from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1_rag.chunking import chunk_documents
from f1_rag.evaluation import ndcg_at_n, qa_accuracy, recall_at_n
from f1_rag.ingest import build_knowledge_base, load_documents
from f1_rag.model_config import MODEL_CONFIG_PATH, load_model_config
from f1_rag.openrouter import DEFAULT_JUDGE_MODEL, chat_completion, judge_answer
from f1_rag.retrieval import _fingerprint
from f1_rag.sources import JOLPICA_API_BASE, jolpica_source_url
from f1_rag.stores import NumpyStore


class ChunkingTests(unittest.TestCase):
    def test_chunks_have_requested_size_and_overlap(self) -> None:
        words = [f"word{index}" for index in range(11)]
        chunks = chunk_documents([{"id": "doc", "text": " ".join(words)}], chunk_size=5, overlap=2)
        self.assertEqual([len(chunk["text"].split()) for chunk in chunks], [5, 5, 5, 2])
        self.assertEqual(chunks[0]["text"].split()[-2:], chunks[1]["text"].split()[:2])

    def test_overlap_must_be_less_than_chunk_size(self) -> None:
        with self.assertRaises(ValueError):
            chunk_documents([{"id": "doc", "text": "words"}], chunk_size=5, overlap=5)


class EvaluationTests(unittest.TestCase):
    def test_source_relevance_metrics_use_ranked_hits(self) -> None:
        hits = [{"source": "other"}, {"source": "expected"}, {"source": "expected"}]
        self.assertEqual(recall_at_n(hits, {"expected"}, 2), 1.0)
        self.assertAlmostEqual(ndcg_at_n(hits, {"expected"}, 10), 1 / 1.5849625007)

    def test_qa_accuracy_is_fraction_correct(self) -> None:
        self.assertEqual(qa_accuracy([{"correct": True}, {"correct": False}]), 0.5)


class ConfigurationTests(unittest.TestCase):
    def test_model_choices_are_loaded_from_central_toml(self) -> None:
        config = load_model_config(MODEL_CONFIG_PATH)
        self.assertEqual(config["embedding"]["repo_id"], "sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(
            config["embedding"]["weights_path"],
            "models/embedding/all-MiniLM-L6-v2/model.safetensors",
        )
        self.assertIn("cohere/north-mini-code:free", config["openrouter"]["answer_models"])
        self.assertEqual(config["openrouter"]["judge_model"], "nvidia/nemotron-3.5-lightning:free")

    def test_invalid_model_config_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "models.toml"
            path.write_text("[embedding]\nrepo_id = 'model'\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must define"):
                load_model_config(path)


class NumpyStoreTests(unittest.TestCase):
    def test_persistent_index_loads_and_returns_nearest_document(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            documents = [
                {"id": "a", "source": "a", "text": "first"},
                {"id": "b", "source": "b", "text": "second"},
            ]
            store = NumpyStore(path)
            store.build(np.array([[1, 0], [0, 1]], dtype=np.float32), documents, "fingerprint")

            restored = NumpyStore(path)
            self.assertTrue(restored.load("fingerprint"))
            self.assertEqual(restored.search(np.array([0, 1], dtype=np.float32), 1)[0]["id"], "b")
            self.assertFalse(NumpyStore(path).load("different-fingerprint"))


class RetrievalTests(unittest.TestCase):
    def test_fingerprint_includes_embedding_model_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            corpus = Path(temporary) / "wikipedia.jsonl"
            corpus.write_text("{}\n", encoding="utf-8")
            fingerprint = _fingerprint(corpus, chunk_size=500, overlap=80)

        self.assertRegex(fingerprint, r"^[0-9a-f]{64}$")


class OpenRouterTests(unittest.TestCase):
    @patch("f1_rag.openrouter._request")
    def test_reasoning_budget_exhaustion_reports_why_completion_is_empty(self, request) -> None:
        request.return_value = {
            "choices": [{"finish_reason": "length", "message": {"content": None}}],
            "usage": {
                "completion_tokens": 900,
                "completion_tokens_details": {"reasoning_tokens": 900},
            },
        }

        with self.assertRaisesRegex(RuntimeError, "output limit.*900 of 900.*reasoning"):
            chat_completion("test/model", [], "test-key", max_tokens=900)

    @patch("f1_rag.openrouter._request")
    def test_default_judge_requests_json_mode(self, request) -> None:
        request.return_value = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": '{"correct": true, "reason": "Supported by the evidence."}'},
                }
            ]
        }

        verdict = judge_answer("Question", "Reference", "Candidate", DEFAULT_JUDGE_MODEL, "test-key")

        self.assertEqual(verdict, {"correct": True, "reason": "Supported by the evidence."})
        self.assertEqual(request.call_args.args[2]["response_format"], {"type": "json_object"})
        self.assertEqual(request.call_args.args[2]["reasoning"], {"effort": "none"})
        self.assertEqual(request.call_args.args[2]["max_tokens"], 500)


class SourceAttributionTests(unittest.TestCase):
    def test_jolpica_source_ids_resolve_to_direct_api_records(self) -> None:
        self.assertEqual(
            jolpica_source_url("Jolpica:1950/1/results"),
            f"{JOLPICA_API_BASE}/1950/1/results/?format=json",
        )
        self.assertEqual(
            jolpica_source_url("Jolpica:2024/driverStandings"),
            f"{JOLPICA_API_BASE}/2024/driverStandings/?format=json",
        )
        self.assertEqual(
            jolpica_source_url("Jolpica:circuits/albert_park"),
            f"{JOLPICA_API_BASE}/circuits/albert_park/?format=json",
        )

    def test_non_jolpica_source_does_not_create_an_external_url(self) -> None:
        self.assertEqual(jolpica_source_url("Wikipedia:Monza"), f"{JOLPICA_API_BASE}/")


class IngestionTests(unittest.TestCase):
    def test_compressed_corpus_is_loaded_when_plain_jsonl_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            corpus = Path(temporary) / "corpora" / "jolpica.jsonl.gz"
            corpus.parent.mkdir()
            with gzip.open(corpus, "wt", encoding="utf-8") as file:
                file.write(json.dumps({"id": "jolpica:1950", "text": "1950 season record"}) + "\n")

            documents = load_documents("jolpica", Path(temporary))

        self.assertEqual(documents[0]["id"], "jolpica:1950")

    def test_outputs_are_independent_and_keep_matching_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input"
            wiki = source / "raw" / "wikipedia"
            jolpica = source / "raw" / "jolpica"
            processed = source / "processed"
            final = source / "final"
            wiki.mkdir(parents=True)
            jolpica.mkdir(parents=True)
            processed.mkdir()
            final.mkdir()
            (wiki / "drivers_wiki.json").write_text(
                json.dumps([{"title": "Nino Farina", "extract": "Nino Farina won the 1950 championship."}]),
                encoding="utf-8",
            )
            (jolpica / "1950_results.json").write_text(
                json.dumps(
                    {
                        "MRData": {
                            "RaceTable": {
                                "season": "1950",
                                "Races": [
                                    {
                                        "round": "1",
                                        "raceName": "British Grand Prix",
                                        "Results": [{"Driver": {"givenName": "Nino", "familyName": "Farina"}}],
                                    }
                                ],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (processed / "champions.json").write_text("[]", encoding="utf-8")
            (final / "qa.jsonl").write_text(
                json.dumps(
                    {
                        "question": "Who won the 1950 championship?",
                        "answer": "Nino Farina",
                        "source": "Jolpica:1950/1/results",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "knowledge_base"

            counts = build_knowledge_base(source, output)

            self.assertEqual(counts["wikipedia"], 1)
            self.assertGreaterEqual(counts["jolpica"], 1)
            self.assertEqual(counts["evaluation"], 1)
            self.assertTrue((output / "corpora" / "wikipedia.jsonl").is_file())
            self.assertTrue((output / "corpora" / "jolpica.jsonl").is_file())
            docs = [
                json.loads(line)
                for line in (output / "corpora" / "jolpica.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertIn("Jolpica:1950/1/results", {doc["source"] for doc in docs})
            race = next(doc for doc in docs if doc["source"] == "Jolpica:1950/1/results")
            self.assertEqual(
                race["source_url"],
                f"{JOLPICA_API_BASE}/1950/1/results/?format=json",
            )
            self.assertTrue((output / "evaluation" / "qa.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
