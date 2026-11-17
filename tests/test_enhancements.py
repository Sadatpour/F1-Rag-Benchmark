from __future__ import annotations

import json
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1_rag.benchmark import BenchmarkConfig, benchmark_store
from f1_rag.model_config import load_model_config
from f1_rag.providers import CompletionResult, ProviderRequestError, ProviderRouter, configured_models
from f1_rag.evaluation import ndcg_ids_at_n, recall_ids_at_n
from f1_rag.sparse import BM25Index, hybrid_search
from f1_rag.stores import create_store


class HybridRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = [
            {"id": "d1", "source": "one", "text": "Formula One world champion"},
            {"id": "d2", "source": "two", "text": "Formula One race winner"},
            {"id": "d3", "source": "three", "text": "A completely different subject"},
        ]

    def test_bm25_prefers_exact_lexical_match(self) -> None:
        hits = BM25Index(self.documents).search("world champion", 2)
        self.assertEqual(hits[0]["id"], "d1")
        self.assertGreater(hits[0]["score"], 0)

    def test_rrf_merges_rankings_and_deduplicates_documents(self) -> None:
        dense = [self.documents[1], self.documents[0]]
        sparse = [self.documents[0], self.documents[2]]

        hits = hybrid_search(dense, sparse, limit=3)

        self.assertEqual({hit["id"] for hit in hits}, {"d1", "d2", "d3"})
        self.assertEqual(hits[0]["id"], "d1")
        self.assertTrue(all(hit["retrieval_method"] == "hybrid_rrf" for hit in hits))


class QrelsMetricTests(unittest.TestCase):
    def test_id_based_recall_and_graded_ndcg_use_document_ids(self) -> None:
        relevant = {"a": 3, "b": 1}
        hits = [{"id": "b"}, {"id": "a"}, {"id": "c"}]

        self.assertEqual(recall_ids_at_n(hits, set(relevant), n=2), 1.0)
        self.assertLess(ndcg_ids_at_n(hits, relevant, n=2), 1.0)
        self.assertEqual(ndcg_ids_at_n([{"id": "a"}, {"id": "b"}], relevant, n=2), 1.0)


class ProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_model_config()

    def test_model_registry_keeps_provider_and_model_id_separate(self) -> None:
        models = configured_models(self.config)

        self.assertEqual(models["grok::grok-4.6"].provider, "xai")
        self.assertEqual(models["grok::grok-4.6"].model_id, "grok-4.6")
        self.assertEqual(models["mistral::mistral-small-2603"].provider, "mistral")
        self.assertEqual(models["gemini::gemma-4-31b-it"].provider, "google")
        self.assertEqual(models["groq::openai/gpt-oss-20b"].provider, "groq")
        self.assertEqual(models["groq::openai/gpt-oss-20b"].model_id, "openai/gpt-oss-20b")

    def test_openai_compatible_response_is_normalized_and_logged_without_prompt(self) -> None:
        model_name = "grok::grok-4.6"
        body = {
            "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3, "cost": 0.0001},
        }

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            @staticmethod
            def read() -> bytes:
                return json.dumps(body).encode()

        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "provider-calls.jsonl"
            router = ProviderRouter(self.config, log_path)
            with patch.dict("os.environ", {"XAI_API_KEY": "test-key"}), patch(
                "f1_rag.providers.urlopen", return_value=Response()
            ) as request:
                result = router.complete(model_name, [{"role": "user", "content": "secret prompt"}])

            self.assertIsInstance(result, CompletionResult)
            self.assertEqual(result.provider, "xai")
            self.assertEqual(result.model_id, "grok-4.6")
            self.assertEqual(result.text, "answer")
            self.assertEqual(result.cost_usd, 0.0001)
            self.assertEqual(request.call_args.args[0].full_url, "https://api.x.ai/v1/chat/completions")
            event = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertEqual(event["status"], "success")
            self.assertNotIn("secret prompt", log_path.read_text(encoding="utf-8"))

    def test_google_gemma_uses_native_generate_content_request_shape(self) -> None:
        router = ProviderRouter(self.config, Path("unused.jsonl"))
        model = router.models["gemini::gemma-4-31b-it"]
        payload, headers, url = router._request_parts(
            router.providers["google"],
            model,
            [
                {"role": "system", "content": "Use JSON."},
                {"role": "user", "content": "Return a result."},
            ],
            "test-key",
            100,
            0,
            "json",
        )

        self.assertEqual(url, "https://generativelanguage.googleapis.com/v1beta/models/gemma-4-31b-it:generateContent")
        self.assertEqual(headers["x-goog-api-key"], "test-key")
        self.assertEqual(payload["systemInstruction"]["parts"][0]["text"], "Use JSON.")
        self.assertEqual(payload["contents"][0]["role"], "user")
        self.assertEqual(payload["generationConfig"]["responseMimeType"], "application/json")

    def test_rotation_uses_configured_order_and_transient_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            router = ProviderRouter(self.config, Path(temporary) / "events.jsonl")
            success = CompletionResult("mistral", "mistral-small-2603", "ok", 9.0, 1, 1, None, "stop")
            with patch.dict("os.environ", {"XAI_API_KEY": "x-test", "MISTRAL_API_KEY": "m-test"}), patch.object(
                router,
                "complete",
                side_effect=[ProviderRequestError("temporary failure", retryable=True), success],
            ) as complete:
                result = router.rotate(["grok::grok-4.6", "mistral::mistral-small-2603"], [])

            self.assertEqual(result.provider, "mistral")
            self.assertEqual([call.args[0] for call in complete.call_args_list], [
                "grok::grok-4.6", "mistral::mistral-small-2603"
            ])


class BenchmarkTests(unittest.TestCase):
    def test_numpy_benchmark_records_repeatable_metrics_and_metadata(self) -> None:
        documents = [
            {"id": "one", "source": "one", "text": "champion"},
            {"id": "two", "source": "two", "text": "race"},
        ]
        vectors = np.array([[1, 0], [0, 1]], dtype=np.float32)
        queries = [{"question": "champion", "source": "one"}]

        with tempfile.TemporaryDirectory() as temporary:
            result = benchmark_store(
                BenchmarkConfig("test", "NumPy (exact)", "test-embedder", 128, 16, 1, repeats=2),
                documents,
                vectors,
                queries,
                lambda _text: np.array([1, 0], dtype=np.float32),
                Path(temporary),
            )
            rows = (Path(temporary) / "runs.jsonl").read_text(encoding="utf-8").splitlines()

        self.assertIsNone(result["error"])
        self.assertEqual(result["results"]["successful_repeats"], 2)
        self.assertEqual(result["results"]["recall_at_k"], 1.0)
        self.assertGreaterEqual(result["results"]["search_latency_ms_p95"], 0)
        self.assertEqual(result["config"]["embedding_model"], "test-embedder")
        self.assertEqual(len(rows), 1)

    @unittest.skipUnless(importlib.util.find_spec("faiss"), "FAISS optional dependency is not installed")
    def test_faiss_store_builds_and_retrieves_same_fixture(self) -> None:
        self._assert_store_retrieves("FAISS")

    @unittest.skipUnless(importlib.util.find_spec("chromadb"), "ChromaDB optional dependency is not installed")
    def test_chroma_store_builds_and_retrieves_same_fixture(self) -> None:
        self._assert_store_retrieves("ChromaDB")

    def _assert_store_retrieves(self, store_name: str) -> None:
        documents = [
            {"id": "one", "source": "one", "text": "champion"},
            {"id": "two", "source": "two", "text": "race"},
        ]
        vectors = np.array([[1, 0], [0, 1]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary)
            store = create_store(store_name, output_path / "index")
            store.build(vectors, documents, "fixture")
            hits = store.search(np.array([1, 0], dtype=np.float32), 1)
            close = getattr(store, "close", None)
            if callable(close):
                close()
            result = benchmark_store(
                BenchmarkConfig("test", store_name, "fixture-model", 128, 16, 1, repeats=2),
                documents,
                vectors,
                [{"question": "champion", "source": "one"}],
                lambda _text: np.array([1, 0], dtype=np.float32),
                output_path / "benchmark",
            )
        self.assertEqual(hits[0]["id"], "one")
        self.assertIsNone(result["error"])
        self.assertEqual(result["results"]["recall_at_k"], 1.0)


if __name__ == "__main__":
    unittest.main()
