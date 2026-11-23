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

from f1_rag.benchmark import BenchmarkConfig, benchmark_store, run_hyperparameter_grid
from f1_rag.model_config import load_model_config
from f1_rag.providers import (
    CompletionResult,
    ProviderRequestError,
    ProviderRouter,
    _classify_http_error,
    configured_models,
)
from f1_rag.evaluation import ndcg_ids_at_n, recall_ids_at_n
from f1_rag.reranking import DEFAULT_RERANKER_MODEL, RERANKER_PROFILES, rerank_candidates
from f1_rag.sparse import BM25Index, build_source_search_index, hybrid_search
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

    def test_bm25_returns_no_hits_when_the_source_has_no_matching_terms(self) -> None:
        self.assertEqual(BM25Index(self.documents).search("unrelated xyz", 3), [])

    def test_direct_source_search_indexes_word_chunks_without_vectors(self) -> None:
        index = build_source_search_index(self.documents, chunk_size=2, overlap=0)
        hits = index.search("world champion", 1)

        self.assertEqual(hits[0]["source"], "one")
        self.assertEqual(hits[0]["text"], "world champion")

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


class RerankingTests(unittest.TestCase):
    def test_cross_encoder_scores_reorder_candidates_and_keep_retrieval_rank(self) -> None:
        class FakeCrossEncoder:
            @staticmethod
            def predict(pairs, **_kwargs):
                return [0.1, 0.9, 0.5]

        candidates = [
            {"id": "a", "text": "first passage", "score": 0.8},
            {"id": "b", "text": "second passage", "score": 0.7},
            {"id": "c", "text": "third passage", "score": 0.6},
        ]
        ranked = rerank_candidates(FakeCrossEncoder(), "question", candidates, final_k=2)

        self.assertEqual([item["id"] for item in ranked], ["b", "c"])
        self.assertEqual([item["retrieval_rank"] for item in ranked], [2, 3])
        self.assertEqual([item["rerank_score"] for item in ranked], [0.9, 0.5])
        self.assertIn(DEFAULT_RERANKER_MODEL, RERANKER_PROFILES)


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

    def test_access_and_daily_quota_errors_are_fallbackable(self) -> None:
        access = _classify_http_error(403, "", "", "access denied by security policy")
        quota = _classify_http_error(429, "quota_exceeded", "", "daily token quota exceeded")
        generic_quota = _classify_http_error(403, "", "", "You have exceeded your quota")
        billing = _classify_http_error(429, "credit_balance_exhausted", "", "Add credits")
        invalid_request = _classify_http_error(400, "invalid_request", "", "bad parameter")

        self.assertEqual(access, ("access_denied", True, 900))
        self.assertEqual(quota, ("quota_exhausted", True, 86400))
        self.assertEqual(generic_quota, ("quota_exhausted", True, 3600))
        self.assertEqual(billing, ("billing_unavailable", True, 3600))
        self.assertEqual(invalid_request, ("request_rejected", False, 0))

    def test_rotation_skips_provider_during_persisted_quota_cooldown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            event_path = Path(temporary) / "provider-calls.jsonl"
            router = ProviderRouter(self.config, event_path)
            openrouter_failure = ProviderRequestError(
                "quota exhausted",
                fallbackable=True,
                category="quota_exhausted",
                cooldown_seconds=86400,
            )
            success = CompletionResult("google", "gemma-4-31b-it", "ok", 9.0, 1, 1, None, "stop")
            with patch.dict(
                "os.environ",
                {"OPENROUTER_API_KEY": "or-test", "GOOGLE_API_KEY": "google-test"},
            ), patch.object(router, "complete", side_effect=[openrouter_failure, success, success]) as complete:
                first = router.rotate(
                    ["openrouter::openrouter/free", "gemini::gemma-4-31b-it"], []
                )
                second = router.rotate(
                    ["openrouter::openrouter/free", "gemini::gemma-4-31b-it"], []
                )

            self.assertEqual(first.provider, "google")
            self.assertEqual(second.provider, "google")
            self.assertEqual([call.args[0] for call in complete.call_args_list], [
                "openrouter::openrouter/free",
                "gemini::gemma-4-31b-it",
                "gemini::gemma-4-31b-it",
            ])
            persisted = json.loads((Path(temporary) / "provider-cooldowns.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["openrouter::openrouter/free"]["category"], "quota_exhausted")

    def test_rotation_error_reports_each_provider_status_and_message(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            router = ProviderRouter(self.config, Path(temporary) / "events.jsonl")
            rotation = self.config["routing"]["answer_rotation"]
            env = {
                router.providers[router.models[name].provider].api_key_env: "test-key"
                for name in rotation
            }
            failures = [
                ProviderRequestError(
                    f"{name} denied",
                    fallbackable=True,
                    category="access_denied",
                    http_status=403,
                    provider_code="permission_denied",
                    provider_message=f"{name} permission denied",
                )
                for name in rotation
            ]
            with patch.dict("os.environ", env), patch.object(router, "complete", side_effect=failures):
                with self.assertRaisesRegex(RuntimeError, "HTTP 403.*permission denied") as raised:
                    router.rotate(rotation, [])

            for name in rotation:
                self.assertIn(name, str(raised.exception))


class BenchmarkTests(unittest.TestCase):
    def test_grid_forwards_tuning_queries_to_each_benchmark(self) -> None:
        documents = [{"id": "one", "source": "one", "text": "champion"}]
        vectors = np.array([[1, 0]], dtype=np.float32)
        queries = [{"question": "champion", "source": "one"}]
        encode_query = lambda _text: np.array([1, 0], dtype=np.float32)

        with tempfile.TemporaryDirectory() as temporary, patch(
            "f1_rag.benchmark.benchmark_store", return_value={"run_id": "expected"}
        ) as benchmark:
            results = run_hyperparameter_grid(
                dataset="test",
                store_names=["NumPy (exact)"],
                embedding_model="test-embedder",
                chunk_sizes=[128],
                overlaps=[16],
                top_ks=[1],
                repeats=1,
                prepare=lambda _size, _overlap: (documents, vectors, encode_query),
                queries=queries,
                output_dir=Path(temporary),
            )

        self.assertEqual(results, [{"run_id": "expected"}])
        self.assertEqual(benchmark.call_args.args[3], queries)

    def test_numpy_benchmark_records_repeatable_metrics_and_metadata(self) -> None:
        documents = [
            {"id": "one", "source": "one", "text": "champion"},
            {"id": "two", "source": "two", "text": "race"},
        ]
        vectors = np.array([[1, 0], [0, 1]], dtype=np.float32)
        queries = [{"question": "champion", "source": "one"}]

        with tempfile.TemporaryDirectory() as temporary:
            result = benchmark_store(
                BenchmarkConfig(
                    "test", "NumPy (exact)", "test-embedder", 128, 16, 2,
                    repeats=2, metric_ks=(1, 2),
                ),
                documents,
                vectors,
                queries,
                lambda _text: np.array([1, 0], dtype=np.float32),
                Path(temporary),
            )
            rows = (Path(temporary) / "runs.jsonl").read_text(encoding="utf-8").splitlines()

        self.assertIsNone(result["error"])
        self.assertEqual(result["results"]["documents"], len(documents))
        self.assertEqual(result["results"]["successful_repeats"], 2)
        self.assertEqual(result["results"]["recall_at_k"], 1.0)
        self.assertEqual(result["results"]["metrics_at_k"]["1"]["recall"], 1.0)
        self.assertEqual(result["results"]["metrics_at_k"]["2"]["ndcg"], 1.0)
        self.assertGreaterEqual(result["results"]["search_latency_ms_p95"], 0)
        self.assertTrue(result["results"]["resource_metrics_available"])
        self.assertIsNotNone(result["results"]["rss_delta_bytes_median"])
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
