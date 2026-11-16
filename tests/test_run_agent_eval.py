import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter"))
from baseline import TfidfIndex
from run_agent_eval import (
    TrackedController,
    TfidfRetriever,
    _make_error_message,
    _write_json,
)


class AgentEvalRunnerTests(unittest.TestCase):
    def test_tfidf_retriever_returns_agent_chunks_and_rejects_filters(self):
        chunk = {
            "id": "chunk-1",
            "text": "Python uses a clear syntax.",
            "source_id": "book.pdf",
            "source": "book.pdf",
            "page": 1,
            "metadata": {},
        }
        retriever = TfidfRetriever(TfidfIndex([chunk]))
        results = retriever.search("Python syntax", filters=None, top_k=1)
        self.assertEqual(results[0]["id"], "chunk-1")
        self.assertEqual(results[0]["source_id"], "book.pdf")
        with self.assertRaises(ValueError):
            retriever.search(
                "Python syntax", filters={"section": "1.2"}, top_k=1
            )

    def test_controller_tracks_model_call_attempts_even_on_error(self):
        class FailingController:
            def decide(self, question, history):
                raise RuntimeError("provider unavailable")

            def close(self):
                pass

        controller = TrackedController(FailingController())
        with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
            controller.decide("question", [])
        self.assertEqual(controller.call_attempts, 1)
        self.assertGreaterEqual(controller.elapsed_ms, 0)

    def test_error_message_redacts_api_key(self):
        message = _make_error_message(
            RuntimeError("request failed for secret-key"), "secret-key"
        )
        self.assertNotIn("secret-key", message)
        self.assertIn("[REDACTED]", message)

    def test_json_progress_file_is_written_atomically_and_can_be_updated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traces.json"
            _write_json(path, [{"query_id": "Q01"}])
            _write_json(path, [{"query_id": "Q01"}, {"query_id": "Q02"}])
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                [{"query_id": "Q01"}, {"query_id": "Q02"}],
            )
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
