from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1_rag.reporting import (
    dataframe_to_csv,
    dataframe_to_pdf,
    load_evaluation_runs,
    save_evaluation_run,
)


class EvaluationReportingTests(unittest.TestCase):
    def test_csv_and_pdf_exports_contain_the_report(self) -> None:
        frame = pd.DataFrame([{"Backend": "NumPy", "Recall@1": 1.0, "RAM delta": "−1.0 KB"}])

        csv_data = dataframe_to_csv(frame)
        pdf_data = dataframe_to_pdf(frame, "Retrieval evaluation")

        self.assertTrue(csv_data.startswith(b"\xef\xbb\xbf"))
        self.assertIn(b"Backend,Recall@1,RAM delta", csv_data)
        self.assertTrue(pdf_data.startswith(b"%PDF-"))
        self.assertIn(b"Retrieval evaluation", pdf_data)

    def test_evaluation_runs_are_persisted_and_loaded_newest_first(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            first = save_evaluation_run(
                output_dir,
                test_type="retrieval",
                dataset="wikipedia",
                configuration={"top_k": 3},
                rows=[{"Recall@3": 1.0}],
            )
            second = save_evaluation_run(
                output_dir,
                test_type="answer_accuracy",
                dataset="wikipedia",
                configuration={"sample_size": 10},
                rows=[{"QA accuracy": 0.8}],
                errors=["one backend failed"],
            )

            records = load_evaluation_runs(output_dir, dataset="wikipedia")
            persisted = [
                json.loads(line)
                for line in (output_dir / "evaluations.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            filtered_records = load_evaluation_runs(output_dir, dataset="jolpica")

        self.assertEqual([record["run_id"] for record in records], [second["run_id"], first["run_id"]])
        self.assertEqual(persisted[1]["errors"], ["one backend failed"])
        self.assertEqual(filtered_records, [])

    def test_invalid_history_is_reported_instead_of_silently_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "evaluations.jsonl"
            path.write_text("{invalid json}\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "line 1"):
                load_evaluation_runs(Path(temporary))

    def test_incomplete_history_record_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "evaluations.jsonl"
            path.write_text(json.dumps({"rows": []}) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "record at line 1"):
                load_evaluation_runs(Path(temporary))


if __name__ == "__main__":
    unittest.main()
