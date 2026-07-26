from __future__ import annotations

import unittest
from pathlib import Path

from method.eval.slide_review_eval import (
    _error_summary,
    _is_retryable_error,
    load_case_assets,
)
from method.pipeline import WorkflowStageError


class SlideReviewEvalTest(unittest.TestCase):
    def test_retryable_error_unwraps_workflow_stage_error(self) -> None:
        error = WorkflowStageError(
            stage="content_validation",
            partial_result={},
            original_error=TimeoutError("temporary timeout"),
        )

        self.assertTrue(_is_retryable_error(error))

    def test_non_retryable_workflow_stage_error_stays_non_retryable(self) -> None:
        error = WorkflowStageError(
            stage="content_validation",
            partial_result={},
            original_error=ValueError("bad state"),
        )

        self.assertFalse(_is_retryable_error(error))

    def test_error_summary_handles_empty_exception_message(self) -> None:
        self.assertEqual(_error_summary(Exception()), "Exception")

    def test_error_summary_uses_first_line(self) -> None:
        self.assertEqual(_error_summary(ValueError("first\nsecond")), "first")

    def test_loads_ground_truth_slide_as_clean_case(self) -> None:
        benchmark_root = Path("/benchmark")
        record = {
            "sample_id": "sample-1",
            "gt_yaml": "split/test/sample-1/gt/slide.yaml",
            "gt_ppt": "split/test/sample-1/gt/slide.pptx",
        }

        assets = load_case_assets(benchmark_root, record, case_type="clean")

        self.assertEqual(assets["case_id"], "sample-1__clean")
        self.assertEqual(assets["case_type"], "clean")
        self.assertEqual(
            assets["pptx_path"],
            benchmark_root / record["gt_ppt"],
        )
        self.assertEqual(
            assets["image_path"],
            benchmark_root / "split/test/sample-1/gt/slide.png",
        )
        self.assertEqual(assets["corruption_record"], {"operations": []})
        self.assertEqual(assets["feedback_episode"], {"feedback_items": []})


if __name__ == "__main__":
    unittest.main()
