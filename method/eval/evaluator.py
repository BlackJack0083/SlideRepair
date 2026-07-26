from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from method.eval.detection import evaluate_detection
from method.eval.stages import (
    _normalize_final_data_source,
    compare_presentations,
    evaluate_content_repair,
    evaluate_data_source_extraction,
    evaluate_function_logic,
    evaluate_parser,
)


class SlideReviewEvaluator:
    """Evaluate detection, stage outputs, and repaired PPTX against benchmark GT."""

    def evaluate_case(
        self,
        *,
        result: dict[str, Any],
        corruption_record: dict[str, Any],
        injected_yaml_path: Path,
        ground_truth_yaml_path: Path,
        ground_truth_pptx_path: Path,
    ) -> dict[str, Any]:
        """Evaluate one completed slide-review case."""
        injected_yaml = yaml.safe_load(injected_yaml_path.read_text(encoding="utf-8"))
        ground_truth_yaml = yaml.safe_load(
            ground_truth_yaml_path.read_text(encoding="utf-8")
        )
        repaired_pptx_path = Path(result["repaired_artifacts"]["pptx_path"])

        detection = evaluate_detection(result["detected_issues"], corruption_record)
        parser = evaluate_parser(
            observed_slide=result["observed_slide"],
            ppt_representation=result["ppt_representation"],
            injected_yaml=injected_yaml,
            injected_yaml_path=injected_yaml_path,
        )
        data_source_extraction = evaluate_data_source_extraction(
            analysis_state=result["slide_analysis_state"],
            injected_yaml=injected_yaml,
        )
        function_logic = evaluate_function_logic(
            analysis_state=result["slide_analysis_state"],
            ground_truth_yaml=ground_truth_yaml,
            ground_truth_yaml_path=ground_truth_yaml_path,
        )
        ground_truth_source = ground_truth_yaml["slide_filters"][0]
        ground_truth_table = ground_truth_source["connection"]["table"]
        expected_final_data_source = {
            "connection": {
                "table": str(
                    ground_truth_table[0]
                    if isinstance(ground_truth_table, list)
                    else ground_truth_table
                ).lower()
            },
            "filters": {
                field: str(ground_truth_source["filters"][field])
                for field in ("city", "block", "start_date", "end_date")
            },
        }
        data_source_validation = {
            "success": _normalize_final_data_source(
                result["analysis_state"]["final_data_source"]
            )
            == expected_final_data_source
        }
        content_repair = evaluate_content_repair(
            corruption_record=corruption_record,
            repaired_pptx_path=repaired_pptx_path,
            ground_truth_pptx_path=ground_truth_pptx_path,
        )
        task_success = compare_presentations(
            repaired_pptx_path,
            ground_truth_pptx_path,
        )
        return {
            "detection": detection,
            "task_success": task_success,
            "stages": {
                "parser": parser,
                "data_source_extraction": data_source_extraction,
                "function_logic": function_logic,
                "data_source_validation": data_source_validation,
                "content_repair": content_repair,
            },
        }

    def evaluate_partial_case(
        self,
        *,
        partial_result: dict[str, Any],
        corruption_record: dict[str, Any],
        injected_yaml_path: Path,
        ground_truth_yaml_path: Path,
    ) -> dict[str, Any]:
        """Evaluate stages that completed before a workflow failure."""
        injected_yaml = yaml.safe_load(injected_yaml_path.read_text(encoding="utf-8"))
        ground_truth_yaml = yaml.safe_load(
            ground_truth_yaml_path.read_text(encoding="utf-8")
        )
        operation_count = len(corruption_record["operations"])

        stages: dict[str, Any] = {
            "parser": {"success": False},
            "data_source_extraction": {"success": False},
            "function_logic": {"success": False},
            "data_source_validation": {"success": False},
            "content_repair": {
                "accuracy": 0.0,
                "success": False,
                "correct": 0,
                "total": operation_count,
            },
        }
        if (
            "observed_slide" in partial_result
            and "ppt_representation" in partial_result
        ):
            stages["parser"] = evaluate_parser(
                observed_slide=partial_result["observed_slide"],
                ppt_representation=partial_result["ppt_representation"],
                injected_yaml=injected_yaml,
                injected_yaml_path=injected_yaml_path,
            )
        if "slide_analysis_state" in partial_result:
            stages["data_source_extraction"] = evaluate_data_source_extraction(
                analysis_state=partial_result["slide_analysis_state"],
                injected_yaml=injected_yaml,
            )
            stages["function_logic"] = evaluate_function_logic(
                analysis_state=partial_result["slide_analysis_state"],
                ground_truth_yaml=ground_truth_yaml,
                ground_truth_yaml_path=ground_truth_yaml_path,
            )
        if "analysis_state" in partial_result:
            ground_truth_source = ground_truth_yaml["slide_filters"][0]
            ground_truth_table = ground_truth_source["connection"]["table"]
            expected_final_data_source = {
                "connection": {
                    "table": str(
                        ground_truth_table[0]
                        if isinstance(ground_truth_table, list)
                        else ground_truth_table
                    ).lower()
                },
                "filters": {
                    field: str(ground_truth_source["filters"][field])
                    for field in ("city", "block", "start_date", "end_date")
                },
            }
            stages["data_source_validation"] = {
                "success": _normalize_final_data_source(
                    partial_result["analysis_state"]["final_data_source"]
                )
                == expected_final_data_source
            }

        return {
            "detection": evaluate_detection(
                partial_result.get("detected_issues", []),
                corruption_record,
            ),
            "task_success": False,
            "stages": stages,
        }


def failure_metrics(corruption_record: dict[str, Any]) -> dict[str, Any]:
    """Return zero-valued metrics for a pipeline failure."""
    operation_count = len(corruption_record["operations"])
    return {
        "detection": evaluate_detection([], corruption_record),
        "task_success": False,
        "stages": {
            "parser": {"success": False},
            "data_source_extraction": {"success": False},
            "function_logic": {"success": False},
            "data_source_validation": {"success": False},
            "content_repair": {
                "accuracy": 0.0,
                "success": False,
                "correct": 0,
                "total": operation_count,
            },
        },
    }
