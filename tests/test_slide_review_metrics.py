from __future__ import annotations

import unittest

from method.eval.detection import aggregate_metrics, evaluate_detection
from method.eval.evaluator import failure_metrics
from method.eval.stages import evaluate_data_source_extraction


class SlideReviewMetricsTest(unittest.TestCase):
    def test_detection_uses_scope_subtype_and_field(self) -> None:
        corruption = {
            "operations": [
                {
                    "error_types": ["scope_error"],
                    "scope_error_type": "missing",
                    "field": "block",
                },
                {
                    "error_types": ["claim_error"],
                    "target": "summary",
                },
            ]
        }
        detected = [
            {
                "error_type": "scope_error",
                "scope_error_type": "missing",
                "field": "block",
            },
            {"error_type": "claim_error", "target": "summary"},
        ]

        metrics = evaluate_detection(detected, corruption)

        self.assertTrue(metrics["error_category"]["exact_match"])
        self.assertTrue(metrics["specific_issue"]["exact_match"])
        self.assertEqual(metrics["error_category"]["f1"], 1.0)
        self.assertEqual(metrics["specific_issue"]["f1"], 1.0)
        self.assertEqual(
            metrics["error_category"]["gold"],
            [{"error_type": "claim_error"}, {"error_type": "scope_error"}],
        )
        self.assertEqual(
            metrics["specific_issue"]["gold"],
            [
                {"error_type": "claim_error", "target": "summary"},
                {
                    "error_type": "scope_error",
                    "scope_error_type": "missing",
                    "field": "block",
                },
            ],
        )

    def test_detection_exact_match_rejects_extra_issue(self) -> None:
        corruption = {
            "operations": [
                {
                    "error_types": ["value_error"],
                    "target": "table",
                }
            ]
        }
        detected = [
            {"error_type": "value_error", "target": "table"},
            {"error_type": "claim_error", "target": "summary"},
        ]

        metrics = evaluate_detection(detected, corruption)

        self.assertFalse(metrics["error_category"]["exact_match"])
        self.assertFalse(metrics["specific_issue"]["exact_match"])
        self.assertEqual(metrics["error_category"]["precision"], 0.5)
        self.assertEqual(metrics["error_category"]["recall"], 1.0)
        self.assertEqual(metrics["specific_issue"]["precision"], 0.5)
        self.assertEqual(metrics["specific_issue"]["recall"], 1.0)

    def test_detection_deduplicates_repeated_scope_labels(self) -> None:
        operation = {
            "error_types": ["scope_error"],
            "scope_error_type": "missing",
            "field": "block",
        }
        detected = [
            {
                "error_type": "scope_error",
                "scope_error_type": "missing",
                "field": "block",
            }
        ]

        metrics = evaluate_detection(
            detected,
            {"operations": [operation, operation]},
        )

        self.assertTrue(metrics["specific_issue"]["exact_match"])
        self.assertEqual(
            metrics["specific_issue"]["gold"],
            [
                {
                    "error_type": "scope_error",
                    "scope_error_type": "missing",
                    "field": "block",
                }
            ],
        )

    def test_aggregate_reports_three_primary_metrics(self) -> None:
        corruption = {
            "operations": [
                {
                    "mutation_type": "value_data_cell",
                    "element_id": "4",
                    "error_types": ["value_error"],
                    "target": "table",
                }
            ]
        }
        successful = failure_metrics(corruption)
        successful["detection"] = evaluate_detection(
            [{"error_type": "value_error", "target": "table"}],
            corruption,
        )
        successful["task_success"] = True
        for stage in (
            "parser",
            "data_source_extraction",
            "function_logic",
            "data_source_validation",
        ):
            successful["stages"][stage]["success"] = True
        successful["stages"]["content_repair"] = {
            "accuracy": 1.0,
            "success": True,
            "correct": 1,
            "total": 1,
        }
        failed = failure_metrics(corruption)

        aggregate = aggregate_metrics([successful, failed])

        self.assertEqual(aggregate["error_category_macro_f1"], 2 / 3)
        self.assertEqual(aggregate["specific_issue_macro_f1"], 2 / 3)
        self.assertEqual(aggregate["error_category_exact_match_rate"], 0.5)
        self.assertEqual(aggregate["specific_issue_exact_match_rate"], 0.5)
        self.assertEqual(aggregate["end_to_end_success_rate"], 0.5)
        self.assertEqual(aggregate["stage_success_rate"]["parser"], 0.5)
        self.assertEqual(aggregate["stage_success_rate"]["content_repair"], 0.5)

    def test_aggregate_macro_f1_uses_gold_label_space(self) -> None:
        corruption = {
            "operations": [
                {
                    "error_types": ["value_error"],
                    "target": "table",
                }
            ]
        }
        metrics = failure_metrics(corruption)
        metrics["detection"] = evaluate_detection(
            [
                {"error_type": "value_error", "target": "table"},
                {"error_type": "none", "target": "all"},
            ],
            corruption,
        )

        aggregate = aggregate_metrics([metrics])

        self.assertEqual(aggregate["error_category_macro_f1"], 1.0)
        self.assertEqual(aggregate["specific_issue_macro_f1"], 1.0)
        self.assertEqual(aggregate["error_category_exact_match_rate"], 0.0)
        self.assertEqual(aggregate["specific_issue_exact_match_rate"], 0.0)

    def test_aggregate_does_not_count_absent_labels_as_false_negatives(self) -> None:
        value_corruption = {
            "operations": [
                {"error_types": ["value_error"], "target": "table"}
            ]
        }
        claim_corruption = {
            "operations": [
                {"error_types": ["claim_error"], "target": "summary"}
            ]
        }
        value_metrics = failure_metrics(value_corruption)
        value_metrics["detection"] = evaluate_detection(
            [{"error_type": "value_error", "target": "table"}],
            value_corruption,
        )
        claim_metrics = failure_metrics(claim_corruption)
        claim_metrics["detection"] = evaluate_detection(
            [{"error_type": "claim_error", "target": "summary"}],
            claim_corruption,
        )

        aggregate = aggregate_metrics([value_metrics, claim_metrics])

        self.assertEqual(aggregate["error_category_macro_f1"], 1.0)
        self.assertEqual(aggregate["specific_issue_macro_f1"], 1.0)

    def test_content_repair_counts_each_injected_operation(self) -> None:
        operation = {
            "mutation_type": "value_data_cell",
            "element_id": "4",
            "error_types": ["value_error"],
            "target": "table",
        }

        metrics = failure_metrics({"operations": [operation, operation]})

        self.assertEqual(metrics["stages"]["content_repair"]["total"], 2)

    def test_data_source_extraction_uses_slide_filter_columns(self) -> None:
        injected_yaml = {
            "template_slide": {
                "elements": [
                    {
                        "id": "3",
                        "role": "caption",
                        "text_binding": {
                            "kind": "caption",
                            "slots": {
                                "Geo_City_Name": {"value": "Guangzhou"},
                                "Geo_Block_Name": {
                                    "value": "International Innovation City"
                                },
                                "Temporal_Start_Year": {"value": "2020"},
                                "Temporal_End_Year": {"value": "2024"},
                            },
                        },
                    },
                    {
                        "id": "4",
                        "role": "table",
                        "data": "./data/element_4.csv",
                    },
                ]
            },
            "slide_filters": [
                {
                    "connection": {"table": ["guangzhou_resale_house"]},
                    "select_columns": ["dim_area", "dim_price"],
                    "fun_tool": {
                        "fun": "Area x Price Cross Pivot",
                        "args": {
                            "metrics": [
                                {
                                    "source_col": "dim_price",
                                    "agg_func": "mean",
                                    "filter_condition": {"trade_sets": 1},
                                }
                            ]
                        },
                    },
                }
            ],
        }
        analysis_state = {
            "tables": [
                {
                    "caption": {
                        "data_source": {
                            "connection": {"table": "guangzhou_resale_house"},
                            "select_columns": ["dim_area", "dim_price"],
                            "filters": {
                                "city": "Guangzhou",
                                "block": "International Innovation City",
                                "start_date": "2020-01-01",
                                "end_date": "2024-12-31",
                            },
                        }
                    }
                }
            ]
        }

        metrics = evaluate_data_source_extraction(
            analysis_state=analysis_state,
            injected_yaml=injected_yaml,
        )

        self.assertTrue(metrics["success"])
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_data_source_extraction_derives_table_from_caption_city(self) -> None:
        injected_yaml = {
            "template_slide": {
                "elements": [
                    {
                        "id": "3",
                        "role": "caption",
                        "text_binding": {
                            "kind": "caption",
                            "slots": {
                                "Geo_City_Name": {"value": "Beijing"},
                                "Geo_Block_Name": {"value": "Lianhuashan"},
                                "Temporal_Start_Year": {"value": "2020"},
                                "Temporal_End_Year": {"value": "2024"},
                            },
                        },
                    },
                    {"id": "4", "role": "chart-bar", "data": "./data/element_4.csv"},
                ]
            },
            "slide_filters": [
                {
                    "connection": {"table": ["guangzhou_new_house"]},
                    "select_columns": ["date_code", "dim_area"],
                    "fun_tool": {"fun": "Supply-Transaction Area", "args": {}},
                }
            ],
        }
        analysis_state = {
            "tables": [
                {
                    "caption": {
                        "data_source": {
                            "connection": {"table": "beijing_new_house"},
                            "select_columns": ["date_code", "dim_area"],
                            "filters": {
                                "city": "Beijing",
                                "block": "Lianhuashan",
                                "start_date": "2020-01-01",
                                "end_date": "2024-12-31",
                            },
                        }
                    }
                }
            ]
        }

        metrics = evaluate_data_source_extraction(
            analysis_state=analysis_state,
            injected_yaml=injected_yaml,
        )

        self.assertTrue(metrics["success"])


if __name__ == "__main__":
    unittest.main()
