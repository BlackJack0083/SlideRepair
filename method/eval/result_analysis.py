from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def analyze_result_directories(result_dirs: list[Path]) -> dict[str, Any]:
    """Aggregate paper metrics from case traces in one or more evaluation runs.

    Failed cases remain in every denominator. Macro F1 is computed over labels
    defined by the gold annotations, so malformed model labels do not become
    additional benchmark classes. They still invalidate case-level exact match.

    Args:
        result_dirs: Directories containing one JSON trace per evaluated case.

    Returns:
        Overall, category-level, issue-level, module, and diagnostic metrics.
    """
    trace_paths = sorted(
        path
        for result_dir in result_dirs
        for path in result_dir.glob("*.json")
    )
    if not trace_paths:
        raise ValueError(f"No case traces found in {result_dirs}")
    traces = [json.loads(path.read_text(encoding="utf-8")) for path in trace_paths]

    category_metrics = _aggregate_detection(traces, "error_category")
    issue_metrics = _aggregate_detection(traces, "specific_issue")
    case_count = len(traces)
    completed_cases = sum(bool(trace["completed"]) for trace in traces)

    error_categories = []
    for label_metric in category_metrics["labels"]:
        error_type = label_metric["label"][0]
        matching_traces = [
            trace
            for trace in traces
            if (error_type,)
            in _records_to_labels(
                trace["metrics"]["detection"]["error_category"]["gold"]
            )
        ]
        error_categories.append(
            {
                **label_metric,
                "end_to_end_success_rate": sum(
                    bool(trace["metrics"]["task_success"])
                    for trace in matching_traces
                )
                / len(matching_traces),
            }
        )

    modules = {}
    for stage in (
        "parser",
        "data_source_extraction",
        "function_logic",
        "data_source_validation",
    ):
        modules[stage] = {
            "value": sum(
                bool(trace["metrics"]["stages"][stage]["success"])
                for trace in traces
            )
            / case_count,
            "unit": "case",
            "support": case_count,
        }

    content_correct = sum(
        trace["metrics"]["stages"]["content_repair"]["correct"]
        for trace in traces
    )
    content_total = sum(
        trace["metrics"]["stages"]["content_repair"]["total"]
        for trace in traces
    )
    modules["content_repair"] = {
        "value": content_correct / content_total if content_total else None,
        "unit": "injected_issue",
        "support": content_total,
    }

    return {
        "result_dirs": [str(result_dir) for result_dir in result_dirs],
        "case_count": case_count,
        "completed_cases": completed_cases,
        "main": {
            "error_category_accuracy": category_metrics["exact_match_accuracy"],
            "error_category_macro_f1": category_metrics["macro_f1"],
            "specific_issue_accuracy": issue_metrics["exact_match_accuracy"],
            "specific_issue_macro_f1": issue_metrics["macro_f1"],
            "end_to_end_success_rate": sum(
                bool(trace["metrics"]["task_success"]) for trace in traces
            )
            / case_count,
            "completion_rate": completed_cases / case_count,
        },
        "error_categories": error_categories,
        "specific_issues": issue_metrics["labels"],
        "modules": modules,
        "diagnostics": {
            "invalid_predicted_error_categories": category_metrics[
                "invalid_predictions"
            ],
            "invalid_predicted_specific_issues": issue_metrics[
                "invalid_predictions"
            ],
        },
    }


def write_analysis_csv(
    analysis: dict[str, Any],
    output_path: Path,
    *,
    model: str,
    client_mode: str,
    configuration: str,
) -> None:
    """Write one evaluation analysis as a long-form CSV.

    Args:
        analysis: Output of :func:`analyze_result_directories`.
        output_path: Destination CSV path.
        model: Model name shown in experiment tables.
        client_mode: Client setting, such as ``deterministic`` or ``llm``.
        configuration: Name of the evaluated workflow configuration.

    Returns:
        None.
    """
    rows: list[dict[str, Any]] = []
    common = {
        "model": model,
        "client_mode": client_mode,
        "configuration": configuration,
        "case_count": analysis["case_count"],
        "completed_cases": analysis["completed_cases"],
        "result_dirs": ",".join(analysis["result_dirs"]),
    }

    for metric, value in analysis["main"].items():
        rows.append(
            {
                **common,
                "section": "main",
                "metric": metric,
                "value": value,
                "support": analysis["case_count"],
            }
        )

    for item in analysis["error_categories"]:
        for metric in (
            "precision",
            "recall",
            "f1",
            "end_to_end_success_rate",
        ):
            rows.append(
                {
                    **common,
                    "section": "error_category",
                    "error_type": item["label"][0],
                    "metric": metric,
                    "value": item[metric],
                    "support": item["support"],
                }
            )

    for item in analysis["specific_issues"]:
        label = item["label"]
        issue_fields = (
            {
                "error_type": label[0],
                "scope_error_type": label[1],
                "field": label[2],
            }
            if label[0] == "scope_error"
            else {"error_type": label[0], "target": label[1]}
        )
        for metric in ("precision", "recall", "f1"):
            rows.append(
                {
                    **common,
                    **issue_fields,
                    "section": "specific_issue",
                    "metric": metric,
                    "value": item[metric],
                    "support": item["support"],
                }
            )

    for module, item in analysis["modules"].items():
        rows.append(
            {
                **common,
                "section": "module",
                "module": module,
                "metric": "success_rate"
                if item["unit"] == "case"
                else "accuracy",
                "value": item["value"],
                "support": item["support"],
                "unit": item["unit"],
            }
        )

    fieldnames = [
        "model",
        "client_mode",
        "configuration",
        "section",
        "error_type",
        "scope_error_type",
        "field",
        "target",
        "module",
        "metric",
        "value",
        "support",
        "unit",
        "case_count",
        "completed_cases",
        "result_dirs",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_detection(
    traces: list[dict[str, Any]],
    block_name: str,
) -> dict[str, Any]:
    pairs = [
        (
            _records_to_labels(
                trace["metrics"]["detection"][block_name]["predicted"]
            ),
            _records_to_labels(trace["metrics"]["detection"][block_name]["gold"]),
        )
        for trace in traces
    ]
    gold_labels = set().union(*(gold for _, gold in pairs))
    label_metrics = []
    for label in sorted(gold_labels):
        true_positive = sum(label in predicted and label in gold for predicted, gold in pairs)
        false_positive = sum(label in predicted and label not in gold for predicted, gold in pairs)
        false_negative = sum(label not in predicted and label in gold for predicted, gold in pairs)
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = true_positive / (true_positive + false_negative)
        label_metrics.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": (
                    2 * precision * recall / (precision + recall)
                    if precision + recall
                    else 0.0
                ),
                "support": true_positive + false_negative,
            }
        )

    invalid_predictions = sorted(
        {
            label
            for predicted, _ in pairs
            for label in predicted
            if label not in gold_labels
        }
    )
    return {
        "exact_match_accuracy": sum(
            predicted == gold for predicted, gold in pairs
        )
        / len(pairs),
        "macro_f1": (
            sum(item["f1"] for item in label_metrics) / len(label_metrics)
            if label_metrics
            else None
        ),
        "labels": label_metrics,
        "invalid_predictions": invalid_predictions,
    }


def _records_to_labels(records: list[dict[str, Any]]) -> set[tuple[str, ...]]:
    labels = set()
    for record in records:
        error_type = str(record["error_type"])
        if "scope_error_type" in record:
            labels.add(
                (
                    error_type,
                    str(record["scope_error_type"]),
                    str(record["field"]),
                )
            )
        elif set(record) == {"error_type"}:
            labels.add((error_type,))
        else:
            labels.add((error_type, str(record["target"])))
    return labels
