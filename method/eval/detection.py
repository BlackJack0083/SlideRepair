from __future__ import annotations

from typing import Any


def aggregate_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the primary detection, task, and stage metrics."""
    error_category_f1 = _aggregate_f1_by_label(case["detection"]["error_category"] for case in metrics)
    specific_issue_f1 = _aggregate_f1_by_label(case["detection"]["specific_issue"] for case in metrics)

    case_count = len(metrics)
    content_correct = sum(case["stages"]["content_repair"]["correct"] for case in metrics)
    content_total = sum(case["stages"]["content_repair"]["total"] for case in metrics)
    stage_success_rate = {
        stage: (
            sum(case["stages"][stage]["success"] for case in metrics) / case_count
            if case_count
            else 0.0
        )
        for stage in (
            "parser",
            "data_source_extraction",
            "function_logic",
            "data_source_validation",
        )
    }
    stage_success_rate["content_repair"] = (
        content_correct / content_total if content_total else 0.0
    )

    return {
        "error_category_macro_f1": error_category_f1["macro_f1"],
        "error_category_exact_match_rate": (
            sum(case["detection"]["error_category"]["exact_match"] for case in metrics) / case_count
            if case_count
            else 0.0
        ),
        "specific_issue_macro_f1": specific_issue_f1["macro_f1"],
        "specific_issue_exact_match_rate": (
            sum(case["detection"]["specific_issue"]["exact_match"] for case in metrics)
            / case_count
            if case_count
            else 0.0
        ),
        "end_to_end_success_rate": (
            sum(case["task_success"] for case in metrics) / case_count
            if case_count
            else 0.0
        ),
        "stage_success_rate": stage_success_rate,
        "error_category_f1_by_label": error_category_f1["f1_by_label"],
        "specific_issue_f1_by_label": specific_issue_f1["f1_by_label"],
    }


def evaluate_detection(
    detected_issues: list[dict[str, Any]],
    corruption_record: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate coarse error type detection and fine-grained issue detection."""
    gold_issues: list[dict[str, Any]] = []
    for operation in corruption_record["operations"]:
        for error_type in operation["error_types"]:
            gold_issues.append({**operation, "error_type": error_type})

    predicted_error_categories = {(issue["error_type"],) for issue in detected_issues}
    gold_error_categories = {(issue["error_type"],) for issue in gold_issues}
    predicted_issue_labels = {_issue_label(issue) for issue in detected_issues}
    gold_issue_labels = {_issue_label(issue) for issue in gold_issues}

    return {
        "error_category": _label_metrics(
            predicted=predicted_error_categories,
            gold=gold_error_categories,
        ),
        "specific_issue": _label_metrics(
            predicted=predicted_issue_labels,
            gold=gold_issue_labels,
        ),
    }


def _label_metrics(
    *,
    predicted: set[tuple[str, ...]],
    gold: set[tuple[str, ...]],
) -> dict[str, Any]:
    labels = sorted(predicted | gold)
    tp = predicted & gold
    fp = predicted - gold
    fn = gold - predicted
    precision = len(tp) / len(predicted) if predicted else 0.0
    recall = len(tp) / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f1_by_label = [
        {
            "label": _label_to_dict(label),
            "precision": 1.0 if label in tp else 0.0,
            "recall": 1.0 if label in tp else 0.0,
        }
        for label in labels
    ]
    for item in f1_by_label:
        item["f1"] = (
            2
            * item["precision"]
            * item["recall"]
            / (item["precision"] + item["recall"])
            if item["precision"] + item["recall"]
            else 0.0
        )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": (
            sum(item["f1"] for item in f1_by_label) / len(f1_by_label)
            if f1_by_label
            else 0.0
        ),
        "exact_match": predicted == gold,
        "predicted": [_label_to_dict(label) for label in sorted(predicted)],
        "gold": [_label_to_dict(label) for label in sorted(gold)],
        "tp": [_label_to_dict(label) for label in sorted(tp)],
        "fp": [_label_to_dict(label) for label in sorted(fp)],
        "fn": [_label_to_dict(label) for label in sorted(fn)],
        "f1_by_label": f1_by_label,
    }


def _aggregate_f1_by_label(
    metric_blocks: Any,
) -> dict[str, Any]:
    blocks = list(metric_blocks)
    gold_labels = set().union(
        *(_records_to_labels(block["gold"]) for block in blocks)
    )
    label_stats: dict[tuple[str, ...], dict[str, int]] = {}
    for block in blocks:
        predicted = _records_to_labels(block["predicted"])
        gold = _records_to_labels(block["gold"])
        for label in gold_labels:
            stats = label_stats.setdefault(label, {"tp": 0, "fp": 0, "fn": 0})
            if label in gold:
                stats["tp" if label in predicted else "fn"] += 1
            elif label in predicted:
                stats["fp"] += 1

    f1_by_label = []
    for label, stats in sorted(label_stats.items()):
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_by_label.append(
            {
                "label": _label_to_dict(label),
                "precision": precision,
                "recall": recall,
                "f1": (
                    2 * precision * recall / (precision + recall)
                    if precision + recall
                    else 0.0
                ),
            }
        )

    return {
        "macro_f1": (
            sum(item["f1"] for item in f1_by_label) / len(f1_by_label)
            if f1_by_label
            else 0.0
        ),
        "f1_by_label": f1_by_label,
    }


def _records_to_labels(items: list[dict[str, Any]]) -> set[tuple[str, ...]]:
    labels: set[tuple[str, ...]] = set()
    for item in items:
        if "scope_error_type" in item:
            label = (
                str(item["error_type"]),
                str(item["scope_error_type"]),
                str(item["field"]),
            )
        elif set(item) == {"error_type"}:
            label = (str(item["error_type"]),)
        else:
            label = (str(item["error_type"]), str(item["target"]))
        labels.add(label)
    return labels


def _issue_label(item: dict[str, Any]) -> tuple[str, ...]:
    """提取 issue label 的元组表示形式。转化为 (error_type, scope_error_type, field) 或 (error_type, target) 的形式。"""
    error_type = item["error_type"]
    if error_type == "scope_error":
        return (error_type, item["scope_error_type"], item["field"])
    return (error_type, item["target"])


def _label_to_dict(label: tuple[str, ...]) -> dict[str, str]:
    """将 issue label 的元组表示形式转化为字典形式。"""
    if len(label) == 1:
        return {"error_type": label[0]}
    if label[0] == "scope_error":
        return {
            "error_type": label[0],
            "scope_error_type": label[1],
            "field": label[2],
        }
    return {"error_type": label[0], "target": label[1]}
