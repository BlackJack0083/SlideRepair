from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_json_object(text: str) -> dict[str, Any]:
    """解析模型返回的 JSON object。

    Args:
        text: 由 `response_format="json_object"` 约束的模型响应文本。

    Returns:
        解析后的 JSON object。

    Raises:
        json.JSONDecodeError: 响应不是合法 JSON 时抛出。
        ValueError: JSON 顶层不是 object 时抛出。
    """
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object, got {type(value).__name__}.")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file.

    Args:
        path: JSONL file path.

    Returns:
        Parsed records in file order.
    """
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def scalar_to_json(value: Any) -> Any:
    """Convert a pandas or NumPy scalar to a JSON-compatible value.

    Args:
        value: Scalar value.

    Returns:
        A native Python scalar or ``None`` for missing values.
    """
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value
