from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path
from typing import Any

from method.schemas import TableAnalysisConfig
from method.utils import Client, parse_json_object, resolve_case_path

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"
DEFAULT_SUMMARY_DATA_SOURCE_PROMPT_PATH = PROMPT_DIR / "summary_data_source_extraction_prompt.txt"
DEFAULT_CAPTION_DATA_SOURCE_PROMPT_PATH = PROMPT_DIR / "caption_data_source_extraction_prompt.txt"
DEFAULT_FUNCTION_LOGIC_PROMPT_PATH = PROMPT_DIR / "function_logic_extraction_prompt.txt"
DIMENSION_SOURCE_COLUMNS = {"date_code", "dim_area", "dim_price"}
METRIC_SOURCE_COLUMNS = {
    "supply_sets",
    "trade_sets",
    "dim_area",
    "dim_price",
    "dim_unit_price",
}


class SlideAnalysisAgent:
    """从 PPT representation 中抽取可执行 data source 和 calculation logic。"""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout_sec: int = 120,
        enable_thinking: bool | None = False,
        client: Any | None = None,
    ):
        """初始化 slide analysis agent。

        Args:
            model: OpenAI-compatible chat model 名称。未注入 `client` 时必填。
            api_key: OpenAI-compatible endpoint 的 API key。
            base_url: OpenAI-compatible endpoint 的 base URL。
            timeout_sec: LLM 请求超时时间，单位为秒。
            enable_thinking: 传给 `Client` 的 provider-specific thinking 开关。
            client: 可选的测试用 LLM client。

        Raises:
            ValueError: 未注入 `client` 且未提供 `model`。
        """
        if client is not None:
            self.client = client
        else:
            if model is None:
                raise ValueError("SlideAnalysisAgent requires model when client is not provided.")
            self.client = Client(
                model=model,
                api_key=api_key,
                base_url=base_url,
                timeout_sec=timeout_sec,
                enable_thinking=enable_thinking,
            )
        self.summary_data_source_prompt = DEFAULT_SUMMARY_DATA_SOURCE_PROMPT_PATH.read_text(
            encoding="utf-8"
        )
        self.caption_data_source_prompt = DEFAULT_CAPTION_DATA_SOURCE_PROMPT_PATH.read_text(
            encoding="utf-8"
        )
        self.function_logic_prompt = DEFAULT_FUNCTION_LOGIC_PROMPT_PATH.read_text(
            encoding="utf-8"
        )

    async def arun(
        self,
        *,
        ppt_representation: dict[str, Any],
        data_source_state: dict[str, Any] | None = None,
        computation_logic: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """对 parser 输出执行 Phase 1 analysis。

        Args:
            ppt_representation: Parser 输出，包含 title、summary，以及带 CSV
                路径的结构化 chart/table body。
            data_source_state: 可选的预构造 source state，包含 `summary` 和
                与 table 对应的 `captions`。提供时跳过 data-source extraction。
            computation_logic: 可选的 table calculation logic 列表。提供时跳过
                function-logic extraction。

        Returns:
            Analysis state，包含 title、summary source、每个 table 的 caption
            source，以及 `calculation_logic`。

        Raises:
            ValueError: 当模型输出不符合要求 schema 时抛出。
        """
        analysis_input = build_analysis_input(ppt_representation)
        if data_source_state is not None and len(data_source_state["captions"]) != len(
            analysis_input["tables"]
        ):
            raise ValueError("data_source_state captions must match table count.")
        if computation_logic is not None and len(computation_logic) != len(
            analysis_input["tables"]
        ):
            raise ValueError("computation_logic must match table count.")

        table_tasks = [
            self._analyze_table(
                index,
                table_input,
                data_source=(
                    data_source_state["captions"][index]
                    if data_source_state is not None
                    else None
                ),
                calculation_logic=(
                    computation_logic[index] if computation_logic is not None else None
                ),
            )
            for index, table_input in enumerate(analysis_input["tables"])
        ]
        if data_source_state is None:
            summary_data_source, *analyzed_tables = await asyncio.gather(
                self._extract_summary_data_source(analysis_input["summary"]),
                *table_tasks,
            )
        else:
            analyzed_tables = await asyncio.gather(*table_tasks)
            summary_data_source = data_source_state["summary"]

        # summary 和 caption 可能同时描述同一组 source slots。这里先保留为
        # 独立证据，后续由 `DataSourceValidationAgent` 聚合为一个
        # `final_data_source`。
        return {
            "title": {
                "text": analysis_input["title"],
                "element_id": analysis_input["title_element_id"],
            },
            "summary": {
                "text": analysis_input["summary"],
                "data_source": summary_data_source,
                "element_id": analysis_input["summary_element_id"],
            },
            "tables": analyzed_tables,
        }

    async def _analyze_table(
        self,
        index: int,
        table_input: dict[str, Any],
        data_source: dict[str, Any] | None = None,
        calculation_logic: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """抽取单个 chart/table 的 caption data source 和 calculation logic。

        Args:
            index: table 在 `analysis_input["tables"]` 中的下标。
            table_input: 单个 visible chart/table 的紧凑表示。
            data_source: 可选的预构造 caption source。提供时不调用 source
                extraction。
            calculation_logic: 可选的预构造 calculation logic。提供时不调用
                function-logic extraction。

        Returns:
            `analysis_state["tables"]` 中的一项。
        """
        if data_source is None and calculation_logic is None:
            data_source, calculation_logic = await asyncio.gather(
                self._extract_caption_data_source(table_input, index),
                self._extract_function_logic(table_input, index),
            )
        elif data_source is None:
            data_source = await self._extract_caption_data_source(table_input, index)
        elif calculation_logic is None:
            calculation_logic = await self._extract_function_logic(table_input, index)
        return {
            "caption": {
                "text": table_input["caption"],
                "data_source": data_source,
                "element_id": table_input["caption_element_id"],
            },
            "body": table_input["body"],
            "data_path": table_input["data_path"],
            "calculation_logic": calculation_logic,
        }

    async def _extract_summary_data_source(
        self,
        summary_text: str,
    ) -> dict[str, Any]:
        """从 summary text 中抽取 data-source slots。

        Args:
            summary_text: PPT summary 文本。

        Returns:
            校验后的 summary data-source object。
        """
        content = await self.client.achat(
            self.summary_data_source_prompt,
            json.dumps({"text": summary_text}, ensure_ascii=False, indent=2),
            response_format="json_object",
        )
        return _validate_data_source(
            parse_json_object(content),
            context="summary",
        )

    async def _extract_caption_data_source(
        self,
        table_input: dict[str, Any],
        index: int,
    ) -> dict[str, Any]:
        """从 caption 和可见表头中抽取 data-source slots。

        Args:
            table_input: 单个 visible chart/table 的紧凑表示。
            index: table 在 `analysis_input["tables"]` 中的下标。

        Returns:
            校验后的 caption data-source object。
        """
        payload = {
            "text": table_input["caption"],
            "row_headers": table_input["row_headers"],
            "column_headers": table_input["column_headers"],
        }
        content = await self.client.achat(
            self.caption_data_source_prompt,
            json.dumps(payload, ensure_ascii=False, indent=2),
            response_format="json_object",
        )
        return _validate_caption_data_source(
            parse_json_object(content),
            context=f"tables[{index}].caption",
        )

    async def _extract_function_logic(
        self,
        table_input: dict[str, Any],
        index: int,
    ) -> dict[str, Any]:
        """从 visible table data 中抽取 table calculation logic。

        Args:
            table_input: 单个 visible chart/table 的紧凑表示。
            index: 用于 validation error message 的 table index。

        Returns:
            序列化为字典的 `TableAnalysisConfig`。
        """
        payload = {
            "table_caption": table_input["caption"],
            "table_data": table_input["table_data"],
        }
        content = await self.client.achat(
            self.function_logic_prompt,
            json.dumps(payload, ensure_ascii=False, indent=2),
            response_format="json_object",
        )
        calculation_logic = parse_json_object(content)
        config = TableAnalysisConfig.model_validate(calculation_logic)
        _validate_calculation_logic_sources(config, context=f"tables[{index}]")
        return config.model_dump(exclude_none=True)


def build_analysis_input(ppt_representation: dict[str, Any]) -> dict[str, Any]:
    """构造发送给 analysis LLM 的紧凑 JSON payload。

    Args:
        ppt_representation: Parser 输出，包含 title、summary、captions 和 body
            CSV 路径。

    Returns:
        包含 table headers 和 CSV rows 的紧凑 analysis input。

    Raises:
        FileNotFoundError: parser 引用的 table body CSV 不存在时抛出。
        ValueError: 引用的 CSV 为空时抛出。
    """
    tables = []
    case_dir = Path(str(ppt_representation["case_dir"]))
    for table in ppt_representation["structured_tables"]:
        body = table["body"]
        data_path_ref = str(body["data_path"])
        data_path = resolve_case_path(data_path_ref, case_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Missing table CSV for analysis: {data_path}")
        header, rows, table_data = _read_csv_for_analysis(data_path)
        tables.append(
            {
                "caption": table["caption"]["text"],
                "caption_element_id": table["caption"]["element_id"],
                "body": body,
                "data_path": data_path_ref,
                "row_headers": [row[0] for row in rows if row],
                "column_headers": header[1:] if len(header) > 1 else header,
                "table_data": table_data,
            }
        )

    return {
        "title": ppt_representation["title"]["text"],
        "title_element_id": ppt_representation["title"]["element_id"],
        "summary": ppt_representation["summary"]["text"],
        "summary_element_id": ppt_representation["summary"]["element_id"],
        "tables": tables,
    }


def _validate_data_source(
    data_source: dict[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    """校验 summary data-source schema。"""
    connection = data_source.get("connection")
    filters = data_source.get("filters")
    if not isinstance(connection, dict) or not isinstance(connection.get("table"), str):
        raise ValueError(f"Analysis {context} missing connection.table.")
    if not isinstance(filters, dict):
        raise ValueError(f"Analysis {context} missing filters object.")

    required_filter_keys = {"city", "block", "start_date", "end_date"}
    if set(filters) != required_filter_keys:
        raise ValueError(
            f"Analysis {context} filters must contain exactly "
            f"{sorted(required_filter_keys)}, got {sorted(filters)}."
        )
    if not all(isinstance(filters[key], str) for key in required_filter_keys):
        raise ValueError(f"Analysis {context} filters values must be strings.")

    validated = {
        "connection": {"table": connection["table"]},
        "filters": {key: filters[key] for key in ("city", "block", "start_date", "end_date")},
    }
    return validated


def _validate_caption_data_source(
    data_source: dict[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    """校验 caption data-source schema。"""
    validated = _validate_data_source(data_source, context=context)
    select_columns = data_source.get("select_columns")
    if not isinstance(select_columns, list) or not all(
        isinstance(column, str) for column in select_columns
    ):
        raise ValueError(f"Analysis {context} select_columns must be string list.")
    validated["select_columns"] = list(select_columns)
    return validated


def _read_csv_for_analysis(path: Path) -> tuple[list[str], list[list[str]], list[dict[str, str]]]:
    """读取 parser 输出的 table body CSV，构造 analysis LLM 输入"""
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"Analysis CSV is empty: {path}")
    header = rows[0]
    body_rows = rows[1:]
    table_data = []
    for row in body_rows:
        item = {}
        for idx, name in enumerate(header):
            item[name] = row[idx] if idx < len(row) else ""
        table_data.append(item)
    return header, body_rows, table_data


def _validate_calculation_logic_sources(
    config: TableAnalysisConfig,
    *,
    context: str,
) -> None:
    for dimension in config.dimensions:
        if dimension.source_col not in DIMENSION_SOURCE_COLUMNS:
            raise ValueError(
                f"Analysis {context} dimension source_col must be a raw database "
                f"column, got {dimension.source_col!r}."
            )
    for metric in config.metrics:
        if metric.source_col not in METRIC_SOURCE_COLUMNS:
            raise ValueError(
                f"Analysis {context} metric source_col must be a raw database "
                f"column, got {metric.source_col!r}."
            )
