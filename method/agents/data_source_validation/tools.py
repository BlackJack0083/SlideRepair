"""DataSourceValidationAgent 使用的 LangChain/LangGraph tools。"""

from __future__ import annotations

import json
import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

import pandas as pd
from langchain.agents import AgentState
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

from method.database import query as database_query

REAL_ESTATE_TABLES = [
    "beijing_new_house",
    "beijing_resale_house",
    "guangzhou_new_house",
    "guangzhou_resale_house",
    "shenzhen_new_house",
    "shenzhen_resale_house",
]


class DataSourceQueryTool:
    """用真实房地产数据库验证 data-source slots。

    该工具刻意只返回证据，不直接给出错误类型。ReAct agent 负责判断 slots 是
    Missing、Invalid、Unmatch 还是 Correct。

    默认检查只在当前 table 内进行。这里不做跨表诊断，因为 data-source 验证
    只需要判断当前 PPT 声称的数据源是否能检索到数据。
    """

    def __init__(self) -> None:
        """初始化数据库驱动的 data-source checker。"""
        self._existing_tables_cache: set[str] | None = None

    def run(
        self,
        table: str,
        city: str,
        block: str,
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        """检查 data-source slots 是否能从数据库中检索到行。

        Args:
            table: PPT 声称的数据库表名。
            city: PPT 声称的 city slot。
            block: PPT 声称的 block/district slot。
            start_date: PPT 声称的开始日期，格式为 `YYYY-MM-DD`。
            end_date: PPT 声称的结束日期，格式为 `YYYY-MM-DD`。

        Returns:
            供 ReAct prompt 使用的布尔证据字典。
        """
        start = pd.to_datetime(start_date, errors="coerce")
        end = pd.to_datetime(end_date, errors="coerce")
        date_valid = not pd.isna(start) and not pd.isna(end) and bool(start <= end)
        if not date_valid:
            return {
                "date_range_error": (
                    "start_date/end_date must be valid dates and start_date must be less than or equal to end_date."
                ),
            }

        if self._existing_tables_cache is None:
            allowed_tables = ", ".join(
                f"'{table_name}'" for table_name in REAL_ESTATE_TABLES
            )
            result = database_query(
                f"""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name IN ({allowed_tables})
                """,  # nosec - table names come from the static allowlist.
            )
            self._existing_tables_cache = {
                str(row["table_name"]) for _, row in result.iterrows()
            }

        table_exists = table in self._existing_tables_cache
        if not table_exists:
            return {"table_exists": False}

        scope_checks = self._query_table_scope(
            table=table,
            city=city,
            block=block,
        )
        full_scope_result = database_query(
            f"""
            SELECT EXISTS (
                SELECT 1
                FROM public.{table}
                WHERE city = :city
                  AND block = :block
                  AND date_code >= :start_date
                  AND date_code <= :end_date
                LIMIT 1
            ) AS full_scope
            """,  # nosec - table name comes from the static allowlist.
            {
                "city": city,
                "block": block,
                "start_date": start_date,
                "end_date": end_date,
            },
        )
        full_scope = bool(full_scope_result.iloc[0]["full_scope"])

        return {
            "table_exists": True,
            "city_exists": scope_checks["table_city"],
            "block_exists": scope_checks["table_block"],
            "table_city_matches": scope_checks["table_city"],
            "city_block_matches": scope_checks["table_city_block"],
            "table_city_block_matches": scope_checks["table_city_block"],
            "full_scope_has_data": full_scope,
        }

    def _query_table_scope(
        self,
        *,
        table: str,
        city: str,
        block: str,
    ) -> dict[str, bool]:
        """在当前表内检查 city/block 组合是否有数据。

        Args:
            table: 已通过 allowlist 和存在性检查的数据库表名。
            city: PPT 声称的城市 slot。
            block: PPT 声称的板块或区域 slot。

        Returns:
            包含 `table_city`、`table_block`、`table_city_block` 和
            `full_scope` 四个布尔检查结果的字典。`full_scope` 默认是
            `False`，由 `run` 在日期合法时补充。
        """
        result = database_query(
            f"""
            SELECT
                EXISTS (
                    SELECT 1
                    FROM public.{table}
                    WHERE city = :city
                    LIMIT 1
                ) AS table_city,
                EXISTS (
                    SELECT 1
                    FROM public.{table}
                    WHERE block = :block
                    LIMIT 1
                ) AS table_block,
                EXISTS (
                    SELECT 1
                    FROM public.{table}
                    WHERE city = :city AND block = :block
                    LIMIT 1
                ) AS table_city_block
            """,  # nosec - table name comes from the static allowlist.
            {
                "city": city,
                "block": block,
            },
        )
        row = result.iloc[0]
        return {
            "table_city": bool(row["table_city"]),
            "table_block": bool(row["table_block"]),
            "table_city_block": bool(row["table_city_block"]),
            "full_scope": False,
        }


@dataclass
class DataSourceValidationContext:
    """Data source validation tools 的运行上下文。

    Args:
        client: 具备 `respond(request)` 的 benchmark client。
        query_tool: 数据库 slot evidence 工具。
    """

    client: Any
    query_tool: Any = field(default_factory=DataSourceQueryTool)


class DataSourceValidationState(AgentState):
    """Data source validation agent 的 LangGraph state。

    Args:
        tool_log: 本轮 ReAct 过程中所有 data-source validation 工具调用记录。
        detected_issues: 本轮 ReAct 过程中显式向 client 澄清的 data-source 问题。
    """

    tool_log: Annotated[list[dict[str, Any]], operator.add]
    detected_issues: Annotated[list[dict[str, Any]], operator.add]


@tool
def slot_query(
    table: str,
    city: str,
    block: str,
    start_date: str,
    end_date: str,
    runtime: ToolRuntime,
) -> Command:
    """用数据库验证 table/city/block/start_date/end_date slots。

    Args:
        table: PPT 声称的数据库表名。
        city: PPT 声称的城市 slot。
        block: PPT 声称的板块或区域 slot。
        start_date: PPT 声称的开始日期，格式为 `YYYY-MM-DD`。
        end_date: PPT 声称的结束日期，格式为 `YYYY-MM-DD`。

    Returns:
        可供 ReAct agent 判断 slot 状态的数据库证据字典。
    """
    result = runtime.context.query_tool.run(
        table=table,
        city=city,
        block=block,
        start_date=start_date,
        end_date=end_date,
    )
    args = {
        "table": table,
        "city": city,
        "block": block,
        "start_date": start_date,
        "end_date": end_date,
    }
    return _command(
        runtime,
        result,
        {
            "tool_log": [
                {"tool": "slot_query", "args": args, "result": result},
            ]
        },
    )


@tool
def ask_client(
    error_type: Literal["scope_error"],
    scope_error_type: Literal["missing", "invalid", "unmatch", "conflict"],
    field: Literal["city", "block", "time_range"],
    description: str,
    runtime: ToolRuntime,
) -> Command:
    """向 client 请求 data-source slot 澄清。

    Args:
        error_type: benchmark error type，data-source validation 中通常是
            `scope_error`。
        scope_error_type: scope 细分类型，例如 `missing`、`invalid`、
            `unmatch` 或 `conflict`。
        field: 需要 client 澄清或修正的单个 feedback 字段。时间范围问题
            使用 `time_range`，不要拆成 `start_date` 或 `end_date`。
        description: agent 对当前问题的简短说明。
    Returns:
        面向 ReAct agent 的客户式回复，只包含 `response`。
    """
    request: dict[str, Any] = {
        "error_type": error_type,
        "scope_error_type": scope_error_type,
        "field": field,
        "description": description,
    }
    response = runtime.context.client.respond(request)

    if "response" not in response or not isinstance(response["response"], str):
        raise ValueError(
            f"ask_client expects client to return only response: {response}"
        )
    visible_response = {"response": response["response"]}

    log_item = {
        "tool": "ask_client",
        "request": request,
        "response": visible_response,
        "confirmed": bool(response.get("confirmed", False)),
    }
    detected_issue = {
        **request,
        "evidence": description,
        "client_response": response["response"],
        "confirmed": bool(response.get("confirmed", False)),
    }
    detected_issue.pop("description")
    return _command(
        runtime,
        visible_response,
        {
            "tool_log": [log_item],
            "detected_issues": [detected_issue],
        },
    )


DATA_SOURCE_VALIDATION_TOOLS = [slot_query, ask_client]


def _command(
    runtime: ToolRuntime,
    result: dict[str, Any],
    update: dict[str, Any],
) -> Command:
    if runtime.tool_call_id is not None:
        update = {
            **update,
            "messages": [
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False, default=str),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    return Command(update=update)
