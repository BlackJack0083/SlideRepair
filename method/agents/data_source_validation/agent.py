"""数据源验证 agent。

该 agent 负责 scope 验证闭环：
1. 将 summary 和 caption 中的 data-source 候选描述交给 ReAct LLM。
2. 让 ReAct LLM 合成 slide 级 final data source。
3. ReAct LLM 按需调用 `ask_client` 和 `slot_query`。
4. 返回最终确认的 `analysis_state["final_data_source"]`。
"""

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field

from .tools import (
    DATA_SOURCE_VALIDATION_TOOLS,
    REAL_ESTATE_TABLES,
    DataSourceValidationContext,
    DataSourceValidationState,
)

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"
DEFAULT_PROMPT_PATH = PROMPT_DIR / "data_source_validation_react_prompt.txt"


class Connection(BaseModel):
    """数据库连接。"""

    model_config = ConfigDict(extra="forbid")

    table: str = Field(min_length=1)


class Filters(BaseModel):
    """Data source 过滤条件。"""

    model_config = ConfigDict(extra="forbid")

    city: str = Field(min_length=1)
    block: str = Field(min_length=1)
    start_date: str = Field(min_length=1)
    end_date: str = Field(min_length=1)


class DataSource(BaseModel):
    """Data source validation agent 的最终输出。"""

    model_config = ConfigDict(extra="forbid")

    connection: Connection
    filters: Filters


class DataSourceValidationAgent:
    """通过 ReAct 和 client 多轮交互验证、修正 data-source slots。

    该类只负责 data-source validation 阶段：把 analysis 阶段抽取到的
    summary/caption source candidates 交给 ReAct agent，由 agent 调用数据库
    查询工具和 client 澄清工具，最终返回 slide-level `final_data_source`。
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout_sec: int = 120,
        enable_thinking: bool | None = False,
    ) -> None:
        """初始化数据源验证 agent。

        Args:
            model: OpenAI-compatible chat model 名称。
            api_key: OpenAI-compatible endpoint 的 API key。
            base_url: OpenAI-compatible endpoint 的 base URL。
            timeout_sec: LLM 请求超时时间，单位为秒。
            enable_thinking: 是否通过 provider-specific `extra_body` 打开模型
                thinking。Qwen tool-calling 默认应关闭，避免内部 parameter
                标签泄漏到 tool arguments。

        Returns:
            None.

        Raises:
            ValueError: 未提供 `model` 时抛出。
        """
        if model is None:
            raise ValueError(
                "DataSourceValidationAgent requires model when using create_agent."
            )

        self.prompt = DEFAULT_PROMPT_PATH.read_text(encoding="utf-8")
        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url=base_url,
            timeout=timeout_sec,
            temperature=0,
            extra_body={"enable_thinking": enable_thinking}
            if enable_thinking is not None
            else None,
        )

    async def arun(
        self,
        analysis_state: dict[str, Any],
        client: Any,
    ) -> dict[str, Any]:
        """让 ReAct agent 合成并验证 slide 级 data-source slots。

        Args:
            analysis_state: 当前 analysis state，包含 summary/caption 抽取结果和可选的 client-confirmed `final_data_source` patch。
            client: 具备 `respond(request)` 的 client simulator 或人工桥接对象。

        Returns:
            包含 `final_data_source`、`tool_log` 和 `detected_issues` 的结果。
        """
        user_payload = build_validation_payload(analysis_state)
        state: DataSourceValidationState = {
            "messages": [
                HumanMessage(
                    content=json.dumps(
                        user_payload,
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            ],
            "tool_log": [],
            "detected_issues": [],
        }
        agent = create_agent(
            model=self.llm,
            tools=DATA_SOURCE_VALIDATION_TOOLS,
            system_prompt=self.prompt,
            response_format=ToolStrategy(
                DataSource,
                handle_errors=True,
            ),
            state_schema=DataSourceValidationState,
            context_schema=DataSourceValidationContext,
        )
        result = await agent.ainvoke(
            state,
            context=DataSourceValidationContext(client=client),
        )
        output: DataSource = result["structured_response"]
        final_data_source = _validate_final_data_source(output.model_dump())
        return {
            "final_data_source": final_data_source,
            "tool_log": result["tool_log"],
            "detected_issues": result["detected_issues"],
        }


def build_validation_payload(
    analysis_state: dict[str, Any],
) -> dict[str, Any]:
    """把 analysis state 转成 ReAct prompt payload。

    Args:
        analysis_state: 当前 analysis state，包含 summary/caption 抽取结果。

    Returns:
        包含 source candidates 的 JSON payload。

    Raises:
        KeyError: `summary`、`tables` 或 table caption 缺少必要字段时抛出。
    """
    candidates = []
    summary_source = analysis_state["summary"]["data_source"]
    if summary_source["connection"]["table"] or any(summary_source["filters"].values()):
        candidates.append(
            {
                "source": "summary",
                "data_source": {
                    "connection": summary_source["connection"],
                    "filters": summary_source["filters"],
                },
            }
        )
    for table in analysis_state["tables"]:
        caption = table["caption"]
        data_source = caption["data_source"]
        candidates.append(
            {
                "source": "caption",
                "data_source": {
                    "connection": data_source["connection"],
                    "filters": data_source["filters"],
                },
            }
        )
    return {"source_candidates": candidates}


def _validate_final_data_source(data_source: dict[str, Any]) -> dict[str, Any]:
    table = data_source["connection"]["table"]
    filters = data_source["filters"]
    if table not in REAL_ESTATE_TABLES:
        raise ValueError(f"Data source validation returned unknown table: {table}")
    start_date = date.fromisoformat(filters["start_date"])
    end_date = date.fromisoformat(filters["end_date"])
    if start_date > end_date:
        raise ValueError(
            "Data source validation returned invalid date range: "
            f"{filters['start_date']} > {filters['end_date']}"
        )
    return data_source
