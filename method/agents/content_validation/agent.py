"""Interactive slide repair agent."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from .artifacts import write_content_artifacts
from .tools import (
    CONTENT_VALIDATION_TOOLS,
    ContentValidationContext,
    ContentValidationState,
)

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"
DEFAULT_PROMPT_PATH = PROMPT_DIR / "content_validation_react_prompt.txt"


class ContentValidationAgent:
    """通过 ReAct tools 验证并修复 chart、table、caption 和 summary。"""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout_sec: int = 120,
        enable_thinking: bool | None = False,
    ) -> None:
        """初始化 content validation agent。

        Args:
            model: OpenAI-compatible chat model 名称。
            api_key: OpenAI-compatible endpoint 的 API key。
            base_url: OpenAI-compatible endpoint 的 base URL。
            timeout_sec: LLM 请求超时时间，单位为秒。
            enable_thinking: 是否通过 provider-specific `extra_body` 打开模型
                thinking。

        Raises:
            ValueError: 未提供 `model` 时抛出。
        """
        if model is None:
            raise ValueError(
                "ContentValidationAgent requires model when using create_agent."
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
        *,
        analysis_state: dict[str, Any],
        client: Any,
        pptx_path: Path,
        artifact_dir: Path,
        scope_dialogue: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """验证并按 client 确认结果更新 content state。

        Args:
            analysis_state: 已经通过 `update_data_source` 写入 final datasource 和
                parser metadata 的 slide state。
            client: 共享的 benchmark client 或人工桥接对象。
            pptx_path: 待修复的源 PPTX 路径。
            artifact_dir: raw/computed/repaired 文件输出目录。
            scope_dialogue: Data-source validation 阶段已经获得 client 确认的
                scope 问答，可作为文本修复授权依据。

        Returns:
            最终 analysis state、工具调用日志、检测到的问题和 repaired artifact 路径。
        """
        artifact_dir.mkdir(parents=True, exist_ok=True)
        state: ContentValidationState = {
            "messages": [
                HumanMessage(
                    content=json.dumps(
                        build_content_payload(analysis_state),
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            ],
            "analysis_state": copy.deepcopy(analysis_state),
            "table_records": [],
            "tool_log": [],
            "detected_issues": [],
        }
        system_prompt = self.prompt
        if scope_dialogue:
            system_prompt += (
                "\n\nData-source validation 阶段已完成以下 client 对话：\n"
                + json.dumps(scope_dialogue, ensure_ascii=False, indent=2)
                + "\n这些回复可作为 scope 文本修复的确认依据；不要为同一修正再次调用 ask_client。"
            )
        agent = create_agent(
            model=self.llm,
            tools=CONTENT_VALIDATION_TOOLS,
            system_prompt=system_prompt,
            state_schema=ContentValidationState,
            context_schema=ContentValidationContext,
        )
        result = await agent.ainvoke(
            state,
            context=ContentValidationContext(
                client=client,
                artifact_dir=artifact_dir,
            ),
        )
        result_state = result["analysis_state"]
        table_records = result["table_records"]
        repaired_artifacts = write_content_artifacts(
            source_pptx=pptx_path,
            analysis_state=result_state,
            artifact_dir=artifact_dir,
        )
        return {
            "analysis_state": result_state,
            "table_records": table_records,
            "tool_log": result["tool_log"],
            "detected_issues": result["detected_issues"],
            "repaired_artifacts": repaired_artifacts,
        }


def build_content_payload(state: dict[str, Any]) -> dict[str, Any]:
    """构造给 content validation ReAct agent 的紧凑 payload。

    Args:
        state: 已归一化 datasource、并带 body metadata 的 slide state。

    Returns:
        包含 summary 和 tables 的 JSON payload。大表数据由 tools 按需读取。
    """
    return {
        "title": {
            "element_id": state["title"]["element_id"],
            "text": state["title"]["text"],
        },
        "data_source": state["final_data_source"],
        "summary": {
            "element_id": state["summary"]["element_id"],
            "text": state["summary"]["text"],
        },
        "tables": [
            {
                "table_index": table_index,
                "caption": {
                    "element_id": table_state["caption"]["element_id"],
                    "text": table_state["caption"]["text"],
                },
                "body": table_state["body"],
                "select_columns": table_state["caption"]["data_source"][
                    "select_columns"
                ],
                "calculation_logic": table_state["calculation_logic"],
            }
            for table_index, table_state in enumerate(state["tables"])
        ],
    }
