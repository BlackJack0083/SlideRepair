from __future__ import annotations

import asyncio
import json
import re
import unittest
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pandas as pd
from langchain.tools import ToolRuntime

from method.agents.client import ClientAgent
from method.agents.data_source_validation import (
    DataSourceValidationAgent,
    build_validation_payload,
)
from method.agents.data_source_validation.agent import _validate_final_data_source
from method.agents.data_source_validation.tools import (
    DATA_SOURCE_VALIDATION_TOOLS,
    DataSourceQueryTool,
    DataSourceValidationContext,
)

SLOT_REPLY_RE = re.compile(
    r"(table|city|block|start_date|end_date)=(.*?)(?=,\s*(?:table|city|block|start_date|end_date)=|\.?$)"
)


class FakeAgent:
    async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        user_payload = json.loads(payload["messages"][0].content)
        conflict_fields = _fake_conflict_fields(user_payload)
        if conflict_fields:
            data_source = _fake_synthesize_final_data_source(user_payload)
            data_source["filters"]["city"] = "Beijing"
            return _structured_response(data_source)

        data_source = _fake_synthesize_final_data_source(user_payload)
        filters = data_source["filters"]
        slots = {
            "table": data_source["connection"]["table"],
            "city": filters["city"],
            "block": filters["block"],
            "start_date": filters["start_date"],
            "end_date": filters["end_date"],
        }

        if not slots["city"]:
            data_source["filters"]["city"] = "Beijing"
            return _structured_response(data_source)

        if slots["city"] == "Beijing" and slots["block"] == "Nanshan CBD":
            data_source["connection"]["table"] = "shenzhen_new_house"
            data_source["filters"]["city"] = "Shenzhen"
            return _structured_response(data_source)

        return _structured_response(data_source)


class FakeClient:
    def respond(self, request: dict[str, Any]) -> dict[str, Any]:
        field = request["field"]
        if request.get("scope_error_type") == "unmatch" and field in {"city", "block"}:
            return {
                "response": (
                    "Please use table=shenzhen_new_house, "
                    "city=Shenzhen, block=Nanshan CBD."
                ),
                "confirmed": True,
            }
        if field == "city":
            return {"response": "Please use city=Beijing.", "confirmed": True}
        return {
            "response": "I do not have a confirmed correction for this request.",
            "confirmed": False,
        }


class FakeLLMClient:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def chat(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": json.loads(user_prompt),
                "kwargs": kwargs,
            }
        )
        return "Sure, please use city=Beijing."


class ToolCallingFakeAgent:
    """用真实 LangChain tools 模拟 ReAct agent 的关键控制流。

    Args:
        tools: `DataSourceValidationAgent` 注入给 agent 的 LangChain tools。
    """

    def __init__(self, tools: list[Any], state: dict[str, Any], context: Any) -> None:
        """初始化测试 agent。

        Args:
            tools: `slot_query` 和 `ask_client`。

        Returns:
            None.
        """
        self.tools = {tool.name: tool for tool in tools}
        self.state = state
        self.context = context

    def invoke_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        command = self.tools[name].func(
            **args,
            runtime=ToolRuntime(
                state=self.state,
                context=self.context,
                config={},
                stream_writer=lambda _: None,
                tool_call_id=f"fake-{name}",
                store=None,
            ),
        )
        update = dict(command.update)
        messages = update.pop("messages", [])
        if "tool_log" in update:
            self.state["tool_log"] = [*self.state["tool_log"], *update.pop("tool_log")]
        if "detected_issues" in update:
            self.state["detected_issues"] = [
                *self.state["detected_issues"],
                *update.pop("detected_issues"),
            ]
        self.state.update(update)
        return json.loads(messages[0].content) if messages else {}

    async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        """模拟一次 ReAct 调用。

        Args:
            payload: LangChain agent 输入，包含序列化后的 validation payload。

        Returns:
            与 LangChain agent 兼容的消息结果。
        """
        user_payload = json.loads(payload["messages"][0].content)
        data_source = _fake_synthesize_final_data_source(user_payload)
        conflict_fields = _fake_conflict_fields(user_payload)
        if conflict_fields:
            response = self._ask_client(
                fields=conflict_fields,
                scope_error_type="conflict",
                description=f"Conflicting data-source fields: {conflict_fields}",
            )
            if not _merge_client_reply(data_source, response):
                raise ValueError(
                    "Data source validation requires client input: conflict fields"
                )

        missing_fields = _missing_scope_fields(data_source)
        if missing_fields:
            response = self._ask_client(
                fields=missing_fields,
                scope_error_type="missing",
                description=f"Missing data-source fields: {missing_fields}",
            )
            if not _merge_client_reply(data_source, response):
                raise ValueError(
                    "Data source validation requires client input: missing fields"
                )

        evidence = self.invoke_tool("slot_query", _slots(data_source))
        if "date_range_error" in evidence:
            response = self._ask_client(
                fields=["time_range"],
                scope_error_type="invalid",
                description=evidence["date_range_error"],
            )
            if not _merge_client_reply(data_source, response):
                raise ValueError(
                    "Data source validation requires client input: invalid date range"
                )
            evidence = self.invoke_tool("slot_query", _slots(data_source))

        if not evidence.get("table_exists", True):
            raise ValueError("Data source extraction returned an unknown table")
        elif not evidence["full_scope_has_data"]:
            response = self._ask_client(
                fields=["city", "block"],
                scope_error_type="unmatch",
                description="Current city/block combination has no database rows.",
            )
            if not _merge_client_reply(data_source, response):
                raise ValueError(
                    "Data source validation requires client input: scope unmatch"
                )
            self.invoke_tool("slot_query", _slots(data_source))

        return _structured_response(data_source)

    def _ask_client(
        self,
        *,
        fields: list[str],
        scope_error_type: str,
        description: str,
    ) -> dict[str, Any]:
        """通过真实 `ask_client` 发起澄清请求。

        Args:
            fields: 需要 client 澄清或修正的 data-source slots。
            scope_error_type: scope 错误细分标签。
            description: 请求说明。
        Returns:
            client tool 返回的 response。
        """
        response: dict[str, Any] = {}
        for field in fields:
            args = {
                "error_type": "scope_error",
                "scope_error_type": scope_error_type,
                "field": field,
                "description": description,
            }
            response = self.invoke_tool("ask_client", args)
            if SLOT_REPLY_RE.search(str(response.get("response", ""))):
                break
        return response


@dataclass
class FakeMessage:
    content: str


def run_data_source_tool(
    name: str,
    args: dict[str, Any],
    *,
    client: Any,
    query_tool: Any | None = None,
    state: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tool = next(item for item in DATA_SOURCE_VALIDATION_TOOLS if item.name == name)
    runtime_state = state or {"messages": [], "tool_log": [], "detected_issues": []}
    command = tool.func(
        **args,
        runtime=ToolRuntime(
            state=runtime_state,
            context=DataSourceValidationContext(
                client=client,
                query_tool=query_tool or DataSourceQueryTool(),
            ),
            config={},
            stream_writer=lambda _: None,
            tool_call_id=f"fake-{name}",
            store=None,
        ),
    )
    update = dict(command.update)
    messages = update.pop("messages", [])
    if "tool_log" in update:
        runtime_state["tool_log"] = [
            *runtime_state["tool_log"],
            *update.pop("tool_log"),
        ]
    if "detected_issues" in update:
        runtime_state["detected_issues"] = [
            *runtime_state["detected_issues"],
            *update.pop("detected_issues"),
        ]
    runtime_state.update(update)
    result = json.loads(messages[0].content) if messages else {}
    return result, runtime_state


class ScenarioQueryTool:
    """可控数据库证据：北京 + Nanshan CBD 被视为组合不匹配。

    Args:
        无。
    """

    def __init__(self) -> None:
        """初始化查询记录器。

        Args:
            无。

        Returns:
            None.
        """
        self.calls: list[dict[str, str]] = []

    def run(
        self,
        table: str,
        city: str,
        block: str,
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        """返回测试用数据库证据。

        Args:
            table: 待验证的数据库表名。
            city: 待验证的城市 slot。
            block: 待验证的板块或区域 slot。
            start_date: 待验证的开始日期。
            end_date: 待验证的结束日期。

        Returns:
            与 `DataSourceQueryTool.run` 兼容的证据字典。
        """
        slots = {
            "table": table,
            "city": city,
            "block": block,
            "start_date": start_date,
            "end_date": end_date,
        }
        self.calls.append(slots)
        full_scope_rows = not (
            table == "beijing_new_house"
            and city == "Beijing"
            and block == "Nanshan CBD"
        )
        return {
            "table_exists": True,
            "city_exists": True,
            "block_exists": True,
            "table_city_matches": True,
            "city_block_matches": full_scope_rows,
            "table_city_block_matches": full_scope_rows,
            "full_scope_has_data": full_scope_rows,
        }


class DateRangeQueryTool:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def run(
        self,
        table: str,
        city: str,
        block: str,
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        slots = {
            "table": table,
            "city": city,
            "block": block,
            "start_date": start_date,
            "end_date": end_date,
        }
        self.calls.append(slots)
        if start_date > end_date:
            return {
                "date_range_error": (
                    "start_date/end_date must be valid dates and start_date must be "
                    "less than or equal to end_date."
                )
            }
        return {
            "table_exists": True,
            "city_exists": True,
            "block_exists": True,
            "table_city_matches": True,
            "city_block_matches": True,
            "table_city_block_matches": True,
            "full_scope_has_data": True,
        }


class FakeDataSourceValidationAgent(DataSourceValidationAgent):
    def __init__(self) -> None:
        pass

    async def arun(
        self,
        analysis_state: dict[str, Any],
        client: Any,
    ) -> dict[str, Any]:
        del client
        result = await FakeAgent().ainvoke(
            {
                "messages": [
                    FakeMessage(
                        json.dumps(
                            build_validation_payload(analysis_state),
                            ensure_ascii=False,
                        )
                    )
                ]
            }
        )
        return {
            "final_data_source": result["structured_response"],
            "tool_log": [],
            "detected_issues": [],
        }


class ToolCallingDataSourceValidationAgent(DataSourceValidationAgent):
    def __init__(self, query_tool: ScenarioQueryTool) -> None:
        self.query_tool = query_tool

    async def arun(
        self,
        analysis_state: dict[str, Any],
        client: Any,
    ) -> dict[str, Any]:
        state = {"messages": [], "tool_log": [], "detected_issues": []}
        agent = ToolCallingFakeAgent(
            DATA_SOURCE_VALIDATION_TOOLS,
            state,
            DataSourceValidationContext(client=client, query_tool=self.query_tool),
        )
        result = await agent.ainvoke(
            {
                "messages": [
                    FakeMessage(
                        json.dumps(
                            build_validation_payload(analysis_state),
                            ensure_ascii=False,
                        )
                    )
                ]
            }
        )
        return {
            "final_data_source": result["structured_response"],
            "tool_log": state["tool_log"],
            "detected_issues": state["detected_issues"],
        }


class DataSourceValidationAgentTest(unittest.TestCase):
    def test_run_with_client_returns_agent_confirmed_missing_slot_repair(self) -> None:
        state = _analysis_state(city="", block="Liangxiang")
        agent = FakeDataSourceValidationAgent()
        result = asyncio.run(agent.arun(state, FakeClient()))
        final_data_source = result["final_data_source"]
        interactions = result["tool_log"]

        self.assertEqual(final_data_source["filters"]["city"], "Beijing")
        self.assertEqual(interactions, [])

    def test_run_with_client_returns_agent_confirmed_unmatch_repair(self) -> None:
        state = _analysis_state(city="Beijing", block="Nanshan CBD")
        agent = FakeDataSourceValidationAgent()
        result = asyncio.run(agent.arun(state, FakeClient()))
        final_data_source = result["final_data_source"]
        interactions = result["tool_log"]

        self.assertEqual(final_data_source["connection"]["table"], "shenzhen_new_house")
        self.assertEqual(final_data_source["filters"]["city"], "Shenzhen")
        self.assertEqual(final_data_source["filters"]["block"], "Nanshan CBD")
        self.assertEqual(interactions, [])

    def test_run_with_client_lets_agent_detect_candidate_conflict(self) -> None:
        state = _analysis_state(city="Beijing", block="Liangxiang")
        state["tables"][0]["caption"]["data_source"]["filters"]["city"] = "Shenzhen"
        agent = FakeDataSourceValidationAgent()
        result = asyncio.run(agent.arun(state, FakeClient()))
        final_data_source = result["final_data_source"]
        interactions = result["tool_log"]

        self.assertEqual(final_data_source["filters"]["city"], "Beijing")
        self.assertEqual(interactions, [])

    def test_ask_client_records_interaction(self) -> None:
        client = FakeClient()

        response, state = run_data_source_tool(
            "ask_client",
            {
                "error_type": "scope_error",
                "scope_error_type": "missing",
                "field": "city",
                "description": "city is missing",
            },
            client=client,
        )

        self.assertEqual(response, {"response": "Please use city=Beijing."})
        self.assertIn("city=Beijing", response["response"])
        self.assertNotIn("matched", response)
        self.assertNotIn("state_patch", response)
        self.assertNotIn("city", response)
        self.assertEqual(state["tool_log"][0]["tool"], "ask_client")
        self.assertEqual(state["tool_log"][0]["request"]["field"], "city")
        self.assertEqual(state["tool_log"][0]["response"], response)
        self.assertEqual(state["detected_issues"][0]["field"], "city")
        self.assertEqual(state["detected_issues"][0]["evidence"], "city is missing")
        self.assertNotIn("target", state["tool_log"][0]["request"])
        self.assertNotIn("target", state["detected_issues"][0])
        self.assertTrue(state["detected_issues"][0]["confirmed"])

    def test_ask_client_returns_plain_customer_reply_when_unmatched(self) -> None:
        response, state = run_data_source_tool(
            "ask_client",
            {
                "error_type": "scope_error",
                "scope_error_type": "missing",
                "field": "block",
                "description": "block is missing",
            },
            client=ClientAgent(feedback_items=[]),
        )

        self.assertEqual(
            response,
            {"response": "I do not have a confirmed correction for this request."},
        )
        self.assertEqual(state["tool_log"][0]["tool"], "ask_client")
        self.assertEqual(state["tool_log"][0]["response"], response)
        self.assertEqual(state["detected_issues"][0]["field"], "block")
        self.assertFalse(state["detected_issues"][0]["confirmed"])

    def test_client_agent_matches_scope_feedback_by_field_and_subtype(self) -> None:
        client = ClientAgent(
            feedback_items=[
                {
                    "scope_error_type": "unmatch",
                    "field": "city",
                    "response": (
                        "Please use table=shenzhen_new_house, "
                        "city=Shenzhen, block=Nanshan CBD."
                    ),
                }
            ]
        )
        response = client.respond(
            {
                "scope_error_type": "unmatch",
                "field": "city",
            }
        )

        self.assertEqual(
            response,
            {
                "response": (
                    "Please use table=shenzhen_new_house, "
                    "city=Shenzhen, block=Nanshan CBD."
                ),
                "confirmed": True,
            },
        )

    def test_client_agent_rejects_different_scope_subtype(self) -> None:
        client = ClientAgent(
            feedback_items=[
                {
                    "scope_error_type": "invalid",
                    "field": "time_range",
                    "response": (
                        "Please use start_date=2020-01-01, end_date=2024-12-31."
                    ),
                }
            ]
        )

        response = client.respond(
            {
                "field": "time_range",
                "scope_error_type": "missing",
            }
        )

        self.assertFalse(response["confirmed"])

    def test_client_agent_rejects_different_scope_field(self) -> None:
        client = ClientAgent(
            feedback_items=[
                {
                    "scope_error_type": "invalid",
                    "field": "city",
                    "response": "Please use city=Beijing.",
                }
            ]
        )

        response = client.respond(
            {
                "field": "block",
                "scope_error_type": "invalid",
            }
        )

        self.assertFalse(response["confirmed"])

    def test_client_agent_rejects_different_content_target(self) -> None:
        client = ClientAgent(
            feedback_items=[
                {
                    "error_type": "claim_error",
                    "target": "caption",
                    "response": "Yes, please apply the proposed update.",
                }
            ]
        )

        response = client.respond(
            {
                "error_type": "claim_error",
                "target": "summary",
            }
        )

        self.assertFalse(response["confirmed"])

    def test_client_agent_llm_mode_rewrites_matched_response(self) -> None:
        simulator_client = FakeLLMClient()
        client = ClientAgent(
            feedback_items=[
                {
                    "scope_error_type": "invalid",
                    "field": "city",
                    "response": "Please use city=Beijing.",
                }
            ],
            mode="llm",
            client=simulator_client,
        )

        response = client.respond(
            {
                "field": "city",
                "error_type": "scope_error",
                "scope_error_type": "invalid",
            }
        )

        self.assertEqual(
            response,
            {"response": "Sure, please use city=Beijing.", "confirmed": True},
        )
        self.assertEqual(len(simulator_client.calls), 1)
        self.assertEqual(
            simulator_client.calls[0]["user_prompt"]["matched_feedback"],
            {"response": "Please use city=Beijing."},
        )

    def test_client_agent_llm_mode_does_not_call_model_without_match(self) -> None:
        simulator_client = FakeLLMClient()
        client = ClientAgent(feedback_items=[], mode="llm", client=simulator_client)

        response = client.respond(
            {
                "field": "city",
                "error_type": "scope_error",
                "scope_error_type": "invalid",
            }
        )

        self.assertEqual(
            response,
            {
                "response": "I do not have a confirmed correction for this request.",
                "confirmed": False,
            },
        )
        self.assertEqual(simulator_client.calls, [])

    def test_validation_payload_keeps_only_scope_slots(self) -> None:
        state = _analysis_state(city="Beijing", block="Liangxiang")

        payload = build_validation_payload(state)

        self.assertEqual(
            [candidate["source"] for candidate in payload["source_candidates"]],
            ["summary", "caption"],
        )
        caption_candidate = payload["source_candidates"][1]["data_source"]
        self.assertNotIn("select_columns", caption_candidate)
        self.assertEqual(caption_candidate["filters"]["block"], "Liangxiang")

    def test_validation_payload_skips_summary_without_scope_claim(self) -> None:
        state = _analysis_state(city="Beijing", block="Liangxiang")
        state["summary"]["data_source"] = {
            "connection": {"table": ""},
            "filters": {
                "city": "",
                "block": "",
                "start_date": "",
                "end_date": "",
            },
        }

        payload = build_validation_payload(state)

        self.assertEqual(payload["source_candidates"][0]["source"], "caption")

    def test_tool_calling_agent_records_missing_slot_interaction(self) -> None:
        state = _analysis_state(city="", block="Liangxiang")
        query_tool = ScenarioQueryTool()
        agent = ToolCallingDataSourceValidationAgent(query_tool)

        result = asyncio.run(agent.arun(state, FakeClient()))
        final_data_source = result["final_data_source"]
        tool_log = result["tool_log"]

        self.assertEqual(final_data_source["filters"]["city"], "Beijing")
        self.assertEqual(
            [item["tool"] for item in tool_log], ["ask_client", "slot_query"]
        )
        ask_client_log = tool_log[0]
        self.assertEqual(ask_client_log["request"]["scope_error_type"], "missing")
        self.assertEqual(ask_client_log["request"]["field"], "city")
        self.assertEqual(len(query_tool.calls), 1)

    def test_tool_calling_agent_queries_then_repairs_unmatch(self) -> None:
        state = _analysis_state(city="Beijing", block="Nanshan CBD")
        query_tool = ScenarioQueryTool()
        agent = ToolCallingDataSourceValidationAgent(query_tool)

        result = asyncio.run(agent.arun(state, FakeClient()))
        final_data_source = result["final_data_source"]
        tool_log = result["tool_log"]

        self.assertEqual(final_data_source["connection"]["table"], "shenzhen_new_house")
        self.assertEqual(final_data_source["filters"]["city"], "Shenzhen")
        self.assertEqual(
            [item["tool"] for item in tool_log],
            ["slot_query", "ask_client", "slot_query"],
        )
        self.assertEqual(tool_log[1]["request"]["scope_error_type"], "unmatch")
        self.assertEqual(len(query_tool.calls), 2)

    def test_tool_calling_agent_asks_time_range_for_invalid_dates(self) -> None:
        state = _real_estate_analysis_state(
            summary_text="Beijing Liangxiang market from 2026 to 2024.",
            caption_text="Beijing Liangxiang Statistics (2026-2024)",
            table="beijing_new_house",
            city="Beijing",
            block="Liangxiang",
            start_date="2026-01-01",
            end_date="2024-12-31",
        )
        query_tool = DateRangeQueryTool()
        agent = ToolCallingDataSourceValidationAgent(query_tool)
        client = ClientAgent(
            feedback_items=[
                {
                    "scope_error_type": "invalid",
                    "field": "time_range",
                    "response": (
                        "Please use start_date=2020-01-01, end_date=2024-12-31."
                    ),
                }
            ]
        )

        result = asyncio.run(agent.arun(state, client))
        final_data_source = result["final_data_source"]
        tool_log = result["tool_log"]

        self.assertEqual(final_data_source["filters"]["start_date"], "2020-01-01")
        self.assertEqual(final_data_source["filters"]["end_date"], "2024-12-31")
        self.assertEqual(
            [item["tool"] for item in tool_log],
            ["slot_query", "ask_client", "slot_query"],
        )
        self.assertEqual(tool_log[1]["request"]["field"], "time_range")
        self.assertEqual(tool_log[1]["request"]["scope_error_type"], "invalid")

    def test_final_data_source_rejects_unknown_table(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown table"):
            _validate_final_data_source(
                {
                    "connection": {
                        "table": "text_to_sql_query_engine_v1.guangzhou_new_house"
                    },
                    "filters": {
                        "city": "Guangzhou",
                        "block": "Lanhuashan",
                        "start_date": "2020-01-01",
                        "end_date": "2024-12-31",
                    },
                }
            )

    def test_final_data_source_rejects_reversed_date_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid date range"):
            _validate_final_data_source(
                {
                    "connection": {"table": "beijing_new_house"},
                    "filters": {
                        "city": "Beijing",
                        "block": "Liangxiang",
                        "start_date": "2026-01-01",
                        "end_date": "2024-12-31",
                    },
                }
            )

    def test_real_estate_beijing_liangxiang_example_validates_without_client(
        self,
    ) -> None:
        state = _real_estate_analysis_state(
            summary_text="From 2020 to 2024, Beijing Liangxiang supply and transaction area both increased.",
            caption_text="Beijing Liangxiang: Historical Supply and Transaction Area Statistics (2020-2024)",
            table="beijing_new_house",
            city="Beijing",
            block="Liangxiang",
            start_date="2020-01-01",
            end_date="2024-12-31",
        )
        query_tool = ScenarioQueryTool()
        agent = ToolCallingDataSourceValidationAgent(query_tool)

        result = asyncio.run(agent.arun(state, FakeClient()))
        final_data_source = result["final_data_source"]
        tool_log = result["tool_log"]

        self.assertEqual(
            final_data_source,
            {
                "connection": {"table": "beijing_new_house"},
                "filters": {
                    "city": "Beijing",
                    "block": "Liangxiang",
                    "start_date": "2020-01-01",
                    "end_date": "2024-12-31",
                },
            },
        )
        self.assertEqual([item["tool"] for item in tool_log], ["slot_query"])
        self.assertEqual(tool_log[0]["args"], _slots(final_data_source))
        self.assertEqual(query_tool.calls, [_slots(final_data_source)])

    def test_real_estate_shenzhen_nanshan_example_keeps_real_date_fields(self) -> None:
        state = _real_estate_analysis_state(
            summary_text="Shenzhen Nanshan CBD resale house area segment statistics from 2021 to 2023.",
            caption_text="2021-2023 Shenzhen Nanshan CBD Resale House Total Area Segment Distribution Statistics",
            table="shenzhen_resale_house",
            city="Shenzhen",
            block="Nanshan CBD",
            start_date="2021-01-01",
            end_date="2023-12-31",
        )
        query_tool = ScenarioQueryTool()
        agent = ToolCallingDataSourceValidationAgent(query_tool)

        result = asyncio.run(agent.arun(state, FakeClient()))
        final_data_source = result["final_data_source"]
        tool_log = result["tool_log"]

        self.assertEqual(
            final_data_source["connection"]["table"], "shenzhen_resale_house"
        )
        self.assertEqual(final_data_source["filters"]["start_date"], "2021-01-01")
        self.assertEqual(final_data_source["filters"]["end_date"], "2023-12-31")
        self.assertNotIn("time_range", final_data_source["filters"])
        self.assertEqual([item["tool"] for item in tool_log], ["slot_query"])
        self.assertEqual(len(query_tool.calls), 1)

    def test_real_estate_candidate_conflict_asks_client_before_query(self) -> None:
        state = _real_estate_analysis_state(
            summary_text="Beijing Liangxiang market from 2020 to 2024 increased.",
            caption_text="2021-2023 Shenzhen Nanshan CBD Resale House Total Area Segment Distribution Statistics",
            table="beijing_new_house",
            city="Beijing",
            block="Liangxiang",
            start_date="2020-01-01",
            end_date="2024-12-31",
        )
        state["tables"][0]["caption"]["data_source"] = {
            "connection": {"table": "shenzhen_resale_house"},
            "select_columns": ["date_code", "area_segment", "total_area"],
            "filters": {
                "city": "Shenzhen",
                "block": "Nanshan CBD",
                "start_date": "2021-01-01",
                "end_date": "2023-12-31",
            },
        }
        query_tool = ScenarioQueryTool()
        agent = ToolCallingDataSourceValidationAgent(query_tool)
        client = ClientAgent(
            feedback_items=[
                {
                    "scope_error_type": "conflict",
                    "field": "city",
                    "response": (
                        "Please use table=shenzhen_resale_house, city=Shenzhen, "
                        "block=Nanshan CBD, start_date=2021-01-01, "
                        "end_date=2023-12-31."
                    ),
                }
            ]
        )

        result = asyncio.run(agent.arun(state, client))
        final_data_source = result["final_data_source"]
        tool_log = result["tool_log"]

        self.assertEqual(
            final_data_source["connection"]["table"], "shenzhen_resale_house"
        )
        self.assertEqual(final_data_source["filters"]["city"], "Shenzhen")
        self.assertEqual(
            [item["tool"] for item in tool_log], ["ask_client", "slot_query"]
        )
        self.assertEqual(tool_log[0]["request"]["scope_error_type"], "conflict")
        self.assertEqual(tool_log[0]["request"]["field"], "city")
        self.assertEqual(query_tool.calls, [_slots(final_data_source)])

    def test_tool_calling_agent_fails_when_client_has_no_matching_feedback(
        self,
    ) -> None:
        state = _analysis_state(city="Beijing", block="Nanshan CBD")
        agent = ToolCallingDataSourceValidationAgent(ScenarioQueryTool())

        with self.assertRaisesRegex(ValueError, "requires client input"):
            asyncio.run(agent.arun(state, ClientAgent(feedback_items=[])))

    def test_slot_query_returns_database_evidence(self) -> None:
        calls: list[tuple[str, dict[str, str] | None]] = []

        def query_func(sql: str, params: dict[str, str] | None = None) -> pd.DataFrame:
            calls.append((sql, params))
            if "information_schema.tables" in sql:
                return pd.DataFrame([{"table_name": "beijing_new_house"}])
            if "AS full_scope" in sql:
                return pd.DataFrame([{"full_scope": False}])
            return pd.DataFrame(
                [
                    {
                        "table_city": True,
                        "table_block": True,
                        "table_city_block": False,
                    }
                ]
            )

        with patch(
            "method.agents.data_source_validation.tools.database_query", query_func
        ):
            evidence = DataSourceQueryTool().run(
                table="beijing_new_house",
                city="Beijing",
                block="Nanshan CBD",
                start_date="2020-01-01",
                end_date="2024-12-31",
            )

        self.assertTrue(evidence["table_exists"])
        self.assertFalse(evidence["full_scope_has_data"])
        self.assertEqual(len(calls), 3)

    def test_slot_query_returns_only_table_evidence_when_table_missing(self) -> None:
        calls: list[tuple[str, dict[str, str] | None]] = []

        def query_func(sql: str, params: dict[str, str] | None = None) -> pd.DataFrame:
            calls.append((sql, params))
            if "information_schema.tables" in sql:
                return pd.DataFrame([{"table_name": "guangzhou_new_house"}])
            raise AssertionError(f"Missing table should not query table scope: {sql}")

        with patch(
            "method.agents.data_source_validation.tools.database_query", query_func
        ):
            evidence = DataSourceQueryTool().run(
                table="beijing_new_house",
                city="Beijing",
                block="Liangxiang",
                start_date="2020-01-01",
                end_date="2024-12-31",
            )

        self.assertEqual(evidence, {"table_exists": False})
        self.assertEqual(len(calls), 1)

    def test_slot_query_skips_full_scope_when_date_range_invalid(self) -> None:
        calls: list[tuple[str, dict[str, str] | None]] = []

        def query_func(sql: str, params: dict[str, str] | None = None) -> pd.DataFrame:
            calls.append((sql, params))
            raise AssertionError(f"Invalid date range should not query database: {sql}")

        with patch(
            "method.agents.data_source_validation.tools.database_query", query_func
        ):
            evidence = DataSourceQueryTool().run(
                table="beijing_new_house",
                city="Beijing",
                block="Liangxiang",
                start_date="bad-date",
                end_date="2024-12-31",
            )

        self.assertEqual(
            evidence,
            {
                "date_range_error": (
                    "start_date/end_date must be valid dates and start_date must be "
                    "less than or equal to end_date."
                )
            },
        )
        self.assertEqual(calls, [])


def _structured_response(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {"structured_response": payload}


def _fake_synthesize_final_data_source(payload: dict[str, Any]) -> dict[str, Any]:
    synthesized = {
        "connection": {"table": ""},
        "filters": {
            "city": "",
            "block": "",
            "start_date": "",
            "end_date": "",
        },
    }
    for candidate in payload["source_candidates"]:
        data_source = candidate["data_source"]
        if not synthesized["connection"]["table"]:
            synthesized["connection"]["table"] = data_source["connection"]["table"]
        for key in ("city", "block", "start_date", "end_date"):
            if not synthesized["filters"][key]:
                synthesized["filters"][key] = data_source["filters"][key]
    return synthesized


def _fake_conflict_fields(payload: dict[str, Any]) -> list[str]:
    conflict_fields: list[str] = []
    slot_paths = {
        "city": ("filters", "city"),
        "block": ("filters", "block"),
    }
    for field, path in slot_paths.items():
        values = []
        for candidate in payload["source_candidates"]:
            value = candidate["data_source"][path[0]][path[1]]
            if value and value not in values:
                values.append(value)
        if len(values) > 1:
            conflict_fields.append(field)
    time_ranges = {
        (
            candidate["data_source"]["filters"]["start_date"],
            candidate["data_source"]["filters"]["end_date"],
        )
        for candidate in payload["source_candidates"]
        if candidate["data_source"]["filters"]["start_date"]
        or candidate["data_source"]["filters"]["end_date"]
    }
    if len(time_ranges) > 1:
        conflict_fields.append("time_range")
    return conflict_fields


def _missing_scope_fields(data_source: dict[str, Any]) -> list[str]:
    fields = []
    filters = data_source["filters"]
    for key in ("city", "block"):
        if not filters[key]:
            fields.append(key)
    if not filters["start_date"] or not filters["end_date"]:
        fields.append("time_range")
    return fields


def _merge_client_reply(
    data_source: dict[str, Any],
    response: dict[str, Any],
) -> bool:
    updated = False
    text = str(response.get("response", ""))
    slots = {
        match.group(1): match.group(2).strip(" .")
        for match in SLOT_REPLY_RE.finditer(text)
    }
    if slots.get("table"):
        data_source["connection"]["table"] = slots["table"]
        updated = True
    for key in ("city", "block", "start_date", "end_date"):
        if slots.get(key):
            data_source["filters"][key] = slots[key]
            updated = True
    return updated


def _slots(data_source: dict[str, Any]) -> dict[str, str]:
    filters = data_source["filters"]
    return {
        "table": data_source["connection"]["table"],
        "city": filters["city"],
        "block": filters["block"],
        "start_date": filters["start_date"],
        "end_date": filters["end_date"],
    }


def _analysis_state(city: str, block: str) -> dict[str, Any]:
    return {
        "summary": {
            "element_id": "summary-1",
            "text": "Example summary",
            "data_source": {
                "connection": {"table": "beijing_new_house"},
                "filters": {
                    "city": city,
                    "block": block,
                    "start_date": "2020-01-01",
                    "end_date": "2024-12-31",
                },
            },
        },
        "tables": [
            {
                "caption": {
                    "element_id": "caption-1",
                    "text": "Example caption",
                    "data_source": {
                        "connection": {"table": "beijing_new_house"},
                        "select_columns": ["date_code", "trade_sets"],
                        "filters": {
                            "city": city,
                            "block": block,
                            "start_date": "2020-01-01",
                            "end_date": "2024-12-31",
                        },
                    },
                },
            }
        ],
    }


def _real_estate_analysis_state(
    *,
    summary_text: str,
    caption_text: str,
    table: str,
    city: str,
    block: str,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    return {
        "summary": {
            "element_id": "summary-1",
            "text": summary_text,
            "data_source": {
                "connection": {"table": table},
                "filters": {
                    "city": city,
                    "block": block,
                    "start_date": start_date,
                    "end_date": end_date,
                },
            },
        },
        "tables": [
            {
                "caption": {
                    "element_id": "caption-1",
                    "text": caption_text,
                    "data_source": {
                        "connection": {"table": table},
                        "select_columns": [
                            "date_code",
                            "supply_sets",
                            "trade_sets",
                            "dim_area",
                        ],
                        "filters": {
                            "city": city,
                            "block": block,
                            "start_date": start_date,
                            "end_date": end_date,
                        },
                    },
                },
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
