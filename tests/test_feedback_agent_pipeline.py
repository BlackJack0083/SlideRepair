from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import unittest
from dataclasses import dataclass
from pathlib import Path

from langchain.tools import ToolRuntime

from method.agents import (
    ClientAgent,
    ContentValidationAgent,
    SlideAnalysisAgent,
    SlideParserAgent,
    SlideReviewInput,
)
from method.pipeline import SlideReviewWorkflow

PRESENTATION_LABEL_RE = re.compile(
    r"\((bar chart|line chart|pie chart|table)\)\s*$", re.I
)
TREND_RE = re.compile(
    r"\b(increase|increased|decrease|decreased|growth|decline|upward|downward)\b", re.I
)


class TestRoleClient:
    async def achat(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Path,
        response_format: str,
    ) -> str:
        del system_prompt, image_path, response_format
        payload = json.loads(user_prompt.split("Input elements:\n", 1)[1])
        elements = list(payload.get("elements", []))
        roles: dict[str, str] = {}
        text_elements = [
            element for element in elements if element.get("type") == "textBox"
        ]
        non_text_elements = [
            element for element in elements if element.get("type") != "textBox"
        ]

        for element in non_text_elements:
            shape = str(element.get("shape") or element.get("type") or "")
            if shape in {"chart-bar", "chart-line", "chart-pie", "table"}:
                roles[str(element["id"])] = shape
            elif element.get("type") == "table":
                roles[str(element["id"])] = "table"
            else:
                roles[str(element["id"])] = "chart-bar"

        caption_ids = []
        summary_ids = []
        for element in text_elements:
            text = str(element.get("text", "")).strip()
            element_id = str(element["id"])
            if PRESENTATION_LABEL_RE.search(text) or "analysis" in text.lower():
                caption_ids.append(element_id)
            elif TREND_RE.search(text) or any(char.isdigit() for char in text):
                summary_ids.append(element_id)

        title_candidates = [
            element
            for element in text_elements
            if str(element["id"]) not in caption_ids + summary_ids
        ]
        title_candidates.sort(
            key=lambda item: (
                item.get("layout", {}).get("y", 999.0),
                len(str(item.get("text", ""))),
            )
        )
        if not title_candidates and text_elements:
            title_candidates = sorted(
                text_elements,
                key=lambda item: (
                    item.get("layout", {}).get("y", 999.0),
                    len(str(item.get("text", ""))),
                ),
            )
        title_id = str(title_candidates[0]["id"]) if title_candidates else None

        for element in text_elements:
            element_id = str(element["id"])
            if element_id in roles:
                continue
            if element_id == title_id:
                roles[element_id] = "title"
            elif element_id in caption_ids:
                roles[element_id] = "caption"
            else:
                roles[element_id] = "summary"

        return json.dumps(
            {
                "roles": [
                    {"id": element_id, "role": role}
                    for element_id, role in sorted(
                        roles.items(), key=lambda item: int(item[0])
                    )
                ]
            }
        )


class TestAnalysisClient:
    async def achat(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Path | None = None,
        response_format: str | None = None,
    ) -> str:
        del image_path, response_format
        payload = json.loads(user_prompt)
        del system_prompt
        if "text" in payload and "table_data" not in payload:
            select_columns = []
            if "row_headers" in payload or "column_headers" in payload:
                select_columns = ["date_code", "supply_sets", "trade_sets", "dim_area"]
            data_source = {
                "connection": {"table": "beijing_new_house"},
                "filters": {
                    "city": "Beijing",
                    "block": "Liangxiang",
                    "start_date": "2020-01-01",
                    "end_date": "2024-12-31",
                },
            }
            if select_columns:
                data_source["select_columns"] = select_columns
            return json.dumps(data_source)

        metric_names = [
            name
            for name in payload["table_data"][0]
            if name not in {"category", "year", "month"}
        ] or ["value"]
        return json.dumps(
            {
                "table_type": "field-constraint",
                "dimensions": [
                    {
                        "source_col": "date_code",
                        "target_col": "year",
                        "method": "period",
                        "time_granularity": "year",
                    }
                ],
                "metrics": [
                    {
                        "name": name,
                        "source_col": "trade_sets",
                        "agg_func": "count",
                        "filter_condition": {"trade_sets": 1},
                    }
                    for name in metric_names
                ],
                "crosstab_row": None,
                "crosstab_col": None,
            }
        )


class TestSummaryValidationClient:
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Path | None = None,
        response_format: str | None = None,
    ) -> str:
        del system_prompt, user_prompt, image_path, response_format
        return json.dumps(
            {
                "status": "pass",
                "evidence": "No contradictory summary claim.",
                "suggested_summary": "",
                "confidence": 0.9,
            }
        )


class PassThroughDataSourceValidationAgent:
    async def arun(self, analysis_state, client):
        del client
        first_caption_source = analysis_state["tables"][0]["caption"]["data_source"]
        filters = first_caption_source["filters"]
        return {
            "final_data_source": {
                "connection": first_caption_source["connection"],
                "filters": {
                    "city": filters["city"],
                    "block": filters["block"],
                    "start_date": filters["start_date"],
                    "end_date": filters["end_date"],
                },
            },
            "tool_log": [],
            "detected_issues": [],
        }


class ContentToolCallingFakeAgent:
    def __init__(self, tools, state, context):
        self.tools = {tool.name: tool for tool in tools}
        self.state = state
        self.context = context

    def invoke_tool(self, name, args):
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
        self.state.update(update)
        return json.loads(messages[0].content) if messages else {}

    async def ainvoke(self, payload):
        user_payload = json.loads(payload["messages"][0].content)
        for table in user_payload["tables"]:
            table_index = table["table_index"]
            raw = self.invoke_tool("sql_retrieve", {"table_index": table_index})
            computed = self.invoke_tool(
                "analysis_execute",
                {
                    "table_index": table_index,
                    "raw_data_path": raw["raw_data_path"],
                },
            )
            comparison = self.invoke_tool(
                "compare_table",
                {
                    "table_index": table_index,
                    "computed_data_path": computed["computed_data_path"],
                },
            )
            if comparison["comparison"]["status"] == "different":
                target = "table" if table["body"]["type"] == "table" else "chart"
                response = self.invoke_tool(
                    "ask_client",
                    {
                        "error_type": "value_error",
                        "target": target,
                        "description": comparison["comparison"]["diff_summary"],
                    },
                )
                if _client_agrees(response["response"]):
                    tool_name = (
                        "modify_table"
                        if table["body"]["type"] == "table"
                        else "modify_chart"
                    )
                    self.invoke_tool(
                        tool_name,
                        {
                            "table_index": table_index,
                            "data_path": computed["computed_data_path"],
                        },
                    )

        for table in user_payload["tables"]:
            table_index = table["table_index"]
            caption_text = table["caption"]["text"]
            caption_label = _presentation_label(caption_text)
            actual_type = _body_presentation_type(table["body"]["type"])
            if caption_label and caption_label != actual_type:
                evidence = (
                    f"caption says '{caption_label}' but body type is '{actual_type}'."
                )
                response = self.invoke_tool(
                    "ask_client",
                    {
                        "error_type": "claim_error",
                        "target": "caption",
                        "description": evidence,
                    },
                )
                if _client_agrees(response["response"]):
                    self.invoke_tool(
                        "modify_textbox",
                        {
                            "element_id": table["caption"]["element_id"],
                            "text": PRESENTATION_LABEL_RE.sub(
                                f"({actual_type.title()})",
                                caption_text.strip(),
                            ),
                        },
                    )
        return {"messages": []}


@dataclass
class FakeMessage:
    content: str


class FakeContentValidationAgent:
    def __init__(self, issues=None):
        self.issues = list(issues or [])

    async def arun(
        self,
        *,
        analysis_state,
        client,
        pptx_path,
        artifact_dir,
        scope_dialogue=None,
    ):
        del client, pptx_path, scope_dialogue
        artifact_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = artifact_dir / "repaired_slide.yaml"
        yaml_path.write_text("title: Example\n", encoding="utf-8")
        return {
            "analysis_state": analysis_state,
            "table_records": [],
            "tool_log": [],
            "detected_issues": self.issues,
            "repaired_artifacts": {
                "yaml_path": str(yaml_path),
                "data_paths": {},
                "pptx_path": None,
            },
        }


class ToolCallingContentValidationAgent(ContentValidationAgent):
    def __init__(self) -> None:
        pass

    async def arun(
        self,
        *,
        analysis_state,
        client,
        pptx_path,
        artifact_dir,
        scope_dialogue=None,
    ):
        from method.agents.content_validation.agent import build_content_payload
        from method.agents.content_validation.artifacts import write_content_artifacts
        from method.agents.content_validation.tools import (
            CONTENT_VALIDATION_TOOLS,
            ContentValidationContext,
        )

        state = {
            "messages": [],
            "analysis_state": copy.deepcopy(analysis_state),
            "table_records": [],
            "tool_log": [],
            "detected_issues": [],
        }
        context = ContentValidationContext(
            client=client,
            artifact_dir=artifact_dir,
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        await ContentToolCallingFakeAgent(
            CONTENT_VALIDATION_TOOLS, state, context
        ).ainvoke(
            {
                "messages": [
                    FakeMessage(
                        json.dumps(
                            build_content_payload(state["analysis_state"]),
                            ensure_ascii=False,
                        )
                    )
                ]
            }
        )
        repaired_artifacts = write_content_artifacts(
            source_pptx=pptx_path,
            analysis_state=state["analysis_state"],
            artifact_dir=artifact_dir,
        )
        return {
            "analysis_state": state["analysis_state"],
            "table_records": state["table_records"],
            "tool_log": state["tool_log"],
            "detected_issues": state["detected_issues"],
            "repaired_artifacts": repaired_artifacts,
        }


class FeedbackPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_dir = Path(
            "examples/benchmark/split/train/s_00163f7ed3ede3ed/injected"
        )
        self.case_dir = None
        if self.sample_dir.exists():
            self.case_dir = next(
                (path for path in sorted(self.sample_dir.iterdir()) if path.is_dir()),
                None,
            )

    def require_injected_case(self) -> Path:
        if self.case_dir is None:
            self.skipTest(f"Injected benchmark fixture not found: {self.sample_dir}")
        return self.case_dir

    def test_slide_parser_agent_builds_ppt_representation_and_csv(self) -> None:
        case_dir = self.require_injected_case()
        agent = SlideParserAgent(client=TestRoleClient())
        result = asyncio.run(
            agent.arun(
                SlideReviewInput(
                    pptx_path=case_dir / "slide.pptx",
                    image_path=case_dir / "slide.png",
                )
            )
        )
        self.assertIn("observed_slide", result)
        self.assertIn("ppt_representation", result)
        self.assertNotIn("role_assignments", result)
        representation = result["ppt_representation"]
        self.assertNotIn("case_id", representation)
        self.assertIn("title", representation)
        self.assertIn("summary", representation)
        self.assertTrue(representation["structured_tables"])
        first_table = representation["structured_tables"][0]
        self.assertIn("header", first_table)
        self.assertIn("body", first_table)
        self.assertIn("data_path", first_table["body"])
        self.assertEqual(first_table["body"]["data_path"], "data.csv")
        self.assertTrue((case_dir / first_table["body"]["data_path"]).exists())
        self.assertIn("element_id", representation["summary"])
        self.assertNotIn("element_ids", representation["summary"])

    def test_pipeline_emits_detected_issues(self) -> None:
        case_dir = self.require_injected_case()
        workflow = SlideReviewWorkflow(
            slide_parser_agent=SlideParserAgent(client=TestRoleClient()),
            slide_analysis_agent=SlideAnalysisAgent(client=TestAnalysisClient()),
            data_source_validation_agent=PassThroughDataSourceValidationAgent(),
            content_validation_agent=FakeContentValidationAgent(
                issues=[
                    {
                        "target": "table",
                        "error_type": "value_error",
                        "evidence": "synthetic mismatch",
                    }
                ]
            ),
        )
        result = asyncio.run(
            workflow.arun(
                SlideReviewInput(
                    pptx_path=case_dir / "slide.pptx",
                    image_path=case_dir / "slide.png",
                ),
                client_agent=ClientAgent(feedback_items=[]),
            ),
        )
        self.assertTrue(result.detected_issues)
        self.assertIn("tables", result.analysis_state)
        self.assertIn("final_data_source", result.analysis_state)
        self.assertIn("body", result.analysis_state["tables"][0])
        first_issue = result.detected_issues[0]
        self.assertIn("target", first_issue)
        self.assertIn("error_type", first_issue)

    @unittest.skipUnless(
        all(
            os.getenv(name)
            for name in ("SQL_USER", "SQL_PASSWORD", "SQL_HOST", "SQL_DB")
        ),
        "requires the SlideRepair PostgreSQL database",
    )
    def test_content_validation_updates_real_caption_presentation_mismatch(
        self,
    ) -> None:
        case_dir = Path(
            "examples/benchmark/split/train/s_00163f7ed3ede3ed/injected/"
            "00163f7ed3ede3ed-claim1-517240f5"
        )
        if not (case_dir / "slide.pptx").exists():
            self.skipTest(f"Injected benchmark fixture not found: {case_dir}")
        workflow = SlideReviewWorkflow(
            slide_parser_agent=SlideParserAgent(client=TestRoleClient()),
            slide_analysis_agent=SlideAnalysisAgent(client=TestAnalysisClient()),
            data_source_validation_agent=PassThroughDataSourceValidationAgent(),
            content_validation_agent=ToolCallingContentValidationAgent(),
        )
        result = asyncio.run(
            workflow.arun(
                SlideReviewInput(
                    pptx_path=case_dir / "slide.pptx",
                    image_path=case_dir / "slide.png",
                ),
                client_agent=ClientAgent(
                    feedback_items=[
                        {
                            "error_type": "claim_error",
                            "target": "caption",
                            "response": "Yes, please apply the proposed update.",
                        }
                    ]
                ),
            ),
        )
        caption_requests = [
            item
            for item in result.content_validation_log
            if item["tool"] == "ask_client"
            and item["request"]["target"] == "caption"
        ]
        caption_updates = [
            item
            for item in result.content_validation_log
            if item["tool"] == "modify_textbox" and item["target"] == "caption"
        ]
        caption_issues = [
            issue
            for issue in result.detected_issues
            if issue["target"] == "caption" and issue["error_type"] == "claim_error"
        ]
        self.assertTrue(caption_requests)
        self.assertTrue(caption_updates)
        self.assertTrue(caption_issues)

    def test_client_matches_exact_caption_update_request(self) -> None:
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
                "target": "caption",
                "description": "Update caption text to match the chart type.",
            }
        )

        self.assertEqual(
            response,
            {
                "response": "Yes, please apply the proposed update.",
                "confirmed": True,
            },
        )


def _client_agrees(response: str) -> bool:
    return any(
        token in response.lower() for token in ("yes", "accept", "agree", "update")
    )


def _presentation_label(text: str) -> str | None:
    match = PRESENTATION_LABEL_RE.search(text.strip())
    if match is None:
        return None
    return match.group(1).lower()


def _body_presentation_type(body_type: str) -> str:
    if body_type == "table":
        return "table"
    if body_type.startswith("chart-"):
        return f"{body_type.removeprefix('chart-')} chart"
    return body_type
