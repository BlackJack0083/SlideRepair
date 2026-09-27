from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from method.agents import (
    ContentValidationAgent,
    DataSourceValidationAgent,
    SlideAnalysisAgent,
    SlideParserAgent,
    SlideReviewResult,
)
from method.agents.types import SlideReviewInput


class WorkflowStageError(RuntimeError):
    """Workflow stage failure with partial trace for evaluation/debugging."""

    def __init__(
        self,
        *,
        stage: str,
        partial_result: dict[str, Any],
        original_error: Exception,
    ) -> None:
        super().__init__(str(original_error))
        self.stage = stage
        self.partial_result = partial_result
        self.original_error = original_error


@dataclass
class WorkflowOverrides:
    """Optional intermediate outputs supplied by an evaluation intervention.

    Normal workflow execution does not construct or use these values. The
    evaluation layer may provide them to measure the downstream effect of a
    particular upstream output.

    Args:
        parser_result: Replacement parser output containing `observed_slide`
            and `ppt_representation`.
        data_source_state: Replacement source state passed to slide analysis.
        computation_logic: Replacement calculation logic passed to slide
            analysis.
        slide_analysis_state: Replacement pre-validation analysis state used for
            trace-level stage evaluation.
        verified_final_data_source: Optional final data source that bypasses
            data-source validation while preserving the analysis state produced
            by slide analysis.
        verified_analysis_state: Replacement post-validation analysis state.
        data_source_issues: Optional scope issues to reuse in the detection
            trace when `verified_analysis_state` is supplied.
        scope_dialogue: Confirmed scope dialogue paired with
            `verified_analysis_state`.
    """

    parser_result: dict[str, Any] | None = None
    data_source_state: dict[str, Any] | None = None
    computation_logic: list[dict[str, Any]] | None = None
    slide_analysis_state: dict[str, Any] | None = None
    verified_final_data_source: dict[str, Any] | None = None
    verified_analysis_state: dict[str, Any] | None = None
    data_source_issues: list[dict[str, Any]] | None = None
    scope_dialogue: list[dict[str, str]] | None = None


class SlideReviewWorkflow:
    """Run slide parsing, state extraction, source repair, and content validation."""

    def __init__(
        self,
        *,
        slide_parser_agent: SlideParserAgent,
        slide_analysis_agent: SlideAnalysisAgent,
        data_source_validation_agent: DataSourceValidationAgent,
        content_validation_agent: ContentValidationAgent,
    ):
        self.slide_parser_agent = slide_parser_agent
        self.slide_analysis_agent = slide_analysis_agent
        self.data_source_validation_agent = data_source_validation_agent
        self.content_validation_agent = content_validation_agent

    async def arun(
        self,
        slide_input: SlideReviewInput,
        *,
        client_agent,
        overrides: WorkflowOverrides | None = None,
    ) -> SlideReviewResult:
        partial: dict[str, Any] = {}
        if overrides is not None and overrides.parser_result is not None:
            parsed = copy.deepcopy(overrides.parser_result)
        else:
            try:
                parsed = await self.slide_parser_agent.arun(slide_input)
            except Exception as exc:
                raise WorkflowStageError(
                    stage="parser",
                    partial_result=partial,
                    original_error=exc,
                ) from exc
        observed_slide = parsed["observed_slide"]
        ppt_representation = parsed["ppt_representation"]
        partial.update(
            {
                "observed_slide": observed_slide,
                "ppt_representation": ppt_representation,
            }
        )
        if overrides is not None and overrides.verified_analysis_state is not None:
            analysis_state = copy.deepcopy(overrides.verified_analysis_state)
            slide_analysis_state = copy.deepcopy(
                overrides.slide_analysis_state or analysis_state
            )
            data_source_tool_log: list[dict[str, Any]] = []
            data_source_issues = list(overrides.data_source_issues or [])
            data_source_validation_log = {
                "final_data_source": analysis_state["final_data_source"],
                "tool_log": data_source_tool_log,
            }
            scope_dialogue = list(overrides.scope_dialogue or [])
        else:
            try:
                analysis_state = await self.slide_analysis_agent.arun(
                    ppt_representation=ppt_representation,
                    data_source_state=(
                        overrides.data_source_state if overrides is not None else None
                    ),
                    computation_logic=(
                        overrides.computation_logic if overrides is not None else None
                    ),
                )
            except Exception as exc:
                raise WorkflowStageError(
                    stage="slide_analysis",
                    partial_result=partial,
                    original_error=exc,
                ) from exc
            slide_analysis_state = analysis_state
            if overrides is not None and overrides.verified_final_data_source is not None:
                final_data_source = copy.deepcopy(
                    overrides.verified_final_data_source
                )
                data_source_tool_log = []
                data_source_issues = list(overrides.data_source_issues or [])
                analysis_state = update_data_source(analysis_state, final_data_source)
                data_source_validation_log = {
                    "final_data_source": final_data_source,
                    "tool_log": data_source_tool_log,
                }
                scope_dialogue = list(overrides.scope_dialogue or [])
            else:
                try:
                    data_source_result = await self.data_source_validation_agent.arun(
                        analysis_state=analysis_state,
                        client=client_agent,
                    )
                except Exception as exc:
                    raise WorkflowStageError(
                        stage="data_source_validation",
                        partial_result=partial,
                        original_error=exc,
                    ) from exc
                final_data_source = data_source_result["final_data_source"]
                data_source_tool_log = data_source_result["tool_log"]
                data_source_issues = data_source_result["detected_issues"]
                analysis_state = update_data_source(analysis_state, final_data_source)
                data_source_validation_log = {
                    "final_data_source": final_data_source,
                    "tool_log": data_source_tool_log,
                }
                scope_dialogue = [
                    {
                        "assistant": issue["evidence"],
                        "human": issue["client_response"],
                    }
                    for issue in data_source_issues
                    if issue["confirmed"]
                ]
        partial["slide_analysis_state"] = slide_analysis_state
        partial.update(
            {
                "analysis_state": analysis_state,
                "data_source_validation_log": data_source_validation_log,
                "detected_issues": list(data_source_issues),
            }
        )
        artifact_dir = slide_input.pptx_path.parent / "review_artifacts"
        try:
            content_result = await self.content_validation_agent.arun(
                analysis_state=analysis_state,
                client=client_agent,
                pptx_path=slide_input.pptx_path,
                artifact_dir=artifact_dir,
                scope_dialogue=scope_dialogue,
            )
        except Exception as exc:
            raise WorkflowStageError(
                stage="content_validation",
                partial_result=partial,
                original_error=exc,
            ) from exc
        analysis_state = content_result["analysis_state"]

        return SlideReviewResult(
            observed_slide=observed_slide,
            ppt_representation=ppt_representation,
            slide_analysis_state=slide_analysis_state,
            analysis_state=analysis_state,
            data_source_validation_log=data_source_validation_log,
            content_validation_log=content_result["tool_log"],
            detected_issues=[
                *data_source_issues,
                *content_result["detected_issues"],
            ],
            table_records=content_result["table_records"],
            repaired_artifacts=content_result["repaired_artifacts"],
        )


def update_data_source(
    analysis_state: dict[str, Any],
    data_source: dict[str, Any],
) -> dict[str, Any]:
    """把确认后的 slide-level datasource 写回 analysis state。

    Args:
        analysis_state: Slide analysis 输出的 state。
        data_source: Data source validation 确认后的 slide-level source。

    Returns:
        datasource 已归一化的 state。
    """
    state = copy.deepcopy(analysis_state)
    state["final_data_source"] = data_source
    sources = [
        state["summary"]["data_source"],
        *(table["caption"]["data_source"] for table in state["tables"]),
    ]
    for source in sources:
        source.update(copy.deepcopy(data_source))
    return state
