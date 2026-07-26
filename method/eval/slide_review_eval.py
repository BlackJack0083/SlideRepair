from __future__ import annotations

import asyncio
import json
import os
import shutil
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from method.agents import (
    ClientAgent,
    ContentValidationAgent,
    DataSourceValidationAgent,
    SlideAnalysisAgent,
    SlideParserAgent,
    SlideReviewInput,
)
from method.eval.detection import aggregate_metrics
from method.eval.evaluator import SlideReviewEvaluator, failure_metrics
from method.pipeline import SlideReviewWorkflow, WorkflowStageError
from method.utils import read_jsonl, scalar_to_json

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


@dataclass(frozen=True)
class SlideReviewEvalConfig:
    benchmark_root: Path
    split: str = "test"
    case_type: Literal["corrupted", "clean"] = "corrupted"
    limit: int = 3
    workers: int = 1
    case_timeout_sec: int = 900
    case_retries: int = 2
    retry_base_sleep_sec: float = 8.0
    client_mode: str = "deterministic"
    output_dir: Path = Path("output/slide_review_eval")


async def run_slide_review_eval(config: SlideReviewEvalConfig) -> dict[str, Any]:
    """Run slide review workflow evaluation on injected benchmark cases.

    The workflow-visible input is limited to `slide.pptx`, `slide.png`, and
    responses exposed through `ClientAgent`. Gold issue records are used only
    by the evaluator after the workflow finishes.

    Args:
        config: Dataset, concurrency, retry, client, and output settings.

    Returns:
        Aggregate evaluation summary. Per-case traces are written to
        `config.output_dir`.
    """
    benchmark_root = config.benchmark_root.resolve()
    manifest_name = (
        "corruptions.jsonl" if config.case_type == "corrupted" else "samples.jsonl"
    )
    records = [
        record
        for record in read_jsonl(benchmark_root / "manifest" / manifest_name)
        if record["split"] == config.split
    ][: config.limit]

    config.output_dir.mkdir(parents=True, exist_ok=True)
    runnable = [
        load_case_assets(
            benchmark_root,
            record,
            case_type=config.case_type,
        )
        for record in records
    ]
    workflow = SlideReviewWorkflow(
        slide_parser_agent=SlideParserAgent(**_agent_config("PARSER")),
        slide_analysis_agent=SlideAnalysisAgent(**_agent_config("ANALYSIS")),
        data_source_validation_agent=DataSourceValidationAgent(
            **_agent_config("DATA_SOURCE")
        ),
        content_validation_agent=ContentValidationAgent(**_agent_config("CONTENT")),
    )

    semaphore = asyncio.Semaphore(config.workers)
    tasks = [
        asyncio.create_task(_run_one_case(assets, config, workflow, semaphore))
        for assets in runnable
    ]
    results = [await task for task in asyncio.as_completed(tasks)]
    completed = sum(result["completed"] for result in results)

    return {
        "benchmark_root": str(benchmark_root),
        "split": config.split,
        "case_type": config.case_type,
        "requested_limit": config.limit,
        "workers": config.workers,
        "written": len(results),
        "succeeded": completed,
        "failed": len(results) - completed,
        "client_mode": config.client_mode,
        "metrics": aggregate_metrics([result["metrics"] for result in results]),
        "output_dir": str(config.output_dir.resolve()),
    }


def load_case_assets(
    benchmark_root: Path,
    record: dict[str, Any],
    *,
    case_type: Literal["corrupted", "clean"],
) -> dict[str, Any]:
    if case_type == "clean":
        ground_truth_yaml_path = benchmark_root / record["gt_yaml"]
        ground_truth_pptx_path = benchmark_root / record["gt_ppt"]
        return {
            "case_id": f"{record['sample_id']}__clean",
            "case_type": "clean",
            "pptx_path": ground_truth_pptx_path,
            "image_path": ground_truth_pptx_path.with_suffix(".png"),
            "injected_yaml_path": ground_truth_yaml_path,
            "ground_truth_yaml_path": ground_truth_yaml_path,
            "ground_truth_pptx_path": ground_truth_pptx_path,
            "corruption_record": {"operations": []},
            "feedback_episode": {"feedback_items": []},
        }

    output_yaml = benchmark_root / record["output_yaml"]
    case_dir = output_yaml.parent
    output_ppt = case_dir / "slide.pptx"
    output_png = case_dir / "slide.png"
    corruption_path = case_dir / "corruption.json"
    feedback_path = case_dir / "feedback_episode.json"
    corruption_record = json.loads(corruption_path.read_text(encoding="utf-8"))
    ground_truth_yaml_path = benchmark_root / corruption_record["expected_repair_yaml"]
    ground_truth_pptx_path = ground_truth_yaml_path.with_name("slide.pptx")
    return {
        "case_id": f"{record['sample_id']}__{record['injection_id']}",
        "case_type": "corrupted",
        "pptx_path": output_ppt,
        "image_path": output_png,
        "injected_yaml_path": output_yaml,
        "ground_truth_yaml_path": ground_truth_yaml_path,
        "ground_truth_pptx_path": ground_truth_pptx_path,
        "corruption_record": corruption_record,
        "feedback_episode": json.loads(feedback_path.read_text(encoding="utf-8")),
    }


def to_jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, date | datetime):
        return value.isoformat()
    return scalar_to_json(value)


async def _run_one_case(
    assets: dict[str, Any],
    config: SlideReviewEvalConfig,
    workflow: SlideReviewWorkflow,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    case_id = assets["case_id"]
    work_dir = config.output_dir / "work" / case_id
    work_dir.mkdir(parents=True, exist_ok=True)
    pptx_path = work_dir / "slide.pptx"
    image_path = work_dir / "slide.png"
    shutil.copyfile(assets["pptx_path"], pptx_path)
    shutil.copyfile(assets["image_path"], image_path)
    slide_input = SlideReviewInput(
        pptx_path=pptx_path,
        image_path=image_path,
    )
    payload: dict[str, Any]
    async with semaphore:
        for attempt in range(config.case_retries + 1):
            try:
                payload = await _attempt_case(
                    case_id=case_id,
                    slide_input=slide_input,
                    assets=assets,
                    config=config,
                    workflow=workflow,
                    attempt=attempt + 1,
                )
                break
            except Exception as exc:  # noqa: BLE001
                payload = _error_payload(
                    case_id=case_id,
                    slide_input=slide_input,
                    attempt=attempt + 1,
                    error=exc,
                    assets=assets,
                )
                if attempt >= config.case_retries or not _is_retryable_error(exc):
                    break
                sleep_sec = config.retry_base_sleep_sec * (2**attempt)
                logger.warning(
                    "Retrying case {} after attempt {} in {} seconds: {}",
                    case_id,
                    attempt + 1,
                    sleep_sec,
                    _error_summary(exc),
                )
                await asyncio.sleep(sleep_sec)

    output_path = config.output_dir / f"{case_id}.json"
    output_path.write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "Finished case {}: completed={}, run_attempt={}",
        case_id,
        payload["completed"],
        payload["run_attempt"],
    )
    return payload


async def _attempt_case(
    *,
    case_id: str,
    slide_input: SlideReviewInput,
    assets: dict[str, Any],
    config: SlideReviewEvalConfig,
    workflow: SlideReviewWorkflow,
    attempt: int,
) -> dict[str, Any]:
    feedback_items = assets["feedback_episode"]["feedback_items"]
    if config.client_mode == "deterministic":
        client_agent = ClientAgent(feedback_items=feedback_items)
    else:
        client_agent = ClientAgent(
            feedback_items=feedback_items,
            mode="llm",
            **_agent_config("CLIENT"),
        )

    result = await asyncio.wait_for(
        workflow.arun(
            slide_input,
            client_agent=client_agent,
        ),
        timeout=config.case_timeout_sec,
    )
    result_dict = result.to_dict()
    metrics = SlideReviewEvaluator().evaluate_case(
        result=result_dict,
        corruption_record=assets["corruption_record"],
        injected_yaml_path=assets["injected_yaml_path"],
        ground_truth_yaml_path=assets["ground_truth_yaml_path"],
        ground_truth_pptx_path=assets["ground_truth_pptx_path"],
    )
    return {
        "case_id": case_id,
        "case_type": assets["case_type"],
        "completed": True,
        "run_attempt": attempt,
        "slide_input": {
            "pptx_path": str(slide_input.pptx_path),
            "image_path": str(slide_input.image_path),
        },
        "stage_summary": {
            "observed_elements": len(result_dict["observed_slide"]["elements"]),
            "tables": len(result_dict["analysis_state"]["tables"]),
            "data_source_tool_calls": len(
                result_dict["data_source_validation_log"]["tool_log"]
            ),
            "content_tool_calls": len(result_dict["content_validation_log"]),
            "detected_issues": len(result_dict["detected_issues"]),
            "table_records": len(result_dict["table_records"]),
            "has_repaired_yaml": bool(result_dict["repaired_artifacts"]["yaml_path"]),
        },
        "result": result_dict,
        "metrics": metrics,
    }


def _error_payload(
    *,
    case_id: str,
    slide_input: SlideReviewInput,
    attempt: int,
    error: Exception,
    assets: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "case_id": case_id,
        "case_type": assets["case_type"],
        "completed": False,
        "run_attempt": attempt,
        "error": _error_summary(error),
        "metrics": failure_metrics(assets["corruption_record"]),
        "slide_input": {
            "pptx_path": str(slide_input.pptx_path),
            "image_path": str(slide_input.image_path),
        },
    }
    if isinstance(error, WorkflowStageError):
        partial_result = error.partial_result
        payload["failed_stage"] = error.stage
        payload["partial_result"] = partial_result
        payload["metrics"] = SlideReviewEvaluator().evaluate_partial_case(
            partial_result=partial_result,
            corruption_record=assets["corruption_record"],
            injected_yaml_path=assets["injected_yaml_path"],
            ground_truth_yaml_path=assets["ground_truth_yaml_path"],
        )
    return payload


def _error_summary(error: Exception) -> str:
    lines = str(error).splitlines()
    return lines[0] if lines else type(error).__name__


def _is_retryable_error(error: Exception) -> bool:
    if isinstance(error, WorkflowStageError):
        error = error.original_error
    return isinstance(
        error,
        (
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
            TimeoutError,
        ),
    )


def _agent_config(prefix: str) -> dict[str, str | None]:
    model = os.getenv(f"{prefix}_DASHSCOPE_MODEL") or os.getenv("DASHSCOPE_MODEL")
    if not model:
        raise RuntimeError(f"{prefix}_DASHSCOPE_MODEL or DASHSCOPE_MODEL is required.")
    return {
        "model": model,
        "api_key": os.getenv(f"{prefix}_DASHSCOPE_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY"),
        "base_url": os.getenv(f"{prefix}_DASHSCOPE_BASE_URL")
        or os.getenv("DASHSCOPE_BASE_URL")
        or DEFAULT_BASE_URL,
    }
