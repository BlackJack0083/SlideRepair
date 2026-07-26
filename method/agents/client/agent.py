from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from method.utils import Client

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"
DEFAULT_PROMPT_PATH = PROMPT_DIR / "client_simulation_prompt.txt"


class ClientAgent:
    """由 case-local feedback items 驱动的共享 benchmark client。

    Args:
        feedback_items: 当前 case 的私有 feedback items。
        mode: `deterministic` 直接返回 feedback response；`llm` 先确定性匹配
            feedback item，再用 LLM 改写成更自然的客户回复。
        client: 可选的 project-local `.chat(...)` client，主要用于测试。
        model: LLM client 使用的模型名。`mode="llm"` 且未传 `client` 时必填。
        api_key: LLM API key。
        base_url: OpenAI-compatible API base URL。
        timeout_sec: LLM 请求超时时间。
    """

    def __init__(
        self,
        *,
        feedback_items: list[dict[str, Any]],
        mode: Literal["deterministic", "llm"] = "deterministic",
        client: Any | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout_sec: int = 120,
    ):
        if mode not in {"deterministic", "llm"}:
            raise ValueError(f"Unsupported client mode: {mode}")
        self.feedback_items = list(feedback_items)
        self.mode = mode
        self.prompt = DEFAULT_PROMPT_PATH.read_text(encoding="utf-8")
        self.client = client
        if self.mode == "llm" and self.client is None:
            if not model:
                raise ValueError("ClientAgent mode='llm' requires model when client is not provided.")
            self.client = Client(
                model=model,
                api_key=api_key,
                base_url=base_url,
                timeout_sec=timeout_sec,
            )

    def respond(self, request: dict[str, Any]) -> dict[str, Any]:
        """响应 agent 发出的澄清或确认请求。

        Args:
        request: 请求对象。Scope 澄清按 `field` 和 `scope_error_type` 匹配；
            caption 澄清按 `error_type` 和 `target` 匹配。

        Returns:
            client 回复。agent 可见的澄清和确认请求只返回客户式 `response`。
        """
        item = self._find_feedback_item(request)
        if item is None:
            return {
                "response": "I do not have a confirmed correction for this request.",
                "confirmed": False,
            }

        if self.mode == "deterministic":
            return {"response": item["response"], "confirmed": True}
        return self.llm_response(request=request, item=item)

    def _find_feedback_item(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any] | None:
        if "field" in request:
            match_fields = ["field", "scope_error_type"]
            is_scope = True
        elif "target" in request:
            match_fields = ["error_type", "target"]
            is_scope = False
        else:
            raise ValueError(f"Unsupported client request: {request}")

        for item in self.feedback_items:
            if ("field" in item) != is_scope:
                continue
            if not all(item[field] == request[field] for field in match_fields):
                continue
            return item
        return None

    def llm_response(
        self,
        *,
        request: dict[str, Any],
        item: dict[str, Any],
    ) -> dict[str, str]:
        payload = {
            "request": request,
            "matched_feedback": {
                "response": item["response"],
            },
        }
        content = self.client.chat(
            self.prompt,
            json.dumps(payload, ensure_ascii=False, indent=2),
            temperature=0.3,
            max_tokens=512,
        )
        return {"response": content, "confirmed": True}
