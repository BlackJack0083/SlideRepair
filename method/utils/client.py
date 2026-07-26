import base64
import mimetypes
import os
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


def _image_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if not mime_type:
        mime_type = "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


class Client:
    """LangChain ChatOpenAI client with the project-local `.chat(...)` interface."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout_sec: int = 120,
        enable_thinking: bool | None = False,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.enable_thinking = enable_thinking
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self._model = ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout_sec,
            temperature=0,
            extra_body={"enable_thinking": self.enable_thinking}
            if self.enable_thinking is not None
            else None,
        )

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Path | None = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        response_format: Literal["json_object"] | None = None,
    ) -> str:
        if image_path is None:
            user_content: list[dict[str, Any]] | str = user_prompt
        else:
            user_content = [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(image_path)},
                },
            ]

        bind_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format == "json_object":
            bind_kwargs["response_format"] = {"type": "json_object"}

        response = self._model.bind(**bind_kwargs).invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content),
            ]
        )
        text = str(response.text).strip()
        if not text:
            raise RuntimeError("Model returned empty content.")
        return text

    async def achat(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Path | None = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        response_format: Literal["json_object"] | None = None,
    ) -> str:
        if image_path is None:
            user_content: list[dict[str, Any]] | str = user_prompt
        else:
            user_content = [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(image_path)},
                },
            ]

        bind_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format == "json_object":
            bind_kwargs["response_format"] = {"type": "json_object"}

        response = await self._model.bind(**bind_kwargs).ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content),
            ]
        )
        text = str(response.text).strip()
        if not text:
            raise RuntimeError("Model returned empty content.")
        return text
