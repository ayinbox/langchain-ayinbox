"""A LangChain tool that runs one AY Inbox agent tool."""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool, ToolException
from pydantic import ConfigDict, Field

from ._errors import AyInboxAPIError, AyInboxError
from ._http import AyInboxToolsClient, ToolDefinition

_EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


class AyInboxToolError(ToolException):
    """A tool call the API refused or could not answer.

    The message names the API's `code` and `request_id` so a model can react to it and a person can
    trace it. `status`, `code`, `request_id` and `reason` are `None` for a network failure.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        code: str | None = None,
        request_id: str | None = None,
        reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.request_id = request_id
        self.reason = reason

    @classmethod
    def from_error(cls, error: AyInboxError) -> AyInboxToolError:
        if isinstance(error, AyInboxAPIError):
            return cls(
                str(error), status=error.status, code=error.code, request_id=error.request_id, reason=error.reason
            )
        return cls(str(error))


def _result_content(data: dict[str, Any]) -> str:
    text = data.get("result_text")
    if isinstance(text, str):
        return text
    result = data.get("result")
    return result if isinstance(result, str) else json.dumps(result)


class AyInboxTool(BaseTool):
    """One AY Inbox agent tool, built from its definition in `GET /ai/tools`.

    Calling it sends the model's arguments to `POST /ai/tools/{name}/call`. The content returned to the
    model is the API's `result_text`, which fences text written by people outside the workspace as data;
    the artifact is the whole `data` object (`name`, `result`, `result_text`, `untrusted`).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: AyInboxToolsClient = Field(exclude=True, repr=False)
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    handle_tool_error: bool = True

    @classmethod
    def from_definition(cls, definition: ToolDefinition, client: AyInboxToolsClient) -> AyInboxTool:
        """Build a tool from one item of `GET /ai/tools?format=ayinbox`."""
        name = definition["name"]
        schema = definition.get("input_schema")
        if not isinstance(schema, dict):
            schema = dict(_EMPTY_SCHEMA)
        elif "properties" not in schema:
            schema = {**schema, "properties": {}}
        description = definition.get("description") or f"AY Inbox tool {name}."
        return cls(name=name, description=description, args_schema=schema, client=client)

    def _run(
        self,
        run_manager: CallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        try:
            data = self.client.call_tool(self.name, kwargs)
        except AyInboxError as error:
            raise AyInboxToolError.from_error(error) from error
        return _result_content(data), data

    async def _arun(
        self,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        try:
            data = await self.client.acall_tool(self.name, kwargs)
        except AyInboxError as error:
            raise AyInboxToolError.from_error(error) from error
        return _result_content(data), data
