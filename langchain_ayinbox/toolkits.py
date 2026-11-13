"""The AY Inbox toolkit: every agent tool a credential may call, as LangChain tools."""

from __future__ import annotations

from collections.abc import Sequence

import httpx
from langchain_core.tools import BaseTool, BaseToolkit
from langchain_core.utils import secret_from_env
from pydantic import ConfigDict, Field, PrivateAttr, SecretStr

from ._http import DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT, AyInboxToolsClient, ToolDefinition
from .tools import AyInboxTool


class AyInboxToolkit(BaseToolkit):
    """AY Inbox agent tools for LangChain.

    The catalogue comes from `GET /ai/tools`, fetched once on the first `get_tools()` or `aget_tools()`.
    It holds only the tools the credential's scopes allow, and never a tool that sends mail.

    Example:
        ```python
        from langchain_ayinbox import AyInboxToolkit

        toolkit = AyInboxToolkit(toolsets=["inbox"])  # reads AYINBOX_API_KEY
        tools = toolkit.get_tools()
        ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_key: SecretStr | None = Field(default_factory=secret_from_env("AYINBOX_API_KEY", default=None))
    """API key (`aybox_...`) or OAuth access token (`ayat_...`). Falls back to `AYINBOX_API_KEY`."""
    base_url: str | None = None
    """Defaults to `AYINBOX_BASE_URL`, then `https://api.ayinbox.com/v1`."""
    toolsets: list[str] | None = None
    """Narrow the catalogue server-side, for example `["inbox"]`. Calls are gated the same either way."""
    include: list[str] | None = None
    """Keep only these tool names. A name the credential is not offered raises `ValueError`."""
    exclude: list[str] | None = None
    """Drop these tool names."""
    timeout: float = DEFAULT_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    http_client: httpx.Client | None = Field(default=None, exclude=True, repr=False)
    async_http_client: httpx.AsyncClient | None = Field(default=None, exclude=True, repr=False)

    _client: AyInboxToolsClient | None = PrivateAttr(default=None)
    _tools: list[BaseTool] | None = PrivateAttr(default=None)

    @property
    def client(self) -> AyInboxToolsClient:
        """The HTTP client the tools share."""
        if self._client is None:
            self._client = AyInboxToolsClient(
                self.api_key.get_secret_value() if self.api_key else None,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
                http_client=self.http_client,
                async_http_client=self.async_http_client,
            )
        return self._client

    def get_tools(self) -> list[BaseTool]:
        """Return the tools, fetching the catalogue on the first call."""
        if self._tools is None:
            self._tools = self._build(self.client.list_tools(self.toolsets))
        return list(self._tools)

    async def aget_tools(self) -> list[BaseTool]:
        """Async variant of `get_tools`."""
        if self._tools is None:
            self._tools = self._build(await self.client.alist_tools(self.toolsets))
        return list(self._tools)

    def _build(self, definitions: Sequence[ToolDefinition]) -> list[BaseTool]:
        names = [d["name"] for d in definitions]
        if self.include is not None:
            missing = sorted(set(self.include) - set(names))
            if missing:
                raise ValueError(
                    "These tools are not offered to this credential (check its scopes and toolsets): "
                    + ", ".join(missing)
                )
        keep = set(self.include) if self.include is not None else set(names)
        keep -= set(self.exclude or ())
        client = self.client
        return [AyInboxTool.from_definition(d, client) for d in definitions if d["name"] in keep]
