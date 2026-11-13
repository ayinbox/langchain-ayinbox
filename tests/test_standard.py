"""LangChain's standard unit tests for tools, run against a tool built from the fixture catalogue."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_ayinbox import AyInboxTool, AyInboxToolsClient

from .conftest import API_KEY, fixture


class TestAyInboxToolStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> BaseTool:
        client = AyInboxToolsClient(API_KEY)
        return AyInboxTool.from_definition(fixture("ai_tools_list.json")["data"][0], client)

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"limit": 10}
