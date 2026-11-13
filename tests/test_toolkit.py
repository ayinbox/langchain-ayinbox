from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from langchain_core.messages import ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import SecretStr

from langchain_ayinbox import (
    AyInboxAPIError,
    AyInboxError,
    AyInboxTool,
    AyInboxToolError,
    AyInboxToolkit,
    AyInboxToolsClient,
    __version__,
)

from .conftest import API_KEY, MockAPI, fixture, json_response

TOOL_CALL = {"name": "list_campaigns", "args": {"limit": 5}, "id": "call_1", "type": "tool_call"}

CATALOGUE = {
    "data": [
        {"name": "list_threads", "description": "List threads.", "input_schema": {"type": "object"}},
        {
            "name": "list_campaigns",
            "description": "List the workspace's campaigns.",
            "input_schema": {"type": "object", "properties": {"limit": {"type": "integer"}}},
        },
        {
            "name": "add_tag",
            "description": "Tag a contact.",
            "input_schema": {
                "type": "object",
                "properties": {"contact_id": {"type": "string"}, "idempotency_key": {"type": "string"}},
                "required": ["contact_id"],
            },
        },
    ]
}


def make_toolkit(api: MockAPI, **kwargs: Any) -> AyInboxToolkit:
    toolkit = AyInboxToolkit(
        api_key=SecretStr(API_KEY), http_client=api.sync_client(), async_http_client=api.async_client(), **kwargs
    )
    toolkit.client._sleep = lambda _seconds: None

    async def no_sleep(_seconds: float) -> None:
        return None

    toolkit.client._asleep = no_sleep
    return toolkit


def test_tool_list_is_converted(api: MockAPI) -> None:
    tools = make_toolkit(api).get_tools()
    assert [t.name for t in tools] == ["list_campaigns"]
    tool = tools[0]
    assert isinstance(tool, AyInboxTool)
    assert tool.description == "List the workspace's campaigns."
    assert tool.args == {"limit": {"type": "integer"}}
    assert convert_to_openai_tool(tool) == {
        "type": "function",
        "function": {
            "name": "list_campaigns",
            "description": "List the workspace's campaigns.",
            "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}}},
        },
    }


def test_list_request_headers_and_query(api: MockAPI) -> None:
    make_toolkit(api, toolsets=["inbox", "campaigns"]).get_tools()
    request = api.requests[0]
    assert request.method == "GET"
    assert str(request.url) == "https://api.ayinbox.com/v1/ai/tools?format=ayinbox&toolsets=inbox%2Ccampaigns"
    assert request.headers["authorization"] == f"Bearer {API_KEY}"
    assert request.headers["user-agent"] == f"langchain-ayinbox/{__version__}"
    assert request.headers["accept"] == "application/json"


def test_catalogue_is_fetched_once(api: MockAPI) -> None:
    toolkit = make_toolkit(api)
    toolkit.get_tools()
    toolkit.get_tools()
    assert len(api.requests) == 1


def test_sync_call_round_trip(api: MockAPI) -> None:
    tool = make_toolkit(api).get_tools()[0]
    message = tool.invoke(TOOL_CALL)
    assert isinstance(message, ToolMessage)
    assert message.content == '{"campaigns":[]}'
    assert message.artifact == fixture("ai_tool_call.json")["data"]
    request = api.requests[-1]
    assert request.method == "POST"
    assert request.url.path == "/v1/ai/tools/list_campaigns/call"
    assert json.loads(request.content) == {"limit": 5}
    assert request.headers["content-type"] == "application/json"
    assert request.headers["authorization"] == f"Bearer {API_KEY}"


def test_plain_invoke_returns_content(api: MockAPI) -> None:
    tool = make_toolkit(api).get_tools()[0]
    assert tool.invoke({}) == '{"campaigns":[]}'
    assert json.loads(api.requests[-1].content) == {}


async def test_async_call_round_trip(api: MockAPI) -> None:
    toolkit = make_toolkit(api)
    tools = await toolkit.aget_tools()
    message = await tools[0].ainvoke(TOOL_CALL)
    assert isinstance(message, ToolMessage)
    assert message.content == '{"campaigns":[]}'
    assert message.artifact["result"] == {"campaigns": []}
    assert json.loads(api.requests[-1].content) == {"limit": 5}


@pytest.mark.parametrize(
    ("fixture_name", "status", "code"),
    [
        ("errors/403_forbidden.json", 403, "forbidden"),
        ("errors/404_not_found.json", 404, "not_found"),
        ("errors/422_unprocessable.json", 422, "unprocessable"),
        ("errors/402_insufficient_credits.json", 402, "insufficient_credits"),
        ("errors/409_idempotency_outcome_unknown.json", 409, "idempotency_outcome_unknown"),
        ("errors/500_internal_error.json", 500, "internal_error"),
    ],
)
def test_api_error_becomes_tool_error(api: MockAPI, fixture_name: str, status: int, code: str) -> None:
    api.responses[("POST", "/v1/ai/tools/list_campaigns/call")] = [json_response(status, fixture(fixture_name))]
    tool = make_toolkit(api).get_tools()[0]
    message = tool.invoke(TOOL_CALL)
    assert isinstance(message, ToolMessage)
    assert message.status == "error"
    assert f"{status} {code}" in str(message.content)
    assert "req-fixture-0001" in str(message.content)


def test_tool_error_carries_fields_when_not_handled(api: MockAPI) -> None:
    api.responses[("POST", "/v1/ai/tools/list_campaigns/call")] = [
        json_response(403, fixture("errors/403_api_key_mailbox_limited.json"))
    ]
    tool = make_toolkit(api).get_tools()[0]
    tool.handle_tool_error = False
    with pytest.raises(AyInboxToolError) as info:
        tool.invoke({"limit": 1})
    assert info.value.status == 403
    assert info.value.code == "api_key_mailbox_limited"
    assert info.value.request_id == "req-fixture-0001"


async def test_async_api_error_becomes_tool_error(api: MockAPI) -> None:
    api.responses[("POST", "/v1/ai/tools/list_campaigns/call")] = [
        json_response(404, fixture("errors/404_not_found.json"))
    ]
    tools = await make_toolkit(api).aget_tools()
    message = await tools[0].ainvoke(TOOL_CALL)
    assert message.status == "error"
    assert "not_found" in message.content and "req-fixture-0001" in message.content


def test_category_rate_limit_without_code(api: MockAPI) -> None:
    api.responses[("POST", "/v1/ai/tools/list_campaigns/call")] = [
        json_response(429, fixture("errors/429_category_rate_limited.json"), {"X-Request-Id": "hdr-req-1"})
    ]
    toolkit = make_toolkit(api, max_retries=0)
    with pytest.raises(AyInboxAPIError) as info:
        toolkit.client.call_tool("list_campaigns", {})
    assert info.value.code == "rate_limit_exceeded"
    assert info.value.request_id == "hdr-req-1"
    assert info.value.retry_after == 12


def test_list_error_raises_api_error(api: MockAPI) -> None:
    api.responses[("GET", "/v1/ai/tools")] = [json_response(403, fixture("errors/403_api_key_mailbox_limited.json"))]
    with pytest.raises(AyInboxAPIError) as info:
        make_toolkit(api).get_tools()
    assert info.value.code == "api_key_mailbox_limited"


def test_get_is_retried_on_503_then_succeeds(api: MockAPI) -> None:
    api.responses[("GET", "/v1/ai/tools")] = [
        json_response(503, fixture("errors/503_service_unavailable.json"), {"Retry-After": "0"}),
        json_response(200, fixture("ai_tools_list.json")),
    ]
    assert len(make_toolkit(api).get_tools()) == 1
    assert len(api.requests) == 2


def test_tool_call_without_idempotency_key_is_not_retried_on_503(api: MockAPI) -> None:
    api.responses[("POST", "/v1/ai/tools/list_campaigns/call")] = [
        json_response(503, fixture("errors/503_service_unavailable.json")),
        json_response(200, fixture("ai_tool_call.json")),
    ]
    toolkit = make_toolkit(api)
    with pytest.raises(AyInboxAPIError):
        toolkit.client.call_tool("list_campaigns", {"limit": 1})
    assert len(api.requests) == 1


def test_tool_call_with_idempotency_key_is_retried_on_503(api: MockAPI) -> None:
    api.responses[("POST", "/v1/ai/tools/list_campaigns/call")] = [
        json_response(503, fixture("errors/503_service_unavailable.json")),
        json_response(200, fixture("ai_tool_call.json")),
    ]
    data = make_toolkit(api).client.call_tool("list_campaigns", {"idempotency_key": "k-1"})
    assert data["name"] == "list_campaigns"
    assert len(api.requests) == 2
    assert all(json.loads(r.content) == {"idempotency_key": "k-1"} for r in api.requests)


def test_tool_call_is_retried_on_429(api: MockAPI) -> None:
    api.responses[("POST", "/v1/ai/tools/list_campaigns/call")] = [
        json_response(429, fixture("errors/429_api_key_rate_limited.json"), {"Retry-After": "0"}),
        json_response(200, fixture("ai_tool_call.json")),
    ]
    assert make_toolkit(api).client.call_tool("list_campaigns")["name"] == "list_campaigns"
    assert len(api.requests) == 2


def test_retries_stop_at_max_retries(api: MockAPI) -> None:
    api.responses[("GET", "/v1/ai/tools")] = [json_response(502, {"error": "Bad Gateway"})]
    with pytest.raises(AyInboxAPIError) as info:
        make_toolkit(api, max_retries=2).get_tools()
    assert info.value.code == "internal_error"
    assert len(api.requests) == 3


def test_network_failure_becomes_tool_error() -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    client = AyInboxToolsClient(API_KEY, http_client=httpx.Client(transport=httpx.MockTransport(fail)))
    tool = AyInboxTool.from_definition(fixture("ai_tools_list.json")["data"][0], client)
    content = tool.invoke({"limit": 1})
    assert "Could not reach AY Inbox" in content
    assert "may or may not have run" in content


def test_include_and_exclude() -> None:
    api = MockAPI()
    api.add("GET", "/v1/ai/tools", json_response(200, CATALOGUE))
    assert [t.name for t in make_toolkit(api, include=["add_tag", "list_threads"]).get_tools()] == [
        "list_threads",
        "add_tag",
    ]
    assert [t.name for t in make_toolkit(api, exclude=["add_tag"]).get_tools()] == ["list_threads", "list_campaigns"]
    assert [t.name for t in make_toolkit(api, include=["add_tag"], exclude=["add_tag"]).get_tools()] == []


def test_include_of_a_tool_not_offered_raises() -> None:
    api = MockAPI()
    api.add("GET", "/v1/ai/tools", json_response(200, CATALOGUE))
    with pytest.raises(ValueError, match="send_reply"):
        make_toolkit(api, include=["send_reply"]).get_tools()


def test_schema_without_properties_gets_an_empty_object() -> None:
    api = MockAPI()
    api.add("GET", "/v1/ai/tools", json_response(200, CATALOGUE))
    tool = make_toolkit(api, include=["list_threads"]).get_tools()[0]
    assert tool.args == {}


def test_env_fallbacks(api: MockAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AYINBOX_API_KEY", API_KEY)
    monkeypatch.setenv("AYINBOX_BASE_URL", "https://example.test/v1/")
    toolkit = AyInboxToolkit(http_client=api.sync_client())
    assert toolkit.client.base_url == "https://example.test/v1"
    toolkit.get_tools()
    assert api.requests[0].url.host == "example.test"
    assert api.requests[0].headers["authorization"] == f"Bearer {API_KEY}"


def test_missing_key_raises() -> None:
    with pytest.raises(AyInboxError, match="AYINBOX_API_KEY"):
        AyInboxToolkit().get_tools()


def test_key_is_never_leaked(api: MockAPI) -> None:
    echo = {"error": "Bad Request", "message": f"bad token {API_KEY}", "code": "bad_request", "request_id": "r1"}
    api.responses[("POST", "/v1/ai/tools/list_campaigns/call")] = [json_response(400, echo)]
    toolkit = make_toolkit(api)
    tool = toolkit.get_tools()[0]
    message = tool.invoke(TOOL_CALL)
    surfaces = [
        repr(toolkit),
        str(toolkit),
        repr(toolkit.client),
        repr(tool),
        str(tool),
        json.dumps(tool.model_dump(mode="json")),
        json.dumps(toolkit.model_dump(mode="json")),
        str(message.content),
        repr(message),
    ]
    for surface in surfaces:
        assert API_KEY not in surface
    with pytest.raises(AyInboxAPIError) as info:
        toolkit.client.call_tool("list_campaigns")
    assert API_KEY not in str(info.value)
    assert API_KEY not in repr(info.value)
    assert API_KEY not in json.dumps(info.value.body)
