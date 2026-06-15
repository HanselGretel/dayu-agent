"""AsyncCursorSdkRunner 测试。"""

from __future__ import annotations

import json
import sys
from types import ModuleType
from types import TracebackType
from typing import Any, Callable, Mapping, cast

import pytest

from dayu.contracts.agent_types import build_user_chat_message
from dayu.contracts.protocols import ToolExecutionContext
from dayu.engine.async_cursor_sdk_runner import AsyncCursorSdkRunner
from dayu.engine.events import (
    CURSOR_CUSTOM_TOOL_METADATA_KEY,
    EventType,
    TOOL_LOOP_OWNER_METADATA_KEY,
    TOOL_LOOP_OWNER_RUNNER,
)


class _FakeToolExecutor:
    """用于验证 Cursor customTools 转调当前 ToolExecutor 的桩。"""

    def __init__(self, tool_names: tuple[str, ...] = ("lookup_filing",)) -> None:
        """初始化执行记录。"""

        self.calls: list[tuple[str, dict[str, Any], ToolExecutionContext | None]] = []
        self.tool_names = tool_names

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> dict[str, Any]:
        """记录工具调用并返回结构化结果。"""

        self.calls.append((name, arguments, context))
        return {"ok": True, "value": {"echo": arguments}}

    def get_schemas(self) -> list[dict[str, Any]]:
        """返回一个可转换为 Cursor custom tool 的 schema。"""

        return [
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": "读取财报信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                        },
                        "required": ["ticker"],
                    },
                },
            }
            for tool_name in self.tool_names
        ]

    def clear_cursors(self) -> None:
        """测试桩无需清理游标。"""

    def get_dup_call_spec(self, name: str) -> None:
        """测试桩不声明重复调用策略。"""

        del name
        return None

    def get_execution_context_param_name(self, name: str) -> None:
        """测试桩不注入执行上下文参数名。"""

        del name
        return None

    def get_tool_display_info(self, name: str) -> tuple[str, list[str] | None]:
        """返回默认展示信息。"""

        return name, None

    def register_response_middleware(
        self,
        callback: Callable[[str, dict[str, Any], ToolExecutionContext | None], dict[str, Any]],
    ) -> None:
        """测试桩不支持 middleware。"""

        del callback


class _FakeCursorCustomToolContext:
    """模拟 Cursor SDK Python 传入的 CustomToolContext 对象。"""

    def __init__(self, tool_call_id: str | None) -> None:
        """初始化工具调用 ID。

        Args:
            tool_call_id: Cursor SDK 传入的工具调用 ID。

        Returns:
            无。

        Raises:
            无。
        """

        self.tool_call_id = tool_call_id


@pytest.mark.asyncio
async def test_cursor_runner_reports_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """缺少 Cursor API Key 时应返回明确错误事件。"""

    monkeypatch.delenv("CURSOR_API_KEY", raising=False)
    runner = AsyncCursorSdkRunner()

    events = [event async for event in runner.call([build_user_chat_message("hello")])]

    assert events[0].type == EventType.ERROR
    assert events[0].metadata["error_type"] == "missing_cursor_api_key"


@pytest.mark.asyncio
async def test_cursor_runner_uses_custom_tools_from_current_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cursor customTools 应转调当前 run 的 ToolExecutor。"""

    monkeypatch.setenv("CURSOR_API_KEY", "cursor_test")
    executor = _FakeToolExecutor(("lookup_filing", "web_search"))
    captured_local: dict[str, object] = {}

    class _FakeResult:
        status = "finished"
        result = "最终回答"
        id = "run_1"

    class _FakeRun:
        def messages(self) -> list[dict[str, object]]:
            return [
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {"type": "text", "text": "最终回答"},
                        ]
                    },
                }
            ]

        def wait(self) -> _FakeResult:
            return _FakeResult()

    class _FakeAgent:
        agent_id = "agent_1"

        def send(self, prompt: str) -> _FakeRun:
            assert "读取苹果财报" in prompt
            raw_tools = captured_local["custom_tools"]
            assert isinstance(raw_tools, Mapping)
            assert tuple(raw_tools.keys()) == ("lookup_filing",)
            raw_tool = raw_tools["lookup_filing"]
            assert isinstance(raw_tool, Mapping)
            execute = cast(
                Callable[[dict[str, str], _FakeCursorCustomToolContext], str],
                raw_tool["execute"],
            )
            assert callable(execute)
            tool_result = execute({"ticker": "AAPL"}, _FakeCursorCustomToolContext("call_1"))
            payload = json.loads(tool_result)
            assert payload["ok"] is True
            return _FakeRun()

    class _FakeAgentContext:
        def __enter__(self) -> _FakeAgent:
            return _FakeAgent()

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            del exc_type, exc, traceback

    class _FakeAgentFactory:
        @staticmethod
        def create(
            *,
            api_key: str,
            model: str,
            name: str,
            local: Mapping[str, object],
        ) -> _FakeAgentContext:
            assert api_key == "cursor_test"
            assert model == "composer-2.5"
            assert name == "cursor-auto"
            captured_local.update(dict(local))
            return _FakeAgentContext()

    fake_module = ModuleType("cursor_sdk")
    setattr(fake_module, "Agent", _FakeAgentFactory)
    monkeypatch.setitem(sys.modules, "cursor_sdk", fake_module)

    runner = AsyncCursorSdkRunner(
        model="composer-2.5",
        name="cursor-auto",
        allowed_tool_names=("lookup_filing",),
        required_tool_names_any=("lookup_filing",),
    )
    runner.set_tools(executor)

    events = [
        event
        async for event in runner.call(
            [build_user_chat_message("读取苹果财报")],
            trace_context={"run_id": "run_outer", "iteration_id": "iter_1"},
        )
    ]

    assert [event.type for event in events] == [
        EventType.TOOL_CALL_DISPATCHED,
        EventType.TOOL_CALL_RESULT,
        EventType.CONTENT_DELTA,
        EventType.CONTENT_COMPLETE,
        EventType.DONE,
    ]
    dispatched = events[0]
    assert dispatched.data["id"] == "call_1"
    assert dispatched.data["name"] == "lookup_filing"
    assert dispatched.data["arguments"] == {"ticker": "AAPL"}
    assert dispatched.data["index_in_iteration"] == 0
    assert dispatched.metadata[TOOL_LOOP_OWNER_METADATA_KEY] == TOOL_LOOP_OWNER_RUNNER
    assert dispatched.metadata[CURSOR_CUSTOM_TOOL_METADATA_KEY] is True
    assert dispatched.metadata["run_id"] == "run_outer"
    assert dispatched.metadata["iteration_id"] == "iter_1"

    tool_result = events[1]
    assert tool_result.data["id"] == "call_1"
    assert tool_result.data["name"] == "lookup_filing"
    assert tool_result.data["arguments"] == {"ticker": "AAPL"}
    assert tool_result.data["result"] == {"ok": True, "value": {"echo": {"ticker": "AAPL"}}}
    assert tool_result.metadata[TOOL_LOOP_OWNER_METADATA_KEY] == TOOL_LOOP_OWNER_RUNNER
    assert tool_result.metadata[CURSOR_CUSTOM_TOOL_METADATA_KEY] is True

    assert events[-2].data == "最终回答"
    assert executor.calls[0][0] == "lookup_filing"
    assert executor.calls[0][1] == {"ticker": "AAPL"}
    context = executor.calls[0][2]
    assert context is not None
    assert context.run_id == "run_outer"
    assert context.iteration_id == "iter_1"
    assert context.tool_call_id == "call_1"
    assert isinstance(events[-1].data, dict)
    assert events[-1].data["called_tool_names"] == ("lookup_filing",)


@pytest.mark.asyncio
async def test_cursor_runner_rejects_answer_when_required_tool_not_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """配置必需工具后，Cursor 未实际调用 Dayu 工具时应拒绝回答。"""

    monkeypatch.setenv("CURSOR_API_KEY", "cursor_test")
    executor = _FakeToolExecutor()

    class _FakeResult:
        status = "finished"
        result = "脱离本地财报的回答"
        id = "run_1"

    class _FakeRun:
        def messages(self) -> list[dict[str, object]]:
            return [
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {"type": "text", "text": "脱离本地财报的回答"},
                        ]
                    },
                }
            ]

        def wait(self) -> _FakeResult:
            return _FakeResult()

    class _FakeAgent:
        agent_id = "agent_1"

        def send(self, prompt: str) -> _FakeRun:
            assert "读取苹果财报" in prompt
            return _FakeRun()

    class _FakeAgentContext:
        def __enter__(self) -> _FakeAgent:
            return _FakeAgent()

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            del exc_type, exc, traceback

    class _FakeAgentFactory:
        @staticmethod
        def create(
            *,
            api_key: str,
            model: str,
            name: str,
            local: Mapping[str, object],
        ) -> _FakeAgentContext:
            assert api_key == "cursor_test"
            assert model == "composer-2.5"
            assert name == "cursor-auto"
            assert "custom_tools" in local
            return _FakeAgentContext()

    fake_module = ModuleType("cursor_sdk")
    setattr(fake_module, "Agent", _FakeAgentFactory)
    monkeypatch.setitem(sys.modules, "cursor_sdk", fake_module)

    runner = AsyncCursorSdkRunner(
        model="composer-2.5",
        name="cursor-auto",
        required_tool_names_any=("lookup_filing",),
    )
    runner.set_tools(executor)

    events = [event async for event in runner.call([build_user_chat_message("读取苹果财报")])]

    assert [event.type for event in events] == [EventType.ERROR]
    assert events[0].metadata["error_type"] == "cursor_required_tool_not_called"
    assert events[0].metadata["required_tool_names_any"] == ("lookup_filing",)
    assert events[0].metadata["called_tool_names"] == ()
    assert executor.calls == []


@pytest.mark.asyncio
async def test_cursor_runner_uses_fallback_tool_call_id_when_context_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cursor custom tool 缺少 tool_call_id 时应使用稳定 fallback。"""

    monkeypatch.setenv("CURSOR_API_KEY", "cursor_test")
    executor = _FakeToolExecutor()

    class _FakeResult:
        status = "finished"
        result = "最终回答"
        id = "run_1"

    class _FakeRun:
        def messages(self) -> list[dict[str, object]]:
            return [
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {"type": "text", "text": "最终回答"},
                        ]
                    },
                }
            ]

        def wait(self) -> _FakeResult:
            return _FakeResult()

    class _FakeAgent:
        agent_id = "agent_1"

        def send(self, prompt: str) -> _FakeRun:
            raw_tools = captured_local["custom_tools"]
            assert isinstance(raw_tools, Mapping)
            raw_tool = raw_tools["lookup_filing"]
            assert isinstance(raw_tool, Mapping)
            execute = cast(
                Callable[[dict[str, str], None], str],
                raw_tool["execute"],
            )
            execute({"ticker": "AAPL"}, None)
            return _FakeRun()

    class _FakeAgentContext:
        def __enter__(self) -> _FakeAgent:
            return _FakeAgent()

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            del exc_type, exc, traceback

    class _FakeAgentFactory:
        @staticmethod
        def create(
            *,
            api_key: str,
            model: str,
            name: str,
            local: Mapping[str, object],
        ) -> _FakeAgentContext:
            captured_local.update(dict(local))
            return _FakeAgentContext()

    captured_local: dict[str, object] = {}
    fake_module = ModuleType("cursor_sdk")
    setattr(fake_module, "Agent", _FakeAgentFactory)
    monkeypatch.setitem(sys.modules, "cursor_sdk", fake_module)

    runner = AsyncCursorSdkRunner(model="composer-2.5", name="cursor-auto")
    runner.set_tools(executor)

    events = [event async for event in runner.call([build_user_chat_message("读取苹果财报")])]

    assert events[0].type == EventType.TOOL_CALL_DISPATCHED
    assert events[0].data["id"] == "cursor_call_0"
    assert events[1].type == EventType.TOOL_CALL_RESULT
    assert events[1].data["id"] == "cursor_call_0"
