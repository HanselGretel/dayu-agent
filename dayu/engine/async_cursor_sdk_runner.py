"""异步 Cursor SDK Runner。

本模块把 Cursor Python SDK 封装到 Dayu 现有 ``AsyncRunner`` 协议下。
Cursor Agent 自带内部 agent loop，因此这里不把 Cursor 的工具调用结果回灌
成 OpenAI-style tool batch，而是通过 local customTools 让 Cursor 在当前进程
直接调用 Dayu 已装配的 ``ToolExecutor``。
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from threading import Lock
from types import ModuleType
from types import TracebackType
from typing import Protocol, TypedDict, cast

from dayu.contracts.agent_types import AgentMessage, JsonValue
from dayu.contracts.cancellation import CancellationToken
from dayu.contracts.protocols import ToolExecutionContext
from dayu.engine.events import (
    StreamEvent,
    content_complete,
    content_delta,
    done_event,
    error_event,
    metadata_event,
    reasoning_delta,
    runner_internal_tool_event_metadata,
    tool_call_dispatched,
    tool_call_result,
)
from dayu.engine.protocols import ToolExecutor
from dayu.log import Log

MODULE = "ENGINE.CURSOR_SDK_RUNNER"
DEFAULT_CURSOR_API_KEY_ENV = "CURSOR_API_KEY"
DEFAULT_CURSOR_MODEL = "auto"
DEFAULT_CURSOR_TIMEOUT_SECONDS = 3600.0
_CURSOR_DASHBOARD_INTEGRATIONS_URL = "https://cursor.com/dashboard/integrations"
_CUSTOM_TOOLS_FIELD = "custom_tools"
_INPUT_SCHEMA_FIELD = "inputSchema"
_EXECUTE_FIELD = "execute"
_CURSOR_TOOL_CALL_ID_FALLBACK_PREFIX = "cursor_call_"


CursorLocalValue = str | Mapping[str, "CursorCustomToolDefinition"]


class CursorCustomToolContextLike(Protocol):
    """Cursor SDK custom tool context 的最小结构。"""

    tool_call_id: str | None


class CursorCustomToolDefinition(TypedDict, total=False):
    """Cursor SDK custom tool 定义。"""

    description: str
    inputSchema: Mapping[str, JsonValue]
    execute: Callable[
        [Mapping[str, JsonValue], Mapping[str, JsonValue] | CursorCustomToolContextLike | None],
        str,
    ]


class CursorRunResult(Protocol):
    """Cursor SDK RunResult 的最小结构协议。"""

    status: str
    result: str | None
    id: str


class CursorRun(Protocol):
    """Cursor SDK Run 的最小结构协议。"""

    def messages(self) -> Iterable[object]:
        """返回 Cursor SDK 消息迭代器。"""

        ...

    def wait(self) -> CursorRunResult:
        """等待 run 结束并返回终态结果。"""

        ...


class CursorAgent(Protocol):
    """Cursor SDK Agent 的最小结构协议。"""

    agent_id: str

    def send(self, prompt: str) -> CursorRun:
        """发送用户 prompt。"""

        ...


class CursorAgentContext(Protocol):
    """Cursor SDK Agent 上下文管理器协议。"""

    def __enter__(self) -> CursorAgent:
        """进入上下文并返回 Agent。"""

        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """退出上下文并释放 Agent 资源。"""

        ...


class CursorAgentFactory(Protocol):
    """Cursor SDK Agent 工厂的最小结构协议。"""

    def create(
        self,
        *,
        api_key: str,
        model: str,
        name: str,
        local: Mapping[str, CursorLocalValue],
    ) -> CursorAgentContext:
        """创建 Cursor Agent。"""

        ...


@dataclass(frozen=True)
class CursorSdkRunOutcome:
    """Cursor SDK 单次运行的同步执行结果。"""

    events: tuple[StreamEvent, ...]
    final_text: str
    status: str
    run_id: str | None
    agent_id: str | None
    called_tool_names: tuple[str, ...]


def _load_cursor_sdk_module() -> ModuleType:
    """加载 Cursor SDK 模块。

    Args:
        无。

    Returns:
        已导入的 ``cursor_sdk`` 模块。

    Raises:
        RuntimeError: 未安装 ``cursor-sdk`` 时抛出。
    """

    try:
        return import_module("cursor_sdk")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "未安装 cursor-sdk，请先安装项目依赖，或执行 `pip install cursor-sdk`。"
        ) from exc


def _load_cursor_agent_factory(module: ModuleType) -> CursorAgentFactory:
    """从动态 SDK 模块中读取 Agent 工厂。

    Args:
        module: ``cursor_sdk`` 模块。

    Returns:
        Cursor SDK Agent 工厂。

    Raises:
        RuntimeError: SDK 模块不包含 Agent 工厂时抛出。
    """

    # Cursor SDK 处于 beta，运行期导入能避免旧 Dayu 使用路径在缺少 SDK 时失败。
    agent_factory = getattr(module, "Agent", None)
    if agent_factory is None:
        raise RuntimeError("cursor-sdk 未导出 Agent，当前安装版本不支持 Agent.create")
    return cast(CursorAgentFactory, agent_factory)


def _read_external_attr(value: object, name: str) -> object | None:
    """读取 Cursor SDK 动态事件上的字段。

    Args:
        value: SDK 返回的消息、结果或嵌套对象。
        name: 字段名。

    Returns:
        字段值；不存在时返回 ``None``。

    Raises:
        无。
    """

    if isinstance(value, Mapping):
        return value.get(name)
    # SDK 事件 dataclass 与 Mapping 会并存，外部 beta 边界只能做窄动态读取。
    return getattr(value, name, None)


def _read_external_string(value: object, name: str) -> str:
    """读取外部对象上的字符串字段。"""

    raw_value = _read_external_attr(value, name)
    return raw_value if isinstance(raw_value, str) else ""


def _read_external_mapping(value: object, name: str) -> Mapping[str, JsonValue] | None:
    """读取外部对象上的 JSON mapping 字段。"""

    raw_value = _read_external_attr(value, name)
    return cast(Mapping[str, JsonValue], raw_value) if isinstance(raw_value, Mapping) else None


def _json_dumps(value: JsonValue | Mapping[str, JsonValue]) -> str:
    """序列化 JSON 值为中文友好的文本。"""

    return json.dumps(value, ensure_ascii=False)


def _normalize_tool_names(raw_tool_names: Sequence[str] | None) -> frozenset[str]:
    """规范化工具名列表。

    Args:
        raw_tool_names: 外部配置传入的工具名序列。

    Returns:
        去空白、去重后的工具名集合。

    Raises:
        无。
    """

    if raw_tool_names is None:
        return frozenset()
    names: set[str] = set()
    for raw_name in raw_tool_names:
        name = str(raw_name).strip()
        if name:
            names.add(name)
    return frozenset(names)


def _is_allowed_tool_name(name: str, allowed_tool_names: frozenset[str]) -> bool:
    """判断工具是否允许暴露给 Cursor。

    Args:
        name: 工具名。
        allowed_tool_names: 允许工具名集合；空集合表示不限制。

    Returns:
        ``True`` 表示可暴露。

    Raises:
        无。
    """

    return not allowed_tool_names or name in allowed_tool_names


def _message_content(message: AgentMessage) -> str:
    """读取 Dayu 消息内容。"""

    content = message.get("content")
    return content if isinstance(content, str) else ""


def _format_messages_for_cursor(messages: Sequence[AgentMessage]) -> str:
    """把 Dayu messages 转换成 Cursor Agent 的单次 prompt。

    Args:
        messages: Dayu 当前运行态消息列表。

    Returns:
        可发送给 Cursor SDK 的 prompt 文本。

    Raises:
        无。
    """

    parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "")).strip() or "unknown"
        content = _message_content(message).strip()
        if not content:
            continue
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts).strip()


class _CursorToolEventCollector:
    """线程安全收集 Cursor custom tool 标准工具事件。"""

    def __init__(self) -> None:
        """初始化事件缓冲。"""

        self._events: list[StreamEvent] = []
        self._lock = Lock()

    def append(self, event: StreamEvent) -> None:
        """追加单个工具事件。

        Args:
            event: 待追加的标准工具事件。

        Returns:
            无。

        Raises:
            无。
        """

        with self._lock:
            self._events.append(event)

    def drain(self) -> list[StreamEvent]:
        """取出并清空当前缓冲的工具事件。

        Args:
            无。

        Returns:
            按追加顺序排列的工具事件列表。

        Raises:
            无。
        """

        with self._lock:
            drained = self._events
            self._events = []
            return drained


def _resolve_cursor_tool_call_id(
    *,
    context: Mapping[str, JsonValue] | CursorCustomToolContextLike | None,
    index: int,
) -> str:
    """解析 Cursor custom tool 的工具调用 ID。

    Args:
        context: Cursor SDK 传入的 custom tool context。
        index: 当前 run 内的工具调用序号。

    Returns:
        Cursor 提供的 tool_call_id；缺失时使用稳定 fallback。

    Raises:
        无。
    """

    tool_call_id = _extract_tool_call_id(context)
    if tool_call_id:
        return tool_call_id
    return f"{_CURSOR_TOOL_CALL_ID_FALLBACK_PREFIX}{index}"


def _schema_to_cursor_tool(
    *,
    schema: Mapping[str, object],
    executor: ToolExecutor,
    trace_context: Mapping[str, JsonValue],
    call_index_provider: Callable[[], int],
    tool_call_recorder: Callable[[str], None],
    tool_event_collector: Callable[[StreamEvent], None],
) -> tuple[str, CursorCustomToolDefinition] | None:
    """把 OpenAI tool schema 转换为 Cursor custom tool。

    Args:
        schema: ``ToolExecutor.get_schemas()`` 返回的单个 OpenAI tool schema。
        executor: 当前 run 的工具执行器。
        trace_context: 外层 Agent 传入的 trace 上下文。
        call_index_provider: 生成工具调用序号的回调。
        tool_call_recorder: 记录已成功执行的工具名。
        tool_event_collector: 收集标准工具请求/结果事件。

    Returns:
        ``(tool_name, definition)``；schema 非法时返回 ``None``。

    Raises:
        无。
    """

    function = schema.get("function")
    if not isinstance(function, Mapping):
        return None
    raw_name = function.get("name")
    if not isinstance(raw_name, str) or not raw_name.strip():
        return None
    name = raw_name.strip()
    raw_description = function.get("description")
    description = raw_description if isinstance(raw_description, str) else ""
    raw_parameters = function.get("parameters")
    parameters = cast(Mapping[str, JsonValue], raw_parameters) if isinstance(raw_parameters, Mapping) else {}

    def execute(
        arguments: Mapping[str, JsonValue],
        context: Mapping[str, JsonValue] | CursorCustomToolContextLike | None = None,
    ) -> str:
        """执行单个 Cursor custom tool 调用。"""

        index = call_index_provider()
        run_id = _trace_context_string(trace_context, "run_id")
        iteration_id = _trace_context_string(trace_context, "iteration_id")
        tool_call_id = _resolve_cursor_tool_call_id(context=context, index=index)
        arguments_dict = dict(arguments)
        internal_tool_metadata = runner_internal_tool_event_metadata(
            run_id=run_id,
            iteration_id=iteration_id,
        )
        tool_event_collector(
            tool_call_dispatched(
                tool_call_id,
                name,
                arguments_dict,
                index_in_iteration=index,
                **internal_tool_metadata,
            )
        )
        result = executor.execute(
            name,
            arguments_dict,
            context=ToolExecutionContext(
                run_id=run_id,
                iteration_id=iteration_id,
                tool_call_id=tool_call_id,
                index_in_iteration=index,
            ),
        )
        tool_event_collector(
            tool_call_result(
                tool_call_id,
                result,
                name=name,
                arguments=arguments_dict,
                index_in_iteration=index,
                **internal_tool_metadata,
            )
        )
        tool_call_recorder(name)
        return _json_dumps(cast(Mapping[str, JsonValue], result))

    return name, {
        "description": description,
        _INPUT_SCHEMA_FIELD: parameters,
        _EXECUTE_FIELD: execute,
    }


def _extract_tool_call_id(context: Mapping[str, JsonValue] | CursorCustomToolContextLike | None) -> str | None:
    """从 Cursor custom tool context 中提取工具调用 ID。"""

    if context is None:
        return None
    if isinstance(context, Mapping):
        raw_call_id = context.get("tool_call_id") or context.get("callId") or context.get("call_id")
    else:
        # Cursor SDK Python 当前传入 CustomToolContext 对象，字段名为 tool_call_id；
        # 保留 Mapping 分支用于测试桩与未来 bridge 兼容。
        raw_call_id = context.tool_call_id
    return raw_call_id if isinstance(raw_call_id, str) and raw_call_id.strip() else None


def _trace_context_string(trace_context: Mapping[str, JsonValue], key: str) -> str | None:
    """读取 trace_context 字符串字段。"""

    value = trace_context.get(key)
    return value if isinstance(value, str) and value.strip() else None


def _extract_trace_context(extra_payloads: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    """从 Runner 透传参数中提取 trace_context。"""

    raw_trace_context = extra_payloads.get("trace_context")
    return cast(Mapping[str, JsonValue], raw_trace_context) if isinstance(raw_trace_context, Mapping) else {}


def _extract_text_from_content_blocks(blocks: object) -> str:
    """从 Cursor assistant content blocks 中提取文本。"""

    if isinstance(blocks, str):
        return blocks
    if not isinstance(blocks, Sequence):
        return ""
    parts: list[str] = []
    for block in blocks:
        block_type = _read_external_string(block, "type")
        if block_type and block_type != "text":
            continue
        text = _read_external_string(block, "text")
        if text:
            parts.append(text)
    return "".join(parts)


def _extract_assistant_text(message: object) -> str:
    """从 Cursor assistant 消息中提取文本。"""

    nested_message = _read_external_attr(message, "message")
    if nested_message is not None:
        nested_content = _read_external_attr(nested_message, "content")
        text = _extract_text_from_content_blocks(nested_content)
        if text:
            return text
    return _read_external_string(message, "text")


def _events_from_cursor_message(message: object) -> tuple[StreamEvent, ...]:
    """把 Cursor SDK 流消息转换为 Dayu 事件。"""

    message_type = _read_external_string(message, "type")
    if message_type == "thinking":
        text = _read_external_string(message, "text")
        return (reasoning_delta(text),) if text else ()
    if message_type == "assistant":
        text = _extract_assistant_text(message)
        return (content_delta(text),) if text else ()
    if message_type == "tool_call":
        call_id = _read_external_string(message, "call_id")
        name = _read_external_string(message, "name")
        status = _read_external_string(message, "status")
        args = _read_external_mapping(message, "args")
        return (
            metadata_event(
                "cursor_tool_call",
                {
                    "call_id": call_id,
                    "name": name,
                    "status": status,
                    "args": dict(args or {}),
                },
            ),
        )
    if message_type == "request":
        request_id = _read_external_string(message, "request_id")
        return (metadata_event("cursor_request_id", request_id),) if request_id else ()
    return ()


class AsyncCursorSdkRunner:
    """基于 Cursor Python SDK 的异步 Runner。

    Args:
        model: Cursor 模型 ID，例如 ``auto`` 或 ``composer-2.5``。
        api_key_env: Cursor API Key 环境变量名。
        cwd: Cursor local runtime 工作目录。
        name: Runner 名称。
        timeout: 单次 run 超时时间。
        supports_stream: 是否声明支持流式输出。
        supports_tool_calling: 是否声明支持工具调用。
        allowed_tool_names: 允许暴露给 Cursor 的工具名；为空表示不限制。
        required_tool_names_any: 必须至少实际调用其一的工具名；为空表示不要求。
        cancellation_token: Host 传入的取消令牌。

    Raises:
        ValueError: 配置为空或 timeout 非法时抛出。
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_CURSOR_MODEL,
        api_key_env: str = DEFAULT_CURSOR_API_KEY_ENV,
        cwd: str = ".",
        name: str | None = None,
        timeout: int | float = DEFAULT_CURSOR_TIMEOUT_SECONDS,
        supports_stream: bool = True,
        supports_tool_calling: bool = True,
        allowed_tool_names: Sequence[str] | None = None,
        required_tool_names_any: Sequence[str] | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> None:
        """初始化 Cursor SDK Runner。"""

        model_name = str(model or "").strip()
        key_env = str(api_key_env or "").strip()
        if not model_name:
            raise ValueError("cursor_sdk runner model 不能为空")
        if not key_env:
            raise ValueError("cursor_sdk runner api_key_env 不能为空")
        timeout_seconds = float(timeout)
        if timeout_seconds <= 0:
            raise ValueError("cursor_sdk runner timeout 必须大于 0")
        self.model = model_name
        self.api_key_env = key_env
        self.cwd = str(Path(cwd or ".").expanduser())
        self.name = str(name or model_name)
        self.timeout = timeout_seconds
        self._supports_stream = supports_stream
        self._supports_tool_calling = supports_tool_calling
        self._allowed_tool_names = _normalize_tool_names(allowed_tool_names)
        self._required_tool_names_any = _normalize_tool_names(required_tool_names_any)
        self._tool_executor: ToolExecutor | None = None
        self._cancellation_token = cancellation_token
        self._tool_call_index = 0
        self._tool_call_index_lock = Lock()
        self._called_tool_names: list[str] = []
        self._called_tool_names_lock = Lock()

    def set_tools(self, executor: ToolExecutor | None) -> None:
        """设置当前 Runner 使用的工具执行器。

        Args:
            executor: 工具执行器；``None`` 表示禁用工具。

        Returns:
            无。

        Raises:
            无。
        """

        self._tool_executor = executor

    def is_supports_tool_calling(self) -> bool:
        """返回是否支持工具调用。

        Args:
            无。

        Returns:
            ``True`` 表示 Cursor local customTools 可暴露工具。

        Raises:
            无。
        """

        return self._supports_tool_calling

    async def close(self) -> None:
        """关闭 Runner 持有的资源。

        Args:
            无。

        Returns:
            无。

        Raises:
            无。
        """

        self._tool_executor = None

    async def call(
        self,
        messages: list[AgentMessage],
        *,
        stream: bool = True,
        **extra_payloads: JsonValue,
    ) -> AsyncIterator[StreamEvent]:
        """调用 Cursor SDK 并返回 Dayu 流式事件。

        Args:
            messages: Dayu 当前消息列表。
            stream: 是否请求流式输出。Cursor Runner 总是通过 SDK run 消息观察输出。
            **extra_payloads: 外层 Agent 透传参数，目前只消费 ``trace_context``。

        Yields:
            Dayu ``StreamEvent``。

        Raises:
            无。运行失败会转换为 ``error_event``。
        """

        del stream
        if self._cancellation_token is not None:
            self._cancellation_token.raise_if_cancelled()
        prompt = _format_messages_for_cursor(messages)
        if not prompt:
            yield error_event("cursor_sdk runner 收到空 prompt", recoverable=False, error_type="empty_prompt")
            return
        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            yield error_event(
                (
                    f"缺少 {self.api_key_env}。请在 Cursor Dashboard Integrations 创建 API Key，"
                    f"并设置环境变量：export {self.api_key_env}=\"cursor_...\"。"
                ),
                recoverable=False,
                error_type="missing_cursor_api_key",
                docs_url=_CURSOR_DASHBOARD_INTEGRATIONS_URL,
            )
            return

        trace_context = _extract_trace_context(extra_payloads)
        self._reset_tool_call_index()
        self._reset_called_tool_names()
        try:
            outcome = await asyncio.wait_for(
                asyncio.to_thread(
                    self._run_cursor_sync,
                    api_key=api_key,
                    prompt=prompt,
                    trace_context=trace_context,
                ),
                timeout=self.timeout,
            )
        except TimeoutError as exc:
            yield error_event(
                f"Cursor SDK run 超时（{self.timeout:.0f}s）",
                exception=exc,
                recoverable=False,
                error_type="cursor_sdk_timeout",
            )
            return
        except Exception as exc:
            yield error_event(
                "Cursor SDK run 启动或执行失败",
                exception=exc,
                recoverable=False,
                error_type="cursor_sdk_error",
            )
            return

        if outcome.status == "error":
            yield error_event(
                "Cursor SDK run 返回 error 状态",
                recoverable=False,
                error_type="cursor_run_error",
                cursor_run_id=outcome.run_id,
                cursor_agent_id=outcome.agent_id,
            )
            return
        if not self._satisfies_required_tool_names(outcome.called_tool_names):
            yield error_event(
                (
                    "Cursor SDK run 未按模型配置调用必需的 Dayu 本地工具；"
                    "为避免脱离本地财报数据回答，本次结果已拒绝。"
                ),
                recoverable=False,
                error_type="cursor_required_tool_not_called",
                required_tool_names_any=tuple(sorted(self._required_tool_names_any)),
                called_tool_names=outcome.called_tool_names,
                cursor_run_id=outcome.run_id,
                cursor_agent_id=outcome.agent_id,
            )
            return
        for event in outcome.events:
            yield event
        yield content_complete(outcome.final_text)
        yield done_event(
            {
                "finish_reason": "stop",
                "cursor_run_id": outcome.run_id,
                "cursor_agent_id": outcome.agent_id,
                "called_tool_names": outcome.called_tool_names,
            }
        )

    def _run_cursor_sync(
        self,
        *,
        api_key: str,
        prompt: str,
        trace_context: Mapping[str, JsonValue],
    ) -> CursorSdkRunOutcome:
        """在线程中同步执行 Cursor SDK run。"""

        sdk_module = _load_cursor_sdk_module()
        agent_factory = _load_cursor_agent_factory(sdk_module)
        tool_event_collector = _CursorToolEventCollector()
        local_options = self._build_local_options(trace_context, tool_event_collector=tool_event_collector)

        def _flush_tool_events() -> None:
            events.extend(tool_event_collector.drain())

        events: list[StreamEvent] = []
        final_text_parts: list[str] = []
        with agent_factory.create(
            api_key=api_key,
            model=self.model,
            name=self.name,
            local=local_options,
        ) as agent:
            run = agent.send(prompt)
            for message in run.messages():
                _flush_tool_events()
                for event in _events_from_cursor_message(message):
                    events.append(event)
                    if event is not None and event.type.value == "content_delta" and isinstance(event.data, str):
                        final_text_parts.append(event.data)
            _flush_tool_events()
            result = run.wait()
            _flush_tool_events()
            result_text = result.result if isinstance(result.result, str) else ""
            final_text = result_text or "".join(final_text_parts)
            run_id = result.id if isinstance(result.id, str) else None
            return CursorSdkRunOutcome(
                events=tuple(events),
                final_text=final_text,
                status=result.status,
                run_id=run_id,
                agent_id=agent.agent_id if isinstance(agent.agent_id, str) else None,
                called_tool_names=self._snapshot_called_tool_names(),
            )

    def _build_local_options(
        self,
        trace_context: Mapping[str, JsonValue],
        *,
        tool_event_collector: _CursorToolEventCollector,
    ) -> Mapping[str, CursorLocalValue]:
        """构造 Cursor SDK local options。"""

        local_options: dict[str, CursorLocalValue] = {"cwd": self.cwd}
        if self._tool_executor is not None and self._supports_tool_calling:
            custom_tools = self._build_custom_tools(
                self._tool_executor,
                trace_context,
                tool_event_collector=tool_event_collector,
            )
            if custom_tools:
                local_options[_CUSTOM_TOOLS_FIELD] = custom_tools
        return local_options

    def _build_custom_tools(
        self,
        executor: ToolExecutor,
        trace_context: Mapping[str, JsonValue],
        *,
        tool_event_collector: _CursorToolEventCollector,
    ) -> Mapping[str, CursorCustomToolDefinition]:
        """根据当前 ToolExecutor 构造 Cursor customTools。"""

        tools: dict[str, CursorCustomToolDefinition] = {}
        for schema in executor.get_schemas():
            converted = _schema_to_cursor_tool(
                schema=cast(Mapping[str, object], schema),
                executor=executor,
                trace_context=trace_context,
                call_index_provider=self._next_tool_call_index,
                tool_call_recorder=self._record_tool_call,
                tool_event_collector=tool_event_collector.append,
            )
            if converted is None:
                Log.warn("跳过非法工具 schema，无法转换为 Cursor custom tool", module=MODULE)
                continue
            name, definition = converted
            if not _is_allowed_tool_name(name, self._allowed_tool_names):
                continue
            tools[name] = definition
        return tools

    def _next_tool_call_index(self) -> int:
        """返回当前 run 内的下一个工具调用序号。"""

        with self._tool_call_index_lock:
            index = self._tool_call_index
            self._tool_call_index += 1
            return index

    def _reset_tool_call_index(self) -> None:
        """重置当前 run 内的工具调用序号。"""

        with self._tool_call_index_lock:
            self._tool_call_index = 0

    def _record_tool_call(self, name: str) -> None:
        """记录 Cursor 实际执行的 Dayu 工具名。

        Args:
            name: 工具名。

        Returns:
            无。

        Raises:
            无。
        """

        with self._called_tool_names_lock:
            self._called_tool_names.append(name)

    def _reset_called_tool_names(self) -> None:
        """清空当前 run 的工具调用记录。

        Args:
            无。

        Returns:
            无。

        Raises:
            无。
        """

        with self._called_tool_names_lock:
            self._called_tool_names = []

    def _snapshot_called_tool_names(self) -> tuple[str, ...]:
        """读取当前 run 的工具调用记录快照。

        Args:
            无。

        Returns:
            工具名元组，顺序与实际执行顺序一致。

        Raises:
            无。
        """

        with self._called_tool_names_lock:
            return tuple(self._called_tool_names)

    def _satisfies_required_tool_names(self, called_tool_names: Sequence[str]) -> bool:
        """判断当前 run 是否满足必需工具调用约束。

        Args:
            called_tool_names: 当前 run 实际成功执行的工具名。

        Returns:
            ``True`` 表示满足配置约束。

        Raises:
            无。
        """

        if not self._required_tool_names_any:
            return True
        return bool(self._required_tool_names_any.intersection(called_tool_names))


__all__ = ["AsyncCursorSdkRunner", "DEFAULT_CURSOR_API_KEY_ENV", "DEFAULT_CURSOR_MODEL"]
