# Cursor API Runner 开发回顾

## 目标

为 `dayu-cli prompt` 与 `prompt --label` 新增 Cursor API Key 访问路径。新增能力必须保持旧的 `openai_compatible` 模型、CLI 参数、用户 workspace 数据和财报仓储读取方式不变。

## 关键设计决策

- 新增 `runner_type=cursor_sdk`，不把 Cursor 伪装成 OpenAI-compatible endpoint。
- 内置模型名为 `cursor-auto`，只加入 `prompt` 与 `prompt_mt` 的 `allowed_names`，不改任何 scene 默认模型。
- `cursor-auto` 的 Cursor SDK `model` 字段当前固定为 `composer-2.5`；如需恢复 Cursor 自动路由，只改配置值回 `auto`，不需要改代码。
- Cursor API Key 通过 `api_key_env` 配置，默认读取 `CURSOR_API_KEY`。
- 原计划使用 stdio MCP server 暴露 Fins 工具；开发时根据 Cursor SDK 2026-06 的 `local.custom_tools` 能力调整为直接把当前 `ToolExecutor` 暴露给 Cursor local Agent。这样工具执行仍在 Host 当前进程内，能复用已经装配好的权限、limits、ticker 上下文和用户数据路径。
- Cursor Agent 自带内部 tool loop，因此 `AsyncCursorSdkRunner` 不向外层 `AsyncAgent` 发送 `TOOL_CALLS_BATCH_DONE`，避免外层再次回填工具消息并重复驱动模型。
- Cursor custom tool 的真实执行会映射为 Dayu 标准 `tool_call_dispatched` / `tool_call_result` 事件，并带 `tool_loop_owner=runner` 与 `cursor_custom_tool=true` metadata；`AsyncAgent` 对这些事件只做 trace/UI 透传，不写入外层 `tool_calls_data`，因此不会触发 `tool_calls_batch_missing` 或重复 tool loop。
- `cursor-auto` 不能只靠 prompt 约束“不要联网”。模型配置新增 `allowed_tool_names` 与 `required_tool_names_any`：默认只向 Cursor 暴露 Dayu 本地财报读取工具，并要求至少实际调用 `list_documents`；若 Cursor 没有触达本地文档目录，Runner 拒绝输出模型答案。

## 重要变更

- `dayu.contracts.model_config` 新增 Cursor SDK 模型配置与 runner 参数类型。
- `dayu.host.agent_builder` 与 `dayu.engine.runner_factory` 新增 `cursor_sdk` 分支。
- `dayu.engine.async_cursor_sdk_runner` 封装 Cursor SDK 动态导入、API Key 校验、`custom_tools` 映射、最终文本事件转换。
- `dayu.engine.async_cursor_sdk_runner` 会记录 custom tool 的真实执行工具名，而不是依赖 Cursor 返回的展示事件判断是否调用工具。
- `dayu.engine.async_cursor_sdk_runner` 在 custom tool 执行点生成标准 `tool_call_dispatched` / `tool_call_result` 事件，供 ToolTrace、CLI 工具状态和调用次数统计复用；这些事件带 runner 内部 loop 标记，外层 `AsyncAgent` 不会据此再次回填工具消息。
- `dayu/config/llm_models.json` 新增 `cursor-auto`，`prompt.json` 与 `prompt_mt.json` 允许显式选择该模型；该模型默认只暴露 `fetch_more` 与 Fins 本地读取/检索/表格/XBRL 工具。
- `ConfigLoader.collect_model_referenced_env_vars()` 现在能从 `api_key_env` 收集 `CURSOR_API_KEY`。
- `dayu.cli.main` 补齐 `python -m dayu.cli.main` 执行入口；`dayu.cli.__init__` 不再提前导入 `dayu.cli.main`，避免 runpy warning。
- 用户 workspace `config` 可增量加入 `cursor-auto`，但保持 `prompt` / `prompt_mt` 的 `default_name` 不变。旧 Dayu 读取配置和继续使用默认模型不会受影响；旧 Dayu 若显式选择 `cursor-auto`，会因为不认识 `runner_type=cursor_sdk` 而报错，这是预期的能力边界。

## 遇到的问题

- 当前开发机需要使用项目根目录 `.venv`。不要复用离线包目录内的 `.venv`，否则会混淆当前源码和旧发行包运行结果。
- 全量 `pip install -e .` 在本机卡在第三方 `docling-parse` wheel 构建；本次真实 prompt 冒烟改为用当前源码路径直接运行，并按需补齐 `cursor-sdk` 与 `edgartools`。
- Cursor SDK 的 custom tool Python 文档不如 TypeScript 示例完整，适配层选择使用 raw local dict 的 `custom_tools` 字段，集中隔离在 Runner 内。
- Cursor SDK 的 stream/tool 事件属于外部 beta 边界，事件字段读取使用窄动态适配，避免向 Host/Service 泄漏 SDK 事件结构。
- `python -m dayu.cli.main` 原本只导入模块没有进入 `main()`，根因是缺少 `if __name__ == "__main__"` 执行块；补齐后还需要移除 `dayu.cli.__init__` 对 `dayu.cli.main` 的提前导入，才能消除 runpy warning。
- 真实调用中 Cursor SDK Python 传入的是 `CustomToolContext` 对象而不是 mapping；`_extract_tool_call_id()` 已按 `tool_call_id` 属性读取，并保留 mapping 分支给测试桩和未来 bridge。
- 没安装 `docling-core` 时，Cursor 即使能调用 Dayu 工具，也会在读取本地 Docling JSON 时失败。本地财报强约束要同时验证工具调用链和文档读取依赖。

## 验证方式

- 新增 `tests/engine/test_async_cursor_sdk_runner.py`，通过 fake `cursor_sdk` 验证缺 key、`custom_tools` 映射、工具执行上下文、工具白名单、必需工具未调用拒绝输出、标准工具事件映射和最终事件。
- 更新 `tests/engine/test_async_agent.py`，验证带 runner 内部 loop 标记的工具结果事件不会触发外层 `tool_calls_batch_missing`。
- 更新 `tests/application/test_agent_builder_extra.py`，验证 Cursor 模型配置能生成 `AgentCreateArgs`。
- 更新 `tests/integration/test_config_loader_e2e.py`，验证内置 `cursor-auto` 可加载、收集 `CURSOR_API_KEY`，并带有本地财报工具约束。
- 当前项目 `.venv` 验证：Python 3.11.15，`python -m dayu.cli.main --help` 可正常进入 CLI 主入口且无 runpy warning。
- 当前项目 `.venv` 验证：`pytest tests/cli/test_main.py -q`、`pytest tests/application/test_console_output.py -q` 通过，`pyright dayu/cli/__init__.py dayu/cli/main.py tests/cli/test_main.py` 通过。
- 当前项目 `.venv` 验证：Cursor 与 CLI 相关目标测试共 24 个通过；本次触达文件 scoped pyright 通过。全量 `pyright` 在当前 `.venv` 下仍失败，主要原因是未安装完整 `docling`、`fastapi`、`playwright`、`streamlit` 可选运行依赖，以及既有 `tests/engine/test_web_tools.py` fake response 类型问题。
- 真实 Cursor 冒烟已在用户 workspace `/Users/hansen/Documents/f/dayu-agent-0.1.4-macos-arm64-offline/workspace` 跑通：`AAPL` 最新可用财报返回 Form `10-Q`、报告期 `2026-03-28`、document id `fil_0000320193-26-000013`。
- 真实 Cursor 本地工具冒烟已在同一 workspace 跑通：`300476.SZ` 最新年报通过 `list_documents` 与 `get_document_sections` 返回 document id `fil_cn_b33863a90d7f8652351036702faee7135ad4c154`，章节数量 `761`。

## 遗留风险

- `cursor-sdk` 当前处于 beta，`local.custom_tools` 的 Python raw dict 字段名若变化，需要只改 `AsyncCursorSdkRunner`。
- 当前 Runner 通过同步 SDK 在线程中执行，外层取消只能中断等待与返回超时，不能保证立即取消 Cursor SDK 内部 run。
- Cursor 内部工具调用现已映射为标准 `tool_call_dispatched` / `tool_call_result` 事件并进入 ToolTrace；但仍不发送 `tool_calls_batch_done`，外层 Agent 不会据此继续多轮工具回填。
- `required_tool_names_any` 只能保证 Cursor 至少触达本地文档目录；具体回答是否充分阅读年报/季报/管理层讨论，仍需依赖 scene prompt、工具结果和后续审计链约束。

## 后续建议

- 后续日常验证优先使用项目根目录 `.venv`：`source .venv/bin/activate && python -m dayu.cli.main ...`。
- 如果 Cursor SDK 后续提供稳定 Python custom tool dataclass，应把 raw dict 映射替换为官方 dataclass，并保留现有 fake SDK 单测作为回归。
