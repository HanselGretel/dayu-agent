"""应用配置加载端到端测试。"""
from pathlib import Path
from typing import Mapping

import pytest

from dayu.startup.config_file_resolver import ConfigFileResolver
from dayu.startup.config_loader import ConfigLoader
from dayu.startup.prompt_assets import FilePromptAssetStore


def _header_value(headers: object, key: str) -> str | None:
    """安全读取 headers 中的字符串值。"""

    if not isinstance(headers, Mapping):
        return None
    value = headers.get(key)
    return value if isinstance(value, str) else None


def test_load_llm_model_with_env_replace(config_fixtures, monkeypatch):
    """测试 LLM 模型配置加载与环境变量替换"""
    monkeypatch.setenv("TEST_API_KEY", "test-key-123")
    loader = ConfigLoader(ConfigFileResolver(config_fixtures))

    model_config = loader.load_llm_model("test_openai")

    assert model_config.get("runner_type") == "openai_compatible"
    assert _header_value(model_config.get("headers"), "Authorization") == "Bearer test-key-123"


def test_prompt_asset_store_keeps_runtime_placeholders(config_fixtures, monkeypatch):
    """测试 task prompt 读取后保留运行时占位符，不做环境变量替换。"""

    prompts_dir = config_fixtures / "prompts" / "tasks"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / "custom_runtime.md").write_text("API key={{TEST_API_KEY}}", encoding="utf-8")

    monkeypatch.setenv("TEST_API_KEY", "prompt-key")
    store = FilePromptAssetStore(ConfigFileResolver(config_fixtures))

    content = store.load_task_prompt("custom_runtime")
    assert "API key={{TEST_API_KEY}}" in content


def test_prompt_asset_store_reads_manifest_file() -> None:
    """验证 prompt 资产仓储可读取 scene manifest。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    content = store.load_scene_manifest("interactive")

    assert content["scene"] == "interactive"


def test_prompt_asset_store_reads_repair_manifest_file() -> None:
    """验证 prompt 资产仓储可读取 repair scene manifest。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    content = store.load_scene_manifest("repair")

    assert content["scene"] == "repair"


def test_prompt_asset_store_reads_prompt_manifest_file() -> None:
    """验证 prompt 资产仓储可读取 prompt scene manifest。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    content = store.load_scene_manifest("prompt")

    assert content["scene"] == "prompt"


def test_prompt_asset_store_reads_overview_manifest_file() -> None:
    """验证 prompt 资产仓储可读取 overview scene manifest。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    content = store.load_scene_manifest("overview")

    assert content["scene"] == "overview"


def test_fallback_to_package_config(tmp_path):
    """测试缺失文件时 fallback 到包内 dayu/config。"""
    # 只创建空目录，触发 fallback
    workspace_config = tmp_path / "config"
    workspace_config.mkdir(parents=True, exist_ok=True)

    loader = ConfigLoader(ConfigFileResolver(workspace_config))
    models = loader.load_llm_models()

    assert isinstance(models, dict)
    assert len(models.keys()) > 0


def test_load_llm_model_rejects_disabled_cli_runner(tmp_path: Path) -> None:
    """验证被禁用的 CLI runner 会在模型加载阶段显式失败。"""

    config_root = tmp_path / "config"
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "llm_models.json").write_text(
        (
            '{\n'
            '  "legacy_cli": {\n'
            '    "runner_type": "cli",\n'
            '    "command": ["codex", "exec"]\n'
            '  }\n'
            '}\n'
        ),
        encoding="utf-8",
    )

    loader = ConfigLoader(ConfigFileResolver(config_root))

    with pytest.raises(ValueError, match="CLI runner"):
        loader.load_llm_model("legacy_cli")


def test_load_builtin_cursor_sdk_model_and_collects_api_key_env() -> None:
    """默认模型目录应包含 Cursor SDK 模型并收集 CURSOR_API_KEY 依赖。"""

    loader = ConfigLoader(ConfigFileResolver())

    model_config = loader.load_llm_model("cursor-auto")
    referenced_env_vars = loader.collect_model_referenced_env_vars(["cursor-auto"])

    assert model_config.get("runner_type") == "cursor_sdk"
    assert model_config.get("model") == "composer-2.5"
    assert model_config.get("api_key_env") == "CURSOR_API_KEY"
    assert model_config.get("required_tool_names_any") == ["list_documents"]
    assert model_config.get("allowed_tool_names") == [
        "fetch_more",
        "list_documents",
        "get_document_sections",
        "read_section",
        "search_document",
        "list_tables",
        "get_table",
        "get_page_content",
        "get_financial_statement",
        "query_xbrl_facts",
    ]
    assert "CURSOR_API_KEY" in referenced_env_vars
