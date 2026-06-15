"""``dayu.cli.main`` 顶层 KeyboardInterrupt 收口测试。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

from dayu.cli.main import main
from dayu.process_lifecycle.exit_codes import EXIT_CODE_SIGINT


CLI_HELP_COMMANDS: tuple[str | None, ...] = (
    None,
    "interactive",
    "prompt",
    "write",
    "download",
    "upload_filing",
    "upload_filings_from",
    "upload_material",
    "process",
    "process_filing",
    "process_material",
    "init",
    "sessions",
    "runs",
    "cancel",
    "host",
    "conv",
)


@pytest.mark.unit
def test_main_returns_exit_code_sigint_on_keyboard_interrupt() -> None:
    """非交互式命令触发 KeyboardInterrupt 时，``main`` 应收口并返回退出码 130。

    sync 信号 handler 在 ``settle_active_runs`` 后 raise KeyboardInterrupt 或
    SystemExit；该测试覆盖在信号 handler 注册之前 KeyboardInterrupt 已抛到
    顶层、由 ``main`` 顶层兜底返回 EXIT_CODE_SIGINT 的边缘场景。
    """

    def _fake_parse() -> argparse.Namespace:
        return argparse.Namespace(command="download", ticker="MCO")

    fake_fins_module = ModuleType("dayu.cli.commands.fins")

    def _fake_run_fins_command(args: argparse.Namespace) -> int:
        del args
        raise KeyboardInterrupt

    fake_fins_module.__dict__["run_fins_command"] = _fake_run_fins_command

    with (
        patch("dayu.cli.main.parse_arguments", side_effect=_fake_parse),
        patch("dayu.cli.main.configure_standard_streams_for_console_output"),
        patch.dict(sys.modules, {"dayu.cli.commands.fins": fake_fins_module}),
    ):
        result = main()

    assert result == EXIT_CODE_SIGINT


@pytest.mark.unit
def test_module_execution_prints_help() -> None:
    """通过 ``python -m dayu.cli.main`` 执行时应进入 CLI 主入口。"""

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "dayu.cli.main", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert "prompt" in result.stdout


@pytest.mark.unit
@pytest.mark.parametrize("command", CLI_HELP_COMMANDS)
def test_existing_cli_commands_print_help(command: str | None) -> None:
    """所有既有 CLI 命令的帮助入口应保持可用。

    Args:
        command: 要验证的子命令；``None`` 表示顶层入口。

    Returns:
        无。

    Raises:
        AssertionError: 当命令无法正常输出帮助信息时抛出。
    """

    repo_root = Path(__file__).resolve().parents[2]
    args = [sys.executable, "-m", "dayu.cli.main"]
    if command is not None:
        args.append(command)
    args.append("--help")

    result = subprocess.run(
        args,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout
