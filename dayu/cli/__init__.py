"""统一 CLI 入口包。"""


def run_main() -> int:
    """运行统一 CLI 主入口。

    Args:
        无。

    Returns:
        主入口退出码。

    Raises:
        无。
    """

    from dayu.cli.main import main

    return main()


__all__ = [
    "run_main",
]
