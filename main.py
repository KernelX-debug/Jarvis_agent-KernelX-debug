"""Entrada compatible: `python main.py chat` desde el repositorio."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def run() -> int:
    from jarvis_agent.cli import main

    return main()


if __name__ == "__main__":
    raise SystemExit(run())
