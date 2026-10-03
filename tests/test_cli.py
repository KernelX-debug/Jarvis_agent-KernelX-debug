from __future__ import annotations

import os

import pytest

import jarvis_agent.cli as cli
from jarvis_agent.models import TurnResult


@pytest.fixture(autouse=True)
def clean_jarvis_environment(monkeypatch) -> None:
    for key in tuple(os.environ):
        if key.startswith("JARVIS_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cli, "_configure_console_encoding", lambda: None)


class AskOnlyAssistant:
    def __init__(self) -> None:
        self.questions: list[str] = []

    def ask(self, question: str) -> TurnResult:
        self.questions.append(question)
        return TurnResult("respuesta segura")


def test_ask_command_uses_query_path_even_for_action_like_text(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    assistant = AskOnlyAssistant()
    monkeypatch.setattr(cli, "build_assistant", lambda _settings: assistant)

    exit_code = cli.main(
        [
            "--env-file",
            str(tmp_path / "missing.env"),
            "ask",
            "abre",
            "la",
            "calculadora",
        ]
    )

    assert exit_code == 0
    assert assistant.questions == ["abre la calculadora"]
    assert "respuesta segura" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("flag", "environment_value", "expected"),
    [
        ("--dry-run", "false", True),
        ("--live", "true", False),
    ],
)
def test_execution_flag_overrides_environment(
    monkeypatch,
    tmp_path,
    flag: str,
    environment_value: str,
    expected: bool,
) -> None:
    monkeypatch.setenv("JARVIS_DRY_RUN", environment_value)
    args = cli.build_parser().parse_args(
        ["--env-file", str(tmp_path / "missing.env"), flag, "actions"]
    )

    settings = cli._settings_from_args(args)

    assert settings.dry_run is expected


def test_no_execution_flag_keeps_safe_default(tmp_path) -> None:
    args = cli.build_parser().parse_args(["--env-file", str(tmp_path / "missing.env"), "actions"])

    settings = cli._settings_from_args(args)

    assert settings.dry_run is True


def test_dry_run_and_live_are_mutually_exclusive() -> None:
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as error:
        parser.parse_args(["--dry-run", "--live", "actions"])

    assert error.value.code == 2


def test_invalid_configuration_fails_before_dispatch(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    monkeypatch.setenv("JARVIS_DRY_RUN", "almost")

    def unexpected_dispatch(*_args, **_kwargs) -> int:
        pytest.fail("No se debe despachar un comando con configuracion invalida.")

    monkeypatch.setattr(cli, "run_ask", unexpected_dispatch)

    exit_code = cli.main(["--env-file", str(tmp_path / "missing.env"), "ask", "pregunta"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "booleano invalido" in captured.err
