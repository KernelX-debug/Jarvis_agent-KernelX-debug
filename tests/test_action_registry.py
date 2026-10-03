from __future__ import annotations

import json
from typing import Any

import pytest

from jarvis_agent.actions.registry import ActionRegistry
from jarvis_agent.actions.system import HandlerOutput
from jarvis_agent.models import ActionRequest, ActionStatus, RiskLevel


class RecordingSystemActions:
    def __init__(self, *, fail_on: str | None = None) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.fail_on = fail_on

    def _record(self, name: str, arguments: dict[str, Any]) -> HandlerOutput:
        self.calls.append((name, dict(arguments)))
        if self.fail_on == name:
            raise RuntimeError("fallo controlado")
        return HandlerOutput(f"{name}: ok", should_stop=name == "stop_agent")

    def open_app(self, arguments: dict[str, Any]) -> HandlerOutput:
        return self._record("open_app", arguments)

    def open_url(self, arguments: dict[str, Any]) -> HandlerOutput:
        return self._record("open_url", arguments)

    def web_search(self, arguments: dict[str, Any]) -> HandlerOutput:
        return self._record("web_search", arguments)

    def tell_time(self, arguments: dict[str, Any]) -> HandlerOutput:
        return self._record("tell_time", arguments)

    def tell_date(self, arguments: dict[str, Any]) -> HandlerOutput:
        return self._record("tell_date", arguments)

    def change_volume(self, arguments: dict[str, Any]) -> HandlerOutput:
        return self._record("change_volume", arguments)

    def mute_volume(self, arguments: dict[str, Any]) -> HandlerOutput:
        return self._record("mute_volume", arguments)

    def show_help(self, arguments: dict[str, Any]) -> HandlerOutput:
        return self._record("show_help", arguments)

    def type_text(self, arguments: dict[str, Any]) -> HandlerOutput:
        return self._record("type_text", arguments)

    def stop_agent(self, arguments: dict[str, Any]) -> HandlerOutput:
        return self._record("stop_agent", arguments)


def test_safe_action_executes_once() -> None:
    system = RecordingSystemActions()
    registry = ActionRegistry(system_actions=system)
    request = ActionRequest("open_app", {"app": "calculator"})

    result = registry.execute(request)

    assert result.status is ActionStatus.EXECUTED
    assert result.success is True
    assert system.calls == [("open_app", {"app": "calculator"})]


def test_request_copies_arguments_before_execution() -> None:
    system = RecordingSystemActions()
    registry = ActionRegistry(system_actions=system)
    arguments = {"app": "calculator"}
    request = ActionRequest("open_app", arguments)
    arguments["app"] = "notepad"

    registry.execute(request)

    assert system.calls == [("open_app", {"app": "calculator"})]


def test_request_deeply_freezes_nested_arguments() -> None:
    nested = {"options": {"targets": ["original"]}}
    request = ActionRequest("future_action", nested)
    nested["options"]["targets"].append("altered")

    assert request.arguments_dict() == {"options": {"targets": ["original"]}}
    with pytest.raises(TypeError):
        request.arguments["options"]["extra"] = True  # type: ignore[index]


def test_sensitive_action_requires_confirmation_before_calling_system() -> None:
    system = RecordingSystemActions()
    registry = ActionRegistry(system_actions=system)
    request = ActionRequest("type_text", {"text": "dato privado"})

    pending = registry.execute(request)

    assert pending.status is ActionStatus.NEEDS_CONFIRMATION
    assert pending.success is False
    assert system.calls == []

    executed = registry.execute(request, confirmed=True)

    assert executed.status is ActionStatus.EXECUTED
    assert system.calls == [("type_text", {"text": "dato privado"})]


@pytest.mark.parametrize("confirmed", [False, True])
def test_dry_run_never_calls_system(confirmed: bool) -> None:
    system = RecordingSystemActions()
    registry = ActionRegistry(system_actions=system, dry_run=True)

    result = registry.execute(
        ActionRequest("type_text", {"text": "no escribir"}),
        confirmed=confirmed,
    )

    assert result.status is ActionStatus.DRY_RUN
    assert result.success is True
    assert system.calls == []


def test_dry_run_preserves_stop_signal_without_stopping_anything() -> None:
    system = RecordingSystemActions()
    registry = ActionRegistry(system_actions=system, dry_run=True)

    result = registry.execute(ActionRequest("stop_agent"))

    assert result.status is ActionStatus.DRY_RUN
    assert result.should_stop is True
    assert system.calls == []


def test_unknown_action_is_blocked_even_when_marked_confirmed() -> None:
    system = RecordingSystemActions()
    registry = ActionRegistry(system_actions=system)

    result = registry.execute(
        ActionRequest("run_shell", {"command": "whoami"}),
        confirmed=True,
    )

    assert result.status is ActionStatus.BLOCKED
    assert result.success is False
    assert registry.risk_for("run_shell") is RiskLevel.BLOCKED
    assert system.calls == []


def test_handler_error_becomes_failed_result() -> None:
    system = RecordingSystemActions(fail_on="tell_time")
    registry = ActionRegistry(system_actions=system)

    result = registry.execute(ActionRequest("tell_time"))

    assert result.status is ActionStatus.FAILED
    assert "fallo controlado" in result.message
    assert system.calls == [("tell_time", {})]


def test_audit_redacts_typed_text(tmp_path) -> None:
    audit_path = tmp_path / "audit" / "actions.jsonl"
    system = RecordingSystemActions()
    registry = ActionRegistry(system_actions=system, audit_path=audit_path)

    registry.execute(
        ActionRequest("type_text", {"text": "secreto", "window": "editor"}),
        confirmed=True,
    )

    record = json.loads(audit_path.read_text(encoding="utf-8"))
    assert record["arguments"] == {"text": "<redacted>", "window": "editor"}
    assert record["message"] == "<redacted>"
    assert "secreto" not in audit_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("action", "arguments", "secret"),
    [
        ("type_text", {"text": "CLAVE-SUPER-SECRETA"}, "CLAVE-SUPER-SECRETA"),
        ("web_search", {"query": "diagnostico privado"}, "diagnostico privado"),
        ("open_url", {"url": "https://example.test/?token=privado"}, "privado"),
    ],
)
def test_dry_run_audit_never_contains_sensitive_payload(
    tmp_path, action: str, arguments: dict[str, object], secret: str
) -> None:
    audit_path = tmp_path / "actions.jsonl"
    registry = ActionRegistry(
        system_actions=RecordingSystemActions(), dry_run=True, audit_path=audit_path
    )

    result = registry.execute(ActionRequest(action, arguments))

    serialized = audit_path.read_text(encoding="utf-8")
    assert result.status is ActionStatus.DRY_RUN
    assert secret not in result.message
    assert secret not in serialized
    assert "<redacted>" in serialized
