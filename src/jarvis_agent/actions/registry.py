from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from jarvis_agent.models import ActionRequest, ActionResult, ActionStatus, RiskLevel

from .system import HandlerOutput, SystemActions


@dataclass(frozen=True, slots=True)
class ActionDefinition:
    name: str
    risk: RiskLevel
    handler: Callable[[dict[str, object]], HandlerOutput]


class AuditLogger:
    _SENSITIVE_ARGUMENTS = {"text", "query", "url", "token", "api_key", "password", "secret"}
    _SENSITIVE_ACTIONS = {"type_text", "web_search", "open_url"}

    def __init__(self, path: Path | None) -> None:
        self.path = path

    def write(self, result: ActionResult) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        arguments = self._redact(result.action.arguments_dict())
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "action": result.action.name,
            "arguments": arguments,
            "status": result.status.value,
            "message": (
                "<redacted>" if result.action.name in self._SENSITIVE_ACTIONS else result.message
            ),
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    @classmethod
    def _redact(cls, value: object, key: str | None = None) -> object:
        if key and key.lower() in cls._SENSITIVE_ARGUMENTS:
            return "<redacted>"
        if isinstance(value, dict):
            return {name: cls._redact(item, name) for name, item in value.items()}
        if isinstance(value, list):
            return [cls._redact(item) for item in value]
        return value


class ActionRegistry:
    """Lista permitida, politica de confirmacion y auditoria de acciones."""

    def __init__(
        self,
        system_actions: SystemActions | None = None,
        *,
        dry_run: bool = False,
        audit_path: Path | None = None,
    ) -> None:
        actions = system_actions or SystemActions()
        self.dry_run = dry_run
        self.audit = AuditLogger(audit_path)
        self._definitions = {
            "open_app": ActionDefinition("open_app", RiskLevel.SAFE, actions.open_app),
            "open_url": ActionDefinition("open_url", RiskLevel.SAFE, actions.open_url),
            "web_search": ActionDefinition("web_search", RiskLevel.SAFE, actions.web_search),
            "tell_time": ActionDefinition("tell_time", RiskLevel.SAFE, actions.tell_time),
            "tell_date": ActionDefinition("tell_date", RiskLevel.SAFE, actions.tell_date),
            "change_volume": ActionDefinition(
                "change_volume", RiskLevel.SAFE, actions.change_volume
            ),
            "mute_volume": ActionDefinition("mute_volume", RiskLevel.SAFE, actions.mute_volume),
            "show_help": ActionDefinition("show_help", RiskLevel.SAFE, actions.show_help),
            "type_text": ActionDefinition("type_text", RiskLevel.CONFIRM, actions.type_text),
            "stop_agent": ActionDefinition("stop_agent", RiskLevel.SAFE, actions.stop_agent),
        }

    @property
    def action_names(self) -> tuple[str, ...]:
        return tuple(self._definitions)

    def risk_for(self, name: str) -> RiskLevel:
        definition = self._definitions.get(name)
        return definition.risk if definition else RiskLevel.BLOCKED

    def execute(self, request: ActionRequest, *, confirmed: bool = False) -> ActionResult:
        definition = self._definitions.get(request.name)
        if definition is None or definition.risk is RiskLevel.BLOCKED:
            return self._finish(
                ActionResult(
                    ActionStatus.BLOCKED,
                    f"La accion '{request.name}' no esta permitida.",
                    request,
                )
            )

        if self.dry_run:
            return self._finish(
                ActionResult(
                    ActionStatus.DRY_RUN,
                    f"Simulacion: ejecutaria la accion '{request.name}'.",
                    request,
                    should_stop=request.name == "stop_agent",
                )
            )

        if definition.risk is RiskLevel.CONFIRM and not confirmed:
            return ActionResult(
                ActionStatus.NEEDS_CONFIRMATION,
                f"La accion '{request.name}' requiere confirmacion.",
                request,
            )

        try:
            output = definition.handler(request.arguments_dict())
            result = ActionResult(
                ActionStatus.EXECUTED,
                output.message,
                request,
                should_stop=output.should_stop,
            )
        except Exception as exc:  # La frontera de acciones convierte fallos en resultados tipados.
            result = ActionResult(
                ActionStatus.FAILED,
                f"No pude ejecutar '{request.name}': {exc}",
                request,
            )
        return self._finish(result)

    def _finish(self, result: ActionResult) -> ActionResult:
        self.audit.write(result)
        return result
