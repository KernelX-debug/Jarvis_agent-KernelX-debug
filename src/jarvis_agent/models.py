from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any


class RiskLevel(StrEnum):
    """Nivel de riesgo que la politica asigna a una accion."""

    SAFE = "safe"
    CONFIRM = "confirm"
    BLOCKED = "blocked"


class ActionStatus(StrEnum):
    EXECUTED = "executed"
    NEEDS_CONFIRMATION = "needs_confirmation"
    BLOCKED = "blocked"
    FAILED = "failed"
    DRY_RUN = "dry_run"


@dataclass(frozen=True, slots=True)
class ActionRequest:
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    original_text: str = ""

    def __post_init__(self) -> None:
        # Congela una copia: una confirmacion nunca puede aprobar argumentos mutados despues.
        object.__setattr__(self, "arguments", _freeze_mapping(self.arguments))

    def arguments_dict(self) -> dict[str, Any]:
        """Entrega una copia mutable solo en la frontera del ejecutor."""

        return {key: _thaw(value) for key, value in self.arguments.items()}


@dataclass(frozen=True, slots=True)
class ActionResult:
    status: ActionStatus
    message: str
    action: ActionRequest
    should_stop: bool = False

    @property
    def success(self) -> bool:
        return self.status in {ActionStatus.EXECUTED, ActionStatus.DRY_RUN}


@dataclass(frozen=True, slots=True)
class SearchHit:
    source: str
    content: str
    score: float
    chunk_index: int


@dataclass(frozen=True, slots=True)
class IngestReport:
    source: str
    files: int
    chunks: int
    skipped: int = 0


@dataclass(frozen=True, slots=True)
class TurnResult:
    text: str
    action_result: ActionResult | None = None
    pending_action: ActionRequest | None = None
    confirmation_id: str | None = None
    sources: tuple[str, ...] = ()
    should_stop: bool = False


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("Los nombres de argumentos de una accion deben ser texto.")
        frozen[key] = _freeze(item)
    return MappingProxyType(frozen)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Tipo de argumento de accion no permitido: {type(value).__name__}.")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value
