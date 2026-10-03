from __future__ import annotations

from typing import Any

import pytest

import jarvis_agent.assistant as assistant_module
from jarvis_agent.actions.registry import ActionRegistry
from jarvis_agent.actions.router import IntentRouter
from jarvis_agent.actions.system import HandlerOutput
from jarvis_agent.assistant import JarvisAssistant
from jarvis_agent.models import ActionStatus, SearchHit


class RecordingSystemActions:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def _record(self, name: str, arguments: dict[str, Any]) -> HandlerOutput:
        self.calls.append((name, dict(arguments)))
        return HandlerOutput(f"respuesta de {name}", should_stop=name == "stop_agent")

    def __getattribute__(self, name: str):
        registered = {
            "open_app",
            "open_url",
            "web_search",
            "tell_time",
            "tell_date",
            "change_volume",
            "mute_volume",
            "show_help",
            "type_text",
            "stop_agent",
        }
        if name in registered:
            return lambda arguments: self._record(name, arguments)
        return object.__getattribute__(self, name)


class RecordingKnowledge:
    def __init__(self, hits: list[SearchHit] | None = None) -> None:
        self.hits = hits or []
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, *, limit: int = 4) -> list[SearchHit]:
        self.calls.append((query, limit))
        return list(self.hits)


class RecordingLanguageModel:
    name = "fake"

    def __init__(self, answer: str = "respuesta local") -> None:
        self.response = answer
        self.calls: list[tuple[str, list[SearchHit]]] = []

    def answer(self, question: str, context: list[SearchHit]) -> str:
        self.calls.append((question, list(context)))
        return self.response


@pytest.fixture
def components():
    system = RecordingSystemActions()
    actions = ActionRegistry(system_actions=system)
    knowledge = RecordingKnowledge()
    model = RecordingLanguageModel()
    assistant = JarvisAssistant(IntentRouter(), actions, knowledge, model)
    return assistant, system, knowledge, model


def test_registered_action_does_not_enter_rag_pipeline(components) -> None:
    assistant, system, knowledge, model = components

    turn = assistant.process("dime la hora")

    assert turn.action_result is not None
    assert turn.action_result.status is ActionStatus.EXECUTED
    assert system.calls == [("tell_time", {})]
    assert knowledge.calls == []
    assert model.calls == []


def test_question_uses_rag_and_never_calls_system() -> None:
    hits = [
        SearchHit("manual.md", "Jarvis funciona sin red.", 1.0, 0),
        SearchHit("manual.md", "Usa una lista permitida.", 0.8, 1),
        SearchHit("seguridad.md", "Las acciones se confirman.", 0.7, 0),
    ]
    system = RecordingSystemActions()
    knowledge = RecordingKnowledge(hits)
    model = RecordingLanguageModel("respuesta basada en contexto")
    assistant = JarvisAssistant(
        IntentRouter(),
        ActionRegistry(system_actions=system),
        knowledge,
        model,
    )

    turn = assistant.process("¿Cómo protege mis datos el manual?")

    assert turn.text == "respuesta basada en contexto"
    assert turn.sources == ("manual.md", "seguridad.md")
    assert knowledge.calls == [("¿Cómo protege mis datos el manual?", 4)]
    assert model.calls == [("¿Cómo protege mis datos el manual?", hits)]
    assert system.calls == []


def test_explicit_ask_never_interprets_question_as_action(components) -> None:
    assistant, system, knowledge, model = components

    turn = assistant.ask("abre la calculadora")

    assert turn.text == "respuesta local"
    assert knowledge.calls == [("abre la calculadora", 4)]
    assert model.calls == [("abre la calculadora", [])]
    assert system.calls == []


def test_valid_confirmation_is_single_use(components, monkeypatch) -> None:
    assistant, system, _knowledge, _model = components
    monkeypatch.setattr(assistant_module.secrets, "token_urlsafe", lambda _length: "token-1")

    pending = assistant.process("escribe hola")

    assert pending.confirmation_id == "token-1"
    assert pending.action_result is not None
    assert pending.action_result.status is ActionStatus.NEEDS_CONFIRMATION
    assert system.calls == []

    accepted = assistant.confirm("token-1", accepted=True)
    replayed = assistant.confirm("token-1", accepted=True)

    assert accepted.action_result is not None
    assert accepted.action_result.status is ActionStatus.EXECUTED
    assert system.calls == [("type_text", {"text": "hola"})]
    assert replayed.action_result is None
    assert "ya fue utilizada" in replayed.text


def test_pending_action_arguments_cannot_be_mutated(components) -> None:
    assistant, system, _knowledge, _model = components
    pending = assistant.process("escribe texto original")

    assert pending.pending_action is not None
    with pytest.raises(TypeError):
        pending.pending_action.arguments["text"] = "texto alterado"  # type: ignore[index]

    accepted = assistant.confirm(pending.confirmation_id or "", accepted=True)

    assert accepted.action_result is not None
    assert system.calls == [("type_text", {"text": "texto original"})]


def test_rejected_confirmation_cancels_without_side_effects(components) -> None:
    assistant, system, _knowledge, _model = components
    pending = assistant.process("escribe dato")

    cancelled = assistant.confirm(pending.confirmation_id or "", accepted=False)

    assert cancelled.action_result is None
    assert "cancelada" in cancelled.text.lower()
    assert system.calls == []


def test_wrong_token_invalidates_pending_confirmation(components) -> None:
    assistant, system, _knowledge, _model = components
    pending = assistant.process("escribe dato")

    wrong = assistant.confirm("token-incorrecto", accepted=True)
    later = assistant.confirm(pending.confirmation_id or "", accepted=True)

    assert "no coincide" in wrong.text
    assert "ya fue utilizada" in later.text
    assert system.calls == []


def test_expired_confirmation_never_executes(components, monkeypatch) -> None:
    assistant, system, _knowledge, _model = components
    assistant.confirmation_ttl = 5.0
    clock = [100.0]
    monkeypatch.setattr(assistant_module.time, "monotonic", lambda: clock[0])
    pending = assistant.process("escribe dato")
    clock[0] = 105.01

    expired = assistant.confirm(pending.confirmation_id or "", accepted=True)

    assert "expiro" in expired.text
    assert expired.action_result is None
    assert system.calls == []


def test_model_failure_is_contained_as_a_text_response(components) -> None:
    assistant, system, knowledge, model = components

    def fail(_question: str, _context: list[SearchHit]) -> str:
        raise RuntimeError("modelo no disponible")

    model.answer = fail
    turn = assistant.process("pregunta sin acción")

    assert "modelo no disponible" in turn.text
    assert knowledge.calls == [("pregunta sin acción", 4)]
    assert system.calls == []
