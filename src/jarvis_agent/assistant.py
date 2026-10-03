from __future__ import annotations

import secrets
import time
from dataclasses import dataclass

from jarvis_agent.actions import ActionRegistry, IntentRouter
from jarvis_agent.knowledge import KnowledgeBase
from jarvis_agent.models import ActionRequest, ActionStatus, TurnResult
from jarvis_agent.providers import LanguageModel
from jarvis_agent.spanish import local_response


@dataclass(slots=True)
class _PendingConfirmation:
    token: str
    request: ActionRequest
    expires_at: float


class JarvisAssistant:
    """Orquestador: intencion -> politica -> accion, o consulta -> RAG -> LLM."""

    def __init__(
        self,
        router: IntentRouter,
        actions: ActionRegistry,
        knowledge: KnowledgeBase,
        language_model: LanguageModel,
        *,
        confirmation_ttl: float = 60.0,
    ) -> None:
        self.router = router
        self.actions = actions
        self.knowledge = knowledge
        self.language_model = language_model
        self.confirmation_ttl = confirmation_ttl
        self._pending: _PendingConfirmation | None = None

    def process(self, text: str) -> TurnResult:
        self._pending = None
        cleaned = text.strip()
        if not cleaned:
            return TurnResult("No escuche ningun comando.")

        request = self.router.parse(cleaned)
        if request is not None:
            result = self.actions.execute(request)
            if result.status is ActionStatus.NEEDS_CONFIRMATION:
                token = secrets.token_urlsafe(18)
                # Se conserva una copia exacta para impedir que cambien los argumentos.
                frozen_request = ActionRequest(
                    request.name, request.arguments_dict(), request.original_text
                )
                self._pending = _PendingConfirmation(
                    token,
                    frozen_request,
                    time.monotonic() + self.confirmation_ttl,
                )
                return TurnResult(
                    self._confirmation_prompt(frozen_request),
                    action_result=result,
                    pending_action=frozen_request,
                    confirmation_id=token,
                )
            return TurnResult(
                result.message,
                action_result=result,
                should_stop=result.should_stop,
            )

        response = local_response(cleaned)
        if response:
            return TurnResult(response)
        return self.ask(cleaned)

    def ask(self, question: str) -> TurnResult:
        """Responde una consulta sin interpretarla nunca como accion del sistema."""

        cleaned = question.strip()
        if not cleaned:
            return TurnResult("La pregunta esta vacia.")
        hits = self.knowledge.search(cleaned, limit=4)
        try:
            answer = self.language_model.answer(cleaned, hits)
        except Exception as exc:
            answer = f"No pude consultar el modelo configurado: {exc}"
        sources = tuple(dict.fromkeys(hit.source for hit in hits))
        return TurnResult(answer, sources=sources)

    def confirm(self, confirmation_id: str, *, accepted: bool) -> TurnResult:
        pending = self._pending
        self._pending = None
        if pending is None or not secrets.compare_digest(pending.token, confirmation_id):
            return TurnResult("La confirmacion no coincide o ya fue utilizada.")
        if time.monotonic() > pending.expires_at:
            return TurnResult("La confirmacion expiro; repite el comando.")
        if not accepted:
            return TurnResult("Accion cancelada. No hice ningun cambio.")
        result = self.actions.execute(pending.request, confirmed=True)
        return TurnResult(
            result.message,
            action_result=result,
            should_stop=result.should_stop,
        )

    @staticmethod
    def _confirmation_prompt(request: ActionRequest) -> str:
        if request.name == "type_text":
            length = len(str(request.arguments.get("text", "")))
            return (
                f"Voy a escribir {length} caracteres en la ventana que tenga el foco. ¿Confirmas?"
            )
        return f"La accion '{request.name}' puede modificar el sistema. ¿Confirmas?"
