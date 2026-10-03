from __future__ import annotations

import re
import unicodedata

from jarvis_agent.models import ActionRequest
from jarvis_agent.spanish import language_pack, unwrap


def _normalized(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _without_accents(text: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFD", text) if unicodedata.category(char) != "Mn"
    )


class IntentRouter:
    """Convierte frases acotadas en acciones tipadas.

    El router es deliberadamente determinista: el LLM responde preguntas, pero
    no puede inventar acciones ni argumentos ejecutables.
    """

    _APP_ALIASES = {
        "navegador": "browser",
        "calculadora": "calculator",
        "bloc de notas": "notepad",
        "notas": "notepad",
        "explorador": "file_manager",
        "explorador de archivos": "file_manager",
    }

    def parse(self, text: str) -> ActionRequest | None:
        original = text.strip()
        command = unwrap(original)
        compacted = " ".join(command.split())
        normalized = _normalized(command)
        preserved_command = compacted.lstrip("¿¡")
        plain = _without_accents(normalized).lstrip("¿¡").rstrip(" .?!")
        if not normalized:
            return None

        # Evita que "no abras..." se convierta en una accion por coincidencia parcial.
        if re.match(r"^(no|nunca|jamas)\b", plain):
            return None

        # Solo las órdenes cerradas aceptan un sufijo de cortesía.
        closed = re.sub(r"[, ]+por favor$", "", plain).strip()
        plain = language_pack()["aliases"].get(closed, closed)
        numbers = {
            "uno": "1",
            "dos": "2",
            "tres": "3",
            "cuatro": "4",
            "cinco": "5",
            "seis": "6",
            "siete": "7",
            "ocho": "8",
            "nueve": "9",
            "diez": "10",
        }
        plain = re.sub(
            r"(?<=volumen )(" + "|".join(numbers) + r")$",
            lambda match: numbers[match.group(0)],
            plain,
        )

        if re.fullmatch(
            r"(?:salir|terminar|detente|desconectate|apagate|cerrar agente|terminar proceso)",
            plain,
        ):
            return ActionRequest("stop_agent", original_text=original)

        if re.fullmatch(r"(?:que hora es|dime la hora|hora)", plain):
            return ActionRequest("tell_time", original_text=original)

        if re.fullmatch(r"(?:que fecha es|dime la fecha|fecha|que dia es)", plain):
            return ActionRequest("tell_date", original_text=original)

        if re.fullmatch(r"(?:ayuda|que puedes hacer|comandos)", plain):
            return ActionRequest("show_help", original_text=original)

        match = re.fullmatch(
            r"(?:abre|abrir|inicia|iniciar) (?:el |la )?"
            r"(navegador|calculadora|bloc de notas|notas|explorador(?: de archivos)?)",
            plain,
        )
        if match:
            return ActionRequest(
                "open_app",
                {"app": self._APP_ALIASES[match.group(1)]},
                original,
            )

        match = re.fullmatch(
            r"(?:busca|buscar)(?: en (?:internet|la web))? (.+)",
            preserved_command,
            flags=re.IGNORECASE,
        )
        if match and match.group(1).strip():
            return ActionRequest("web_search", {"query": match.group(1).strip()}, original)

        match = re.fullmatch(
            r"(?:abre|abrir) (?:la )?(?:pagina|página|web|sitio)(?: web)? (.+)",
            preserved_command,
            flags=re.IGNORECASE,
        )
        if match and match.group(1).strip():
            return ActionRequest("open_url", {"url": match.group(1).strip()}, original)

        match = re.fullmatch(r"(?:sube|aumenta|incrementa) (?:el )?volumen(?: (\d+))?", plain)
        if match:
            return ActionRequest(
                "change_volume",
                {"direction": "up", "steps": self._steps(match.group(1))},
                original,
            )

        match = re.fullmatch(r"(?:baja|reduce) (?:el )?volumen(?: (\d+))?", plain)
        if match:
            return ActionRequest(
                "change_volume",
                {"direction": "down", "steps": self._steps(match.group(1))},
                original,
            )

        if re.fullmatch(r"(?:silencia|silenciar|mutea) (?:el )?(?:audio|volumen)", plain):
            return ActionRequest("mute_volume", original_text=original)

        match = re.fullmatch(
            r"(?:escribe|escribir|teclea) (?:el texto )?(.+)",
            preserved_command,
            flags=re.IGNORECASE,
        )
        if match and match.group(1).strip():
            return ActionRequest("type_text", {"text": match.group(1).strip()}, original)

        return None

    @staticmethod
    def _steps(raw: str | None) -> int:
        if not raw:
            return 3
        return min(max(int(raw), 1), 10)
