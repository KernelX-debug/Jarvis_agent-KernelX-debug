"""Conocimiento de comandos disponible incluso sin red, LLM o índice personal."""

import json
import re
import unicodedata
from functools import lru_cache
from importlib.resources import files


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.casefold())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return " ".join(text.split()).strip(" ¿¡?!.,;:")


@lru_cache(maxsize=1)
def language_pack() -> dict:
    return json.loads(
        files("jarvis_agent.resources").joinpath("spanish.json").read_text(encoding="utf-8")
    )


def unwrap(text: str) -> str:
    """Retira solo prefijos acotados; conserva mayúsculas y contenido del dictado."""
    value = text.strip().lstrip("¿¡")
    value = re.sub(r"^(?:oye\s+)?jarvis\b[ ,:]*", "", value, flags=re.I)
    value = value.lstrip("¿¡ ")
    value = re.sub(r"^por favor[, ]+", "", value, flags=re.I)
    value = re.sub(r"^(?:puedes|podrías|podrias)\s+", "", value, flags=re.I)
    # No retirar 'por favor' al final: puede pertenecer al texto dictado o búsqueda.
    return value.strip()


def local_response(text: str) -> str | None:
    return language_pack()["responses"].get(normalize(unwrap(text)))
