from __future__ import annotations

import pytest

from jarvis_agent.actions.router import IntentRouter


@pytest.fixture
def router() -> IntentRouter:
    return IntentRouter()


@pytest.mark.parametrize(
    ("spoken", "name", "arguments"),
    [
        ("  DIME   LA HORA  ", "tell_time", {}),
        ("Dime la hora.", "tell_time", {}),
        ("¿Qué hora es?", "tell_time", {}),
        ("qué fecha es", "tell_date", {}),
        ("¿qué puedes hacer?".strip("¿?"), "show_help", {}),
        ("apágate", "stop_agent", {}),
        ("abre la calculadora", "open_app", {"app": "calculator"}),
        ("Abre la calculadora.", "open_app", {"app": "calculator"}),
        ("abrir el bloc de notas", "open_app", {"app": "notepad"}),
        ("inicia explorador de archivos", "open_app", {"app": "file_manager"}),
        ("busca en internet café peruano", "web_search", {"query": "café peruano"}),
        ("abre la página example.org", "open_url", {"url": "example.org"}),
        ("aumenta el volumen", "change_volume", {"direction": "up", "steps": 3}),
        ("baja volumen 4", "change_volume", {"direction": "down", "steps": 4}),
        ("silencia el audio", "mute_volume", {}),
        (
            "escribe el texto Hola, Perú",
            "type_text",
            {"text": "Hola, Perú"},
        ),
    ],
)
def test_parse_supported_intents(
    router: IntentRouter,
    spoken: str,
    name: str,
    arguments: dict[str, object],
) -> None:
    request = router.parse(spoken)

    assert request is not None
    assert request.name == name
    assert request.arguments == arguments
    assert request.original_text == spoken.strip()


@pytest.mark.parametrize(
    "spoken",
    [
        "",
        "   ",
        "no abras el navegador",
        "nunca abre la calculadora",
        "jamás silencies el audio",
        "abre la calculadora y borra todo",
        "ejecuta powershell",
        "sube volumen mucho",
    ],
)
def test_parse_rejects_empty_negated_partial_or_unknown_commands(
    router: IntentRouter,
    spoken: str,
) -> None:
    assert router.parse(spoken) is None


@pytest.mark.parametrize(
    ("spoken", "steps"),
    [
        ("sube volumen 0", 1),
        ("sube volumen 1", 1),
        ("sube volumen 10", 10),
        ("sube volumen 999", 10),
    ],
)
def test_volume_steps_are_clamped(
    router: IntentRouter,
    spoken: str,
    steps: int,
) -> None:
    request = router.parse(spoken)

    assert request is not None
    assert request.arguments["steps"] == steps


def test_action_matching_requires_the_complete_utterance(router: IntentRouter) -> None:
    assert router.parse("abre navegador; apaga el equipo") is None
