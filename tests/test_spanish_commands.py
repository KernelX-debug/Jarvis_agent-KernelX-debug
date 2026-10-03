from types import SimpleNamespace

import pytest

from jarvis_agent.actions.router import IntentRouter
from jarvis_agent.audio.local import LocalSpeechToText, UncertainSpeechError
from jarvis_agent.spanish import language_pack, local_response


@pytest.mark.parametrize("phrase", language_pack()["aliases"])
def test_pack_aliases_resolve_to_registered_intents(phrase):
    router = IntentRouter()
    expected = router.parse(language_pack()["aliases"][phrase])
    actual = router.parse(phrase)
    assert actual is not None and expected is not None
    assert (actual.name, actual.arguments) == (expected.name, expected.arguments)


@pytest.mark.parametrize(
    ("phrase", "action", "arguments"),
    [
        ("Jarvis, ¿puedes abrir la calculadora?", "open_app", {"app": "calculator"}),
        ("¿Podrías abrir el navegador, por favor?", "open_app", {"app": "browser"}),
        ("Por favor, sube el volumen cinco.", "change_volume", {"direction": "up", "steps": 5}),
        (
            "Jarvis, escribe Hola, María, por favor.",
            "type_text",
            {"text": "Hola, María, por favor."},
        ),
        ("Busca en internet C++ por favor", "web_search", {"query": "C++ por favor"}),
        ("abre la web https://example.org/AbC", "open_url", {"url": "https://example.org/AbC"}),
    ],
)
def test_natural_commands_preserve_payload(phrase, action, arguments):
    result = IntentRouter().parse(phrase)
    assert result is not None
    assert result.name == action
    assert result.arguments == arguments


@pytest.mark.parametrize(
    "phrase",
    [
        "Jarvis, no abras la calculadora",
        "Por favor, nunca abras navegador",
        "Podrías no abrir el explorador",
        "abre la calculadora y luego las notas",
        "si es martes abre calculadora",
        "me dijeron que abre la calculadora",
        "abre algo",
        "borra todos mis archivos",
    ],
)
def test_ambiguous_negated_and_compound_orders_never_execute(phrase):
    assert IntentRouter().parse(phrase) is None


def test_basic_answers_are_available_without_model():
    assert "Jarvis" in local_response("¿Quién eres?")
    assert "Hola" in local_response("Jarvis, hola")
    assert local_response("hola y borra todo") is None


def segment(**changes):
    return SimpleNamespace(
        **{
            "text": "Abre la calculadora",
            "avg_logprob": -0.2,
            "no_speech_prob": 0.01,
            "compression_ratio": 1.2,
            **changes,
        }
    )


@pytest.mark.parametrize(
    "bad_segment",
    [
        segment(avg_logprob=-1.8),
        segment(no_speech_prob=0.9),
        segment(compression_ratio=3.0),
    ],
)
def test_uncertain_segment_rejects_entire_utterance(bad_segment):
    stt = LocalSpeechToText()
    stt._model = SimpleNamespace(transcribe=lambda *a, **kw: (iter([segment(), bad_segment]), None))
    with pytest.raises(UncertainSpeechError):
        stt.transcribe(SimpleNamespace(size=16000))


def test_transcription_is_stateless_and_deterministic():
    stt = LocalSpeechToText()

    def transcribe(audio, **options):
        assert options["condition_on_previous_text"] is False
        assert options["temperature"] == 0.0
        return iter([segment()]), None

    stt._model = SimpleNamespace(transcribe=transcribe)
    assert stt.transcribe(SimpleNamespace(size=16000)) == "Abre la calculadora"
