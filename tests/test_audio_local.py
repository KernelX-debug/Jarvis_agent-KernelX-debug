from __future__ import annotations

import pytest

from jarvis_agent.audio.local import extract_wake_command


@pytest.mark.parametrize(
    ("transcript", "wake_word", "expected"),
    [
        ("Jarvis abre la calculadora", "jarvis", "abre la calculadora"),
        ("oye JARVIS, dime la hora", "jarvis", "dime la hora"),
        ("jarvis: busca café", "jarvis", "busca café"),
        ("jarvis...   silencia el audio", "jarvis", "silencia el audio"),
        ("jarvis", "jarvis", ""),
        ("  comando directo  ", "", "comando directo"),
        ("j.rvis: ayuda", "j.rvis", "ayuda"),
    ],
)
def test_extract_wake_command(
    transcript: str,
    wake_word: str,
    expected: str,
) -> None:
    assert extract_wake_command(transcript, wake_word) == expected


@pytest.mark.parametrize(
    ("transcript", "wake_word"),
    [
        ("abre la calculadora", "jarvis"),
        ("jarvisito abre la calculadora", "jarvis"),
        ("", "jarvis"),
    ],
)
def test_extract_wake_command_returns_none_when_wake_word_is_absent(
    transcript: str,
    wake_word: str,
) -> None:
    assert extract_wake_command(transcript, wake_word) is None
