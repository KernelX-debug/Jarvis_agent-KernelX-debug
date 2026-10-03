from __future__ import annotations

import os
from pathlib import Path

import pytest

from jarvis_agent.config import Settings, load_dotenv


@pytest.fixture(autouse=True)
def clean_jarvis_environment(monkeypatch) -> None:
    for key in tuple(os.environ):
        if key.startswith("JARVIS_"):
            monkeypatch.delenv(key, raising=False)


def test_default_settings_are_local_and_create_derived_paths() -> None:
    settings = Settings.from_env(None)

    assert settings.language == "es"
    assert settings.stt_device == "cpu"
    assert settings.llm_provider == "none"
    assert settings.microphone is None
    assert settings.dry_run is True
    assert settings.knowledge_db == Path(".jarvis/knowledge.sqlite3")
    assert settings.audit_log == Path(".jarvis/actions.jsonl")


def test_settings_parse_environment_values(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("JARVIS_NAME", "Aurora")
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path / "datos"))
    monkeypatch.setenv("JARVIS_DRY_RUN", "sí")
    monkeypatch.setenv("JARVIS_TTS_ENABLED", "off")
    monkeypatch.setenv("JARVIS_TTS_RATE", "190")
    monkeypatch.setenv("JARVIS_MICROPHONE", "004")
    monkeypatch.setenv("JARVIS_WAKE_WORD", "  Computadora ")
    monkeypatch.setenv("JARVIS_LLM_PROVIDER", "OLLAMA")
    monkeypatch.setenv("JARVIS_OLLAMA_URL", "http://127.0.0.1:11434///")
    monkeypatch.setenv("JARVIS_MAX_NEW_TOKENS", "512")
    monkeypatch.setenv("JARVIS_MAX_FILE_MB", "8")

    settings = Settings.from_env(None)

    assert settings.name == "Aurora"
    assert settings.data_dir == tmp_path / "datos"
    assert settings.dry_run is True
    assert settings.tts_enabled is False
    assert settings.tts_rate == 190
    assert settings.microphone == 4
    assert settings.wake_word == "computadora"
    assert settings.llm_provider == "ollama"
    assert settings.ollama_url == "http://127.0.0.1:11434"
    assert settings.max_new_tokens == 512
    assert settings.max_file_mb == 8


def test_microphone_name_is_preserved(monkeypatch) -> None:
    monkeypatch.setenv("JARVIS_MICROPHONE", "USB Array")

    assert Settings.from_env(None).microphone == "USB Array"


def test_numeric_values_fall_back_or_are_bounded(monkeypatch) -> None:
    monkeypatch.setenv("JARVIS_TTS_RATE", "not-a-number")
    monkeypatch.setenv("JARVIS_MAX_NEW_TOKENS", "-5")
    monkeypatch.setenv("JARVIS_MAX_FILE_MB", "0")

    settings = Settings.from_env(None)

    assert settings.tts_rate == 175
    assert settings.max_new_tokens == 32
    assert settings.max_file_mb == 1


def test_invalid_dry_run_value_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv("JARVIS_DRY_RUN", "tru")

    with pytest.raises(ValueError, match="booleano invalido"):
        Settings.from_env(None)


def test_dotenv_does_not_overwrite_existing_environment(monkeypatch, tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comentario\n"
        "JARVIS_NAME='Nombre del archivo'\n"
        "JARVIS_LANGUAGE=es-PE\n"
        "INVALID-KEY=ignored\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("JARVIS_NAME", "Nombre del entorno")

    load_dotenv(env_file)

    assert os.environ["JARVIS_NAME"] == "Nombre del entorno"
    assert os.environ["JARVIS_LANGUAGE"] == "es-PE"
    assert "INVALID-KEY" not in os.environ


@pytest.mark.parametrize("provider", ["cloud", "openai", "unknown"])
def test_invalid_llm_provider_is_rejected(monkeypatch, provider: str) -> None:
    monkeypatch.setenv("JARVIS_LLM_PROVIDER", provider)

    with pytest.raises(ValueError, match="JARVIS_LLM_PROVIDER"):
        Settings.from_env(None)


@pytest.mark.parametrize("variable", ["JARVIS_OLLAMA_URL", "JARVIS_OPENAI_BASE_URL"])
def test_non_http_model_endpoint_is_rejected(monkeypatch, variable: str) -> None:
    monkeypatch.setenv(variable, "file:///tmp/model")

    with pytest.raises(ValueError, match=variable):
        Settings.from_env(None)


def test_huggingface_provider_requires_model(monkeypatch) -> None:
    monkeypatch.setenv("JARVIS_LLM_PROVIDER", "huggingface")

    with pytest.raises(ValueError, match="JARVIS_HF_MODEL"):
        Settings.from_env(None)


def test_with_overrides_revalidates_settings() -> None:
    settings = Settings()

    updated = settings.with_overrides(dry_run=True, llm_provider="huggingface", hf_model="x/y")

    assert updated.dry_run is True
    assert updated.hf_model == "x/y"
    assert settings.dry_run is True
