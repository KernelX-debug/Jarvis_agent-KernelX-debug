from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import urlparse


def load_dotenv(path: Path) -> None:
    """Carga pares simples CLAVE=VALOR sin sobrescribir el entorno existente."""

    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key.replace("_", "").isalnum():
            os.environ.setdefault(key, value)


def _as_bool(value: str | None, default: bool = False, *, strict: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "si", "sí", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    if strict:
        raise ValueError(f"Valor booleano invalido: {value!r}.")
    return default


def _as_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


@dataclass(frozen=True, slots=True)
class Settings:
    name: str = "Jarvis"
    language: str = "es"
    data_dir: Path = Path(".jarvis")
    dry_run: bool = True
    stt_model: str = "small"
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    microphone: str | int | None = None
    tts_enabled: bool = True
    tts_rate: int = 175
    wake_word: str = "jarvis"
    llm_provider: str = "none"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:3b"
    openai_base_url: str = "http://127.0.0.1:8080/v1"
    openai_model: str = "local-model"
    hf_model: str = ""
    max_new_tokens: int = 384
    max_file_mb: int = 20

    @classmethod
    def from_env(cls, env_file: str | Path | None = ".env") -> Settings:
        if env_file:
            load_dotenv(Path(env_file))

        raw_microphone = os.getenv("JARVIS_MICROPHONE", "").strip()
        microphone: str | int | None = raw_microphone or None
        if raw_microphone.isdigit():
            microphone = int(raw_microphone)

        settings = cls(
            name=os.getenv("JARVIS_NAME", "Jarvis").strip() or "Jarvis",
            language=os.getenv("JARVIS_LANGUAGE", "es").strip() or "es",
            data_dir=Path(os.getenv("JARVIS_DATA_DIR", ".jarvis")).expanduser(),
            dry_run=_as_bool(os.getenv("JARVIS_DRY_RUN"), True, strict=True),
            stt_model=os.getenv("JARVIS_STT_MODEL", "small").strip() or "small",
            stt_device=os.getenv("JARVIS_STT_DEVICE", "cpu").strip() or "cpu",
            stt_compute_type=(os.getenv("JARVIS_STT_COMPUTE_TYPE", "int8").strip() or "int8"),
            microphone=microphone,
            tts_enabled=_as_bool(os.getenv("JARVIS_TTS_ENABLED"), True),
            tts_rate=_as_int(os.getenv("JARVIS_TTS_RATE"), 175),
            wake_word=os.getenv("JARVIS_WAKE_WORD", "jarvis").strip().lower() or "jarvis",
            llm_provider=os.getenv("JARVIS_LLM_PROVIDER", "none").strip().lower() or "none",
            ollama_url=(os.getenv("JARVIS_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")),
            ollama_model=os.getenv("JARVIS_OLLAMA_MODEL", "qwen2.5:3b").strip(),
            openai_base_url=(
                os.getenv("JARVIS_OPENAI_BASE_URL", "http://127.0.0.1:8080/v1").rstrip("/")
            ),
            openai_model=os.getenv("JARVIS_OPENAI_MODEL", "local-model").strip(),
            hf_model=os.getenv("JARVIS_HF_MODEL", "").strip(),
            max_new_tokens=max(32, _as_int(os.getenv("JARVIS_MAX_NEW_TOKENS"), 384)),
            max_file_mb=max(1, _as_int(os.getenv("JARVIS_MAX_FILE_MB"), 20)),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.llm_provider not in {"none", "ollama", "openai_compatible", "huggingface"}:
            raise ValueError(
                "JARVIS_LLM_PROVIDER debe ser: none, ollama, openai_compatible o huggingface."
            )
        if self.llm_provider == "huggingface" and not self.hf_model:
            raise ValueError("Configura JARVIS_HF_MODEL para usar el proveedor huggingface.")
        for label, value in (
            ("JARVIS_OLLAMA_URL", self.ollama_url),
            ("JARVIS_OPENAI_BASE_URL", self.openai_base_url),
        ):
            parsed = urlparse(value)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError(f"{label} debe ser una URL http o https valida.")

    def with_overrides(self, **changes: object) -> Settings:
        updated = replace(self, **changes)
        updated.validate()
        return updated

    @property
    def knowledge_db(self) -> Path:
        return self.data_dir / "knowledge.sqlite3"

    @property
    def audit_log(self) -> Path:
        return self.data_dir / "actions.jsonl"
