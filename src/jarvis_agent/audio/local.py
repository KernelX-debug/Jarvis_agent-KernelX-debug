from __future__ import annotations

import os
import re
import time
from collections import deque
from typing import Any


class UncertainSpeechError(RuntimeError):
    """La frase completa debe repetirse antes de interpretar acciones."""


class MicrophoneCapture:
    """Captura una frase con umbral ambiental y fin por silencio."""

    def __init__(
        self,
        device: str | int | None = None,
        *,
        sample_rate: int = 16_000,
        wait_timeout: float = 6.0,
        phrase_limit: float = 15.0,
        silence_seconds: float = 0.9,
    ) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.wait_timeout = wait_timeout
        self.phrase_limit = phrase_limit
        self.silence_seconds = silence_seconds

    def record_phrase(self) -> Any:
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError('Instala la voz local: pip install -e ".[voice]"') from exc

        block_seconds = 0.1
        block_size = int(self.sample_rate * block_seconds)
        ambient_levels: list[float] = []
        pre_roll: deque[Any] = deque(maxlen=3)
        recorded: list[Any] = []
        started = False
        started_at = 0.0
        last_voice = 0.0
        wait_started = time.monotonic()

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=block_size,
                device=self.device,
            ) as stream:
                print("[Calibrando ruido: guarda silencio un instante...]")
                while True:
                    block, _overflowed = stream.read(block_size)
                    mono = np.asarray(block[:, 0], dtype=np.float32).copy()
                    now = time.monotonic()
                    rms = float(np.sqrt(np.mean(np.square(mono))))

                    if not started and len(ambient_levels) < 5:
                        ambient_levels.append(rms)
                        pre_roll.append(mono)
                        if len(ambient_levels) == 5:
                            print("[Habla ahora]")
                        continue
                    ambient = float(np.median(ambient_levels)) if ambient_levels else 0.0
                    threshold = max(0.012, ambient * 3.0)

                    if not started:
                        pre_roll.append(mono)
                        if rms >= threshold:
                            started = True
                            started_at = now
                            last_voice = now
                            recorded.extend(pre_roll)
                        elif now - wait_started >= self.wait_timeout:
                            return np.empty(0, dtype=np.float32)
                        continue

                    recorded.append(mono)
                    if rms >= threshold:
                        last_voice = now
                    if now - last_voice >= self.silence_seconds:
                        break
                    if now - started_at >= self.phrase_limit:
                        break
        except sd.PortAudioError as exc:
            raise RuntimeError(
                "No pude abrir el microfono. Ejecuta 'jarvis devices' y configura "
                "JARVIS_MICROPHONE."
            ) from exc

        return np.concatenate(recorded) if recorded else np.empty(0, dtype=np.float32)

    @staticmethod
    def list_devices() -> list[dict[str, object]]:
        try:
            import sounddevice as sd
        except ImportError:
            return []
        devices: list[dict[str, object]] = []
        try:
            available = sd.query_devices()
        except sd.PortAudioError:
            return []
        for index, item in enumerate(available):
            if int(item["max_input_channels"]) > 0:
                devices.append(
                    {
                        "index": index,
                        "name": str(item["name"]),
                        "channels": int(item["max_input_channels"]),
                        "sample_rate": int(item["default_samplerate"]),
                    }
                )
        return devices


class LocalSpeechToText:
    def __init__(
        self,
        model: str = "small",
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "es",
        min_log_probability: float = -1.0,
    ) -> None:
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.min_log_probability = min_log_probability
        self._model = None

    def transcribe(self, audio: Any) -> str:
        if getattr(audio, "size", 0) == 0:
            return ""
        if self._model is None:
            if os.name == "nt":
                os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError('Instala la voz local: pip install -e ".[voice]"') from exc
            try:
                self._model = WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"No pude cargar el modelo STT '{self.model_name}': {exc}"
                ) from exc
        try:
            segments, _info = self._model.transcribe(
                audio,
                language=self.language,
                vad_filter=True,
                beam_size=5,
                temperature=0.0,
                condition_on_previous_text=False,
            )
            items = list(segments)
            if any(
                segment.avg_logprob < self.min_log_probability
                or segment.no_speech_prob > 0.6
                or segment.compression_ratio > 2.4
                for segment in items
            ):
                raise UncertainSpeechError(
                    "La transcripción es poco fiable. Repite una sola orden con claridad."
                )
            return " ".join(segment.text.strip() for segment in items).strip()
        except UncertainSpeechError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Fallo la transcripcion local: {exc}") from exc


class LocalTextToSpeech:
    def __init__(self, *, enabled: bool = True, rate: int = 175, language: str = "es") -> None:
        self.enabled = enabled
        self.rate = rate
        self.language = language
        self._engine = None

    def speak(self, text: str) -> None:
        if not self.enabled or not text:
            return
        if self._engine is None:
            try:
                import pyttsx3
            except ImportError as exc:
                raise RuntimeError('Instala la voz local: pip install -e ".[voice]"') from exc
            try:
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", self.rate)
                self._select_voice()
            except Exception as exc:
                raise RuntimeError(f"No pude inicializar la voz del sistema: {exc}") from exc
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception as exc:
            raise RuntimeError(f"Fallo la sintesis de voz local: {exc}") from exc

    def _select_voice(self) -> None:
        voices = self._engine.getProperty("voices")
        for voice in voices:
            descriptor = f"{getattr(voice, 'id', '')} {getattr(voice, 'name', '')}".lower()
            languages = " ".join(str(item).lower() for item in getattr(voice, "languages", []))
            if any(marker in f"{descriptor} {languages}" for marker in ("spanish", "es-", "es_")):
                self._engine.setProperty("voice", voice.id)
                break


def extract_wake_command(transcript: str, wake_word: str) -> str | None:
    """Devuelve lo que sigue a la palabra de activacion o None si no aparece."""

    if not wake_word.strip():
        return transcript.strip()
    pattern = re.compile(rf"\b{re.escape(wake_word.strip())}\b[\s,;:.-]*", re.IGNORECASE)
    match = pattern.search(transcript)
    if match is None:
        return None
    return transcript[match.end() :].strip()
