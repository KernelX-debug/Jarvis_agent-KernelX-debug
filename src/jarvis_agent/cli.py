from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from jarvis_agent.actions import ActionRegistry, IntentRouter
from jarvis_agent.assistant import JarvisAssistant
from jarvis_agent.config import Settings
from jarvis_agent.knowledge import KnowledgeBase
from jarvis_agent.knowledge.ingest import ResourceIngestor
from jarvis_agent.models import TurnResult
from jarvis_agent.providers import build_language_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jarvis",
        description="Agente local por voz con acciones permitidas y conocimiento propio.",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Archivo de configuracion (por defecto .env).",
    )
    parser.add_argument("--data-dir", help="Directorio local para indice y auditoria.")
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=None,
        help="Simula las acciones sin modificar el sistema.",
    )
    execution.add_argument(
        "--live",
        dest="dry_run",
        action="store_false",
        help="Permite ejecutar las acciones registradas en el sistema.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("chat", help="Bucle interactivo por texto.")

    once = subparsers.add_parser("once", help="Ejecuta una sola orden de texto.")
    once.add_argument("text", nargs="+", help="Orden o pregunta.")
    once.add_argument(
        "--yes",
        action="store_true",
        help="Confirma una accion sensible en esta invocacion.",
    )

    voice = subparsers.add_parser("voice", help="Modo de voz local, pulsar para hablar.")
    voice.add_argument(
        "--continuous",
        action="store_true",
        help="Escucha continuamente y exige la palabra de activacion.",
    )
    voice.add_argument("--no-tts", action="store_true", help="No reproduce respuestas por voz.")
    voice.add_argument("--wake-word", help="Palabra de activacion para modo continuo.")

    ingest = subparsers.add_parser("ingest", help="Indexa archivos, carpetas o URLs.")
    ingest.add_argument("sources", nargs="+", help="Ruta local o URL http(s).")

    ingest_hf = subparsers.add_parser(
        "ingest-hf", help="Descarga documentos permitidos de Hugging Face y los indexa."
    )
    ingest_hf.add_argument("repo_id", help="Repositorio, por ejemplo organizacion/nombre.")
    ingest_hf.add_argument("--repo-type", choices=("dataset", "model", "space"), default="dataset")
    ingest_hf.add_argument(
        "--revision",
        help="Commit o tag. Para reproducibilidad es preferible un hash de commit.",
    )

    ask = subparsers.add_parser("ask", help="Pregunta usando el indice local y el LLM configurado.")
    ask.add_argument("question", nargs="+", help="Pregunta.")

    doctor = subparsers.add_parser("doctor", help="Revisa entorno, extras y configuracion.")
    doctor.add_argument("--json", action="store_true", help="Salida JSON.")

    subparsers.add_parser("devices", help="Lista microfonos detectados.")
    subparsers.add_parser("actions", help="Lista acciones habilitadas y su riesgo.")
    return parser


def _settings_from_args(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env(args.env_file)
    changes: dict[str, object] = {}
    if args.data_dir:
        changes["data_dir"] = Path(args.data_dir).expanduser()
    if args.dry_run is not None:
        changes["dry_run"] = args.dry_run
    return settings.with_overrides(**changes) if changes else settings


def build_assistant(settings: Settings) -> JarvisAssistant:
    knowledge = KnowledgeBase(settings.knowledge_db)
    actions = ActionRegistry(
        dry_run=settings.dry_run,
        audit_path=settings.audit_log,
    )
    return JarvisAssistant(
        IntentRouter(),
        actions,
        knowledge,
        build_language_model(settings),
    )


def _print_turn(turn: TurnResult) -> None:
    print(f"Jarvis> {turn.text}")
    if turn.sources:
        print("Fuentes:")
        for source in turn.sources:
            print(f"  - {source}")


def _confirm_in_console(assistant: JarvisAssistant, turn: TurnResult) -> TurnResult:
    if not turn.confirmation_id:
        return turn
    answer = input("Confirmar desde el teclado [s/N]: ").strip().lower()
    return assistant.confirm(turn.confirmation_id, accepted=answer in {"s", "si", "sí", "y", "yes"})


def run_chat(settings: Settings) -> int:
    assistant = build_assistant(settings)
    suffix = " (SIMULACION)" if settings.dry_run else ""
    print(f"{settings.name} local listo{suffix}. Escribe 'ayuda' o 'salir'.")
    while True:
        try:
            text = input("Tu> ")
        except EOFError:
            print()
            return 0
        turn = assistant.process(text)
        _print_turn(turn)
        if turn.confirmation_id:
            turn = _confirm_in_console(assistant, turn)
            _print_turn(turn)
        if turn.should_stop:
            return 0


def run_once(settings: Settings, text: str, *, auto_confirm: bool) -> int:
    assistant = build_assistant(settings)
    turn = assistant.process(text)
    _print_turn(turn)
    if turn.confirmation_id and auto_confirm:
        turn = assistant.confirm(turn.confirmation_id, accepted=True)
        _print_turn(turn)
    elif turn.confirmation_id:
        print("No se ejecuto. Repite con --yes o usa el modo interactivo para confirmar.")
        return 3
    return 0 if not turn.action_result or turn.action_result.success else 2


def run_ask(settings: Settings, question: str) -> int:
    turn = build_assistant(settings).ask(question)
    _print_turn(turn)
    return 0


def run_voice(settings: Settings, *, continuous: bool, no_tts: bool, wake_word: str | None) -> int:
    from jarvis_agent.audio import (
        LocalSpeechToText,
        LocalTextToSpeech,
        MicrophoneCapture,
        extract_wake_command,
    )
    from jarvis_agent.audio.local import UncertainSpeechError

    assistant = build_assistant(settings)
    capture = MicrophoneCapture(settings.microphone)
    stt = LocalSpeechToText(
        settings.stt_model,
        device=settings.stt_device,
        compute_type=settings.stt_compute_type,
        language=settings.language,
    )
    tts = LocalTextToSpeech(
        enabled=settings.tts_enabled and not no_tts,
        rate=settings.tts_rate,
        language=settings.language,
    )
    active_word = (wake_word or settings.wake_word).strip().lower()
    intro = (
        f"Modo continuo. Di {active_word} antes de cada orden. Control C para salir."
        if continuous
        else "Pulsa Enter para hablar; escribe q y Enter para salir."
    )
    print(intro)
    _safe_speak(tts, f"{settings.name} local listo.")

    while True:
        if not continuous:
            key = input("[Enter = hablar | q = salir] ").strip().lower()
            if key in {"q", "salir"}:
                return 0
        print("[Escuchando...]")
        audio = capture.record_phrase()
        if getattr(audio, "size", 0) == 0:
            print("[No se detecto voz]")
            continue
        print("[Transcribiendo localmente...]")
        try:
            transcript = stt.transcribe(audio)
        except UncertainSpeechError as exc:
            print(f"Jarvis> {exc}")
            _safe_speak(tts, str(exc))
            continue
        if not transcript:
            print("[No se entendio la frase]")
            continue
        print(f"Tu> {transcript}")

        command = transcript
        if continuous:
            extracted = extract_wake_command(transcript, active_word)
            if extracted is None:
                continue
            if not extracted:
                _safe_speak(tts, "Te escucho.")
                continue
            command = extracted

        turn = assistant.process(command)
        _print_turn(turn)
        _safe_speak(tts, turn.text)
        if turn.confirmation_id:
            # La voz no se considera autenticacion; la confirmacion es deliberadamente manual.
            turn = _confirm_in_console(assistant, turn)
            _print_turn(turn)
            _safe_speak(tts, turn.text)
        if turn.should_stop:
            return 0


def _safe_speak(tts: object, text: str) -> None:
    try:
        tts.speak(text)  # type: ignore[attr-defined]
    except RuntimeError as exc:
        print(f"[TTS no disponible: {exc}]")


def run_ingest(settings: Settings, sources: list[str]) -> int:
    knowledge = KnowledgeBase(settings.knowledge_db)
    ingestor = ResourceIngestor(knowledge, max_file_mb=settings.max_file_mb)
    failed = 0
    for source in sources:
        try:
            report = ingestor.ingest(source)
            print(
                f"{report.source}: {report.files} archivo(s), "
                f"{report.chunks} fragmento(s), {report.skipped} omitido(s)."
            )
        except Exception as exc:
            failed += 1
            print(f"Error al ingerir {source}: {exc}", file=sys.stderr)
    return 1 if failed else 0


def run_ingest_hf(settings: Settings, repo_id: str, *, repo_type: str, revision: str | None) -> int:
    ingestor = ResourceIngestor(
        KnowledgeBase(settings.knowledge_db), max_file_mb=settings.max_file_mb
    )
    report = ingestor.ingest_huggingface(repo_id, repo_type=repo_type, revision=revision)
    print(
        f"{report.source}: {report.files} archivo(s), "
        f"{report.chunks} fragmento(s), {report.skipped} omitido(s)."
    )
    if revision is None:
        print("Aviso: fija --revision a un commit para obtener resultados reproducibles.")
    return 0


def run_doctor(settings: Settings, *, as_json: bool) -> int:
    packages = {
        "faster_whisper": importlib.util.find_spec("faster_whisper") is not None,
        "sounddevice": importlib.util.find_spec("sounddevice") is not None,
        "pyttsx3": importlib.util.find_spec("pyttsx3") is not None,
        "pyautogui": importlib.util.find_spec("pyautogui") is not None,
        "huggingface_hub": importlib.util.find_spec("huggingface_hub") is not None,
        "transformers": importlib.util.find_spec("transformers") is not None,
        "pypdf": importlib.util.find_spec("pypdf") is not None,
    }
    version = sys.version_info
    supported_python = (3, 11) <= version[:2] < (3, 14)
    knowledge = KnowledgeBase(settings.knowledge_db)
    report = {
        "python": platform.python_version(),
        "python_supported": supported_python,
        "platform": platform.platform(),
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "provider": settings.llm_provider,
        "stt_model": settings.stt_model,
        "stt_device": settings.stt_device,
        "dry_run": settings.dry_run,
        "data_dir": str(settings.data_dir.resolve()),
        "knowledge": knowledge.stats(),
        "packages": packages,
        "voice_ready": all(packages[name] for name in ("faster_whisper", "sounddevice", "pyttsx3")),
    }
    if as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        mark = lambda value: "OK" if value else "FALTA"  # noqa: E731
        print(f"Python {report['python']}: {mark(supported_python)} (recomendado 3.12)")
        print(f"Sistema: {report['platform']}")
        print(f"FFmpeg: {mark(bool(report['ffmpeg']))}")
        print(f"Proveedor LLM: {settings.llm_provider}")
        print(f"STT: {settings.stt_model} en {settings.stt_device}")
        print(f"Modo simulacion: {'si' if settings.dry_run else 'no'}")
        print(f"Datos: {report['data_dir']}")
        stats = report["knowledge"]
        print(f"Conocimiento: {stats['sources']} fuentes / {stats['chunks']} fragmentos")
        print("Extras:")
        for package, installed in packages.items():
            print(f"  {mark(installed):5} {package}")
        if not report["voice_ready"]:
            print('Instala voz con: python -m pip install -e ".[voice]"')
    return 0 if supported_python else 1


def run_devices() -> int:
    from jarvis_agent.audio import MicrophoneCapture

    devices = MicrophoneCapture.list_devices()
    if not devices:
        print('No se detectaron entradas o falta el extra ".[voice]".')
        return 1
    for item in devices:
        print(
            f"[{item['index']}] {item['name']} — {item['channels']} canal(es), "
            f"{item['sample_rate']} Hz"
        )
    return 0


def run_actions(settings: Settings) -> int:
    registry = ActionRegistry(dry_run=settings.dry_run)
    for name in registry.action_names:
        print(f"{name:16} {registry.risk_for(name).value}")
    return 0


def _configure_console_encoding() -> None:
    """Evita fallos con acentos/transcripciones cuando Windows usa una pagina ANSI."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    _configure_console_encoding()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = _settings_from_args(args)
        if args.command == "chat":
            return run_chat(settings)
        if args.command == "once":
            return run_once(settings, " ".join(args.text), auto_confirm=args.yes)
        if args.command == "voice":
            return run_voice(
                settings,
                continuous=args.continuous,
                no_tts=args.no_tts,
                wake_word=args.wake_word,
            )
        if args.command == "ingest":
            return run_ingest(settings, args.sources)
        if args.command == "ingest-hf":
            return run_ingest_hf(
                settings,
                args.repo_id,
                repo_type=args.repo_type,
                revision=args.revision,
            )
        if args.command == "ask":
            return run_ask(settings, " ".join(args.question))
        if args.command == "doctor":
            return run_doctor(settings, as_json=args.json)
        if args.command == "devices":
            return run_devices()
        if args.command == "actions":
            return run_actions(settings)
    except KeyboardInterrupt:
        print("\nInterrumpido.")
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
