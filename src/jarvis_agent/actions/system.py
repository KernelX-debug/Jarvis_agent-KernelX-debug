from __future__ import annotations

import ctypes
import platform
import shutil

# Solo se usan ejecutables resueltos y argumentos de listas cerradas.
import subprocess  # nosec B404
import time
import webbrowser
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import quote_plus, urlparse


@dataclass(frozen=True, slots=True)
class HandlerOutput:
    message: str
    should_stop: bool = False


class SystemActions:
    """Implementaciones acotadas; nunca recibe ni ejecuta comandos de shell."""

    def __init__(self, language: str = "es") -> None:
        self.language = language
        self.system = platform.system().lower()

    def open_app(self, arguments: dict[str, Any]) -> HandlerOutput:
        app = str(arguments.get("app", ""))
        if app == "browser":
            webbrowser.open("https://www.google.com", new=2)
            return HandlerOutput("He abierto el navegador.")

        commands = self._application_commands().get(app, [])
        for command in commands:
            executable = command[0]
            resolved_executable = shutil.which(executable)
            if resolved_executable:
                # Lista permitida y shell desactivado.
                subprocess.Popen(  # nosec B603
                    [resolved_executable, *command[1:]],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    shell=False,
                )
                labels = {
                    "calculator": "la calculadora",
                    "notepad": "el editor de texto",
                    "file_manager": "el explorador de archivos",
                }
                return HandlerOutput(f"He abierto {labels.get(app, app)}.")
        raise RuntimeError(f"No encontre una aplicacion compatible para '{app}'.")

    def open_url(self, arguments: dict[str, Any]) -> HandlerOutput:
        url = str(arguments.get("url", "")).strip()
        if not url:
            raise ValueError("La URL esta vacia.")
        if "://" not in url:
            url = f"https://{url}"
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Solo se permiten direcciones http o https validas.")
        webbrowser.open(url, new=2)
        return HandlerOutput(f"He abierto {parsed.netloc}.")

    def web_search(self, arguments: dict[str, Any]) -> HandlerOutput:
        query = str(arguments.get("query", "")).strip()
        if not query or len(query) > 500:
            raise ValueError("La busqueda debe tener entre 1 y 500 caracteres.")
        webbrowser.open(f"https://www.google.com/search?q={quote_plus(query)}", new=2)
        return HandlerOutput(f"Estoy buscando: {query}.")

    def tell_time(self, _arguments: dict[str, Any]) -> HandlerOutput:
        return HandlerOutput(f"Son las {datetime.now():%H:%M}.")

    def tell_date(self, _arguments: dict[str, Any]) -> HandlerOutput:
        today = datetime.now()
        months = (
            "enero",
            "febrero",
            "marzo",
            "abril",
            "mayo",
            "junio",
            "julio",
            "agosto",
            "septiembre",
            "octubre",
            "noviembre",
            "diciembre",
        )
        return HandlerOutput(f"Hoy es {today.day} de {months[today.month - 1]} de {today.year}.")

    def change_volume(self, arguments: dict[str, Any]) -> HandlerOutput:
        direction = str(arguments.get("direction", ""))
        steps = int(arguments.get("steps", 3))
        if direction not in {"up", "down"} or not 1 <= steps <= 10:
            raise ValueError("Cambio de volumen invalido.")
        self._send_volume("up" if direction == "up" else "down", steps)
        verb = "aumentado" if direction == "up" else "reducido"
        return HandlerOutput(f"He {verb} el volumen.")

    def mute_volume(self, _arguments: dict[str, Any]) -> HandlerOutput:
        self._send_volume("mute", 1)
        return HandlerOutput("He alternado el silencio del sistema.")

    def type_text(self, arguments: dict[str, Any]) -> HandlerOutput:
        text = str(arguments.get("text", ""))
        if not text or len(text) > 2_000:
            raise ValueError("El texto debe tener entre 1 y 2000 caracteres.")
        try:
            import pyautogui
            import pyperclip
        except ImportError as exc:
            raise RuntimeError(
                'Instala la opcion de automatizacion: pip install -e ".[automation]"'
            ) from exc

        pyautogui.FAILSAFE = True
        previous_clipboard: str | None = None
        with suppress(Exception):
            previous_clipboard = pyperclip.paste()
        print("Cambia a la ventana de destino. El texto se pegará en 5 segundos.")
        time.sleep(5)
        try:
            pyperclip.copy(text)
            pyautogui.hotkey("ctrl" if self.system != "darwin" else "command", "v")
            time.sleep(0.2)
        finally:
            if previous_clipboard is not None:
                pyperclip.copy(previous_clipboard)
        return HandlerOutput("He escrito el texto en la ventana activa.")

    def show_help(self, _arguments: dict[str, Any]) -> HandlerOutput:
        return HandlerOutput(
            "Puedo abrir el navegador, calculadora, notas o explorador; buscar en internet; "
            "abrir una web; indicar hora y fecha; controlar el volumen; escribir texto con "
            "confirmacion; consultar tus documentos; y cerrar el agente."
        )

    def stop_agent(self, _arguments: dict[str, Any]) -> HandlerOutput:
        return HandlerOutput("Entendido. Desconectando el agente. Hasta pronto.", should_stop=True)

    def _application_commands(self) -> dict[str, list[list[str]]]:
        if self.system == "windows":
            return {
                "calculator": [["calc.exe"]],
                "notepad": [["notepad.exe"]],
                "file_manager": [["explorer.exe"]],
            }
        if self.system == "darwin":
            return {
                "calculator": [["open", "-a", "Calculator"]],
                "notepad": [["open", "-a", "TextEdit"]],
                "file_manager": [["open", "."]],
            }
        return {
            "calculator": [["gnome-calculator"], ["kcalc"]],
            "notepad": [["gedit"], ["kate"], ["mousepad"]],
            "file_manager": [["xdg-open", "."]],
        }

    def _send_volume(self, key: str, steps: int) -> None:
        if self.system == "windows":
            keycodes = {"mute": 0xAD, "down": 0xAE, "up": 0xAF}
            for _ in range(steps):
                ctypes.windll.user32.keybd_event(keycodes[key], 0, 0, 0)  # type: ignore[attr-defined]
                ctypes.windll.user32.keybd_event(keycodes[key], 0, 2, 0)  # type: ignore[attr-defined]
            return
        if self.system == "darwin":
            scripts = {
                "up": "set volume output volume ((output volume of (get volume settings)) + 5)",
                "down": "set volume output volume ((output volume of (get volume settings)) - 5)",
                "mute": "set volume output muted not (output muted of (get volume settings))",
            }
            executable = shutil.which("osascript")
            if not executable:
                raise RuntimeError("No se encontro osascript para controlar el volumen.")
            for _ in range(steps):
                # Script interno de una lista cerrada.
                subprocess.run(  # nosec B603
                    [executable, "-e", scripts[key]], check=True, shell=False
                )
            return
        args = {
            "up": ["pactl", "set-sink-volume", "@DEFAULT_SINK@", "+5%"],
            "down": ["pactl", "set-sink-volume", "@DEFAULT_SINK@", "-5%"],
            "mute": ["pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle"],
        }
        executable = shutil.which("pactl")
        if not executable:
            raise RuntimeError("No se encontro pactl para controlar el volumen.")
        for _ in range(steps):
            # Argumentos internos de una lista cerrada.
            subprocess.run(  # nosec B603
                [executable, *args[key][1:]], check=True, shell=False
            )
