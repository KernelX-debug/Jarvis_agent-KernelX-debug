# Jarvis Agent Local

Asistente de voz local y de código abierto para ejecutar acciones concretas del
ordenador y consultar documentos. Reconoce español con Faster-Whisper y responde
con la voz del sistema. Sus comandos básicos no necesitan un modelo de lenguaje.

**Estado:** MVP en desarrollo. **Licencia:** MIT. **Python:** 3.11–3.13 (3.12 recomendado).

## Funciones

| Capacidad | Ejemplo |
| --- | --- |
| Aplicaciones permitidas | «¿Puedes abrir la calculadora?» |
| Editor de texto | «Por favor, abre el editor de texto» |
| Archivos | «Abre mis archivos» |
| Navegador | «Abre el navegador web» |
| Búsqueda web | «Busca en internet recetas peruanas» |
| URL | «Abre la web example.org» |
| Hora y fecha | «¿Qué día es hoy?», «Me dices la hora» |
| Volumen | «Sube el volumen cinco», «Baja un poco el volumen» |
| Alternar silencio | «Alterna el silencio» |
| Dictado | «Escribe Hola, María»; requiere confirmación por teclado |
| Ayuda e identidad | «Ayuda», «Hola», «¿Quién eres?» |
| Salir | «Cierra Jarvis»; cierra el asistente, no el ordenador |
| Documentos | Recuperación de fragmentos con sus fuentes |

Incluye 36 variantes adicionales de comandos y 12 respuestas locales en español.
Admite prefijos «Jarvis», «por favor», «puedes» y «podrías», puntuación habitual
del reconocimiento y números del uno al diez para volumen. Conserva mayúsculas
en búsquedas, URLs y dictado. Da **una acción a la vez**.

No se admiten acciones arbitrarias, shell, borrados, instalaciones ni apagado del
equipo. Silenciar alterna el estado: repetirlo puede volver a activar el audio.

## Requisitos y compatibilidad

- Micrófono y salida de audio para voz.
- Internet durante la instalación y la primera descarga del modelo Whisper.
- CPU para inferencia; no se exige GPU. Modelo predeterminado: `small`, CPU `int8`.

Windows se ha comprobado manualmente. El núcleo se prueba en CI en Windows y
Linux. Hay adaptadores para macOS, pero su audio y automatización necesitan
validación manual. En Linux pueden hacer falta PortAudio, eSpeak y `pactl`;
el dictado depende del escritorio y puede estar limitado bajo Wayland. En macOS
son necesarios los permisos de micrófono y accesibilidad correspondientes.

## Instalación

Descarga o clona el repositorio y abre una terminal en su carpeta.

### Windows / PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[voice]"
.\.venv\Scripts\jarvis.exe doctor
.\.venv\Scripts\jarvis.exe chat
```

No requiere activar el entorno ni cambiar la política de ejecución de PowerShell.
Si prefieres activarlo:

```powershell
.\.venv\Scripts\Activate.ps1
jarvis voice
```

### Linux / macOS

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[voice]'
.venv/bin/jarvis doctor
.venv/bin/jarvis voice
```

### Entorno reproducible con uv

Se incluyen `uv.lock` y `.python-version`:

```powershell
uv sync --locked --extra voice --extra dev
uv run --no-sync jarvis doctor
uv run --no-sync jarvis voice
```

El paquete no incluye modelos, credenciales, grabaciones ni rutas personales.
Cada usuario descarga su modelo y selecciona su dispositivo local.

## Primer uso

Jarvis arranca en **simulación**: reconoce y muestra acciones sin ejecutarlas.

```powershell
jarvis once "¿Puedes abrir la calculadora?"
jarvis chat
jarvis voice
```

Para acciones reales coloca `--live` antes del subcomando:

```powershell
jarvis --live chat
jarvis --live voice
```

En voz pulsa Enter, guarda silencio durante la calibración, espera **Habla ahora**
y da una orden. Usa `q` y Enter o `Ctrl+C` para salir.

```powershell
jarvis devices
jarvis voice --continuous --wake-word jarvis
```

El modo continuo transcribe frases y busca la palabra de activación; no usa un
detector dedicado de wake word. La escucha se suspende durante la respuesta
hablada. Transcripciones con indicadores débiles se rechazan completas y se pide
repetir. Los umbrales son heurísticos, no un porcentaje de precisión ni una
garantía de que todas las frases aceptadas sean correctas.

Para dictado instala el extra de automatización. Tras confirmar tendrás cinco
segundos para cambiar a la ventana de destino. No hay selección automática de
ventana: comprueba el foco antes del pegado.

```powershell
python -m pip install -e ".[automation]"
jarvis --live voice
```

## Configuración

Copia `.env.example` a `.env` sin sobrescribir tu configuración existente.
Las variables del entorno tienen prioridad. Se busca `.env` en el directorio
actual; también puedes indicar `--env-file RUTA`.

| Variable | Predeterminado | Uso |
| --- | --- | --- |
| `JARVIS_DRY_RUN` | `true` | Simulación; `--live` la desactiva |
| `JARVIS_DATA_DIR` | `.jarvis` | Índice y auditoría, relativos al directorio actual |
| `JARVIS_STT_MODEL` | `small` | Tamaño Whisper o carpeta CTranslate2 local |
| `JARVIS_STT_DEVICE` | `cpu` | CPU o CUDA configurada por el usuario |
| `JARVIS_STT_COMPUTE_TYPE` | `int8` | Tipo de cálculo compatible con el dispositivo |
| `JARVIS_MICROPHONE` | vacío | Índice o nombre obtenido con `jarvis devices` |
| `JARVIS_LANGUAGE` | `es` | Idioma de transcripción |
| `JARVIS_TTS_ENABLED` | `true` | Respuestas habladas |
| `JARVIS_LLM_PROVIDER` | `none` | Proveedor para preguntas generales |
| `JARVIS_MAX_FILE_MB` | `20` | Límite por archivo de conocimiento |

Para un equipo lento se puede probar `JARVIS_STT_MODEL=base`; evalúa la precisión
con tus propias órdenes antes de habilitar acciones reales.

## Conocimiento y conversación

El catálogo `src/jarvis_agent/resources/spanish.json` está empaquetado y funciona
sin configuración. Sus expresiones son originales y se distribuyen bajo MIT.
Puedes ampliar alias apuntando a órdenes existentes; nuevas acciones requieren
código y pruebas. Los documentos nunca registran acciones automáticamente.

```powershell
jarvis ingest ./documentos
jarvis ingest ./manual.md https://example.org/documentacion
jarvis ask "¿Cómo se configura el proyecto?"
```

Admite TXT, Markdown, RST, JSON, JSONL, CSV, HTML y algunos formatos de código.
SQLite FTS5 realiza búsqueda léxica. Sin LLM muestra fragmentos; con LLM genera
una respuesta usando el contexto. `ask` nunca ejecuta acciones.

```powershell
python -m pip install -e ".[pdf]"
jarvis ingest ./manual.pdf
python -m pip install -e ".[hub]"
jarvis ingest-hf organizacion/repositorio --repo-type dataset --revision COMMIT
```

El conector HF importa documentos, no pesos ni audio de entrenamiento. El límite
de ingesta se aplica al procesar archivos, no al tamaño total de descarga del
repositorio. Revisa su tamaño y licencia antes de descargarlo.

Para conversación general instala un servidor local por separado y configura
un proveedor en `.env`. Ejemplo con Ollama:

```dotenv
JARVIS_LLM_PROVIDER=ollama
JARVIS_OLLAMA_URL=http://127.0.0.1:11434
JARVIS_OLLAMA_MODEL=nombre-del-modelo-que-instalaste
```

Para llama.cpp o LM Studio:

```dotenv
JARVIS_LLM_PROVIDER=openai_compatible
JARVIS_OPENAI_BASE_URL=http://127.0.0.1:8080/v1
JARVIS_OPENAI_MODEL=nombre-del-modelo-servido
```

También existe `JARVIS_LLM_PROVIDER=huggingface`, con `JARVIS_HF_MODEL` y el extra
`.[llm-hf]`. Instala PyTorch y requiere más recursos. `trust_remote_code` permanece
desactivado. Un endpoint remoto recibirá preguntas y fragmentos recuperados;
usa un servidor local para mantenerlos en el equipo.

## Solución de problemas

- **WinError 1314:** al cargar Whisper en Windows se desactivan por defecto los
  symlinks de HF. Si otra librería ya importó HF, establece
  `$env:HF_HUB_DISABLE_SYMLINKS="1"` antes de iniciar Jarvis.
- **Aviso de HF_TOKEN:** el modelo público admite descarga sin token. Mantén
  cualquier token en el entorno o `.env`, nunca en Git.
- **No escucha:** revisa permisos, `jarvis devices` y `JARVIS_MICROPHONE`.
  Espera a «Habla ahora» y reduce el ruido ambiental.
- **Extras FALTA:** `pyautogui`, `transformers` y `pypdf` son opcionales según la
  función. `doctor` comprueba instalación, no precisión de voz.
- **No conversa:** configura un LLM para preguntas abiertas. Saludos e identidad
  sí están disponibles sin modelo.

## Arquitectura y privacidad

```text
Micrófono → Whisper → filtro de calidad → router español → acción permitida
                                              ↓
                                  respuesta local o consulta RAG → LLM
```

Las confirmaciones caducan y usan argumentos inmutables. El LLM redacta
respuestas y no ejecuta herramientas. El audio no se guarda por defecto.
La auditoría omite el contenido de textos, búsquedas y URLs. Los documentos
indexados permanecen en SQLite local; la base no está cifrada.

## Desarrollo

```powershell
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check src tests main.py
```

Las pruebas usan dobles para audio y sistema. Cubren español, negaciones,
ambigüedad, dictado, rechazo de audio incierto y confirmaciones. No miden precisión
real de Whisper: usa grabaciones independientes siguiendo la
[guía de evaluación](docs/evaluacion-voz.md).

Referencias técnicas:
[Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) y
[caché de Hugging Face](https://huggingface.co/docs/huggingface_hub/main/package_reference/environment_variables).
No se redistribuyeron corpus externos ni se entrenó un modelo nuevo.

Consulta [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md) y
[LICENSE](LICENSE).
