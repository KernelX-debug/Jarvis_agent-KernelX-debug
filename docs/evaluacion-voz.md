# Evaluación de órdenes de voz

Los recursos españoles son reglas y respuestas originales bajo MIT. Indexar
documentación no cambia los pesos de Whisper ni enseña nuevas acciones.

## Protocolo reproducible

1. Inicia `jarvis --dry-run voice` con el mismo modelo y micrófono.
2. Pronuncia cada frase tres veces: ambiente tranquilo, distancia habitual y ruido
   moderado. Espera a «Habla ahora».
3. Compara transcripción y acción reconocida con el resultado esperado.
4. Anota modelo, sistema, dispositivo y condiciones. No publiques voces sin permiso.
5. Repite tras los cambios con ejemplos separados de los usados para ajustar reglas.

| Frase | Resultado esperado |
| --- | --- |
| Jarvis, puedes abrir la calculadora | open_app: calculator |
| Por favor, abre el editor de texto | open_app: notepad |
| Abre mis archivos | open_app: file_manager |
| Qué día es hoy | tell_date |
| Sube el volumen cinco | change_volume: up, 5 |
| Escribe Hola, María | type_text, conservando mayúsculas |
| No abras el navegador | ninguna acción |
| Abre la calculadora y borra todo | ninguna acción |
| Abre algo | ninguna acción |
| Si es martes abre la calculadora | ninguna acción |
| Hola | respuesta local |
| Silencio o ruido | ninguna acción |

Mide aciertos de intención, errores de transcripción, rechazos de órdenes válidas
y falsas activaciones. Informa número de muestras y condiciones; no equipares
las pruebas con texto a precisión de voz.

## Recursos para una fase de entrenamiento

- [FLEURS](https://huggingface.co/datasets/google/fleurs): consultar `es_419`.
- [MLS](https://huggingface.co/datasets/facebook/multilingual_librispeech): español
  con transcripción; empezar por una selección pequeña.
- [Mozilla Data Collective](https://datacollective.mozillafoundation.org/):
  distribución oficial de Common Voice.

No son dependencias ni se descargan automáticamente. Revisa licencias, tamaño y
condiciones actuales; guarda la revisión y separa hablantes entre entrenamiento
y evaluación. El entrenamiento y la conversión CTranslate2 son trabajo futuro.
