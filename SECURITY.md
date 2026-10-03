# Seguridad

Jarvis es software alfa y puede controlar aplicaciones del equipo. Ejecútalo
primero con `--dry-run` y revisa cada acción sensible antes de confirmarla.

Principios del proyecto:

- no existe una acción de shell arbitrario;
- un LLM nunca ejecuta directamente una orden;
- las acciones se registran explícitamente y validan sus argumentos;
- escribir en la ventana activa requiere confirmación;
- el audio no se guarda por defecto;
- secretos y tokens solo se leen desde variables de entorno o `.env`.

No se incluyen acciones de apagar el equipo, borrar archivos, elevar permisos,
instalar software ni enviar mensajes. Si se agregan en el futuro deberán estar
deshabilitadas por defecto y requerir una política específica.

Para reportar una vulnerabilidad, abre un aviso privado al responsable del
repositorio antes de publicar detalles explotables.
