# Contribuir

1. Usa Python 3.11, 3.12 o 3.13.
2. Crea un entorno virtual e instala `pip install -e ".[dev]"`.
3. Ejecuta `pytest` y `ruff check .` antes de proponer cambios.
4. Las nuevas acciones deben declarar su nivel de riesgo, validar todos sus
   argumentos y tener pruebas sin producir efectos reales.
5. No actives `trust_remote_code` ni añadas ejecución de shell libre.

Los adaptadores de voz, LLM y conocimiento deben conservar interfaces pequeñas
y permitir pruebas con implementaciones simuladas.
