from __future__ import annotations

"""Configuración de calibración para la detección/clasificación de troncos.

Concentra las constantes de calibración del módulo de troncos
(conversión pixeles -> unidades reales y reglas de categoría) de forma
accesible y sin depender de la aplicación completa.
"""

# Factor de calibración: diámetro_real = diámetro_px / PIXELS_PER_UNIT.
# Se asume que el ancho del bounding box (en px) aproxima el diámetro real,
# por lo que por defecto es 1.0 (1 px = 1 unidad).
PIXELS_PER_UNIT = 1.0

# Limite inferior de diámetro válido; por debajo la categoría es -1 (ignorado).
SPEC_VALID_MIN_DIAMETER = 14.0

# Reglas estrictas de categoría (sobre d_real redondeado a 1 decimal).
# Categoría -> (min_inclusive, max_inclusive o None si no tiene tope).
CATEGORY_RANGES = {
    0: (14.0, 15.9),
    1: (16.0, 22.9),
    2: (23.0, 27.9),
    3: (28.0, 33.9),
    4: (34.0, 37.9),
    5: (38.0, None),
}

# Categorías válidas (0..5) usadas por contadores.
VALID_CATEGORIES = (0, 1, 2, 3, 4, 5)

# Contador de "Excepciones": cualquier diámetro por debajo del mínimo
# (d_real < SPEC_VALID_MIN_DIAMETER, que el clasificador marca como -1).
# Se cuenta aparte, ubicado al final de las categorías, e incluye en el total.
EXCEPTION_CATEGORY = 6

# Todas las categorías con contador (0..5 + excepciones).
ALL_COUNT_CATEGORIES = VALID_CATEGORIES + (EXCEPTION_CATEGORY,)

# Setting opcional de la BD para sobrescribir el factor de calibración.
PIXELS_PER_UNIT_SETTING = "troncos_pixels_per_unit"


def get_pixels_per_unit(settings: dict = None) -> float:
    """Devuelve el factor de calibración activo.

    Prioriza el valor guardado en settings (BD) si existe y es un número
    positivo; en caso contrario usa PIXELS_PER_UNIT.
    """
    if settings:
        raw = settings.get(PIXELS_PER_UNIT_SETTING)
        if raw not in (None, ""):
            try:
                value = float(raw)
                if value > 0:
                    return value
            except (TypeError, ValueError):
                pass
    return PIXELS_PER_UNIT