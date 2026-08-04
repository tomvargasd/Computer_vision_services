from __future__ import annotations

"""Clasificación de troncos por diámetro real según reglas de negocio.

Categoría   Rango (d_real)
0           14.0 <= d <= 15.9
1           16.0 <= d <= 22.9
2           23.0 <= d <= 27.9
3           28.0 <= d <= 33.9
4           34.0 <= d <= 37.9
5           d >= 38.0
-1          d < 14.0  (Ignorado / Out of bounds)
"""

from src.modules.log_config import CATEGORY_RANGES, VALID_CATEGORIES


def classify_diameter(d_real: float) -> int:
    """Asigna la categoría (0..5) a un diámetro real, o -1 si es inválido.

    El valor de d_real ya debe venir redondeado a 1 decimal (ver metrics.py).
    """
    if d_real is None:
        return -1
    for category in VALID_CATEGORIES:
        lo, hi = CATEGORY_RANGES[category]
        if hi is None:
            if d_real >= lo:
                return category
        else:
            if lo <= d_real <= hi:
                return category
    return -1