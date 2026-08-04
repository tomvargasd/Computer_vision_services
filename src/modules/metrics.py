from __future__ import annotations

"""Conversión del diámetro en pixeles a unidades reales.

Formula:  d_real = d_px / PIXELS_PER_UNIT
"""

from src.modules.log_config import get_pixels_per_unit, PIXELS_PER_UNIT


def diameter_to_real(d_px: float, pixels_per_unit: float = None) -> float:
    """Convierte un diámetro en pixeles a unidades reales.

    Args:
        d_px: diámetro en pixeles (obtenido de la máscara/contorno).
        pixels_per_unit: factor de calibración. Si es None usa el valor
            por defecto de log_config.

    Returns:
        diámetro real redondeado a 1 decimal.
    """
    if pixels_per_unit is None:
        pixels_per_unit = PIXELS_PER_UNIT
    if pixels_per_unit <= 0 or d_px is None:
        return 0.0
    return round(float(d_px) / float(pixels_per_unit), 1)