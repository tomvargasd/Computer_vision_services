from __future__ import annotations

"""Función portable de detección de troncos.

Orquesta: segmentación -> conversión a diámetro real -> clasificación.

Uso típico:

    mask, d_real, category = process_log_detection(frame, [x1, y1, x2, y2])
"""

from typing import Optional, Tuple

from src.modules.segmentation import segment_log_mask
from src.modules.metrics import diameter_to_real
from src.modules.classifier import classify_diameter
from src.modules.log_config import get_pixels_per_unit


def process_log_detection(
    frame,
    bbox,
    pixels_per_unit: float = None,
    settings: dict = None,
) -> Tuple[Optional["object"], Optional[float], int]:
    """Procesa un tronco dado su frame y bounding box.

    Args:
        frame: imagen BGR completa.
        bbox: coordenadas [x1, y1, x2, y2] del tronco detectado.
        pixels_per_unit: factor de calibración (opcional).
        settings: dict de settings (BD) para sobrescribir el factor (opcional).

    Returns:
        (mask, d_real, category):
          - mask: máscara binaria del tronco con el tamaño del frame
            (np.ndarray uint8) o None si no se pudo segmentar.
          - d_real: diámetro real estimado redondeado a 1 decimal
            (float) o None si no se pudo segmentar.
          - category: categoría asignada (0..5) o -1 si es inválida/ignorada.
    """
    if pixels_per_unit is None:
        pixels_per_unit = get_pixels_per_unit(settings)

    # Máscara segmentada (solo para visualización; recortada al bounding box).
    result = segment_log_mask(frame, bbox)
    mask = result[0] if result is not None else None

    # Diámetro de clasificación = ancho del bounding box (el ancho del box
    # aproxima el diámetro real del tronco). Robusto y siempre disponible.
    x1, y1, x2, y2 = bbox[:4]
    d_px = float(abs(int(x2) - int(x1)))

    d_real = diameter_to_real(d_px, pixels_per_unit)
    category = classify_diameter(d_real)
    return mask, d_real, category