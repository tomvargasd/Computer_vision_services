from __future__ import annotations

"""Segmentación de la cara transversal del tronco (Opción B - CPU/FPS).

Recibe un frame y la bounding box [x1, y1, x2, y2] del tronco detectado,
extrae la ROI, aplica umbralizado de Otsu y obtiene el contorno más grande
para definir la máscara binaria del tronco y estimar su diámetro en pixeles.
"""

import cv2
import numpy as np
from typing import Optional, Tuple

# Padding relativo alrededor del bbox para capturar el borde del tronco.
ROI_PADDING_RATIO = 0.25


def _clamp_roi(x: int, y: int, w: int, h: int, W: int, H: int) -> Tuple[int, int, int, int]:
    """Recorta el rectángulo para que quede dentro de la imagen."""
    x = max(0, x)
    y = max(0, y)
    w = min(W, x + w) - x
    h = min(H, y + h) - y
    return x, y, w, h


def largest_contour(binary: np.ndarray) -> Optional[np.ndarray]:
    """Devuelve el contorno con mayor área de la imagen binaria (o None)."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def segment_log_roi(gray_roi: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    """Segmenta el tronco dentro de una ROI en escala de grises.

    Prueba las dos polaridades de Otsu (tronco más claro o más oscuro que el
    fondo) y elige el contorno con mayor área que represente una región
    plausible dentro de la ROI.

    Retorna (máscara local binaria uint8, diámetro_px) o None si no se halla
    un contorno lo bastante significativo.
    """
    if gray_roi.size == 0:
        return None

    blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    roi_area = float(gray_roi.shape[0] * gray_roi.shape[1])
    if roi_area <= 0:
        return None

    candidates = []
    for inverted in (False, True):
        mode = cv2.THRESH_BINARY + cv2.THRESH_OTSU
        if inverted:
            mode = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        _, binary = cv2.threshold(blurred, 0, 255, mode)

        # Operaciones morfológicas para cerrar huecos y limpiar ruido.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        contour = largest_contour(binary)
        if contour is None:
            continue
        area = cv2.contourArea(contour)
        ratio = area / roi_area
        if ratio < 0.03 or ratio > 0.95:
            continue
        perimeter = cv2.arcLength(contour, True)
        compactness = (4.0 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0.0
        candidates.append((compactness, area, contour))

    if not candidates:
        return None

    # La cara transversal del tronco es aproximadamente circular: elegimos el
    # contorno más compacto (compactness ~1), desempatando por área.
    compactness, _area, contour = max(candidates, key=lambda c: (c[0], c[1]))

    # Diámetro como el de la circunferencia mínima que encierra el contorno.
    (_cx, _cy), radius = cv2.minEnclosingCircle(contour)
    d_px = 2.0 * radius

    local_mask = np.zeros_like(gray_roi)
    cv2.drawContours(local_mask, [contour], -1, 255, -1)
    return local_mask, d_px


def segment_log_mask(frame: np.ndarray, bbox) -> Optional[Tuple[np.ndarray, float]]:
    """Extrae la máscara del tronco y su diámetro en pixeles.

    Args:
        frame: imagen BGR completa.
        bbox: una secuencia [x1, y1, x2, y2] del tronco detectado.

    Returns:
        (mask_frame, d_px) donde mask_frame es la máscara binaria (uint8 0/255)
        con el mismo tamaño que el frame, y d_px el diámetro en pixeles.
        None si no se pudo segmentar un contorno válido.
    """
    if frame is None or bbox is None:
        return None
    H, W = frame.shape[:2]

    x1, y1, x2, y2 = (int(v) for v in bbox[:4])
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(bw * ROI_PADDING_RATIO)
    pad_y = int(bh * ROI_PADDING_RATIO)

    rx, ry, rw, rh = _clamp_roi(x1 - pad_x, y1 - pad_y, bw + 2 * pad_x, bh + 2 * pad_y, W, H)
    if rw <= 0 or rh <= 0:
        return None

    roi = frame[ry:ry + rh, rx:rx + rw]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    result = segment_log_roi(gray_roi)
    if result is None:
        return None
    local_mask, _d_px = result

    mask_frame = np.zeros((H, W), dtype=np.uint8)
    mask_frame[ry:ry + rh, rx:rx + rw] = local_mask

    # La máscara nunca debe salirse del bounding box del tronco.
    mask_frame[:y1, :] = 0
    mask_frame[y2:, :] = 0
    mask_frame[:, :x1] = 0
    mask_frame[:, x2:] = 0

    # Recalcular el diámetro sobre la máscara ya recortada.
    contour = largest_contour(mask_frame)
    if contour is None:
        return None
    area = cv2.contourArea(contour)
    bbox_area = float(bw * bh)
    if bbox_area <= 0 or area < bbox_area * 0.03:
        return None

    (_cx, _cy), radius = cv2.minEnclosingCircle(contour)
    d_px = min(2.0 * radius, float(bw))
    if d_px <= 0:
        return None
    return mask_frame, d_px