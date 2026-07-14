from __future__ import annotations

import threading

import torch

from src.database import get_settings


def get_device() -> str:
    """Devuelve 'cuda:0' si hay CUDA disponible, 'cpu' en caso contrario."""
    return "cuda:0" if torch.cuda.is_available() else "cpu"


# ── Multi‑detección global (hasta 4 pipelines simultáneos) ──────────

_multi_lock = threading.Lock()
_multi_count = 0
MAX_MULTI = 4


def multi_acquire() -> bool:
    """Intenta reservar un slot de pipeline.
    Retorna True si:
      - multi‑detección está desactivada (modo normal), o
      - hay menos de 4 pipelines activos (y lo incrementa).
    Retorna False si ya hay 4 pipelines y multi‑detección está activa.
    """
    global _multi_count
    s = get_settings()
    if s.get("multi_detection", "0") != "1":
        return True
    with _multi_lock:
        if _multi_count < MAX_MULTI:
            _multi_count += 1
            return True
        return False


def multi_release() -> None:
    """Libera un slot de pipeline (solo si multi‑detección está activa)."""
    global _multi_count
    s = get_settings()
    if s.get("multi_detection", "0") != "1":
        return
    with _multi_lock:
        if _multi_count > 0:
            _multi_count -= 1


def is_multi_enabled() -> bool:
    s = get_settings()
    return s.get("multi_detection", "0") == "1"
