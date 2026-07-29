from __future__ import annotations

import threading
import time

import torch

from datetime import datetime

from src.database import get_settings, save_module_counters, load_module_counters, reset_module_counters


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


# ── BasePersistPipeline (v3.0) ─────────────────────────────────────────

class BasePersistPipeline:
    """Mixin para pipelines que necesitan persistencia de contadores."""

    def _init_persistence(self, module_id, source_id):
        self._module_id = module_id
        self._source_id = source_id
        self._last_persist = 0.0
        self._persist_interval = 1.0
        self._daily_date = datetime.now().strftime("%Y-%m-%d")

    def _check_daily_reset(self, counters: dict):
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._daily_date:
            save_module_counters(self._module_id, self._source_id, counters)
            reset_module_counters(self._module_id, self._source_id)
            self._daily_date = today
            self._on_daily_reset()
            return True
        return False

    def _on_daily_reset(self):
        pass

    def _persist_counters(self, counters: dict):
        self._check_daily_reset(counters)
        now = time.time()
        if now - self._last_persist < self._persist_interval:
            return
        self._last_persist = now
        save_module_counters(self._module_id, self._source_id, counters)

    def _load_counters(self) -> dict:
        return load_module_counters(self._module_id, self._source_id)
