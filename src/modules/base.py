from __future__ import annotations

import torch


def get_device() -> str:
    """Devuelve 'cuda:0' si hay CUDA disponible, 'cpu' en caso contrario."""
    return "cuda:0" if torch.cuda.is_available() else "cpu"
