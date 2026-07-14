"""
vision/acciones.py — Pipeline de Detección de Acciones / Pose
=============================================================

Funciones:
  • deteccion_acciones — detecta personas y dibuja esqueleto de pose (YOLO11-pose)
"""
from __future__ import annotations

import cv2
import os
import json
import uuid
import time
import threading
import numpy as np
from collections import deque
from typing import Optional, Dict, List, Tuple

from ultralytics import YOLO

from src.utils import get_device
from src.config import BASE_DIR

POSE_MODEL  = "yolo11n-pose.pt"
CONF_THRESH = 0.35
IOU_THRESH  = 0.50
JPEG_Q      = 80

MAX_FACE_CAPS = 3
MAX_BODY_CAPS = 3
CAP_THROTTLE_S = 3.0

CAPTURES_BASE = os.path.join(BASE_DIR, "static", "uploads", "captures")

# ── Reglas centralizadas ─────────────────────────────────────────────────────
_RULES = {
    "kp_min_conf": 0.40,
    "smooth_n": 6,
    "violencia": {
        "enabled": True, "threshold": 0.55,
        "punch_raise": 0.22, "punch_raise_max": 0.60, "punch_ext": 0.62, "punch_elbow_angle": 140,
        "kick_raise": 0.25, "kick_ankle_clear": 0.15,
        "both_raised_min": 0.15,
        "vel_factor": 0.25, "leg_vel_penalty": 0.40,
    },
    "robo": {
        "enabled": True, "threshold": 0.55,
        "arm_ext": 0.60, "arm_ext_vel": 0.65, "arm_vel": 0.38,
        "min_horiz": 0.30,
        "lean_max": 0.25, "lean_min_score": 0.30,
    },
    "sospechoso": {
        "enabled": True, "threshold": 0.50,
        "crouch_drop": 0.55,
        "reach_below": 0.60, "reach_ext": 0.58, "reach_vel": 0.30,
        "hunch_min": 0.15,
    },
    "celular": {
        "enabled": True, "threshold": 0.50,
        "wrist_to_ear_max": 0.30,
        "arm_extension_max": 0.55,
        "hands_together_max": 0.30,
        "wrist_vel_max": 0.12,
    },
    "caida": {
        "enabled": True, "threshold": 0.60,
        "fall_hip_vel": 0.30,
        "fall_body_angle": 30,
        "fall_angle_vel": 5.0,
        "fallen_hip_ratio": 0.45,
        "fallen_body_angle": 50,
        "knee_ground_dist": 0.18,
        "recovery_vel": -0.20,
        "standing_hip_ratio": 0.40,
    },
}

# ── Colores de alerta (BGR) ─────────────────────────────────────────────────
_CLR_V  = ( 50,  40, 235)   # rojo — violencia
_CLR_R  = (  0, 110, 255)   # naranja — robo/amenaza
_CLR_S  = (  0, 220, 255)   # amarillo — sospechoso
_CLR_PH = (200,   0, 255)   # magenta — uso de celular
_CLR_FP = (  0, 165, 255)   # naranja claro — caída parcial
_CLR_FC = (128,   0, 128)   # púrpura — caída completa

# ── Pares de keypoints COCO-17 para dibujar el esqueleto ─────────────────
_SKELETON = [
    (0,  1,  (  0, 222, 255)),
    (0,  2,  (  0, 222, 255)),
    (1,  3,  ( 20, 200, 230)),
    (2,  4,  ( 20, 200, 230)),
    (5,  6,  ( 85,  42,  24)),
    (5,  11, ( 70,  55,  50)),
    (6,  12, ( 70,  55,  50)),
    (11, 12, ( 55,  68,  75)),
    (5,  7,  (150, 200, 100)),
    (7,  9,  (180, 220,  80)),
    (6,  8,  (150, 200, 100)),
    (8,  10, (180, 220,  80)),
    (11, 13, (100, 100, 255)),
    (13, 15, ( 60,  60, 255)),
    (12, 14, (100, 150, 255)),
    (14, 16, ( 60, 110, 255)),
]

_KP_COLOR   = (255, 255, 255)
_BBOX_COLOR = (  0, 255, 255)

# ── Índices COCO-17 ───────────────────────────────────────────────────────
_NOSE   = 0
_L_EYE, _R_EYE = 1, 2
_L_EAR, _R_EAR = 3, 4
_L_SHO, _R_SHO = 5, 6
_L_ELB, _R_ELB = 7, 8
_L_WRI, _R_WRI = 9, 10
_L_HIP, _R_HIP = 11, 12
_L_KNE, _R_KNE = 13, 14
_L_ANK, _R_ANK = 15, 16

_K_VIOLENCIA  = "violencia"
_K_ROBO       = "robo"
_K_SOSPECHOSO = "sospechoso"
_K_CELULAR    = "celular"
_K_CAIDA      = "caida"
_ALERT_KEYS   = (_K_VIOLENCIA, _K_ROBO, _K_SOSPECHOSO, _K_CELULAR, _K_CAIDA)

# ── Enseñar (teach-by-example) ─────────────────────────────────────────────
_TEACH_DATA_FILE = "_teach_data.json"
_TEACH_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), _TEACH_DATA_FILE)
_TEACH_SAMPLES: List[dict] = []
_TEACH_SIM_RATIO   = 0.25   # distancia media / bbox_diag para considerar match
_TEACH_MIN_SCORE   = 0.55   # score mínimo cuando hay match (cubre todos los thresholds)
_LOG_WINDOW_SECONDS = 3.0
_LOG_MAXLEN        = 90


def _load_teach_samples() -> None:
    global _TEACH_SAMPLES
    if os.path.exists(_TEACH_DATA_PATH):
        try:
            with open(_TEACH_DATA_PATH) as f:
                _TEACH_SAMPLES = json.load(f).get("samples", [])
        except Exception:
            _TEACH_SAMPLES = []


def _save_teach_sample(sample: dict) -> None:
    global _TEACH_SAMPLES
    _TEACH_SAMPLES.append(sample)
    with open(_TEACH_DATA_PATH, "w") as f:
        json.dump({"samples": _TEACH_SAMPLES}, f, indent=2, default=str)


def _teach_kp_dist(kps_a: np.ndarray, kps_b: np.ndarray) -> float:
    dists = []
    for i in range(17):
        if (kps_a[i, 0] > 1 and kps_a[i, 1] > 1 and
            kps_b[i, 0] > 1 and kps_b[i, 1] > 1):
            dists.append(float(np.linalg.norm(kps_a[i] - kps_b[i])))
    return float(np.mean(dists)) if dists else -1.0


def _apply_teach_bonus(scores: dict, kps: np.ndarray, bbox_diag: float) -> dict:
    """Reemplaza el score por un valor alto si la pose coincide con una muestra enseñada."""
    if not _TEACH_SAMPLES or bbox_diag < 1:
        return scores
    result = dict(scores)
    thresh_px = bbox_diag * _TEACH_SIM_RATIO
    for sample in _TEACH_SAMPLES:
        action = sample.get("action")
        if action not in result:
            continue
        best_d = float("inf")
        for entry in sample.get("log", []):
            stored = np.array(entry.get("kps"))
            if stored.shape != (17, 2):
                continue
            d = _teach_kp_dist(kps, stored)
            if 0 <= d < best_d:
                best_d = d
        if best_d < float("inf") and best_d < thresh_px:
            confidence = 1.0 - best_d / thresh_px  # 0.0 en el borde, 1.0 en match exacto
            teach_score = _TEACH_MIN_SCORE + 0.40 * confidence  # 0.55 → 0.95
            result[action] = max(result[action], min(1.0, teach_score))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Auxiliares de geometría 2D
# ─────────────────────────────────────────────────────────────────────────────

def _kpt(kps: np.ndarray, conf: Optional[np.ndarray], idx: int) -> Optional[np.ndarray]:
    if conf is not None and conf[idx] < _RULES["kp_min_conf"]:
        return None
    x, y = float(kps[idx, 0]), float(kps[idx, 1])
    return np.array([x, y], dtype=np.float32) if x > 1.0 and y > 1.0 else None


def _d(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _line_dev(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    n = np.linalg.norm(ab)
    if n < 1e-6:
        return _d(p, a)
    return float(abs(np.cross(ab, p - a) / n))


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    na = np.linalg.norm(ba)
    nc = np.linalg.norm(bc)
    if na < 1e-6 or nc < 1e-6:
        return 0.0
    cos_a = float(np.dot(ba, bc) / (na * nc))
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))


def _body_angle(sho_mid: np.ndarray, hip_mid: np.ndarray) -> float:
    dy = hip_mid[1] - sho_mid[1]
    dx = hip_mid[0] - sho_mid[0]
    if abs(dy) < 1e-6:
        return 90.0
    return float(np.degrees(np.arctan2(abs(dx), abs(dy))))


# ─────────────────────────────────────────────────────────────────────────────
# Detectores individuales
# ─────────────────────────────────────────────────────────────────────────────

def _detect_violence(
    kp, torso_h, arm_ref, sho_mid, hip_mid,
    l_sho, r_sho, l_elb, r_elb, l_wri, r_wri, l_kne, r_kne,
    l_ank, r_ank,
    prev_wrists: Optional[dict],
) -> float:
    rules = _RULES[_K_VIOLENCIA]

    def _punch(sho, elb, wri) -> float:
        if sho is None or elb is None or wri is None:
            return 0.0
        elbow_angle = _angle(sho, elb, wri)
        if elbow_angle < rules["punch_elbow_angle"]:
            return 0.0
        raise_r = (sho[1] - wri[1]) / torso_h
        ext = _d(sho, wri) / arm_ref
        if rules["punch_raise"] < raise_r < rules["punch_raise_max"] and ext > rules["punch_ext"]:
            return float(np.clip(raise_r * 1.8 * ext, 0, 1))
        return 0.0

    def _threat_swing(sho, wri, prev_wri) -> float:
        if sho is None or wri is None or prev_wri is None:
            return 0.0
        ext = _d(sho, wri) / arm_ref
        vel = _d(wri, prev_wri) / torso_h
        if ext > 0.55 and vel > 0.32:
            if hip_mid is not None and abs(wri[1] - hip_mid[1]) / torso_h < 0.18:
                return 0.0
            if sho_mid is not None and abs(wri[1] - sho_mid[1]) / torso_h < 0.13:
                return 0.0
            # Reject vertical raises (e.g. celebration, waving)
            if abs(wri[0] - sho[0]) / torso_h < 0.20:
                return 0.0
            return float(np.clip(vel * ext * 1.2, 0, 1))
        return 0.0

    def _kick(kne, ank) -> float:
        if kne is None or hip_mid is None:
            return 0.0
        raise_r = (hip_mid[1] - kne[1]) / torso_h
        if raise_r < rules["kick_raise"]:
            return 0.0
        if ank is not None and ank[1] > kne[1] + torso_h * rules["kick_ankle_clear"]:
            return 0.0
        return float(np.clip((raise_r - rules["kick_raise"]) * 3.5, 0, 1))

    punch_l = _punch(l_sho, l_elb, l_wri)
    punch_r = _punch(r_sho, r_elb, r_wri)
    kick_l = _kick(l_kne, l_ank)
    kick_r = _kick(r_kne, r_ank)

    both_raised = 0.0
    if (l_sho is not None and r_sho is not None
            and l_wri is not None and r_wri is not None):
        rl = (l_sho[1] - l_wri[1]) / torso_h
        rr = (r_sho[1] - r_wri[1]) / torso_h
        if rl > rules["both_raised_min"] and rr > rules["both_raised_min"]:
            both_raised = float(np.clip((rl + rr) * 0.75, 0, 1))

    wri_vel = 0.0
    if prev_wrists is not None:
        vels = []
        if l_wri is not None and prev_wrists.get("l") is not None:
            vels.append(_d(l_wri, prev_wrists["l"]) / torso_h)
        if r_wri is not None and prev_wrists.get("r") is not None:
            vels.append(_d(r_wri, prev_wrists["r"]) / torso_h)
        if vels:
            wri_vel = float(np.clip(max(vels) * rules["vel_factor"], 0, 1))

    threat_l = _threat_swing(l_sho, l_wri, prev_wrists.get("l") if prev_wrists else None)
    threat_r = _threat_swing(r_sho, r_wri, prev_wrists.get("r") if prev_wrists else None)

    return float(np.clip(max(
        punch_l * 0.80 + wri_vel * 0.20,
        punch_r * 0.80 + wri_vel * 0.20,
        kick_l, kick_r,
        both_raised * 0.30,
        threat_l, threat_r,
    ), 0, 1))


def _detect_robbery(
    kp, torso_h, arm_ref, sho_mid, hip_mid,
    l_sho, r_sho, l_elb, r_elb, l_wri, r_wri,
    prev_wrists: Optional[dict],
) -> float:
    rules = _RULES[_K_ROBO]

    def _arm_point(sho, elb, wri, prev_wri) -> float:
        if sho is None or elb is None or wri is None:
            return 0.0
        if hip_mid is not None and wri[1] > hip_mid[1] + torso_h * 0.15:
            return 0.0
        ext = _d(sho, wri) / arm_ref
        dx = abs(wri[0] - sho[0])
        dy = abs(wri[1] - sho[1])
        horiz = dx / (dx + dy + 1e-6)
        if horiz < rules["min_horiz"]:
            return 0.0
        straight = 1.0 - float(np.clip(
            _line_dev(elb, sho, wri) / (torso_h * 0.30 + 1e-6), 0, 1
        ))
        vel = 0.0
        if prev_wri is not None:
            vel = _d(wri, prev_wri) / torso_h
        if ext > rules["arm_ext_vel"] and vel > rules["arm_vel"]:
            return float(np.clip(ext * vel * 1.1, 0, 1))
        return float(np.clip(ext * horiz * straight, 0, 1))

    point_l = _arm_point(l_sho, l_elb, l_wri, prev_wrists.get("l") if prev_wrists else None)
    point_r = _arm_point(r_sho, r_elb, r_wri, prev_wrists.get("r") if prev_wrists else None)
    point_score = max(point_l, point_r)
    if point_score > rules["lean_min_score"] and sho_mid is not None and hip_mid is not None:
        lean_bonus = float(np.clip(
            abs(sho_mid[0] - hip_mid[0]) / torso_h * 0.35, 0, rules["lean_max"]
        ))
        point_score = float(np.clip(point_score + lean_bonus, 0, 1))
    return point_score


def _detect_suspicious(
    kp, torso_h, arm_ref, sho_mid, hip_mid,
    l_sho, r_sho, l_wri, r_wri, l_kne, r_kne, l_ank, r_ank,
    prev_wrists: Optional[dict],
) -> float:
    rules = _RULES[_K_SOSPECHOSO]

    def _crouch(kne) -> float:
        if kne is None or hip_mid is None:
            return 0.0
        knee_drop = (kne[1] - hip_mid[1]) / torso_h
        if knee_drop < rules["crouch_drop"]:
            return float(np.clip((rules["crouch_drop"] - knee_drop) / rules["crouch_drop"] * 1.3, 0, 1))
        return 0.0

    def _reach_down(sho, wri, prev_wri) -> float:
        if sho is None or wri is None or hip_mid is None or prev_wri is None:
            return 0.0
        below = (wri[1] - hip_mid[1]) / torso_h
        ext = _d(sho, wri) / arm_ref
        vel = _d(wri, prev_wri) / torso_h
        if below > rules["reach_below"] and ext > rules["reach_ext"] and vel > rules["reach_vel"]:
            return float(np.clip(below * ext * vel, 0, 1))
        return 0.0

    def _hunch() -> float:
        if sho_mid is not None and hip_mid is not None:
            fwd = (abs(sho_mid[0] - hip_mid[0]) / torso_h) - rules["hunch_min"]
            return float(np.clip(fwd * 1.7, 0, 1)) if fwd > 0 else 0.0
        return 0.0

    unusual_arm = 0.0
    if prev_wrists is not None:
        for sho, wri, prev_wri in [
            (l_sho, l_wri, prev_wrists.get("l")),
            (r_sho, r_wri, prev_wrists.get("r")),
        ]:
            if sho is not None and wri is not None and prev_wri is not None:
                ext = _d(sho, wri) / arm_ref
                vel = _d(wri, prev_wri) / torso_h
                if ext > 0.50 and vel > 0.28 and (sho_mid is None or abs(wri[1] - sho_mid[1]) / torso_h > 0.20):
                    unusual_arm = max(unusual_arm, float(np.clip(ext * vel * 0.7, 0, 1)))

    crouch_l = _crouch(l_kne)
    crouch_r = _crouch(r_kne)
    # Ambas rodillas deben estar dobladas para ser agachado real
    crouch = min(crouch_l, crouch_r) if crouch_l > 0 and crouch_r > 0 else 0.0
    reach = 0.0
    if prev_wrists is not None:
        reach = max(_reach_down(l_sho, l_wri, prev_wrists.get("l")),
                    _reach_down(r_sho, r_wri, prev_wrists.get("r")))
    hunch = _hunch()

    return float(np.clip(
        max(crouch * 0.85, reach * 0.80, hunch * 0.50, unusual_arm * 0.45), 0, 1
    ))


def _detect_phone(
    kp, torso_h, arm_ref, sho_mid, hip_mid,
    l_sho, r_sho, l_elb, r_elb, l_wri, r_wri, l_ear, r_ear,
    nose,
    prev_wrists: Optional[dict],
) -> float:
    rules = _RULES[_K_CELULAR]
    score = 0.0

    # Llamando: muñeca cerca de oreja + brazo flexionado + velocidad baja
    for wri, ear, elb, sho, wri_prev in [
        (l_wri, l_ear, l_elb, l_sho, prev_wrists.get("l") if prev_wrists else None),
        (r_wri, r_ear, r_elb, r_sho, prev_wrists.get("r") if prev_wrists else None),
    ]:
        if wri is not None and ear is not None and elb is not None and sho is not None:
            wrist_to_ear = _d(wri, ear) / torso_h
            arm_ext = _d(sho, wri) / arm_ref
            wri_vel = _d(wri, wri_prev) / torso_h if wri_prev is not None else 0

            if wri_vel > rules["wrist_vel_max"]:
                continue

            if not (wri[1] < elb[1] - torso_h * 0.05):
                continue

            if wrist_to_ear < rules["wrist_to_ear_max"] and arm_ext < rules["arm_extension_max"]:
                call_score = float(np.clip(
                    (1.0 - wrist_to_ear / rules["wrist_to_ear_max"]) * 0.6 +
                    (1.0 - arm_ext / rules["arm_extension_max"]) * 0.2 +
                    (1.0 - min(wri_vel, 0.15) / 0.15) * 0.2,
                    0, 1
                ))
                score = max(score, call_score)

    # Texteando: ambas muñecas juntas + cabeza inclinada hacia abajo
    if l_wri is not None and r_wri is not None and nose is not None and sho_mid is not None:
        if nose[1] > sho_mid[1] + torso_h * 0.10:
            hands_dist = _d(l_wri, r_wri) / torso_h
            avg_wrist_y = (l_wri[1] + r_wri[1]) / 2
            if hip_mid is not None:
                wrist_height = (hip_mid[1] - avg_wrist_y) / torso_h
                if hands_dist < rules["hands_together_max"] and 0.10 < wrist_height < 0.60:
                    text_score = float(np.clip(
                        (1.0 - hands_dist / rules["hands_together_max"]) * 0.6 +
                        (1.0 - abs(wrist_height - 0.30) / 0.30) * 0.2 +
                        0.2,
                        0, 1
                    ))
                    score = max(score, text_score)

    return float(np.clip(score, 0, 1))


def _detect_fall(
    kp, torso_h, sho_mid, hip_mid,
    l_kne, r_kne, l_ank, r_ank,
    l_sho, r_sho,
    prev_fall_state: Optional[dict],
) -> Tuple[float, dict]:
    rules = _RULES[_K_CAIDA]

    fs = prev_fall_state or {}
    state = fs.get("state", "standing")
    hip_hist = fs.get("hip_hist", deque(maxlen=5))
    ang_hist = fs.get("ang_hist", deque(maxlen=5))
    subtype = fs.get("subtype", "")
    fall_count = fs.get("fall_count", 0)

    if sho_mid is None or hip_mid is None or torso_h is None or torso_h < 10:
        return 0.0, {
            "state": "standing", "subtype": "",
            "hip_hist": hip_hist, "ang_hist": ang_hist,
            "fall_count": fall_count,
        }

    body_angle = _body_angle(sho_mid, hip_mid)
    hip_y = hip_mid[1]

    hip_hist.append(hip_y)
    ang_hist.append(body_angle)

    # Velocidad de cadera promediada sobre últimos 3 frames
    if len(hip_hist) >= 3:
        hip_vel = (hip_hist[-1] - hip_hist[-3]) / (torso_h * 2.0)
    elif len(hip_hist) >= 2:
        hip_vel = (hip_hist[-1] - hip_hist[-2]) / torso_h
    else:
        hip_vel = 0.0

    # Velocidad del ángulo corporal
    if len(ang_hist) >= 3:
        ang_vel = (ang_hist[-1] - ang_hist[-3]) / 2.0
    elif len(ang_hist) >= 2:
        ang_vel = ang_hist[-1] - ang_hist[-2]
    else:
        ang_vel = 0.0

    sho_hip_vert = abs(sho_mid[1] - hip_mid[1]) / torso_h if torso_h > 0 else 1.0

    knee_ground_l = False
    if l_kne is not None and l_ank is not None:
        knee_ground_l = abs(l_kne[1] - l_ank[1]) / torso_h < rules["knee_ground_dist"]
    knee_ground_r = False
    if r_kne is not None and r_ank is not None:
        knee_ground_r = abs(r_kne[1] - r_ank[1]) / torso_h < rules["knee_ground_dist"]

    new_subtype = ""
    new_state = state
    score = 0.0

    if prev_fall_state is None and sho_hip_vert < rules["fallen_hip_ratio"] and body_angle > rules["fallen_body_angle"]:
        new_state = "fallen"
        new_subtype = "completa"
        score = 0.85
        fall_count += 1

    elif state == "standing":
        if (hip_vel > rules["fall_hip_vel"] and body_angle > rules["fall_body_angle"]
                and ang_vel > rules["fall_angle_vel"]):
            new_state = "falling"
            new_subtype = "parcial"
            score = 0.35
        elif knee_ground_l or knee_ground_r:
            new_state = "fallen"
            new_subtype = "parcial"
            score = 0.60
            fall_count += 1
        else:
            score = 0.0

    elif state == "falling":
        if sho_hip_vert < rules["fallen_hip_ratio"] and body_angle > rules["fallen_body_angle"]:
            new_state = "fallen"
            new_subtype = "completa"
            score = 0.90
            fall_count += 1
        elif abs(hip_vel) < 0.05 and body_angle < 25:
            new_state = "standing"
            new_subtype = ""
            score = 0.0
        else:
            progress = min(1.0, max(0.0, (body_angle - rules["fall_body_angle"]) / 45.0))
            score = 0.35 + progress * 0.40
            new_subtype = "parcial"

    elif state == "fallen":
        if sho_hip_vert < rules["fallen_hip_ratio"] and body_angle > rules["fallen_body_angle"]:
            new_subtype = "completa"
        else:
            new_subtype = "parcial"
        score = 0.85 if new_subtype == "parcial" else 0.95
        if hip_vel < rules["recovery_vel"]:
            new_state = "getting_up"
            new_subtype = "parcial"
            score = 0.50

    elif state == "getting_up":
        score = 0.40
        new_subtype = "parcial"
        if sho_hip_vert > rules["standing_hip_ratio"] and body_angle < 20:
            new_state = "standing"
            new_subtype = ""
            score = 0.0

    fall_state = {
        "state": new_state,
        "subtype": new_subtype,
        "hip_hist": hip_hist,
        "ang_hist": ang_hist,
        "fall_count": fall_count,
    }

    return score, fall_state


# ─────────────────────────────────────────────────────────────────────────────
# Análisis de pose — integrador
# ─────────────────────────────────────────────────────────────────────────────

def analyze_pose(
    kps: np.ndarray,
    kps_conf: Optional[np.ndarray],
    prev_wrists: Optional[dict] = None,
    prev_fall_state: Optional[dict] = None,
) -> Tuple[dict, dict, dict]:
    """
    Analiza la pose de UNA persona y devuelve:
      raw_scores : {"violencia": 0-1, "robo": 0-1, "sospechoso": 0-1, "celular": 0-1, "caida": 0-1}
      curr_wrists: {"l": ndarray|None, "r": ndarray|None}
      fall_state : dict con estado de la máquina de caída
    """
    kp = lambda idx: _kpt(kps, kps_conf, idx)
    l_sho = kp(_L_SHO);  r_sho = kp(_R_SHO)
    l_elb = kp(_L_ELB);  r_elb = kp(_R_ELB)
    l_wri = kp(_L_WRI);  r_wri = kp(_R_WRI)
    l_hip = kp(_L_HIP);  r_hip = kp(_R_HIP)
    l_kne = kp(_L_KNE);  r_kne = kp(_R_KNE)
    l_ank = kp(_L_ANK);  r_ank = kp(_R_ANK)
    l_ear = kp(_L_EAR);  r_ear = kp(_R_EAR)
    nose  = kp(_NOSE)

    sho_mid = (l_sho + r_sho) / 2 if l_sho is not None and r_sho is not None else None
    hip_mid = (l_hip + r_hip) / 2 if l_hip is not None and r_hip is not None else None
    torso_h = _d(sho_mid, hip_mid) if sho_mid is not None and hip_mid is not None else None

    scores = {k: 0.0 for k in _ALERT_KEYS}
    if torso_h is None or torso_h < 10.0:
        return scores, {"l": l_wri, "r": r_wri}, {
            "state": "standing", "subtype": "",
            "hip_hist": deque(maxlen=5), "ang_hist": deque(maxlen=5),
            "fall_count": 0,
        }

    arm_ref = torso_h * 1.4

    scores[_K_VIOLENCIA] = _detect_violence(
        kp, torso_h, arm_ref, sho_mid, hip_mid,
        l_sho, r_sho, l_elb, r_elb, l_wri, r_wri, l_kne, r_kne, l_ank, r_ank,
        prev_wrists,
    )
    scores[_K_ROBO] = _detect_robbery(
        kp, torso_h, arm_ref, sho_mid, hip_mid,
        l_sho, r_sho, l_elb, r_elb, l_wri, r_wri,
        prev_wrists,
    )
    scores[_K_SOSPECHOSO] = _detect_suspicious(
        kp, torso_h, arm_ref, sho_mid, hip_mid,
        l_sho, r_sho, l_wri, r_wri, l_kne, r_kne, l_ank, r_ank,
        prev_wrists,
    )
    scores[_K_CELULAR] = _detect_phone(
        kp, torso_h, arm_ref, sho_mid, hip_mid,
        l_sho, r_sho, l_elb, r_elb, l_wri, r_wri, l_ear, r_ear,
        nose,
        prev_wrists,
    )
    fall_score, fall_state = _detect_fall(
        kp, torso_h, sho_mid, hip_mid,
        l_kne, r_kne, l_ank, r_ank, l_sho, r_sho,
        prev_fall_state,
    )
    scores[_K_CAIDA] = fall_score

    return scores, {"l": l_wri, "r": r_wri}, fall_state


# ─────────────────────────────────────────────────────────────────────────────
class AccionesPipeline:
    """Pipeline de pose/esqueleto para una fuente de video."""

    def __init__(
        self,
        source_id: int,
        source_path: str,
        func_state: dict,
        conf_thresh: float = CONF_THRESH,
        half: bool = False,
        model_path: str = None,
        fps_limit: float = 0.0,
    ):
        self.source_id   = source_id
        self.source_path = source_path
        self.func_state  = func_state
        self.conf_thresh = conf_thresh
        self.half        = half
        self.model_path  = model_path or POSE_MODEL
        self.fps_limit   = fps_limit

        self.model: Optional[YOLO] = None

        self._frame: Optional[np.ndarray] = None
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.current_persons: int = 0
        self.current_alerts: dict = {k: 0 for k in _ALERT_KEYS}
        self.total_alerts:   dict = {k: 0 for k in _ALERT_KEYS}
        self._prev_person_data: list = []
        self._next_tid: int = 1

        self.capture_count: int = 0
        self._cap_face: Dict[int, List[str]] = {}
        self._cap_body: Dict[int, List[str]] = {}
        self._last_cap_ts: Dict[int, Dict[str, float]] = {}

        self._person_log: Dict[int, deque] = {}
        self._ref_cap_taken: set = set()

        self._cap_dir = os.path.join(CAPTURES_BASE, f"acciones_{source_id}")
        os.makedirs(self._cap_dir, exist_ok=True)

    # ── Control ──────────────────────────────────────────────────────────

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"acciones-pipe-{self.source_id}"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._thread = None

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Hilo principal ────────────────────────────────────────────────────

    def _run(self) -> None:
        self.model = YOLO(self.model_path)
        self.model.to(get_device())
        print(f"[AccionesPipeline] Modelo pose cargado: {self.model_path}")

        src = self.source_path
        cap = cv2.VideoCapture(int(src) if src.isdigit() else src)

        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret:
                if isinstance(src, str) and "://" not in src:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            annotated = self._process(frame)
            with self._lock:
                self._frame = annotated
            time.sleep(self.fps_limit)

        cap.release()

    # ── Capturas ─────────────────────────────────────────────────────────

    def _try_capture_face(
        self, frame: np.ndarray, tid: int, kps: np.ndarray,
        kps_conf: Optional[np.ndarray], bx1: int, by1: int, bx2: int, by2: int,
    ) -> None:
        if len(self._cap_face.get(tid, [])) >= MAX_FACE_CAPS:
            return
        now = time.time()
        ts_d = self._last_cap_ts.setdefault(tid, {})
        if now - ts_d.get("face", 0.0) < CAP_THROTTLE_S:
            return

        h_frame, w_frame = frame.shape[:2]
        body_h = by2 - by1

        face_kps = []
        for ki in (0, 1, 2, 3, 4):
            if kps[ki, 0] > 1 and kps[ki, 1] > 1:
                if kps_conf is None or kps_conf[ki] >= _RULES["kp_min_conf"]:
                    face_kps.append(kps[ki])

        if face_kps:
            xs = [p[0] for p in face_kps]
            ys = [p[1] for p in face_kps]
            cx, cy = float(np.mean(xs)), float(np.mean(ys))
            pad = body_h * 0.22
            x1 = max(0, int(cx - pad))
            y1 = max(0, int(cy - pad))
            x2 = min(w_frame, int(cx + pad))
            y2 = min(h_frame, int(cy + pad))
        else:
            face_h = max(int(body_h * 0.25), 1)
            x1, y1 = max(0, bx1), max(0, by1)
            x2, y2 = min(w_frame, bx2), min(h_frame, by1 + face_h)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 18 or crop.shape[1] < 18:
            return

        tid_dir = os.path.join(self._cap_dir, str(tid))
        os.makedirs(tid_dir, exist_ok=True)
        fname = f"face_{int(time.time()*1000)}.jpg"
        cv2.imwrite(os.path.join(tid_dir, fname), crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
        url = f"/static/uploads/captures/acciones_{self.source_id}/{tid}/{fname}"
        self._cap_face.setdefault(tid, []).append(url)
        ts_d["face"] = now
        self.capture_count += 1

    def _try_capture_body(
        self, frame: np.ndarray, tid: int,
        bx1: int, by1: int, bx2: int, by2: int,
    ) -> None:
        if len(self._cap_body.get(tid, [])) >= MAX_BODY_CAPS:
            return
        now = time.time()
        ts_d = self._last_cap_ts.setdefault(tid, {})
        if now - ts_d.get("body", 0.0) < CAP_THROTTLE_S:
            return

        h_frame, w_frame = frame.shape[:2]
        pad_x = int((bx2 - bx1) * 0.06)
        x1 = max(0, bx1 - pad_x)
        y1 = max(0, by1)
        x2 = min(w_frame, bx2 + pad_x)
        y2 = min(h_frame, by2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 40 or crop.shape[1] < 20:
            return

        tid_dir = os.path.join(self._cap_dir, str(tid))
        os.makedirs(tid_dir, exist_ok=True)
        fname = f"body_{int(time.time()*1000)}.jpg"
        cv2.imwrite(os.path.join(tid_dir, fname), crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
        url = f"/static/uploads/captures/acciones_{self.source_id}/{tid}/{fname}"
        self._cap_body.setdefault(tid, []).append(url)
        ts_d["body"] = now
        self.capture_count += 1

    # ── Procesado de frame ────────────────────────────────────────────────

    def _process(self, frame: np.ndarray) -> np.ndarray:
        annotated = frame.copy()
        h, w = frame.shape[:2]

        results = self.model(
            frame,
            conf=self.conf_thresh,
            iou=IOU_THRESH,
            half=self.half,
            verbose=False,
        )
        r = results[0]

        deteccion_on  = self.func_state.get("deteccion_acciones",   True)
        violencia_on  = self.func_state.get("deteccion_violencia",   True)
        robo_on       = self.func_state.get("deteccion_robo",        True)
        sospechoso_on = self.func_state.get("deteccion_sospechosa",  True)
        celular_on    = self.func_state.get("deteccion_celular",     True)
        caida_on      = self.func_state.get("deteccion_caida",       True)

        persons      = 0
        new_pdata: list = []

        if r.keypoints is not None and deteccion_on:
            kps_all  = r.keypoints.xy.cpu().numpy()
            conf_all = (
                r.keypoints.conf.cpu().numpy()
                if r.keypoints.conf is not None else None
            )
            boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []

            curr_ctr = []
            for i in range(len(kps_all)):
                if len(boxes) > i:
                    bx1, by1, bx2, by2 = boxes[i]
                    curr_ctr.append(np.array(
                        [(bx1 + bx2) / 2, (by1 + by2) / 2], dtype=np.float32
                    ))
                else:
                    curr_ctr.append(np.zeros(2, dtype=np.float32))

            prev    = self._prev_person_data
            matched = [None] * len(kps_all)
            used    = set()
            for ci, cc in enumerate(curr_ctr):
                best_pi, best_d = None, 120.0
                for pi, pd in enumerate(prev):
                    if pi in used:
                        continue
                    d = float(np.linalg.norm(cc - pd["centroid"]))
                    if d < best_d:
                        best_d, best_pi = d, pi
                if best_pi is not None:
                    matched[ci] = best_pi
                    used.add(best_pi)

            for i, kps in enumerate(kps_all):
                persons += 1
                conf_i = conf_all[i] if conf_all is not None else None

                prev_pd   = prev[matched[i]] if matched[i] is not None else None
                prev_wr   = prev_pd["wrists"]       if prev_pd else None
                prev_fs   = prev_pd.get("fall_state") if prev_pd else None
                score_buf = prev_pd["score_buf"]    if prev_pd else {
                    k: deque([0.0] * _RULES["smooth_n"], maxlen=_RULES["smooth_n"])
                    for k in _ALERT_KEYS
                }
                if prev_pd is not None:
                    tid = prev_pd["tid"]
                else:
                    tid = self._next_tid
                    self._next_tid += 1
                prev_alert = prev_pd["alert_active"] if prev_pd else {
                    k: False for k in _ALERT_KEYS
                }

                raw, curr_wr, fall_state = analyze_pose(kps, conf_i, prev_wr, prev_fs)

                # ── Teach bonus (antes de suavizar) ──
                bbox_diag = 0.0
                bx1_l = by1_l = bx2_l = by2_l = 0
                if len(boxes) > i:
                    bx1_l, by1_l, bx2_l, by2_l = map(int, boxes[i])
                    bbox_diag = float(np.sqrt((bx2_l - bx1_l)**2 + (by2_l - by1_l)**2))
                if _TEACH_SAMPLES and bbox_diag > 1:
                    raw = _apply_teach_bonus(raw, kps, bbox_diag)

                for k in _ALERT_KEYS:
                    score_buf[k].append(raw[k])
                smooth = {k: float(np.mean(score_buf[k])) for k in score_buf}

                # ── Log rotativo 3s para enseñar ──
                if bbox_diag > 1:
                    if tid not in self._person_log:
                        self._person_log[tid] = deque(maxlen=_LOG_MAXLEN)
                    self._person_log[tid].append({
                        "ts":       time.time(),
                        "kps":      kps.tolist(),
                        "bbox":     [bx1_l, by1_l, bx2_l, by2_l],
                        "centroid": curr_ctr[i].tolist(),
                        "scores":   {k: float(smooth[k]) for k in _ALERT_KEYS},
                    })

                new_pdata.append({
                    "centroid":     curr_ctr[i],
                    "wrists":       curr_wr,
                    "score_buf":    score_buf,
                    "tid":          tid,
                    "fall_state":   fall_state,
                    "alert_active": {k: False for k in _ALERT_KEYS},
                })

                # ── Construir lista de alertas activas para esta persona
                active: list = []
                cap_trigger  = False  # violencia/robo/celular → face + body
                fall_trigger = False  # caída → body

                if violencia_on and smooth[_K_VIOLENCIA] >= _RULES[_K_VIOLENCIA]["threshold"]:
                    active.append(("VIOLENCIA",    _CLR_V, smooth[_K_VIOLENCIA]))
                    new_pdata[-1]["alert_active"][_K_VIOLENCIA] = True
                    if not prev_alert[_K_VIOLENCIA]:
                        self.total_alerts[_K_VIOLENCIA] += 1
                    cap_trigger = True

                if robo_on and smooth[_K_ROBO] >= _RULES[_K_ROBO]["threshold"]:
                    active.append(("ROBO/AMENAZA", _CLR_R, smooth[_K_ROBO]))
                    new_pdata[-1]["alert_active"][_K_ROBO] = True
                    if not prev_alert[_K_ROBO]:
                        self.total_alerts[_K_ROBO] += 1
                    cap_trigger = True

                if sospechoso_on and smooth[_K_SOSPECHOSO] >= _RULES[_K_SOSPECHOSO]["threshold"]:
                    active.append(("SOSPECHOSO",   _CLR_S, smooth[_K_SOSPECHOSO]))
                    new_pdata[-1]["alert_active"][_K_SOSPECHOSO] = True
                    if not prev_alert[_K_SOSPECHOSO]:
                        self.total_alerts[_K_SOSPECHOSO] += 1

                if celular_on and smooth[_K_CELULAR] >= _RULES[_K_CELULAR]["threshold"]:
                    active.append(("USO CELULAR",  _CLR_PH, smooth[_K_CELULAR]))
                    new_pdata[-1]["alert_active"][_K_CELULAR] = True
                    if not prev_alert[_K_CELULAR]:
                        self.total_alerts[_K_CELULAR] += 1
                    cap_trigger = True

                if caida_on and smooth[_K_CAIDA] >= _RULES[_K_CAIDA]["threshold"]:
                    subtype = fall_state.get("subtype", "")
                    if subtype == "completa":
                        lbl, clr = "CAIDA COMPLETA", _CLR_FC
                    elif subtype == "parcial":
                        lbl, clr = "CAIDA PARCIAL", _CLR_FP
                    else:
                        lbl, clr = "CAIDA", _CLR_FP
                    active.append((lbl, clr, smooth[_K_CAIDA]))
                    new_pdata[-1]["alert_active"][_K_CAIDA] = True
                    if not prev_alert[_K_CAIDA]:
                        self.total_alerts[_K_CAIDA] += 1
                    fall_trigger = True

                # ── Bounding box
                if len(boxes) > i:
                    bx1, by1, bx2, by2 = map(int, boxes[i])
                    box_clr = active[0][1] if active else _BBOX_COLOR
                    cv2.rectangle(annotated, (bx1, by1), (bx2, by2), box_clr, 2)

                    if cap_trigger:
                        self._try_capture_face(frame, tid, kps, conf_i, bx1, by1, bx2, by2)
                        self._try_capture_body(frame, tid, bx1, by1, bx2, by2)
                    if fall_trigger:
                        self._try_capture_body(frame, tid, bx1, by1, bx2, by2)
                    if tid not in self._ref_cap_taken:
                        self._ref_cap_taken.add(tid)
                        self._try_capture_body(frame, tid, bx1, by1, bx2, by2)

                    id_txt = f"ID[{tid}]"
                    (itw, ith), _ = cv2.getTextSize(id_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                    cv2.rectangle(
                        annotated,
                        (bx1, by1), (bx1 + itw + 8, by1 + ith + 6),
                        box_clr, -1,
                    )
                    cv2.putText(
                        annotated, id_txt, (bx1 + 4, by1 + ith + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA,
                    )

                    lbl_y = by1 - 6
                    for lbl_txt, lbl_clr, score in reversed(active):
                        txt = f"{lbl_txt} {score:.0%}"
                        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
                        cv2.rectangle(
                            annotated,
                            (bx1, lbl_y - th - 4), (bx1 + tw + 6, lbl_y + 2),
                            lbl_clr, -1,
                        )
                        cv2.putText(
                            annotated, txt, (bx1 + 3, lbl_y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA,
                        )
                        lbl_y -= (th + 7)

                # ── Esqueleto
                for (ka, kb, color) in _SKELETON:
                    xa, ya = kps[ka]
                    xb, yb = kps[kb]
                    if xa < 1 or ya < 1 or xb < 1 or yb < 1:
                        continue
                    if conf_i is not None and (
                        conf_i[ka] < _RULES["kp_min_conf"] or conf_i[kb] < _RULES["kp_min_conf"]
                    ):
                        continue
                    cv2.line(
                        annotated, (int(xa), int(ya)), (int(xb), int(yb)),
                        color, 2, cv2.LINE_AA,
                    )

                # ── Keypoints
                for ki, (xk, yk) in enumerate(kps):
                    if xk < 1 or yk < 1:
                        continue
                    if conf_i is not None and conf_i[ki] < _RULES["kp_min_conf"]:
                        continue
                    cv2.circle(annotated, (int(xk), int(yk)), 3, _KP_COLOR, -1, cv2.LINE_AA)

        self._prev_person_data = new_pdata
        self.current_persons   = persons

        # ── Podar logs de TIDs que ya no están en escena ──
        active_tids = {p["tid"] for p in new_pdata}
        for tid in list(self._person_log):
            if tid not in active_tids:
                del self._person_log[tid]

        self.current_alerts    = {
            k: sum(1 for p in new_pdata if p["alert_active"][k])
            for k in _ALERT_KEYS
        }

        # ── HUD inferior izquierdo
        cv2.putText(
            annotated, f"Personas: {persons}",
            (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA,
        )

        # ── Indicadores de alerta (esquina superior derecha)
        clr_map = {
            _K_VIOLENCIA: _CLR_V,
            _K_ROBO:      _CLR_R,
            _K_SOSPECHOSO: _CLR_S,
            _K_CELULAR:   _CLR_PH,
            _K_CAIDA:     _CLR_FC,
        }
        short_labels = {
            _K_VIOLENCIA:  "V",
            _K_ROBO:       "R",
            _K_SOSPECHOSO: "S",
            _K_CELULAR:    "P",
            _K_CAIDA:      "F",
        }
        alert_items = []
        for k in _ALERT_KEYS:
            cnt = self.current_alerts.get(k, 0)
            if cnt:
                alert_items.append((f"{short_labels[k]}:{cnt}", clr_map[k]))
        ax = w - 12
        for txt, clr in reversed(alert_items):
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
            ax -= (tw + 12)
            cv2.rectangle(annotated, (ax - 4, 8), (ax + tw + 6, 8 + th + 8), clr, -1)
            cv2.putText(
                annotated, txt, (ax, 8 + th + 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA,
            )

        return annotated

    # ── API pública ───────────────────────────────────────────────────────

    def get_frame_jpeg(self) -> Optional[bytes]:
        with self._lock:
            frame = self._frame
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
        return buf.tobytes() if ok else None

    def get_stats(self) -> dict:
        all_tids = set(self._cap_face) | set(self._cap_body)
        captures = {
            str(tid): {
                "face": self._cap_face.get(tid, []),
                "body": self._cap_body.get(tid, []),
            }
            for tid in all_tids
        }
        return {
            "source_id":       self.source_id,
            "current_persons": self.current_persons,
            "alerts":          dict(self.total_alerts),
            "capture_count":   self.capture_count,
            "captures":        captures,
        }

    def get_teach_data(self) -> dict:
        now = time.time()
        result: dict = {}
        for tid, log in self._person_log.items():
            window = [e for e in log if now - e["ts"] <= _LOG_WINDOW_SECONDS]
            if window:
                result[str(tid)] = {
                    "log": window,
                    "captures": {
                        "face": list(self._cap_face.get(tid, [])),
                        "body": list(self._cap_body.get(tid, [])),
                    },
                }
        return result

    def update_func_state(self, func_state: dict) -> None:
        self.func_state.update(func_state)


# ─────────────────────────────────────────────────────────────────────────────
class AccionesManager:
    """Singleton que gestiona pipelines de detección de acciones."""

    _instance: Optional["AccionesManager"] = None
    _class_lock = threading.Lock()

    def __init__(self) -> None:
        self.pipelines: Dict[int, AccionesPipeline] = {}
        self._lock = threading.Lock()

    @classmethod
    def get(cls) -> "AccionesManager":
        with cls._class_lock:
            if cls._instance is None:
                cls._instance = AccionesManager()
        return cls._instance

    def start(
        self,
        source_id: int,
        source_path: str,
        func_state: dict,
        conf_thresh: float = CONF_THRESH,
        half: bool = False,
        model_path: str = None,
        fps_limit: float = 0.0,
    ) -> None:
        self.stop_all()
        with self._lock:
            p = AccionesPipeline(source_id, source_path, func_state.copy(), conf_thresh, half, model_path, fps_limit)
            p.start()
            self.pipelines[source_id] = p

    def stop(self, source_id: int) -> None:
        with self._lock:
            p = self.pipelines.pop(source_id, None)
        if p:
            p.stop()

    def stop_all(self) -> None:
        with self._lock:
            ids = list(self.pipelines.keys())
        for sid in ids:
            self.stop(sid)

    def is_running(self, source_id: int) -> bool:
        with self._lock:
            p = self.pipelines.get(source_id)
        if p is None:
            return False
        if not p.is_alive():
            with self._lock:
                self.pipelines.pop(source_id, None)
            return False
        return True

    def get_frame_jpeg(self, source_id: int) -> Optional[bytes]:
        with self._lock:
            p = self.pipelines.get(source_id)
        return p.get_frame_jpeg() if p else None

    def get_stats(self, source_id: int) -> Optional[dict]:
        with self._lock:
            p = self.pipelines.get(source_id)
        return p.get_stats() if p else None

    def get_teach_data(self, source_id: int) -> Optional[dict]:
        with self._lock:
            p = self.pipelines.get(source_id)
        return p.get_teach_data() if p else None

    def update_func_state(self, func_state: dict) -> None:
        with self._lock:
            pipelines = list(self.pipelines.values())
        for p in pipelines:
            p.update_func_state(func_state)


# ── Cargar muestras de enseñanza al importar ──
_load_teach_samples()
