"""
vision/acciones.py — Pipeline de Detección de Acciones / Pose
=============================================================

Funciones:
  • deteccion_acciones — detecta personas y dibuja esqueleto de pose (YOLO11-pose)
"""
from __future__ import annotations

import cv2
import os
import time
import threading
import numpy as np
from collections import deque
from typing import Optional, Dict, List, Tuple

from ultralytics import YOLO

from src.utils import get_device

POSE_MODEL  = "yolo11n-pose.pt"   # se descarga automáticamente la 1ª vez
CONF_THRESH = 0.35
IOU_THRESH  = 0.50
JPEG_Q      = 80

MAX_FACE_CAPS = 3         # máx capturas de rostro por persona
MAX_BODY_CAPS = 3         # máx capturas de cuerpo completo por persona
CAP_THROTTLE_S = 3.0      # segundos mínimos entre capturas del mismo tipo

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAPTURES_BASE = os.path.join(_PROJECT_ROOT, "static", "uploads", "captures")

# ── Pares de keypoints COCO-17 para dibujar el esqueleto ─────────────────
# (índice_a, índice_b, color_BGR)
_SKELETON = [
    # Cara — amarillo (accent)
    (0,  1,  (  0, 222, 255)),   # nariz → ojo_izq
    (0,  2,  (  0, 222, 255)),   # nariz → ojo_der
    (1,  3,  ( 20, 200, 230)),   # ojo_izq → oreja_izq
    (2,  4,  ( 20, 200, 230)),   # ojo_der → oreja_der
    # Torso — navy (brand)
    (5,  6,  ( 85,  42,  24)),   # hombro_izq → hombro_der
    (5,  11, ( 70,  55,  50)),   # hombro_izq → cadera_izq
    (6,  12, ( 70,  55,  50)),   # hombro_der → cadera_der
    (11, 12, ( 55,  68,  75)),   # cadera_izq → cadera_der
    # Brazo izquierdo — gris/teal
    (5,  7,  (150, 200, 100)),   # hombro_izq → codo_izq
    (7,  9,  (180, 220,  80)),   # codo_izq → muñeca_izq
    # Brazo derecho — gris/teal
    (6,  8,  (150, 200, 100)),   # hombro_der → codo_der
    (8,  10, (180, 220,  80)),   # codo_der → muñeca_der
    # Pierna izquierda — rojo semántico
    (11, 13, (100, 100, 255)),   # cadera_izq → rodilla_izq
    (13, 15, ( 60,  60, 255)),   # rodilla_izq → tobillo_izq
    # Pierna derecha — rojo semántico
    (12, 14, (100, 150, 255)),   # cadera_der → rodilla_der
    (14, 16, ( 60, 110, 255)),   # rodilla_der → tobillo_der
]

_KP_COLOR   = (255, 255, 255)
_BBOX_COLOR = (  0, 255, 255)

# ── Índices COCO-17 ───────────────────────────────────────────────────────
_NOSE              = 0
_L_EYE, _R_EYE     = 1, 2
_L_EAR, _R_EAR     = 3, 4
_L_SHO, _R_SHO     = 5, 6
_L_ELB, _R_ELB     = 7, 8
_L_WRI, _R_WRI     = 9, 10
_L_HIP, _R_HIP     = 11, 12
_L_KNE, _R_KNE     = 13, 14
_L_ANK, _R_ANK     = 15, 16

# ── Umbrales de alerta (aplicados tras suavizado temporal) ────────────────
_KP_MIN_CONF = 0.40   # confianza mínima de keypoint aceptado
_SMOOTH_N    = 6      # frames de suavizado por persona
_V_THRESH    = 0.50   # violencia
_R_THRESH    = 0.52   # robo / amenaza
_S_THRESH    = 0.45   # actividad sospechosa

# ── Colores de alerta (BGR) ───────────────────────────────────────────────
_CLR_V = ( 50,  40, 235)   # rojo — violencia
_CLR_R = (  0, 110, 255)   # naranja — robo/amenaza
_CLR_S = (  0, 220, 255)   # amarillo — sospechoso


# ─────────────────────────────────────────────────────────────────────────────
# Auxiliares de geometría 2D
# ─────────────────────────────────────────────────────────────────────────────

def _kpt(kps: np.ndarray, conf: Optional[np.ndarray], idx: int) -> Optional[np.ndarray]:
    """Devuelve el keypoint como array float32 si es confiable, None si no."""
    if conf is not None and conf[idx] < _KP_MIN_CONF:
        return None
    x, y = float(kps[idx, 0]), float(kps[idx, 1])
    return np.array([x, y], dtype=np.float32) if x > 1.0 and y > 1.0 else None


def _d(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _line_dev(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Desviación perpendicular de p a la línea a→b."""
    ab = b - a
    n = np.linalg.norm(ab)
    if n < 1e-6:
        return _d(p, a)
    return float(abs(np.cross(ab, p - a) / n))


# ─────────────────────────────────────────────────────────────────────────────
# Análisis de pose — geometría pura COCO-17, sin dependencias externas
# ─────────────────────────────────────────────────────────────────────────────

def analyze_pose(
    kps: np.ndarray,
    kps_conf: Optional[np.ndarray],
    prev_wrists: Optional[dict] = None,
) -> Tuple[dict, dict]:
    """
    Analiza la pose de UNA persona y devuelve:
      raw_scores : {"violencia": 0-1, "robo": 0-1, "sospechoso": 0-1}
      curr_wrists: {"l": ndarray|None, "r": ndarray|None}

    Todos los scores están normalizados por la altura del torso (hombro→cadera),
    lo que los hace robustos frente a cambios de escala y distancia a la cámara.
    """
    kp = lambda idx: _kpt(kps, kps_conf, idx)  # noqa: E731
    l_sho = kp(_L_SHO);  r_sho = kp(_R_SHO)
    l_elb = kp(_L_ELB);  r_elb = kp(_R_ELB)
    l_wri = kp(_L_WRI);  r_wri = kp(_R_WRI)
    l_hip = kp(_L_HIP);  r_hip = kp(_R_HIP)
    l_kne = kp(_L_KNE);  r_kne = kp(_R_KNE)
    l_ank = kp(_L_ANK);  r_ank = kp(_R_ANK)

    sho_mid = (l_sho + r_sho) / 2 if l_sho is not None and r_sho is not None else None
    hip_mid = (l_hip + r_hip) / 2 if l_hip is not None and r_hip is not None else None
    torso_h = _d(sho_mid, hip_mid) if sho_mid is not None and hip_mid is not None else None

    scores = {"violencia": 0.0, "robo": 0.0, "sospechoso": 0.0}
    if torso_h is None or torso_h < 10.0:
        return scores, {"l": l_wri, "r": r_wri}

    arm_ref = torso_h * 1.4   # longitud de referencia de brazo

    # ── VIOLENCIA ──────────────────────────────────────────────────────────
    #   • Puñetazo: muñeca sobre hombro + brazo extendido (+ velocidad alta)
    #   • Patada:   rodilla sobre nivel de cadera
    #   • Ambos brazos alzados simultáneamente (agresión)
    #   • Movimiento de brazo MUY rápido y extendido (amenaza tipo machete)

    def _punch(sho, wri) -> float:
        if sho is None or wri is None:
            return 0.0
        raise_r = (sho[1] - wri[1]) / torso_h   # + = muñeca por encima del hombro
        ext = _d(sho, wri) / arm_ref
        if raise_r > 0.18 and ext > 0.58:
            return float(np.clip(raise_r * 1.8 * ext, 0, 1))
        return 0.0

    # Movimiento de brazo muy rápido y extendido (amenaza tipo machete)
    def _threat_swing(sho, wri, prev_wri) -> float:
        if sho is None or wri is None or prev_wri is None:
            return 0.0
        ext = _d(sho, wri) / arm_ref
        vel = _d(wri, prev_wri) / torso_h
        # Más sensible: ext > 0.55 y vel > 0.32
        if ext > 0.55 and vel > 0.32:
            # Penaliza si la muñeca está muy cerca de la cadera (evita trotar)
            if hip_mid is not None and abs(wri[1] - hip_mid[1]) / torso_h < 0.18:
                return 0.0
            # Penaliza si la muñeca está muy cerca de la cabeza (evita saludar)
            if sho_mid is not None and abs(wri[1] - sho_mid[1]) / torso_h < 0.13:
                return 0.0
            return float(np.clip(vel * ext * 1.2, 0, 1))
        return 0.0

    def _kick(kne) -> float:
        if kne is None or hip_mid is None:
            return 0.0
        raise_r = (hip_mid[1] - kne[1]) / torso_h   # + = rodilla sobre cadera
        return float(np.clip((raise_r - 0.22) * 3.5, 0, 1))

    punch_l = _punch(l_sho, l_wri)
    punch_r = _punch(r_sho, r_wri)
    kick_l  = _kick(l_kne)
    kick_r  = _kick(r_kne)

    both_raised = 0.0
    if (l_sho is not None and r_sho is not None
            and l_wri is not None and r_wri is not None):
        rl = (l_sho[1] - l_wri[1]) / torso_h
        rr = (r_sho[1] - r_wri[1]) / torso_h
        if rl > 0.12 and rr > 0.12:
            both_raised = float(np.clip((rl + rr) * 0.75, 0, 1))

    wri_vel = 0.0
    if prev_wrists is not None:
        vels = []
        if l_wri is not None and prev_wrists.get("l") is not None:
            vels.append(_d(l_wri, prev_wrists["l"]) / torso_h)
        if r_wri is not None and prev_wrists.get("r") is not None:
            vels.append(_d(r_wri, prev_wrists["r"]) / torso_h)
        if vels:
            wri_vel = float(np.clip(max(vels) * 0.30, 0, 1))

    # Amenaza tipo machete: brazo extendido y movimiento muy rápido
    threat_l = 0.0
    threat_r = 0.0
    if prev_wrists is not None:
        threat_l = _threat_swing(l_sho, l_wri, prev_wrists.get("l"))
        threat_r = _threat_swing(r_sho, r_wri, prev_wrists.get("r"))

    scores["violencia"] = float(np.clip(max(
        punch_l * 0.75 + wri_vel * 0.25,
        punch_r * 0.75 + wri_vel * 0.25,
        kick_l,
        kick_r,
        both_raised * 0.55 + wri_vel * 0.45,
        threat_l,
        threat_r,
    ), 0, 1))

    # ── ROBO / AMENAZA CON ARMA ────────────────────────────────────────────
    #   • Brazo apuntando: extendido + horizontal + recto (arma apuntada)
    #   • O brazo extendido y movimiento rápido (amenaza)
    #   • Bonus por inclinación del torso hacia adelante (pose de amenaza)

    def _arm_point(sho, elb, wri, prev_wri) -> float:
        if sho is None or elb is None or wri is None:
            return 0.0
        ext   = _d(sho, wri) / arm_ref
        dx    = abs(wri[0] - sho[0])
        dy    = abs(wri[1] - sho[1])
        horiz = dx / (dx + dy + 1e-6)
        straight = 1.0 - float(np.clip(
            _line_dev(elb, sho, wri) / (torso_h * 0.30 + 1e-6), 0, 1
        ))
        # Si hay movimiento rápido y brazo extendido, cuenta como amenaza
        vel = 0.0
        if prev_wri is not None:
            vel = _d(wri, prev_wri) / torso_h
        if ext > 0.55 and vel > 0.32:
            return float(np.clip(ext * vel * 1.1, 0, 1))
        return float(np.clip(ext * horiz * straight, 0, 1))

    point_l = _arm_point(l_sho, l_elb, l_wri, prev_wrists.get("l") if prev_wrists else None)
    point_r = _arm_point(r_sho, r_elb, r_wri, prev_wrists.get("r") if prev_wrists else None)
    lean_bonus = 0.0
    if sho_mid is not None and hip_mid is not None:
        lean_bonus = float(np.clip(
            abs(sho_mid[0] - hip_mid[0]) / torso_h * 0.35, 0, 0.30
        ))
    scores["robo"] = float(np.clip(max(point_l, point_r) + lean_bonus, 0, 1))


    # ── ACTIVIDAD SOSPECHOSA OPTIMIZADA ───────────────────────────────────
    #   • Agachado: rodillas muy cerca de caderas (knee_drop < 0.65)
    #   • Rastreando: muñeca debajo de cadera + brazo extendido + velocidad
    #   • Husmeando: hombros adelantados respecto a caderas
    #   • Movimientos de brazos inusuales (no caminar/correr)

    def _crouch(kne, ank) -> float:
        if kne is None or ank is None or hip_mid is None:
            return 0.0
        knee_drop = (kne[1] - hip_mid[1]) / torso_h
        if knee_drop < 0.65:
            return float(np.clip((0.65 - knee_drop) / 0.65 * 1.3, 0, 1))
        return 0.0

    def _reach_down(sho, wri, prev_wri) -> float:
        if sho is None or wri is None or hip_mid is None or prev_wri is None:
            return 0.0
        below = (wri[1] - hip_mid[1]) / torso_h
        ext   = _d(sho, wri) / arm_ref
        vel   = _d(wri, prev_wri) / torso_h
        if below > 0.55 and ext > 0.55 and vel > 0.18:
            return float(np.clip(below * ext * vel, 0, 1))
        return 0.0

    def _hunch() -> float:
        if sho_mid is not None and hip_mid is not None:
            fwd = (abs(sho_mid[0] - hip_mid[0]) / torso_h) - 0.10
            return float(np.clip(fwd * 1.7, 0, 1)) if fwd > 0 else 0.0
        return 0.0

    # Movimientos de brazos inusuales (no caminar/correr)
    unusual_arm = 0.0
    if prev_wrists is not None:
        for sho, wri, prev_wri in [
            (l_sho, l_wri, prev_wrists.get("l")),
            (r_sho, r_wri, prev_wrists.get("r")),
        ]:
            if sho is not None and wri is not None and prev_wri is not None:
                ext = _d(sho, wri) / arm_ref
                vel = _d(wri, prev_wri) / torso_h
                # Detecta movimientos rápidos con brazo extendido pero no arriba (no saludar)
                if ext > 0.45 and vel > 0.22 and (sho_mid is None or abs(wri[1] - sho_mid[1]) / torso_h > 0.18):
                    unusual_arm = max(unusual_arm, float(np.clip(ext * vel * 0.7, 0, 1)))

    crouch = max(_crouch(l_kne, l_ank), _crouch(r_kne, r_ank))
    reach  = 0.0
    if prev_wrists is not None:
        reach = max(_reach_down(l_sho, l_wri, prev_wrists.get("l")), _reach_down(r_sho, r_wri, prev_wrists.get("r")))
    hunch  = _hunch()

    scores["sospechoso"] = float(np.clip(
        max(crouch * 0.85, reach * 0.80, hunch * 0.50, unusual_arm * 0.65), 0, 1
    ))

    return scores, {"l": l_wri, "r": r_wri}


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
    ):
        self.source_id   = source_id
        self.source_path = source_path
        self.func_state  = func_state
        self.conf_thresh = conf_thresh
        self.half        = half
        self.model_path  = model_path or POSE_MODEL

        self.model: Optional[YOLO] = None

        self._frame: Optional[np.ndarray] = None
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Stats
        self.current_persons: int = 0
        self.current_alerts: dict = {"violencia": 0, "robo": 0, "sospechoso": 0}
        self.total_alerts:   dict = {"violencia": 0, "robo": 0, "sospechoso": 0}
        self._prev_person_data: list = []
        self._next_tid: int = 1

        # Capturas por persona TID
        self.capture_count: int = 0
        self._cap_face: Dict[int, List[str]] = {}    # {tid: [url, ...]}
        self._cap_body: Dict[int, List[str]] = {}    # {tid: [url, ...]}
        self._last_cap_ts: Dict[int, Dict[str, float]] = {}  # {tid: {"face":ts, "body":ts}}

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

        cap.release()

    # ── Procesado de frame ────────────────────────────────────────────────

    def _try_capture_face(
        self, frame: np.ndarray, tid: int, kps: np.ndarray,
        kps_conf: Optional[np.ndarray], bx1: int, by1: int, bx2: int, by2: int,
    ) -> None:
        """Recorta rostro (nariz+orejas o top 25% del bbox si no hay keypoints)."""
        if len(self._cap_face.get(tid, [])) >= MAX_FACE_CAPS:
            return
        now = time.time()
        ts_d = self._last_cap_ts.setdefault(tid, {})
        if now - ts_d.get("face", 0.0) < CAP_THROTTLE_S:
            return

        h_frame, w_frame = frame.shape[:2]
        body_h = by2 - by1

        # Intentar centrar en keypoints de cara (0–4)
        face_kps = []
        for ki in (0, 1, 2, 3, 4):
            if kps[ki, 0] > 1 and kps[ki, 1] > 1:
                if kps_conf is None or kps_conf[ki] >= _KP_MIN_CONF:
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
            # Fallback: 25% superior del bbox
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
        """Recorta cuerpo completo."""
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

        deteccion_on  = self.func_state.get("deteccion_acciones",  True)
        violencia_on  = self.func_state.get("deteccion_violencia",  True)
        robo_on       = self.func_state.get("deteccion_robo",       True)
        sospechoso_on = self.func_state.get("deteccion_sospechosa", True)

        persons      = 0
        frame_alerts = {"violencia": 0, "robo": 0, "sospechoso": 0}
        new_pdata: list = []

        if r.keypoints is not None and deteccion_on:
            kps_all  = r.keypoints.xy.cpu().numpy()           # (N, 17, 2)
            conf_all = (
                r.keypoints.conf.cpu().numpy()
                if r.keypoints.conf is not None else None
            )
            boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []

            # Centroides actuales (centro del bbox)
            curr_ctr = []
            for i in range(len(kps_all)):
                if len(boxes) > i:
                    bx1, by1, bx2, by2 = boxes[i]
                    curr_ctr.append(np.array(
                        [(bx1 + bx2) / 2, (by1 + by2) / 2], dtype=np.float32
                    ))
                else:
                    curr_ctr.append(np.zeros(2, dtype=np.float32))

            # Matching nearest-centroid con datos del frame anterior
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

                # Estado previo del mismo individuo (por matching de centroide)
                prev_pd   = prev[matched[i]] if matched[i] is not None else None
                prev_wr   = prev_pd["wrists"]       if prev_pd else None
                score_buf = prev_pd["score_buf"]    if prev_pd else {
                    k: deque([0.0] * _SMOOTH_N, maxlen=_SMOOTH_N)
                    for k in ("violencia", "robo", "sospechoso")
                }
                # ID persistente: heredar o asignar nuevo
                if prev_pd is not None:
                    tid = prev_pd["tid"]
                else:
                    tid = self._next_tid
                    self._next_tid += 1
                # Estado de alertas previo (para detectar onset)
                prev_alert = prev_pd["alert_active"] if prev_pd else {
                    k: False for k in ("violencia", "robo", "sospechoso")
                }

                # Scores geométricos + suavizado temporal
                raw, curr_wr = analyze_pose(kps, conf_i, prev_wr)
                for k in ("violencia", "robo", "sospechoso"):
                    score_buf[k].append(raw[k])
                smooth = {k: float(np.mean(score_buf[k])) for k in score_buf}

                new_pdata.append({
                    "centroid":     curr_ctr[i],
                    "wrists":       curr_wr,
                    "score_buf":    score_buf,
                    "tid":          tid,
                    "alert_active": {k: False for k in ("violencia", "robo", "sospechoso")},
                })

                # Alertas activas para esta persona
                # Solo se incrementa total_alerts en el onset (False→True)
                active: list = []
                capture_trigger = False   # dispara capturas si hay alerta de violencia o robo
                if violencia_on  and smooth["violencia"]  >= _V_THRESH:
                    active.append(("VIOLENCIA",    _CLR_V, smooth["violencia"]))
                    new_pdata[-1]["alert_active"]["violencia"] = True
                    if not prev_alert["violencia"]:
                        self.total_alerts["violencia"] += 1
                    capture_trigger = True
                if robo_on       and smooth["robo"]        >= _R_THRESH:
                    active.append(("ROBO/AMENAZA", _CLR_R, smooth["robo"]))
                    new_pdata[-1]["alert_active"]["robo"] = True
                    if not prev_alert["robo"]:
                        self.total_alerts["robo"] += 1
                    capture_trigger = True
                if sospechoso_on and smooth["sospechoso"]  >= _S_THRESH:
                    active.append(("SOSPECHOSO",   _CLR_S, smooth["sospechoso"]))
                    new_pdata[-1]["alert_active"]["sospechoso"] = True
                    if not prev_alert["sospechoso"]:
                        self.total_alerts["sospechoso"] += 1

                # ── Bounding box
                if len(boxes) > i:
                    bx1, by1, bx2, by2 = map(int, boxes[i])
                    box_clr = active[0][1] if active else _BBOX_COLOR
                    cv2.rectangle(annotated, (bx1, by1), (bx2, by2), box_clr, 2)

                    # Capturas de rostro + cuerpo si hay alerta de violencia o robo
                    if capture_trigger:
                        self._try_capture_face(frame, tid, kps, conf_i, bx1, by1, bx2, by2)
                        self._try_capture_body(frame, tid, bx1, by1, bx2, by2)

                    # Etiqueta de ID en esquina superior izquierda del bbox
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

                    # Etiquetas de alerta encima del bbox
                    lbl_y = by1 - 6
                    for lbl_txt, lbl_clr, score in reversed(active):
                        txt = f"{lbl_txt} {score:.0%}"
                        (tw, th), _ = cv2.getTextSize(
                            txt, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1
                        )
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
                        conf_i[ka] < _KP_MIN_CONF or conf_i[kb] < _KP_MIN_CONF
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
                    if conf_i is not None and conf_i[ki] < _KP_MIN_CONF:
                        continue
                    cv2.circle(annotated, (int(xk), int(yk)), 3, _KP_COLOR, -1, cv2.LINE_AA)

        self._prev_person_data = new_pdata
        self.current_persons   = persons
        self.current_alerts    = {
            k: sum(1 for p in new_pdata if p["alert_active"][k])
            for k in ("violencia", "robo", "sospechoso")
        }

        # ── HUD inferior izquierdo
        cv2.putText(
            annotated, f"Personas: {persons}",
            (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA,
        )

        # ── Indicadores de alerta (esquina superior derecha)
        alert_items = []
        if frame_alerts["violencia"]:
            alert_items.append((f"V:{frame_alerts['violencia']}", _CLR_V))
        if frame_alerts["robo"]:
            alert_items.append((f"R:{frame_alerts['robo']}", _CLR_R))
        if frame_alerts["sospechoso"]:
            alert_items.append((f"S:{frame_alerts['sospechoso']}", _CLR_S))
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
    ) -> None:
        self.stop_all()
        with self._lock:
            p = AccionesPipeline(source_id, source_path, func_state.copy(), conf_thresh, half, model_path)
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

    def update_func_state(self, func_state: dict) -> None:
        with self._lock:
            pipelines = list(self.pipelines.values())
        for p in pipelines:
            p.update_func_state(func_state)
