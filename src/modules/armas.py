"""
vision/armas.py — Pipeline de Detección de Armas
=================================================

Funciones:
  • deteccion_arma   — detecta armas blancas y de fuego
  • captura_rostro   — captura automática de rostro del portador del arma
  • tipo_arma        — clasifica el tipo de arma (SIEMPRE activa, no se puede desactivar)
"""
from __future__ import annotations

import cv2
import os
import time
import threading
import numpy as np
from typing import Optional, Dict, List

from ultralytics import YOLO

from src.utils import get_device
from src.modules.base import multi_acquire, multi_release, is_multi_enabled
from src.config import BASE_DIR

# Palabras clave para identificar clase persona en cualquier modelo
_PERSON_KEYWORDS = {"person", "persona", "people", "human", "pedestrian", "man", "woman", "without_weapon"}
# Palabras clave para EXCLUIR como arma (clases que no son armas ni personas)
_IGNORE_KEYWORDS = {"car", "truck", "bus", "bicycle", "motorcycle", "background", "neutral"}

DEFAULT_MODEL   = "yolo11n.pt"
PERSON_CLS      = 0
# Clases weapon en COCO80 (modelo por defecto sin custom)
COCO_WEAPON_CLS = {43}            # 43 = knife (única arma en COCO)
CONF_THRESH        = 0.35         # umbral para modelo por defecto
CONF_THRESH_CUSTOM = 0.20         # umbral más bajo para modelos custom
IOU_THRESH      = 0.50
JPEG_Q          = 80
MAX_CAPS_PER_ID = 6               # máximo de capturas por weapon tracker ID
CAP_THROTTLE_S  = 3.0             # segundos mínimos entre capturas del mismo ID

# Config de ByteTrack ajustada para armas (relativa al directorio de ejecución)
import os as _os
_TRACKER_CFG = _os.path.join(BASE_DIR, "bytetrack_armas.yaml")

CAPTURES_BASE = os.path.join(BASE_DIR, "static", "uploads", "captures")


def _weapon_type(class_name: str) -> str:
    """Clasifica el nombre de clase en 'arma_blanca', 'arma_fuego' o 'arma'."""
    n = class_name.lower().replace("-", "_").replace(" ", "_")
    if any(k in n for k in (
        "gun", "pistol", "rifle", "firearm", "revolver", "shotgun",
        "handgun", "arma_fuego", "weapon_fire", "glock", "ak", "smg",
        "submachine", "assault", "sniper", "carbine", "cannon",
    )):
        return "arma_fuego"
    if any(k in n for k in (
        "knife", "blade", "cuchillo", "arma_blanca", "dagger",
        "machete", "katana", "navaja", "punal", "sword", "axe",
        "razor", "shank", "shiv",
    )):
        return "arma_blanca"
    # Genérico: cualquier cosa que tenga "weapon", "arm", "gun" parcial
    if any(k in n for k in ("weapon", "arm", "firearm")):
        return "arma_fuego"
    return "arma"


# ─────────────────────────────────────────────────────────────────────────────
class ArmasPipeline:
    """Pipeline de análisis de video para una fuente específica de armas."""

    def __init__(
        self,
        source_id: int,
        source_path: str,
        func_state: dict,
        conf_thresh: float = None,
        half: bool = False,
        model_path: str = None,
        fps_limit: float = 0.0,
    ):
        self.source_id   = source_id
        self.source_path = source_path
        self.func_state  = func_state
        self.model_path  = model_path or DEFAULT_MODEL
        self.half        = half
        is_default = (self.model_path == DEFAULT_MODEL)
        self.conf_thresh = conf_thresh if conf_thresh is not None else (CONF_THRESH if is_default else CONF_THRESH_CUSTOM)
        self.fps_limit   = fps_limit

        self.model: Optional[YOLO]         = None   # cargado en hilo
        self._person_model: Optional[YOLO] = None   # secundario si el principal no tiene clase 0

        self._frame: Optional[np.ndarray] = None
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # ── Stats ────────────────────────────────────────────────────────
        self.weapon_current = 0         # armas visibles en el frame actual
        self.total_weapons = 0           # acumulado histórico (por track ID)
        self.total_blanca  = 0
        self.total_fuego   = 0
        self.capture_count = 0           # total capturas guardadas
        self._captures: Dict[int, List[str]] = {}  # {weapon_tid: ["url1", ...]}
        self._last_cap_ts: Dict[int, float]  = {}  # throttle por weapon_tid
        # Sets para deduplicar por track ID
        self._weapon_ids_seen: set = set()
        self._blanca_ids_seen: set = set()
        self._fuego_ids_seen: set = set()

        # Directorio de capturas para esta fuente
        self._cap_dir = os.path.join(CAPTURES_BASE, str(source_id))
        os.makedirs(self._cap_dir, exist_ok=True)

        # ── Validación 3×5 frames ─────────────────────────────────────
        self._round_buffer: List[bool] = []      # 5 frames del round actual
        self._round_results: List[float] = []    # ratios de rounds completados (máx 3)
        self._alert_active = False
        self._alert_timestamp: Optional[float] = None

    # ── Control ──────────────────────────────────────────────────────────

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"armas-pipe-{self.source_id}",
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

    def _make_error_frame(self, msg: str) -> np.ndarray:
        h, w = 480, 640
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(
            frame, "ERROR DE FUENTE", (int(w * 0.22), h // 2 - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (85, 42, 24), 2, cv2.LINE_AA,
        )
        max_chars = 55
        parts = [msg[j: j + max_chars] for j in range(0, min(len(msg), max_chars * 3), max_chars)]
        for i, part in enumerate(parts):
            cv2.putText(
                frame, part, (20, h // 2 + 10 + i * 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA,
            )
        cv2.putText(
            frame, "Verifica la ruta o permisos de la fuente", (20, h - 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 120, 120), 1, cv2.LINE_AA,
        )
        return frame

    def _run(self) -> None:
        # Cargar modelo principal en el hilo (no bloquea el servidor)
        self.model = YOLO(self.model_path)
        self.model.to(get_device())

        # Log de clases para diagnóstico
        print(f"[ArmasPipeline] Modelo: {self.model_path}")
        print(f"[ArmasPipeline] Clases ({len(self.model.names)}): {dict(self.model.names)}")

        # Si ninguna clase se llama "person" → cargar YOLO11n de respaldo para personas
        model_class_names = [n.lower() for n in self.model.names.values()]
        main_has_person = any(k in n for n in model_class_names for k in _PERSON_KEYWORDS)
        if not main_has_person and self.func_state.get("captura_rostro"):
            print("[ArmasPipeline] Cargando modelo secundario para detección de personas...")
            self._person_model = YOLO(DEFAULT_MODEL)
            self._person_model.to(get_device())

        try:
            src = int(self.source_path)
        except (ValueError, TypeError):
            src = self.source_path

        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            err = self._make_error_frame(f"No se puede abrir: {self.source_path}")
            with self._lock:
                self._frame = err
            while not self._stop.is_set():
                time.sleep(0.5)
            return

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while not self._stop.is_set() and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # Loop si es archivo local
                if isinstance(src, str) and "://" not in src:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            annotated = self._process(frame)
            with self._lock:
                self._frame = annotated
            time.sleep(self.fps_limit)

        cap.release()

    # ── Procesado de frame ────────────────────────────────────────────────

    def _process(self, frame: np.ndarray) -> np.ndarray:
        annotated = frame.copy()
        h, w = frame.shape[:2]
        # ── Inferencia principal ─────────────────────────────────────────
        is_default_model = (self.model_path == DEFAULT_MODEL)
        eff_conf = self.conf_thresh

        results = self.model.track(
            frame,
            persist=True,
            conf=eff_conf,
            iou=IOU_THRESH,
            half=self.half,
            verbose=False,
            tracker=_TRACKER_CFG,
        )
        r = results[0]
        boxes = r.boxes if r.boxes is not None else []
        names = self.model.names

        armed_persons: List[tuple] = []  # (tid, x1,y1,x2,y2, conf) — man_with_weapon rastreado
        weapon_boxes: List[tuple]  = []  # (x1,y1,x2,y2, wtype, conf, tid or None) — presencia

        for box in boxes:
            cls   = int(box.cls[0])
            conf  = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cname = names.get(cls, "").lower()

            if conf < eff_conf:
                continue

            if any(k in cname for k in _IGNORE_KEYWORDS):
                continue

            # Persona armada — necesita ID de tracker estable
            if "with_weapon" in cname:
                if box.id is None:
                    continue
                armed_persons.append((int(box.id[0]), x1, y1, x2, y2, conf))
                continue

            # Persona sin arma — ignorar para lógica de detección
            if any(k in cname for k in _PERSON_KEYWORDS):
                continue

            # Con modelo default solo clases COCO de armas
            if is_default_model and cls not in COCO_WEAPON_CLS:
                continue

            wid = int(box.id[0]) if box.id is not None else None
            weapon_boxes.append((x1, y1, x2, y2, _weapon_type(cname), conf, wid))

        # ── Acumulado por track ID (deduplicado) ────────────────────────
        for (wx1, wy1, wx2, wy2, wtype, wconf, wid) in weapon_boxes:
            if wid is not None:
                if wid not in self._weapon_ids_seen:
                    self._weapon_ids_seen.add(wid)
                    self.total_weapons += 1
                    if wtype == "arma_blanca":
                        if wid not in self._blanca_ids_seen:
                            self._blanca_ids_seen.add(wid)
                            self.total_blanca += 1
                    elif wtype == "arma_fuego":
                        if wid not in self._fuego_ids_seen:
                            self._fuego_ids_seen.add(wid)
                            self.total_fuego += 1
            else:
                # Sin track ID — contar como detección única (no deduplicable)
                self.total_weapons += 1
                if wtype == "arma_blanca":
                    self.total_blanca += 1
                elif wtype == "arma_fuego":
                    self.total_fuego += 1
        for (atid, apx1, apy1, apx2, apy2, _) in armed_persons:
            if atid not in self._weapon_ids_seen:
                self._weapon_ids_seen.add(atid)
                self.total_weapons += 1
                self.total_blanca += 1  # armed_persons se cuentan como blanca (portador)

        # ── Validación 3×5 frames ─────────────────────────────────────
        if self.func_state.get("deteccion_arma", True):
            has_weapon = len(weapon_boxes) + len(armed_persons) > 0
            self._round_buffer.append(has_weapon)
            if len(self._round_buffer) >= 5:
                round_ratio = sum(self._round_buffer) / 5.0
                self._round_results.append(round_ratio)
                self._round_buffer = []
                if len(self._round_results) >= 3:
                    avg_ratio = sum(self._round_results) / 3.0
                    if avg_ratio >= 0.3:
                        self._alert_active = True
                        self._alert_timestamp = time.time()
                        # Capturar portadores visibles en este frame
                        for (atid, apx1, apy1, apx2, apy2, _) in armed_persons:
                            if self.func_state.get("captura_rostro"):
                                self._try_capture(frame, atid, apx1, apy1, apx2, apy2)
                    self._round_results = []

        # ── Dibujar + lógica ─────────────────────────────────────────────
        deteccion_on = self.func_state.get("deteccion_arma", True)
        self.weapon_current = len(weapon_boxes) + len(armed_persons)

        if deteccion_on:
            # Armas sueltas — presencia sin ID
            for (wx1, wy1, wx2, wy2, wtype, wconf, wid) in weapon_boxes:
                color = (0, 255, 255) if wtype == "arma_fuego" else (0, 0, 220)
                cv2.rectangle(annotated, (wx1, wy1), (wx2, wy2), color, 2)
                type_label = "Arma de Fuego" if wtype == "arma_fuego" else "Arma Blanca"
                ty = wy1 - 8 if wy1 > 20 else wy2 + 18
                cv2.putText(annotated, f"{type_label}  {wconf:.0%}",
                            (wx1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)

            # Persona armada — ID estable de la persona
            for (tid, px1, py1, px2, py2, pconf) in armed_persons:
                color = (85, 42, 24)
                cv2.rectangle(annotated, (px1, py1), (px2, py2), color, 2)
                ty = py1 - 8 if py1 > 20 else py2 + 18
                cv2.putText(annotated, f"ID[{tid}] - conf. {int(pconf * 100)}%  Portador",
                            (px1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)

                if self.func_state.get("captura_rostro"):
                    self._try_capture(frame, tid, px1, py1, px2, py2)

        # ── HUD ──────────────────────────────────────────────────────────
        cv2.putText(
            annotated,
            f"Armas: {self.total_weapons}  En zona: {self.weapon_current}  Capturas: {self.capture_count}",
            (12, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
        )

        return annotated

    def _try_capture(
        self,
        frame: np.ndarray,
        person_tid: int,
        px1: int, py1: int, px2: int, py2: int,
    ) -> None:
        """Captura la región de cabeza de la persona armada (top 35 % de su bbox)."""
        if len(self._captures.get(person_tid, [])) >= MAX_CAPS_PER_ID:
            return

        now = time.time()
        if now - self._last_cap_ts.get(person_tid, 0.0) < CAP_THROTTLE_S:
            return

        # Top 35 % del bbox como región de cabeza
        head_h  = max(int((py2 - py1) * 0.35), 1)
        y_top   = max(0, py1)
        y_bot   = min(frame.shape[0], py1 + head_h)
        x_left  = max(0, px1)
        x_right = min(frame.shape[1], px2)

        crop = frame[y_top:y_bot, x_left:x_right]
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            return

        tid_dir  = os.path.join(self._cap_dir, str(person_tid))
        os.makedirs(tid_dir, exist_ok=True)
        ts       = int(time.time() * 1000)
        filename = f"cap_{ts}.jpg"
        cv2.imwrite(os.path.join(tid_dir, filename), crop, [cv2.IMWRITE_JPEG_QUALITY, 88])

        url = f"/static/uploads/captures/{self.source_id}/{person_tid}/{filename}"
        if person_tid not in self._captures:
            self._captures[person_tid] = []
        self._captures[person_tid].append(url)
        self._last_cap_ts[person_tid] = now
        self.capture_count += 1

    # ── API pública ───────────────────────────────────────────────────────

    def get_frame_jpeg(self) -> Optional[bytes]:
        with self._lock:
            frame = self._frame
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
        return buf.tobytes() if ok else None

    def get_stats(self) -> dict:
        return {
            "source_id":       self.source_id,
            "weapon_count":    self.weapon_current,
            "total_weapons":   self.total_weapons,
            "total_blanca":    self.total_blanca,
            "total_fuego":     self.total_fuego,
            "capture_count":   self.capture_count,
            "captures": {str(tid): urls for tid, urls in self._captures.items()},
            "alert_active":    self._alert_active,
            "alert_timestamp": self._alert_timestamp,
        }

    def reset(self) -> None:
        self.weapon_current = 0
        self.total_weapons  = 0
        self.total_blanca   = 0
        self.total_fuego    = 0
        self.capture_count  = 0
        self._captures.clear()
        self._last_cap_ts.clear()
        self._weapon_ids_seen.clear()
        self._blanca_ids_seen.clear()
        self._fuego_ids_seen.clear()
        self._round_buffer = []
        self._round_results = []
        self._alert_active = False
        self._alert_timestamp = None

    def update_func_state(self, func_state: dict) -> None:
        self.func_state.update(func_state)


# ─────────────────────────────────────────────────────────────────────────────
class ArmasManager:
    """Singleton que gestiona pipelines de detección de armas."""

    _instance: Optional["ArmasManager"] = None
    _class_lock = threading.Lock()

    def __init__(self) -> None:
        self.pipelines: Dict[int, ArmasPipeline] = {}
        self._lock = threading.Lock()

    @classmethod
    def get(cls) -> "ArmasManager":
        with cls._class_lock:
            if cls._instance is None:
                cls._instance = ArmasManager()
        return cls._instance

    def start(
        self,
        source_id: int,
        source_path: str,
        func_state: dict,
        conf_thresh: float = None,
        half: bool = False,
        model_path: str = None,
        fps_limit: float = 0.0,
    ) -> None:
        if not multi_acquire():
            raise RuntimeError("Límite de 4 reproducciones simultáneas alcanzado")
        if not is_multi_enabled():
            self.stop_all()
        with self._lock:
            p = ArmasPipeline(source_id, source_path, func_state.copy(), conf_thresh, half, model_path, fps_limit)
            p.start()
            self.pipelines[source_id] = p

    def stop(self, source_id: int) -> None:
        with self._lock:
            p = self.pipelines.pop(source_id, None)
        if p:
            p.stop()
            multi_release()

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

    def update_func_state(self, func_state: dict) -> None:
        with self._lock:
            for p in self.pipelines.values():
                p.update_func_state(func_state)

    def get_frame_jpeg(self, source_id: int) -> Optional[bytes]:
        with self._lock:
            p = self.pipelines.get(source_id)
        return p.get_frame_jpeg() if p else None

    def get_stats(self, source_id: int) -> Optional[dict]:
        with self._lock:
            p = self.pipelines.get(source_id)
        return p.get_stats() if p else None

    def reset(self, source_id: int) -> None:
        with self._lock:
            p = self.pipelines.get(source_id)
        if p:
            p.reset()
