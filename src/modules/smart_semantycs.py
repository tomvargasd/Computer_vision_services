"""
src/modules/smart_semantycs.py
=============================
Pipeline de visión con vocabulario abierto (YOLO-E / YOLOE) para el módulo
"Smart Semantycs". Genera contadores y logs a partir de una "skill" (JSON
producido por Gemini en la Fase de interpretación).

Arquitectura (mismo patrón que personas.py):
  SmartSemntycsPipeline → un hilo por sesión activa
  SmartSemntycsManager  → singleton; SOLO 1 sesión activa a la vez dentro
                          de este módulo (no usa el límite global de 4).
"""
from __future__ import annotations

import os
import time
import threading
import logging
from typing import Optional, Dict, List

import cv2
import numpy as np
import torch

from src.config import BASE_DIR, CAPTURES_FOLDER
from src.database import (
    upsert_semantycs_counter, list_semantycs_counters,
    clear_semantycs_counters, insert_semantycs_log, clear_semantycs_logs,
)

logger = logging.getLogger("vision.smart_semantycs")

JPEG_Q  = 72
CONF_THRESH = 0.25
IOU_THRESH  = 0.50
TRACKER     = "bytetrack.yaml"

# Colores por defecto para clases no asociadas a ninguna regla
DEFAULT_COLOR = (255, 255, 255)   # blanco


def _iou(ax, ay, axx, ayy, bx, byy, bxx, byyy):
    """IoU de dos cajas xyxy."""
    ix1 = max(ax, bx)
    iy1 = max(ay, byy)
    ix2 = min(axx, bxx)
    iy2 = min(ayy, byyy)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, (axx - ax) * (ayy - ay))
    area_b = max(0.0, (bxx - bx) * (byyy - byy))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _matches_condition(cls_name: str, xyxy, boxes, condition: dict) -> bool:
    """Evalúa una condition del contrato de skill sobre una caja dada.

    condition = {"detect": [...], "overlap": [...], "min_overlap": 0.3}
    """
    detect = condition.get("detect") or []
    if cls_name not in detect:
        return False

    overlap = condition.get("overlap")
    if not overlap:
        return True

    min_overlap = float(condition.get("min_overlap", 0.3))
    x1, y1, x2, y2 = xyxy
    for other in boxes:
        if other["cls_name"] in overlap:
            if _iou(x1, y1, x2, y2, *other["xyxy"]) >= min_overlap:
                return True
    return False


class SmartSemntycsPipeline:
    """Pipeline de análisis para una sesión de Smart Semantycs."""

    def __init__(self, session_id: str, source_path: str, source_type: str,
                 classes: List[str], skill: dict, conf: float = CONF_THRESH,
                 fps_limit: float = 0.0):
        self.session_id  = session_id
        self.source_path = source_path
        self.source_type = source_type
        self.classes     = list(classes)     # nombres LVIS (orden = índice)
        self.skill       = skill or {}
        self.conf_thresh = conf
        self.fps_limit   = fps_limit

        self._counters_def = {c["id"]: c for c in self.skill.get("counters", [])}
        self._logs_def     = {l["id"]: l for l in self.skill.get("logs", [])}

        # Estado del hilo
        self._frame: Optional[np.ndarray] = None
        self._lock     = threading.Lock()
        self._stop     = threading.Event()
        self._paused   = threading.Event()   # set = pausado
        self._thread: Optional[threading.Thread] = None
        self.model     = None

        # Contadores en memoria (inicializados con lo persistido)
        self._counters: Dict[str, int] = {}
        self._counted_tracks: Dict[str, set] = {}   # counter_id -> {track_id}
        self._logged_tracks: Dict[str, set] = {}    # log_id -> {track_id}
        self._seen_tracks: set = set()

        # Capturas
        self._captures_dir = os.path.join(CAPTURES_FOLDER, "semantycs", session_id)
        os.makedirs(self._captures_dir, exist_ok=True)

        self._last_persist = 0.0
        self._persist_interval = 1.0
        self._h = 0
        self._w = 0

    # ── Control ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        saved = list_semantycs_counters(self.session_id)
        for c in saved:
            self._counters[c["counter_id"]] = c["value"]
        self._stop.clear()
        self._paused.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True,
            name=f"semantycs-{self.session_id[:8]}",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._thread = None
        self._persist_counters()
        del self.model
        torch.cuda.empty_cache()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def is_paused(self) -> bool:
        return self._paused.is_set()

    def reset(self) -> None:
        """Limpia contadores, logs, dedup y capturas (mantiene video+skill)."""
        self._counters = {}
        self._counted_tracks = {}
        self._logged_tracks = {}
        self._seen_tracks = set()
        clear_semantycs_counters(self.session_id)
        clear_semantycs_logs(self.session_id)
        if os.path.isdir(self._captures_dir):
            for f in os.listdir(self._captures_dir):
                try:
                    os.remove(os.path.join(self._captures_dir, f))
                except OSError:
                    pass

    # ── Bucle principal ──────────────────────────────────────────────────────

    def _make_error_frame(self, msg: str) -> np.ndarray:
        h, w = 480, 640
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(frame, "SOURCE ERROR", (int(w * 0.25), h // 2 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (85, 42, 24), 2, cv2.LINE_AA)
        max_chars = 55
        for i, part in enumerate([msg[j:j + max_chars] for j in range(0, min(len(msg), max_chars * 3), max_chars)]):
            cv2.putText(frame, part, (20, h // 2 + 10 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "Check the source path or permissions", (20, h - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 120, 120), 1, cv2.LINE_AA)
        return frame

    def _run(self) -> None:
        try:
            from ultralytics import YOLOE
            # El text-encoder (mobileclip2_b.ts) se busca en el CWD o en weights_dir.
            # Redirigimos weights_dir a la carpeta de modelos montada para que, si el
            # archivo ya está precacheado ahí (deploy.sh), no se re-descargue (242 MB).
            from src.config import MODELS_FOLDER
            try:
                from ultralytics.utils import SETTINGS
                SETTINGS["weights_dir"] = MODELS_FOLDER
            except Exception:
                pass
            # Encuentra el modelo en el CWD o en la ruta de modelos.
            model_path = self._resolve_model_path()
            self.model = YOLOE(model_path)
            self.model.to("cpu" if not torch.cuda.is_available() else "cuda:0")
            if self.classes:
                self.model.set_classes(self.classes)

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
                if self._paused.is_set():
                    time.sleep(0.05)
                    continue

                ret, frame = cap.read()
                if not ret:
                    if self.source_type == "video":
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break

                if self._h == 0:
                    self._h, self._w = frame.shape[:2]

                annotated = self._process(frame)
                with self._lock:
                    self._frame = annotated
                self._persist_counters()

                if self.fps_limit and self.fps_limit > 0:
                    time.sleep(self.fps_limit)

            cap.release()
        except Exception:
            logger.exception("Fatal error in smart_semantycs pipeline %s", self.session_id)
        finally:
            self._persist_counters()

    def _resolve_model_path(self) -> str:
        """Devuelve la ruta del .pt de YOLOE (CWD o static/uploads/models)."""
        from src.config import MODELS_FOLDER
        import glob
        cwd_pt = glob.glob(os.path.join(os.getcwd(), "yoloe-*-seg.pt"))
        if cwd_pt:
            return cwd_pt[0]
        mdir_pt = glob.glob(os.path.join(MODELS_FOLDER, "yoloe-*-seg.pt"))
        if mdir_pt:
            return mdir_pt[0]
        raise FileNotFoundError(
            "No se encontró un modelo YOLOE (.pt). Súbelo en Settings > Smart Semantycs."
        )

    # ── Procesado de un frame ────────────────────────────────────────────────

    def _process(self, frame: np.ndarray) -> np.ndarray:
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf_thresh,
            iou=IOU_THRESH,
            verbose=False,
            tracker=TRACKER,
        )

        annotated = frame.copy()
        r = results[0]
        boxes = r.boxes if r.boxes is not None else []
        now = time.time()

        parsed = []   # [{track_id, cls_idx, cls_name, conf, xyxy}]
        for box in boxes:
            if box.id is None:
                continue
            cls_idx = int(box.cls[0])
            if cls_idx >= len(self.classes):
                continue
            parsed.append({
                "track_id": int(box.id[0]),
                "cls_idx": cls_idx,
                "cls_name": self.classes[cls_idx],
                "conf": float(box.conf[0]),
                "xyxy": tuple(float(v) for v in box.xyxy[0].tolist()),
            })

        # ── Evaluar contadores y logs (dedup por track_id) ──────────────────
        fired_tracks = {}   # track_id -> color a dibujar
        for counter_id, cdef in self._counters_def.items():
            cond = cdef.get("condition") or {}
            for det in parsed:
                tid = det["track_id"]
                if tid in self._counted_tracks.get(counter_id, set()):
                    continue
                if _matches_condition(det["cls_name"], det["xyxy"], parsed, cond):
                    self._counted_tracks.setdefault(counter_id, set()).add(tid)
                    self._counters[counter_id] = self._counters.get(counter_id, 0) + 1
                    fired_tracks[tid] = cdef.get("color", "#22C55E")

        needs_capture = False
        for log_id, ldef in self._logs_def.items():
            cond = ldef.get("condition") or {}
            for det in parsed:
                tid = det["track_id"]
                if tid in self._logged_tracks.get(log_id, set()):
                    continue
                if _matches_condition(det["cls_name"], det["xyxy"], parsed, cond):
                    needs_capture = True
                    break   # ya hay al menos un evento nuevo este frame
            if needs_capture:
                break

        # ── Dibujar bounding boxes ──────────────────────────────────────────
        for det in parsed:
            tid = det["track_id"]
            x1, y1, x2, y2 = [int(v) for v in det["xyxy"]]
            color = DEFAULT_COLOR
            if tid in fired_tracks:
                color = self._hex_to_bgr(fired_tracks[tid])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{det['cls_name'].split('/')[0]} ID[{tid}] {int(det['conf'] * 100)}%"
            ty = y1 - 8 if y1 > 20 else y2 + 18
            cv2.putText(annotated, label, (x1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)
            self._seen_tracks.add(tid)

        # ── Persistir logs + captura si hubo eventos nuevos ─────────────────
        if needs_capture:
            self._persist_logs(parsed, annotated)

        # HUD: título de la skill y estado
        summary = self.skill.get("summary", "")
        if summary:
            cv2.putText(annotated, summary[:60], (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return annotated

    def _persist_logs(self, parsed: list, annotated: np.ndarray) -> None:
        capture_path = None
        for log_id, ldef in self._logs_def.items():
            cond = ldef.get("condition") or {}
            for det in parsed:
                tid = det["track_id"]
                if tid in self._logged_tracks.get(log_id, set()):
                    continue
                if _matches_condition(det["cls_name"], det["xyxy"], parsed, cond):
                    if capture_path is None:
                        capture_path = self._save_capture(annotated)
                    insert_semantycs_log(
                        self.session_id, log_id,
                        ldef.get("label", log_id),
                        ldef.get("event", ""),
                        ldef.get("priority", "info"),
                        capture_path,
                    )
                    self._logged_tracks.setdefault(log_id, set()).add(tid)

    def _save_capture(self, annotated: np.ndarray) -> str:
        fname = f"{int(time.time() * 1000)}.jpg"
        dest = os.path.join(self._captures_dir, fname)
        cv2.imwrite(dest, annotated, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
        rel = os.path.relpath(dest, BASE_DIR)
        return rel.replace(os.sep, "/")

    @staticmethod
    def _hex_to_bgr(hex_color: str):
        h = hex_color.lstrip("#")
        if len(h) != 6:
            return DEFAULT_COLOR
        try:
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        except ValueError:
            return DEFAULT_COLOR
        return (b, g, r)

    # ── Persistencia ─────────────────────────────────────────────────────────

    def _persist_counters(self) -> None:
        now = time.time()
        if now - self._last_persist < self._persist_interval:
            return
        self._last_persist = now
        for counter_id, cdef in self._counters_def.items():
            upsert_semantycs_counter(
                self.session_id, counter_id,
                cdef.get("label", counter_id),
                cdef.get("color", "#22C55E"),
                self._counters.get(counter_id, 0),
            )

    # ── API pública ──────────────────────────────────────────────────────────

    def get_frame_jpeg(self) -> Optional[bytes]:
        with self._lock:
            frame = self._frame
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
        return buf.tobytes() if ok else None

    def get_stats(self) -> dict:
        counters = [
            {"id": cid, "label": cdef.get("label", cid),
             "color": cdef.get("color", "#22C55E"),
             "value": self._counters.get(cid, 0)}
            for cid, cdef in self._counters_def.items()
        ]
        return {
            "session_id": self.session_id,
            "running": self.is_alive(),
            "paused": self.is_paused(),
            "counters": counters,
            "seen_tracks": len(self._seen_tracks),
        }


# ─────────────────────────────────────────────────────────────────────────────
class SmartSemntycsManager:
    """Singleton; gestiona pipelines de sesiones. SOLO 1 activa a la vez.

    NO usa multi_acquire/multi_release: el límite global de 4 pipelines de los
    módulos de detección existentes queda intacto.
    """

    _instance: Optional["SmartSemntycsManager"] = None
    _class_lock = threading.Lock()

    def __init__(self) -> None:
        self.pipelines: Dict[str, SmartSemntycsPipeline] = {}
        self._lock = threading.Lock()

    @classmethod
    def get(cls) -> "SmartSemntycsManager":
        with cls._class_lock:
            if cls._instance is None:
                cls._instance = SmartSemntycsManager()
        return cls._instance

    def start(self, session_id: str, source_path: str, source_type: str,
              classes: List[str], skill: dict, conf: float = CONF_THRESH,
              fps_limit: float = 0.0) -> None:
        # Regla del módulo: detener cualquier pipeline previo (otra sesión o la misma).
        with self._lock:
            ids = list(self.pipelines.keys())
        for sid in ids:
            if sid != session_id:
                self.stop(sid)
        self.stop(session_id)

        p = SmartSemntycsPipeline(
            session_id, source_path, source_type, classes, skill, conf, fps_limit,
        )
        p.start()
        with self._lock:
            self.pipelines[session_id] = p

    def stop(self, session_id: str) -> None:
        with self._lock:
            p = self.pipelines.pop(session_id, None)
        if p:
            p.stop()

    def stop_all(self) -> None:
        with self._lock:
            ids = list(self.pipelines.keys())
        for sid in ids:
            self.stop(sid)

    def is_running(self, session_id: str) -> bool:
        with self._lock:
            p = self.pipelines.get(session_id)
        if p is None:
            return False
        if not p.is_alive():
            with self._lock:
                self.pipelines.pop(session_id, None)
            return False
        return True

    def get_pipeline(self, session_id: str) -> Optional[SmartSemntycsPipeline]:
        with self._lock:
            return self.pipelines.get(session_id)

    def pause(self, session_id: str) -> None:
        with self._lock:
            p = self.pipelines.get(session_id)
        if p:
            p.pause()

    def resume(self, session_id: str) -> None:
        with self._lock:
            p = self.pipelines.get(session_id)
        if p:
            p.resume()

    def reset(self, session_id: str) -> None:
        with self._lock:
            p = self.pipelines.get(session_id)
        if p:
            p.reset()

    def get_frame_jpeg(self, session_id: str) -> Optional[bytes]:
        with self._lock:
            p = self.pipelines.get(session_id)
        return p.get_frame_jpeg() if p else None

    def get_stats(self, session_id: str) -> Optional[dict]:
        with self._lock:
            p = self.pipelines.get(session_id)
        return p.get_stats() if p else None
