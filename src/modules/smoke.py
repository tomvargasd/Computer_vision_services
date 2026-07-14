from __future__ import annotations

import cv2
import os
import time
import threading
import numpy as np
from typing import Optional, Dict
from ultralytics import YOLO
from datetime import datetime

from src.utils import get_device

MODEL_NAME  = "yolo11n.pt"
CONF_THRESH = 0.35
IOU_THRESH  = 0.50
JPEG_Q      = 72

RED    = (0, 0, 255)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
WHITE  = (255, 255, 255)

SMOKE_CLASSES = [0]

CAPTURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "static", "uploads", "captures", "smoke",
)


class SmokePipeline:
    def __init__(self, source_id: int, source_path: str, func_state: dict,
                 conf_thresh: float = CONF_THRESH, half: bool = False,
                 model_path: str = None,
                 target_classes: Optional[list] = None,
                 fps_limit: float = 0.0):
        self.source_id   = source_id
        self.source_path = source_path
        self.func_state  = func_state
        self.conf_thresh = conf_thresh
        self.half        = half
        self.model_path  = model_path or MODEL_NAME
        self.target_classes = target_classes or SMOKE_CLASSES
        self.fps_limit   = fps_limit

        self.model: Optional[YOLO] = None
        self._frame: Optional[np.ndarray] = None
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._h = 0
        self._w = 0

        self.smoke_detected  = False
        self.alert_triggered = False
        self.first_detection_time: Optional[str] = None
        self._evidence_saved = False

        os.makedirs(CAPTURES_DIR, exist_ok=True)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True,
            name=f"smoke-pipe-{self.source_id}",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._thread = None

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _make_error_frame(self, msg: str) -> np.ndarray:
        h, w = 480, 640
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(frame, "ERROR DE FUENTE", (int(w * 0.25), h // 2 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (85, 42, 24), 2, cv2.LINE_AA)
        for i, part in enumerate([msg[j:j+55] for j in range(0, min(len(msg), 165), 55)]):
            cv2.putText(frame, part, (20, h // 2 + 10 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "Verifica la ruta o permisos de la fuente", (20, h - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 120, 120), 1, cv2.LINE_AA)
        return frame

    def _run(self) -> None:
        self.model = YOLO(self.model_path)
        self.model.to(get_device())

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

        first_frame = True
        while not self._stop.is_set() and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if isinstance(src, str) and "://" not in src:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            if first_frame:
                self._h, self._w = frame.shape[:2]
                first_frame = False

            annotated = self._process(frame)
            with self._lock:
                self._frame = annotated
            time.sleep(self.fps_limit)

        cap.release()

    def _process(self, frame: np.ndarray) -> np.ndarray:
        h, w = self._h, self._w
        annotated = frame.copy()

        results = self.model.track(
            frame, persist=True, conf=self.conf_thresh,
            iou=IOU_THRESH, half=self.half, verbose=False,
            tracker="bytetrack.yaml",
        )
        r = results[0]
        boxes = r.boxes if r.boxes is not None else []

        frame_has_smoke = False
        for box in boxes:
            cls = int(box.cls[0])
            if cls not in self.target_classes:
                continue
            frame_has_smoke = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            color = RED if self.smoke_detected else ORANGE
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"SMOKE/FIRE {conf:.2f}", (x1, max(y1 - 8, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)

        if frame_has_smoke:
            if not self.smoke_detected:
                self.smoke_detected = True
                self.first_detection_time = datetime.now().strftime("%H:%M:%S")
                self.alert_triggered = True
                if not self._evidence_saved:
                    self._evidence_saved = True
                    self._save_evidence(annotated)

        overlay = annotated.copy()
        lines = []
        if self.smoke_detected:
            lines.append(("HUMO / FUEGO DETECTADO", RED))
            lines.append((f"Primera vez: {self.first_detection_time or '--'}", ORANGE))
        else:
            lines.append(("MONITOREANDO...", YELLOW))

        lh = 20
        for i, (txt, clr) in enumerate(lines):
            y = h - 30 - (len(lines) - 1 - i) * lh
            cv2.putText(overlay, txt, (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, clr, 2, cv2.LINE_AA)

        if self.smoke_detected:
            alert_w = 280
            alert_h = 50
            ax = w - alert_w - 16
            ay = h - alert_h - 16
            cv2.rectangle(overlay, (ax, ay), (ax + alert_w, ay + alert_h), RED, -1)
            cv2.putText(overlay, "HUMO / FUEGO DETECTADO", (ax + 10, ay + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, WHITE, 2, cv2.LINE_AA)
            cv2.putText(overlay, f"{self.first_detection_time or ''}", (ax + 10, ay + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, WHITE, 1, cv2.LINE_AA)

        return overlay

    def _save_evidence(self, frame: np.ndarray) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"smoke_{self.source_id}_{ts}.jpg"
        path = os.path.join(CAPTURES_DIR, fname)
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])

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
            "smoke_detected":  self.smoke_detected,
            "alert_triggered": self.alert_triggered,
            "first_detection":  self.first_detection_time,
        }

    def reset(self) -> None:
        self.smoke_detected    = False
        self.alert_triggered   = False
        self.first_detection_time = None
        self._evidence_saved   = False

    def set_classes(self, classes: list) -> None:
        self.target_classes = classes


class SmokeManager:
    _instance: Optional["SmokeManager"] = None
    _class_lock = threading.Lock()

    def __init__(self) -> None:
        self.pipelines: Dict[int, SmokePipeline] = {}
        self._lock = threading.Lock()

    @classmethod
    def get(cls) -> "SmokeManager":
        with cls._class_lock:
            if cls._instance is None:
                cls._instance = SmokeManager()
        return cls._instance

    def start(self, source_id: int, source_path: str, func_state: dict,
              conf_thresh: float = CONF_THRESH, half: bool = False,
              model_path: str = None, fps_limit: float = 0.0) -> None:
        self.stop_all()
        with self._lock:
            p = SmokePipeline(source_id, source_path, func_state.copy(),
                              conf_thresh, half, model_path,
                              fps_limit=fps_limit)
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

    def reset(self, source_id: int) -> None:
        with self._lock:
            p = self.pipelines.get(source_id)
        if p:
            p.reset()

    def update_func_state(self, func_state: dict) -> None:
        with self._lock:
            for p in self.pipelines.values():
                p.func_state.update(func_state)
