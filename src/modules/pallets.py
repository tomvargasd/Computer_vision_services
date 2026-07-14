from __future__ import annotations

import cv2
import time
import threading
import numpy as np
from typing import Optional, Dict
from ultralytics import YOLO

from src.utils import get_device

MODEL_NAME  = "yolo11n.pt"
CONF_THRESH = 0.35
IOU_THRESH  = 0.50
JPEG_Q      = 72

PURPLE = (200, 0, 200)
YELLOW = (0, 255, 255)
WHITE  = (255, 255, 255)


class PalletsPipeline:
    def __init__(self, source_id: int, source_path: str, func_state: dict,
                 conf_thresh: float = CONF_THRESH, half: bool = False,
                 model_path: str = None,
                 area_x1: int = 25, area_y1: int = 25,
                 area_x2: int = 75, area_y2: int = 75,
                 classes: Optional[list] = None,
                 fps_limit: float = 0.0):
        self.source_id   = source_id
        self.source_path = source_path
        self.func_state  = func_state
        self.conf_thresh = conf_thresh
        self.half        = half
        self.model_path  = model_path or MODEL_NAME
        self.area_x1     = area_x1
        self.area_y1     = area_y1
        self.area_x2     = area_x2
        self.area_y2     = area_y2
        self.classes     = classes  # None = todas las clases
        self.fps_limit   = fps_limit

        self.model = None
        self._frame: Optional[np.ndarray] = None
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.total_in       = 0
        self.total_out      = 0
        self._track_state   = {}  # tid -> {enter_ts, exit_ts, absent_ts, counted_in}
        self.current_objects = 0
        self._h = 0
        self._w = 0

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True,
            name=f"pallets-pipe-{self.source_id}",
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
        rx1 = int(w * self.area_x1 / 100)
        ry1 = int(h * self.area_y1 / 100)
        rx2 = int(w * self.area_x2 / 100)
        ry2 = int(h * self.area_y2 / 100)

        track_kw = dict(
            persist=True, conf=self.conf_thresh,
            iou=IOU_THRESH, half=self.half, verbose=False,
            tracker="bytetrack.yaml",
        )
        if self.classes is not None:
            track_kw["classes"] = self.classes
        results = self.model.track(frame, **track_kw)

        annotated  = frame.copy()
        r          = results[0]
        boxes      = r.boxes if r.boxes is not None else []
        active_ids = set()
        now        = time.monotonic()
        COUNTED_CLASS = 0  # Solo clase 0 (Pallet) cuenta como IN/OUT
        MIN_DWELL     = 5.0   # segundos dentro del área antes de contar IN
        MIN_EXIT      = 2.0   # segundos fuera antes de contar OUT
        MAX_ABSENT    = 3.0   # segundos sin ver el track para descartarlo

        seen_class0: set = set()

        for box in boxes:
            if box.id is None:
                continue
            tid = int(box.id[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            active_ids.add(tid)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), YELLOW, 2)
            label = f"ID[{tid}] - conf. {int(conf * 100)}%"
            ty = y1 - 8 if y1 > 20 else y2 + 18
            cv2.putText(annotated, label, (x1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, YELLOW, 2, cv2.LINE_AA)

            if cls != COUNTED_CLASS or not self.func_state.get("conteo"):
                continue

            seen_class0.add(tid)

            if tid not in self._track_state:
                self._track_state[tid] = {
                    "enter_ts":   None,
                    "exit_ts":    None,
                    "absent_ts":  None,
                    "counted_in": False,
                }
            st = self._track_state[tid]
            st["absent_ts"] = None  # visible this frame

            inside = rx1 <= cx <= rx2 and ry1 <= cy <= ry2

            if inside:
                if st["enter_ts"] is None:
                    st["enter_ts"] = now  # start dwell timer
                st["exit_ts"] = None

                if not st["counted_in"] and (now - st["enter_ts"]) >= MIN_DWELL:
                    self.total_in += 1
                    st["counted_in"] = True
            else:
                st["enter_ts"] = None

                if st["counted_in"]:
                    if st["exit_ts"] is None:
                        st["exit_ts"] = now
                    elif (now - st["exit_ts"]) >= MIN_EXIT:
                        self.total_out += 1
                        st["counted_in"] = False
                        st["exit_ts"] = None

        # Tracks no vistos este frame
        for tid in list(self._track_state):
            if tid not in seen_class0:
                st = self._track_state[tid]
                if st["absent_ts"] is None:
                    st["absent_ts"] = now
                elif (now - st["absent_ts"]) >= MAX_ABSENT:
                    if st["counted_in"]:
                        self.total_out += 1
                    del self._track_state[tid]

        self.current_objects = len(active_ids)

        cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), PURPLE, 2)
        cv2.putText(annotated, "Area de conteo", (rx1 + 4, max(ry1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, PURPLE, 1, cv2.LINE_AA)

        cv2.putText(annotated, f"IN {self.total_in}   OUT {self.total_out}",
                    (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, YELLOW, 2, cv2.LINE_AA)
        cv2.putText(annotated, f"Objetos: {self.current_objects}",
                    (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)

        return annotated

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
            "current_objects": self.current_objects,
            "in_count":        self.total_in,
            "out_count":       self.total_out,
        }

    def set_area(self, x1: int, y1: int, x2: int, y2: int) -> None:
        self.area_x1 = max(0, min(100, x1))
        self.area_y1 = max(0, min(100, y1))
        self.area_x2 = max(0, min(100, x2))
        self.area_y2 = max(0, min(100, y2))
        if self.area_x1 > self.area_x2:
            self.area_x1, self.area_x2 = self.area_x2, self.area_x1
        if self.area_y1 > self.area_y2:
            self.area_y1, self.area_y2 = self.area_y2, self.area_y1

    def set_classes(self, classes: list) -> None:
        self.classes = classes

    def reset(self) -> None:
        self.total_in  = 0
        self.total_out = 0
        self._track_state.clear()


class PalletsManager:
    _instance: Optional["PalletsManager"] = None
    _class_lock = threading.Lock()

    def __init__(self) -> None:
        self.pipelines: Dict[int, PalletsPipeline] = {}
        self._lock = threading.Lock()

    @classmethod
    def get(cls) -> "PalletsManager":
        with cls._class_lock:
            if cls._instance is None:
                cls._instance = PalletsManager()
        return cls._instance

    def start(self, source_id: int, source_path: str, func_state: dict,
              conf_thresh: float = CONF_THRESH, half: bool = False,
              model_path: str = None,
              area_x1: int = 25, area_y1: int = 25,
              area_x2: int = 75, area_y2: int = 75,
              classes: Optional[list] = None,
              fps_limit: float = 0.0) -> None:
        self.stop_all()
        with self._lock:
            p = PalletsPipeline(source_id, source_path, func_state.copy(),
                                conf_thresh, half, model_path,
                                area_x1, area_y1, area_x2, area_y2,
                                classes=classes, fps_limit=fps_limit)
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

    def update_func_state(self, func_state: dict) -> None:
        with self._lock:
            for p in self.pipelines.values():
                p.func_state.update(func_state)

    def get_frame_jpeg(self, source_id: int) -> Optional[bytes]:
        with self._lock:
            p = self.pipelines.get(source_id)
        return p.get_frame_jpeg() if p else None

    def get_stats(self, source_id: int) -> Optional[dict]:
        with self._lock:
            p = self.pipelines.get(source_id)
        return p.get_stats() if p else None

    def set_area(self, source_id: int, x1: int, y1: int, x2: int, y2: int) -> None:
        with self._lock:
            p = self.pipelines.get(source_id)
        if p:
            p.set_area(x1, y1, x2, y2)

    def set_classes(self, classes: list) -> None:
        with self._lock:
            for p in self.pipelines.values():
                p.set_classes(classes)

    def reset(self, source_id: int) -> None:
        with self._lock:
            p = self.pipelines.get(source_id)
        if p:
            p.reset()
