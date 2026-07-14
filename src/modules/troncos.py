from __future__ import annotations

import cv2
import time
import threading
import numpy as np
from typing import Optional, Dict
from ultralytics import YOLO

from src.utils import get_device
from src.modules.base import multi_acquire, multi_release, is_multi_enabled

MODEL_NAME  = "yolo11n.pt"
CONF_THRESH = 0.35
IOU_THRESH  = 0.50
JPEG_Q      = 72

PURPLE = (200, 0, 200)
YELLOW = (0, 255, 255)
WHITE  = (255, 255, 255)


class TroncosPipeline:
    def __init__(self, source_id: int, source_path: str, func_state: dict,
                 conf_thresh: float = CONF_THRESH, half: bool = False,
                 model_path: str = None, line_x_pct: int = 50,
                 fps_limit: float = 0.0):
        self.source_id   = source_id
        self.source_path = source_path
        self.func_state  = func_state
        self.conf_thresh = conf_thresh
        self.half        = half
        self.model_path  = model_path or MODEL_NAME
        self.line_x_pct  = line_x_pct
        self.fps_limit   = fps_limit

        self.model = None
        self._frame: Optional[np.ndarray] = None
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.total_in   = 0
        self.total_out  = 0
        self._prev_cx: Dict[int, int] = {}
        self._cross_state: Dict[int, str] = {}
        self.current_objects = 0
        self._h = 0
        self._w = 0

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True,
            name=f"troncos-pipe-{self.source_id}",
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
        h, w   = self._h, self._w
        line_x = int(w * self.line_x_pct / 100)

        results = self.model.track(
            frame, persist=True, conf=self.conf_thresh,
            iou=IOU_THRESH, half=self.half, verbose=False,
            tracker="bytetrack.yaml",
        )

        annotated  = frame.copy()
        r          = results[0]
        boxes      = r.boxes if r.boxes is not None else []
        active_ids = set()

        for box in boxes:
            if box.id is None:
                continue
            tid = int(box.id[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            active_ids.add(tid)

            if self.func_state.get("conteo"):
                prev = self._prev_cx.get(tid)
                if prev is not None:
                    state = self._cross_state.get(tid, "none")
                    crossed_left  = prev < line_x and cx >= line_x
                    crossed_right = prev > line_x and cx <= line_x

                    if crossed_left:
                        if state == "none":
                            self.total_in += 1
                            self._cross_state[tid] = "inside"
                    elif crossed_right:
                        if state in ("none", "inside"):
                            self.total_out += 1
                            self._cross_state[tid] = "done"

                self._prev_cx[tid] = cx

            cv2.rectangle(annotated, (x1, y1), (x2, y2), YELLOW, 2)
            label = f"ID[{tid}] - conf. {int(conf * 100)}%"
            ty = y1 - 8 if y1 > 20 else y2 + 18
            cv2.putText(annotated, label, (x1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, YELLOW, 2, cv2.LINE_AA)

        gone = set(self._prev_cx.keys()) - active_ids
        for tid in gone:
            self._prev_cx.pop(tid, None)

        self.current_objects = len(active_ids)

        if self.func_state.get("conteo"):
            cv2.line(annotated, (line_x, 0), (line_x, h), PURPLE, 2)
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

    def set_line_x(self, pct: int) -> None:
        self.line_x_pct = max(0, min(100, pct))

    def reset(self) -> None:
        self.total_in  = 0
        self.total_out = 0
        self._prev_cx.clear()
        self._cross_state.clear()


class TroncosManager:
    _instance: Optional["TroncosManager"] = None
    _class_lock = threading.Lock()

    def __init__(self) -> None:
        self.pipelines: Dict[int, TroncosPipeline] = {}
        self._lock = threading.Lock()

    @classmethod
    def get(cls) -> "TroncosManager":
        with cls._class_lock:
            if cls._instance is None:
                cls._instance = TroncosManager()
        return cls._instance

    def start(self, source_id: int, source_path: str, func_state: dict,
              conf_thresh: float = CONF_THRESH, half: bool = False,
              model_path: str = None, line_x_pct: int = 50,
              fps_limit: float = 0.0) -> None:
        if not multi_acquire():
            raise RuntimeError("Límite de 4 reproducciones simultáneas alcanzado")
        if not is_multi_enabled():
            self.stop_all()
        with self._lock:
            p = TroncosPipeline(source_id, source_path, func_state.copy(),
                                conf_thresh, half, model_path, line_x_pct,
                                fps_limit=fps_limit)
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
                p.func_state.update(func_state)

    def get_frame_jpeg(self, source_id: int) -> Optional[bytes]:
        with self._lock:
            p = self.pipelines.get(source_id)
        return p.get_frame_jpeg() if p else None

    def get_stats(self, source_id: int) -> Optional[dict]:
        with self._lock:
            p = self.pipelines.get(source_id)
        return p.get_stats() if p else None

    def set_line_x(self, source_id: int, pct: int) -> None:
        with self._lock:
            p = self.pipelines.get(source_id)
        if p:
            p.set_line_x(pct)

    def reset(self, source_id: int) -> None:
        with self._lock:
            p = self.pipelines.get(source_id)
        if p:
            p.reset()
