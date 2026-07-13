from __future__ import annotations

import cv2
import os
import time
import threading
import numpy as np
from typing import Optional, Dict, List
from ultralytics import YOLO
from datetime import datetime
from dataclasses import dataclass

from src.utils import get_device

MODEL_NAME  = "yolo11n.pt"
CONF_THRESH = 0.35
PLATE_CONF  = 0.35
IOU_THRESH  = 0.50
JPEG_Q      = 80

GREEN  = (0, 255, 0)
RED    = (0, 0, 255)
YELLOW = (0, 255, 255)
CYAN   = (255, 255, 0)
WHITE  = (255, 255, 255)

DEFAULT_VEHICLE_CLASSES = [2, 3, 5, 7]

CAPTURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "static", "uploads", "captures", "vehiculos",
)

try:
    import pytesseract
    _TESSERACT_OK = True
except ImportError:
    _TESSERACT_OK = False


@dataclass
class VehiculoRegistro:
    track_id: int
    timestamp: str
    image_file: str
    direction: str
    plate_text: str = ""
    plate_crop: str = ""


def _detect_plate_cv2(vehicle_roi: np.ndarray) -> tuple[str, Optional[np.ndarray]]:
    gray = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
    for blur_size, canny_low, canny_high in [(5, 30, 150), (3, 20, 100)]:
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        edges = cv2.Canny(blurred, canny_low, canny_high)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / h if h > 0 else 0
            area_ratio = (w * h) / (vehicle_roi.shape[0] * vehicle_roi.shape[1])
            if aspect < 1.5 or aspect > 6.0:
                continue
            if w < 40 or h < 12:
                continue
            if area_ratio < 0.02 or area_ratio > 0.6:
                continue
            plate_roi = vehicle_roi[y:y+h, x:x+w]
            if plate_roi.size > 0:
                text = _ocr_plate(plate_roi)
                if text and len(text) >= 3:
                    return text, plate_roi
    h_roi, w_roi = vehicle_roi.shape[:2]
    bottom = vehicle_roi[h_roi // 2:, :]
    if bottom.size > 0:
        text = _ocr_plate(bottom)
        if text and len(text) >= 2:
            return text, bottom
    return "", None


def _ocr_plate(plate_roi: np.ndarray) -> str:
    if not _TESSERACT_OK:
        return ""
    try:
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        config = "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
        text = pytesseract.image_to_string(thresh, config=config).strip()
        text = "".join(c for c in text if c.isalnum() or c == "-")
        return text
    except Exception:
        return ""


class VehiculosPipeline:
    def __init__(self, source_id: int, source_path: str, func_state: dict,
                 conf_thresh: float = CONF_THRESH, half: bool = False,
                 model_path: str = None,
                 plate_model_path: str = None,
                 plate_conf_thresh: float = PLATE_CONF,
                 classes: Optional[list] = None,
                 line_mode: str = "horizontal", line_pos: int = 50):
        self.source_id   = source_id
        self.source_path = source_path
        self.func_state  = func_state
        self.conf_thresh = conf_thresh
        self.half        = half
        self.model_path  = model_path or MODEL_NAME
        self.plate_model_path = plate_model_path
        self.plate_conf_thresh = plate_conf_thresh
        self.classes     = classes or DEFAULT_VEHICLE_CLASSES
        self.line_mode   = line_mode
        self.line_pos    = line_pos

        self.model: Optional[YOLO] = None
        self.plate_model: Optional[YOLO] = None
        self._frame: Optional[np.ndarray] = None
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._h = 0
        self._w = 0

        self.total_in       = 0
        self.total_out      = 0
        self.total_vehicles = 0
        self.plates_count   = 0

        self._prev_pos: Dict[int, tuple] = {}
        self._counted: set = set()
        self._captured: set = set()

        self.registros: List[VehiculoRegistro] = []
        self.placas_detectadas: List[dict] = []
        self._max_registros = 50

        self.plate_detection_enabled = False

        os.makedirs(CAPTURES_DIR, exist_ok=True)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True,
            name=f"vehiculos-pipe-{self.source_id}",
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

        if self.plate_model_path:
            try:
                self.plate_model = YOLO(self.plate_model_path)
                self.plate_model.to(get_device())
                self.plate_detection_enabled = True
            except Exception as e:
                self.plate_model = None

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

        cap.release()

    def _process(self, frame: np.ndarray) -> np.ndarray:
        h, w = self._h, self._w
        annotated = frame.copy()

        track_kw = dict(
            persist=True, conf=self.conf_thresh,
            iou=IOU_THRESH, half=self.half, verbose=False,
            tracker="bytetrack.yaml",
        )
        if self.classes is not None:
            track_kw["classes"] = self.classes

        results = self.model.track(frame, **track_kw)
        r = results[0]
        boxes = r.boxes if r.boxes is not None else []

        if self.line_mode == "horizontal":
            line_px = int(h * self.line_pos / 100)
            draw_p1, draw_p2 = (0, line_px), (w, line_px)
        else:
            line_px = int(w * self.line_pos / 100)
            draw_p1, draw_p2 = (line_px, 0), (line_px, h)

        active_ids = set()

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

            color = CYAN
            direction_label = ""
            if tid in self._counted:
                direction_label = next((r.direction for r in reversed(self.registros) if r.track_id == tid), "")
                color = GREEN if direction_label == "in" else RED

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"V-{tid} {direction_label}"
            ty = y1 - 8 if y1 > 20 else y2 + 18
            cv2.putText(annotated, label, (x1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)

            if not self.func_state.get("conteo"):
                continue

            prev = self._prev_pos.get(tid)
            self._prev_pos[tid] = (cx, cy)
            if prev is None:
                continue

            if tid in self._counted:
                continue

            crossed = False
            direction = None
            if self.line_mode == "horizontal":
                if prev[1] < line_px <= cy:
                    crossed = True
                    direction = "in"
                elif prev[1] > line_px >= cy:
                    crossed = True
                    direction = "out"
            else:
                if prev[0] < line_px <= cx:
                    crossed = True
                    direction = "in"
                elif prev[0] > line_px >= cx:
                    crossed = True
                    direction = "out"

            if crossed:
                self._counted.add(tid)
                if direction == "in":
                    self.total_in += 1
                else:
                    self.total_out += 1
                self.total_vehicles += 1

                if tid not in self._captured:
                    self._captured.add(tid)
                    crop = frame[y1:y2, x1:x2]
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fname = f"veh_{self.source_id}_{tid}_{ts}.jpg"
                    path = os.path.join(CAPTURES_DIR, fname)
                    cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])

                    plate_text = ""
                    plate_crop_file = ""
                    if self.plate_detection_enabled and crop.size > 0:
                        plate_roi = None
                        if self.plate_model:
                            plate_results = self.plate_model(crop, conf=self.plate_conf_thresh, verbose=False, max_det=5)
                            for pr in plate_results:
                                pb = pr.boxes
                                if pb is None:
                                    continue
                                for box in pb:
                                    px1, py1, px2, py2 = map(int, box.xyxy[0])
                                    px1 = max(0, px1); py1 = max(0, py1)
                                    px2 = min(crop.shape[1], px2); py2 = min(crop.shape[0], py2)
                                    plate_roi = crop[py1:py2, px1:px2]
                                    if plate_roi.size > 0:
                                        plate_text = _ocr_plate(plate_roi)
                                        if plate_text:
                                            break
                                if plate_text:
                                    break
                        if not plate_text:
                            plate_text, plate_roi = _detect_plate_cv2(crop)

                        if plate_text:
                            self.plates_count += 1
                            plate_fname = f"plate_{self.source_id}_{tid}_{ts}.jpg"
                            plate_path = os.path.join(CAPTURES_DIR, plate_fname)
                            if plate_roi is not None and plate_roi.size > 0:
                                cv2.imwrite(plate_path, plate_roi, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
                                plate_crop_file = plate_fname
                            self.placas_detectadas.append({
                                "track_id": tid,
                                "timestamp": ts,
                                "plate": plate_text or "---",
                                "image": plate_crop_file,
                            })

                    reg = VehiculoRegistro(
                        track_id=tid,
                        timestamp=datetime.now().strftime("%H:%M:%S"),
                        image_file=fname,
                        direction=direction or "",
                        plate_text=plate_text,
                        plate_crop=plate_crop_file,
                    )
                    self.registros.append(reg)
                    if len(self.registros) > self._max_registros:
                        old = self.registros.pop(0)
                        old_path = os.path.join(CAPTURES_DIR, old.image_file)
                        if os.path.exists(old_path):
                            os.remove(old_path)

        cv2.line(annotated, draw_p1, draw_p2, YELLOW, 2)
        cv2.putText(annotated, f"IN {self.total_in}  OUT {self.total_out}",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2, cv2.LINE_AA)
        cv2.putText(annotated, f"Vehiculos: {self.total_vehicles}",
                    (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2, cv2.LINE_AA)

        return annotated

    def get_frame_jpeg(self) -> Optional[bytes]:
        with self._lock:
            frame = self._frame
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
        return buf.tobytes() if ok else None

    def get_stats(self) -> dict:
        regs = []
        for r in self.registros:
            regs.append({
                "track_id": r.track_id,
                "timestamp": r.timestamp,
                "image": r.image_file,
                "direction": r.direction,
                "plate_text": r.plate_text,
                "plate_crop": r.plate_crop,
            })
        return {
            "source_id": self.source_id,
            "total_in": self.total_in,
            "total_out": self.total_out,
            "total_vehicles": self.total_vehicles,
            "plates_count": self.plates_count,
            "registros": regs,
            "placas": list(self.placas_detectadas),
        }

    def set_line_mode(self, mode: str) -> None:
        self.line_mode = mode

    def set_line_pos(self, pos: int) -> None:
        self.line_pos = max(0, min(100, pos))

    def set_plate_detection(self, enabled: bool) -> None:
        self.plate_detection_enabled = enabled

    def reset(self) -> None:
        self.total_in = 0
        self.total_out = 0
        self.total_vehicles = 0
        self.plates_count = 0
        self._prev_pos.clear()
        self._counted.clear()
        self._captured.clear()
        self.registros.clear()
        self.placas_detectadas.clear()


class VehiculosManager:
    _instance: Optional["VehiculosManager"] = None
    _class_lock = threading.Lock()

    def __init__(self) -> None:
        self.pipelines: Dict[int, VehiculosPipeline] = {}
        self._lock = threading.Lock()

    @classmethod
    def get(cls) -> "VehiculosManager":
        with cls._class_lock:
            if cls._instance is None:
                cls._instance = VehiculosManager()
        return cls._instance

    def start(self, source_id: int, source_path: str, func_state: dict,
              conf_thresh: float = CONF_THRESH, half: bool = False,
              model_path: str = None,
              plate_model_path: str = None,
              plate_conf_thresh: float = PLATE_CONF,
              classes: Optional[list] = None,
              line_mode: str = "horizontal", line_pos: int = 50) -> None:
        self.stop_all()
        with self._lock:
            p = VehiculosPipeline(source_id, source_path, func_state.copy(),
                                  conf_thresh, half, model_path,
                                  plate_model_path=plate_model_path,
                                  plate_conf_thresh=plate_conf_thresh,
                                  classes=classes,
                                  line_mode=line_mode, line_pos=line_pos)
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

    def reset(self, source_id: int) -> None:
        with self._lock:
            p = self.pipelines.get(source_id)
        if p:
            p.reset()

    def set_line_mode(self, source_id: int, mode: str) -> None:
        with self._lock:
            p = self.pipelines.get(source_id)
        if p:
            p.set_line_mode(mode)

    def set_line_pos(self, source_id: int, pos: int) -> None:
        with self._lock:
            p = self.pipelines.get(source_id)
        if p:
            p.set_line_pos(pos)

    def set_plate_detection(self, source_id: int, enabled: bool) -> None:
        with self._lock:
            p = self.pipelines.get(source_id)
        if p:
            p.set_plate_detection(enabled)
