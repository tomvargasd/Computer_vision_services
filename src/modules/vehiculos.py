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
from concurrent.futures import ThreadPoolExecutor

from src.utils import get_device

MODEL_NAME  = "yolo11n.pt"
CONF_THRESH = 0.35
PLATE_CONF  = 0.35
IOU_THRESH  = 0.50
JPEG_Q      = 80
OCR_WORKERS = 4

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

try:
    import easyocr
    _EASYOCR_OK = True
except ImportError:
    _EASYOCR_OK = False

_easyocr_reader = None
_easyocr_init_lock = threading.Lock()
_easyocr_infer_lock = threading.Lock()


def _get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None and _EASYOCR_OK:
        with _easyocr_init_lock:
            if _easyocr_reader is None:
                try:
                    _easyocr_reader = easyocr.Reader(['en'], gpu=False)
                except Exception:
                    pass
    return _easyocr_reader


@dataclass
class VehiculoRegistro:
    track_id: int
    timestamp: str
    image_file: str
    direction: str
    plate_text: str = ""
    plate_crop: str = ""


def _deskew_plate(plate_roi: np.ndarray) -> np.ndarray:
    h, w = plate_roi.shape[:2]
    if h < 10 or w < 30:
        return plate_roi
    gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100,
                            minLineLength=w // 3, maxLineGap=10)
    if lines is None:
        return plate_roi
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        if abs(angle) < 30:
            angles.append(angle)
    if not angles:
        return plate_roi
    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return plate_roi
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    return cv2.warpAffine(plate_roi, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _preprocess_plate(plate_roi: np.ndarray) -> list[np.ndarray]:
    h, w = plate_roi.shape[:2]
    if h < 8 or w < 20:
        return []
    deskewed = _deskew_plate(plate_roi)
    target_h = max(48, h * 2)
    scale = target_h / h
    resized = cv2.resize(deskewed, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, 9, 15, 15)
    variants = []
    _, otsu = cv2.threshold(bilateral, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)
    adapt = cv2.adaptiveThreshold(bilateral, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 25, 8)
    variants.append(adapt)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(bilateral)
    _, clahe_thresh = cv2.threshold(enhanced, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(clahe_thresh)
    kernel_sharp = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(bilateral, -1, kernel_sharp)
    _, sharp_thresh = cv2.threshold(sharpened, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(sharp_thresh)
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = []
    for v in variants:
        v = cv2.morphologyEx(v, cv2.MORPH_CLOSE, kernel_morph, iterations=1)
        v = cv2.morphologyEx(v, cv2.MORPH_OPEN, kernel_morph, iterations=1)
        cleaned.append(v)
    return cleaned


def _ocr_tesseract(plate_roi: np.ndarray) -> str:
    if not _TESSERACT_OK:
        return ""
    try:
        variants = _preprocess_plate(plate_roi)
        if not variants:
            return ""
        config = (
            "--psm 7 "
            "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
        )
        candidates = []
        for img in variants:
            try:
                text = pytesseract.image_to_string(img, config=config).strip()
                text = "".join(c for c in text if c.isalnum() or c == "-")
                if len(text) >= 3:
                    candidates.append(text)
            except Exception:
                pass
        if not candidates:
            config_alt = (
                "--psm 8 "
                "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
            )
            for img in variants:
                try:
                    text = pytesseract.image_to_string(
                        img, config=config_alt).strip()
                    text = "".join(c for c in text if c.isalnum() or c == "-")
                    if len(text) >= 2:
                        candidates.append(text)
                except Exception:
                    pass
        if not candidates:
            return ""
        from collections import Counter
        return Counter(candidates).most_common(1)[0][0]
    except Exception:
        return ""


def _ocr_easyocr(plate_roi: np.ndarray) -> str:
    reader = _get_easyocr_reader()
    if reader is None:
        return ""
    try:
        resized = cv2.resize(plate_roi, None, fx=2, fy=2,
                             interpolation=cv2.INTER_CUBIC)
        with _easyocr_infer_lock:
            results = reader.readtext(resized)
        texts = []
        for (bbox, text, conf) in results:
            clean = "".join(c for c in text if c.isalnum() or c == "-")
            if conf >= 0.3 and len(clean) >= 3:
                texts.append((clean.upper(), conf))
        if texts:
            texts.sort(key=lambda x: -x[1])
            return texts[0][0]
        return ""
    except Exception:
        return ""


def _ocr_combined(plate_roi: np.ndarray) -> str:
    tesseract_text = _ocr_tesseract(plate_roi)
    easyocr_text = _ocr_easyocr(plate_roi)
    if tesseract_text and easyocr_text:
        if tesseract_text == easyocr_text:
            return tesseract_text
        if tesseract_text in easyocr_text or easyocr_text in tesseract_text:
            longer = tesseract_text if len(tesseract_text) >= len(easyocr_text) else easyocr_text
            return longer
        return easyocr_text
    elif tesseract_text:
        return tesseract_text
    elif easyocr_text:
        return easyocr_text
    return ""


def _detect_plate_cv2(vehicle_roi: np.ndarray) -> tuple[str, Optional[np.ndarray]]:
    h_roi, w_roi = vehicle_roi.shape[:2]
    if h_roi < 20 or w_roi < 40:
        return "", None
    gray = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, 9, 15, 15)
    candidates = []
    for blur, canny_lo, canny_hi in [(5, 30, 150), (3, 20, 100), (7, 50, 200)]:
        blurred = cv2.GaussianBlur(bilateral, (blur, blur), 0)
        edges = cv2.Canny(blurred, canny_lo, canny_hi)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        edges = cv2.dilate(edges, kernel, iterations=1)
        cnts, _ = cv2.findContours(edges, cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            aspect = w / h if h > 0 else 0
            if aspect < 1.5 or aspect > 6.0:
                continue
            if w < 40 or h < 10:
                continue
            area_ratio = (w * h) / (h_roi * w_roi)
            if area_ratio < 0.01 or area_ratio > 0.5:
                continue
            rect_area = w * h
            contour_area = cv2.contourArea(approx)
            extent = contour_area / rect_area if rect_area > 0 else 0
            if extent < 0.3:
                continue
            plate_roi = vehicle_roi[y:y+h, x:x+w]
            if plate_roi.size > 0:
                candidates.append(plate_roi)
    for cand in candidates:
        text = _ocr_combined(cand)
        if text and len(text) >= 4:
            return text, cand
    for cand in candidates:
        text = _ocr_combined(cand)
        if text and len(text) >= 3:
            return text, cand
    bottom = vehicle_roi[h_roi // 2:, :]
    if bottom.size > 0:
        text = _ocr_combined(bottom)
        if text and len(text) >= 3:
            return text, bottom
    lower = vehicle_roi[int(h_roi * 0.6):, :]
    if lower.size > 0:
        text = _ocr_combined(lower)
        if text and len(text) >= 2:
            return text, lower
    return "", None


class VehiculosPipeline:
    def __init__(self, source_id: int, source_path: str, func_state: dict,
                 conf_thresh: float = CONF_THRESH, half: bool = False,
                 model_path: str = None,
                 plate_model_path: str = None,
                 plate_conf_thresh: float = PLATE_CONF,
                 classes: Optional[list] = None,
                 line_mode: str = "horizontal", line_pos: int = 50,
                 fps_limit: float = 0.0):
        self.source_id   = source_id
        self.source_path = source_path
        self.func_state  = func_state
        self.conf_thresh = conf_thresh
        self.half        = half
        self.model_path  = model_path or MODEL_NAME
        self.fps_limit   = fps_limit
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

        self._ocr_pool: Optional[ThreadPoolExecutor] = None

        self.registros: List[VehiculoRegistro] = []
        self.placas_detectadas: List[dict] = []
        self._max_registros = 50

        self.plate_detection_enabled = True

        os.makedirs(CAPTURES_DIR, exist_ok=True)

    def start(self) -> None:
        self._stop.clear()
        self._ocr_pool = ThreadPoolExecutor(max_workers=OCR_WORKERS)
        self._thread = threading.Thread(
            target=self._run, daemon=True,
            name=f"vehiculos-pipe-{self.source_id}",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._ocr_pool:
            self._ocr_pool.shutdown(wait=True)
            self._ocr_pool = None
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
        for i, part in enumerate(
            [msg[j:j+55] for j in range(0, min(len(msg), 165), 55)]
        ):
            cv2.putText(frame, part, (20, h // 2 + 10 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200),
                        1, cv2.LINE_AA)
        cv2.putText(frame, "Verifica la ruta o permisos de la fuente",
                    (20, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (120, 120, 120), 1, cv2.LINE_AA)
        return frame

    def _run(self) -> None:
        self.model = YOLO(self.model_path)
        self.model.to(get_device())

        if self.plate_model_path:
            try:
                self.plate_model = YOLO(self.plate_model_path)
                self.plate_model.to(get_device())
            except Exception:
                self.plate_model = None

        _get_easyocr_reader()

        try:
            src = int(self.source_path)
        except (ValueError, TypeError):
            src = self.source_path

        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            err = self._make_error_frame(
                f"No se puede abrir: {self.source_path}")
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

        for box in boxes:
            if box.id is None:
                continue
            tid = int(box.id[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            color = CYAN
            direction_label = ""
            if tid in self._counted:
                direction_label = next(
                    (rg.direction for rg in reversed(self.registros)
                     if rg.track_id == tid), "")
                color = GREEN if direction_label == "in" else RED

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"V-{tid} {direction_label}"
            ty = y1 - 8 if y1 > 20 else y2 + 18
            cv2.putText(annotated, label, (x1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2,
                        cv2.LINE_AA)

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
                    hf, wf = frame.shape[:2]
                    pad_x = int((x2 - x1) * 0.3)
                    pad_y = int((y2 - y1) * 0.3)
                    cx1 = max(0, x1 - pad_x)
                    cy1 = max(0, y1 - pad_y)
                    cx2 = min(wf, x2 + pad_x)
                    cy2 = min(hf, y2 + pad_y)
                    crop = frame[cy1:cy2, cx1:cx2]
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fname = f"veh_{self.source_id}_{tid}_{ts}.jpg"
                    path = os.path.join(CAPTURES_DIR, fname)
                    cv2.imwrite(path, crop,
                                [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])

                    item = {
                        "track_id": tid,
                        "crop": crop.copy(),
                        "direction": direction or "",
                        "fname": fname,
                        "ts": ts,
                        "time_str": datetime.now().strftime("%H:%M:%S"),
                    }
                    if self._ocr_pool:
                        try:
                            future = self._ocr_pool.submit(
                                self._process_ocr_item, item)
                            future.add_done_callback(self._on_ocr_result)
                        except RuntimeError:
                            pass

        cv2.line(annotated, draw_p1, draw_p2, YELLOW, 2)
        cv2.putText(annotated,
                    f"IN {self.total_in}  OUT {self.total_out}",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2,
                    cv2.LINE_AA)
        cv2.putText(annotated,
                    f"Vehiculos: {self.total_vehicles}",
                    (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2,
                    cv2.LINE_AA)
        return annotated

    def _detect_plate(self, crop: np.ndarray) -> tuple[str, Optional[np.ndarray]]:
        if self.plate_model and crop.size > 0:
            try:
                plate_results = self.plate_model(
                    crop, conf=self.plate_conf_thresh,
                    verbose=False, max_det=5)
                for pr in plate_results:
                    pb = pr.boxes
                    if pb is None:
                        continue
                    for pbox in pb:
                        px1, py1, px2, py2 = map(int, pbox.xyxy[0])
                        px1 = max(0, px1)
                        py1 = max(0, py1)
                        px2 = min(crop.shape[1], px2)
                        py2 = min(crop.shape[0], py2)
                        plate_roi = crop[py1:py2, px1:px2]
                        if plate_roi.size > 0:
                            text = _ocr_combined(plate_roi)
                            if text and len(text) >= 3:
                                return text, plate_roi
            except Exception:
                pass
        return _detect_plate_cv2(crop)

    def _process_ocr_item(self, item: dict) -> dict:
        tid = item["track_id"]
        crop = item["crop"]
        plate_text = ""
        plate_crop_file = ""

        if self.plate_detection_enabled and crop.size > 0:
            plate_text, plate_roi = self._detect_plate(crop)
            if plate_text and plate_roi is not None and plate_roi.size > 0:
                plate_fname = (
                    f"plate_{self.source_id}_{tid}_{item['ts']}.jpg")
                plate_path = os.path.join(CAPTURES_DIR, plate_fname)
                cv2.imwrite(plate_path, plate_roi,
                            [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
                plate_crop_file = plate_fname

        reg = VehiculoRegistro(
            track_id=tid,
            timestamp=item["time_str"],
            image_file=item["fname"],
            direction=item["direction"],
            plate_text=plate_text,
            plate_crop=plate_crop_file,
        )
        result = {"reg": reg, "plate_text": plate_text}
        if plate_text:
            result["plate_data"] = {
                "track_id": tid,
                "timestamp": item["ts"],
                "plate": plate_text,
                "image": plate_crop_file,
            }
        return result

    def _on_ocr_result(self, future) -> None:
        try:
            result = future.result()
        except Exception:
            return
        if not result:
            return
        with self._lock:
            self.registros.append(result["reg"])
            if len(self.registros) > self._max_registros:
                old = self.registros.pop(0)
                old_path = os.path.join(CAPTURES_DIR, old.image_file)
                if os.path.exists(old_path):
                    os.remove(old_path)
            if result.get("plate_text"):
                self.plates_count += 1
                self.placas_detectadas.append(result["plate_data"])

    def get_frame_jpeg(self) -> Optional[bytes]:
        with self._lock:
            frame = self._frame
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
        return buf.tobytes() if ok else None

    def get_stats(self) -> dict:
        with self._lock:
            regs = [{
                "track_id": r.track_id,
                "timestamp": r.timestamp,
                "image": r.image_file,
                "direction": r.direction,
                "plate_text": r.plate_text,
                "plate_crop": r.plate_crop,
            } for r in self.registros]
            placas = list(self.placas_detectadas)
        return {
            "source_id": self.source_id,
            "total_in": self.total_in,
            "total_out": self.total_out,
            "total_vehicles": self.total_vehicles,
            "plates_count": self.plates_count,
            "registros": regs,
            "placas": placas,
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
        with self._lock:
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
              line_mode: str = "horizontal", line_pos: int = 50,
              fps_limit: float = 0.0) -> None:
        self.stop_all()
        with self._lock:
            p = VehiculosPipeline(
                source_id, source_path, func_state.copy(),
                conf_thresh, half, model_path,
                plate_model_path=plate_model_path,
                plate_conf_thresh=plate_conf_thresh,
                classes=classes,
                line_mode=line_mode, line_pos=line_pos,
                fps_limit=fps_limit,
            )
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

    def set_plate_detection(self, source_id: int,
                            enabled: bool) -> None:
        with self._lock:
            p = self.pipelines.get(source_id)
        if p:
            p.set_plate_detection(enabled)
