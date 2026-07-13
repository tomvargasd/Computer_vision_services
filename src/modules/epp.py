from __future__ import annotations

import cv2
import os
import time
import threading
from datetime import datetime
import numpy as np
from typing import Optional, Dict, List, Set
from collections import defaultdict
from ultralytics import YOLO

from src.utils import get_device

MODEL_NAME  = "yolo11n.pt"
CONF_THRESH = 0.35
IOU_THRESH  = 0.50
JPEG_Q      = 80

EPP_CLASS_MAP = {
    0: "boots",
    1: "earmuffs",
    2: "glasses",
    3: "gloves",
    4: "helmet",
    6: "vest",
}
EPP_CLASS_IDS = set(EPP_CLASS_MAP.keys())
PERSON_CLS = 5

GREEN  = (0, 255, 0)
RED    = (0, 0, 255)
YELLOW = (0, 255, 255)
CYAN   = (255, 255, 0)
WHITE  = (255, 255, 255)

WINDOW_DURATION = 1.0
MAJORITY_RATIO  = 0.50
CONFIRM_CHECKS  = 3

CAPTURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "static", "uploads", "captures", "epp",
)


class EppPipeline:
    def __init__(self, source_id: int, source_path: str, func_state: dict,
                 conf_thresh: float = CONF_THRESH, half: bool = False,
                 model_path: str = None):
        self.source_id   = source_id
        self.source_path = source_path
        self.func_state  = func_state
        self.conf_thresh = conf_thresh
        self.half        = half
        self.model_path  = model_path or MODEL_NAME

        self.model: Optional[YOLO] = None
        self._frame: Optional[np.ndarray] = None
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._h = 0
        self._w = 0

        self.total_protected   = 0
        self.total_unprotected = 0
        self._protected_tids: Set[int] = set()

        self._person_epp: Dict[int, Dict[int, List[bool]]] = {}
        self._person_status: Dict[int, str] = {}
        self._person_finalized: Dict[int, bool] = {}
        self._person_windows: Dict[int, List[bool]] = {}
        self._person_window_frames: Dict[int, int] = {}
        self._person_window_start: Dict[int, float] = {}
        self._alerts_sent: Set[int] = set()
        self._epp_counts: Dict[str, int] = defaultdict(int)
        self._last_switch_version = 0
        self._current_switch_version = 0
        self._evidencias: List[dict] = []

        self.master_enabled = True
        self.required_epp: Set[int] = {4, 6}

        os.makedirs(CAPTURES_DIR, exist_ok=True)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True,
            name=f"epp-pipe-{self.source_id}",
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

    def _epp_items_for_person(self, px1, py1, px2, py2, epp_boxes) -> Dict[int, bool]:
        found: Dict[int, bool] = {}
        for epp_cls in EPP_CLASS_IDS:
            found[epp_cls] = False
        for box in epp_boxes:
            cls   = int(box.cls[0])
            if cls not in EPP_CLASS_IDS:
                continue
            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
            ix1 = max(px1, bx1); iy1 = max(py1, by1)
            ix2 = min(px2, bx2); iy2 = min(py2, by2)
            if ix1 < ix2 and iy1 < iy2:
                p_area = (px2 - px1) * (py2 - py1)
                i_area = (ix2 - ix1) * (iy2 - iy1)
                ratio = i_area / p_area if p_area > 0 else 0
                if ratio >= 0.05:
                    found[cls] = True
        return found

    def _evaluate_window(self, track_id: int) -> Optional[bool]:
        obs = self._person_epp.get(track_id)
        if obs is None:
            return None
        window_frames = self._person_window_frames.get(track_id, 0)
        if window_frames < 5:
            return None
        results = {}
        for epp_cls in self.required_epp:
            history = obs.get(epp_cls, [])
            window = history[-window_frames:] if len(history) >= window_frames else history
            if not window:
                results[epp_cls] = False
                continue
            detected_ratio = sum(window) / len(window)
            results[epp_cls] = detected_ratio > MAJORITY_RATIO
        if all(results.values()):
            return True
        return False

    def _save_evidence(self, frame, track_id, status):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        fname = f"epp_alerta_{self.source_id}_{track_id}_{ts}.jpg"
        fpath = os.path.join(CAPTURES_DIR, fname)
        cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        entry = {
            "track_id": track_id,
            "status": status,
            "capture_path": f"/static/uploads/captures/epp/{fname}",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "date": datetime.now().strftime("%d/%m/%Y"),
        }
        self._evidencias.insert(0, entry)
        if len(self._evidencias) > 50:
            self._evidencias = self._evidencias[:50]

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

        persons = []
        epp_items = []
        for box in boxes:
            cls = int(box.cls[0])
            if cls == PERSON_CLS:
                persons.append(box)
            elif cls in EPP_CLASS_IDS:
                epp_items.append(box)

        if self._current_switch_version != self._last_switch_version:
            self._person_status.clear()
            self._person_finalized.clear()
            self._person_epp.clear()
            self._person_windows.clear()
            self._person_window_frames.clear()
            self._person_window_start.clear()
            self._last_switch_version = self._current_switch_version

        current_tids: Set[int] = set()
        for pbox in persons:
            if pbox.id is None:
                continue
            tid = int(pbox.id[0])
            current_tids.add(tid)
            px1, py1, px2, py2 = map(int, pbox.xyxy[0])

            if tid not in self._person_epp:
                self._person_epp[tid] = {c: [] for c in EPP_CLASS_IDS}
                self._person_status[tid] = "analyzing"
                self._person_finalized[tid] = False
                self._person_window_frames[tid] = 0
                self._person_window_start[tid] = time.monotonic()

            items = self._epp_items_for_person(px1, py1, px2, py2, epp_items)
            for epp_cls in EPP_CLASS_IDS:
                self._person_epp[tid][epp_cls].append(items.get(epp_cls, False))
            self._person_window_frames[tid] += 1

            if not self.master_enabled:
                self._person_status[tid] = "tracking"
                status_label = f"ID[{tid}]"
                color = CYAN
            elif self._person_finalized.get(tid):
                status_label = f"ID[{tid}] {self._person_status[tid].upper()}"
                color = GREEN if self._person_status[tid] == "protected" else RED
            else:
                elapsed = time.monotonic() - self._person_window_start[tid]
                if elapsed >= WINDOW_DURATION:
                    verdict = self._evaluate_window(tid)
                    if verdict is True:
                        self._person_windows.setdefault(tid, []).append(True)
                    elif verdict is False:
                        self._person_windows.setdefault(tid, []).append(False)

                    wins = self._person_windows.get(tid, [])
                    if len(wins) >= CONFIRM_CHECKS:
                        if any(wins):
                            self._person_status[tid] = "protected"
                            self._person_finalized[tid] = True
                            self._protected_tids.add(tid)
                            self.total_protected += 1
                            for epp_cls in EPP_CLASS_IDS:
                                if self._person_epp[tid].get(epp_cls) and any(self._person_epp[tid][epp_cls]):
                                    epp_name = EPP_CLASS_MAP[epp_cls]
                                    self._epp_counts[epp_name] += 1
                        else:
                            self._person_status[tid] = "unprotected"
                            self._person_finalized[tid] = True
                            self.total_unprotected += 1
                            if tid not in self._alerts_sent:
                                self._alerts_sent.add(tid)
                                self._save_evidence(annotated, tid, "sin_epp")
                    else:
                        self._person_window_frames[tid] = 0
                        self._person_window_start[tid] = time.monotonic()

                color_map = {"protected": GREEN, "unprotected": RED, "analyzing": YELLOW}
                color = color_map.get(self._person_status.get(tid, "analyzing"), CYAN)
                status_label = f"ID[{tid}] {self._person_status.get(tid, 'analyzing').upper()}"
                if not self._person_finalized.get(tid):
                    window_progress = min(len(self._person_windows.get(tid, [])) + 1, CONFIRM_CHECKS)
                    status_label += f" [{window_progress}/{CONFIRM_CHECKS}]"

            cv2.rectangle(annotated, (px1, py1), (px2, py2), color, 2)
            ty = py1 - 10 if py1 > 24 else py2 + 20
            cv2.putText(annotated, status_label, (px1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)

        for ebox in epp_items:
            cls = int(ebox.cls[0])
            if ebox.id is None:
                continue
            ex1, ey1, ex2, ey2 = map(int, ebox.xyxy[0])
            conf = float(ebox.conf[0])
            epp_name = EPP_CLASS_MAP.get(cls, f"cls_{cls}")
            cv2.rectangle(annotated, (ex1, ey1), (ex2, ey2), WHITE, 1)
            cv2.putText(annotated, f"{epp_name} {conf:.2f}", (ex1, ey1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, WHITE, 1, cv2.LINE_AA)

        ox, oy = 12, 30
        lh = 20
        overlay = annotated.copy()
        lines = []
        if self.master_enabled:
            lines.append((f"PROTEGIDOS: {self.total_protected}", GREEN))
            lines.append((f"SIN EPP: {self.total_unprotected}", RED))
        else:
            lines.append(("MODO: SOLO DETECCION", CYAN))
            lines.append((f"PERSONAS: {len(persons)}", WHITE))
        lines.append((f"EPP: {', '.join(EPP_CLASS_MAP.get(c, str(c)) for c in sorted(self.required_epp))}", WHITE))

        h2 = len(lines) * lh + 10
        cv2.rectangle(overlay, (ox - 6, oy - 22), (ox + 260, oy + h2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0, annotated)

        for i, (txt, clr) in enumerate(lines):
            cv2.putText(annotated, txt, (ox, oy + i * lh),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, clr, 1, cv2.LINE_AA)

        return annotated

    def get_frame_jpeg(self) -> Optional[bytes]:
        with self._lock:
            frame = self._frame
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
        return buf.tobytes() if ok else None

    def get_stats(self) -> dict:
        epp_ranking = sorted(self._epp_counts.items(), key=lambda x: -x[1])
        return {
            "source_id":         self.source_id,
            "protected":         self.total_protected,
            "unprotected":       self.total_unprotected,
            "epp_ranking":       epp_ranking,
            "evidencias":        self._evidencias[:20],
            "master_enabled":    self.master_enabled,
            "required_epp":      sorted(self.required_epp),
            "active_persons":    len(self._person_status),
        }

    def set_master(self, enabled: bool) -> None:
        self.master_enabled = enabled
        if not enabled:
            self._person_status.clear()
            self._person_finalized.clear()

    def set_required_epp(self, classes: Set[int]) -> None:
        self.required_epp = classes & EPP_CLASS_IDS
        self._current_switch_version += 1

    def reset(self) -> None:
        self.total_protected   = 0
        self.total_unprotected = 0
        self._protected_tids.clear()
        self._person_epp.clear()
        self._person_status.clear()
        self._person_finalized.clear()
        self._person_windows.clear()
        self._person_window_frames.clear()
        self._person_window_start.clear()
        self._alerts_sent.clear()
        self._epp_counts.clear()
        self._evidencias.clear()


class EppManager:
    _instance: Optional["EppManager"] = None
    _class_lock = threading.Lock()

    def __init__(self) -> None:
        self.pipelines: Dict[int, EppPipeline] = {}
        self._lock = threading.Lock()

    @classmethod
    def get(cls) -> "EppManager":
        with cls._class_lock:
            if cls._instance is None:
                cls._instance = EppManager()
        return cls._instance

    def start(self, source_id: int, source_path: str, func_state: dict,
              conf_thresh: float = CONF_THRESH, half: bool = False,
              model_path: str = None) -> None:
        self.stop_all()
        with self._lock:
            p = EppPipeline(source_id, source_path, func_state.copy(),
                            conf_thresh, half, model_path)
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

    def set_master(self, source_id: int, enabled: bool) -> None:
        with self._lock:
            p = self.pipelines.get(source_id)
        if p:
            p.set_master(enabled)

    def set_required_epp(self, source_id: int, classes: Set[int]) -> None:
        with self._lock:
            p = self.pipelines.get(source_id)
        if p:
            p.set_required_epp(classes)

    def reset(self, source_id: int) -> None:
        with self._lock:
            p = self.pipelines.get(source_id)
        if p:
            p.reset()
