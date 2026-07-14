from __future__ import annotations

import cv2
import os
import time
import threading
from datetime import datetime
import numpy as np
from typing import Optional, Dict
from ultralytics import YOLO

from src.config import BASE_DIR
from src.utils import get_device
from src.modules.base import multi_acquire, multi_release, is_multi_enabled

MODEL_NAME  = "yolo11n.pt"
CONF_THRESH = 0.45
IOU_THRESH  = 0.50
JPEG_Q      = 72

PURPLE = (200, 0, 200)
YELLOW = (0, 255, 255)
WHITE  = (255, 255, 255)
GREEN  = (0, 255, 0)
RED    = (0, 0, 255)
CYAN   = (255, 255, 0)

SHORT_WINDOW_SIZE = 15
LONG_WINDOW_SIZE  = 30
MIN_FRAMES_CLASSIFY = 10
STABILITY_WINDOW    = 8
CONFIDENCE_THRESH   = 0.75
MIN_FRAMES_CONFIDENCE = 15
BOOT_SEARCH_EXPANSION = 1.8
REVERIFY_THRESHOLD    = 0.60
CLASSIFY_THRESHOLD    = 0.65
REVERIFY_SECONDS      = 5.0

CAPTURES_DIR = os.path.join(
    BASE_DIR, "static", "uploads", "captures", "reglamento",
)


class AreaPipeline:
    def __init__(self, source_id: int, source_path: str, func_state: dict,
                 conf_thresh: float = CONF_THRESH, half: bool = False,
                 model_path: str = None, min_time: int = 10,
                 area_x1: int = 30, area_y1: int = 30,
                 area_x2: int = 70, area_y2: int = 70,
                 fps_limit: float = 0.0):
        self.source_id   = source_id
        self.source_path = source_path
        self.func_state  = func_state
        self.conf_thresh = conf_thresh
        self.half        = half
        self.model_path  = model_path or MODEL_NAME
        self.min_time    = min_time
        self.area_x1     = area_x1
        self.area_y1     = area_y1
        self.area_x2     = area_x2
        self.area_y2     = area_y2
        self.fps_limit   = fps_limit

        self.model = None
        self._frame: Optional[np.ndarray] = None
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.total_con_botas      = 0
        self.total_sin_botas      = 0
        self.total_cumplimientos  = 0
        self.total_incumplimientos = 0
        self._h = 0
        self._w = 0

        self._person_boot_status  = {}
        self._person_frames       = {}
        self._obs_short           = {}
        self._obs_long            = {}
        self._reverify_done       = {}
        self._status_before_rev   = {}
        self._entry_time          = {}
        self._seconds_in_area     = {}
        self._counted             = set()
        self._alerts_sent         = set()
        self._exited              = set()

        self.evidencias = []

        os.makedirs(CAPTURES_DIR, exist_ok=True)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True,
            name=f"reglamento-pipe-{self.source_id}",
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

    def _find_boots_for_person(self, px1, py1, px2, py2, boots, area_coords):
        pw = px2 - px1
        ph = py2 - py1
        exp = BOOT_SEARCH_EXPANSION
        ex1 = int(px1 - pw * (exp - 1) / 2)
        ey1 = int(py1 - ph * (exp - 1) / 2)
        ex2 = int(px2 + pw * (exp - 1) / 2)
        ey2 = int(py2 + ph * (exp - 1) / 2)

        area_x1, area_y1 = area_coords[0]
        area_x2, area_y2 = area_coords[2]

        boots_in_person = 0
        boots_in_area   = 0
        boots_near      = 0
        for b in boots:
            bx1, by1, bx2, by2 = b['bbox']
            bcx = (bx1 + bx2) / 2
            bcy = (by1 + by2) / 2

            if bx1 >= ex1 and by1 >= ey1 and bx2 <= ex2 and by2 <= ey2:
                boots_in_person += 1
            if area_x1 <= bcx <= area_x2 and area_y1 <= bcy <= area_y2:
                boots_in_area += 1
            dist = max(0, abs(bcx - (px1 + px2) / 2) - pw / 2,
                            abs(bcy - (py1 + py2) / 2) - ph / 2)
            if dist <= max(pw, ph) * 0.3:
                boots_near += 1

        return min(boots_in_person, 2), min(boots_in_area, 2), min(boots_near, 2)

    def _classify_person(self, track_id, boots_in_area, boots_near,
                          person_bbox, all_persons, area_coords):
        if track_id not in self._person_frames:
            self._person_frames[track_id] = 0
            self._obs_short[track_id] = []
            self._obs_long[track_id] = []

        self._person_frames[track_id] += 1
        total_boots = boots_in_area + boots_near

        self._obs_short[track_id].append(total_boots)
        self._obs_long[track_id].append(total_boots)

        if len(self._obs_short[track_id]) > SHORT_WINDOW_SIZE:
            self._obs_short[track_id] = self._obs_short[track_id][-SHORT_WINDOW_SIZE:]
        if len(self._obs_long[track_id]) > LONG_WINDOW_SIZE:
            self._obs_long[track_id] = self._obs_long[track_id][-LONG_WINDOW_SIZE:]

        if self._person_frames[track_id] < MIN_FRAMES_CLASSIFY:
            return False

        obs = self._obs_long[track_id]
        window = obs[-STABILITY_WINDOW:] if len(obs) >= STABILITY_WINDOW else obs
        unique_vals = len(set(window))
        avg_boots = sum(window) / len(window) if window else 0

        if len(obs) >= MIN_FRAMES_CONFIDENCE:
            cw = obs[-MIN_FRAMES_CONFIDENCE:]
            c_avg = sum(cw) / len(cw)
            target = round(c_avg)
            consistent = sum(1 for v in cw if abs(v - target) <= 1)
            ratio = consistent / len(cw)
            if ratio >= CONFIDENCE_THRESH:
                return True
            return False

        stable = unique_vals <= (3 if len(window) >= 10 else 2)
        has_b = sum(1 for v in window if v >= 1)
        no_b  = sum(1 for v in window if v == 0)
        total = len(window)
        clear_trend = (has_b / total >= 0.7) or (no_b / total >= 0.7)
        return stable and clear_trend

    def _calc_dual_avg(self, track_id):
        short = self._obs_short.get(track_id, [])
        long  = self._obs_long.get(track_id, [])
        avg_s = sum(short) / len(short) if short else 0.0
        avg_l = sum(long) / len(long) if long else 0.0
        return max(avg_s, avg_l), avg_s, avg_l

    def _save_evidence(self, frame, track_id, boot_status, seconds):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        fname = f"evidencia_{self.source_id}_{track_id}_{ts}.jpg"
        fpath = os.path.join(CAPTURES_DIR, fname)
        cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        entry = {
            "track_id": track_id,
            "boot_status": boot_status,
            "seconds": round(seconds, 1),
            "capture_path": f"/static/uploads/captures/reglamento/{fname}",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "date": datetime.now().strftime("%d/%m/%Y"),
        }
        self.evidencias.insert(0, entry)
        if len(self.evidencias) > 100:
            self.evidencias = self.evidencias[:100]
        return fpath

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
        area_mid_y = (ry1 + ry2) // 2
        area_coords = [(rx1, ry1), (rx2, ry1), (rx2, ry2), (rx1, ry2)]

        results = self.model.track(
            frame, persist=True, conf=self.conf_thresh,
            iou=IOU_THRESH, half=self.half, verbose=False,
            tracker="bytetrack.yaml",
        )

        annotated = frame.copy()
        r = results[0]
        boxes = r.boxes if r.boxes is not None else []
        current_tids = set()

        persons = []
        boots   = []
        for box in boxes:
            if box.id is None:
                continue
            tid = int(box.id[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            entry = {'bbox': [x1, y1, x2, y2], 'conf': conf, 'track_id': tid}
            if cls == 0:
                persons.append(entry)
            elif cls == 1:
                boots.append(entry)

        for p in persons:
            px1, py1, px2, py2 = p['bbox']
            tid = p['track_id']
            current_tids.add(tid)

            boots_in_p, boots_in_a, boots_near = self._find_boots_for_person(
                px1, py1, px2, py2, boots, area_coords,
            )

            person_bottom = py2
            is_inside_area = (rx1 <= (px1 + px2) / 2 <= rx2 and
                              person_bottom >= area_mid_y)

            now = time.monotonic()

            if is_inside_area:
                if tid not in self._entry_time:
                    self._entry_time[tid] = now
                elapsed = now - self._entry_time[tid]
                self._seconds_in_area[tid] = elapsed

                if (tid in self._person_boot_status and
                    tid in self._seconds_in_area and
                    self._seconds_in_area[tid] >= REVERIFY_SECONDS and
                    tid not in self._reverify_done and
                    self._person_boot_status[tid] == 'sin_botas'):

                    prev = self._person_boot_status[tid]
                    self._status_before_rev[tid] = prev
                    can = self._classify_person(tid, boots_in_a, boots_near,
                                                 [px1, py1, px2, py2],
                                                 persons, area_coords)
                    if can:
                        avg_f, _, _ = self._calc_dual_avg(tid)
                        if avg_f >= REVERIFY_THRESHOLD:
                            self._person_boot_status[tid] = 'con_botas'
                    self._reverify_done[tid] = True

                if (tid not in self._person_boot_status and
                    tid not in self._exited):
                    can = self._classify_person(tid, boots_in_a, boots_near,
                                                 [px1, py1, px2, py2],
                                                 persons, area_coords)
                    if can:
                        avg_f, _, _ = self._calc_dual_avg(tid)
                        if avg_f >= CLASSIFY_THRESHOLD:
                            self._person_boot_status[tid] = 'con_botas'
                        else:
                            self._person_boot_status[tid] = 'sin_botas'

            exit_through_bottom = person_bottom >= ry2

            if exit_through_bottom and tid in self._entry_time and tid not in self._counted:
                elapsed = self._seconds_in_area.get(tid, 0)
                boot_st = self._person_boot_status.get(tid, 'sin_determinar')

                compliance = 'incumplio'
                if boot_st == 'con_botas' and elapsed >= self.min_time:
                    compliance = 'cumplio'

                if compliance == 'cumplio':
                    self.total_cumplimientos += 1
                else:
                    self.total_incumplimientos += 1
                    self._save_evidence(annotated, tid, boot_st, elapsed)

                if boot_st == 'con_botas':
                    self.total_con_botas += 1
                elif boot_st == 'sin_botas':
                    self.total_sin_botas += 1

                self._counted.add(tid)
                self._exited.add(tid)

                from src.database import insert_reglamento_detection
                try:
                    insert_reglamento_detection(
                        source_id=self.source_id,
                        track_id=tid,
                        boot_status=boot_st,
                        time_compliance=compliance,
                        seconds_in_area=elapsed,
                    )
                except Exception:
                    pass

            boot_st = self._person_boot_status.get(tid, None)
            seconds = self._seconds_in_area.get(tid, 0)

            if boot_st == 'con_botas':
                color = GREEN
                label = f"ID[{tid}] CON BOTAS {seconds:.1f}s"
            elif boot_st == 'sin_botas':
                color = RED
                label = f"ID[{tid}] SIN BOTAS {seconds:.1f}s"
            elif boot_st is None and tid in self._entry_time:
                color = CYAN
                label = f"ID[{tid}] LOADING {seconds:.1f}s"
            else:
                color = CYAN
                label = f"ID[{tid}]"

            cv2.rectangle(annotated, (px1, py1), (px2, py2), color, 2)
            ty = py1 - 10 if py1 > 24 else py2 + 20
            cv2.putText(annotated, label, (px1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)

        cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), PURPLE, 2)
        mid_x = (rx1 + rx2) // 2
        cv2.line(annotated, (rx1, area_mid_y), (rx2, area_mid_y), PURPLE, 1,
                 cv2.LINE_AA)
        cv2.putText(annotated, "Area de analisis", (rx1 + 4, max(ry1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, PURPLE, 1, cv2.LINE_AA)

        ox = 12
        oy = 30
        lh = 20
        overlay = annotated.copy()
        cv2.rectangle(overlay, (ox - 6, oy - 22),
                      (ox + 240, oy + 4 * lh + 4), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0, annotated)

        items = [
            (f"CON BOTAS: {self.total_con_botas}", GREEN),
            (f"SIN BOTAS: {self.total_sin_botas}", RED),
            (f"CUMPLIMIENTO: {self.total_cumplimientos}", GREEN),
            (f"INCUMPLIMIENTO: {self.total_incumplimientos}", RED),
        ]
        for i, (txt, clr) in enumerate(items):
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
        return {
            "source_id":          self.source_id,
            "con_botas":          self.total_con_botas,
            "sin_botas":          self.total_sin_botas,
            "cumplimientos":      self.total_cumplimientos,
            "incumplimientos":    self.total_incumplimientos,
            "evidencias":         self.evidencias[:20],
            "area_x1":            self.area_x1,
            "area_y1":            self.area_y1,
            "area_x2":            self.area_x2,
            "area_y2":            self.area_y2,
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

    def set_min_time(self, t: int) -> None:
        self.min_time = max(1, t)

    def reset(self) -> None:
        self.total_con_botas      = 0
        self.total_sin_botas      = 0
        self.total_cumplimientos  = 0
        self.total_incumplimientos = 0
        self._person_boot_status.clear()
        self._person_frames.clear()
        self._obs_short.clear()
        self._obs_long.clear()
        self._reverify_done.clear()
        self._status_before_rev.clear()
        self._entry_time.clear()
        self._seconds_in_area.clear()
        self._counted.clear()
        self._alerts_sent.clear()
        self._exited.clear()
        self.evidencias.clear()


class ReglamentoManager:
    _instance: Optional["ReglamentoManager"] = None
    _class_lock = threading.Lock()

    def __init__(self) -> None:
        self.pipelines: Dict[int, AreaPipeline] = {}
        self._lock = threading.Lock()

    @classmethod
    def get(cls) -> "ReglamentoManager":
        with cls._class_lock:
            if cls._instance is None:
                cls._instance = ReglamentoManager()
        return cls._instance

    def start(self, source_id: int, source_path: str, func_state: dict,
              conf_thresh: float = CONF_THRESH, half: bool = False,
              model_path: str = None, min_time: int = 10,
              area_x1: int = 30, area_y1: int = 30,
              area_x2: int = 70, area_y2: int = 70,
              fps_limit: float = 0.0) -> None:
        if not multi_acquire():
            raise RuntimeError("Límite de 4 reproducciones simultáneas alcanzado")
        if not is_multi_enabled():
            self.stop_all()
        with self._lock:
            p = AreaPipeline(source_id, source_path, func_state.copy(),
                                   conf_thresh, half, model_path, min_time,
                                   area_x1, area_y1, area_x2, area_y2,
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

    def set_area(self, source_id: int, x1: int, y1: int,
                 x2: int, y2: int) -> None:
        with self._lock:
            p = self.pipelines.get(source_id)
        if p:
            p.set_area(x1, y1, x2, y2)

    def set_min_time(self, source_id: int, t: int) -> None:
        with self._lock:
            p = self.pipelines.get(source_id)
        if p:
            p.set_min_time(t)

    def reset(self, source_id: int) -> None:
        with self._lock:
            p = self.pipelines.get(source_id)
        if p:
            p.reset()
