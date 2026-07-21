from __future__ import annotations

import cv2
import os
import time
import threading
from datetime import datetime
import numpy as np
from typing import Optional, Dict, List, Tuple
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
ORANGE = (0, 165, 255)

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


def _point_in_polygon(pt: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    x, y = pt
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


class AreaPipeline:
    def __init__(self, source_id: int, source_path: str, func_state: dict,
                 conf_thresh: float = CONF_THRESH, half: bool = False,
                 model_path: str = None, min_time: int = 10,
                 area_x1: int = 30, area_y1: int = 30,
                 area_x2: int = 70, area_y2: int = 70,
                 line_mode: str = "rectangle", line_pos: int = 50,
                 inverted: bool = False, jpeg_q: int = JPEG_Q,
                 max_dim: int = 0, frame_step: int = 1,
                 custom_rect: Optional[List[Tuple[float, float]]] = None,
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
        self.line_mode   = line_mode
        self.line_pos    = line_pos
        self.inverted    = inverted
        self.jpeg_q      = jpeg_q
        self.max_dim     = max_dim
        self.frame_step  = max(1, frame_step)
        self._rect_points = custom_rect or [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
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

        self.total_in        = 0
        self.total_out       = 0
        self.current_in_area = 0

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

        self._track_area_state: Dict[int, dict] = {}
        self._prev_pos: Dict[int, Tuple[float, float]] = {}
        self._counted_tracks: set = set()
        self._person_first_seen: Dict[int, float] = {}
        self._pipeline_start_time: Optional[float] = None
        self._warmup_seconds = 2.0

        self._frame_count = 0

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
        self._frame_count = 0
        while not self._stop.is_set() and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if isinstance(src, str) and "://" not in src:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            if first_frame:
                self._h, self._w = frame.shape[:2]
                if self.max_dim > 0:
                    scale = min(self.max_dim / self._w, self.max_dim / self._h, 1.0)
                    if scale < 1.0:
                        new_w = int(self._w * scale)
                        new_h = int(self._h * scale)
                        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                        self._w, self._h = new_w, new_h
                first_frame = False

            self._frame_count += 1
            if self._frame_count % self.frame_step != 0:
                with self._lock:
                    pass
                time.sleep(0.001)
                continue

            annotated = self._process(frame)
            with self._lock:
                self._frame = annotated
            time.sleep(self.fps_limit)

        cap.release()

    def _process(self, frame: np.ndarray) -> np.ndarray:
        h, w = self._h, self._w
        is_area_mode = self.line_mode in ("rectangle", "custom_rect")

        if is_area_mode:
            if self.line_mode == "rectangle":
                rx1 = int(w * self.area_x1 / 100)
                ry1 = int(h * self.area_y1 / 100)
                rx2 = int(w * self.area_x2 / 100)
                ry2 = int(h * self.area_y2 / 100)
                area_coords = [(rx1, ry1), (rx2, ry1), (rx2, ry2), (rx1, ry2)]
                poly_pts = area_coords
                area_ok = True
            else:
                poly_pts = [(int(w * px), int(h * py)) for px, py in self._rect_points]
                area_ok = len(poly_pts) >= 3
        else:
            line_pos = int((h if self.line_mode == "horizontal" else w) * self.line_pos / 100)

        results = self.model.track(
            frame, persist=True, conf=self.conf_thresh,
            iou=IOU_THRESH, half=self.half, verbose=False,
            tracker="bytetrack.yaml",
        )

        annotated = frame.copy()
        r = results[0]
        boxes = r.boxes if r.boxes is not None else []
        current_tids = set()
        now = time.monotonic()

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

        if self._pipeline_start_time is None:
            self._pipeline_start_time = now

        for p in persons:
            px1, py1, px2, py2 = p['bbox']
            tid = p['track_id']
            cx = (px1 + px2) // 2
            cy = (py1 + py2) // 2
            person_bottom = py2
            current_tids.add(tid)

            if tid not in self._person_first_seen:
                self._person_first_seen[tid] = now
            time_since_first_seen = now - self._person_first_seen[tid]
            is_warmup = (now - self._pipeline_start_time) < self._warmup_seconds

            if is_area_mode and area_ok:
                if self.line_mode == "rectangle":
                    inside = rx1 <= cx <= rx2 and ry1 <= cy <= ry2
                else:
                    inside = _point_in_polygon((cx, cy), poly_pts)

                if tid not in self._track_area_state:
                    self._track_area_state[tid] = {"counted_in": False}
                st = self._track_area_state[tid]

                if inside and not st["counted_in"]:
                    is_legacy = (
                        is_warmup and
                        time_since_first_seen < self._warmup_seconds and
                        tid not in self._prev_pos
                    )
                    if not is_legacy:
                        self.total_in += 1
                    st["counted_in"] = True
                elif not inside and st["counted_in"]:
                    self.total_out += 1
                    st["counted_in"] = False

                was_in_area = tid in self._entry_time

                if inside:
                    if not was_in_area:
                        self._entry_time[tid] = now
                    elapsed = now - self._entry_time[tid]
                    self._seconds_in_area[tid] = elapsed

                    if self.line_mode == "rectangle":
                        boot_area_coords = area_coords
                    else:
                        xs = [p[0] for p in poly_pts]
                        ys = [p[1] for p in poly_pts]
                        bmin_x, bmax_x = min(xs), max(xs)
                        bmin_y, bmax_y = min(ys), max(ys)
                        boot_area_coords = [(bmin_x, bmin_y), (bmax_x, bmin_y), (bmax_x, bmax_y), (bmin_x, bmax_y)]

                    boots_in_p, boots_in_a, boots_near = self._find_boots_for_person(
                        px1, py1, px2, py2, boots, boot_area_coords,
                    )

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

                if was_in_area and not inside and tid not in self._counted:
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

                if not inside and tid in self._entry_time:
                    self._entry_time.pop(tid, None)
                    self._seconds_in_area.pop(tid, None)

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

            else:
                prev = self._prev_pos.get(tid)
                if prev is not None and tid not in self._counted_tracks:
                    if self.line_mode == "horizontal":
                        if prev[1] < line_pos <= cy:
                            self.total_in += 1
                            self._counted_tracks.add(tid)
                        elif prev[1] > line_pos >= cy:
                            self.total_out += 1
                            self._counted_tracks.add(tid)
                    else:
                        if prev[0] < line_pos <= cx:
                            self.total_in += 1
                            self._counted_tracks.add(tid)
                        elif prev[0] > line_pos >= cx:
                            self.total_out += 1
                            self._counted_tracks.add(tid)

                cv2.rectangle(annotated, (px1, py1), (px2, py2), YELLOW, 2)
                label = f"ID[{tid}]"
                ty = py1 - 8 if py1 > 20 else py2 + 18
                cv2.putText(annotated, label, (px1, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, YELLOW, 2, cv2.LINE_AA)

            self._prev_pos[tid] = (cx, cy)

        gone_ids = set(self._prev_pos.keys()) - current_tids
        for tid in gone_ids:
            self._prev_pos.pop(tid, None)
            self._person_first_seen.pop(tid, None)
        gone_c = set(self._counted_tracks) - current_tids
        for tid in gone_c:
            self._counted_tracks.discard(tid)
        gone_a = set(self._track_area_state.keys()) - current_tids
        for tid in gone_a:
            st = self._track_area_state[tid]
            if st["counted_in"]:
                self.total_out += 1
            del self._track_area_state[tid]

        active_in_area = sum(1 for s in self._track_area_state.values() if s.get("counted_in"))
        self.current_in_area = active_in_area
        display_in = self.total_out if self.inverted else self.total_in
        display_out = self.total_in if self.inverted else self.total_out

        if is_area_mode and area_ok:
            if self.line_mode == "rectangle":
                cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), PURPLE, 2)
                cv2.putText(annotated, "Area de analisis", (rx1 + 4, max(ry1 - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, PURPLE, 1, cv2.LINE_AA)
            else:
                pts_arr = np.array(poly_pts)
                cv2.polylines(annotated, [pts_arr], True, PURPLE, 2)
                for i, pt in enumerate(poly_pts):
                    cv2.circle(annotated, pt, 5, GREEN, -1)
                    cv2.putText(annotated, str(i+1), (pt[0]+8, pt[1]+8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
                if poly_pts:
                    cx = int(sum(p[0] for p in poly_pts) / len(poly_pts))
                    cy_t = int(sum(p[1] for p in poly_pts) / len(poly_pts))
                    cv2.putText(annotated, "Area personalizada", (cx - 50, max(cy_t - 10, 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, PURPLE, 1, cv2.LINE_AA)
        else:
            if self.line_mode == "horizontal":
                cv2.line(annotated, (0, line_pos), (w, line_pos), PURPLE, 2)
                cv2.putText(annotated, "Linea de conteo", (4, max(line_pos - 8, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, PURPLE, 1, cv2.LINE_AA)
            else:
                cv2.line(annotated, (line_pos, 0), (line_pos, h), PURPLE, 2)
                cv2.putText(annotated, "Linea de conteo", (max(line_pos + 4, 4), 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, PURPLE, 1, cv2.LINE_AA)

        ox = 12
        oy = 30
        lh = 20

        if is_area_mode:
            items = [
                (f"ENTRADAS: {display_in}", GREEN),
                (f"SALIDAS: {display_out}", RED),
                (f"CON BOTAS: {self.total_con_botas}", GREEN),
                (f"SIN BOTAS: {self.total_sin_botas}", RED),
                (f"CUMPLIMIENTO: {self.total_cumplimientos}", GREEN),
                (f"INCUMPLIMIENTO: {self.total_incumplimientos}", RED),
                (f"EN AREA: {self.current_in_area}", CYAN),
            ]
        else:
            items = [
                (f"ENTRADAS: {display_in}", GREEN),
                (f"SALIDAS: {display_out}", RED),
            ]

        n_items = len(items)
        overlay = annotated.copy()
        cv2.rectangle(overlay, (ox - 6, oy - 22),
                      (ox + 240, oy + n_items * lh + 4), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0, annotated)

        for i, (txt, clr) in enumerate(items):
            cv2.putText(annotated, txt, (ox, oy + i * lh),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, clr, 1, cv2.LINE_AA)

        return annotated

    def get_frame_jpeg(self) -> Optional[bytes]:
        with self._lock:
            frame = self._frame
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_q])
        return buf.tobytes() if ok else None

    def get_stats(self) -> dict:
        display_in = self.total_out if self.inverted else self.total_in
        display_out = self.total_in if self.inverted else self.total_out
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
            "total_in":           self.total_in,
            "total_out":          self.total_out,
            "current_in_area":    self.current_in_area,
            "display_in":         display_in,
            "display_out":        display_out,
            "inverted":           self.inverted,
            "line_mode":          self.line_mode,
            "line_pos":           self.line_pos,
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

    def set_line_mode(self, mode: str) -> None:
        if mode in ("horizontal", "vertical", "rectangle", "custom_rect"):
            self.line_mode = mode

    def set_line_pos(self, pct: int) -> None:
        self.line_pos = max(0, min(100, pct))

    def set_inverted(self, inv: bool) -> None:
        self.inverted = inv

    def set_jpeg_q(self, q: int) -> None:
        self.jpeg_q = max(10, min(100, q))

    def set_max_dim(self, d: int) -> None:
        self.max_dim = max(0, d)

    def set_frame_step(self, s: int) -> None:
        self.frame_step = max(1, s)

    def set_custom_rect(self, points: List[Tuple[float, float]]) -> None:
        if len(points) >= 3:
            self._rect_points = points[:4]

    def reset(self) -> None:
        self.total_con_botas      = 0
        self.total_sin_botas      = 0
        self.total_cumplimientos  = 0
        self.total_incumplimientos = 0
        self.total_in             = 0
        self.total_out            = 0
        self.current_in_area      = 0
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
        self._track_area_state.clear()
        self._prev_pos.clear()
        self._counted_tracks.clear()
        self._person_first_seen.clear()
        self._pipeline_start_time = None
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
              line_mode: str = "rectangle", line_pos: int = 50,
              inverted: bool = False, jpeg_q: int = JPEG_Q,
              max_dim: int = 0, frame_step: int = 1,
              custom_rect: Optional[List[Tuple[float, float]]] = None,
              fps_limit: float = 0.0) -> None:
        if not multi_acquire():
            raise RuntimeError("Límite de 4 reproducciones simultáneas alcanzado")
        if not is_multi_enabled():
            self.stop_all()
        with self._lock:
            p = AreaPipeline(source_id, source_path, func_state.copy(),
                                   conf_thresh, half, model_path, min_time,
                                   area_x1, area_y1, area_x2, area_y2,
                                   line_mode=line_mode, line_pos=line_pos,
                                   inverted=inverted, jpeg_q=jpeg_q,
                                   max_dim=max_dim, frame_step=frame_step,
                                   custom_rect=custom_rect,
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

    def set_line_mode(self, source_id: int, mode: str) -> None:
        with self._lock:
            p = self.pipelines.get(source_id)
        if p:
            p.set_line_mode(mode)

    def set_line_pos(self, source_id: int, pct: int) -> None:
        with self._lock:
            p = self.pipelines.get(source_id)
        if p:
            p.set_line_pos(pct)

    def set_inverted(self, source_id: int, inv: bool) -> None:
        with self._lock:
            p = self.pipelines.get(source_id)
        if p:
            p.set_inverted(inv)

    def set_jpeg_q(self, source_id: int, q: int) -> None:
        with self._lock:
            p = self.pipelines.get(source_id)
        if p:
            p.set_jpeg_q(q)

    def set_max_dim(self, source_id: int, d: int) -> None:
        with self._lock:
            p = self.pipelines.get(source_id)
        if p:
            p.set_max_dim(d)

    def set_frame_step(self, source_id: int, s: int) -> None:
        with self._lock:
            p = self.pipelines.get(source_id)
        if p:
            p.set_frame_step(s)

    def set_custom_rect(self, source_id: int, points: List[Tuple[float, float]]) -> None:
        with self._lock:
            p = self.pipelines.get(source_id)
        if p:
            p.set_custom_rect(points)

    def reset(self, source_id: int) -> None:
        with self._lock:
            p = self.pipelines.get(source_id)
        if p:
            p.reset()
