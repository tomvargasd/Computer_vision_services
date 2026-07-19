from __future__ import annotations

import cv2
import os
import json
import uuid
import time
import threading
import numpy as np
from collections import deque
from typing import Optional, Dict, List, Tuple
from ultralytics import YOLO
from datetime import datetime

from src.utils import get_device
from src.modules.base import multi_acquire, multi_release, is_multi_enabled
from src.config import BASE_DIR

MODEL_NAME  = "yolo11n.pt"
POSE_MODEL  = "yolo11n-pose.pt"
CONF_THRESH = 0.35
IOU_THRESH  = 0.50
JPEG_Q      = 72
TANK_CLASSES = [0]

RED    = (0, 0, 255)
GREEN  = (0, 255, 0)
YELLOW = (0, 255, 255)
PURPLE = (200, 0, 200)
ORANGE = (0, 165, 255)
WHITE  = (255, 255, 255)
CYAN   = (255, 255, 0)
MAGENTA = (200, 0, 255)

CAPTURES_DIR = os.path.join(BASE_DIR, "static", "uploads", "captures", "tanques_gas")
_TEACH_DATA_FILE = "_teach_data_tanques.json"
_TEACH_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), _TEACH_DATA_FILE)
_TEACH_SAMPLES: List[dict] = []
_TEACH_SIM_RATIO   = 0.25
_TEACH_MIN_SCORE   = 0.55
_TEACH_LOG_SECONDS = 2.0
_TEACH_LOG_MAXLEN  = 90
_ACTION_COOLDOWN   = 5.0

_SKELETON = [
    (0, 1, (255, 222, 0)), (0, 2, (255, 222, 0)),
    (1, 3, (230, 200, 20)), (2, 4, (230, 200, 20)),
    (5, 6, (24, 42, 85)), (5, 11, (50, 55, 70)),
    (6, 12, (50, 55, 70)), (11, 12, (75, 68, 55)),
    (5, 7, (100, 200, 150)), (7, 9, (80, 220, 180)),
    (6, 8, (100, 200, 150)), (8, 10, (80, 220, 180)),
    (11, 13, (255, 100, 100)), (13, 15, (255, 60, 60)),
    (12, 14, (255, 150, 100)), (14, 16, (255, 110, 60)),
]
_KP_COLOR = (255, 255, 255)


def _load_teach_samples():
    global _TEACH_SAMPLES
    if os.path.exists(_TEACH_DATA_PATH):
        try:
            with open(_TEACH_DATA_PATH) as f:
                _TEACH_SAMPLES = json.load(f).get("samples", [])
        except Exception:
            _TEACH_SAMPLES = []


def _save_teach_sample(sample: dict):
    global _TEACH_SAMPLES
    _TEACH_SAMPLES.append(sample)
    with open(_TEACH_DATA_PATH, "w") as f:
        json.dump({"samples": _TEACH_SAMPLES}, f, indent=2, default=str)


def _kp_dist(kps_a: np.ndarray, kps_b: np.ndarray) -> float:
    dists = []
    for i in range(17):
        if (kps_a[i, 0] > 1 and kps_a[i, 1] > 1 and
            kps_b[i, 0] > 1 and kps_b[i, 1] > 1):
            dists.append(float(np.linalg.norm(kps_a[i] - kps_b[i])))
    return float(np.mean(dists)) if dists else -1.0


def _point_in_polygon(pt, polygon) -> bool:
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


class TanquesGasPipeline:
    def __init__(self, source_id: int, source_path: str, func_state: dict,
                 conf_thresh: float = CONF_THRESH, half: bool = False,
                 model_path: str = None, pose_model_path: str = None,
                 smoke_model_path: str = None,
                 line_mode: str = "horizontal", line_pos: int = 50,
                 fps_limit: float = 0.0):
        self.source_id   = source_id
        self.source_path = source_path
        self.func_state  = func_state
        self.conf_thresh = conf_thresh
        self.half        = half
        self.model_path  = model_path or MODEL_NAME
        self.classes     = TANK_CLASSES
        self.pose_model_path = pose_model_path or POSE_MODEL
        self.smoke_model_path = smoke_model_path or MODEL_NAME
        self.line_mode   = line_mode
        self.line_pos    = line_pos
        self.fps_limit   = fps_limit

        self.model: Optional[YOLO] = None
        self.pose_model: Optional[YOLO] = None
        self.smoke_model: Optional[YOLO] = None

        self._frame: Optional[np.ndarray] = None
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._h = 0
        self._w = 0

        # ── Conteo ──
        self.total_in   = 0
        self.total_out  = 0
        self.current_objects = 0
        self._prev_pos = {}
        self._cross_state = {}
        self._counted_tracks = set()
        self._prev_cx: Dict[int, int] = {}
        self._line_p1 = (0.25, 0.25)
        self._line_p2 = (0.75, 0.75)
        self._rect_points = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
        self._custom_editing = False

        # ── Acciones / Teach ──
        self.current_persons = 0
        self._person_log: Dict[int, deque] = {}
        self._prev_person_data: list = []
        self._next_tid = 1
        self._person_actions: Dict[int, str] = {}
        self._action_log: List[dict] = []
        self._action_count: Dict[str, int] = {}
        self._per_person_action_count: Dict[str, Dict[int, int]] = {}
        self._teach_mode = False
        self._teach_paused = False
        self._teach_captured_frame: Optional[np.ndarray] = None
        self._teach_captured_kps: Optional[list] = None
        self._teach_captured_tid: Optional[int] = None
        self._teach_captured_bbox: Optional[list] = None
        self._person_cooldown: Dict[int, Dict[str, float]] = {}
        self._action_detected_log: List[dict] = []

        # ── Humo/Fuego ──
        self.smoke_detected = False
        self.alert_triggered = False
        self.first_detection_time: Optional[str] = None
        self._evidence_saved = False
        os.makedirs(CAPTURES_DIR, exist_ok=True)

        # ── Areas restringidas ──
        self.restricted_areas: List[dict] = []

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True,
            name=f"tanques-gas-pipe-{self.source_id}",
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

        if self.func_state.get("deteccion_acciones"):
            try:
                self.pose_model = YOLO(self.pose_model_path)
                self.pose_model.to(get_device())
            except Exception:
                self.pose_model = None

        if self.func_state.get("deteccion_humo"):
            try:
                self.smoke_model = YOLO(self.smoke_model_path)
                self.smoke_model.to(get_device())
            except Exception:
                self.smoke_model = None

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

        # ── Run main det model (ByteTrack) for counting ──
        if self.func_state.get("conteo") or self.func_state.get("areas_restringidas"):
            results = self.model.track(
                frame, persist=True, conf=self.conf_thresh,
                iou=IOU_THRESH, half=self.half, verbose=False,
                tracker="bytetrack.yaml", classes=self.classes,
            )
            r = results[0]
            boxes = r.boxes if r.boxes is not None else []
        else:
            boxes = []

        active_ids = set()

        # ── Conteo ──
        if self.func_state.get("conteo"):
            annotated = self._process_counting(annotated, boxes, active_ids)

        # ── Areas restringidas ──
        if self.func_state.get("areas_restringidas"):
            annotated = self._process_restricted_areas(annotated, boxes)

        # ── Deteccion de acciones (pose) ──
        if self.func_state.get("deteccion_acciones"):
            annotated = self._process_actions(annotated)

        # ── Deteccion de humo/fuego ──
        if self.func_state.get("deteccion_humo"):
            annotated = self._process_smoke(annotated)

        # ── HUD ──
        cv2.putText(annotated, f"Objetos: {self.current_objects}",
                    (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)
        if self._teach_mode:
            cv2.putText(annotated, "MODO ENSEÑAR - Haga clic en una persona", (12, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, MAGENTA, 2, cv2.LINE_AA)

        return annotated

    def _process_counting(self, frame: np.ndarray, boxes, active_ids: set) -> np.ndarray:
        h, w = self._h, self._w

        if self.line_mode == "horizontal":
            line_pos = int(h * self.line_pos / 100)
            draw_p1, draw_p2 = (0, line_pos), (w, line_pos)
        elif self.line_mode == "vertical":
            line_pos = int(w * self.line_pos / 100)
            draw_p1, draw_p2 = (line_pos, 0), (line_pos, h)
        elif self.line_mode == "rectangle":
            x1 = int(w * self._rect_points[0][0])
            y1 = int(h * self._rect_points[0][1])
            x2 = int(w * self._rect_points[2][0])
            y2 = int(h * self._rect_points[2][1])
        elif self.line_mode == "custom_line":
            p1x = int(w * self._line_p1[0])
            p1y = int(h * self._line_p1[1])
            p2x = int(w * self._line_p2[0])
            p2y = int(h * self._line_p2[1])
            draw_p1, draw_p2 = (p1x, p1y), (p2x, p2y)
        elif self.line_mode == "custom_rect":
            pass

        for box in boxes:
            if box.id is None:
                continue
            tid = int(box.id[0])
            conf = float(box.conf[0])
            x1b, y1b, x2b, y2b = map(int, box.xyxy[0])
            cx = (x1b + x2b) // 2
            cy = (y1b + y2b) // 2
            active_ids.add(tid)

            cv2.rectangle(frame, (x1b, y1b), (x2b, y2b), YELLOW, 2)
            label = f"ID[{tid}]"
            ty = y1b - 8 if y1b > 20 else y2b + 18
            cv2.putText(frame, label, (x1b, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, YELLOW, 2, cv2.LINE_AA)

            # ── Rectangle counting ──
            if self.line_mode == "rectangle":
                inside = (x1 < cx < x2 and y1 < cy < y2)
                if tid not in self._counted_tracks and inside:
                    self.total_in += 1
                    self._counted_tracks.add(tid)
                # Check exit
                if tid in self._counted_tracks and not inside:
                    if self._cross_state.get(tid) == "inside":
                        self.total_out += 1
                    self._cross_state[tid] = "outside"

                if inside:
                    cv2.rectangle(frame, (x1b, y1b), (x2b, y2b), GREEN, 3)

            # ── Line counting ──
            elif self.line_mode in ("horizontal", "vertical"):
                prev = self._prev_pos.get(tid)
                self._prev_pos[tid] = (cx, cy)
                if prev is None:
                    continue
                if tid in self._counted_tracks:
                    continue
                crossed = False
                direction = None
                if self.line_mode == "horizontal":
                    if prev[1] < line_pos <= cy:
                        crossed = True
                        direction = "in"
                    elif prev[1] > line_pos >= cy:
                        crossed = True
                        direction = "out"
                else:
                    prev_cx = self._prev_cx.get(tid)
                    if prev_cx is not None:
                        crossed_left = prev_cx < line_pos and cx >= line_pos
                        crossed_right = prev_cx > line_pos and cx <= line_pos
                        if crossed_left:
                            crossed = True; direction = "in"
                        elif crossed_right:
                            crossed = True; direction = "out"
                    self._prev_cx[tid] = cx

                if crossed:
                    self._counted_tracks.add(tid)
                    if direction == "in":
                        self.total_in += 1
                    else:
                        self.total_out += 1

            # ── Custom line counting ──
            elif self.line_mode == "custom_line":
                prev = self._prev_pos.get(tid)
                self._prev_pos[tid] = (cx, cy)
                if prev is None:
                    continue
                if tid in self._counted_tracks:
                    continue

                p1 = np.array([p1x, p1y])
                p2 = np.array([p2x, p2y])
                prev_pt = np.array(prev)
                curr_pt = np.array([cx, cy])

                def side(p, a, b):
                    return np.cross(b - a, p - a)

                s1 = side(prev_pt, p1, p2)
                s2 = side(curr_pt, p1, p2)
                if s1 * s2 < 0:
                    self._counted_tracks.add(tid)
                    if s2 > 0:
                        self.total_in += 1
                    else:
                        self.total_out += 1

        # Draw counting overlays
        if self.line_mode == "horizontal":
            cv2.line(frame, draw_p1, draw_p2, PURPLE, 2)
            cv2.putText(frame, f"Linea Horizontal  IN {self.total_in}  OUT {self.total_out}",
                        (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2, cv2.LINE_AA)
        elif self.line_mode == "vertical":
            cv2.line(frame, draw_p1, draw_p2, PURPLE, 2)
            cv2.putText(frame, f"Linea Vertical  IN {self.total_in}  OUT {self.total_out}",
                        (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2, cv2.LINE_AA)
        elif self.line_mode == "rectangle":
            cv2.rectangle(frame, (x1, y1), (x2, y2), PURPLE, 2)
            cv2.putText(frame, f"Area  IN {self.total_in}  OUT {self.total_out}",
                        (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2, cv2.LINE_AA)
        elif self.line_mode == "custom_line":
            cv2.line(frame, draw_p1, draw_p2, PURPLE, 2)
            cv2.circle(frame, draw_p1, 6, GREEN, -1)
            cv2.circle(frame, draw_p2, 6, GREEN, -1)
            cv2.putText(frame, f"Custom Line  IN {self.total_in}  OUT {self.total_out}",
                        (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2, cv2.LINE_AA)
        elif self.line_mode == "custom_rect":
            pts = []
            for i, (px, py) in enumerate(self._rect_points):
                pt = (int(w * px), int(h * py))
                pts.append(pt)
                cv2.circle(frame, pt, 6, GREEN, -1)
                cv2.putText(frame, str(i+1), (pt[0]+8, pt[1]+8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)
            if len(pts) == 4:
                cv2.polylines(frame, [np.array(pts)], True, PURPLE, 2)
            cv2.putText(frame, f"Custom Rect  IN {self.total_in}  OUT {self.total_out}",
                        (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2, cv2.LINE_AA)

        # Cleanup stale tracks
        gone = set(self._prev_pos.keys()) - active_ids
        for tid in gone:
            self._prev_pos.pop(tid, None)
            self._prev_cx.pop(tid, None)

        gone_c = set(self._counted_tracks) - active_ids
        for tid in gone_c:
            if tid not in active_ids:
                self._counted_tracks.discard(tid)
                self._cross_state.pop(tid, None)

        self.current_objects = len(active_ids)
        return frame

    def _process_restricted_areas(self, frame: np.ndarray, boxes) -> np.ndarray:
        h, w = self._h, self._w

        for area_data in self.restricted_areas:
            pts_raw = area_data.get("points", [])
            restrict_type = area_data.get("restrict_type", "ambos")
            if len(pts_raw) < 3:
                continue
            pts_px = [(int(x * w), int(y * h)) for x, y in pts_raw]

            overlay = frame.copy()
            cv2.fillPoly(overlay, [np.array(pts_px)], (0, 0, 255))
            frame = cv2.addWeighted(overlay, 0.1, frame, 0.9, 0)
            cv2.polylines(frame, [np.array(pts_px)], True, RED, 2)

            for box in boxes:
                if box.id is None:
                    continue
                tid = int(box.id[0])
                cls = int(box.cls[0])
                x1b, y1b, x2b, y2b = map(int, box.xyxy[0])
                cx, cy = (x1b + x2b) // 2, (y1b + y2b) // 2

                is_person = (cls == 0)
                is_tank = (cls == 1)

                should_alert = False
                if restrict_type == "persona" and is_person:
                    should_alert = True
                elif restrict_type == "tanque" and is_tank:
                    should_alert = True
                elif restrict_type == "ambos":
                    should_alert = True

                if should_alert and _point_in_polygon((cx, cy), pts_px):
                    cv2.rectangle(frame, (x1b, y1b), (x2b, y2b), RED, 3)
                    cv2.putText(frame, "RESTRINGIDO", (x1b, y1b - 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, cv2.LINE_AA)

        return frame

    def _process_actions(self, frame: np.ndarray) -> np.ndarray:
        if self.pose_model is None:
            return frame
        h, w = self._h, self._w

        results = self.pose_model(
            frame, conf=self.conf_thresh, iou=IOU_THRESH,
            half=self.half, verbose=False,
        )
        r = results[0]

        if r.keypoints is None:
            return frame

        kps_all = r.keypoints.xy.cpu().numpy()
        conf_all = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None
        boxes_data = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []

        curr_ctr = []
        for i in range(len(kps_all)):
            if len(boxes_data) > i:
                bx1, by1, bx2, by2 = boxes_data[i]
                curr_ctr.append(np.array([(bx1 + bx2) / 2, (by1 + by2) / 2], dtype=np.float32))
            else:
                curr_ctr.append(np.zeros(2, dtype=np.float32))

        prev = self._prev_person_data
        matched = [None] * len(kps_all)
        used = set()
        for ci, cc in enumerate(curr_ctr):
            best_pi, best_d = None, 120.0
            for pi, pd in enumerate(prev):
                if pi in used:
                    continue
                d = float(np.linalg.norm(cc - pd["centroid"]))
                if d < best_d:
                    best_d, best_pi = d, pi
            if best_pi is not None:
                matched[ci] = best_pi
                used.add(best_pi)

        new_pdata = []
        now = time.time()

        for i, kps in enumerate(kps_all):
            conf_i = conf_all[i] if conf_all is not None else None
            prev_pd = prev[matched[i]] if matched[i] is not None else None

            if prev_pd is not None:
                tid = prev_pd["tid"]
            else:
                tid = self._next_tid
                self._next_tid += 1

            bbox_diag = 0.0
            bx1_l = by1_l = bx2_l = by2_l = 0
            if len(boxes_data) > i:
                bx1_l, by1_l, bx2_l, by2_l = map(int, boxes_data[i])
                bbox_diag = float(np.sqrt((bx2_l - bx1_l)**2 + (by2_l - by1_l)**2))

            # ── Teach log (rotating 2s buffer) ──
            if not self._teach_paused and bbox_diag > 1:
                if tid not in self._person_log:
                    self._person_log[tid] = deque(maxlen=_TEACH_LOG_MAXLEN)
                self._person_log[tid].append({
                    "ts": now,
                    "kps": kps.tolist(),
                    "bbox": [bx1_l, by1_l, bx2_l, by2_l],
                    "centroid": curr_ctr[i].tolist(),
                })

            # ── Teach capture ──
            if self._teach_mode and not self._teach_paused:
                if self._teach_captured_tid is None:
                    self._teach_captured_tid = tid
                    self._teach_captured_kps = kps.tolist()
                    self._teach_captured_bbox = [bx1_l, by1_l, bx2_l, by2_l]
                    self._teach_captured_frame = frame.copy()
                    self._teach_paused = True

            # ── Recognize actions ──
            detected_action = None
            if _TEACH_SAMPLES and bbox_diag > 1:
                detected_action = self._match_action(kps, bbox_diag)

            new_pdata.append({
                "centroid": curr_ctr[i],
                "tid": tid,
            })

            # ── Draw ──
            if len(boxes_data) > i:
                bx1, by1, bx2, by2 = map(int, boxes_data[i])
                box_clr = CYAN if detected_action else GREEN
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_clr, 2)
                id_txt = f"ID[{tid}]"
                cv2.putText(frame, id_txt, (bx1 + 4, by1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1, cv2.LINE_AA)

                if detected_action:
                    cv2.putText(frame, detected_action, (bx1, by1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, MAGENTA, 2, cv2.LINE_AA)

            # ── Skeleton ──
            for (ka, kb, color) in _SKELETON:
                xa, ya = kps[ka]
                xb, yb = kps[kb]
                if xa < 1 or ya < 1 or xb < 1 or yb < 1:
                    continue
                cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), color, 2, cv2.LINE_AA)

            for ki, (xk, yk) in enumerate(kps):
                if xk < 1 or yk < 1:
                    continue
                cv2.circle(frame, (int(xk), int(yk)), 3, _KP_COLOR, -1, cv2.LINE_AA)

            # ── Action recognition with cooldown ──
            if detected_action:
                cooldowns = self._person_cooldown.setdefault(tid, {})
                last_time = cooldowns.get(detected_action, 0)
                if now - last_time >= _ACTION_COOLDOWN:
                    cooldowns[detected_action] = now
                    self._action_count[detected_action] = self._action_count.get(detected_action, 0) + 1
                    per_p = self._per_person_action_count.setdefault(detected_action, {})
                    per_p[tid] = per_p.get(tid, 0) + 1
                    self._action_detected_log.append({
                        "ts": now,
                        "tid": tid,
                        "action": detected_action,
                    })
                    if len(self._action_detected_log) > 200:
                        self._action_detected_log = self._action_detected_log[-200:]

        self._prev_person_data = new_pdata
        self.current_persons = len(kps_all)

        active_tids = {p["tid"] for p in new_pdata}
        for tid in list(self._person_log):
            if tid not in active_tids:
                del self._person_log[tid]

        # ── Action log display (top-right) ──
        recent = [e for e in self._action_detected_log if now - e["ts"] <= 10]
        y_offset = 60
        for entry in reversed(recent[-5:]):
            lbl = f"Persona ID {entry['tid']} : {entry['action']}"
            cv2.putText(frame, lbl, (w - 320, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, MAGENTA, 1, cv2.LINE_AA)
            y_offset += 18

        return frame

    def _match_action(self, kps: np.ndarray, bbox_diag: float) -> Optional[str]:
        if not _TEACH_SAMPLES or bbox_diag < 1:
            return None
        thresh_px = bbox_diag * _TEACH_SIM_RATIO
        best_action = None
        best_score = 0.0

        for sample in _TEACH_SAMPLES:
            action = sample.get("action")
            best_d = float("inf")
            for entry in sample.get("log", []):
                stored = np.array(entry.get("kps"))
                if stored.shape != (17, 2):
                    continue
                d = _kp_dist(kps, stored)
                if 0 <= d < best_d:
                    best_d = d
            if best_d < float("inf") and best_d < thresh_px:
                confidence = 1.0 - best_d / thresh_px
                score = _TEACH_MIN_SCORE + 0.40 * confidence
                if score > best_score:
                    best_score = score
                    best_action = action

        return best_action

    def _process_smoke(self, frame: np.ndarray) -> np.ndarray:
        if self.smoke_model is None:
            return frame
        h, w = self._h, self._w

        results = self.smoke_model.track(
            frame, persist=True, conf=self.conf_thresh,
            iou=IOU_THRESH, half=self.half, verbose=False,
            tracker="bytetrack.yaml",
        )
        r = results[0]
        boxes = r.boxes if r.boxes is not None else []

        frame_has_smoke = False
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            frame_has_smoke = True
            color = RED if self.smoke_detected else ORANGE
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"HUMO/FUEGO {conf:.2f}", (x1, max(y1 - 8, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)

        if frame_has_smoke:
            if not self.smoke_detected:
                self.smoke_detected = True
                self.first_detection_time = datetime.now().strftime("%H:%M:%S")
                self.alert_triggered = True
                if not self._evidence_saved:
                    self._evidence_saved = True
                    self._save_evidence(frame)

        if self.smoke_detected:
            cv2.putText(frame, "!!! HUMO / FUEGO DETECTADO !!!",
                        (12, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, cv2.LINE_AA)
            if self.first_detection_time:
                cv2.putText(frame, f"Desde: {self.first_detection_time}",
                            (12, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 1, cv2.LINE_AA)

        return frame

    def _save_evidence(self, frame: np.ndarray) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"smoke_tanques_{self.source_id}_{ts}.jpg"
        path = os.path.join(CAPTURES_DIR, fname)
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])

    # ── Public API ──

    def get_frame_jpeg(self) -> Optional[bytes]:
        with self._lock:
            frame = self._frame
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
        return buf.tobytes() if ok else None

    def get_stats(self) -> dict:
        action_log_recent = [e for e in self._action_detected_log if time.time() - e["ts"] <= 10]
        return {
            "source_id": self.source_id,
            "current_objects": self.current_objects,
            "in_count": self.total_in,
            "out_count": self.total_out,
            "line_mode": self.line_mode,
            "line_pos": self.line_pos,
            "smoke_detected": self.smoke_detected,
            "alert_triggered": self.alert_triggered,
            "first_detection": self.first_detection_time,
            "current_persons": self.current_persons,
            "action_count": dict(self._action_count),
            "action_log": action_log_recent,
            "teach_mode": self._teach_mode,
            "teach_paused": self._teach_paused,
        }

    def get_teach_data(self) -> dict:
        now = time.time()
        result: dict = {}
        for tid, log in self._person_log.items():
            window = [e for e in log if now - e["ts"] <= _TEACH_LOG_SECONDS]
            if window:
                result[str(tid)] = {"log": window}
        return result

    def get_teach_capture(self) -> Optional[dict]:
        if self._teach_captured_frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", self._teach_captured_frame,
                                [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
        if not ok:
            return None
        import base64
        return {
            "frame_b64": base64.b64encode(buf.tobytes()).decode(),
            "kps": self._teach_captured_kps,
            "tid": self._teach_captured_tid,
            "bbox": self._teach_captured_bbox,
        }

    def start_teach(self) -> None:
        self._teach_mode = True
        self._teach_paused = False
        self._teach_captured_tid = None
        self._teach_captured_kps = None
        self._teach_captured_bbox = None
        self._teach_captured_frame = None
        self._person_log.clear()

    def cancel_teach(self) -> None:
        self._teach_mode = False
        self._teach_paused = False
        self._teach_captured_tid = None
        self._teach_captured_kps = None
        self._teach_captured_bbox = None
        self._teach_captured_frame = None

    def save_teach_action(self, action: str) -> bool:
        if action and self._teach_captured_tid is not None:
            tid = self._teach_captured_tid
            log_entries = list(self._person_log.get(tid, deque(maxlen=_TEACH_LOG_MAXLEN)))
            sample = {
                "id": str(uuid.uuid4()),
                "action": action,
                "log": log_entries,
                "created_at": time.time(),
            }
            _save_teach_sample(sample)
            self._teach_mode = False
            self._teach_paused = False
            self._teach_captured_tid = None
            self._teach_captured_kps = None
            self._teach_captured_bbox = None
            self._teach_captured_frame = None
            return True
        return False

    def get_actions_info(self) -> dict:
        actions = {}
        for sample in _TEACH_SAMPLES:
            action = sample.get("action", "unknown")
            if action not in actions:
                actions[action] = {"count": 0, "per_person": {}}
        for action, data in actions.items():
            data["count"] = self._action_count.get(action, 0)
            per_p = self._per_person_action_count.get(action, {})
            data["per_person"] = {str(k): v for k, v in per_p.items()}
        return actions

    def set_line_mode(self, mode: str) -> None:
        if mode in ("horizontal", "vertical", "rectangle", "custom_line", "custom_rect"):
            self.line_mode = mode
            self._counted_tracks.clear()

    def set_line_pos(self, pct: int) -> None:
        self.line_pos = max(0, min(100, pct))

    def set_custom_line(self, p1x: float, p1y: float, p2x: float, p2y: float) -> None:
        self._line_p1 = (p1x, p1y)
        self._line_p2 = (p2x, p2y)
        self._counted_tracks.clear()

    def set_custom_rect(self, points: List[Tuple[float, float]]) -> None:
        if len(points) == 4:
            self._rect_points = points
            self._counted_tracks.clear()

    def set_rect_area(self, x1: int, y1: int, x2: int, y2: int) -> None:
        self._rect_points = [
            (x1 / 100, y1 / 100),
            (x2 / 100, y1 / 100),
            (x2 / 100, y2 / 100),
            (x1 / 100, y2 / 100),
        ]
        self._counted_tracks.clear()

    def add_restricted_area(self, points: List[Tuple[float, float]], restrict_type: str = "ambos") -> None:
        self.restricted_areas.append({
            "id": str(uuid.uuid4()),
            "points": points,
            "restrict_type": restrict_type,
        })

    def remove_restricted_area(self, area_id: str) -> None:
        self.restricted_areas = [a for a in self.restricted_areas if a.get("id") != area_id]

    def clear_restricted_areas(self) -> None:
        self.restricted_areas.clear()

    def set_pose_model(self, path: str) -> None:
        self.pose_model_path = path
        try:
            self.pose_model = YOLO(path)
            self.pose_model.to(get_device())
        except Exception:
            pass

    def set_smoke_model(self, path: str) -> None:
        self.smoke_model_path = path
        try:
            self.smoke_model = YOLO(path)
            self.smoke_model.to(get_device())
        except Exception:
            pass

    def reset(self) -> None:
        self.total_in = 0
        self.total_out = 0
        self._prev_pos.clear()
        self._prev_cx.clear()
        self._cross_state.clear()
        self._counted_tracks.clear()
        self.smoke_detected = False
        self.alert_triggered = False
        self.first_detection_time = None
        self._evidence_saved = False
        self._action_count.clear()
        self._per_person_action_count.clear()
        self._action_detected_log.clear()
        self._person_cooldown.clear()


class TanquesGasManager:
    _instance: Optional["TanquesGasManager"] = None
    _class_lock = threading.Lock()

    def __init__(self) -> None:
        self.pipelines: Dict[int, TanquesGasPipeline] = {}
        self._lock = threading.Lock()

    @classmethod
    def get(cls) -> "TanquesGasManager":
        with cls._class_lock:
            if cls._instance is None:
                cls._instance = TanquesGasManager()
        return cls._instance

    def start(self, source_id: int, source_path: str, func_state: dict,
              conf_thresh: float = CONF_THRESH, half: bool = False,
              model_path: str = None, pose_model_path: str = None,
              smoke_model_path: str = None,
              line_mode: str = "horizontal", line_pos: int = 50,
              fps_limit: float = 0.0) -> None:
        if not multi_acquire():
            raise RuntimeError("Límite de 4 reproducciones simultáneas alcanzado")
        if not is_multi_enabled():
            self.stop_all()
        with self._lock:
            p = TanquesGasPipeline(source_id, source_path, func_state.copy(),
                                   conf_thresh, half, model_path,
                                   pose_model_path, smoke_model_path,
                                   line_mode, line_pos, fps_limit=fps_limit)
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
            pipelines = list(self.pipelines.values())
        for p in pipelines:
            p.func_state.update(func_state)

    def get_frame_jpeg(self, source_id: int) -> Optional[bytes]:
        with self._lock:
            p = self.pipelines.get(source_id)
        return p.get_frame_jpeg() if p else None

    def get_stats(self, source_id: int) -> Optional[dict]:
        with self._lock:
            p = self.pipelines.get(source_id)
        return p.get_stats() if p else None

    def get_teach_data(self, source_id: int) -> Optional[dict]:
        with self._lock:
            p = self.pipelines.get(source_id)
        return p.get_teach_data() if p else None

    def get_teach_capture(self, source_id: int) -> Optional[dict]:
        with self._lock:
            p = self.pipelines.get(source_id)
        return p.get_teach_capture() if p else None

    def start_teach(self, source_id: int) -> None:
        with self._lock:
            p = self.pipelines.get(source_id)
        if p:
            p.start_teach()

    def cancel_teach(self, source_id: int) -> None:
        with self._lock:
            p = self.pipelines.get(source_id)
        if p:
            p.cancel_teach()

    def save_teach_action(self, source_id: int, action: str) -> bool:
        with self._lock:
            p = self.pipelines.get(source_id)
        return p.save_teach_action(action) if p else False

    def get_actions_info(self, source_id: int) -> dict:
        with self._lock:
            p = self.pipelines.get(source_id)
        return p.get_actions_info() if p else {}

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

    def set_custom_line(self, source_id: int, p1x: float, p1y: float,
                        p2x: float, p2y: float) -> None:
        with self._lock:
            p = self.pipelines.get(source_id)
        if p:
            p.set_custom_line(p1x, p1y, p2x, p2y)

    def set_custom_rect(self, source_id: int, points: List[Tuple[float, float]]) -> None:
        with self._lock:
            p = self.pipelines.get(source_id)
        if p:
            p.set_custom_rect(points)

    def set_rect_area(self, source_id: int, x1: int, y1: int, x2: int, y2: int) -> None:
        with self._lock:
            p = self.pipelines.get(source_id)
        if p:
            p.set_rect_area(x1, y1, x2, y2)

    def add_restricted_area(self, source_id: int, points: list,
                            restrict_type: str = "ambos") -> None:
        with self._lock:
            p = self.pipelines.get(source_id)
        if p:
            p.add_restricted_area(points, restrict_type)

    def remove_restricted_area(self, source_id: int, area_id: str) -> None:
        with self._lock:
            p = self.pipelines.get(source_id)
        if p:
            p.remove_restricted_area(area_id)

    def clear_restricted_areas(self, source_id: int) -> None:
        with self._lock:
            p = self.pipelines.get(source_id)
        if p:
            p.clear_restricted_areas()

    def reset(self, source_id: int) -> None:
        with self._lock:
            p = self.pipelines.get(source_id)
        if p:
            p.reset()


_load_teach_samples()
