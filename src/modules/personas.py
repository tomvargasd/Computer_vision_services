"""
vision/personas.py — Pipeline de Detección de Personas
=======================================================
Funciones implementadas
  • conteo     — conteo cruzado IN/OUT con línea configurable (ByteTrack IDs)
  • permanencia — tiempo en escena por track ID
  • heatmap    — mapa de calor acumulativo con decaimiento suave

Arquitectura:
  PersonasPipeline → un hilo por fuente activa
  PersonasManager  → singleton que gestiona todos los pipelines
"""
from __future__ import annotations

import cv2
import time
import threading
import numpy as np
from typing import Optional, Dict
from ultralytics import YOLO

from src.utils import get_device
from src.modules.base import multi_acquire, multi_release, is_multi_enabled
from src.config import BASE_DIR

MODEL_NAME  = "yolo11n.pt"   # descarga automática en ~/.ultralytics/ la 1ª vez
PERSON_CLS  = 0              # clase "person" en COCO
CONF_THRESH = 0.35
IOU_THRESH  = 0.50
JPEG_Q      = 72             # calidad JPEG para el stream MJPEG

PURPLE = (200, 0, 200)
YELLOW = (0, 255, 255)
WHITE  = (255, 255, 255)


# ─────────────────────────────────────────────────────────────────────────────
class PersonasPipeline:
    """
    Un pipeline de análisis de video para una fuente específica.

    El parámetro func_state es un dict mutable {func_id: bool} que
    puede actualizarse en caliente desde el hilo principal sin detener
    el pipeline.
    """

    def __init__(self, source_id: int, source_path: str, func_state: dict, conf_thresh: float = CONF_THRESH, half: bool = False, model_path: str = None, line_y_pct: int = 85, fps_limit: float = 0.0):
        self.source_id   = source_id
        self.source_path = source_path
        self.func_state  = func_state          # dict compartido, se actualiza externamente
        self.conf_thresh = conf_thresh
        self.half        = half
        self.model_path  = model_path or MODEL_NAME
        self.line_y_pct  = line_y_pct
        self.fps_limit   = fps_limit

        # Modelo — se carga en el hilo para que /start responda de inmediato
        self.model = None

        # Estado de hilo
        self._frame: Optional[np.ndarray] = None
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # ── Conteo cruzado ──────────────────────────────────────────────────
        self.total_in  = 0
        self.total_out = 0
        self._prev_cy: Dict[int, int] = {}      # {track_id: cy anterior}
        # Estados de cruce por ID: "none" | "inside" | "done"
        # none  → nunca ha cruzado la línea
        # inside → cruzó hacia arriba (entró), aún no ha salido
        # done  → completó el ciclo entrada+salida, ya no se cuenta
        self._cross_state: Dict[int, str] = {}

        # ── Permanencia ─────────────────────────────────────────────────────
        self._first_seen: Dict[int, float] = {} # {track_id: timestamp 1ª aparición}

        # ── Heatmap ─────────────────────────────────────────────────────────
        self._heatmap_acc: Optional[np.ndarray] = None

        # ── Stats públicos (leídos por /api/personas/sources/<id>/stats) ────
        self.current_persons = 0
        self._active_ids: set = set()
        self._h = 0
        self._w = 0

    # ── Control del pipeline ─────────────────────────────────────────────────

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"personas-pipe-{self.source_id}",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._thread = None

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Bucle principal ──────────────────────────────────────────────────────

    def _make_error_frame(self, msg: str) -> np.ndarray:
        """Genera un frame negro con mensaje de error para mostrar en el stream."""
        h, w = 480, 640
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Línea de fondo semi-visible
        cv2.putText(frame, "ERROR DE FUENTE", (int(w * 0.25), h // 2 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (85, 42, 24), 2, cv2.LINE_AA)
        # Mensaje corto (split si es muy largo)
        max_chars = 55
        for i, part in enumerate([msg[j:j+max_chars] for j in range(0, min(len(msg), max_chars*3), max_chars)]):
            cv2.putText(frame, part, (20, h // 2 + 10 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "Verifica la ruta o permisos de la fuente", (20, h - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 120, 120), 1, cv2.LINE_AA)
        return frame

    def _run(self) -> None:
        # Cargar modelo en el hilo (evita bloquear /start)
        self.model = YOLO(self.model_path)
        self.model.to(get_device())

        # Soporte para índice de cámara numérico o ruta/URL
        try:
            src = int(self.source_path)
        except (ValueError, TypeError):
            src = self.source_path

        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            # Mostrar error en el stream en lugar de negro
            err = self._make_error_frame(f"No se puede abrir: {self.source_path}")
            with self._lock:
                self._frame = err
            # Mantener vivo para que el stream sirva el frame de error
            while not self._stop.is_set():
                time.sleep(0.5)
            return

        # Reducir buffer para streams (reduce latencia)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        first_frame = True
        while not self._stop.is_set() and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # Archivo de video → loop; stream caído → salir
                if isinstance(src, str) and "://" not in src:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            if first_frame:
                self._h, self._w = frame.shape[:2]
                self._heatmap_acc = np.zeros((self._h, self._w), dtype=np.float32)
                first_frame = False

            annotated = self._process(frame)

            with self._lock:
                self._frame = annotated
            time.sleep(self.fps_limit)

        cap.release()

    # ── Procesado de un frame ─────────────────────────────────────────────────

    def _process(self, frame: np.ndarray) -> np.ndarray:
        h, w   = self._h, self._w
        line_y = int(h * self.line_y_pct / 100)

        # ── Inferencia + tracking ────────────────────────────────────────────
        results = self.model.track(
            frame,
            persist=True,
            classes=[PERSON_CLS],
            conf=self.conf_thresh,
            iou=IOU_THRESH,
            half=self.half,
            verbose=False,
            tracker="bytetrack.yaml",
        )

        annotated  = frame.copy()
        r          = results[0]
        boxes      = r.boxes if r.boxes is not None else []
        now        = time.time()
        active_ids = set()

        for box in boxes:
            if box.id is None:
                continue

            tid        = int(box.id[0])
            conf       = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            active_ids.add(tid)

            # ── Conteo cruzado (máquina de estados por ID) ──────────────────
            if self.func_state.get("conteo"):
                prev = self._prev_cy.get(tid)
                if prev is not None:
                    state = self._cross_state.get(tid, "none")
                    # Coordenadas Y crecen hacia abajo; line_y está al 75 % del alto.
                    # Subir  (cy disminuye, cruza de abajo→arriba): ENTRADA
                    # Bajar  (cy aumenta,  cruza de arriba→abajo):  SALIDA
                    crossed_up   = prev > line_y and cy <= line_y
                    crossed_down = prev < line_y and cy >= line_y

                    if crossed_up:
                        if state == "none":
                            # Entra por primera vez
                            self.total_in += 1
                            self._cross_state[tid] = "inside"
                        # state == "inside": ya estaba dentro, ignora rebote
                        # state == "done":   ciclo completo, no contar de nuevo

                    elif crossed_down:
                        if state in ("none", "inside"):
                            # Sale (state "none" cubre personas que ya estaban
                            # dentro cuando arrancó el sistema)
                            self.total_out += 1
                            self._cross_state[tid] = "done"
                        # state == "done": ya salió antes, ignorar

                self._prev_cy[tid] = cy

            # ── Primera aparición (base para permanencia) ────────────────────
            if tid not in self._first_seen:
                self._first_seen[tid] = now

            # ── Acumular en heatmap ──────────────────────────────────────────
            if self.func_state.get("heatmap") and self._heatmap_acc is not None:
                rw = max(1, (x2 - x1) // 2)
                rh = max(1, (y2 - y1) // 2)
                cv2.ellipse(
                    self._heatmap_acc, (cx, cy), (rw, rh),
                    0, 0, 360, 4.0, -1,
                )

            # ── Dibujar bounding box ─────────────────────────────────────────
            cv2.rectangle(annotated, (x1, y1), (x2, y2), YELLOW, 2)

            # ── Etiqueta: ID[ID] - conf. XX% ─────────────────────────────────
            label = f"ID[{tid}] - conf. {int(conf * 100)}%"
            if self.func_state.get("permanencia") and tid in self._first_seen:
                secs  = int(now - self._first_seen[tid])
                m, s  = divmod(secs, 60)
                label += f"  {m}:{s:02d}" if m else f"  {s}s"

            ty = y1 - 8 if y1 > 20 else y2 + 18
            cv2.putText(
                annotated, label, (x1, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, YELLOW, 2, cv2.LINE_AA,
            )

        # ── Limpiar IDs que salieron de escena ───────────────────────────────
        gone = set(self._prev_cy.keys()) - active_ids
        for tid in gone:
            self._prev_cy.pop(tid, None)
            # _first_seen se conserva (historial completo de permanencia)

        self.current_persons = len(active_ids)
        self._active_ids      = active_ids

        # ── Overlay del mapa de calor ─────────────────────────────────────────
        if self.func_state.get("heatmap") and self._heatmap_acc is not None:
            if self._heatmap_acc.max() > 0:
                norm     = cv2.normalize(self._heatmap_acc, None, 0, 255, cv2.NORM_MINMAX)
                hm_color = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
                mask     = (self._heatmap_acc > 0.5).astype(np.float32)
                mask3    = np.stack([mask, mask, mask], axis=2)
                annotated = cv2.addWeighted(
                    annotated, 1.0,
                    (hm_color * mask3).astype(np.uint8), 0.45, 0,
                )
            # Decaimiento gradual → el mapa "olvida" posiciones antiguas
            self._heatmap_acc *= 0.996

        # ── Línea de conteo + HUD ─────────────────────────────────────────────
        if self.func_state.get("conteo"):
            cv2.line(annotated, (0, line_y), (w, line_y), PURPLE, 2)
            cv2.putText(
                annotated,
                f"IN {self.total_in}   OUT {self.total_out}",
                (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, YELLOW, 2, cv2.LINE_AA,
            )

        # ── Contador de personas activas (siempre visible) ───────────────────
        cv2.putText(
            annotated,
            f"Personas: {self.current_persons}",
            (12, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )

        return annotated

    # ── API pública ──────────────────────────────────────────────────────────

    def set_line_y(self, pct: int) -> None:
        self.line_y_pct = max(0, min(100, pct))

    def get_frame_jpeg(self) -> Optional[bytes]:
        with self._lock:
            frame = self._frame
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
        return buf.tobytes() if ok else None

    def get_stats(self) -> dict:
        now    = time.time()
        active = self._active_ids
        dwell  = sorted(
            [{"id": tid, "seconds": int(now - t)}
             for tid, t in self._first_seen.items() if tid in active],
            key=lambda x: -x["seconds"],
        )
        return {
            "source_id":       self.source_id,
            "current_persons": self.current_persons,
            "in_count":        self.total_in,
            "out_count":       self.total_out,
            "max_dwell":       dwell[0]["seconds"]  if dwell else None,
            "min_dwell":       dwell[-1]["seconds"] if dwell else None,
            "dwell_times":     dwell[:20],
        }

    def reset(self) -> None:
        self.total_in  = 0
        self.total_out = 0
        self._prev_cy.clear()
        self._cross_state.clear()
        self._first_seen.clear()
        if self._heatmap_acc is not None:
            self._heatmap_acc[:] = 0


# ─────────────────────────────────────────────────────────────────────────────
class PersonasManager:
    """
    Singleton que gestiona un PersonasPipeline por fuente activa.
    Thread-safe.
    """

    _instance: Optional["PersonasManager"] = None
    _class_lock = threading.Lock()

    def __init__(self) -> None:
        self.pipelines: Dict[int, PersonasPipeline] = {}
        self._lock = threading.Lock()

    @classmethod
    def get(cls) -> "PersonasManager":
        with cls._class_lock:
            if cls._instance is None:
                cls._instance = PersonasManager()
        return cls._instance

    # ── Control de pipelines ─────────────────────────────────────────────────

    def start(self, source_id: int, source_path: str, func_state: dict, conf_thresh: float = CONF_THRESH, half: bool = False, model_path: str = None, line_y_pct: int = 85, fps_limit: float = 0.0) -> None:
        if not multi_acquire():
            raise RuntimeError("Límite de 4 reproducciones simultáneas alcanzado")
        if not is_multi_enabled():
            self.stop_all()
        with self._lock:
            p = PersonasPipeline(source_id, source_path, func_state.copy(), conf_thresh, half, model_path, line_y_pct, fps_limit)
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
            # auto-limpiar pipeline muerto
            with self._lock:
                self.pipelines.pop(source_id, None)
            return False
        return True

    # ── Actualización en caliente del estado de funciones ────────────────────

    def update_func_state(self, func_state: dict) -> None:
        """Propaga el estado de funciones a todos los pipelines activos."""
        with self._lock:
            for p in self.pipelines.values():
                p.func_state.update(func_state)

    # ── Acceso a frames y stats ──────────────────────────────────────────────

    def get_frame_jpeg(self, source_id: int) -> Optional[bytes]:
        with self._lock:
            p = self.pipelines.get(source_id)
        return p.get_frame_jpeg() if p else None

    def get_stats(self, source_id: int) -> Optional[dict]:
        with self._lock:
            p = self.pipelines.get(source_id)
        return p.get_stats() if p else None

    def set_line_y(self, source_id: int, pct: int) -> None:
        with self._lock:
            p = self.pipelines.get(source_id)
        if p:
            p.set_line_y(pct)

    def reset(self, source_id: int) -> None:
        with self._lock:
            p = self.pipelines.get(source_id)
        if p:
            p.reset()
