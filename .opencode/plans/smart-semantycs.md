# Smart Semantycs — Plan de Implementación

Módulo nuevo e independiente de visión con vocabulario abierto basado en **YOLO-E (YOLOE)**
más **Gemini** para interpretación de prompts. El sistema se identifica globalmente como
**Smart-AI · por SmartCloud**.

---

## 0. Resumen ejecutivo

- Nuevo módulo "Smart Semantycs" en la sidebar (item nuevo, no toca "Sessions" existente en `/`).
- Backend: pipeline YOLOE independiente (`SmartSemntycsPipeline` + `SmartSemntycsManager`, patrón idéntico a `personas.py`).
- **Solo 1 sesión activa a la vez** (al iniciar una, se detiene la anterior).
- Cada sesión: sube/vincula 1 video o stream → escribe 1 prompt → el LLM (Gemini) genera una **"skill"** (JSON con clases, contadores y logs) → arranca la detección → **todo queda bloqueado**.
- Contadores (máx. 5) arriba, video al centro con bounding boxes + controles translúcidos (pause/resume/reset), logs abajo (scroll, nuevos arriba, clic → popup con captura), chat a la derecha.
- **CPU → modelo nano · GPU → modelo XL** (pesos subidos manualmente en Settings).
- Vocabulario: clases LVIS (~1200) de `ultralytics/cfg/datasets/lvis.yaml` como referencia para `model.set_classes(...)`.
- Branding "Smart-AI · por SmartCloud" en todo el sistema.

Verificado: `ultralytics 8.4.41` instalado y `from ultralytics import YOLOE` funciona.

---

## 1. Modelo YOLOE (backend)

### 1.1 Carga y selección de tamaño

| Entorno | Modelo | Nota |
|---|---|---|
| CPU | `yoloe-26n-seg.pt` (nano) | pequeño, 4.8M params, viable en CPU |
| GPU (CUDA) | `yoloe-26x-seg.pt` (XL) | 69.9M params, máxima efectividad |

- Decisión de dispositivo: reusar `get_device()` de `src/modules/base.py:13` (`cuda:0` / `cpu`).
- Pesos: el usuario los sube en **Settings → pestaña Smart Semantycs**. Rutas persistentes en settings:
  - `smart_semantycs_model_nano` (ruta .pt)
  - `smart_semantycs_model_xl` (ruta .pt)
  - `smart_semantycs_model_auto` = `"1"` (nano si CPU / XL si GPU) u `"0"` (modelo fijo `smart_semantycs_model_fixed`)
- Si no hay ruta subida para el tamaño requerido → `jsonify` error claro pidiendo subirlo (no auto-descarga, según lo acordado).

### 1.2 Inferencia con prompts de texto

```python
from ultralytics import YOLOE

model = YOLOE(model_path)      # yoloe-26n-seg.pt | yoloe-26x-seg.pt
model.to(get_device())
model.set_classes(classes)     # ["person", "bicycle"] — nombres LVIS, 1 sola vez por sesión
results = model.predict(frame) # o model.track(frame, persist=True, ...)
```

- `set_classes()` se llama **una sola vez al iniciar** la sesión (coherente con prompt bloqueado).
- **Tracking**: los docs indican soporte de `track`. Se implementa intentando `model.track(..., persist=True, tracker="bytetrack.yaml")`; si YOLOE no lo soporta → **fallback**: `model.predict()` + ByteTrack manual (`ultralytics.trackers.ByteTrack`) sobre las cajas. (Ver Fase de verificación, §8.)
- `agnostic_nms=True` ya es el default de YOLOE (evita duplicados entre clases del vocabulario amplio).
- Resultados: `boxes.xyxy`, `boxes.conf`, `boxes.cls`, `boxes.id` (si track) + `masks` (no se usan en v1).

---

## 2. Vocabulario LVIS

- Se extrae la lista de nombres del LVIS yaml (1200 clases, `0: aerosol can/spray can … 1202: zucchini/courgette`) a un archivo Python estático:
  `src/modules/smart_semantycs_vocab.py` → `LVIS_NAMES: list[str]` (1203 entradas) y helpers:
  - `normalize_name(name) -> str` (lower, strip)
  - `find_lvis_candidates(text) -> list[str]` (búsqueda por substring para ayudar al mapeo)
- La lista completa se incrusta (resumida/compactada) en el system prompt de Gemini para que el mapeo use **nombres reales** de LVIS y no invente.
- Fuente de verdad del vocabulario: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/lvis.yaml (se copia a mano al archivo de vocab; validar count=1203).

---

## 3. Interpretación con Gemini (generación de la "skill")

### 3.1 Endpoint
`POST /api/semantycs/sessions/<id>/interpret` → body `{ "prompt": "..." }`.

### 3.2 Flujo
1. Validar que la sesión tenga video/stream vinculado (si no → 400).
2. Guardar mensaje de usuario en la sesión.
3. Llamar a Gemini (reusar `_call_gemini` de `src/app.py:2255`, usando `gemini_api_key` de settings, temperatura 0.3, throttle global existente).
4. Gemini responde **solo JSON** según el contrato de §3.3.
5. **Validación estricta** en servidor (`_validate_skill`): si `feasible=false` → guardar mensaje del modelo explicando en lenguaje genérico que **no es posible detectar eso por el momento y que se requeriría un entrenamiento especializado** (sin nombres de tecnologías) y no arrancar nada.
6. Persistir la skill en la sesión; habilitar botón de iniciar.

### 3.3 Contrato JSON (la "plantilla/rompecabezas")

```json
{
  "feasible": true,
  "reason": "…",                       // solo si feasible=false (mensaje genérico)
  "summary": "Detectaré y contaré personas en bicicleta en todo el encuadre.",
  "detection_area": "full_frame",      // v1: solo full_frame
  "classes": ["person", "bicycle"],    // nombres LVIS exactos (máx 15)
  "counters": [                        // máx 5 — SIEMPRE sobre lo solicitado por el prompt
    {
      "id": "c_person_bicycle",
      "label": "Personas en bicicleta",
      "color": "#22C55E",
      "condition": {"detect": ["person"], "overlap": ["bicycle"], "min_overlap": 0.30}
    }
  ],
  "logs": [                            // eventos que generan log + captura — coherentes con el mismo prompt
    {
      "id": "l_person_bicycle",
      "label": "Persona en bicicleta detectada",
      "event": "person_bicycle_detected",
      "priority": "info",              // info | warning | critical
      "condition": {"detect": ["person"], "overlap": ["bicycle"], "min_overlap": 0.30}
    }
  ]
}
```

> **Regla de coherencia**: `classes`, `counters` y `logs` se derivan **del mismo prompt**.
> No deben mezclar conceptos distintos. Si el usuario pide "personas en bicicleta", tanto el
> contador como el log se refieren a persona+bicicleta. Si pide contarlo también quiere ver el
> evento/log de lo mismo; si solo pidiera "detectar y **alertar**" (sin contar) podrían generarse
> solo logs sin contadores. El LLM debe mantener todo alineado con lo que pidió.

- **Tipos de condition soportados en v1** (DSL cerrado, sin ejecución de código arbitrario):
  - `{ "detect": ["cls"...] }` → coincide si el track pertenece a alguna de esas clases LVIS.
  - `{ "detect": [...], "overlap": [...], "min_overlap": 0.3 }` → el objeto detectado **se solapa** ≥ min_overlap (IoU) con un objeto de las clases overlap (ej. persona sobre bicicleta).
  - (Extensible después: `line_cross`, `area_entry`…).
- **Contadores**: cuentan **tracks únicos** (por `track_id` + rule id) que alguna vez cumplen la condición → 1 por track, sin dobles.
- **Logs**: generan entrada de log + **captura** (frame completo con boxes dibujados, guardada en `static/uploads/captures/semantycs/<session_id>/`), con deduplicación por `track_id` + rule.
- Validaciones servidor: máx 5 contadores, ids únicos, clases existentes en LVIS, condición soportada, `feasible` coherente. Si algo no valida → se rechaza con mensaje al chat y se pide re-generar.

### 3.4 System prompt de Gemini (resumen de directivas)
- Eres el cerebro de interpretación del módulo de visión con vocabulario abierto.
- Recibes el prompt del usuario + la lista LVIS. Debes decidir:
  1. ¿Es factible detectar con el vocabulario disponible? Si pides geometría/imposibles (ej. "moneda triangular con 3 agujeros") → `feasible=false`.
  2. ¿Solo detectar o también contar? → si "cuenta" se usan contadores con track; si solo "detecta/alerta" se usan logs.
  3. Mapear a nombres LVIS exactos.
  4. Generar contadores (máx 5) y logs con condiciones válidas, **siempre alineados con el mismo prompt** (nada de contar una cosa y alertar otra distinta).
- Reglas: responder SOLO JSON válido, sin markdown, sin nombres de tecnologías en `summary`/`reason`.

---

## 4. Pipeline y Manager

Nuevo `src/modules/smart_semantycs.py`:

- `SmartSemntycsPipeline(BasePersistPipeline)`:
  - `__init__(session_id, source_path, source_type, classes, skill, conf, fps_limit)`
  - Carga YOLOE en el hilo, `set_classes(classes)`, abre `cv2.VideoCapture` (loop para video, salir si stream cae).
  - Por frame: inferencia+track → dibuja boxes con clase+conf+ID → evalúa contadores/logs (dedup por track) → guarda capturas → encola JPEG (`_frame`) para MJPEG.
  - **Pausa**: evento `_paused`. Para video: al reanudar sigue donde se quedó (no avanza el cap). Para stream: al reanudar se salta al tiempo actual simplemente no acumulando frames y leyendo el siguiente (sin buffer; el stream en vivo siempre es "actual").
  - **Reset** (con confirmación en UI): limpia contadores/logs/capturas y `reset_module_counters`/borrado de logs de la sesión (mantiene video+skill).
  - **Stop**: guarda contadores, libera modelo, `torch.cuda.empty_cache()`, `multi_release()`.
  - FPS limit reutilizando patrón `fps_limit` (timesleep) y `_persist_counters` cada ~1s.
- `SmartSemntycsManager` singleton:
  - dict `pipelines[session_id]`, lock thread-safe, mismo patrón que `PersonasManager`.
  - **Regla 1 activa (solo dentro de Smart Semantycs)**: en `start()` se detiene cualquier pipeline previo del propio módulo.
  - **Independiente del límite global de 4**: NO usa `multi_acquire`/`multi_release`. El contador global de 4 pipelines queda intacto para que los módulos de detección existentes (personas, armas, etc.) sigan permitiendo hasta 4 reproducciones simultáneas como hasta ahora.
  - Helpers: `get_frame_jpeg`, `get_stats`, `pause/resume/reset/stop`.

---

## 5. Base de datos (`src/database.py`)

Nuevas tablas (creadas en `init_db()`):

```sql
CREATE TABLE IF NOT EXISTS semantycs_sessions (
  id          TEXT PRIMARY KEY,            -- uuid
  title       TEXT NOT NULL DEFAULT 'Nueva sesión',
  video_path  TEXT,                        -- ruta relativa o RTSP
  video_type  TEXT CHECK(video_type IN ('video','stream')),
  prompt      TEXT DEFAULT '',
  skill       TEXT DEFAULT '{}',           -- JSON del contrato §3.3
  state       TEXT NOT NULL DEFAULT 'no_video',  -- no_video|video|prompted|running|stopped
  created_at  TEXT NOT NULL DEFAULT (datetime('now','localtime')),
  updated_at  TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);

CREATE TABLE IF NOT EXISTS semantycs_messages (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id  TEXT NOT NULL,
  role        TEXT NOT NULL CHECK(role IN ('user','model','system')),
  content     TEXT NOT NULL,
  kind        TEXT NOT NULL DEFAULT 'text',   -- text | skill | error
  created_at  TEXT NOT NULL DEFAULT (datetime('now','localtime')),
  FOREIGN KEY (session_id) REFERENCES semantycs_sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS semantycs_counters (
  session_id  TEXT NOT NULL,
  counter_id  TEXT NOT NULL,
  label       TEXT NOT NULL,
  color       TEXT NOT NULL DEFAULT '#22C55E',
  value       INTEGER NOT NULL DEFAULT 0,
  updated_at  TEXT NOT NULL DEFAULT (datetime('now','localtime')),
  PRIMARY KEY (session_id, counter_id),
  FOREIGN KEY (session_id) REFERENCES semantycs_sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS semantycs_logs (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id  TEXT NOT NULL,
  log_id      TEXT NOT NULL,               -- id de la rule
  label       TEXT NOT NULL,
  event       TEXT DEFAULT '',
  priority    TEXT NOT NULL DEFAULT 'info',
  capture_path TEXT,                       -- imagen de evidencia
  created_at  TEXT NOT NULL DEFAULT (datetime('now','localtime')),
  FOREIGN KEY (session_id) REFERENCES semantycs_sessions(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_semantycs_logs_session ON semantycs_logs(session_id, created_at DESC);
```

Funciones nuevas (estilo de las existentes): CRUD de sesiones/mensajes, `upsert_semantycs_counter`, `list_semantycs_counters`, `insert_semantycs_log`, `list_semantycs_logs` (ORDER BY created_at DESC, LIMIT ~200), `clear_semantycs_logs`, `clear_semantycs_counters`, `delete_semantycs_session` (cascade + borrado de capturas en disco).

---

## 6. API (en `src/app.py`)

| Ruta | Método | Descripción |
|---|---|---|
| `/smart-semantycs` | GET | Página del módulo |
| `/api/semantycs/sessions` | POST | Crear sesión |
| `/api/semantycs/sessions` | GET | Listar (historial) |
| `/api/semantycs/sessions/<id>` | GET/DELETE | Detalle+mensajes / borrar |
| `/api/semantycs/sessions/<id>/video` | POST | Vincular video subido o stream (valida estado `no_video`) |
| `/api/semantycs/sessions/<id>/interpret` | POST | Enviar prompt → Gemini → skill (valida video presente) |
| `/api/semantycs/sessions/<id>/start` | POST | Arranca pipeline + bloquea (`prompted→running`); detiene sesiones previas |
| `/api/semantycs/sessions/<id>/pause` | POST | Pausar (video: continúa donde iba; stream: salta a lo actual) |
| `/api/semantycs/sessions/<id>/resume` | POST | Reanudar |
| `/api/semantycs/sessions/<id>/reset` | POST | Reset detección con confirmación (limpia contadores/logs) |
| `/api/semantycs/sessions/<id>/stop` | POST | Detener pipeline (`running→stopped`) |
| `/api/semantycs/sessions/<id>/stream` | GET | MJPEG (`multipart/x-mixed-replace`) |
| `/api/semantycs/sessions/<id>/state` | GET | Estado + counters + últimos logs (polling 1–2s) |
| `/api/semantycs/sessions/<id>/logs/<log_row_id>` | GET | Evidencia para popup |
| `/api/semantycs/model/status` | GET | device (CPU/GPU), modelo activo (nano/XL), rutas configuradas |
| `/api/semantycs/upload-video` | POST | Reusa `/api/sources/upload-video` para subir archivo |

- Bloqueo de UI por estado (`no_video` → `video` → `prompted` → `running`):
  - `no_video`: input de prompt deshabilitado; botón "Cargar video" visible sobre la bandeja.
  - `video`: prompt habilitado, video enlazado pero no se reproduce.
  - `prompted`: skill listo; botón iniciar; video sigue sin reproducirse.
  - `running`: **bloqueado** — video, prompt y skill congelados; solo controles pause/resume/reset/stop.
- Rate limit igual que `/api/chat` (2s) y throttle Gemini existente.

---

## 7. Frontend

### 7.1 Layout `templates/smart_semantycs.html` (extiende `base.html`)

```
┌────────────────────────────────────────────────────────┐
│ sidebar (base) │  contadores (altura fija ~90px)   │ chat│
│ (menú general) ├────────────────────────────────────┤ (der│
│                │  VIDEO (flex-1, boxes + controles) │ ~300│
│                │  ── overlay transparente:          │ px) │
│                │     [⏸/▶] [⟲ Reset] [■ Stop]      │     │
│                ├────────────────────────────────────┤     │
│                │  LOGS (altura ~180px, scroll,      │     │
│                │  nuevos arriba, clic → popup)      │     │
└────────────────────────────────────────────────────────┘
```

- Chat derecho: historial de sesiones + "Nueva sesión semántica" + mensajes + input (mismo patrón visual de `index.html`/`chat.js` pero con la lógica de Smart Semantycs en `static/js/smart-semantycs.js`).
- Botón "Cargar video" encima del input cuando no hay video; tras subirlo se muestra chip del archivo vinculado.
- Animación de "interpretando…" (typing + spinner) mientras Gemini analiza; mismo estilo de `showTypingBubble`.
- Contadores: tarjetas con label, color y valor; vacías al inicio (span "Sin contadores").
- Logs: filas con prioridad (info/warning/critical → colores), hora, label; clic → popup con imagen de evidencia (reutilizar patrón `evidence-modal.js`).
- Polling `/state` cada 1.5s mientras `running`.

### 7.2 Settings — pestaña Smart Semantycs (`templates/settings.html`)

- Subir modelo **nano** (.pt) y modelo **XL** (.pt) → endpoints de subida que guardan ruta en settings.
- Selector: "Automático (nano en CPU / XL en GPU)" o "Modelo fijo" con selector nano/XL.
- Status card: device detectado (CPU/GPU), modelo que se usará, peso/ruta.
- Enlaces a pesos oficiales (urls de releases de Ultralytics ya documentadas).

### 7.3 Branding global "Smart-AI · por SmartCloud"

- `templates/base.html`: header del sidebar `{{ settings.system_name }}` → texto "Smart-AI · por SmartCloud"; footer version; item de nav nuevo **Smart Semantycs** (icono + link `/smart-semantycs`).
- `templates/index.html` (Sessions): actualizar títulos/welcome con el branding.
- Login overlay (`base.html`) y títulos de páginas: usar "Smart-AI · por SmartCloud".
- `src/database.py` DEFAULT_SETTINGS: `system_name` default → `Smart-AI`.

---

## 8. Archivos

**Nuevos**
| Archivo | Propósito |
|---|---|
| `src/modules/smart_semantycs.py` | Pipeline + Manager YOLOE |
| `src/modules/smart_semantycs_vocab.py` | `LVIS_NAMES` (1203) + helpers |
| `templates/smart_semantycs.html` | Vista del módulo (layout definido) |
| `static/js/smart-semantycs.js` | Lógica frontend |
| `static/css/smart-semantycs.css` | Estilos (o `<style>` en el template) |

**Modificados**
| Archivo | Cambio |
|---|---|
| `src/database.py` | 4 tablas + CRUD semántico + `system_name` default |
| `src/app.py` | Rutas §6 + `_validate_skill` + prompt de Gemini + wiring del manager |
| `templates/base.html` | Nav item + branding |
| `templates/settings.html` | Pestaña Smart Semantycs (modelos nano/XL) |
| `templates/index.html` | Branding |
| `requirements.txt` | Mantener `ultralytics>=8.4.0` (instalado 8.4.41, YOLOE ok) |

---

## 9. Orden de implementación

1. **Fase 0 — Verificación YOLOE**: script de prueba (fuera del repo o en `scripts/`) que cargue `yoloe-26n-seg.pt`, `set_classes(["person","bicycle"])` y pruebe `predict` y `track` sobre un frame; confirmar IDs de track. Decidir camino de tracking.
2. **Fase 1 — BD**: tablas + funciones en `database.py`; `system_name`.
3. **Fase 2 — Vocabulario**: `smart_semantycs_vocab.py` con las 1203 clases LVIS.
4. **Fase 3 — Módulo backend**: pipeline + manager + evaluador de skill (contadores/logs/dedup/capturas).
5. **Fase 4 — API**: rutas §6 + Gemini interpret (contrato §3) + validación.
6. **Fase 5 — Frontend**: template + CSS + JS (layout, chat, video, controles, logs, popup).
7. **Fase 6 — Settings**: pestaña Smart Semantycs (subida nano/XL, selector, status).
8. **Fase 7 — Branding**: base.html, index.html, login.
9. **Fase 8 — Pruebas E2E**: flujos de los ejemplos 1 y 2 del usuario; bloqueo; pause/resume; reset con confirmación; evidencia popup; 1 sesión activa.

---

## 10. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| `YOLOE.track()` no soportado/estable | Fase 0 lo valida; fallback ByteTrack manual sobre `predict()`. |
| YOLOE 26x pesado (196 GFLOPS) en GPU débil | fps_limit por fuente + opcional frame_step; XL solo en GPU. |
| CPU con 640px lento | nano + fps_limit ajustable (como módulos actuales). |
| Gemini devuelve JSON inválido | `_validate_skill` con retry (máx 2) pidiendo JSON puro; mensaje de error al chat si falla. |
| Clases inventadas fuera de LVIS | System prompt con vocabulario + `find_lvis_candidates`; validación de pertenencia. |
| Capturas duplicadas | Dedup por `track_id` + rule id en pipeline. |
| Subida de .pt grandes | Reusar `MODELS_FOLDER` y validación `.pt` (máx content length 500MB ya configurado). |

---

## 11. Notas de coherencia con lo acordado

- **Bloqueo total** tras iniciar: video, prompt y skill inmutables en estado `running`.
- **1 sesión activa** (regla exclusiva del módulo Smart Semantycs): `SmartSemntycsManager.start()` detiene pipelines previos del propio módulo; NO afecta el límite de 4 de los módulos existentes.
- **CPU nano / GPU XL**: automático por `get_device()`; configurable en Settings.
- **Máx 5 contadores** validado en servidor.
- **No se nombran tecnologías** en mensajes de chat al usuario (solo lenguaje general + sugerencia de entrenamiento especializado).
- **Stream**: pause→resume salta a lo actual sin buffer; **video**: reanuda donde se quedó.
- **Evidencia**: cada log genera captura; clic abre popup.
