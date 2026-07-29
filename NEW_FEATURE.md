# CVVision v3.0 — Persistencia, KPIs y Analytics

## Visión General

Refactorización completa del sistema para agregar **persistencia de datos por fuente**, **corrección de KPIs**, **módulo de Analytics** con dashboards especializados y **visor de evidencias** modal.

---

## 1. Base de Datos — Nuevo Esquema

### Tablas Genéricas

```sql
-- Contadores persistentes por fuente (cada módulo guarda aquí sus KPIs)
CREATE TABLE module_counters (
    module_id   TEXT    NOT NULL,
    source_id   INTEGER NOT NULL,
    counter_key TEXT    NOT NULL,
    int_value   INTEGER NOT NULL DEFAULT 0,
    float_value REAL    DEFAULT NULL,
    str_value   TEXT    DEFAULT NULL,
    updated_at  TEXT    NOT NULL DEFAULT (datetime('now','localtime')),
    PRIMARY KEY (module_id, source_id, counter_key)
);

-- Eventos / Alertas históricas por fuente
CREATE TABLE module_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    module_id    TEXT    NOT NULL,
    source_id    INTEGER NOT NULL,
    event_type   TEXT    NOT NULL,
    label        TEXT    DEFAULT '',
    description  TEXT    DEFAULT '',
    event_data   TEXT    NOT NULL DEFAULT '{}',
    capture_path TEXT    DEFAULT NULL,
    extra_paths  TEXT    DEFAULT NULL,     -- JSON array para múltiples capturas
    created_at   TEXT    NOT NULL DEFAULT (datetime('now','localtime'))
);

CREATE INDEX idx_counters_lookup ON module_counters(module_id, source_id);
CREATE INDEX idx_events_module  ON module_events(module_id, source_id);
CREATE INDEX idx_events_time    ON module_events(created_at);
CREATE INDEX idx_events_type    ON module_events(module_id, event_type);
```

### Tablas Específicas (se mantienen y mejoran)

- `reglamento_detections` — se agrega columna `missing_items TEXT`
- `carga_descarga_detections` — se mantiene igual

### Funciones de Acceso en `database.py`

```python
def save_module_counters(module_id, source_id, counters: dict) -> None
def load_module_counters(module_id, source_id) -> dict
def reset_module_counters(module_id, source_id) -> None

def insert_module_event(module_id, source_id, event_type, label='', description='',
                        event_data=None, capture_path=None, extra_paths=None) -> int
def get_module_events(module_id, source_id=None, event_type=None, days=7, limit=100) -> list
def get_module_analytics(module_id, source_id=None, days=7) -> dict
```

---

## 2. Corrección de KPIs por Módulo

### 2.1 Tanques de Gas (`tanques_gas.py`)

**Layout:** 2 pestañas

| Pestaña | Elementos |
|---------|-----------|
| **Principal** | Video feed + Entradas / Salidas + Acciones Aprendidas / Detectadas + Botón Enseñar + Estado Humo/Fuego + Alertas de Intrusión (log con capturas) |
| **Áreas Restringidas** | Editor de polígonos (existente) + lista de áreas |

**KPIs:**
| KPI | Tipo | Persiste |
|-----|------|----------|
| Entradas | Contador entero | ✅ |
| Salidas | Contador entero | ✅ |
| Acciones aprendidas | Contador entero | ✅ |
| Acciones detectadas | Contador entero | ✅ |
| Humo/Fuego detectado | Estado + contador | ✅ |
| Intrusiones | Conteo + eventos | ✅ |

**Eventos que genera:**
- `intrusion` — cuando se viola área restringida (captura completa)
- `smoke_detected` — cuando se detecta humo/fuego
- `action_detected` — cuando se detecta una acción aprendida

### 2.2 Reglamento (`reglamento.py`)

**Se mantiene todo igual**, solo se agrega:
- Columna `missing_items` en `reglamento_detections` → JSON con items faltantes
- En evidence popup: checklist mostrando qué items tenía presente y cuáles no
- Items a verificar: botas (con_botas / sin_botas) — no hay EPP aquí

### 2.3 Vehículos (`vehiculos.py`)

**KPIs finales:**
| KPI | Tipo |
|-----|------|
| Entradas (IN) | Contador |
| Salidas (OUT) | Contador |
| Placas detectadas | Contador |

Se elimina **Vehículos totales**.

### 2.4 Carga y Descarga (`carga_descarga.py`)

**KPIs finales:**
| KPI | Tipo | Nota |
|-----|------|------|
| Entrada (antes IN) | Contador | Renombrado |
| Salida (antes OUT) | Contador | Renombrado |

Se elimina **Objetos en escena**.

**Invertir contadores:** Al hacer clic en invertir:
1. Intercambia los valores actuales: `temp = entrada; entrada = salida; salida = temp`
2. La lógica de conteo se invierte (lo que era entrada ahora es salida y viceversa)
3. Los contadores persisten inmediatamente con el nuevo estado

### 2.5 Detección de Personas (`personas.py`)

**KPIs finales:**
| KPI Actual | Nuevo |
|------------|-------|
| Personas dentro | Personas en el área |
| Entradas (IN) | Mantener |
| Salidas (OUT) | Mantener |
| Permanencia máx/min | Mantener |

### 2.6 Armas (`armas.py`)

**KPIs finales:**
| KPI | Tipo |
|-----|------|
| Alertas de arma | Contador de eventos |
| Armas blancas detectadas | Contador acumulado |
| Armas de fuego detectadas | Contador acumulado |
| Capturas de rostro | Contador |

Se elimina **Armas únicas en escena**.

### 2.7 Acciones (`acciones.py`)

**KPIs finales:**
| KPI | Tipo |
|-----|------|
| Alertas de violencia | Contador |
| Alertas de robo/amenaza | Contador |
| Alertas de actividad sospechosa | Contador |
| Alertas de uso de celular | Contador |
| Alertas de caídas | Contador |
| Acciones enseñadas | Contador (teach samples guardados) |

Se elimina **Personas en escena**.

### 2.8 Troncos (`troncos.py`)

**KPIs finales:** Solo **Conteo total**. Se elimina **Troncos en escena**.

### 2.9 Pallets (`pallets.py`)

**KPIs finales:** Solo **Conteo total**. Se elimina **Pallets en escena**.

### 2.10 Cajas (`cajas.py`)

**KPIs finales:** Solo **Conteo total**. Se elimina **Objetos en escena**.

### 2.11 Humo/Fuego (`smoke.py`)

**Se mantiene todo igual.**

### 2.12 EPP (`epp.py`)

**KPIs finales:**
| KPI | Tipo |
|-----|------|
| Personas sin EPP completo | Contador |
| Alertas generadas | Contador |
| Compliance rate | Porcentaje |

**Evidence popup:** Checklist dinámico basado en `EPP_CLASS_MAP` — muestra checkmarks solo de los ítems que están activos en la configuración de la fuente (clases de EPP requeridas).

---

## 3. Persistencia de Contadores

### Mecanismo General

Cada pipeline sigue este ciclo de vida con persistencia:

```
Pipeline.start()
  → load_module_counters(module_id, source_id) → restaura contadores
  → inicia thread

Pipeline._run() [cada frame]
  → procesa frame, actualiza contadores en memoria
  → [cada 1 segundo] save_module_counters(module_id, source_id, counters)

Pipeline.stop()
  → save_module_counters(module_id, source_id, counters)  # guardado final
  → join thread
```

### Clase Base (`BasePersistPipeline` en `base.py`)

```python
class BasePersistPipeline:
    """Mixin para pipelines que necesitan persistencia."""

    def _init_persistence(self, module_id, source_id):
        self._module_id = module_id
        self._source_id = source_id
        self._last_persist = 0
        self._persist_interval = 1.0  # segundos

    def _persist_counters(self, counters: dict):
        """Guarda contadores a DB."""
        now = time.time()
        if now - self._last_persist < self._persist_interval:
            return
        self._last_persist = now
        save_module_counters(self._module_id, self._source_id, counters)

    def _load_counters(self) -> dict:
        """Carga contadores desde DB."""
        return load_module_counters(self._module_id, self._source_id)
```

### Configuraciones Geométricas por Fuente

Las configuraciones como líneas de conteo, áreas, modos de línea, etc. deben persistirse **por fuente**, no globalmente. Se usará una tabla adicional:

```sql
CREATE TABLE source_configs (
    source_id INTEGER NOT NULL,
    config_key TEXT NOT NULL,
    config_value TEXT NOT NULL,
    PRIMARY KEY (source_id, config_key)
);
```

O bien, se puede almacenar como JSON en una columna `config` de la tabla `sources`. Opción recomendada:

```sql
ALTER TABLE sources ADD COLUMN config TEXT NOT NULL DEFAULT '{}';
```

Donde `config` es un JSON con todas las configuraciones específicas de esa fuente. Ejemplo:
```json
{
    "line_y": 85,
    "area_x1": 25,
    "area_y1": 25,
    "area_x2": 75,
    "area_y2": 75,
    "line_mode": "horizontal",
    "line_pos": 50
}
```

---

## 4. Módulo de Analytics

### 4.1 Sidebar y Ruteo

- Nuevo item en sidebar: **📊 Analytics** (entre el último módulo y "Sistema")
- Ruta principal: `/analytics`
- Rutas hijas: `/analytics/<module_id>`

### 4.2 Página Principal (`templates/analytics/index.html`)

```
┌─────────────────────────────────────────────────────┐
│  Analytics · Métricas del Sistema                   │
│                                                     │
│  [Buscar módulo...]                                 │
│                                                     │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐     │
│  │👤    │ │🔫    │ │🏃    │ │🪵    │ │📦    │     │
│  │Pers. │ │Armas │ │Acc.  │ │Tron. │ │Cajas │     │
│  │123   │ │45    │ │67    │ │89    │ │101   │     │
│  │ IN   │ │Alert │ │Vio   │ │Total │ │Total │     │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘     │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐     │
│  │📦    │ │👢    │ │🚚    │ │🚗    │ │🔥    │     │
│  │Pall. │ │Regl. │ │C/Des │ │Veh.  │ │Humo  │     │
│  │...   │ │...   │ │...   │ │...   │ │...   │     │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘     │
│                                                     │
│  ⚡ En vivo ahora: 3 pipelines activos              │
└─────────────────────────────────────────────────────┘
```

- Cada card muestra: icono, nombre, KPI principal, indicador de actividad (si pipeline activo → animación)
- Al hacer clic → dashboard especializado
- Polling cada 3s para actualizar KPIs de pipelines activos

### 4.3 Dashboard por Módulo

Cada dashboard se genera en base a eventos y contadores. Ejemplos:

#### Dashboard Personas (`/analytics/personas`)
```
┌─ KPIs ─────────────────────────────────────────────┐
│  👥 Personas en área: 5   📥 Entradas: 120         │
│  📤 Salidas: 115          ⏱ Permanencia: 2m 34s    │
└─────────────────────────────────────────────────────┘
┌─ Timeline: Flujo de personas ──────────────────────┐
│  [Gráfica Chart.js: línea de entradas/salidas/área  │
│   por hora, últimos 7 días]                         │
└─────────────────────────────────────────────────────┘
┌─ Actividad por hora ───────────────────────────────┐
│  [Gráfica Chart.js: heatmap de densidad por hora    │
│   del día]                                          │
└─────────────────────────────────────────────────────┘
```

#### Dashboard Armas (`/analytics/armas`)
```
┌─ KPIs ─────────────────────────────────────────────┐
│  🔫 Alertas: 12    🔪 Blancas: 5     💥 Fuego: 7   │
│  📸 Capturas rostro: 8                             │
└─────────────────────────────────────────────────────┘
┌─ Timeline: Alertas de armas ───────────────────────┐
│  [Gráfica Chart.js: barras apiladas blanca/fuego    │
│   por día]                                          │
└─────────────────────────────────────────────────────┘
┌─ Últimas alertas ──────────────────────────────────┐
│  [Lista con timestamp, tipo, miniatura → popup]     │
└─────────────────────────────────────────────────────┘
```

#### Dashboard Reglamento (`/analytics/reglamento`)
```
┌─ KPIs ─────────────────────────────────────────────┐
│  👢 Con botas: 85%   ❌ Sin botas: 15%              │
│  ✅ Cumplimientos: 50   ⛔ Incumplimientos: 10      │
└─────────────────────────────────────────────────────┘
┌─ Timeline: Compliance ─────────────────────────────┐
│  [Gráfica Chart.js: línea de compliance % por día]  │
└─────────────────────────────────────────────────────┘
┌─ Evidencias recientes ────────────────────────────┐
│  [Lista con miniatura, timestamp → popup con        │
│   checklist de items presentes/ausentes]            │
└─────────────────────────────────────────────────────┘
```

#### Dashboard Tanques de Gas (`/analytics/tanques_gas`)
```
┌─ KPIs ─────────────────────────────────────────────┐
│  📥 Entradas: 45   📤 Salidas: 40                  │
│  🚫 Intrusiones: 3  🔥 Humo: 1  🎯 Acciones: 12   │
└─────────────────────────────────────────────────────┘
┌─ Timeline multicapa ───────────────────────────────┐
│  [Gráfica Chart.js: múltiples líneas (entradas,     │
│   intrusiones, humo) en diferentes colores]         │
└─────────────────────────────────────────────────────┘
┌─ Alertas de intrusión ────────────────────────────┐
│  [Lista con captura completa, timestamp → popup]    │
└─────────────────────────────────────────────────────┘
┌─ Detecciones de humo ─────────────────────────────┐
│  [Lista con captura, timestamp → popup]             │
└─────────────────────────────────────────────────────┘
```

#### Dashboard Vehículos (`/analytics/vehiculos`)
```
┌─ KPIs ─────────────────────────────────────────────┐
│  📥 Entradas: 67   📤 Salidas: 62                  │
│  🏷 Placas detectadas: 23                          │
└─────────────────────────────────────────────────────┘
┌─ Timeline: Flujo vehicular ───────────────────────┐
│  [Gráfica Chart.js: barras IN/OUT por hora/día]    │
└─────────────────────────────────────────────────────┘
┌─ Últimas placas reconocidas ──────────────────────┐
│  [Lista: ABCD-123 | Entrada | 14:32:15 → popup     │
│   con captura del vehículo + crop de placa]         │
└─────────────────────────────────────────────────────┘
```

### 4.4 API de Analytics

```python
GET /api/analytics/<module_id>/summary?days=7
→ {
    "counters": { ... },       # contadores actuales
    "daily": [                 # agregación por día
        {"day": "2026-07-20", "entradas": 10, "salidas": 8, ...}
    ],
    "events": [ ... ],         # últimos 50 eventos
    "active": true/false,      # pipeline activo?
    "sources": [ ... ]         # fuentes configuradas
  }

GET /api/analytics/<module_id>/events?days=7&source_id=X&event_type=Y&limit=50
→ { "events": [ ... ] }

GET /api/analytics/<module_id>/timeline?days=7&granularity=hour
→ { "data": [ {"label": "2026-07-27 14:00", "entradas": 5, ...} ] }

GET /api/evidences/<module_id>/<event_id>
→ { "event": { ... }, "capture_url": "...", "extra_urls": [...] }
```

### 4.5 Chart.js Integración

- Cargar Chart.js v4 desde CDN: `https://cdn.jsdelivr.net/npm/chart.js`
- Gráficas: bar, line, doughnut (para compliance %), stacked bar
- Timeline con zoom (Chart.js zoom plugin opcional)
- Tema coherente con la paleta del sistema (#182A55, #22C55E, #EF4444, etc.)

---

## 5. Modal de Evidencias

### Componente Reutilizable

```html
<!-- templates/components/evidence_modal.html -->
<div class="ev-modal-overlay" id="ev-modal" hidden>
  <div class="ev-modal-container">
    <div class="ev-modal-header">
      <h3 class="ev-modal-title" id="ev-title">Evidencia</h3>
      <button class="ev-modal-close" id="ev-close">&times;</button>
    </div>
    <div class="ev-modal-body">
      <div class="ev-modal-image">
        <img id="ev-image" src="" alt="Evidencia">
      </div>
      <div class="ev-modal-sidebar">
        <div class="ev-meta">
          <div class="ev-meta-row">
            <span class="ev-meta-label">Módulo</span>
            <span class="ev-meta-value" id="ev-module"></span>
          </div>
          <div class="ev-meta-row">
            <span class="ev-meta-label">Fuente</span>
            <span class="ev-meta-value" id="ev-source"></span>
          </div>
          <div class="ev-meta-row">
            <span class="ev-meta-label">Fecha</span>
            <span class="ev-meta-value" id="ev-date"></span>
          </div>
          <div class="ev-meta-row">
            <span class="ev-meta-label">Hora</span>
            <span class="ev-meta-value" id="ev-time"></span>
          </div>
          <div class="ev-meta-row">
            <span class="ev-meta-label">Tipo</span>
            <span class="ev-meta-value" id="ev-type"></span>
          </div>
        </div>
        <div class="ev-checklist" id="ev-checklist" hidden>
          <h4>Elementos detectados</h4>
          <ul id="ev-checklist-list"></ul>
        </div>
        <div class="ev-description" id="ev-description"></div>
        <div class="ev-extra-images" id="ev-extra" hidden>
          <h4>Capturas adicionales</h4>
          <div class="ev-extra-grid" id="ev-extra-grid"></div>
        </div>
      </div>
    </div>
  </div>
</div>
```

### Funcionalidad
- Abre al hacer clic en cualquier evidencia (en live view o analytics)
- Carga datos vía `GET /api/evidences/<module>/<event_id>`
- Muestra imagen principal + metadata + checklist si aplica
- Para **Reglamento**: checklist con check/cross de items EPP
- Para **EPP**: checklist dinámico con los ítems activos en configuración
- Para **Armas**: muestra captura de arma + captura de rostro (extra_paths)
- Para **Intrusiones**: muestra captura completa del incidente

---

## 6. Integración en Live Views

### Estructura de Pestañas para Tanques de Gas

En `tanques_gas_live.html`:

```
┌─ Topbar ────────────────────────────────────────────┐
│  ← Volver | Tanque 1 | [video] | [● En vivo]       │
│  [Reset] [Stop]                                     │
├─────────────────────────────────────────────────────┤
│  [📊 Principal] [📍 Áreas Restringidas]             │
├──────────┬──────────────────────────────────────────┤
│          │  ┌─ KPIs ──────────────────────────┐     │
│          │  │ 📥 Entradas: 45  📤 Salidas: 40 │     │
│          │  │ 🎯 Acciones: 12                  │     │
│  VIDEO   │  │ 📚 Aprendidas: 5                 │     │
│  FEED    │  │ 🔥 Humo: ● Monitoreando          │     │
│          │  └──────────────────────────────────┘     │
│          │  [🧠 Enseñar acción]                      │
│          │                                           │
│          │  ── Alertas ──                            │
│          │  │ 🚫 Intrusión | 14:32:15 | [👁] │     │
│          │  │ 🚫 Intrusión | 14:30:01 | [👁] │     │
│          │  │ 🔥 Humo detectado | 13:15:00 | [👁] │ │
│          └──────────────────────────────────────────┘
```

### Evidence Click → Modal

En todas las listas de alertas/evidencias, cada item tiene un botón "👁" que abre el modal de evidencia con la captura completa.

```javascript
async function openEvidence(moduleId, eventId) {
    const res = await fetch(`/api/evidences/${moduleId}/${eventId}`);
    const data = await res.json();
    // Poblar modal
    document.getElementById('ev-image').src = data.capture_url;
    document.getElementById('ev-module').textContent = data.event.module_id;
    document.getElementById('ev-source').textContent = data.event.source_id;
    document.getElementById('ev-date').textContent = data.event.date || '—';
    document.getElementById('ev-time').textContent = data.event.time || data.event.created_at;
    document.getElementById('ev-type').textContent = data.event.event_type;
    // Mostrar checklist si existe
    if (data.event.event_data?.checklist) {
        showChecklist(data.event.event_data.checklist);
    }
    // Mostrar modal
    document.getElementById('ev-modal').hidden = false;
}
```

---

## 7. Persistencia de Configuraciones por Fuente

### `sources.config` (JSON)

Cada fuente almacena su configuración específica. Al guardar settings de línea/área:

```python
# En app.py, al setear line-y:
@app.route("/api/personas/sources/<int:source_id>/line-y", methods=["POST"])
def personas_line_y(source_id):
    data = request.get_json(silent=True) or {}
    pct = max(0, min(100, int(data.get("pct", 85))))
    # Guardar global (compatibilidad)
    set_setting("personas_line_y", str(pct))
    # Guardar por fuente
    save_source_config(source_id, "personas_line_y", str(pct))
    PersonasManager.get().set_line_y(source_id, pct)
    return jsonify({"line_y_pct": pct})
```

Al iniciar un pipeline, se cargan las configuraciones de fuente + global como fallback:

```python
def _load_source_config(source_id, key, default=None):
    config = get_source_config(source_id)
    if key in config:
        return config[key]
    # Fallback a global
    return get_settings().get(key, default)
```

### Funciones en `database.py`

```python
def save_source_config(source_id: int, key: str, value: str) -> None
def get_source_config(source_id: int) -> dict
def get_source_config_value(source_id: int, key: str, default=None) -> str
def delete_source_config(source_id: int) -> None  # al eliminar fuente
```

---

## 8. Estructura Modular para Nuevos Módulos

### Para agregar un nuevo módulo, se necesita:

```
1. src/modules/nuevo_modulo.py
   - Clase NuevoPipeline(BasePersistPipeline)
   - Clase NuevoManager(BaseManager)

2. src/database.py
   - Agregar a MODULES_META con functions y labels
   - Agregar DEFAULT_SETTINGS si aplica

3. src/app.py   (o refactorizar a fábrica de rutas)
   - Acaso: usar un decorador/fábrica que registre rutas automáticamente:
     register_module_routes(app, "nuevo_modulo", NuevoManager)

4. templates/nuevo_modulo_live.html
   - Template de live view con KPIs, controles, evidencias

5. templates/analytics/nuevo_modulo.html
   - Dashboard de analytics

6. La persistencia funciona automáticamente vía module_counters y module_events
```

### Fábrica de Rutas Propuesta

```python
def register_module_routes(app, module_id, manager_class, extra_routes=None):
    """Registra las rutas estándar (start/stop/stats/reset/stream) para un módulo."""

    @app.route(f"/api/{module_id}/sources/<int:source_id>/start", methods=["POST"])
    def module_start(source_id):
        ...

    @app.route(f"/api/{module_id}/sources/<int:source_id>/stop", methods=["POST"])
    def module_stop(source_id):
        ...

    # ... etc

    if extra_routes:
        extra_routes(app)
```

Esto reduce `app.py` de ~1561 líneas a ~400 líneas y hace trivial agregar nuevos módulos.

---

## 9. Resumen de Cambios por Archivo

### `src/database.py`
- Agregar tablas: `module_counters`, `module_events`
- Modificar `sources`: agregar columna `config TEXT DEFAULT '{}'`
- Agregar funciones: `save_module_counters`, `load_module_counters`, `reset_module_counters`
- Agregar funciones: `insert_module_event`, `get_module_events`, `get_module_analytics`
- Agregar funciones: `save_source_config`, `get_source_config`, `get_source_config_value`

### `src/modules/base.py`
- Agregar `BasePersistPipeline` (mixin/herencia)
- Agregar helpers de persistencia

### `src/app.py`
- Refactorizar con `register_module_routes()` (reducir código repetitivo)
- Agregar rutas de analytics API
- Agregar ruta de evidencias API

### `src/modules/*.py` (todos los módulos)
- Heredar de `BasePersistPipeline`
- Integrar carga/guardado de contadores
- Corregir KPIs según sección 2
- Guardar eventos al detectar alertas

### `templates/*_live.html` (todos)
- Corregir KPIs mostrados
- Agregar modal de evidencias (incluir componente)
- Agregar event listeners para abrir evidencias

### `templates/tanques_gas_live.html`
- Refactorizar a 2 pestañas (Principal / Áreas Restringidas)
- Integrar humo/fuego y acciones en pestaña principal
- Botón enseñar visible

### `templates/base.html`
- Agregar "📊 Analytics" al sidebar

### Archivos Nuevos

| Archivo | Propósito |
|---------|-----------|
| `templates/analytics.html` | Página principal de Analytics (grid de módulos) |
| `templates/analytics_personas.html` | Dashboard Personas |
| `templates/analytics_armas.html` | Dashboard Armas |
| `templates/analytics_acciones.html` | Dashboard Acciones |
| `templates/analytics_troncos.html` | Dashboard Troncos |
| `templates/analytics_pallets.html` | Dashboard Pallets |
| `templates/analytics_cajas.html` | Dashboard Cajas |
| `templates/analytics_reglamento.html` | Dashboard Reglamento |
| `templates/analytics_carga_descarga.html` | Dashboard Carga/Descarga |
| `templates/analytics_epp.html` | Dashboard EPP |
| `templates/analytics_smoke.html` | Dashboard Humo/Fuego |
| `templates/analytics_vehiculos.html` | Dashboard Vehículos |
| `templates/analytics_tanques_gas.html` | Dashboard Tanques de Gas |
| `templates/components/evidence_modal.html` | Modal de evidencias reutilizable |
| `static/js/analytics.js` | Lógica de Chart.js y polling de analytics |
| `static/js/evidence-modal.js` | Lógica del modal de evidencias |

---

## 10. Orden de Implementación Sugerido

1. **Fase 1 — Base de datos**: crear tablas, funciones, migración
2. **Fase 2 — Clase base**: `BasePersistPipeline` + helpers
3. **Fase 3 — Módulos individuales**: uno por uno, integrar persistencia y corregir KPIs
4. **Fase 4 — Source config**: persistencia de configuraciones por fuente
5. **Fase 5 — Modal evidencias**: componente + integración en live views
6. **Fase 6 — Refactor app.py**: fábrica de rutas
7. **Fase 7 — Analytics**: backend (API) + frontend (templates + Chart.js)
8. **Fase 8 — Sidebar + navegación**: integrar analytics en el menú

---

## 11. Dependencias Nuevas

- **Chart.js v4** (CDN) — gráficas en analytics. No requiere npm, se carga desde CDN en templates.
- Sin otras dependencias externas. Todo el resto es Python estándar + paquetes existentes.

---

## Notas Técnicas Adicionales

- **Thread safety**: El acceso a DB debe usar `get_conn()` que ya es thread-safe (cada llamada abre su propia conexión).
- **Rendimiento**: Las escrituras a `module_counters` cada 1s por pipeline son ligeras (UPDATE con PK). Para `module_events` son INSERTs que también son rápidos.
- **Limpieza histórica**: Se puede agregar un job opcional que limpie eventos > 30 días.
- **Migración**: La base de datos existente `cvvision.db` se migra con `ALTER TABLE` y creación de nuevas tablas (no destructivo).
