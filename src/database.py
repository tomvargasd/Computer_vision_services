import sqlite3
import os
import json

from src.config import DB_PATH, APP_VERSION


MODULES_META = {
    "personas": {
        "label": "Detección de Personas",
        "functions": {
            "conteo":      {"label": "Conteo de personas",       "description": "Cuenta el número total de personas en escena"},
            "permanencia": {"label": "Tiempo de permanencia",    "description": "Mide el tiempo que cada persona permanece en zona"},
            "heatmap":     {"label": "Mapa de calor",            "description": "Genera mapa de densidad de movimiento"},
        },
    },
    "armas": {
        "label": "Detección de Armas",
        "functions": {
            "deteccion_arma": {"label": "Detección de arma",             "description": "Detecta presencia de armas en escena"},
            "captura_rostro": {"label": "Captura automática de rostro",  "description": "Captura y guarda foto del portador del arma"},
            "tipo_arma":      {"label": "Clasificar tipo de arma",       "description": "Clasifica entre arma blanca o arma de fuego", "locked": True},
        },
    },
    "acciones": {
        "label": "Detección de Acciones",
        "functions": {
            "deteccion_acciones":  {"label": "Detección de acciones",        "description": "Dibuja esqueleto de pose COCO-17 sobre cada persona"},
            "deteccion_violencia": {"label": "Alertas de violencia",         "description": "Detecta golpes, patadas y agresión física"},
            "deteccion_robo":      {"label": "Alertas de robo / amenaza",   "description": "Detecta brazo apuntando y posturas de amenaza"},
            "deteccion_sospechosa":{"label": "Actividad sospechosa",        "description": "Detecta agachado, rastreo y movimientos furtivos"},
            "deteccion_celular":  {"label": "Uso de celular",            "description": "Detecta personas usando teléfono (llamando o texteando)"},
            "deteccion_caida":    {"label": "Detección de caídas",       "description": "Detecta caídas parciales (tropiezo/arrodillado) y completas"},
        },
    },
    "troncos": {
        "label": "Conteo de Troncos",
        "functions": {
            "conteo": {"label": "Conteo de troncos", "description": "Cuenta troncos que cruzan la línea vertical"},
        },
    },
    "pallets": {
        "label": "Conteo de Pallets",
        "functions": {
            "conteo": {"label": "Conteo de pallets", "description": "Cuenta pallets dentro del área de reconocimiento"},
        },
    },
    "cajas": {
        "label": "Conteo de Cajas",
        "functions": {
            "conteo": {"label": "Conteo de cajas", "description": "Cuenta cajas que cruzan la línea horizontal"},
        },
    },
    "reglamento": {
        "label": "Detección de Reglamento",
        "functions": {
            "deteccion_botas": {"label": "Detección de botas",  "description": "Detecta si las personas usan botas dentro del área"},
            "conteo_tiempo":   {"label": "Conteo de tiempo",    "description": "Mide el tiempo de permanencia en el área"},
            "alertas":         {"label": "Alertas",             "description": "Genera alertas visuales por incumplimiento"},
            "analytics":       {"label": "Analytics",           "description": "Estadísticas y gráficos del módulo"},
        },
    },
    "epp": {
        "label": "Detección de EPP",
        "functions": {
            "deteccion_epp": {"label": "Detección de EPP",       "description": "Detecta equipo de protección personal en personas"},
            "alertas":       {"label": "Alertas sin EPP",        "description": "Captura y alerta cuando una persona no tiene EPP completo"},
            "analytics":     {"label": "Analytics",              "description": "Ranking de EPP más usado y estadísticas"},
        },
    },
    "carga_descarga": {
        "label": "Detección de Carga y Descarga",
        "functions": {
            "conteo":   {"label": "Conteo de carga/descarga", "description": "Cuenta objetos que cruzan la línea de conteo"},
            "analytics":{"label": "Analytics",                "description": "Estadísticas y gráficos del módulo"},
        },
    },
    "smoke": {
        "label": "Detección de Humo/Fuego",
        "functions": {
            "deteccion_humo": {"label": "Detección de humo/fuego", "description": "Detecta presencia de humo o fuego en escena"},
        },
    },
    "vehiculos": {
        "label": "Reconocimiento de Vehículos",
        "functions": {
            "conteo":           {"label": "Conteo de vehículos",     "description": "Cuenta vehículos que cruzan la línea de conteo"},
            "deteccion_placas": {"label": "Detección de placas",    "description": "Reconoce placas en vehículos detectados"},
        },
    },
    "tanques_gas": {
        "label": "Tanques de Gas",
        "functions": {
            "conteo":              {"label": "Conteo de tanques",               "description": "Cuenta tanques que cruzan línea o ingresan a área"},
            "deteccion_acciones":  {"label": "Detección de acciones",           "description": "Detección de acciones con sistema de enseñanza"},
            "deteccion_humo":      {"label": "Detección de humo/fuego",         "description": "Detecta presencia de humo o fuego en escena"},
            "areas_restringidas":  {"label": "Áreas restringidas",              "description": "Define áreas prohibidas para personas o tanques"},
        },
    },
}


_FPS_MODULES = [
    "personas", "armas", "acciones", "troncos", "pallets",
    "cajas", "reglamento", "carga_descarga", "epp", "smoke", "vehiculos",
    "tanques_gas",
]
FPS_DEFAULTS = [
    (f"{m}_fps_limit_video", "0.02")
    for m in _FPS_MODULES
] + [
    (f"{m}_fps_limit_stream", "0.0")
    for m in _FPS_MODULES
]

DEFAULT_SETTINGS = [
    ("system_name", "Computer Vision"), ("version", APP_VERSION), ("logo", ""),
    ("multi_detection", "0"),
    *FPS_DEFAULTS,
    ("armas_model", ""), ("personas_model", ""), ("acciones_model", ""),
    ("troncos_model", ""), ("pallets_model", ""), ("cajas_model", ""),
    ("reglamento_model", ""),
    ("epp_model", ""),
    ("epp_conf", "0.35"),
    ("epp_half", "0"),
    ("personas_conf", "0.35"), ("armas_conf", "0.20"), ("acciones_conf", "0.35"),
    ("troncos_conf", "0.35"), ("pallets_conf", "0.35"), ("cajas_conf", "0.35"),
    ("reglamento_conf", "0.45"),
    ("personas_half", "0"), ("armas_half", "0"), ("acciones_half", "0"),
    ("troncos_half", "0"), ("pallets_half", "0"), ("cajas_half", "0"),
    ("reglamento_half", "0"),
    ("personas_line_y", "85"), ("troncos_line_x", "50"),
    ("pallets_area_x1", "25"), ("pallets_area_y1", "25"),
    ("pallets_area_x2", "75"), ("pallets_area_y2", "75"),
    ("pallets_classes", "0,1,2,3"),
    ("cajas_line_y", "85"),
    ("reglamento_area_x1", "30"), ("reglamento_area_y1", "30"),
    ("reglamento_area_x2", "70"), ("reglamento_area_y2", "70"),
    ("reglamento_min_time", "10"),
    ("reglamento_jpeg_q", "72"),
    ("reglamento_max_dim", "0"),
    ("reglamento_frame_step", "1"),
    ("carga_descarga_conf", "0.35"), ("carga_descarga_half", "0"),
    ("carga_descarga_line_mode", "horizontal"),
    ("carga_descarga_line_pos", "50"),
    ("carga_descarga_cur_model", "0"),
    ("carga_descarga_models", "[]"),
    ("smoke_model", ""),
    ("smoke_conf", "0.35"),
    ("smoke_half", "0"),
    ("vehiculos_model", ""),
    ("vehiculos_conf", "0.35"),
    ("vehiculos_half", "0"),
    ("vehiculos_plate_model", ""),
    ("vehiculos_plate_conf", "0.35"),
    ("vehiculos_line_mode", "horizontal"),
    ("vehiculos_line_pos", "50"),
    ("tanques_gas_model", ""),
    ("tanques_gas_conf", "0.35"),
    ("tanques_gas_half", "0"),
    ("tanques_gas_pose_model", ""),
    ("tanques_gas_smoke_model", ""),
    ("tanques_gas_line_mode", "horizontal"),
    ("tanques_gas_line_pos", "50"),
]


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS settings (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS modules (
                module_id TEXT PRIMARY KEY,
                enabled   INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS module_functions (
                module_id TEXT NOT NULL,
                func_id   TEXT NOT NULL,
                enabled   INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (module_id, func_id)
            );

            CREATE TABLE IF NOT EXISTS sources (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                module_id  TEXT    NOT NULL,
                name       TEXT    NOT NULL,
                type       TEXT    NOT NULL CHECK(type IN ('video','stream')),
                path       TEXT    NOT NULL,
                created_at TEXT    NOT NULL DEFAULT (datetime('now','localtime'))
            );

            CREATE TABLE IF NOT EXISTS reglamento_detections (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id        INTEGER NOT NULL,
                track_id         INTEGER NOT NULL,
                boot_status      TEXT    NOT NULL,
                time_compliance  TEXT    NOT NULL,
                seconds_in_area  REAL    NOT NULL,
                capture_path     TEXT,
                created_at       TEXT    NOT NULL DEFAULT (datetime('now','localtime'))
            );

            CREATE TABLE IF NOT EXISTS carga_descarga_detections (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id        INTEGER NOT NULL,
                track_id         INTEGER NOT NULL,
                direction        TEXT    NOT NULL,
                model_id         TEXT,
                created_at       TEXT    NOT NULL DEFAULT (datetime('now','localtime'))
            );

            -- CVVision v3.0: persistencia de contadores por fuente
            CREATE TABLE IF NOT EXISTS module_counters (
                module_id   TEXT    NOT NULL,
                source_id   INTEGER NOT NULL,
                counter_key TEXT    NOT NULL,
                int_value   INTEGER NOT NULL DEFAULT 0,
                float_value REAL    DEFAULT NULL,
                str_value   TEXT    DEFAULT NULL,
                updated_at  TEXT    NOT NULL DEFAULT (datetime('now','localtime')),
                PRIMARY KEY (module_id, source_id, counter_key)
            );

            -- CVVision v3.0: eventos / alertas históricas por fuente
            CREATE TABLE IF NOT EXISTS module_events (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                module_id    TEXT    NOT NULL,
                source_id    INTEGER NOT NULL,
                event_type   TEXT    NOT NULL,
                label        TEXT    DEFAULT '',
                description  TEXT    DEFAULT '',
                event_data   TEXT    NOT NULL DEFAULT '{}',
                capture_path TEXT    DEFAULT NULL,
                extra_paths  TEXT    DEFAULT NULL,
                created_at   TEXT    NOT NULL DEFAULT (datetime('now','localtime'))
            );

            CREATE INDEX IF NOT EXISTS idx_counters_lookup ON module_counters(module_id, source_id);
            CREATE INDEX IF NOT EXISTS idx_events_module  ON module_events(module_id, source_id);
            CREATE INDEX IF NOT EXISTS idx_events_time    ON module_events(created_at);
            CREATE INDEX IF NOT EXISTS idx_events_type    ON module_events(module_id, event_type);
        """)

        # Migraciones v3.0 (no destructivas)
        try:
            conn.execute("ALTER TABLE sources ADD COLUMN config TEXT NOT NULL DEFAULT '{}'")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE reglamento_detections ADD COLUMN missing_items TEXT DEFAULT NULL")
        except Exception:
            pass

        for k, v in DEFAULT_SETTINGS:
            conn.execute("INSERT OR IGNORE INTO settings(key,value) VALUES(?,?)", (k, v))

        for mod_id, meta in MODULES_META.items():
            conn.execute("INSERT OR IGNORE INTO modules(module_id,enabled) VALUES(?,0)", (mod_id,))
            for func_id, fmeta in meta["functions"].items():
                default_enabled = 1 if fmeta.get("locked") else 0
                conn.execute(
                    "INSERT OR IGNORE INTO module_functions(module_id,func_id,enabled) VALUES(?,?,?)",
                    (mod_id, func_id, default_enabled),
                )

        conn.execute(
            "UPDATE module_functions SET enabled=1 WHERE module_id='armas' AND func_id='tipo_arma'"
        )
        conn.commit()


# ── Settings ────────────────────────────────────────────────────────────────

def get_settings():
    with get_conn() as conn:
        rows = conn.execute("SELECT key, value FROM settings").fetchall()
    return {r["key"]: r["value"] for r in rows}


def set_setting(key, value):
    with get_conn() as conn:
        conn.execute("INSERT OR REPLACE INTO settings(key,value) VALUES(?,?)", (key, str(value)))
        conn.commit()


# ── Módulos ─────────────────────────────────────────────────────────────────

def get_modules_state():
    with get_conn() as conn:
        mod_rows  = conn.execute("SELECT module_id, enabled FROM modules").fetchall()
        func_rows = conn.execute("SELECT module_id, func_id, enabled FROM module_functions").fetchall()

    mod_enabled  = {r["module_id"]: bool(r["enabled"]) for r in mod_rows}
    func_enabled = {}
    for r in func_rows:
        func_enabled.setdefault(r["module_id"], {})[r["func_id"]] = bool(r["enabled"])

    result = {}
    for mod_id, meta in MODULES_META.items():
        functions = {}
        for func_id, fmeta in meta["functions"].items():
            functions[func_id] = {**fmeta, "enabled": func_enabled.get(mod_id, {}).get(func_id, False)}
        result[mod_id] = {"label": meta["label"], "enabled": mod_enabled.get(mod_id, False), "functions": functions}
    return result


def db_toggle_module(module_id):
    with get_conn() as conn:
        row = conn.execute("SELECT enabled FROM modules WHERE module_id=?", (module_id,)).fetchone()
        if row is None:
            return None
        new_val = 0 if row["enabled"] else 1
        conn.execute("UPDATE modules SET enabled=? WHERE module_id=?", (new_val, module_id))
        conn.commit()
    return bool(new_val)


def db_toggle_function(module_id, func_id):
    fmeta = MODULES_META.get(module_id, {}).get("functions", {}).get(func_id, {})
    if fmeta.get("locked"):
        return None
    with get_conn() as conn:
        row = conn.execute(
            "SELECT enabled FROM module_functions WHERE module_id=? AND func_id=?",
            (module_id, func_id),
        ).fetchone()
        if row is None:
            return None
        new_val = 0 if row["enabled"] else 1
        conn.execute(
            "UPDATE module_functions SET enabled=? WHERE module_id=? AND func_id=?",
            (new_val, module_id, func_id),
        )
        conn.commit()
    return bool(new_val)


# ── Fuentes ─────────────────────────────────────────────────────────────────

def get_sources(module_id=None):
    with get_conn() as conn:
        if module_id:
            rows = conn.execute(
                "SELECT * FROM sources WHERE module_id=? ORDER BY created_at DESC", (module_id,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM sources ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


def add_source(module_id, name, src_type, path):
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO sources(module_id, name, type, path) VALUES(?,?,?,?)",
            (module_id, name.strip(), src_type, path.strip()),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM sources WHERE id=?", (cur.lastrowid,)).fetchone()
    return dict(row)


def update_source(source_id, name=None, path=None):
    with get_conn() as conn:
        if name is not None:
            conn.execute("UPDATE sources SET name=? WHERE id=?", (name.strip(), source_id))
        if path is not None:
            conn.execute("UPDATE sources SET path=? WHERE id=?", (path.strip(), source_id))
        conn.commit()
        row = conn.execute("SELECT * FROM sources WHERE id=?", (source_id,)).fetchone()
    return dict(row) if row else None


def get_source(source_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM sources WHERE id=?", (source_id,)).fetchone()
    return dict(row) if row else None


def delete_source(source_id):
    with get_conn() as conn:
        row = conn.execute("SELECT path FROM sources WHERE id=?", (source_id,)).fetchone()
        affected = conn.execute("DELETE FROM sources WHERE id=?", (source_id,)).rowcount
        conn.commit()
    return {"deleted": affected > 0, "path": dict(row)["path"] if row else None}


# ── Reglamento detections ───────────────────────────────────────────────────

def insert_reglamento_detection(source_id, track_id, boot_status,
                                 time_compliance, seconds_in_area,
                                 capture_path=None):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO reglamento_detections
               (source_id, track_id, boot_status, time_compliance,
                seconds_in_area, capture_path)
               VALUES (?,?,?,?,?,?)""",
            (source_id, track_id, boot_status, time_compliance,
             seconds_in_area, capture_path),
        )
        conn.commit()


def get_reglamento_detections(source_id=None, days=7):
    with get_conn() as conn:
        if source_id:
            rows = conn.execute(
                """SELECT * FROM reglamento_detections
                   WHERE source_id=? AND created_at >= datetime('now', ?)
                   ORDER BY created_at DESC""",
                (source_id, f'-{days} days'),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM reglamento_detections
                   WHERE created_at >= datetime('now', ?)
                   ORDER BY created_at DESC""",
                (f'-{days} days',),
            ).fetchall()
    return [dict(r) for r in rows]


def get_reglamento_analytics(source_id, days=7):
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT boot_status, time_compliance,
                      COUNT(*) as count,
                      ROUND(AVG(seconds_in_area), 1) as avg_seconds
               FROM reglamento_detections
               WHERE source_id=? AND created_at >= datetime('now', ?)
               GROUP BY boot_status, time_compliance""",
            (source_id, f'-{days} days'),
        ).fetchall()
        daily = conn.execute(
            """SELECT DATE(created_at) as day,
                      SUM(CASE WHEN boot_status='con_botas' THEN 1 ELSE 0 END) as con_botas,
                      SUM(CASE WHEN boot_status='sin_botas' THEN 1 ELSE 0 END) as sin_botas,
                      SUM(CASE WHEN time_compliance='cumplio' THEN 1 ELSE 0 END) as cumplimientos,
                      SUM(CASE WHEN time_compliance='incumplio' THEN 1 ELSE 0 END) as incumplimientos
               FROM reglamento_detections
               WHERE source_id=? AND created_at >= datetime('now', ?)
               GROUP BY DATE(created_at)
               ORDER BY day ASC""",
            (source_id, f'-{days} days'),
        ).fetchall()
    return {"summary": [dict(r) for r in rows], "daily": [dict(r) for r in daily]}


# ── Carga / Descarga detections ─────────────────────────────────────────────

def insert_carga_descarga_detection(source_id, track_id, direction, model_id=None):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO carga_descarga_detections
               (source_id, track_id, direction, model_id)
               VALUES (?,?,?,?)""",
            (source_id, track_id, direction, model_id),
        )
        conn.commit()


def get_carga_descarga_analytics(source_id, days=7):
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT direction, COUNT(*) as count
               FROM carga_descarga_detections
               WHERE source_id=? AND created_at >= datetime('now', ?)
               GROUP BY direction""",
            (source_id, f'-{days} days'),
        ).fetchall()
        daily = conn.execute(
            """SELECT DATE(created_at) as day,
                      SUM(CASE WHEN direction='in' THEN 1 ELSE 0 END) as entradas,
                      SUM(CASE WHEN direction='out' THEN 1 ELSE 0 END) as salidas
               FROM carga_descarga_detections
               WHERE source_id=? AND created_at >= datetime('now', ?)
               GROUP BY DATE(created_at)
               ORDER BY day ASC""",
            (source_id, f'-{days} days'),
        ).fetchall()
    return {"summary": [dict(r) for r in rows], "daily": [dict(r) for r in daily]}


# ── Module Counters (Persistencia v3.0) ────────────────────────────────────

def save_module_counters(module_id: str, source_id: int, counters: dict) -> None:
    with get_conn() as conn:
        for key, value in counters.items():
            if isinstance(value, bool):
                iv = 1 if value else 0
                fv = None
                sv = None
            elif isinstance(value, int):
                iv = value
                fv = None
                sv = None
            elif isinstance(value, float):
                iv = 0
                fv = value
                sv = None
            elif isinstance(value, dict) or isinstance(value, list):
                iv = 0
                fv = None
                sv = json.dumps(value)
            else:
                iv = 0
                fv = None
                sv = str(value)
            conn.execute("""
                INSERT INTO module_counters(module_id, source_id, counter_key, int_value, float_value, str_value)
                VALUES (?,?,?,?,?,?)
                ON CONFLICT(module_id, source_id, counter_key)
                DO UPDATE SET int_value=excluded.int_value, float_value=excluded.float_value,
                              str_value=excluded.str_value, updated_at=datetime('now','localtime')
            """, (module_id, source_id, key, iv, fv, sv))
        conn.commit()


def load_module_counters(module_id: str, source_id: int) -> dict:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT counter_key, int_value, float_value, str_value FROM module_counters WHERE module_id=? AND source_id=?",
            (module_id, source_id),
        ).fetchall()
    result = {}
    for r in rows:
        k = r["counter_key"]
        if r["float_value"] is not None:
            result[k] = r["float_value"]
        elif r["str_value"] is not None:
            result[k] = r["str_value"]
        else:
            result[k] = r["int_value"]
    return result


def reset_module_counters(module_id: str, source_id: int) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM module_counters WHERE module_id=? AND source_id=?", (module_id, source_id))
        conn.commit()


# ── Module Events (Alertas históricas v3.0) ────────────────────────────────

def insert_module_event(module_id: str, source_id: int, event_type: str,
                        label: str = '', description: str = '',
                        event_data: dict = None, capture_path: str = None,
                        extra_paths: list = None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO module_events(module_id, source_id, event_type, label, description, event_data, capture_path, extra_paths)
               VALUES (?,?,?,?,?,?,?,?)""",
            (module_id, source_id, event_type, label, description,
             json.dumps(event_data or {}), capture_path,
             json.dumps(extra_paths) if extra_paths else None),
        )
        conn.commit()
    return cur.lastrowid


def get_module_events(module_id: str, source_id: int = None,
                      event_type: str = None, days: int = 7,
                      limit: int = 100) -> list:
    with get_conn() as conn:
        sql = "SELECT * FROM module_events WHERE module_id=?"
        params = [module_id]
        if source_id is not None:
            sql += " AND source_id=?"
            params.append(source_id)
        if event_type:
            sql += " AND event_type=?"
            params.append(event_type)
        sql += " AND created_at >= datetime('now', ?)"
        params.append(f'-{days} days')
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["event_data"] = json.loads(d.get("event_data", "{}"))
        d["extra_paths"] = json.loads(d["extra_paths"]) if d.get("extra_paths") else []
        result.append(d)
    return result


def get_module_analytics(module_id: str, source_id: int = None, days: int = 7) -> dict:
    with get_conn() as conn:
        params = [module_id, f'-{days} days']
        src_filter = " AND source_id=?" if source_id is not None else ""
        if source_id is not None:
            params.insert(1, source_id)

        event_counts = conn.execute(
            f"SELECT event_type, COUNT(*) as count FROM module_events WHERE module_id=?{src_filter} AND created_at >= datetime('now', ?) GROUP BY event_type",
            params,
        ).fetchall()

        daily = conn.execute(
            f"SELECT DATE(created_at) as day, event_type, COUNT(*) as count FROM module_events WHERE module_id=?{src_filter} AND created_at >= datetime('now', ?) GROUP BY DATE(created_at), event_type ORDER BY day ASC",
            params,
        ).fetchall()

        last_events = conn.execute(
            f"SELECT id, event_type, label, created_at, capture_path FROM module_events WHERE module_id=?{src_filter} AND created_at >= datetime('now', ?) ORDER BY created_at DESC LIMIT 50",
            params,
        ).fetchall()

    return {
        "event_counts": {r["event_type"]: r["count"] for r in event_counts},
        "daily": [dict(r) for r in daily],
        "last_events": [dict(r) for r in last_events],
    }


# ── Source Config (Configuración por fuente v3.0) ──────────────────────────

def save_source_config(source_id: int, key: str, value: str) -> None:
    with get_conn() as conn:
        row = conn.execute("SELECT config FROM sources WHERE id=?", (source_id,)).fetchone()
        if row is None:
            return
        config = json.loads(row["config"])
        config[key] = value
        conn.execute("UPDATE sources SET config=? WHERE id=?", (json.dumps(config), source_id))
        conn.commit()


def get_source_config(source_id: int) -> dict:
    with get_conn() as conn:
        row = conn.execute("SELECT config FROM sources WHERE id=?", (source_id,)).fetchone()
    if row is None:
        return {}
    return json.loads(row["config"])


def get_source_config_value(source_id: int, key: str, default=None):
    config = get_source_config(source_id)
    return config.get(key, default)


def delete_source_config(source_id: int) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE sources SET config='{}' WHERE id=?", (source_id,))
        conn.commit()
