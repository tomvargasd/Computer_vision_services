import sqlite3
import os
import json

from src.config import DB_PATH, APP_VERSION


MODULES_META = {
    "personas": {
        "label": "People Detection",
        "functions": {
            "conteo":      {"label": "People count",      "description": "Counts the total number of people in the scene"},
            "permanencia": {"label": "Dwell time",        "description": "Measures how long each person stays in the area"},
            "heatmap":     {"label": "Heatmap",           "description": "Generates a motion density map"},
        },
    },
    "armas": {
        "label": "Weapon Detection",
        "functions": {
            "deteccion_arma": {"label": "Weapon detection",            "description": "Detects the presence of weapons in the scene"},
            "captura_rostro": {"label": "Automatic face capture",      "description": "Captures and saves a photo of the weapon carrier"},
            "tipo_arma":      {"label": "Classify weapon type",        "description": "Classifies between a bladed weapon or a firearm", "locked": True},
        },
    },
    "acciones": {
        "label": "Action Detection",
        "functions": {
            "deteccion_acciones":  {"label": "Action detection",         "description": "Draws a COCO-17 pose skeleton over each person"},
            "deteccion_violencia": {"label": "Violence alerts",          "description": "Detects punches, kicks and physical aggression"},
            "deteccion_robo":      {"label": "Theft / threat alerts",   "description": "Detects pointing arms and threatening postures"},
            "deteccion_sospechosa":{"label": "Suspicious activity",     "description": "Detects crouching, stalking and furtive movements"},
            "deteccion_celular":   {"label": "Cell phone use",          "description": "Detects people using a phone (calling or texting)"},
            "deteccion_caida":     {"label": "Fall detection",          "description": "Detects partial (stumble/kneel) and complete falls"},
        },
    },
    "troncos": {
        "label": "Log Counting",
        "functions": {
            "conteo": {"label": "Log count", "description": "Counts logs crossing the vertical line"},
            "diametro": {"label": "Diameter classification", "description": "Estimates the real diameter of each counted log and classifies it (categories 0-5). Always active.", "locked": True},
        },
    },
    "pallets": {
        "label": "Pallet Counting",
        "functions": {
            "conteo": {"label": "Pallet count", "description": "Counts pallets within the detection area"},
        },
    },
    "cajas": {
        "label": "Box Counting",
        "functions": {
            "conteo": {"label": "Box count", "description": "Counts boxes crossing the horizontal line"},
        },
    },
    "reglamento": {
        "label": "Compliance Detection",
        "functions": {
            "deteccion_botas": {"label": "Boot detection",  "description": "Detects whether people are wearing boots within the area"},
            "conteo_tiempo":   {"label": "Time count",      "description": "Measures dwell time in the area"},
            "alertas":         {"label": "Alerts",          "description": "Generates visual alerts for non-compliance"},
            "analytics":       {"label": "Analytics",       "description": "Module statistics and charts"},
        },
    },
    "epp": {
        "label": "PPE Detection",
        "functions": {
            "deteccion_epp": {"label": "PPE detection",        "description": "Detects personal protective equipment on people"},
            "alertas":       {"label": "Missing PPE alerts",   "description": "Captures and alerts when a person lacks full PPE"},
            "analytics":     {"label": "Analytics",            "description": "Ranking of most used PPE and statistics"},
        },
    },
    "carga_descarga": {
        "label": "Loading/Unloading Detection",
        "functions": {
            "conteo":   {"label": "Load/unload count", "description": "Counts objects crossing the counting line"},
            "analytics":{"label": "Analytics",         "description": "Module statistics and charts"},
        },
    },
    "smoke": {
        "label": "Smoke/Fire Detection",
        "functions": {
            "deteccion_humo": {"label": "Smoke/fire detection", "description": "Detects the presence of smoke or fire in the scene"},
        },
    },
    "vehiculos": {
        "label": "Vehicle Recognition",
        "functions": {
            "conteo":           {"label": "Vehicle count",            "description": "Counts vehicles crossing the counting line"},
            "deteccion_placas": {"label": "License plate detection",  "description": "Recognizes plates on detected vehicles"},
        },
    },
    "tanques_gas": {
        "label": "Gas Tanks",
        "functions": {
            "conteo":              {"label": "Tank count",                "description": "Counts tanks crossing the line or entering the area"},
            "deteccion_acciones":  {"label": "Action detection",          "description": "Action detection with teaching system"},
            "deteccion_humo":      {"label": "Smoke/fire detection",      "description": "Detects the presence of smoke or fire in the scene"},
            "areas_restringidas":  {"label": "Restricted areas",          "description": "Defines prohibited areas for people or tanks"},
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
    ("troncos_pixels_per_unit", "1.0"),
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
    ("tanques_gas_smoke_conf", "0.35"),
    ("tanques_gas_line_mode", "horizontal"),
    ("tanques_gas_line_pos", "50"),
    ("gemini_api_key", ""),
    ("modules_order", ""),
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

            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT 'New session',
                created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
            );

            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user','model')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
            );

            -- Acceso: cuentas de usuario
            CREATE TABLE IF NOT EXISTS access_accounts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                email       TEXT    NOT NULL UNIQUE,
                password    TEXT    NOT NULL,
                access_type TEXT    NOT NULL CHECK(access_type IN ('full','limited')),
                created_at  TEXT    NOT NULL DEFAULT (datetime('now','localtime')),
                active      INTEGER NOT NULL DEFAULT 1
            );

            -- Acceso: logs de inicio/cierre de sesión
            CREATE TABLE IF NOT EXISTS access_logs (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                email        TEXT    NOT NULL,
                action       TEXT    NOT NULL CHECK(action IN ('login','logout','expired','failed')),
                ip_address   TEXT    NOT NULL DEFAULT '',
                location_data TEXT   NOT NULL DEFAULT '{}',
                user_agent   TEXT    NOT NULL DEFAULT '',
                reason       TEXT    DEFAULT '',
                created_at   TEXT    NOT NULL DEFAULT (datetime('now','localtime'))
            );

            -- Acceso: módulos permitidos para acceso limitado
            CREATE TABLE IF NOT EXISTS access_limited_modules (
                module_id TEXT PRIMARY KEY
            );

            -- Acceso: lista negra de IPs
            CREATE TABLE IF NOT EXISTS access_ip_blacklist (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ip_address  TEXT    NOT NULL UNIQUE,
                reason      TEXT    DEFAULT '',
                created_by  TEXT    DEFAULT '',
                created_at  TEXT    NOT NULL DEFAULT (datetime('now','localtime'))
            );

            CREATE INDEX IF NOT EXISTS idx_access_logs_email ON access_logs(email);
            CREATE INDEX IF NOT EXISTS idx_access_logs_created ON access_logs(created_at);
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

        # Migración: el ancho del bbox se toma como diámetro real (factor 1.0).
        conn.execute(
            "UPDATE settings SET value='1.0' "
            "WHERE key='troncos_pixels_per_unit' AND value='15.0'"
        )

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

        # Poblar módulos de detección en access_limited_modules por defecto
        all_mod_ids = list(MODULES_META.keys())
        for mid in all_mod_ids:
            conn.execute("INSERT OR IGNORE INTO access_limited_modules(module_id) VALUES(?)", (mid,))

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


def get_module_analytics_daily(module_id: str, source_id: int = None, days: int = 7) -> list:
    """Eventos diarios desglosados por fuente y tipo (para dashboards analíticos).

    Devuelve filas {day, source_id, event_type, count} para el módulo dado,
    opcionalmente filtrado por fuente y rango de días.
    """
    with get_conn() as conn:
        params = [module_id, f'-{days} days']
        src_filter = " AND source_id=?" if source_id is not None else ""
        if source_id is not None:
            params.insert(1, source_id)
        rows = conn.execute(
            f"""SELECT DATE(created_at) as day, source_id, event_type, COUNT(*) as count
                FROM module_events
                WHERE module_id=?{src_filter} AND created_at >= datetime('now', ?)
                GROUP BY DATE(created_at), source_id, event_type
                ORDER BY day ASC, source_id ASC""",
            params,
        ).fetchall()
    return [dict(r) for r in rows]


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


# ── Chat Sessions ────────────────────────────────────────────────────────────

import uuid

def create_chat_session(title: str = "New session") -> str:
    session_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO chat_sessions(id, title) VALUES (?, ?)",
            (session_id, title),
        )
        conn.commit()
    return session_id


def get_chat_sessions() -> list:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at FROM chat_sessions ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_chat_session(session_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, title, created_at, updated_at FROM chat_sessions WHERE id=?",
            (session_id,),
        ).fetchone()
    return dict(row) if row else None


def update_chat_session_title(session_id: str, title: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE chat_sessions SET title=?, updated_at=datetime('now','localtime') WHERE id=?",
            (title, session_id),
        )
        conn.commit()


def touch_chat_session(session_id: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE chat_sessions SET updated_at=datetime('now','localtime') WHERE id=?",
            (session_id,),
        )
        conn.commit()


def delete_chat_session(session_id: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM chat_messages WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM chat_sessions WHERE id=?", (session_id,))
        conn.commit()


def add_chat_message(session_id: str, role: str, content: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO chat_messages(session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        conn.commit()
    return cur.lastrowid


def get_chat_messages(session_id: str) -> list:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, role, content, created_at FROM chat_messages WHERE session_id=? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ── Acceso ──────────────────────────────────────────────────────────────────

SECRET_KEY_SETTING = "_access_secret_key"

def get_or_create_secret_key() -> str:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT value FROM settings WHERE key=?", (SECRET_KEY_SETTING,)
        ).fetchone()
        if row:
            return row["value"]
    import secrets
    key = secrets.token_hex(32)
    set_setting(SECRET_KEY_SETTING, key)
    return key


def add_account(email: str, password: str, access_type: str) -> dict:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO access_accounts(email, password, access_type) VALUES (?,?,?)",
            (email.strip().lower(), password, access_type),
        )
        conn.commit()
        row = conn.execute(
            "SELECT id, email, access_type, created_at, active FROM access_accounts WHERE id=?",
            (cur.lastrowid,),
        ).fetchone()
    return dict(row)


def get_accounts() -> list:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, email, access_type, created_at, active FROM access_accounts ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_account_by_email(email: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM access_accounts WHERE email=? AND active=1",
            (email.strip().lower(),),
        ).fetchone()
    return dict(row) if row else None


def get_account_by_id(account_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, email, access_type, created_at, active FROM access_accounts WHERE id=?",
            (account_id,),
        ).fetchone()
    return dict(row) if row else None


def delete_account(account_id: int) -> bool:
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM access_accounts WHERE id=?", (account_id,))
        conn.commit()
    return cur.rowcount > 0


def update_account_password(account_id: int, new_password: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE access_accounts SET password=? WHERE id=?",
            (new_password, account_id),
        )
        conn.commit()
    return cur.rowcount > 0


def add_access_log(email: str, action: str, ip_address: str = "",
                   location_data: dict = None, user_agent: str = "",
                   reason: str = "") -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO access_logs(email, action, ip_address, location_data, user_agent, reason) VALUES (?,?,?,?,?,?)",
            (email, action, ip_address, json.dumps(location_data or {}), user_agent, reason),
        )
        conn.commit()
    return cur.lastrowid


def get_access_logs(page: int = 1, per_page: int = 50, email_filter: str = None) -> dict:
    with get_conn() as conn:
        where = ""
        params = []
        if email_filter:
            where = " WHERE email=? "
            params.append(email_filter.strip().lower())
        total_row = conn.execute(
            f"SELECT COUNT(*) as cnt FROM access_logs{where}", params
        ).fetchone()
        total = total_row["cnt"] if total_row else 0
        offset = (page - 1) * per_page
        rows = conn.execute(
            f"SELECT * FROM access_logs{where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params + [per_page, offset],
        ).fetchall()
    return {
        "logs": [dict(r) for r in rows],
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": max(1, (total + per_page - 1) // per_page),
    }


def block_ip(ip_address: str, reason: str = "", created_by: str = "") -> dict | None:
    with get_conn() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO access_ip_blacklist(ip_address, reason, created_by) VALUES (?,?,?)",
                (ip_address, reason, created_by),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM access_ip_blacklist WHERE id=?", (cur.lastrowid,)
            ).fetchone()
            return dict(row) if row else None
        except Exception:
            return None


def unblock_ip(entry_id: int) -> bool:
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM access_ip_blacklist WHERE id=?", (entry_id,))
        conn.commit()
    return cur.rowcount > 0


def get_blacklist() -> list:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM access_ip_blacklist ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def is_ip_blocked(ip_address: str) -> bool:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM access_ip_blacklist WHERE ip_address=?", (ip_address,)
        ).fetchone()
    return row is not None


def get_limited_modules() -> list:
    with get_conn() as conn:
        rows = conn.execute("SELECT module_id FROM access_limited_modules").fetchall()
    return [r["module_id"] for r in rows]


def set_limited_modules(module_ids: list) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM access_limited_modules")
        for mid in module_ids:
            conn.execute("INSERT INTO access_limited_modules(module_id) VALUES(?)", (mid,))
        conn.commit()
