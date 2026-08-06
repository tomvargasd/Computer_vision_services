from __future__ import annotations

from flask import Flask, render_template, jsonify, request, Response, stream_with_context, session, g
from flask_cors import CORS
from flask_compress import Compress
import os
import uuid
import time
import json
import random
import secrets
import requests
import re
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename

from src.config import (
    BASE_DIR, UPLOAD_FOLDER, VIDEOS_FOLDER, MODELS_FOLDER, CAPTURES_FOLDER,
    ALLOWED_IMG, ALLOWED_VIDEO, ALLOWED_MODEL, APP_VERSION,
)
from src.database import (
    init_db, get_settings, set_setting,
    get_modules_state, db_toggle_module, db_toggle_function,
    get_sources, get_source, add_source, update_source, update_source_fps, delete_source,
    MODULES_META,
    save_module_counters, load_module_counters, reset_module_counters,
    insert_module_event, get_module_events, get_module_analytics,
    get_module_analytics_daily,
    save_source_config, get_source_config, get_source_config_value,
    delete_source_config,
    create_chat_session, get_chat_sessions, get_chat_session,
    update_chat_session_title, touch_chat_session,
    delete_chat_session, add_chat_message, get_chat_messages,
    # funciones de acceso
    get_or_create_secret_key, add_account, get_accounts,
    get_account_by_email, get_account_by_id, delete_account,
    update_account_password, add_access_log, get_access_logs,
    block_ip, unblock_ip, get_blacklist, is_ip_blocked,
    get_limited_modules, set_limited_modules,
    create_semantycs_session, get_semantycs_sessions, get_semantycs_session,
    update_semantycs_session, delete_semantycs_session,
    add_semantycs_message, get_semantycs_messages,
    upsert_semantycs_counter, list_semantycs_counters, clear_semantycs_counters,
    insert_semantycs_log, list_semantycs_logs, clear_semantycs_logs,
)

from src.modules.smart_semantycs import SmartSemntycsManager
from src.modules.smart_semantycs_vocab import (
    LVIS_NAMES, to_lvis, lvis_exists, find_lvis_candidates, best_lvis_match,
)
from src.modules.base import get_device

from src.modules.personas import PersonasManager
from src.modules.armas import ArmasManager
from src.modules.acciones import AccionesManager
from src.modules.troncos import TroncosManager
from src.modules.pallets import PalletsManager
from src.modules.cajas import CajasManager
from src.modules.reglamento import ReglamentoManager
from src.modules.carga_descarga import CargaDescargaManager
from src.modules.epp import EppManager
from src.modules.smoke import SmokeManager
from src.modules.vehiculos import VehiculosManager
from src.modules.tanques_gas import TanquesGasManager


# ─────────────────────────────────────────────
# Vistas de Analytics (v3.0)
# ─────────────────────────────────────────────

ANALYTICS_TEMPLATES = {
    "personas": "analytics_personas.html",
    "armas": "analytics_armas.html",
    "acciones": "analytics_acciones.html",
    "troncos": "analytics_troncos.html",
    "pallets": "analytics_pallets.html",
    "cajas": "analytics_cajas.html",
    "reglamento": "analytics_reglamento.html",
    "carga_descarga": "analytics_carga_descarga.html",
    "epp": "analytics_epp.html",
    "smoke": "analytics_smoke.html",
    "vehiculos": "analytics_vehiculos.html",
    "tanques_gas": "analytics_tanques_gas.html",
}


def _normalize_path(path, src_type):
    if src_type != "video":
        return path
    if path.startswith(("http://", "https://", "rtsp://", "rtmp://")):
        return path
    abspath = os.path.abspath(path)
    if os.path.exists(abspath):
        rel = os.path.relpath(abspath, BASE_DIR)
        if not rel.startswith(".."):
            return rel
    return path


def _get_fps_limit(source, src_type):
    fps_str = source.get("fps_limit", "")
    if fps_str:
        try:
            return float(fps_str)
        except (ValueError, TypeError):
            pass
    return 0.2 if src_type == "video" else 0.0


app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"),
            static_folder=os.path.join(BASE_DIR, "static"))
CORS(app)
Compress(app)

app.secret_key = get_or_create_secret_key()
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIDEOS_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(CAPTURES_FOLDER, exist_ok=True)

init_db()
set_setting("version", APP_VERSION)


def allowed_img(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMG


def allowed_video(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO


def allowed_model(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_MODEL


@app.context_processor
def inject_globals():
    access_info = {
        "authenticated": "email" in session,
        "email": session.get("email", ""),
        "access_type": session.get("access_type", ""),
    }
    return {
        "settings": get_settings(),
        "modules": get_modules_state(),
        "access": access_info,
        "limited_modules": get_limited_modules() if access_info.get("access_type") == "limited" else [],
    }


PUBLIC_PREFIXES = ("/static/", "/api/access/")

@app.before_request
def check_access():
    path = request.path

    for prefix in PUBLIC_PREFIXES:
        if path.startswith(prefix):
            return None

    ip = request.remote_addr or ""
    if ip and is_ip_blocked(ip):
        return jsonify({"error": "Access denied: blocked IP"}), 403

    email = session.get("email")
    expires_at = session.get("expires_at")
    access_type = session.get("access_type")

    g.access_authenticated = False
    g.access_email = ""
    g.access_type = ""

    if email and expires_at:
        try:
            exp = datetime.fromisoformat(expires_at)
            if datetime.now() < exp:
                g.access_authenticated = True
                g.access_email = email
                g.access_type = access_type
                return None
            else:
                add_access_log(email, "expired", ip,
                               user_agent=request.headers.get("User-Agent", ""))
                session.clear()
        except Exception:
            session.clear()

    if path.startswith("/api/"):
        return jsonify({"error": "Not authenticated", "needs_login": True}), 401

    return None


# ─────────────────────────────────────────────
# Vistas de Analytics (v3.0)
# ─────────────────────────────────────────────

@app.route("/analytics")
def analytics_view():
    modules = get_modules_state()
    return render_template("analytics.html", modules=modules)


@app.route("/analytics/<module_id>")
def analytics_module_view(module_id):
    if module_id not in MODULES_META:
        return render_template("404.html"), 404
    if g.get("access_type") == "limited" and module_id not in get_limited_modules():
        return render_template("404.html"), 404
    tmpl = ANALYTICS_TEMPLATES.get(module_id)
    if not tmpl:
        return render_template("404.html"), 404
    modules = get_modules_state()
    settings = get_settings()
    return render_template(tmpl, module_id=module_id, module=modules[module_id], settings=settings)


# ─────────────────────────────────────────────
# Vistas
# ─────────────────────────────────────────────

@app.route("/")
def index():
    modules       = get_modules_state()
    sources_count = {mid: len(get_sources(mid)) for mid in MODULES_META}
    return render_template("index.html", modules=modules, sources_count=sources_count)


@app.route("/module/<module_id>")
def module_view(module_id):
    if module_id not in MODULES_META:
        return render_template("404.html"), 404
    if g.get("access_type") == "limited" and module_id not in get_limited_modules():
        return render_template("404.html"), 404
    modules    = get_modules_state()
    sources    = get_sources(module_id)
    func_state = {
        fid: fdata["enabled"]
        for fid, fdata in modules[module_id]["functions"].items()
    }
    settings   = get_settings()
    return render_template(
        "module.html",
        module_id=module_id,
        sources=sources,
        func_state=func_state,
        module=modules[module_id],
        settings=settings,
    )


@app.route("/module/<module_id>/live/<int:source_id>")
def live_view(module_id, source_id):
    if module_id not in MODULES_META:
        return render_template("404.html"), 404
    sources = get_sources(module_id)
    src = next((s for s in sources if s["id"] == source_id), None)
    if not src:
        return render_template("404.html"), 404
    modules    = get_modules_state()
    func_state = {
        fid: fdata["enabled"]
        for fid, fdata in modules[module_id]["functions"].items()
    }
    settings = get_settings()

    tmpl_map = {
        "personas": "personas_live.html",
        "armas": "armas_live.html",
        "acciones": "acciones_live.html",
        "troncos": "troncos_live.html",
        "pallets": "pallets_live.html",
        "cajas": "cajas_live.html",
        "reglamento": "reglamento_live.html",
        "carga_descarga": "carga_descarga_live.html",
        "epp": "epp_live.html",
        "smoke": "smoke_live.html",
        "vehiculos": "vehiculos_live.html",
        "tanques_gas": "tanques_gas_live.html",
    }
    tmpl = tmpl_map.get(module_id)
    if not tmpl:
        return render_template("404.html"), 404
    return render_template(tmpl, source=src, func_state=func_state,
                           module=modules[module_id], settings=settings)


# ─────────────────────────────────────────────
# Módulos — toggle
# ─────────────────────────────────────────────

@app.route("/api/modules/<module_id>/toggle", methods=["POST"])
def api_toggle_module(module_id):
    if g.get("access_type") == "limited":
        return jsonify({"error": "You don't have permission"}), 403
    if module_id not in MODULES_META:
        return jsonify({"error": "Module not found"}), 404
    enabled = db_toggle_module(module_id)
    if not enabled:
        stop_map = {
            "personas": PersonasManager,
            "armas": ArmasManager,
            "acciones": AccionesManager,
            "troncos": TroncosManager,
            "pallets": PalletsManager,
            "cajas": CajasManager,
            "reglamento": ReglamentoManager,
            "carga_descarga": CargaDescargaManager,
            "epp": EppManager,
            "smoke": SmokeManager,
            "vehiculos": VehiculosManager,
            "tanques_gas": TanquesGasManager,
        }
        mgr = stop_map.get(module_id)
        if mgr:
            mgr.get().stop_all()
    return jsonify({"module": module_id, "enabled": enabled})


@app.route("/api/modules/<module_id>/functions/<func_id>/toggle", methods=["POST"])
def api_toggle_function(module_id, func_id):
    if g.get("access_type") == "limited":
        return jsonify({"error": "You don't have permission"}), 403
    if module_id not in MODULES_META:
        return jsonify({"error": "Module not found"}), 404
    if func_id not in MODULES_META[module_id]["functions"]:
        return jsonify({"error": "Function not found"}), 404
    fmeta = MODULES_META[module_id]["functions"][func_id]
    if fmeta.get("locked"):
        return jsonify({"error": "This function cannot be disabled", "locked": True}), 403
    enabled = db_toggle_function(module_id, func_id)

    func_state = {
        fid: fdata["enabled"]
        for fid, fdata in get_modules_state()[module_id]["functions"].items()
    }
    mgr_map = {
        "personas": PersonasManager,
        "armas": ArmasManager,
        "acciones": AccionesManager,
        "troncos": TroncosManager,
        "pallets": PalletsManager,
        "cajas": CajasManager,
        "reglamento": ReglamentoManager,
        "carga_descarga": CargaDescargaManager,
        "epp": EppManager,
        "smoke": SmokeManager,
        "vehiculos": VehiculosManager,
        "tanques_gas": TanquesGasManager,
    }
    mgr = mgr_map.get(module_id)
    if mgr:
        mgr.get().update_func_state(func_state)
    return jsonify({"module": module_id, "function": func_id, "enabled": enabled})


@app.route("/api/status", methods=["GET"])
def api_status():
    modules = get_modules_state()
    status  = {}
    for mod_id, mod in modules.items():
        active_funcs = sum(1 for f in mod["functions"].values() if f["enabled"])
        status[mod_id] = {
            "label": mod["label"],
            "enabled": mod["enabled"],
            "active_functions": active_funcs,
            "total_functions": len(mod["functions"]),
            "sources": len(get_sources(mod_id)),
        }
    return jsonify(status)


# ─────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────

@app.route("/settings")
def settings_view():
    if g.get("access_type") == "limited":
        return render_template("404.html"), 404
    modules = get_modules_state()
    limited_modules = get_limited_modules()
    return render_template("settings.html", modules=modules, settings=get_settings(),
                           limited_modules=limited_modules)


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        for k, v in data.items():
            set_setting(k, v)
        return jsonify({"saved": True})
    return jsonify(get_settings())


@app.route("/api/settings/upload/<field>", methods=["POST"])
def api_upload(field):
    if "file" not in request.files:
        return jsonify({"error": "No file was sent"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    target_map = {
        "logo": (UPLOAD_FOLDER, ALLOWED_IMG),
        "video": (VIDEOS_FOLDER, ALLOWED_VIDEO),
        "model": (MODELS_FOLDER, ALLOWED_MODEL),
    }
    if field not in target_map:
        return jsonify({"error": "Invalid field"}), 400
    dest_dir, allowed_set = target_map[field]
    ext = f.filename.rsplit(".", 1)[1].lower() if "." in f.filename else ""
    if ext not in allowed_set:
        return jsonify({"error": f"File type not allowed ({ext})"}), 400
    fname = secure_filename(f.filename)
    dest = os.path.join(dest_dir, fname)
    f.save(dest)
    if field == "logo":
        logos = _logos_list()
        if fname not in logos:
            logos.append(fname)
        _save_logos(logos)
        set_setting("logo", fname)
        return jsonify({"path": fname})
    rel = os.path.join("static", "uploads", field + "s" if field != "logo" else "uploads", fname)
    return jsonify({"path": rel})


def _logos_list():
    raw = get_settings().get("logos", "[]")
    try:
        return json.loads(raw)
    except Exception:
        return []


def _save_logos(logos):
    set_setting("logos", json.dumps(list(dict.fromkeys(logos))))


def _logos_payload():
    active = get_settings().get("logo", "")
    logos = _logos_list()
    if active and active not in logos:
        logos.insert(0, active)
        _save_logos(logos)
    return {"logos": logos, "active": active}


@app.route("/api/settings/logos", methods=["GET"])
def api_logos():
    return jsonify(_logos_payload())


@app.route("/api/settings/logo", methods=["POST"])
def api_upload_logo():
    if "logo" not in request.files:
        return jsonify({"error": "No file was sent"}), 400
    f = request.files["logo"]
    if not f.filename:
        return jsonify({"error": "Empty name"}), 400
    ext = f.filename.rsplit(".", 1)[1].lower() if "." in f.filename else ""
    if ext not in ALLOWED_IMG:
        return jsonify({"error": "Format not allowed"}), 400
    fname = secure_filename(f.filename)
    f.save(os.path.join(UPLOAD_FOLDER, fname))

    logos = _logos_list()
    if fname not in logos:
        logos.append(fname)
    _save_logos(logos)

    set_setting("logo", fname)
    payload = {"logo": fname, **_logos_payload()}
    return jsonify(payload)


@app.route("/api/settings/logo/active", methods=["POST"])
def api_set_active_logo():
    data = request.get_json(silent=True) or {}
    fname = (data.get("filename") or "").strip()
    if not fname:
        return jsonify({"error": "Filename required"}), 400
    logos = _logos_list()
    if fname not in logos:
        return jsonify({"error": "Logo not found"}), 404
    set_setting("logo", fname)
    return jsonify(_logos_payload())


@app.route("/api/settings/logo/delete", methods=["POST"])
def api_delete_logo():
    data = request.get_json(silent=True) or {}
    fname = (data.get("filename") or "").strip()
    if not fname:
        return jsonify({"error": "Filename required"}), 400
    logos = [x for x in _logos_list() if x != fname]
    _save_logos(logos)
    try:
        os.remove(os.path.join(UPLOAD_FOLDER, fname))
    except OSError:
        pass
    if get_settings().get("logo") == fname:
        set_setting("logo", logos[0] if logos else "")
    return jsonify(_logos_payload())


@app.route("/api/<module_id>/upload-model", methods=["POST"])
def api_upload_module_model(module_id):
    if module_id not in MODULES_META:
        return jsonify({"error": "Module not found"}), 404
    if "model" not in request.files:
        return jsonify({"error": "No file was sent"}), 400
    f = request.files["model"]
    if not f.filename.endswith(".pt"):
        return jsonify({"error": "Only .pt files"}), 400
    fname = secure_filename(f.filename)
    f.save(os.path.join(MODELS_FOLDER, fname))
    rel = os.path.join("static", "uploads", "models", fname)
    set_setting(f"{module_id}_model", rel)
    return jsonify({"model": fname})


@app.route("/api/<module_id>/model", methods=["DELETE"])
def api_remove_module_model(module_id):
    if module_id not in MODULES_META:
        return jsonify({"error": "Module not found"}), 404
    set_setting(f"{module_id}_model", "")
    return jsonify({"removed": True})


@app.route("/api/<module_id>/settings/conf", methods=["POST"])
def api_set_conf(module_id):
    if module_id not in MODULES_META:
        return jsonify({"error": "Module not found"}), 404
    data = request.get_json(silent=True) or {}
    val = str(float(data.get("conf", 0.35)))
    set_setting(f"{module_id}_conf", val)
    return jsonify({"conf": val})


@app.route("/api/<module_id>/settings/half", methods=["POST"])
def api_set_half(module_id):
    if module_id not in MODULES_META:
        return jsonify({"error": "Module not found"}), 404
    data = request.get_json(silent=True) or {}
    val = "1" if data.get("half") else "0"
    set_setting(f"{module_id}_half", val)
    return jsonify({"half": val})


@app.route("/api/sources/<int:source_id>/fps", methods=["POST"])
def api_set_source_fps(source_id):
    src = get_source(source_id)
    if not src:
        return jsonify({"error": "Source not found"}), 404
    data = request.get_json(silent=True) or {}
    fps_limit = str(float(data.get("fps_limit", 0.0)))
    update_source_fps(source_id, fps_limit)
    return jsonify({"fps_limit": fps_limit})


@app.route("/api/pallets/settings/classes", methods=["POST"])
def pallets_set_classes():
    data = request.get_json(silent=True) or {}
    raw = data.get("classes", "0,1,2,3")
    ids = [int(c.strip()) for c in raw.split(",") if c.strip()]
    if not ids or any(i < 0 or i > 255 for i in ids):
        return jsonify({"error": "Invalid class list"}), 400
    val = ",".join(str(i) for i in ids)
    set_setting("pallets_classes", val)
    PalletsManager.get().set_classes(ids)
    return jsonify({"classes": ids})


@app.route("/api/carga_descarga/settings/models", methods=["GET", "POST"])
def carga_descarga_settings_models():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        models = data.get("models", [])
        set_setting("carga_descarga_models", json.dumps(models))
        return jsonify({"saved": True})
    raw = get_settings().get("carga_descarga_models", "[]")
    try:
        models = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        models = []
    return jsonify(models)


@app.route("/api/carga_descarga/settings/upload-model", methods=["POST"])
def carga_descarga_upload_model():
    if "file" not in request.files:
        return jsonify({"error": "No file was sent"}), 400
    f = request.files["file"]
    if not f.filename.endswith(".pt"):
        return jsonify({"error": "Only .pt files"}), 400
    fname = secure_filename(f.filename)
    dest = os.path.join(MODELS_FOLDER, fname)
    f.save(dest)
    rel = os.path.join("static", "uploads", "models", fname)
    return jsonify({"path": rel})


# ─────────────────────────────────────────────
# Helpers para func_state por módulo
# ─────────────────────────────────────────────

def _func_state_for(module_id: str) -> dict:
    return {
        fid: fdata["enabled"]
        for fid, fdata in get_modules_state()[module_id]["functions"].items()
    }


# ─────────────────────────────────────────────
# API de módulos — helpers genéricos
# ─────────────────────────────────────────────

_MANAGERS = {
    "personas": PersonasManager,
    "armas": ArmasManager,
    "acciones": AccionesManager,
    "troncos": TroncosManager,
    "pallets": PalletsManager,
    "cajas": CajasManager,
    "reglamento": ReglamentoManager,
    "carga_descarga": CargaDescargaManager,
    "epp": EppManager,
    "smoke": SmokeManager,
    "vehiculos": VehiculosManager,
    "tanques_gas": TanquesGasManager,
}


def _get_manager(module_id):
    cls = _MANAGERS.get(module_id)
    return cls.get() if cls else None


def _stream_response(module_id, source_id):
    manager = _get_manager(module_id)
    def generate():
        while manager.is_running(source_id):
            jpeg = manager.get_frame_jpeg(source_id)
            if jpeg is None:
                time.sleep(0.033)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )
            time.sleep(1 / 30)
    return Response(
        stream_with_context(generate()),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─────────────────────────────────────────────
# Fábrica de rutas estándar para módulos
# ─────────────────────────────────────────────

def _register_standard_routes(app, module_id, manager_class, extra_routes=None):
    _MANAGERS[module_id] = manager_class

    @app.route(f"/api/{module_id}/sources/<int:source_id>/stop",
               endpoint=f"{module_id}_stop", methods=["POST"])
    def _stop(source_id):
        manager_class.get().stop(source_id)
        return jsonify({"stopped": source_id})

    @app.route(f"/api/{module_id}/sources/<int:source_id>/stats",
               endpoint=f"{module_id}_stats")
    def _stats(source_id):
        stats = manager_class.get().get_stats(source_id)
        if stats is None:
            return jsonify({"error": "Pipeline not active"}), 404
        return jsonify(stats)

    @app.route(f"/api/{module_id}/sources/<int:source_id>/reset",
               endpoint=f"{module_id}_reset", methods=["POST"])
    def _reset(source_id):
        manager_class.get().reset(source_id)
        return jsonify({"reset": source_id})

    @app.route(f"/api/{module_id}/stream/<int:source_id>",
               endpoint=f"{module_id}_stream")
    def _stream(source_id):
        return _stream_response(module_id, source_id)

    if extra_routes:
        extra_routes(app)


# ─────────────────────────────────────────────
# API Personas
# ─────────────────────────────────────────────

_register_standard_routes(app, "personas", PersonasManager)

@app.route("/api/personas/sources/<int:source_id>/start", methods=["POST"])
def personas_start(source_id):
    sources = get_sources("personas")
    src = next((s for s in sources if s["id"] == source_id), None)
    if not src:
        return jsonify({"error": "Source not found"}), 404
    s = get_settings()
    fps_limit = _get_fps_limit(src, src["type"])
    try:
        PersonasManager.get().start(source_id, src["path"],
            _func_state_for("personas"),
            float(s.get("personas_conf", "0.35")),
            s.get("personas_half", "0") == "1",
            s.get("personas_model") or None,
            int(s.get("personas_line_y", "85")),
            fps_limit=fps_limit)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 429
    return jsonify({"started": source_id})


@app.route("/api/personas/sources/<int:source_id>/line-y", methods=["POST"])
def personas_line_y(source_id):
    data = request.get_json(silent=True) or {}
    pct = max(0, min(100, int(data.get("pct", 85))))
    set_setting("personas_line_y", str(pct))
    PersonasManager.get().set_line_y(source_id, pct)
    return jsonify({"line_y_pct": pct})


# ─────────────────────────────────────────────
# API Armas
# ─────────────────────────────────────────────

_register_standard_routes(app, "armas", ArmasManager)

@app.route("/api/armas/sources/<int:source_id>/start", methods=["POST"])
def armas_start(source_id):
    sources = get_sources("armas")
    src = next((s for s in sources if s["id"] == source_id), None)
    if not src:
        return jsonify({"error": "Source not found"}), 404
    s = get_settings()
    fps_limit = _get_fps_limit(src, src["type"])
    try:
        ArmasManager.get().start(source_id, src["path"],
            _func_state_for("armas"),
            float(s.get("armas_conf", "0.20")),
            s.get("armas_half", "0") == "1",
            s.get("armas_model") or None,
            fps_limit=fps_limit)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 429
    return jsonify({"started": source_id})


# ─────────────────────────────────────────────
# API Acciones
# ─────────────────────────────────────────────

_register_standard_routes(app, "acciones", AccionesManager)

@app.route("/api/acciones/sources/<int:source_id>/start", methods=["POST"])
def acciones_start(source_id):
    sources = get_sources("acciones")
    src = next((s for s in sources if s["id"] == source_id), None)
    if not src:
        return jsonify({"error": "Source not found"}), 404
    s = get_settings()
    fps_limit = _get_fps_limit(src, src["type"])
    try:
        AccionesManager.get().start(source_id, src["path"],
            _func_state_for("acciones"),
            float(s.get("acciones_conf", "0.35")),
            s.get("acciones_half", "0") == "1",
            s.get("acciones_model") or None,
            fps_limit=fps_limit)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 429
    return jsonify({"started": source_id})


@app.route("/api/acciones/sources/<int:source_id>/teach/data")
def acciones_teach_data(source_id):
    mgr = AccionesManager.get()
    data = mgr.get_teach_data(source_id)
    if data is None:
        return jsonify({"error": "Pipeline not active"}), 404
    return jsonify(data)


@app.route("/api/acciones/sources/<int:source_id>/teach/save", methods=["POST"])
def acciones_teach_save(source_id):
    body = request.get_json()
    if not body:
        return jsonify({"error": "JSON required"}), 400
    tid = body.get("person_id")
    action = body.get("action")
    if tid is None or action not in ("violencia", "robo", "sospechoso", "celular", "caida"):
        return jsonify({"error": "person_id and action required (violencia|robo|sospechoso|celular|caida)"}), 400
    mgr = AccionesManager.get()
    pipeline = mgr.pipelines.get(source_id) if hasattr(mgr, "pipelines") else None
    if pipeline is None:
        return jsonify({"error": "Pipeline not active"}), 404
    log = list(pipeline._person_log.get(tid, []))
    sample = {
        "id": str(uuid.uuid4()),
        "action": action,
        "source_id": source_id,
        "person_id": tid,
        "ts": time.time(),
        "log": [dict(e) for e in log],
        "captures": {
            "face": list(pipeline._cap_face.get(tid, [])),
            "body": list(pipeline._cap_body.get(tid, [])),
        },
    }
    from src.modules.acciones import _save_teach_sample
    _save_teach_sample(sample)
    return jsonify({"ok": True, "sample_id": sample["id"]})


# ─────────────────────────────────────────────
# API Troncos
# ─────────────────────────────────────────────

_register_standard_routes(app, "troncos", TroncosManager)

@app.route("/api/troncos/sources/<int:source_id>/start", methods=["POST"])
def troncos_start(source_id):
    sources = get_sources("troncos")
    src = next((s for s in sources if s["id"] == source_id), None)
    if not src:
        return jsonify({"error": "Source not found"}), 404
    s = get_settings()
    fps_limit = _get_fps_limit(src, src["type"])
    try:
        TroncosManager.get().start(source_id, src["path"],
            _func_state_for("troncos"),
            float(s.get("troncos_conf", "0.35")),
            s.get("troncos_half", "0") == "1",
            s.get("troncos_model") or None,
            int(s.get("troncos_line_x", "50")),
            fps_limit=fps_limit)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 429
    return jsonify({"started": source_id})


@app.route("/api/troncos/sources/<int:source_id>/line-x", methods=["POST"])
def troncos_line_x(source_id):
    data = request.get_json(silent=True) or {}
    pct = max(0, min(100, int(data.get("pct", 50))))
    set_setting("troncos_line_x", str(pct))
    TroncosManager.get().set_line_x(source_id, pct)
    return jsonify({"line_x_pct": pct})


@app.route("/api/troncos/settings/pixels-per-unit", methods=["POST"])
def troncos_pixels_per_unit():
    data = request.get_json(silent=True) or {}
    try:
        value = float(data.get("value"))
    except (TypeError, ValueError):
        return jsonify({"error": "Valor inválido"}), 400
    if not (0 < value <= 1000):
        return jsonify({"error": "El factor debe estar entre 0 y 1000"}), 400
    set_setting("troncos_pixels_per_unit", f"{value:.6f}".rstrip("0").rstrip("."))
    TroncosManager.get().set_pixels_per_unit(value)
    return jsonify({"pixels_per_unit": value})


# ─────────────────────────────────────────────
# API Pallets
# ─────────────────────────────────────────────

_register_standard_routes(app, "pallets", PalletsManager)

@app.route("/api/pallets/sources/<int:source_id>/start", methods=["POST"])
def pallets_start(source_id):
    sources = get_sources("pallets")
    src = next((s for s in sources if s["id"] == source_id), None)
    if not src:
        return jsonify({"error": "Source not found"}), 404
    s = get_settings()
    classes_str = s.get("pallets_classes", "0,1,2,3")
    classes = [int(c.strip()) for c in classes_str.split(",") if c.strip()]
    fps_limit = _get_fps_limit(src, src["type"])
    try:
        PalletsManager.get().start(source_id, src["path"],
            _func_state_for("pallets"),
            float(s.get("pallets_conf", "0.35")),
            s.get("pallets_half", "0") == "1",
            s.get("pallets_model") or None,
            int(s.get("pallets_area_x1", "25")),
            int(s.get("pallets_area_y1", "25")),
            int(s.get("pallets_area_x2", "75")),
            int(s.get("pallets_area_y2", "75")),
            classes=classes,
            fps_limit=fps_limit)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 429
    return jsonify({"started": source_id})


@app.route("/api/pallets/sources/<int:source_id>/area", methods=["POST"])
def pallets_area(source_id):
    data = request.get_json(silent=True) or {}
    x1 = int(data.get("x1", 25)); y1 = int(data.get("y1", 25))
    x2 = int(data.get("x2", 75)); y2 = int(data.get("y2", 75))
    set_setting("pallets_area_x1", str(x1)); set_setting("pallets_area_y1", str(y1))
    set_setting("pallets_area_x2", str(x2)); set_setting("pallets_area_y2", str(y2))
    PalletsManager.get().set_area(source_id, x1, y1, x2, y2)
    return jsonify({"x1": x1, "y1": y1, "x2": x2, "y2": y2})


# ─────────────────────────────────────────────
# API Cajas
# ─────────────────────────────────────────────

_register_standard_routes(app, "cajas", CajasManager)

@app.route("/api/cajas/sources/<int:source_id>/start", methods=["POST"])
def cajas_start(source_id):
    sources = get_sources("cajas")
    src = next((s for s in sources if s["id"] == source_id), None)
    if not src:
        return jsonify({"error": "Source not found"}), 404
    s = get_settings()
    fps_limit = _get_fps_limit(src, src["type"])
    try:
        CajasManager.get().start(source_id, src["path"],
            _func_state_for("cajas"),
            float(s.get("cajas_conf", "0.35")),
            s.get("cajas_half", "0") == "1",
            s.get("cajas_model") or None,
            int(s.get("cajas_line_y", "85")),
            fps_limit=fps_limit)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 429
    return jsonify({"started": source_id})


@app.route("/api/cajas/sources/<int:source_id>/line-y", methods=["POST"])
def cajas_line_y(source_id):
    data = request.get_json(silent=True) or {}
    pct = max(0, min(100, int(data.get("pct", 85))))
    set_setting("cajas_line_y", str(pct))
    CajasManager.get().set_line_y(source_id, pct)
    return jsonify({"line_y_pct": pct})


# ─────────────────────────────────────────────
# API Reglamento
# ─────────────────────────────────────────────

_register_standard_routes(app, "reglamento", ReglamentoManager)

@app.route("/api/reglamento/sources/<int:source_id>/start", methods=["POST"])
def reglamento_start(source_id):
    sources = get_sources("reglamento")
    src = next((s for s in sources if s["id"] == source_id), None)
    if not src:
        return jsonify({"error": "Source not found"}), 404
    s = get_settings()
    fps_limit = _get_fps_limit(src, src["type"])
    try:
        ReglamentoManager.get().start(source_id, src["path"],
            _func_state_for("reglamento"),
            float(s.get("reglamento_conf", "0.45")),
            s.get("reglamento_half", "0") == "1",
            s.get("reglamento_model") or None,
            int(s.get("reglamento_min_time", "10")),
            int(s.get("reglamento_area_x1", "30")),
            int(s.get("reglamento_area_y1", "30")),
            int(s.get("reglamento_area_x2", "70")),
            int(s.get("reglamento_area_y2", "70")),
            jpeg_q=int(s.get("reglamento_jpeg_q", "72")),
            max_dim=int(s.get("reglamento_max_dim", "0")),
            frame_step=int(s.get("reglamento_frame_step", "1")),
            fps_limit=fps_limit)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 429
    return jsonify({"started": source_id})


@app.route("/api/reglamento/sources/<int:source_id>/area", methods=["POST"])
def reglamento_area(source_id):
    data = request.get_json(silent=True) or {}
    x1 = int(data.get("x1", 30)); y1 = int(data.get("y1", 30))
    x2 = int(data.get("x2", 70)); y2 = int(data.get("y2", 70))
    set_setting("reglamento_area_x1", str(x1)); set_setting("reglamento_area_y1", str(y1))
    set_setting("reglamento_area_x2", str(x2)); set_setting("reglamento_area_y2", str(y2))
    ReglamentoManager.get().set_area(source_id, x1, y1, x2, y2)
    return jsonify({"x1": x1, "y1": y1, "x2": x2, "y2": y2})


@app.route("/api/reglamento/sources/<int:source_id>/min-time", methods=["POST"])
def reglamento_min_time(source_id):
    data = request.get_json(silent=True) or {}
    t = max(1, min(300, int(data.get("seconds", 10))))
    set_setting("reglamento_min_time", str(t))
    ReglamentoManager.get().set_min_time(source_id, t)
    return jsonify({"min_time": t})


@app.route("/api/reglamento/sources/<int:source_id>/analytics")
def reglamento_analytics(source_id):
    days = request.args.get("days", 7, type=int)
    from src.database import get_reglamento_analytics
    return jsonify(get_reglamento_analytics(source_id, days))


@app.route("/api/reglamento/sources/<int:source_id>/evidencias")
def reglamento_evidencias(source_id):
    stats = ReglamentoManager.get().get_stats(source_id)
    if stats is None:
        return jsonify([])
    return jsonify(stats.get("evidencias", []))


@app.route("/api/reglamento/settings/jpeg-q", methods=["POST"])
def reglamento_jpeg_q():
    data = request.get_json(silent=True) or {}
    q = max(10, min(100, int(data.get("jpeg_q", 72))))
    set_setting("reglamento_jpeg_q", str(q))
    mgr = ReglamentoManager.get()
    with mgr._lock:
        for sid in mgr.pipelines:
            mgr.pipelines[sid].set_jpeg_q(q)
    return jsonify({"jpeg_q": q})


@app.route("/api/reglamento/settings/max-dim", methods=["POST"])
def reglamento_max_dim():
    data = request.get_json(silent=True) or {}
    d = max(0, min(4096, int(data.get("max_dim", 0))))
    set_setting("reglamento_max_dim", str(d))
    mgr = ReglamentoManager.get()
    with mgr._lock:
        for sid in mgr.pipelines:
            mgr.pipelines[sid].set_max_dim(d)
    return jsonify({"max_dim": d})


@app.route("/api/reglamento/settings/frame-step", methods=["POST"])
def reglamento_frame_step():
    data = request.get_json(silent=True) or {}
    s = max(1, min(10, int(data.get("frame_step", 1))))
    set_setting("reglamento_frame_step", str(s))
    mgr = ReglamentoManager.get()
    with mgr._lock:
        for sid in mgr.pipelines:
            mgr.pipelines[sid].set_frame_step(s)
    return jsonify({"frame_step": s})


# ─────────────────────────────────────────────
# API Carga / Descarga
# ─────────────────────────────────────────────

_register_standard_routes(app, "carga_descarga", CargaDescargaManager)

@app.route("/api/carga_descarga/sources/<int:source_id>/start", methods=["POST"])
def carga_descarga_start(source_id):
    sources = get_sources("carga_descarga")
    src = next((s for s in sources if s["id"] == source_id), None)
    if not src:
        return jsonify({"error": "Source not found"}), 404
    s = get_settings()
    line_mode = s.get("carga_descarga_line_mode", "horizontal")
    line_pos  = int(s.get("carga_descarga_line_pos", "50"))
    raw_models = s.get("carga_descarga_models", "[]")
    try:
        models = json.loads(raw_models)
    except (json.JSONDecodeError, TypeError):
        models = []
    model_path = None; classes = None
    if models:
        idx = max(0, min(int(s.get("carga_descarga_cur_model", "0")), len(models) - 1))
        m = models[idx]
        model_path = m.get("path")
        cls_str = m.get("classes", "")
        classes = [int(c.strip()) for c in cls_str.split(",") if c.strip()] if cls_str else None
    fps_limit = _get_fps_limit(src, src["type"])
    try:
        CargaDescargaManager.get().start(source_id, src["path"],
            _func_state_for("carga_descarga"),
            float(s.get("carga_descarga_conf", "0.35")),
            s.get("carga_descarga_half", "0") == "1",
            model_path, classes, line_mode, line_pos,
            fps_limit=fps_limit)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 429
    return jsonify({"started": source_id})


@app.route("/api/carga_descarga/sources/<int:source_id>/line-mode", methods=["POST"])
def carga_descarga_line_mode(source_id):
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "horizontal")
    if mode not in ("horizontal", "vertical"):
        return jsonify({"error": "Invalid mode"}), 400
    set_setting("carga_descarga_line_mode", mode)
    CargaDescargaManager.get().set_line_mode(source_id, mode)
    return jsonify({"line_mode": mode})


@app.route("/api/carga_descarga/sources/<int:source_id>/invert", methods=["POST"])
def carga_descarga_invert(source_id):
    data = request.get_json(silent=True) or {}
    inverted = data.get("inverted", False)
    CargaDescargaManager.get().set_inverted(source_id, inverted)
    return jsonify({"inverted": inverted})


@app.route("/api/carga_descarga/sources/<int:source_id>/line-pos", methods=["POST"])
def carga_descarga_line_pos(source_id):
    data = request.get_json(silent=True) or {}
    pct = max(0, min(100, int(data.get("pct", 50))))
    set_setting("carga_descarga_line_pos", str(pct))
    CargaDescargaManager.get().set_line_pos(source_id, pct)
    return jsonify({"line_pos": pct})


@app.route("/api/carga_descarga/sources/<int:source_id>/reload-model", methods=["POST"])
def carga_descarga_reload_model(source_id):
    data = request.get_json(silent=True) or {}
    raw = get_settings().get("carga_descarga_models", "[]")
    try:
        models = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        models = []
    idx = int(data.get("model_idx", 0))
    if not models or idx < 0 or idx >= len(models):
        return jsonify({"error": "Model not found"}), 404
    m = models[idx]
    cls_str = m.get("classes", "")
    classes = [int(c.strip()) for c in cls_str.split(",") if c.strip()] if cls_str else None
    set_setting("carga_descarga_cur_model", str(idx))
    CargaDescargaManager.get().reload_model(source_id, m.get("path"), classes, str(idx))
    return jsonify({"model_idx": idx, "name": m.get("name")})


@app.route("/api/carga_descarga/sources/<int:source_id>/analytics")
def carga_descarga_analytics(source_id):
    days = request.args.get("days", 7, type=int)
    from src.database import get_carga_descarga_analytics
    return jsonify(get_carga_descarga_analytics(source_id, days))


@app.route("/api/carga_descarga/sources/<int:source_id>/models")
def carga_descarga_models_list(source_id):
    raw = get_settings().get("carga_descarga_models", "[]")
    try:
        models = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        models = []
    cur = int(get_settings().get("carga_descarga_cur_model", "0"))
    for i, m in enumerate(models):
        m["idx"] = i; m["active"] = (i == cur)
    return jsonify(models)


# ─────────────────────────────────────────────
# API EPP
# ─────────────────────────────────────────────

_register_standard_routes(app, "epp", EppManager)

@app.route("/api/epp/sources/<int:source_id>/start", methods=["POST"])
def epp_start(source_id):
    sources = get_sources("epp")
    src = next((s for s in sources if s["id"] == source_id), None)
    if not src:
        return jsonify({"error": "Source not found"}), 404
    s = get_settings()
    fps_limit = _get_fps_limit(src, src["type"])
    try:
        EppManager.get().start(source_id, src["path"],
            _func_state_for("epp"),
            float(s.get("epp_conf", "0.35")),
            s.get("epp_half", "0") == "1",
            s.get("epp_model") or None,
            fps_limit=fps_limit)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 429
    return jsonify({"started": source_id})


@app.route("/api/epp/sources/<int:source_id>/master", methods=["POST"])
def epp_master(source_id):
    data = request.get_json(silent=True) or {}
    enabled = data.get("enabled", True)
    EppManager.get().set_master(source_id, enabled)
    return jsonify({"master_enabled": enabled})


@app.route("/api/epp/sources/<int:source_id>/required-epp", methods=["POST"])
def epp_required(source_id):
    data = request.get_json(silent=True) or {}
    classes = set(int(c) for c in data.get("classes", [4, 6]))
    EppManager.get().set_required_epp(source_id, classes)
    return jsonify({"required": sorted(classes)})


# ─────────────────────────────────────────────
# API Smoke / Humo-Fuego
# ─────────────────────────────────────────────

_register_standard_routes(app, "smoke", SmokeManager)

@app.route("/api/smoke/sources/<int:source_id>/start", methods=["POST"])
def smoke_start(source_id):
    sources = get_sources("smoke")
    src = next((s for s in sources if s["id"] == source_id), None)
    if not src:
        return jsonify({"error": "Source not found"}), 404
    s = get_settings()
    fps_limit = _get_fps_limit(src, src["type"])
    try:
        SmokeManager.get().start(source_id, src["path"],
            _func_state_for("smoke"),
            float(s.get("smoke_conf", "0.35")),
            s.get("smoke_half", "0") == "1",
            s.get("smoke_model") or None,
            fps_limit=fps_limit)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 429
    return jsonify({"started": source_id})


# ─────────────────────────────────────────────
# API Vehículos
# ─────────────────────────────────────────────

_register_standard_routes(app, "vehiculos", VehiculosManager)

@app.route("/api/vehiculos/sources/<int:source_id>/start", methods=["POST"])
def vehiculos_start(source_id):
    sources = get_sources("vehiculos")
    src = next((s for s in sources if s["id"] == source_id), None)
    if not src:
        return jsonify({"error": "Source not found"}), 404
    s = get_settings()
    fps_limit = _get_fps_limit(src, src["type"])
    try:
        VehiculosManager.get().start(source_id, src["path"],
            _func_state_for("vehiculos"),
            float(s.get("vehiculos_conf", "0.35")),
            s.get("vehiculos_half", "0") == "1",
            s.get("vehiculos_model") or None,
            plate_model_path=s.get("vehiculos_plate_model") or None,
            plate_conf_thresh=float(s.get("vehiculos_plate_conf", "0.35")),
            classes=None,
            line_mode=s.get("vehiculos_line_mode", "horizontal"),
            line_pos=int(s.get("vehiculos_line_pos", "50")),
            fps_limit=fps_limit)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 429
    return jsonify({"started": source_id})


@app.route("/api/vehiculos/sources/<int:source_id>/line-mode", methods=["POST"])
def vehiculos_line_mode(source_id):
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "horizontal")
    if mode not in ("horizontal", "vertical"):
        return jsonify({"error": "Invalid mode"}), 400
    set_setting("vehiculos_line_mode", mode)
    VehiculosManager.get().set_line_mode(source_id, mode)
    return jsonify({"mode": mode})


@app.route("/api/vehiculos/sources/<int:source_id>/line-pos", methods=["POST"])
def vehiculos_line_pos(source_id):
    data = request.get_json(silent=True) or {}
    pos = max(0, min(100, int(data.get("pos", 50))))
    set_setting("vehiculos_line_pos", str(pos))
    VehiculosManager.get().set_line_pos(source_id, pos)
    return jsonify({"pos": pos})


@app.route("/api/vehiculos/sources/<int:source_id>/plate-detection", methods=["POST"])
def vehiculos_plate_detection(source_id):
    data = request.get_json(silent=True) or {}
    enabled = data.get("enabled", False)
    VehiculosManager.get().set_plate_detection(source_id, enabled)
    return jsonify({"plate_detection": enabled})


@app.route("/api/vehiculos/upload-plate-model", methods=["POST"])
def vehiculos_upload_plate_model():
    if "model" not in request.files:
        return jsonify({"error": "No file was sent"}), 400
    f = request.files["model"]
    if not f.filename.endswith(".pt"):
        return jsonify({"error": "Only .pt files"}), 400
    fname = secure_filename(f.filename)
    f.save(os.path.join(MODELS_FOLDER, fname))
    rel = os.path.join("static", "uploads", "models", fname)
    set_setting("vehiculos_plate_model", rel)
    return jsonify({"model": fname})


# ─────────────────────────────────────────────
# API Tanques de Gas
# ─────────────────────────────────────────────

_register_standard_routes(app, "tanques_gas", TanquesGasManager)

@app.route("/api/tanques_gas/sources/<int:source_id>/start", methods=["POST"])
def tanques_gas_start(source_id):
    sources = get_sources("tanques_gas")
    src = next((s for s in sources if s["id"] == source_id), None)
    if not src:
        return jsonify({"error": "Source not found"}), 404
    s = get_settings()
    fps_limit = _get_fps_limit(src, src["type"])
    try:
        TanquesGasManager.get().start(source_id, src["path"],
            _func_state_for("tanques_gas"),
            float(s.get("tanques_gas_conf", "0.35")),
            s.get("tanques_gas_half", "0") == "1",
            s.get("tanques_gas_model") or None,
            s.get("tanques_gas_pose_model") or None,
            s.get("tanques_gas_smoke_model") or None,
            float(s.get("tanques_gas_smoke_conf", "0.35")),
            s.get("tanques_gas_line_mode", "horizontal"),
            int(s.get("tanques_gas_line_pos", "50")),
            fps_limit=fps_limit)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 429
    return jsonify({"started": source_id})


@app.route("/api/tanques_gas/sources/<int:source_id>/line-mode", methods=["POST"])
def tanques_gas_line_mode(source_id):
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "horizontal")
    if mode not in ("horizontal", "vertical", "rectangle", "custom_line", "custom_rect"):
        return jsonify({"error": "Invalid mode"}), 400
    set_setting("tanques_gas_line_mode", mode)
    TanquesGasManager.get().set_line_mode(source_id, mode)
    return jsonify({"line_mode": mode})


@app.route("/api/tanques_gas/sources/<int:source_id>/line-pos", methods=["POST"])
def tanques_gas_line_pos(source_id):
    data = request.get_json(silent=True) or {}
    pct = max(0, min(100, int(data.get("pct", 50))))
    set_setting("tanques_gas_line_pos", str(pct))
    TanquesGasManager.get().set_line_pos(source_id, pct)
    return jsonify({"line_pos": pct})


@app.route("/api/tanques_gas/sources/<int:source_id>/custom-line", methods=["POST"])
def tanques_gas_custom_line(source_id):
    data = request.get_json(silent=True) or {}
    p1x = float(data.get("p1x", 0.25))
    p1y = float(data.get("p1y", 0.25))
    p2x = float(data.get("p2x", 0.75))
    p2y = float(data.get("p2y", 0.75))
    TanquesGasManager.get().set_custom_line(source_id, p1x, p1y, p2x, p2y)
    return jsonify({"p1x": p1x, "p1y": p1y, "p2x": p2x, "p2y": p2y})


@app.route("/api/tanques_gas/sources/<int:source_id>/custom-rect", methods=["POST"])
def tanques_gas_custom_rect(source_id):
    data = request.get_json(silent=True) or {}
    points = data.get("points", [])
    if len(points) != 4:
        return jsonify({"error": "Exactly 4 points are required"}), 400
    TanquesGasManager.get().set_custom_rect(source_id, points)
    return jsonify({"points": points})


@app.route("/api/tanques_gas/sources/<int:source_id>/rect-area", methods=["POST"])
def tanques_gas_rect_area(source_id):
    data = request.get_json(silent=True) or {}
    x1 = int(data.get("x1", 20)); y1 = int(data.get("y1", 20))
    x2 = int(data.get("x2", 80)); y2 = int(data.get("y2", 80))
    TanquesGasManager.get().set_rect_area(source_id, x1, y1, x2, y2)
    return jsonify({"x1": x1, "y1": y1, "x2": x2, "y2": y2})


# ── Restricted Areas ──

@app.route("/api/tanques_gas/sources/<int:source_id>/restricted-areas", methods=["GET", "POST"])
def tanques_gas_restricted_areas(source_id):
    if request.method == "GET":
        with TanquesGasManager.get()._lock:
            p = TanquesGasManager.get().pipelines.get(source_id)
        if p:
            return jsonify(p.restricted_areas)
        return jsonify([])
    data = request.get_json(silent=True) or {}
    points = data.get("points", [])
    restrict_type = data.get("restrict_type", "ambos")
    TanquesGasManager.get().add_restricted_area(source_id, points, restrict_type)
    with TanquesGasManager.get()._lock:
        p = TanquesGasManager.get().pipelines.get(source_id)
    return jsonify(p.restricted_areas if p else [])


@app.route("/api/tanques_gas/sources/<int:source_id>/restricted-areas/<area_id>", methods=["DELETE"])
def tanques_gas_remove_restricted_area(source_id, area_id):
    from urllib.parse import unquote
    area_id = unquote(area_id)
    TanquesGasManager.get().remove_restricted_area(source_id, area_id)
    return jsonify({"removed": area_id})


# ── Teach ──

@app.route("/api/tanques_gas/sources/<int:source_id>/teach/start", methods=["POST"])
def tanques_gas_teach_start(source_id):
    TanquesGasManager.get().start_teach(source_id)
    return jsonify({"teach_mode": True})


@app.route("/api/tanques_gas/sources/<int:source_id>/teach/cancel", methods=["POST"])
def tanques_gas_teach_cancel(source_id):
    TanquesGasManager.get().cancel_teach(source_id)
    return jsonify({"teach_mode": False})


@app.route("/api/tanques_gas/sources/<int:source_id>/teach/capture")
def tanques_gas_teach_capture(source_id):
    data = TanquesGasManager.get().get_teach_capture(source_id)
    if data is None:
        return jsonify(None)
    return jsonify(data)


@app.route("/api/tanques_gas/sources/<int:source_id>/teach/save", methods=["POST"])
def tanques_gas_teach_save(source_id):
    data = request.get_json(silent=True) or {}
    action = data.get("action", "").strip()
    if not action:
        return jsonify({"error": "Action required"}), 400
    ok = TanquesGasManager.get().save_teach_action(source_id, action)
    return jsonify({"ok": ok})


@app.route("/api/tanques_gas/sources/<int:source_id>/teach/samples")
def tanques_gas_teach_samples(source_id):
    from src.modules.tanques_gas import _TEACH_SAMPLES
    return jsonify({"samples": _TEACH_SAMPLES})


@app.route("/api/tanques_gas/sources/<int:source_id>/actions-info")
def tanques_gas_actions_info(source_id):
    info = TanquesGasManager.get().get_actions_info(source_id)
    return jsonify(info)


@app.route("/api/tanques_gas/upload-pose-model", methods=["POST"])
def tanques_gas_upload_pose_model():
    if "model" not in request.files:
        return jsonify({"error": "No file was sent"}), 400
    f = request.files["model"]
    if not f.filename.endswith(".pt"):
        return jsonify({"error": "Only .pt files"}), 400
    fname = secure_filename(f.filename)
    f.save(os.path.join(MODELS_FOLDER, fname))
    rel = os.path.join("static", "uploads", "models", fname)
    set_setting("tanques_gas_pose_model", rel)
    return jsonify({"model": fname})


@app.route("/api/tanques_gas/upload-smoke-model", methods=["POST"])
def tanques_gas_upload_smoke_model():
    if "model" not in request.files:
        return jsonify({"error": "No file was sent"}), 400
    f = request.files["model"]
    if not f.filename.endswith(".pt"):
        return jsonify({"error": "Only .pt files"}), 400
    fname = secure_filename(f.filename)
    f.save(os.path.join(MODELS_FOLDER, fname))
    rel = os.path.join("static", "uploads", "models", fname)
    set_setting("tanques_gas_smoke_model", rel)
    return jsonify({"model": fname})


# ─────────────────────────────────────────────
# API Fuentes (sources) CRUD
# ─────────────────────────────────────────────

@app.route("/api/sources", methods=["POST"])
def api_add_source():
    if g.get("access_type") == "limited":
        return jsonify({"error": "You don't have permission to add sources"}), 403
    data = request.get_json(silent=True) or {}
    module_id = data.get("module_id")
    name = data.get("name", "").strip()
    src_type = data.get("type", "stream")
    path = data.get("path", "").strip()
    if module_id not in MODULES_META:
        return jsonify({"error": "Invalid module"}), 400
    if not name or not path:
        return jsonify({"error": "Name and path are required"}), 400
    if src_type not in ("video", "stream"):
        return jsonify({"error": "Invalid type"}), 400
    path = _normalize_path(path, src_type)
    src = add_source(module_id, name, src_type, path)
    return jsonify(src), 201


@app.route("/api/sources/<int:source_id>", methods=["PUT", "DELETE"])
def api_source_crud(source_id):
    if g.get("access_type") == "limited":
        return jsonify({"error": "You don't have permission to modify sources"}), 403
    if request.method == "DELETE":
        src = get_source(source_id)
        if src:
            mgr = _get_manager(src["module_id"])
            if mgr:
                mgr.stop(source_id)
        result = delete_source(source_id)
        return jsonify(result)
    data = request.get_json(silent=True) or {}
    name = data.get("name")
    path = data.get("path")
    if path is not None:
        sources = get_sources()
        existing = next((s for s in sources if s["id"] == source_id), None)
        src_type = existing["type"] if existing else "stream"
        path = _normalize_path(path, src_type)
    src = update_source(source_id, name=name, path=path)
    if not src:
        return jsonify({"error": "Source not found"}), 404
    return jsonify(src)


@app.route("/api/sources/upload-video", methods=["POST"])
def api_upload_video():
    if g.get("access_type") == "limited":
        return jsonify({"error": "You don't have permission to upload videos"}), 403
    if "video" not in request.files:
        return jsonify({"error": "No file was sent"}), 400
    f = request.files["video"]
    if not f.filename:
        return jsonify({"error": "Empty name"}), 400
    ext = f.filename.rsplit(".", 1)[1].lower() if "." in f.filename else ""
    if ext not in ALLOWED_VIDEO:
        return jsonify({"error": "Video format not allowed"}), 400
    fname = secure_filename(f.filename)
    dest = os.path.join(VIDEOS_FOLDER, fname)
    f.save(dest)
    return jsonify({"path": os.path.join("static", "uploads", "videos", fname)})


# ─────────────────────────────────────────────
# Source Config API (v3.0)
# ─────────────────────────────────────────────

@app.route("/api/sources/<int:source_id>/config", methods=["GET", "POST"])
def api_source_config(source_id):
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        for key, value in data.items():
            save_source_config(source_id, key, str(value))
        return jsonify({"saved": True})
    return jsonify(get_source_config(source_id))


@app.route("/api/sources/<int:source_id>/config/<key>", methods=["GET", "POST"])
def api_source_config_key(source_id, key):
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        value = data.get("value", "")
        save_source_config(source_id, key, str(value))
        return jsonify({"key": key, "value": value})
    default = request.args.get("default")
    value = get_source_config_value(source_id, key, default)
    return jsonify({"key": key, "value": value})


# ─────────────────────────────────────────────
# Analytics API (v3.0)
# ─────────────────────────────────────────────

@app.route("/api/analytics/<module_id>/summary")
def api_analytics_summary(module_id):
    if module_id not in MODULES_META:
        return jsonify({"error": "Module not found"}), 404
    days = request.args.get("days", 7, type=int)
    source_id = request.args.get("source_id", None, type=int)
    mgr = _get_manager(module_id)
    active_sources = []
    if mgr:
        with mgr._lock:
            active_sources = list(mgr.pipelines.keys())

    counters = {}
    if active_sources:
        for sid in active_sources:
            stats = mgr.get_stats(sid)
            if stats:
                counters[str(sid)] = stats
    else:
        for src in get_sources(module_id):
            sid = src["id"]
            persisted = load_module_counters(module_id, sid)
            if persisted:
                counters[str(sid)] = persisted

    data = get_module_analytics(module_id, source_id, days)

    return jsonify({
        "counters": counters,
        "events": data["last_events"],
        "daily": data["daily"],
        "active": len(active_sources) > 0,
        "active_sources": active_sources,
        "event_counts": data["event_counts"],
    })


@app.route("/api/analytics/<module_id>/events")
def api_analytics_events(module_id):
    if module_id not in MODULES_META:
        return jsonify({"error": "Module not found"}), 404
    days = request.args.get("days", 7, type=int)
    source_id = request.args.get("source_id", None, type=int)
    event_type = request.args.get("event_type", None)
    limit = request.args.get("limit", 100, type=int)
    events = get_module_events(module_id, source_id, event_type, days, limit)
    return jsonify({"events": events})


@app.route("/api/analytics/<module_id>/timeline")
def api_analytics_timeline(module_id):
    if module_id not in MODULES_META:
        return jsonify({"error": "Module not found"}), 404
    days = request.args.get("days", 7, type=int)
    source_id = request.args.get("source_id", None, type=int)
    data = get_module_analytics(module_id, source_id, days)
    return jsonify({"data": data["daily"]})


# ─────────────────────────────────────────────
# API Troncos — Dashboard Analítico (v3.0)
# ─────────────────────────────────────────────

CAT_CLASSES = (0, 1, 2, 3, 4, 5)
EXCEPTION_CATEGORY = 6
CAT_NAMES_TYPES_LOOKUP = {f"cat_{c}": c for c in CAT_CLASSES}
CAT_NAMES_TYPES_LOOKUP["cat_exceptions"] = EXCEPTION_CATEGORY
ALL_CATEGORIES = CAT_CLASSES + (EXCEPTION_CATEGORY,)


@app.route("/api/analytics/troncos/comprehensive")
def api_troncos_comprehensive():
    days = request.args.get("days", 29, type=int)
    days = max(1, min(days, 3650))
    source_id = request.args.get("source_id", None, type=int)

    all_sources = get_sources("troncos")
    sources = all_sources
    if source_id is not None:
        sources = [s for s in sources if s["id"] == source_id]
    if not sources:
        return jsonify({
            "kpis": {}, "counts": {}, "sources": [], "dailyTimeline": [],
            "categoryDistribution": {}, "recentEvents": [],
            "availableSources": [{"id": s["id"], "name": s["name"]} for s in all_sources],
            "empty": True,
        })

    source_ids = [s["id"] for s in sources]
    source_map = {s["id"]: s for s in sources}
    today_str = datetime.now().strftime("%Y-%m-%d")

    daily_raw = get_module_analytics_daily("troncos", source_id, days)

    cat_total = {c: 0 for c in ALL_CATEGORIES}
    src_cat = {sid: {c: 0 for c in ALL_CATEGORIES} for sid in source_ids}
    daily_agg = {}

    for d in daily_raw:
        cat = CAT_NAMES_TYPES_LOOKUP.get(d["event_type"])
        if cat is None:
            continue
        sid = d["source_id"]
        if sid not in src_cat:
            continue
        cnt = d["count"]
        day = d["day"]

        cat_total[cat] += cnt
        src_cat[sid][cat] += cnt
        if day not in daily_agg:
            daily_agg[day] = {c: 0 for c in ALL_CATEGORIES}
        daily_agg[day][cat] += cnt

    daily_timeline = []
    for day, cats in sorted(daily_agg.items()):
        daily_timeline.append({"day": day, "total": sum(cats.values()), "counts": cats})

    total_count = sum(cat_total.values())
    today_total = 0
    for r in daily_timeline:
        if r["day"] == today_str:
            today_total = r["total"]
            break

    daily_totals = [r["total"] for r in daily_timeline if r["total"] > 0]
    promedio_diario = round(sum(daily_totals) / len(daily_totals), 1) if daily_totals else 0
    pico_diario = max(daily_totals) if daily_totals else 0
    pico_dia = ""
    for r in daily_timeline:
        if r["total"] == pico_diario:
            pico_dia = r["day"]
            break

    # Eventos recientes (con diámetro para promediar)
    raw_events = get_module_events("troncos", source_id, days=days, limit=500)
    recent_events = []
    diam_sum = 0
    diam_count = 0
    for e in raw_events:
        cat = CAT_NAMES_TYPES_LOOKUP.get(e["event_type"])
        if cat is None:
            continue
        if e["source_id"] not in source_map:
            continue
        event_data = e.get("event_data") or {}
        diameter = event_data.get("diameter") if isinstance(event_data, dict) else None
        recent_events.append({
            "id": e["id"],
            "source_id": e["source_id"],
            "source_name": source_map.get(e["source_id"], {}).get("name", f"Source {e['source_id']}"),
            "event_type": e["event_type"],
            "category": cat,
            "label": e.get("label", ""),
            "created_at": e.get("created_at", ""),
            "has_capture": bool(e.get("capture_path")),
            "diameter": diameter,
        })
        if cat in CAT_CLASSES and isinstance(diameter, (int, float)) and diameter > 0:
            diam_sum += diameter
            diam_count += 1

    avg_diameter = round(diam_sum / diam_count, 1) if diam_count else None

    # Desglose por fuente
    source_list = []
    for sid in source_ids:
        src_total_cur = sum(src_cat[sid].values())
        contrib = round((src_total_cur / total_count * 100), 1) if total_count > 0 else 0
        source_list.append({
            "id": sid,
            "name": source_map[sid]["name"],
            "total": src_total_cur,
            "counts": src_cat[sid],
            "contribucion": contrib,
        })
    source_list.sort(key=lambda x: x["total"], reverse=True)

    kpis = {
        "totalCount": total_count,
        "today": today_total,
        "promedioDiario": promedio_diario,
        "picoDiario": pico_diario,
        "picoDia": pico_dia,
        "totalEventos": total_count,
        "avgDiameter": avg_diameter,
    }

    return jsonify({
        "kpis": kpis,
        "counts": cat_total,
        "sources": source_list,
        "dailyTimeline": daily_timeline,
        "categoryDistribution": cat_total,
        "recentEvents": recent_events,
        "availableSources": [{"id": s["id"], "name": s["name"]} for s in all_sources],
        "empty": False,
    })


# ─────────────────────────────────────────────
# API Tanques de Gas — Dashboard Analítico (v3.0)
# ─────────────────────────────────────────────

import random
from datetime import datetime, timedelta

@app.route("/api/analytics/tanques_gas/comprehensive")
def api_tanques_gas_comprehensive():
    days = request.args.get("days", 29, type=int)
    days = max(1, min(days, 3650))
    source_id = request.args.get("source_id", None, type=int)

    all_sources = get_sources("tanques_gas")
    sources = all_sources
    if source_id is not None:
        sources = [s for s in sources if s["id"] == source_id]
    if not sources:
        return jsonify({
            "kpis": {}, "sources": [], "dailyTimeline": [],
            "eventTypeDistribution": {}, "actionDistribution": {},
            "recentEvents": [],
            "availableSources": [{"id": s["id"], "name": s["name"]} for s in all_sources],
            "empty": True,
        })

    source_ids = [s["id"] for s in sources]
    source_map  = {s["id"]: s for s in sources}

    today_str = datetime.now().strftime("%Y-%m-%d")

    # ── Eventos diarios por fuente (fuente de verdad) ──
    daily_raw = get_module_analytics_daily("tanques_gas", source_id, days)

    entries_by_source  = {}
    exits_by_source    = {}
    acciones_by_source = {}
    action_dist        = {}
    event_type_dist    = {"entry": 0, "exit": 0, "smoke_detected": 0, "action": 0}
    smoke_sources      = set()
    daily_agg          = {}

    for d in daily_raw:
        day = d["day"]
        et  = d["event_type"]
        cnt = d["count"]
        sid = d["source_id"]

        if day not in daily_agg:
            daily_agg[day] = {"day": day, "entradas": 0, "salidas": 0, "acciones": 0, "smoke": 0}

        if et in ("entry", "entrada"):
            daily_agg[day]["entradas"] += cnt
            entries_by_source[sid] = entries_by_source.get(sid, 0) + cnt
            event_type_dist["entry"] += cnt
        elif et in ("exit", "salida"):
            daily_agg[day]["salidas"] += cnt
            exits_by_source[sid] = exits_by_source.get(sid, 0) + cnt
            event_type_dist["exit"] += cnt
        elif et == "smoke_detected":
            daily_agg[day]["smoke"] += cnt
            event_type_dist["smoke_detected"] += cnt
            smoke_sources.add(sid)
        else:
            daily_agg[day]["acciones"] += cnt
            acciones_by_source[sid] = acciones_by_source.get(sid, 0) + cnt
            action_dist[et] = action_dist.get(et, 0) + cnt
            event_type_dist["action"] += cnt

    total_entradas = sum(entries_by_source.values())
    total_salidas  = sum(exits_by_source.values())
    total_acciones = sum(acciones_by_source.values())
    total_smoke_events = event_type_dist["smoke_detected"]

    ocupacion_neta = total_entradas - total_salidas
    ratio_es = round(total_entradas / total_salidas, 2) if total_salidas > 0 else total_entradas

    today_events = sum(d["count"] for d in daily_raw if d["day"] == today_str)
    total_eventos = sum(v for v in event_type_dist.values())

    daily_timeline = sorted(daily_agg.values(), key=lambda x: x["day"])

    # KPIs históricos
    daily_totals = [d["entradas"] + d["salidas"] + d["acciones"] + d["smoke"] for d in daily_timeline]
    promedio_diario = round(sum(daily_totals) / len(daily_totals), 1) if daily_totals else 0
    pico_diario = max(daily_totals) if daily_totals else 0
    pico_dia = ""
    for d in daily_timeline:
        if d["entradas"] + d["salidas"] + d["acciones"] + d["smoke"] == pico_diario:
            pico_dia = d["day"]
            break

    # Desglose por fuente (todas las fuentes del conjunto filtrado)
    source_list = []
    for sid in source_ids:
        ent = entries_by_source.get(sid, 0)
        sal = exits_by_source.get(sid, 0)
        acc = acciones_by_source.get(sid, 0)
        smk = sid in smoke_sources
        contrib = round((ent / total_entradas * 100), 1) if total_entradas > 0 else 0
        source_list.append({
            "id": sid,
            "name": source_map[sid]["name"],
            "entradas": ent,
            "salidas": sal,
            "acciones": acc,
            "smoke": smk,
            "contribucion": contrib,
        })

    source_list.sort(key=lambda x: x["entradas"], reverse=True)

    fuente_top_id = max(entries_by_source, key=entries_by_source.get) if entries_by_source else None
    fuente_top = None
    if fuente_top_id:
        fuente_top = {
            "id": fuente_top_id,
            "name": source_map[fuente_top_id]["name"],
            "entradas": entries_by_source[fuente_top_id],
        }

    kpis = {
        "totalEntradas": total_entradas,
        "totalSalidas": total_salidas,
        "ocupacionNeta": ocupacion_neta,
        "ratioES": ratio_es,
        "eventosHoy": today_events,
        "alertasHumo": len(smoke_sources),
        "promedioDiario": promedio_diario,
        "picoDiario": pico_diario,
        "picoDia": pico_dia,
        "totalEventos": total_eventos,
        "totalAcciones": total_acciones,
        "totalSmoke": total_smoke_events,
        "fuenteTop": fuente_top,
    }

    # Eventos recientes (filtrados por fuente y período)
    raw_events = get_module_events("tanques_gas", source_id, days=days, limit=200)
    recent_events = []
    for e in raw_events:
        recent_events.append({
            "id": e["id"],
            "source_id": e["source_id"],
            "source_name": source_map.get(e["source_id"], {}).get("name", f"Source {e['source_id']}"),
            "event_type": e["event_type"],
            "label": e.get("label", ""),
            "created_at": e.get("created_at", ""),
            "has_capture": bool(e.get("capture_path")),
        })

    return jsonify({
        "kpis": kpis,
        "sources": source_list,
        "dailyTimeline": daily_timeline,
        "eventTypeDistribution": event_type_dist,
        "actionDistribution": action_dist,
        "recentEvents": recent_events,
        "availableSources": [{"id": s["id"], "name": s["name"]} for s in all_sources],
        "totalEvents": len(raw_events),
        "empty": False,
    })


@app.route("/api/analytics/tanques_gas/seed", methods=["POST"])
def api_tanques_gas_seed():
    if g.get("access_type") == "limited":
        return jsonify({"error": "You don't have permission"}), 403
    sources = get_sources("tanques_gas")
    if not sources:
        return jsonify({"error": "No sources registered. Register at least one source first."}), 400

    # Clear existing data
    from src.database import get_conn
    with get_conn() as conn:
        conn.execute("DELETE FROM module_events WHERE module_id='tanques_gas'")
        conn.execute("DELETE FROM module_counters WHERE module_id='tanques_gas'")
        conn.commit()

    # Reset smoke status in active pipelines
    mgr = _get_manager("tanques_gas")
    if mgr:
        with mgr._lock:
            for sid, pipeline in list(mgr.pipelines.items()):
                try:
                    pipeline.smoke_detected = False
                    pipeline.alert_triggered = False
                    pipeline.total_in = 0
                    pipeline.total_out = 0
                    pipeline._action_count = {}
                    pipeline.first_detection_time = None
                    pipeline._counted_tracks = set()
                except Exception:
                    pass

    source_ids  = [s["id"] for s in sources]
    source_names = {s["id"]: s["name"] for s in sources}
    n_sources = len(source_ids)

    action_types = [
        ("levantar", 0.40), ("transportar", 0.30),
        ("inspeccionar", 0.20), ("descargar", 0.10),
    ]
    action_labels_map = {
        "levantar": "Lifting tank",
        "transportar": "Transporting tank",
        "inspeccionar": "Inspecting tank",
        "descargar": "Unloading tank",
    }

    # Base activity per source (weight determines relative intensity)
    source_weights = [1.0, 3.5, 2.0, 0.8, 2.5]  # F1, F2, F3, F4, F5...
    entry_exit_ratio = [0.75, 0.72, 0.70, 0.55, 0.70]  # entry/(entry+exit)

    start_date = datetime(2026, 7, 1)
    end_date   = datetime(2026, 7, 29)
    all_events = []

    base_entries_per_day = 30
    current = start_date
    while current <= end_date:
        weekday = current.weekday()  # 0=Mon, 6=Sun
        is_weekend = weekday >= 5
        day_factor = 0.3 if is_weekend else 1.3

        # 5% chance of anomaly day
        is_anomaly = random.random() < 0.05
        anomaly_factor = 0 if random.random() < 0.5 else 2.5  # maintenance or spike

        day_str = current.strftime("%Y-%m-%d")

        for idx, sid in enumerate(source_ids):
            w_idx = min(idx, len(source_weights) - 1)
            weight = source_weights[w_idx]
            e_ratio = entry_exit_ratio[min(idx, len(entry_exit_ratio) - 1)]

            if is_anomaly:
                daily_base = max(0, int(base_entries_per_day * weight * anomaly_factor * random.uniform(0.8, 1.2)))
            else:
                daily_base = max(0, int(base_entries_per_day * weight * day_factor * random.uniform(0.7, 1.3)))

            daily_entries = max(0, int(daily_base * e_ratio))
            daily_exits   = max(0, daily_base - daily_entries)

            # Add some randomness
            daily_entries = max(0, int(daily_entries * random.uniform(0.85, 1.15)))
            daily_exits   = max(0, int(daily_exits * random.uniform(0.85, 1.15)))

            # Generate timestamps distributed across 8am-6pm
            hours_pool = []
            for h in range(8, 19):
                freq = 1.0
                if 9 <= h <= 11:
                    freq = 2.5  # morning peak
                elif 15 <= h <= 17:
                    freq = 2.0  # afternoon peak
                elif 12 <= h <= 13:
                    freq = 0.5  # lunch lull
                hours_pool.extend([h] * int(freq * 4))

            for _ in range(daily_entries):
                h = random.choice(hours_pool)
                m = random.randint(0, 59)
                s = random.randint(0, 59)
                ts = f"{day_str} {h:02d}:{m:02d}:{s:02d}"
                all_events.append((sid, "entry", "Tank detected", "", ts))

            for _ in range(daily_exits):
                h = random.choice(hours_pool)
                m = random.randint(0, 59)
                s = random.randint(0, 59)
                ts = f"{day_str} {h:02d}:{m:02d}:{s:02d}"
                all_events.append((sid, "exit", "Tank exiting", "", ts))

            # Actions — 10-30% of entries
            action_count = max(0, int(daily_entries * random.uniform(0.10, 0.30)))
            for _ in range(action_count):
                h = random.choice(hours_pool)
                m = random.randint(0, 59)
                s = random.randint(0, 59)
                ts = f"{day_str} {h:02d}:{m:02d}:{s:02d}"
                # Pick action type by weighted random
                r = random.random()
                cum = 0
                chosen_action = "levantar"
                for at, prob in action_types:
                    cum += prob
                    if r <= cum:
                        chosen_action = at
                        break
                all_events.append((sid, chosen_action, action_labels_map[chosen_action], "", ts))

            # Smoke — rare events
            if random.random() < 0.03:  # ~3% of source-days
                h = random.randint(8, 17)
                m = random.randint(0, 59)
                s = random.randint(0, 59)
                ts = f"{day_str} {h:02d}:{m:02d}:{s:02d}"
                all_events.append((sid, "smoke_detected", "Smoke or fire detected", "", ts))

        current += timedelta(days=1)

    # Batch insert events
    with get_conn() as conn:
        conn.executemany(
            """INSERT INTO module_events(module_id, source_id, event_type, label, description, created_at)
               VALUES ('tanques_gas', ?, ?, ?, ?, ?)""",
            all_events,
        )
        conn.commit()

    # Recalculate and save counters per source
    for sid in source_ids:
        totals = {"entry": 0, "exit": 0, "smoke_detected": 0}
        actions_local = {}
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT event_type, COUNT(*) as cnt FROM module_events WHERE module_id='tanques_gas' AND source_id=? GROUP BY event_type",
                (sid,),
            ).fetchall()
            for r in rows:
                et = r["event_type"]
                cnt = r["cnt"]
                if et in ("entry",):
                    totals["entry"] += cnt
                elif et in ("exit",):
                    totals["exit"] += cnt
                elif et == "smoke_detected":
                    totals["smoke_detected"] += cnt
                else:
                    actions_local[et] = actions_local.get(et, 0) + cnt

        counters = {
            "entrada": totals["entry"],
            "salida": totals["exit"],
            "smoke_detected": totals["smoke_detected"],
            "action_count": json.dumps(actions_local),
        }
        save_module_counters("tanques_gas", sid, counters)

    # Get total generated
    with get_conn() as conn:
        total_gen = conn.execute(
            "SELECT COUNT(*) as cnt FROM module_events WHERE module_id='tanques_gas'"
        ).fetchone()["cnt"]

    return jsonify({
        "success": True,
        "message": f"Data generated: {total_gen} events in {n_sources} sources (2026-07-01 → 2026-07-29)",
        "total_events": total_gen,
        "sources_used": n_sources,
    })


@app.route("/api/analytics/tanques_gas/clear", methods=["POST"])
def api_tanques_gas_clear():
    if g.get("access_type") == "limited":
        return jsonify({"error": "You don't have permission"}), 403
    from src.database import get_conn
    with get_conn() as conn:
        conn.execute("DELETE FROM module_events WHERE module_id='tanques_gas'")
        conn.execute("DELETE FROM module_counters WHERE module_id='tanques_gas'")
        conn.commit()

    mgr = _get_manager("tanques_gas")
    if mgr:
        with mgr._lock:
            for pipeline in list(mgr.pipelines.values()):
                try:
                    pipeline.smoke_detected = False
                    pipeline.alert_triggered = False
                    pipeline.total_in = 0
                    pipeline.total_out = 0
                    pipeline._action_count = {}
                    pipeline.first_detection_time = None
                    pipeline._counted_tracks = set()
                except Exception:
                    pass

    return jsonify({"success": True, "message": "Data deleted successfully"})


# ─────────────────────────────────────────────
# Chat API (v3.0)
# ─────────────────────────────────────────────

_MODULE_ALIASES = {
    "tanques_gas": ["tanques de gas", "gas", "tanque", "tg"],
    "personas": ["personas", "persona", "gente", "people"],
    "armas": ["armas", "arma", "weapons", "gun"],
    "acciones": ["acciones", "accion", "actions", "violencia", "robo", "caida"],
    "troncos": ["troncos", "tronco", "logs", "madera"],
    "pallets": ["pallets", "pallet", "tarimas", "tarima"],
    "cajas": ["cajas", "caja", "boxes", "box"],
    "reglamento": ["reglamento", "regulacion", "botas", "normas"],
    "carga_descarga": ["carga", "descarga", "carga y descarga", "loading"],
    "epp": ["epp", "protección", "casco", "chaleco"],
    "smoke": ["smoke", "humo", "fuego", "fire", "incendio"],
    "vehiculos": ["vehiculos", "vehiculo", "vehículo", "carros", "autos", "placa"],
}

_MODULE_KPI_DESCRIPTIONS = {
    "tanques_gas": "entradas, salidas, ocupación neta, ratio E/S, acciones detectadas, alertas de humo",
    "personas": "conteo de personas, tiempo de permanencia, mapa de calor",
    "armas": "detección de armas, captura de rostro, tipo de arma (blanca/fuego)",
    "acciones": "violencia, robo/amenaza, actividad sospechosa, uso de celular, caídas",
    "troncos": "conteo de troncos",
    "pallets": "conteo de pallets",
    "cajas": "conteo de cajas",
    "reglamento": "detección de botas, cumplimiento de tiempo, alertas",
    "carga_descarga": "conteo de carga (entradas/salidas)",
    "epp": "detección de EPP, alertas sin EPP, ranking de EPP",
    "smoke": "detección de humo/fuego",
    "vehiculos": "conteo de vehículos, detección de placas",
}

def _detect_modules(prompt: str) -> list:
    prompt_lower = prompt.lower()
    detected = []
    for mod_id, aliases in _MODULE_ALIASES.items():
        for alias in aliases:
            if alias in prompt_lower:
                detected.append(mod_id)
                break
    return detected

def _extract_days(prompt: str) -> int:
    prompt_lower = prompt.lower()
    patterns = [
        (r'(\d+)\s*dias?', 1), (r'(\d+)\s*días?', 1),
        (r'(\d+)\s*d', 1), (r'\bhoy\b', 1), (r'\bayer\b', 1),
        (r'\besta\s*semana\b', 7), (r'\beste\s*mes\b', 30),
        (r'\bmes\b', 30), (r'\bsemana\b', 7),
        (r'ultimos?\s*(\d+)', 1),
    ]
    for pattern, default in patterns:
        m = re.search(pattern, prompt_lower)
        if m:
            try:
                return int(m.group(1)) if m.lastindex else default
            except (ValueError, IndexError):
                return default
    return 7

def _fetch_module_data(module_id: str, days: int) -> dict:
    from src.database import get_module_analytics, load_module_counters
    sources = get_sources(module_id)
    summary = get_module_analytics(module_id, days=days)

    counters = {}
    for src in sources:
        c = load_module_counters(module_id, src["id"])
        if c:
            counters[src["name"]] = c

    events = get_module_events(module_id, days=days, limit=10)
    return {
        "module_id": module_id,
        "label": MODULES_META[module_id]["label"],
        "kpis": _MODULE_KPI_DESCRIPTIONS.get(module_id, ""),
        "sources": [{"id": s["id"], "name": s["name"], "type": s["type"]} for s in sources],
        "event_counts": summary.get("event_counts", {}),
        "total_events": sum(summary.get("event_counts", {}).values()),
        "counters": counters,
        "recent_events": [
            {"type": e["event_type"], "label": e.get("label", ""), "time": e.get("created_at", "")}
            for e in events[:5]
        ],
    }

import time as time_module

_gemini_last_call = 0.0

def _call_gemini(api_key: str, system_prompt: str, user_prompt: str, history: list = None) -> str:
    global _gemini_last_call
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={api_key}"

    contents = []
    if history:
        for msg in history:
            role = "model" if msg["role"] == "model" else "user"
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })
    contents.append({
        "role": "user",
        "parts": [{"text": user_prompt}]
    })

    payload = {
        "system_instruction": {
            "parts": [{"text": system_prompt}]
        },
        "contents": contents,
        "generationConfig": {
            "temperature": 0.3,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 8192,
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
        ],
    }

    # Global throttle: max 1 request a Gemini cada 3 segundos
    now = time_module.time()
    since_last = now - _gemini_last_call
    if since_last < 3 and _gemini_last_call > 0:
        time_module.sleep(3 - since_last)
    _gemini_last_call = time_module.time()

    max_retries = 3
    last_error = ""

    for attempt in range(max_retries):
        backoff = {1: 10, 2: 30}.get(attempt, 0)
        try:
            if attempt > 0:
                time_module.sleep(backoff)

            r = requests.post(url, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return "No se obtuvo respuesta del modelo. Intenta reformular tu pregunta."
            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                return "Respuesta vacía del modelo. Intenta de nuevo."
            return parts[0].get("text", "Error: Sin texto en la respuesta.")

        except requests.exceptions.Timeout:
            last_error = "La solicitud a Gemini tardó demasiado. Intenta con un prompt más simple."
        except requests.exceptions.HTTPError as e:
            status = r.status_code
            try:
                error_body = r.text[:500]
            except Exception:
                error_body = "(no se pudo leer)"
            if status == 403:
                return "La API Key de Gemini no es válida o no tiene acceso al modelo. Verifica en https://aistudio.google.com"
            if status == 429:
                last_error = f"Gemini 429: {error_body}"
                if attempt < max_retries - 1:
                    continue
                last_error = "Límite de peticiones de Gemini excedido tras reintentos. Revisa tu plan en https://aistudio.google.com"
            else:
                last_error = f"Error HTTP {status} de Gemini: {error_body}"
        except Exception as e:
            last_error = f"Error de conexión con Gemini: {str(e)}"

    return last_error or "Error al comunicarse con Gemini. Verifica tu conexión e intenta de nuevo."

def _build_system_prompt(detected_modules: list, module_data: dict) -> str:
    lines = [
        "Eres un analista de datos del sistema CVVision (Computer Vision).",
        "Trabajas EXCLUSIVAMENTE con datos de módulos de detección.",
        "",
        "REGLAS ESTRICTAS:",
        "- NO generes imágenes ni analices imágenes bajo ninguna circunstancia.",
        "- NO accedas a datos de configuración del sistema.",
        "- NO generes PDFs ni documentos descargables.",
        "- Responde SIEMPRE en Markdown.",
        "- Usa tablas, listas y formato para clarity.",
        "- Si faltan datos o la solicitud no es clara, sugiere prompts alternativos.",
        "- No menciones que eres una IA. Habla como si fueras el sistema.",
        "- Prioriza datos concretos sobre explicaciones extensas.",
        "- Si el usuario pide algo fuera del alcance (imágenes, PDFs, configuración), responde amablemente que no está disponible.",
        "",
    ]

    if detected_modules:
        lines.append("MÓDULOS DETECTADOS:")
        for mod_id in detected_modules:
            md = module_data.get(mod_id, {})
            lines.append(f"\n### {md.get('label', mod_id)}")
            lines.append(f"- KPIs disponibles: {md.get('kpis', '—')}")
            if md.get("sources"):
                lines.append(f"- Fuentes activas: {len(md['sources'])}")
            if md.get("event_counts"):
                lines.append(f"- Conteo de eventos por tipo: {json.dumps(md['event_counts'], indent=2)}")
            if md.get("counters"):
                for src_name, c in md["counters"].items():
                    filtered = {k: v for k, v in c.items() if isinstance(v, (int, float))}
                    if filtered:
                        lines.append(f"- Contadores ({src_name}): {json.dumps(filtered)}")
            if md.get("recent_events"):
                lines.append(f"- Eventos recientes: {len(md['recent_events'])}")
            lines.append("")
    else:
        lines.append("No se detectaron módulos específicos en la consulta.")
        lines.append("Sugiere al usuario que pregunte sobre uno de estos módulos:")
        for mod_id, meta in MODULES_META.items():
            lines.append(f"- {meta['label']}: {_MODULE_KPI_DESCRIPTIONS.get(mod_id, '')}")
        lines.append("")

    lines.append("INSTRUCCIONES DEL USUARIO:")
    return "\n".join(lines)


_chat_rate_limit = {}  # ip -> last_request_time

@app.route("/api/chat", methods=["POST"])
def api_chat():
    try:
        data = request.get_json(silent=True) or {}
        prompt = (data.get("prompt") or "").strip()
        session_id = data.get("session_id") or ""

        if not prompt:
            return jsonify({"error": "The prompt cannot be empty"}), 400

        # Rate limit: 1 request cada 2 segundos (evitar doble-click accidental)
        client_ip = request.remote_addr or "unknown"
        now = time_module.time()
        last = _chat_rate_limit.get(client_ip, 0)
        elapsed = now - last
        if elapsed < 2 and last > 0:
            wait_for = int(2 - elapsed) + 1
            return jsonify({
                "error": f"You must wait {wait_for} seconds before sending another message.",
                "retry_after": wait_for,
            }), 429
        _chat_rate_limit[client_ip] = now

        settings = get_settings()
        api_key = settings.get("gemini_api_key", "").strip()
        if not api_key:
            return jsonify({"error": "No Gemini API Key configured. Go to Settings > General to add it."}), 400

        # Crear o reusar sesión
        existing_messages = []
        if not session_id:
            session_id = create_chat_session(prompt[:60])
        else:
            session = get_chat_session(session_id)
            if not session:
                session_id = create_chat_session(prompt[:60])
            else:
                touch_chat_session(session_id)
                existing_messages = get_chat_messages(session_id)
                if len(existing_messages) == 0:
                    update_chat_session_title(session_id, prompt[:60])

        # Guardar mensaje del usuario
        add_chat_message(session_id, "user", prompt)

        # Detectar módulos y fetch data
        detected = _detect_modules(prompt)
        days = _extract_days(prompt)

        module_data = {}
        for mod_id in detected:
            try:
                module_data[mod_id] = _fetch_module_data(mod_id, days)
            except Exception as e:
                module_data[mod_id] = {"error": str(e)}

        # Construir prompts para Gemini
        system_prompt = _build_system_prompt(detected, module_data)

        # Llamar a Gemini con historial multi-turno
        reply = _call_gemini(api_key, system_prompt, prompt, existing_messages)

        # Guardar respuesta
        add_chat_message(session_id, "model", reply)

        return jsonify({
            "reply": reply,
            "session_id": session_id,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal error: {type(e).__name__}: {str(e)}"}), 500


@app.route("/api/chat/sessions", methods=["GET"])
def api_chat_sessions():
    sessions = get_chat_sessions()
    return jsonify({"sessions": sessions})


@app.route("/api/chat/session", methods=["POST"])
def api_chat_create_session():
    title = (request.get_json(silent=True) or {}).get("title", "New session")
    session_id = create_chat_session(title)
    return jsonify({"session_id": session_id, "title": title}), 201


@app.route("/api/chat/session/<session_id>", methods=["GET", "DELETE"])
def api_chat_session(session_id):
    if request.method == "DELETE":
        delete_chat_session(session_id)
        return jsonify({"deleted": True})
    session = get_chat_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404
    messages = get_chat_messages(session_id)
    return jsonify({"session": session, "messages": messages})


# ─────────────────────────────────────────────
# Modules Order API
# ─────────────────────────────────────────────

@app.route("/api/modules/order", methods=["GET", "POST"])
def api_modules_order():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        order = data.get("order", [])
        valid = [m for m in order if m in MODULES_META]
        set_setting("modules_order", ",".join(valid))
        return jsonify({"saved": True, "order": valid})
    raw = get_settings().get("modules_order", "")
    if raw:
        order = [m.strip() for m in raw.split(",") if m.strip() in MODULES_META]
    else:
        order = list(MODULES_META.keys())
    return jsonify({"order": order})


# ─────────────────────────────────────────────
# Evidence API (v3.0)
# ─────────────────────────────────────────────

@app.route("/api/evidences/<module_id>/<int:event_id>")
def api_evidence(module_id, event_id):
    if module_id not in MODULES_META:
        return jsonify({"error": "Module not found"}), 404
    events = get_module_events(module_id, limit=100)
    event = next((e for e in events if e["id"] == event_id), None)
    if not event:
        return jsonify({"error": "Event not found"}), 404
    capture_url = event.get("capture_path")
    extra_paths = event.get("extra_paths", [])
    return jsonify({
        "event": event,
        "capture_url": capture_url,
        "extra_urls": extra_paths,
    })


# ─────────────────────────────────────────────
# Acceso — Login / Logout
# ─────────────────────────────────────────────

@app.route("/api/access/login", methods=["POST"])
def api_access_login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    location = data.get("location", {})
    ip = request.remote_addr or ""
    ua = request.headers.get("User-Agent", "")

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    account = get_account_by_email(email)
    if not account or account["password"] != password:
        add_access_log(email, "failed", ip, location, ua, "Invalid credentials")
        return jsonify({"error": "Invalid credentials"}), 401

    if not account.get("active", 1):
        return jsonify({"error": "Account disabled"}), 403

    access_type = account["access_type"]
    hours = 24 if access_type == "full" else 1
    expires_at = (datetime.now() + timedelta(hours=hours)).isoformat()

    session["email"] = email
    session["access_type"] = access_type
    session["expires_at"] = expires_at
    session["ip_address"] = ip
    session["location"] = location

    add_access_log(email, "login", ip, location, ua)

    return jsonify({
        "success": True,
        "email": email,
        "access_type": access_type,
        "expires_at": expires_at,
    })


@app.route("/api/access/logout", methods=["POST"])
def api_access_logout():
    email = session.get("email", "")
    ip = request.remote_addr or ""
    ua = request.headers.get("User-Agent", "")
    if email:
        add_access_log(email, "logout", ip, user_agent=ua)
    session.clear()
    return jsonify({"success": True})


@app.route("/api/access/check")
def api_access_check():
    return jsonify({
        "authenticated": g.get("access_authenticated", False),
        "email": g.get("access_email", ""),
        "access_type": g.get("access_type", ""),
    })


# ─────────────────────────────────────────────
# Acceso — Accounts CRUD
# ─────────────────────────────────────────────

@app.route("/api/access/accounts", methods=["GET", "POST"])
def api_access_accounts():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        email = (data.get("email") or "").strip().lower()
        password = data.get("password") or ""
        access_type = data.get("access_type", "limited")
        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400
        if access_type not in ("full", "limited"):
            return jsonify({"error": "Type must be full or limited"}), 400
        try:
            account = add_account(email, password, access_type)
            return jsonify({"account": account, "password": password}), 201
        except Exception:
            return jsonify({"error": "The email is already registered"}), 409
    accounts = get_accounts()
    return jsonify({"accounts": accounts})


@app.route("/api/access/accounts/<int:account_id>", methods=["DELETE"])
def api_access_delete_account(account_id):
    if delete_account(account_id):
        return jsonify({"deleted": True})
    return jsonify({"error": "Account not found"}), 404


@app.route("/api/access/accounts/<int:account_id>/reset-password", methods=["POST"])
def api_access_reset_password(account_id):
    new_password = secrets.token_hex(6)
    if update_account_password(account_id, new_password):
        return jsonify({"password": new_password})
    return jsonify({"error": "Account not found"}), 404


@app.route("/api/access/accounts/by-email/<email>", methods=["GET"])
def api_access_account_by_email(email):
    account = get_account_by_email(email.strip().lower())
    if account:
        return jsonify({"exists": True, "access_type": account["access_type"]})
    return jsonify({"exists": False})


# ─────────────────────────────────────────────
# Acceso — Logs
# ─────────────────────────────────────────────

@app.route("/api/access/logs")
def api_access_logs():
    page = request.args.get("page", 1, type=int)
    email_filter = request.args.get("email", None)
    result = get_access_logs(page=page, email_filter=email_filter)
    return jsonify(result)


# ─────────────────────────────────────────────
# Acceso — IP Blacklist
# ─────────────────────────────────────────────

@app.route("/api/access/blacklist", methods=["GET", "POST"])
def api_access_blacklist():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        ip = (data.get("ip") or "").strip()
        reason = data.get("reason", "")
        if not ip:
            return jsonify({"error": "IP required"}), 400
        entry = block_ip(ip, reason, session.get("email", ""))
        if entry:
            return jsonify({"entry": entry}), 201
        return jsonify({"error": "The IP is already blocked"}), 409
    return jsonify({"entries": get_blacklist()})


@app.route("/api/access/blacklist/<int:entry_id>", methods=["DELETE"])
def api_access_unblock(entry_id):
    if unblock_ip(entry_id):
        return jsonify({"deleted": True})
    return jsonify({"error": "Entry not found"}), 404


# ─────────────────────────────────────────────
# Acceso — Limited modules permissions
# ─────────────────────────────────────────────

@app.route("/api/access/limited-modules", methods=["GET", "POST"])
def api_access_limited_modules():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        modules = data.get("modules", [])
        valid = [m for m in modules if m in MODULES_META]
        set_limited_modules(valid)
        return jsonify({"modules": valid})
    return jsonify({"modules": get_limited_modules()})


# ═════════════════════════════════════════════════════════════
# Smart Semantycs — visión con vocabulario abierto (YOLOE + Gemini)
# ═════════════════════════════════════════════════════════════

_LVIS_SYSTEM_PROMPT = """Eres el cerebro de interpretación de un módulo de visión con vocabulario abierto.
Recibes el prompt del usuario y un vocabulario de clases (LVIS). Debes decidir:

1. ¿Es factible detectar lo pedido con el vocabulario disponible?
   Si pide algo imposible por imágenes de cámara (ej. "moneda triangular con 3
   agujeros") o requiere entrenamiento especializado -> "feasible": false.
2. ¿Solo detectar/alerta o también contar? Si pide "cuenta/cuantos",
   genera contadores (deduplicados por track); si solo pide "detecta/alerta",
   genera logs. Mantén contadores y logs SIEMPRE alineados con el mismo prompt.
3. Mapea SIEMPRE a nombres EXACTOS del vocabulario proporcionado (no inventes).
   El usuario puede escribir en plural, singular o en español ("personas", "gente",
   "person"). Traduce y normaliza SIEMPRE al nombre LVIS canónico en inglés
   (ej. "personas" -> "person", "coches" -> "car"). No repitas la palabra del usuario
   si no es el nombre LVIS exacto.
4. Genera contadores (máx 5) y logs con condiciones válidas.

Reglas de salida:
- Respuesta SOLO JSON válido, sin markdown ni texto extra.
- En "reason" (solo si feasible=false) usa lenguaje general, SIN nombres de
  tecnologías: di que no es posible detectarlo por el momento y que se requeriría
  un entrenamiento especializado.
- "classes": lista de nombres LVIS exactos (máx 15).

Patrón "algo que tenga objeto" (X que tiene Y), ej. "autos que tengan ruedas":
- Contador PRECISO de X con Y: condition {"detect": ["X"], "overlap": ["Y"],
  "min_overlap": 0.30}  -> cuenta solo cuando X se solapa con Y.
- Si solo se detecta X sin Y: no marcar (ese contador ya no dispara).
- Log de ALERTA/EXCEPCIÓN cuando se detecta solo el objeto Y sin el X (algo):
  condition {"detect": ["Y"], "missing": ["X"]}; label tipo "Se detectó el objeto
  Y pero no el X (algo)", priority "warning". Este es un caso de excepción a usar
  en escenarios de este tipo, no para todo.

Condiciones admitidas:
    {"detect": ["cls", ...]}  -> el objeto es de alguna de esas clases.
    {"detect": [...], "overlap": [...], "min_overlap": 0.30} -> el detectado se
       solapa >= 30% con un objeto de las clases "overlap" (ej. persona sobre
       bicicleta). Usar para contadores precisos de "X que tiene Y".
    {"detect": ["Y"], "missing": ["X"]} -> Y está presente pero X NO está en el
       frame. Para log de excepción "solo se detectó el objeto Y, no el X".

VOCABULARIO LVIS DEL PROMPT:
{vocab}"""


@app.route("/smart-semantycs")
def smart_semantycs_view():
    return render_template("smart_semantycs.html")


@app.route("/api/semantycs/sessions", methods=["POST"])
def api_semantycs_create_session():
    sid = create_semantycs_session()["id"]
    return jsonify({"session_id": sid}), 201


@app.route("/api/semantycs/sessions", methods=["GET"])
def api_semantycs_sessions():
    return jsonify({"sessions": get_semantycs_sessions()})


@app.route("/api/semantycs/sessions/<session_id>", methods=["GET", "DELETE"])
def api_semantycs_session(session_id):
    s = get_semantycs_session(session_id)
    if not s:
        return jsonify({"error": "Session not found"}), 404
    if request.method == "DELETE":
        SmartSemntycsManager.get().stop(session_id)
        delete_semantycs_session(session_id)
        return jsonify({"deleted": True})
    s["messages"] = get_semantycs_messages(session_id)
    try:
        s["skill"] = json.loads(s["skill"] or "{}")
    except Exception:
        s["skill"] = {}
    return jsonify({"session": s})


def _semantycs_source_path(s: dict) -> str | None:
    vpath = s.get("video_path") or ""
    vtype = s.get("video_type")
    if not vpath:
        return None
    if vtype == "stream" or vpath.startswith(("rtsp://", "rtmp://", "http://", "https://")):
        return vpath
    if vpath.startswith("static/"):
        return os.path.join(BASE_DIR, vpath.replace("/", os.sep))
    abspath = os.path.abspath(vpath)
    if os.path.exists(abspath):
        return abspath
    return os.path.join(BASE_DIR, vpath)


@app.route("/api/semantycs/sessions/<session_id>/video", methods=["POST"])
def api_semantycs_video(session_id):
    s = get_semantycs_session(session_id)
    if not s:
        return jsonify({"error": "Session not found"}), 404
    data = request.get_json(silent=True) or {}
    vtype = data.get("type") or s.get("video_type")
    if vtype not in ("video", "stream"):
        return jsonify({"error": "Invalid video type"}), 400
    path = (data.get("path") or "").strip()
    if not path:
        return jsonify({"error": "path required"}), 400
    state = "video" if s["state"] == "no_video" else s["state"]
    update_semantycs_session(session_id, video_path=path, video_type=vtype, state=state)
    return jsonify({"saved": True, "path": path, "type": vtype})


def _canonical_lvis(name: str) -> str | None:
    c = to_lvis(name)
    if c:
        return c
    cands = find_lvis_candidates(name, 1)
    if cands:
        return cands[0]
    return best_lvis_match(name)


def _normalize_skill(skill: dict) -> dict | None:
    """Valida y normaliza una skill de Gemini.

    Convierte clases/condiciones a nombres LVIS canónicos para que el pipeline
    haga comparación exacta. Devuelve None si la skill es inválida.
    """
    try:
        if skill.get("feasible") is not True:
            return skill

        classes_raw = skill.get("classes") or []
        if not isinstance(classes_raw, list) or not classes_raw or len(classes_raw) > 15:
            return None
        classes = []
        for c in classes_raw:
            if not isinstance(c, str):
                continue
            canon = _canonical_lvis(c)
            if not canon:
                return None
            if canon not in classes:
                classes.append(canon)
        if not classes:
            return None
        skill["classes"] = classes
        class_set = set(classes)

        def norm_cond(cond):
            if not isinstance(cond, dict):
                return None
            detect = cond.get("detect")
            if not isinstance(detect, list) or not detect:
                return None
            new_detect = []
            for d in detect:
                if not isinstance(d, str):
                    return None
                canon = _canonical_lvis(d)
                if not canon or canon not in class_set:
                    return None
                if canon not in new_detect:
                    new_detect.append(canon)
            new_cond = {"detect": new_detect}
            missing = cond.get("missing")
            if missing:
                if not isinstance(missing, list) or not missing:
                    return None
                new_missing = []
                for m in missing:
                    if not isinstance(m, str):
                        return None
                    canon = _canonical_lvis(m)
                    if not canon or canon not in class_set:
                        return None
                    if canon not in new_missing:
                        new_missing.append(canon)
                new_cond["missing"] = new_missing
            overlap = cond.get("overlap")
            if overlap:
                if not isinstance(overlap, list) or not overlap:
                    return None
                new_overlap = []
                for o in overlap:
                    if not isinstance(o, str):
                        return None
                    canon = _canonical_lvis(o)
                    if not canon or canon not in class_set:
                        return None
                    if canon not in new_overlap:
                        new_overlap.append(canon)
                new_cond["overlap"] = new_overlap
                try:
                    new_cond["min_overlap"] = float(cond.get("min_overlap", 0.3))
                except Exception:
                    new_cond["min_overlap"] = 0.3
            return new_cond

        counters = []
        seen = set()
        for c in (skill.get("counters") or [])[:5]:
            if not isinstance(c, dict) or not c.get("id"):
                return None
            cid = str(c["id"])
            if cid in seen:
                return None
            seen.add(cid)
            cond = norm_cond(c.get("condition"))
            if cond is None:
                return None
            counters.append({
                "id": cid,
                "label": c.get("label") or cid,
                "color": c.get("color") or "#22C55E",
                "condition": cond,
            })
        skill["counters"] = counters

        logs = []
        seen = set()
        for l in skill.get("logs") or []:
            if not isinstance(l, dict) or not l.get("id"):
                return None
            lid = str(l["id"])
            if lid in seen:
                return None
            seen.add(lid)
            cond = norm_cond(l.get("condition"))
            if cond is None:
                return None
            priority = l.get("priority", "info")
            if priority not in ("info", "warning", "critical"):
                priority = "info"
            logs.append({
                "id": lid,
                "label": l.get("label") or lid,
                "event": l.get("event") or "",
                "priority": priority,
                "condition": cond,
            })
        skill["logs"] = logs
        return skill
    except Exception:
        return None


def _parse_skill_json(reply: str) -> dict | None:
    if not reply:
        return None
    raw = reply.strip()
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        chunk = raw[start:end + 1]
    else:
        chunk = raw
    try:
        d = json.loads(chunk)
        return d if isinstance(d, dict) else None
    except Exception:
        return None


@app.route("/api/semantycs/sessions/<session_id>/interpret", methods=["POST"])
def api_semantycs_interpret(session_id):
    s = get_semantycs_session(session_id)
    if not s:
        return jsonify({"error": "Session not found"}), 404
    if not (s.get("video_path") or ""):
        return jsonify({"error": "Primero vincula un video o stream."}), 400

    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "Prompt cannot be empty"}), 400

    client_ip = request.remote_addr or "unknown"
    now = time.time()
    last = _chat_rate_limit.get(client_ip, 0)
    elapsed = now - last
    if elapsed < 2 and last > 0:
        wait_for = int(2 - elapsed) + 1
        return jsonify({"error": f"Wait {wait_for}s before sending.", "retry_after": wait_for}), 429
    _chat_rate_limit[client_ip] = now

    settings = get_settings()
    api_key = settings.get("gemini_api_key", "").strip()
    if not api_key:
        return jsonify({"error": "No Gemini API Key configured. Go to Settings > General."}), 400

    add_semantycs_message(session_id, "user", prompt)
    update_semantycs_session(session_id, prompt=prompt, state="video")

    system_prompt = _LVIS_SYSTEM_PROMPT.replace("{vocab}", "\n".join(LVIS_NAMES))
    reply = _call_gemini(api_key, system_prompt, prompt)

    skill = _parse_skill_json(reply)
    normalized = _normalize_skill(skill) if skill is not None else None
    if normalized is None:
        add_semantycs_message(
            session_id, "model",
            "No se pudo interpretar tu solicitud. Intenta reformularla, por "
            "ejemplo: 'detecta y cuenta personas en bicicleta'.",
            kind="error",
        )
        return jsonify({"parsed": False})

    update_semantycs_session(session_id, skill=json.dumps(normalized))

    if normalized.get("feasible"):
        summary = normalized.get("summary") or "Detector configurado."
        add_semantycs_message(session_id, "model", f"Listo. {summary}", kind="skill")
        update_semantycs_session(session_id, state="prompted")
    else:
        reason = normalized.get("reason") or (
            "No es posible detectar eso por el momento; requeriría un "
            "entrenamiento especializado."
        )
        add_semantycs_message(session_id, "model", reason, kind="error")
        update_semantycs_session(session_id, state="video")
    return jsonify({"parsed": True})


@app.route("/api/semantycs/sessions/<session_id>/start", methods=["POST"])
def api_semantycs_start(session_id):
    s = get_semantycs_session(session_id)
    if not s:
        return jsonify({"error": "Session not found"}), 404
    if s["state"] not in ("prompted", "stopped"):
        return jsonify({"error": "La sesión no está lista para iniciar."}), 400
    if not (s.get("video_path") or ""):
        return jsonify({"error": "No hay video vinculado."}), 400
    try:
        skill = json.loads(s["skill"] or "{}")
    except Exception:
        skill = {}
    if not skill or skill.get("feasible") is not True:
        return jsonify({"error": "La skill no es válida."}), 400
    classes = skill.get("classes") or []
    if not classes:
        return jsonify({"error": "La skill no tiene clases."}), 400

    src_path = _semantycs_source_path(s)
    if not src_path:
        return jsonify({"error": "No se pudo resolver la fuente."}), 400

    settings = get_settings()
    try:
        conf = float(settings.get("smart_semantycs_conf", "0.25"))
    except (TypeError, ValueError):
        conf = 0.25
    # Sin throttle artificial: Smart Semantycs no duerme entre frames.
    fps = 0.0

    SmartSemntycsManager.get().start(
        session_id, src_path, s["video_type"], classes, skill, conf, fps,
    )
    update_semantycs_session(session_id, state="running")
    return jsonify({"started": session_id})


@app.route("/api/semantycs/sessions/<session_id>/pause", methods=["POST"])
def api_semantycs_pause(session_id):
    SmartSemntycsManager.get().pause(session_id)
    return jsonify({"paused": session_id})


@app.route("/api/semantycs/sessions/<session_id>/resume", methods=["POST"])
def api_semantycs_resume(session_id):
    SmartSemntycsManager.get().resume(session_id)
    return jsonify({"resumed": session_id})


@app.route("/api/semantycs/sessions/<session_id>/reset", methods=["POST"])
def api_semantycs_reset(session_id):
    s = get_semantycs_session(session_id)
    if not s:
        return jsonify({"error": "Session not found"}), 404
    SmartSemntycsManager.get().reset(session_id)
    return jsonify({"reset": session_id})


@app.route("/api/semantycs/sessions/<session_id>/stop", methods=["POST"])
def api_semantycs_stop(session_id):
    s = get_semantycs_session(session_id)
    if not s:
        return jsonify({"error": "Session not found"}), 404
    SmartSemntycsManager.get().stop(session_id)
    update_semantycs_session(session_id, state="stopped")
    return jsonify({"stopped": session_id})


@app.route("/api/semantycs/sessions/<session_id>/stream")
def api_semantycs_stream(session_id):
    mgr = SmartSemntycsManager.get()

    def generate():
        while mgr.is_running(session_id):
            jpeg = mgr.get_frame_jpeg(session_id)
            if jpeg is None:
                time.sleep(0.033)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )
            time.sleep(1 / 30)

    return Response(
        stream_with_context(generate()),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/semantycs/sessions/<session_id>/state")
def api_semantycs_state(session_id):
    s = get_semantycs_session(session_id)
    if not s:
        return jsonify({"error": "Session not found"}), 404
    mgr = SmartSemntycsManager.get()
    running = mgr.is_running(session_id)
    stats = mgr.get_stats(session_id)
    if stats:
        counters = stats["counters"]
    else:
        counters = list_semantycs_counters(session_id)
    try:
        skill = json.loads(s["skill"] or "{}")
    except Exception:
        skill = {}
    return jsonify({
        "state": "running" if running else s["state"],
        "running": running,
        "paused": bool(stats and stats["paused"]),
        "video_path": s["video_path"],
        "video_type": s["video_type"],
        "prompt": s["prompt"],
        "skill": skill,
        "counters": counters,
        "logs": list_semantycs_logs(session_id, 200),
    })


@app.route("/api/semantycs/sessions/<session_id>/logs/<int:log_row_id>")
def api_semantycs_log_evidence(session_id, log_row_id):
    logs = list_semantycs_logs(session_id, 1000)
    log = next((l for l in logs if l["id"] == log_row_id), None)
    if not log:
        return jsonify({"error": "Log not found"}), 404
    return jsonify({"log": log})


@app.route("/api/semantycs/model/status")
def api_semantycs_model_status():
    settings = get_settings()
    device = get_device()
    use_gpu = device != "cpu"
    auto = settings.get("smart_semantycs_model_auto", "1") == "1"

    SIZES = {
        "nano":   ("yoloe-26n-seg.pt", "smart_semantycs_model_nano"),
        "medium": ("yoloe-26m-seg.pt", "smart_semantycs_model_medium"),
        "xl":     ("yoloe-26x-seg.pt", "smart_semantycs_model_xl"),
    }
    on_disk = {}
    for key, (fname, _skey) in SIZES.items():
        on_disk[key] = os.path.exists(os.path.join(MODELS_FOLDER, fname))

    # Selección automática: CPU → nano, GPU → medium (fallback xl).
    auto_rank = ["nano"] if not use_gpu else ["medium", "xl"]
    auto_pick = None
    for key in auto_rank:
        fname, skey = SIZES[key]
        if settings.get(skey) or on_disk[key]:
            auto_pick = key
            break

    if auto:
        pick = auto_pick
    else:
        fixed = settings.get("smart_semantycs_model_fixed", "")
        pick = fixed if fixed in SIZES else None

    if pick:
        fname, skey = SIZES[pick]
        model_path = settings.get(skey, "")
        if not model_path and on_disk[pick]:
            model_path = f"static/uploads/models/{fname}"
        active = f"{pick.capitalize()} ({fname})" if model_path else None
    else:
        model_path = ""
        active = None

    paths = {}
    for key, (fname, skey) in SIZES.items():
        paths[key + "_path"] = settings.get(skey, "") or (
            f"static/uploads/models/{fname}" if on_disk[key] else ""
        )

    return jsonify({
        "device": device,
        "use_gpu": use_gpu,
        "auto": auto,
        "active": active,
        "model_path": model_path,
        "nano_path": paths["nano_path"],
        "medium_path": paths["medium_path"],
        "xl_path": paths["xl_path"],
    })
