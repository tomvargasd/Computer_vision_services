from flask import Flask, render_template, jsonify, request, Response, stream_with_context
from flask_cors import CORS
from flask_compress import Compress
import os
import uuid
import time
import json
import random
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename

from src.config import (
    BASE_DIR, UPLOAD_FOLDER, VIDEOS_FOLDER, MODELS_FOLDER, CAPTURES_FOLDER,
    ALLOWED_IMG, ALLOWED_VIDEO, ALLOWED_MODEL, APP_VERSION,
)
from src.database import (
    init_db, get_settings, set_setting,
    get_modules_state, db_toggle_module, db_toggle_function,
    get_sources, get_source, add_source, update_source, delete_source,
    MODULES_META,
    save_module_counters, load_module_counters, reset_module_counters,
    insert_module_event, get_module_events, get_module_analytics,
    save_source_config, get_source_config, get_source_config_value,
    delete_source_config,
)

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


def _get_fps_limit(module_id, src_type, settings):
    key = f"{module_id}_fps_limit_{src_type}"
    return float(settings.get(key, "0.02" if src_type == "video" else "0.0"))


app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"),
            static_folder=os.path.join(BASE_DIR, "static"))
CORS(app)
Compress(app)

app.secret_key = os.urandom(32)
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
    return {"settings": get_settings(), "modules": get_modules_state()}


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
    if module_id not in MODULES_META:
        return jsonify({"error": "Módulo no encontrado"}), 404
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
    if module_id not in MODULES_META:
        return jsonify({"error": "Módulo no encontrado"}), 404
    if func_id not in MODULES_META[module_id]["functions"]:
        return jsonify({"error": "Función no encontrada"}), 404
    fmeta = MODULES_META[module_id]["functions"][func_id]
    if fmeta.get("locked"):
        return jsonify({"error": "Esta función no se puede desactivar", "locked": True}), 403
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
    modules = get_modules_state()
    return render_template("settings.html", modules=modules, settings=get_settings())


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
        return jsonify({"error": "No se envió archivo"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Nombre de archivo vacío"}), 400

    target_map = {
        "logo": (UPLOAD_FOLDER, ALLOWED_IMG),
        "video": (VIDEOS_FOLDER, ALLOWED_VIDEO),
        "model": (MODELS_FOLDER, ALLOWED_MODEL),
    }
    if field not in target_map:
        return jsonify({"error": "Campo inválido"}), 400
    dest_dir, allowed_set = target_map[field]
    ext = f.filename.rsplit(".", 1)[1].lower() if "." in f.filename else ""
    if ext not in allowed_set:
        return jsonify({"error": f"Tipo de archivo no permitido ({ext})"}), 400
    fname = secure_filename(f.filename)
    dest = os.path.join(dest_dir, fname)
    f.save(dest)
    if field == "logo":
        set_setting("logo", fname)
        return jsonify({"path": fname})
    rel = os.path.join("static", "uploads", field + "s" if field != "logo" else "uploads", fname)
    return jsonify({"path": rel})


@app.route("/api/settings/logo", methods=["POST"])
def api_upload_logo():
    if "logo" not in request.files:
        return jsonify({"error": "No se envió archivo"}), 400
    f = request.files["logo"]
    if not f.filename:
        return jsonify({"error": "Nombre vacío"}), 400
    ext = f.filename.rsplit(".", 1)[1].lower() if "." in f.filename else ""
    if ext not in ALLOWED_IMG:
        return jsonify({"error": "Formato no permitido"}), 400
    fname = secure_filename(f.filename)
    f.save(os.path.join(UPLOAD_FOLDER, fname))
    set_setting("logo", fname)
    return jsonify({"logo": fname})


@app.route("/api/<module_id>/upload-model", methods=["POST"])
def api_upload_module_model(module_id):
    if module_id not in MODULES_META:
        return jsonify({"error": "Módulo no encontrado"}), 404
    if "model" not in request.files:
        return jsonify({"error": "No se envió archivo"}), 400
    f = request.files["model"]
    if not f.filename.endswith(".pt"):
        return jsonify({"error": "Solo archivos .pt"}), 400
    fname = secure_filename(f.filename)
    f.save(os.path.join(MODELS_FOLDER, fname))
    rel = os.path.join("static", "uploads", "models", fname)
    set_setting(f"{module_id}_model", rel)
    return jsonify({"model": fname})


@app.route("/api/<module_id>/model", methods=["DELETE"])
def api_remove_module_model(module_id):
    if module_id not in MODULES_META:
        return jsonify({"error": "Módulo no encontrado"}), 404
    set_setting(f"{module_id}_model", "")
    return jsonify({"removed": True})


@app.route("/api/<module_id>/settings/conf", methods=["POST"])
def api_set_conf(module_id):
    if module_id not in MODULES_META:
        return jsonify({"error": "Módulo no encontrado"}), 404
    data = request.get_json(silent=True) or {}
    val = str(float(data.get("conf", 0.35)))
    set_setting(f"{module_id}_conf", val)
    return jsonify({"conf": val})


@app.route("/api/<module_id>/settings/half", methods=["POST"])
def api_set_half(module_id):
    if module_id not in MODULES_META:
        return jsonify({"error": "Módulo no encontrado"}), 404
    data = request.get_json(silent=True) or {}
    val = "1" if data.get("half") else "0"
    set_setting(f"{module_id}_half", val)
    return jsonify({"half": val})


@app.route("/api/<module_id>/settings/fps", methods=["POST"])
def api_set_fps(module_id):
    if module_id not in MODULES_META:
        return jsonify({"error": "Módulo no encontrado"}), 404
    data = request.get_json(silent=True) or {}
    for src_type in ("video", "stream"):
        if src_type in data:
            val = str(float(data[src_type]))
            set_setting(f"{module_id}_fps_limit_{src_type}", val)
    return jsonify({"saved": True})


@app.route("/api/pallets/settings/classes", methods=["POST"])
def pallets_set_classes():
    data = request.get_json(silent=True) or {}
    raw = data.get("classes", "0,1,2,3")
    ids = [int(c.strip()) for c in raw.split(",") if c.strip()]
    if not ids or any(i < 0 or i > 255 for i in ids):
        return jsonify({"error": "Lista de clases inválida"}), 400
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
        return jsonify({"error": "No se envió archivo"}), 400
    f = request.files["file"]
    if not f.filename.endswith(".pt"):
        return jsonify({"error": "Solo archivos .pt"}), 400
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
            return jsonify({"error": "Pipeline no activo"}), 404
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
        return jsonify({"error": "Fuente no encontrada"}), 404
    s = get_settings()
    fps_limit = _get_fps_limit("personas", src["type"], s)
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
        return jsonify({"error": "Fuente no encontrada"}), 404
    s = get_settings()
    fps_limit = _get_fps_limit("armas", src["type"], s)
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
        return jsonify({"error": "Fuente no encontrada"}), 404
    s = get_settings()
    fps_limit = _get_fps_limit("acciones", src["type"], s)
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
        return jsonify({"error": "Pipeline no activo"}), 404
    return jsonify(data)


@app.route("/api/acciones/sources/<int:source_id>/teach/save", methods=["POST"])
def acciones_teach_save(source_id):
    body = request.get_json()
    if not body:
        return jsonify({"error": "JSON requerido"}), 400
    tid = body.get("person_id")
    action = body.get("action")
    if tid is None or action not in ("violencia", "robo", "sospechoso", "celular", "caida"):
        return jsonify({"error": "person_id y action requeridos (violencia|robo|sospechoso|celular|caida)"}), 400
    mgr = AccionesManager.get()
    pipeline = mgr.pipelines.get(source_id) if hasattr(mgr, "pipelines") else None
    if pipeline is None:
        return jsonify({"error": "Pipeline no activo"}), 404
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
        return jsonify({"error": "Fuente no encontrada"}), 404
    s = get_settings()
    fps_limit = _get_fps_limit("troncos", src["type"], s)
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


# ─────────────────────────────────────────────
# API Pallets
# ─────────────────────────────────────────────

_register_standard_routes(app, "pallets", PalletsManager)

@app.route("/api/pallets/sources/<int:source_id>/start", methods=["POST"])
def pallets_start(source_id):
    sources = get_sources("pallets")
    src = next((s for s in sources if s["id"] == source_id), None)
    if not src:
        return jsonify({"error": "Fuente no encontrada"}), 404
    s = get_settings()
    classes_str = s.get("pallets_classes", "0,1,2,3")
    classes = [int(c.strip()) for c in classes_str.split(",") if c.strip()]
    fps_limit = _get_fps_limit("pallets", src["type"], s)
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
        return jsonify({"error": "Fuente no encontrada"}), 404
    s = get_settings()
    fps_limit = _get_fps_limit("cajas", src["type"], s)
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
        return jsonify({"error": "Fuente no encontrada"}), 404
    s = get_settings()
    fps_limit = _get_fps_limit("reglamento", src["type"], s)
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
        return jsonify({"error": "Fuente no encontrada"}), 404
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
    fps_limit = _get_fps_limit("carga_descarga", src["type"], s)
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
        return jsonify({"error": "Modo inválido"}), 400
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
        return jsonify({"error": "Modelo no encontrado"}), 404
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
        return jsonify({"error": "Fuente no encontrada"}), 404
    s = get_settings()
    fps_limit = _get_fps_limit("epp", src["type"], s)
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
        return jsonify({"error": "Fuente no encontrada"}), 404
    s = get_settings()
    fps_limit = _get_fps_limit("smoke", src["type"], s)
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
        return jsonify({"error": "Fuente no encontrada"}), 404
    s = get_settings()
    fps_limit = _get_fps_limit("vehiculos", src["type"], s)
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
        return jsonify({"error": "Modo inválido"}), 400
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
        return jsonify({"error": "No se envió archivo"}), 400
    f = request.files["model"]
    if not f.filename.endswith(".pt"):
        return jsonify({"error": "Solo archivos .pt"}), 400
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
        return jsonify({"error": "Fuente no encontrada"}), 404
    s = get_settings()
    fps_limit = _get_fps_limit("tanques_gas", src["type"], s)
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
        return jsonify({"error": "Modo inválido"}), 400
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
        return jsonify({"error": "Se requieren exactamente 4 puntos"}), 400
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
        return jsonify({"error": "Acción requerida"}), 400
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
        return jsonify({"error": "No se envió archivo"}), 400
    f = request.files["model"]
    if not f.filename.endswith(".pt"):
        return jsonify({"error": "Solo archivos .pt"}), 400
    fname = secure_filename(f.filename)
    f.save(os.path.join(MODELS_FOLDER, fname))
    rel = os.path.join("static", "uploads", "models", fname)
    set_setting("tanques_gas_pose_model", rel)
    return jsonify({"model": fname})


@app.route("/api/tanques_gas/upload-smoke-model", methods=["POST"])
def tanques_gas_upload_smoke_model():
    if "model" not in request.files:
        return jsonify({"error": "No se envió archivo"}), 400
    f = request.files["model"]
    if not f.filename.endswith(".pt"):
        return jsonify({"error": "Solo archivos .pt"}), 400
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
    data = request.get_json(silent=True) or {}
    module_id = data.get("module_id")
    name = data.get("name", "").strip()
    src_type = data.get("type", "stream")
    path = data.get("path", "").strip()
    if module_id not in MODULES_META:
        return jsonify({"error": "Módulo inválido"}), 400
    if not name or not path:
        return jsonify({"error": "Nombre y ruta requeridos"}), 400
    if src_type not in ("video", "stream"):
        return jsonify({"error": "Tipo inválido"}), 400
    path = _normalize_path(path, src_type)
    src = add_source(module_id, name, src_type, path)
    return jsonify(src), 201


@app.route("/api/sources/<int:source_id>", methods=["PUT", "DELETE"])
def api_source_crud(source_id):
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
        return jsonify({"error": "Fuente no encontrada"}), 404
    return jsonify(src)


@app.route("/api/sources/upload-video", methods=["POST"])
def api_upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No se envió archivo"}), 400
    f = request.files["video"]
    if not f.filename:
        return jsonify({"error": "Nombre vacío"}), 400
    ext = f.filename.rsplit(".", 1)[1].lower() if "." in f.filename else ""
    if ext not in ALLOWED_VIDEO:
        return jsonify({"error": "Formato de video no permitido"}), 400
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
        return jsonify({"error": "Módulo no encontrado"}), 404
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
        return jsonify({"error": "Módulo no encontrado"}), 404
    days = request.args.get("days", 7, type=int)
    source_id = request.args.get("source_id", None, type=int)
    event_type = request.args.get("event_type", None)
    limit = request.args.get("limit", 100, type=int)
    events = get_module_events(module_id, source_id, event_type, days, limit)
    return jsonify({"events": events})


@app.route("/api/analytics/<module_id>/timeline")
def api_analytics_timeline(module_id):
    if module_id not in MODULES_META:
        return jsonify({"error": "Módulo no encontrado"}), 404
    days = request.args.get("days", 7, type=int)
    source_id = request.args.get("source_id", None, type=int)
    data = get_module_analytics(module_id, source_id, days)
    return jsonify({"data": data["daily"]})


# ─────────────────────────────────────────────
# API Tanques de Gas — Dashboard Analítico (v3.0)
# ─────────────────────────────────────────────

import random
from datetime import datetime, timedelta

@app.route("/api/analytics/tanques_gas/comprehensive")
def api_tanques_gas_comprehensive():
    sources = get_sources("tanques_gas")
    if not sources:
        return jsonify({"kpis": {}, "sources": [], "dailyTimeline": [], "eventTypeDistribution": {}, "actionDistribution": {}, "recentEvents": [], "empty": True})

    source_ids = [s["id"] for s in sources]
    source_map  = {s["id"]: s for s in sources}

    counters_all = {}
    for sid in source_ids:
        counters_all[sid] = load_module_counters("tanques_gas", sid)

    today_str = datetime.now().strftime("%Y-%m-%d")

    # ── KPIs ──
    total_entradas = sum(counters_all[sid].get("entrada", 0) if isinstance(counters_all[sid], dict) else 0 for sid in source_ids)
    total_salidas  = sum(counters_all[sid].get("salida", 0) if isinstance(counters_all[sid], dict) else 0 for sid in source_ids)

    analytics = get_module_analytics("tanques_gas", days=365)
    daily_raw = analytics["daily"]

    today_events = sum(d["count"] for d in daily_raw if d["day"] == today_str)

    total_eventos = sum(d["count"] for d in daily_raw)

    total_acciones = 0
    entries_by_source = {}
    exits_by_source = {}
    acciones_by_source = {}
    smoke_sources = []

    for sid in source_ids:
        c = counters_all.get(sid, {})
        if not isinstance(c, dict):
            c = {}
        ent = c.get("entrada", 0) or 0
        sal = c.get("salida", 0) or 0
        entries_by_source[sid] = ent
        exits_by_source[sid] = sal

        act_count = c.get("action_count", {})
        if isinstance(act_count, str):
            try:
                act_count = json.loads(act_count)
            except Exception:
                act_count = {}
        if isinstance(act_count, dict):
            acc = sum(v for v in act_count.values() if isinstance(v, (int, float)))
        else:
            acc = 0
        acciones_by_source[sid] = acc
        total_acciones += acc

        sd = c.get("smoke_detected", 0)
        if sd:
            smoke_sources.append(sid)


    ocupacion_neta = total_entradas - total_salidas
    ratio_es = round(total_entradas / total_salidas, 2) if total_salidas > 0 else total_entradas

    # Daily timeline — aggregate by date
    daily_agg = {}
    for d in daily_raw:
        day = d["day"]
        et = d["event_type"]
        cnt = d["count"]
        if day not in daily_agg:
            daily_agg[day] = {"day": day, "entradas": 0, "salidas": 0, "acciones": 0, "smoke": 0}
        if et in ("entry", "entrada"):
            daily_agg[day]["entradas"] += cnt
        elif et in ("exit", "salida"):
            daily_agg[day]["salidas"] += cnt
        elif et == "smoke_detected":
            daily_agg[day]["smoke"] += cnt
        else:
            daily_agg[day]["acciones"] += cnt

    daily_timeline = sorted(daily_agg.values(), key=lambda x: x["day"])

    # Event type distribution
    event_type_dist = {"entry": 0, "exit": 0, "smoke_detected": 0, "action": 0}
    for d in daily_raw:
        et = d["event_type"]
        cnt = d["count"]
        if et in ("entry", "entrada"):
            event_type_dist["entry"] += cnt
        elif et in ("exit", "salida"):
            event_type_dist["exit"] += cnt
        elif et == "smoke_detected":
            event_type_dist["smoke_detected"] += cnt
        else:
            event_type_dist["action"] += cnt

    # Action distribution
    action_dist = {}
    for sid in source_ids:
        c = counters_all.get(sid, {})
        if isinstance(c, dict):
            ac = c.get("action_count", {})
            if isinstance(ac, str):
                try:
                    ac = json.loads(ac)
                except Exception:
                    ac = {}
            if isinstance(ac, dict):
                for act_name, act_cnt in ac.items():
                    action_dist[act_name] = action_dist.get(act_name, 0) + (act_cnt if isinstance(act_cnt, (int, float)) else 0)

    # Per-source breakdown
    total_all = total_entradas + total_salidas + total_acciones
    source_list = []
    for sid in source_ids:
        c = counters_all.get(sid, {})
        if not isinstance(c, dict):
            c = {}
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

    # Historical KPIs from daily data
    daily_totals = [d["entradas"] + d["salidas"] + d["acciones"] + d["smoke"] for d in daily_timeline]
    promedio_diario = round(sum(daily_totals) / len(daily_totals), 1) if daily_totals else 0
    pico_diario = max(daily_totals) if daily_totals else 0
    pico_dia = ""
    for d in daily_timeline:
        if d["entradas"] + d["salidas"] + d["acciones"] + d["smoke"] == pico_diario:
            pico_dia = d["day"]
            break

    fuente_top_id = max(entries_by_source, key=entries_by_source.get) if entries_by_source else None
    fuente_top = None
    if fuente_top_id:
        fuente_top = {
            "id": fuente_top_id,
            "name": source_map[fuente_top_id]["name"],
            "entradas": entries_by_source[fuente_top_id],
        }

    total_smoke_events = event_type_dist.get("smoke_detected", 0)

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
        "totalEventos": total_eventos if total_eventos > 0 else total_entradas + total_salidas + total_acciones + total_smoke_events,
        "totalAcciones": total_acciones,
        "totalSmoke": total_smoke_events,
        "fuenteTop": fuente_top,
    }

    # Recent events
    raw_events = get_module_events("tanques_gas", days=365, limit=200)
    recent_events = []
    for e in raw_events:
        recent_events.append({
            "id": e["id"],
            "source_id": e["source_id"],
            "source_name": source_map.get(e["source_id"], {}).get("name", f"Fuente {e['source_id']}"),
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
        "totalEvents": len(raw_events),
        "empty": False,
    })


@app.route("/api/analytics/tanques_gas/seed", methods=["POST"])
def api_tanques_gas_seed():
    sources = get_sources("tanques_gas")
    if not sources:
        return jsonify({"error": "No hay fuentes registradas. Registre al menos una fuente primero."}), 400

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
        "levantar": "Levantando tanque",
        "transportar": "Transportando tanque",
        "inspeccionar": "Inspeccionando tanque",
        "descargar": "Descargando tanque",
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
                all_events.append((sid, "entry", "Tanque detectado", "", ts))

            for _ in range(daily_exits):
                h = random.choice(hours_pool)
                m = random.randint(0, 59)
                s = random.randint(0, 59)
                ts = f"{day_str} {h:02d}:{m:02d}:{s:02d}"
                all_events.append((sid, "exit", "Tanque en salida", "", ts))

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
                all_events.append((sid, "smoke_detected", "Humo o fuego detectado", "", ts))

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
        "message": f"Datos generados: {total_gen} eventos en {n_sources} fuentes (2026-07-01 → 2026-07-29)",
        "total_events": total_gen,
        "sources_used": n_sources,
    })


@app.route("/api/analytics/tanques_gas/clear", methods=["POST"])
def api_tanques_gas_clear():
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

    return jsonify({"success": True, "message": "Datos eliminados correctamente"})


# ─────────────────────────────────────────────
# Evidence API (v3.0)
# ─────────────────────────────────────────────

@app.route("/api/evidences/<module_id>/<int:event_id>")
def api_evidence(module_id, event_id):
    if module_id not in MODULES_META:
        return jsonify({"error": "Módulo no encontrado"}), 404
    events = get_module_events(module_id, limit=100)
    event = next((e for e in events if e["id"] == event_id), None)
    if not event:
        return jsonify({"error": "Evento no encontrado"}), 404
    capture_url = event.get("capture_path")
    extra_paths = event.get("extra_paths", [])
    return jsonify({
        "event": event,
        "capture_url": capture_url,
        "extra_urls": extra_paths,
    })
