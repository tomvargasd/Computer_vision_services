from flask import Flask, render_template, jsonify, request, Response, stream_with_context
from flask_cors import CORS
import os
import uuid
import time
import json
from werkzeug.utils import secure_filename

from src.config import (
    BASE_DIR, UPLOAD_FOLDER, VIDEOS_FOLDER, MODELS_FOLDER, CAPTURES_FOLDER,
    ALLOWED_IMG, ALLOWED_VIDEO, ALLOWED_MODEL, APP_VERSION,
)
from src.database import (
    init_db, get_settings, set_setting,
    get_modules_state, db_toggle_module, db_toggle_function,
    get_sources, add_source, update_source, delete_source,
    MODULES_META,
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
# API Personas
# ─────────────────────────────────────────────

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


@app.route("/api/personas/sources/<int:source_id>/stop", methods=["POST"])
def personas_stop(source_id):
    PersonasManager.get().stop(source_id)
    return jsonify({"stopped": source_id})


@app.route("/api/personas/sources/<int:source_id>/stats")
def personas_stats(source_id):
    stats = PersonasManager.get().get_stats(source_id)
    if stats is None:
        return jsonify({"error": "Pipeline no activo"}), 404
    return jsonify(stats)


@app.route("/api/personas/sources/<int:source_id>/reset", methods=["POST"])
def personas_reset(source_id):
    PersonasManager.get().reset(source_id)
    return jsonify({"reset": source_id})


@app.route("/api/personas/sources/<int:source_id>/line-y", methods=["POST"])
def personas_line_y(source_id):
    data = request.get_json(silent=True) or {}
    pct = max(0, min(100, int(data.get("pct", 85))))
    set_setting("personas_line_y", str(pct))
    PersonasManager.get().set_line_y(source_id, pct)
    return jsonify({"line_y_pct": pct})


@app.route("/api/personas/stream/<int:source_id>")
def personas_stream(source_id):
    return _stream_response("personas", source_id)


# ─────────────────────────────────────────────
# API Armas
# ─────────────────────────────────────────────

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


@app.route("/api/armas/sources/<int:source_id>/stop", methods=["POST"])
def armas_stop(source_id):
    ArmasManager.get().stop(source_id)
    return jsonify({"stopped": source_id})


@app.route("/api/armas/sources/<int:source_id>/stats")
def armas_stats(source_id):
    stats = ArmasManager.get().get_stats(source_id)
    if stats is None:
        return jsonify({"error": "Pipeline no activo"}), 404
    return jsonify(stats)


@app.route("/api/armas/sources/<int:source_id>/reset", methods=["POST"])
def armas_reset(source_id):
    ArmasManager.get().reset(source_id)
    return jsonify({"reset": source_id})


@app.route("/api/armas/stream/<int:source_id>")
def armas_stream(source_id):
    return _stream_response("armas", source_id)


# ─────────────────────────────────────────────
# API Acciones
# ─────────────────────────────────────────────

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


@app.route("/api/acciones/sources/<int:source_id>/stop", methods=["POST"])
def acciones_stop(source_id):
    AccionesManager.get().stop(source_id)
    return jsonify({"stopped": source_id})


@app.route("/api/acciones/sources/<int:source_id>/stats")
def acciones_stats(source_id):
    stats = AccionesManager.get().get_stats(source_id)
    if stats is None:
        return jsonify({"error": "Pipeline no activo"}), 404
    return jsonify(stats)


@app.route("/api/acciones/sources/<int:source_id>/reset", methods=["POST"])
def acciones_reset(source_id):
    AccionesManager.get().reset(source_id)
    return jsonify({"reset": source_id})


@app.route("/api/acciones/stream/<int:source_id>")
def acciones_stream(source_id):
    return _stream_response("acciones", source_id)


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


@app.route("/api/troncos/sources/<int:source_id>/stop", methods=["POST"])
def troncos_stop(source_id):
    TroncosManager.get().stop(source_id)
    return jsonify({"stopped": source_id})


@app.route("/api/troncos/sources/<int:source_id>/stats")
def troncos_stats(source_id):
    stats = TroncosManager.get().get_stats(source_id)
    if stats is None:
        return jsonify({"error": "Pipeline no activo"}), 404
    return jsonify(stats)


@app.route("/api/troncos/sources/<int:source_id>/reset", methods=["POST"])
def troncos_reset(source_id):
    TroncosManager.get().reset(source_id)
    return jsonify({"reset": source_id})


@app.route("/api/troncos/sources/<int:source_id>/line-x", methods=["POST"])
def troncos_line_x(source_id):
    data = request.get_json(silent=True) or {}
    pct = max(0, min(100, int(data.get("pct", 50))))
    set_setting("troncos_line_x", str(pct))
    TroncosManager.get().set_line_x(source_id, pct)
    return jsonify({"line_x_pct": pct})


@app.route("/api/troncos/stream/<int:source_id>")
def troncos_stream(source_id):
    return _stream_response("troncos", source_id)


# ─────────────────────────────────────────────
# API Pallets
# ─────────────────────────────────────────────

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


@app.route("/api/pallets/sources/<int:source_id>/stop", methods=["POST"])
def pallets_stop(source_id):
    PalletsManager.get().stop(source_id)
    return jsonify({"stopped": source_id})


@app.route("/api/pallets/sources/<int:source_id>/stats")
def pallets_stats(source_id):
    stats = PalletsManager.get().get_stats(source_id)
    if stats is None:
        return jsonify({"error": "Pipeline no activo"}), 404
    return jsonify(stats)


@app.route("/api/pallets/sources/<int:source_id>/reset", methods=["POST"])
def pallets_reset(source_id):
    PalletsManager.get().reset(source_id)
    return jsonify({"reset": source_id})


@app.route("/api/pallets/sources/<int:source_id>/area", methods=["POST"])
def pallets_area(source_id):
    data = request.get_json(silent=True) or {}
    x1 = int(data.get("x1", 25)); y1 = int(data.get("y1", 25))
    x2 = int(data.get("x2", 75)); y2 = int(data.get("y2", 75))
    set_setting("pallets_area_x1", str(x1)); set_setting("pallets_area_y1", str(y1))
    set_setting("pallets_area_x2", str(x2)); set_setting("pallets_area_y2", str(y2))
    PalletsManager.get().set_area(source_id, x1, y1, x2, y2)
    return jsonify({"x1": x1, "y1": y1, "x2": x2, "y2": y2})


@app.route("/api/pallets/stream/<int:source_id>")
def pallets_stream(source_id):
    return _stream_response("pallets", source_id)


# ─────────────────────────────────────────────
# API Cajas
# ─────────────────────────────────────────────

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


@app.route("/api/cajas/sources/<int:source_id>/stop", methods=["POST"])
def cajas_stop(source_id):
    CajasManager.get().stop(source_id)
    return jsonify({"stopped": source_id})


@app.route("/api/cajas/sources/<int:source_id>/stats")
def cajas_stats(source_id):
    stats = CajasManager.get().get_stats(source_id)
    if stats is None:
        return jsonify({"error": "Pipeline no activo"}), 404
    return jsonify(stats)


@app.route("/api/cajas/sources/<int:source_id>/reset", methods=["POST"])
def cajas_reset(source_id):
    CajasManager.get().reset(source_id)
    return jsonify({"reset": source_id})


@app.route("/api/cajas/sources/<int:source_id>/line-y", methods=["POST"])
def cajas_line_y(source_id):
    data = request.get_json(silent=True) or {}
    pct = max(0, min(100, int(data.get("pct", 85))))
    set_setting("cajas_line_y", str(pct))
    CajasManager.get().set_line_y(source_id, pct)
    return jsonify({"line_y_pct": pct})


@app.route("/api/cajas/stream/<int:source_id>")
def cajas_stream(source_id):
    return _stream_response("cajas", source_id)


# ─────────────────────────────────────────────
# API Reglamento
# ─────────────────────────────────────────────

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
            fps_limit=fps_limit)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 429
    return jsonify({"started": source_id})


@app.route("/api/reglamento/sources/<int:source_id>/stop", methods=["POST"])
def reglamento_stop(source_id):
    ReglamentoManager.get().stop(source_id)
    return jsonify({"stopped": source_id})


@app.route("/api/reglamento/sources/<int:source_id>/stats")
def reglamento_stats(source_id):
    stats = ReglamentoManager.get().get_stats(source_id)
    if stats is None:
        return jsonify({"error": "Pipeline no activo"}), 404
    return jsonify(stats)


@app.route("/api/reglamento/sources/<int:source_id>/reset", methods=["POST"])
def reglamento_reset(source_id):
    ReglamentoManager.get().reset(source_id)
    return jsonify({"reset": source_id})


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


@app.route("/api/reglamento/stream/<int:source_id>")
def reglamento_stream(source_id):
    return _stream_response("reglamento", source_id)


# ─────────────────────────────────────────────
# API Carga / Descarga
# ─────────────────────────────────────────────

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


@app.route("/api/carga_descarga/sources/<int:source_id>/stop", methods=["POST"])
def carga_descarga_stop(source_id):
    CargaDescargaManager.get().stop(source_id)
    return jsonify({"stopped": source_id})


@app.route("/api/carga_descarga/sources/<int:source_id>/stats")
def carga_descarga_stats(source_id):
    stats = CargaDescargaManager.get().get_stats(source_id)
    if stats is None:
        return jsonify({"error": "Pipeline no activo"}), 404
    return jsonify(stats)


@app.route("/api/carga_descarga/sources/<int:source_id>/reset", methods=["POST"])
def carga_descarga_reset(source_id):
    CargaDescargaManager.get().reset(source_id)
    return jsonify({"reset": source_id})


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


@app.route("/api/carga_descarga/stream/<int:source_id>")
def carga_descarga_stream(source_id):
    return _stream_response("carga_descarga", source_id)


# ─────────────────────────────────────────────
# API EPP
# ─────────────────────────────────────────────

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


@app.route("/api/epp/sources/<int:source_id>/stop", methods=["POST"])
def epp_stop(source_id):
    EppManager.get().stop(source_id)
    return jsonify({"stopped": source_id})


@app.route("/api/epp/sources/<int:source_id>/stats")
def epp_stats(source_id):
    stats = EppManager.get().get_stats(source_id)
    if stats is None:
        return jsonify({"error": "Pipeline no activo"}), 404
    return jsonify(stats)


@app.route("/api/epp/sources/<int:source_id>/reset", methods=["POST"])
def epp_reset(source_id):
    EppManager.get().reset(source_id)
    return jsonify({"reset": source_id})


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


@app.route("/api/epp/stream/<int:source_id>")
def epp_stream(source_id):
    return _stream_response("epp", source_id)


# ─────────────────────────────────────────────
# API Smoke / Humo-Fuego
# ─────────────────────────────────────────────

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


@app.route("/api/smoke/sources/<int:source_id>/stop", methods=["POST"])
def smoke_stop(source_id):
    SmokeManager.get().stop(source_id)
    return jsonify({"stopped": source_id})


@app.route("/api/smoke/sources/<int:source_id>/stats")
def smoke_stats(source_id):
    stats = SmokeManager.get().get_stats(source_id)
    if stats is None:
        return jsonify({"error": "Pipeline no activo"}), 404
    return jsonify(stats)


@app.route("/api/smoke/sources/<int:source_id>/reset", methods=["POST"])
def smoke_reset(source_id):
    SmokeManager.get().reset(source_id)
    return jsonify({"reset": source_id})


@app.route("/api/smoke/stream/<int:source_id>")
def smoke_stream(source_id):
    return _stream_response("smoke", source_id)


# ─────────────────────────────────────────────
# API Vehículos
# ─────────────────────────────────────────────

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


@app.route("/api/vehiculos/sources/<int:source_id>/stop", methods=["POST"])
def vehiculos_stop(source_id):
    VehiculosManager.get().stop(source_id)
    return jsonify({"stopped": source_id})


@app.route("/api/vehiculos/sources/<int:source_id>/stats")
def vehiculos_stats(source_id):
    stats = VehiculosManager.get().get_stats(source_id)
    if stats is None:
        return jsonify({"error": "Pipeline no activo"}), 404
    return jsonify(stats)


@app.route("/api/vehiculos/sources/<int:source_id>/reset", methods=["POST"])
def vehiculos_reset(source_id):
    VehiculosManager.get().reset(source_id)
    return jsonify({"reset": source_id})


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


@app.route("/api/vehiculos/stream/<int:source_id>")
def vehiculos_stream(source_id):
    return _stream_response("vehiculos", source_id)


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
