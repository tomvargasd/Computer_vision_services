from flask import Flask
from src.config import BASE_DIR, APP_VERSION
from src.database import init_db, set_setting


def create_app() -> Flask:
    app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"),
                static_folder=os.path.join(BASE_DIR, "static"))
    app.secret_key = os.urandom(32)
    app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

    init_db()

    set_setting("version", APP_VERSION)

    from src.routes.dashboard import dashboard_bp
    from src.routes.toggle import toggle_bp
    from src.routes.settings_bp import settings_bp
    from src.routes.api_modules import api_modules_bp

    app.register_blueprint(dashboard_bp)
    app.register_blueprint(toggle_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(api_modules_bp)

    return app


import os
