from flask import Flask, jsonify
from .config import get_config
from .extensions import cors
from .routes.chat import chat_bp

def create_app(config_name: str | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(get_config(config_name))

    # Apply CORS (origins from config)
    cors.init_app(app, 
                  resources={r"/*": {"origins": app.config["CORS_ALLOW_ORIGINS"]}})
    
    cors.init_app(app)

    # Blueprints
    app.register_blueprint(chat_bp)

    # Health check
    @app.get("/healthz")
    def healthz():
        return jsonify({"status": "ok"})

    return app