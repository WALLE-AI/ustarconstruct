import os

class BaseConfig:
    JSON_AS_ASCII = False
    JSONIFY_PRETTYPRINT_REGULAR = False

    # OpenAI-style client settings
    API_KEY = os.getenv("API_KEY", "")
    BASE_URL = os.getenv("BASE_URL", "")  # optional custom gateway
    MODEL = os.getenv("MODEL", "")
    OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
    DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"

    # CORS
    CORS_ALLOW_ORIGINS = os.getenv(
        "CORS_ALLOW_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1,http://localhost,http://localhost:4173"
    ).split(",")

    # Supabase (history store)
    CHAT_STORE_ENABLED = os.getenv("CHAT_STORE_ENABLED", "true").lower() == "true"
    SUPABASE_URL = os.getenv("SUPABASE_URL", "")
    SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
    SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")  # recommended for server-side inserts
    # table names (customizable if you already have schema)
    SUPABASE_TABLE_CONVERSATIONS = os.getenv("SUPABASE_TABLE_CONVERSATIONS", "conversations")
    SUPABASE_TABLE_MESSAGES = os.getenv("SUPABASE_TABLE_MESSAGES", "messages")

class DevConfig(BaseConfig):
    DEBUG = True

class ProdConfig(BaseConfig):
    DEBUG = False

def get_config(name: str | None):
    if not name:
        return DevConfig
    name = name.lower()
    if name in ("prod", "production"):
        return ProdConfig
    return DevConfig