from __future__ import annotations
from typing import Optional
from flask import current_app
from supabase import create_client, Client

def get_supabase() -> Optional[Client]:
    if not current_app.config.get("CHAT_STORE_ENABLED", False):
        return None
    url = current_app.config.get("SUPABASE_URL", "")
    key = (
        current_app.config.get("SUPABASE_SERVICE_ROLE_KEY", "")
        or current_app.config.get("SUPABASE_ANON_KEY", "")
    )
    if not url or not key:
        return None
    client = create_client(url, key)
    # schema = current_app.config.get("SUPABASE_SCHEMA", "public")
    # # supabase-py v2 支持 schema 切换：
    return client