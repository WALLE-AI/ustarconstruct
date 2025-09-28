from __future__ import annotations
from typing import Any, Dict, Optional
from datetime import datetime, timezone
from flask import current_app
from .supabase_client import get_supabase

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def create_conversation(title: str, user_id: Optional[str]) -> Optional[str]:
    sb = get_supabase()
    if not sb:
        return None
    sb = sb.schema("public")
    table = current_app.config.get("SUPABASE_TABLE_CONVERSATIONS", "conversations")

    data = {"title": title, "user_id": user_id, "created_at": _now_iso()}
    try:
        res = sb.table(table).insert(data).execute()
        row = (res.data or [])[0]
        return row.get("id")
    except Exception as e:
        print("Failed to create conversation.",e)
        return None

def append_message(conversation_id: str, role: str, content: Any, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
    sb = get_supabase()
    if not sb:
        return None
    table = current_app.config.get("SUPABASE_TABLE_MESSAGES", "messages")
    payload = {
        "conversation_id": conversation_id,
        "role": role,
        "content": content,   # consider jsonb in schema if you store structured content
        "metadata": metadata or {},
        "created_at": _now_iso(),
    }
    res = sb.table(table).insert(payload).execute()
    try:
        row = (res.data or [])[0]
        return row.get("id")
    except Exception:
        return None

def persist_chat_history(
    req: Any,
    plan_text: str,
    answer_text: str,
    conversation_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Optional[str]:
    """Persist a whole turn to Supabase (best-effort)."""
    sb = get_supabase()
    if not sb:
        return None

    # Guess a title
    last_user_text = ""
    if req and getattr(req, "messages", None):
        for m in reversed(req.messages):
            if (m.sender or "").lower() in ("user", "human"):
                last_user_text = m.get_text()
                if last_user_text:
                    break
    title = (last_user_text or "New Chat").strip()[:40]

    conv_id = conversation_id or create_conversation(title=title, user_id=user_id) or ""

    # Save request messages
    if getattr(req, "messages", None):
        for m in req.messages:
            role = (m.sender or "user").lower()
            if role =="ai":
                role = "assistant"
            content = m.get_text()
            meta: Dict[str, Any] = {}
            if m.images:
                meta["images_count"] = len(m.images)
            append_message(conv_id, role, content, meta)

    # Save assistant answer
    meta = {"plan": plan_text}
    append_message(conv_id, "assistant", answer_text, meta)

    return conv_id