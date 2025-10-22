from __future__ import annotations
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, ConfigDict

# --------- Data Models (keep parity with frontend payload) ---------

class InlineData(BaseModel):
    data: str
    mimeType: str

class ImagePart(BaseModel):
    inlineData: InlineData

class FrontendMessage(BaseModel):
    # Be lenient and map later to OpenAI-compatible roles
    sender: str
    text: Optional[str] = None
    content: Optional[str] = None
    value: Optional[str] = None
    images: Optional[List[ImagePart]] = None
    model_config = ConfigDict(extra='allow')

    def get_text(self) -> str:
        return (self.text or self.content or self.value or "").strip()

class ChatRequest(BaseModel):
    messages: List[FrontendMessage]
    use_web_search: Optional[bool] = False
    knowledge_base: Optional[str] = None
    # Added: persistence-friendly fields from frontend
    conversationId: Optional[str] = None
    userId: Optional[str] = None
    model_config = ConfigDict(extra='allow')

# --- Role normalization ---
def normalize_role(sender: str) -> str:
    s = (sender or "").lower()
    if s in ("assistant", "gpt", "ai", "bot"):
        return "assistant"
    if s in ("user", "human"):
        return "user"
    if s == "system":
        return "system"
    return "user"


# --- Convert to OpenAI messages ---
def to_openai_messages(payload: ChatRequest) -> List[Dict[str, Any]]:
    openai_messages: List[Dict[str, Any]] = []

    # sys_prefix = system_prompt
    # openai_messages.append({"role": "system", "content": sys_prefix})

    flags_note = f"(search={'on' if payload.use_web_search else 'off'}, kb={payload.knowledge_base or 'none'})"
    openai_messages.append({"role": "system", "content": f"Context flags: {flags_note}"})

    for m in payload.messages:
        role = normalize_role(m.sender)
        text = m.get_text()

        if m.images:
            content: List[Dict[str, Any]] = []
            if text:
                content.append({"type": "text", "text": text})
            for p in m.images:
                data_url = f"data:{p.inlineData.mimeType};base64,{p.inlineData.data}"
                content.append({"type": "image_url", "image_url": {"url": data_url}})
            openai_messages.append({"role": role, "content": content})
        else:
            openai_messages.append({"role": role, "content": text or ""})

    return openai_messages