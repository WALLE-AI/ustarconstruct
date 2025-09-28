from flask import Blueprint, request, Response, jsonify, stream_with_context
from pydantic import ValidationError
from typing import Any

from app.schemas.chat import ChatRequest, normalize_role
from app.services.streaming import openai_stream_generator
from app.services.chat_store import persist_chat_history

chat_bp = Blueprint("chat", __name__)

SEP = "---RESPONSE---"

def _wrap_and_persist(gen, req: ChatRequest):
    """Yield chunks while accumulating plan/answer to persist at the end."""
    plan_buf = []
    answer_buf = []
    seen_sep = False

    for chunk in gen:
        try:
            text = chunk.decode("utf-8", "ignore")
        except Exception:
            text = ""

        if not seen_sep:
            idx = text.find(SEP)
            if idx != -1:
                before = text[:idx]
                after = text[idx + len(SEP):]
                if before:
                    plan_buf.append(before)
                seen_sep = True
                if after:
                    answer_buf.append(after)
            else:
                plan_buf.append(text)
        else:
            answer_buf.append(text)

        yield chunk  # always forward to client immediately

    # Persist after stream finishes (best-effort)
    # try:
    #     persist_chat_history(
    #         req=req,
    #         plan_text="".join(plan_buf).strip(),
    #         answer_text="".join(answer_buf).strip(),
    #         conversation_id=req.conversationId,
    #         user_id=req.userId,
    #     )
    # except Exception:
    #     # do not break the response if persistence fails
    #     print("Failed to persist chat history.")

@chat_bp.post("/stream-chat")
def stream_chat():
    try:
        payload_json: dict[str, Any] = request.get_json(silent=True) or {}
        req = ChatRequest.model_validate(payload_json)

        # Ensure the last message is from user (after normalization, to be safe)
        if not req.messages or normalize_role(req.messages[-1].sender) != "user":
            return jsonify({"detail": "Last message must be from the user."}), 400

        gen = openai_stream_generator(req)
        wrapped = _wrap_and_persist(gen, req)
        return Response(stream_with_context(wrapped), mimetype="text/plain; charset=utf-8")
    except ValidationError as ve:
        return jsonify({"detail": f"Invalid payload: {ve}"}), 400
    except Exception as e:
        return jsonify({"detail": str(e)}), 500