from typing import Iterator
from flask import current_app

# OpenAI SDK (>=1.0)
from openai import OpenAIError

from app.schemas.chat import ChatRequest, to_openai_messages
from app.services.openai_client import build_openai_client
from app.utils.instructions import build_system_instruction

def openai_stream_generator(req: ChatRequest) -> Iterator[bytes]:
    api_key = req.model_extra["model_config"]["api_key"]
    client = build_openai_client(api_key=api_key, model_provider_name=req.model_extra["model_config"]["provider"]) 
    model =req.model_extra["model_config"]["model_name"]
    temperature = req.model_extra["model_config"]["temperature"] if req.model_extra["model_config"]["temperature"] is not None else current_app.config.get("OPENAI_TEMPERATURE", 0.3)
    demo_mode = current_app.config.get("DEMO_MODE", False)
    context_length = req.model_extra["model_config"]["context_length"] if req.model_extra["model_config"]["context_length"] is not None else 2048

    # Demo fallback when no key is configured
    if client is None or demo_mode:
        plan = "- Understand the question\n- Outline 2–3 steps\n- Provide final answer\n"
        yield plan.encode("utf-8")
        yield b"\n---RESPONSE---\n"
        if client is None:
            yield b"(Demo mode) No model is configured. Please set OPENAI_API_KEY."
        else:
            yield b"(Demo mode) DEMO_MODE=true; not calling model."
        return

    # system_instruction = build_system_instruction(req)
    oa_messages = to_openai_messages(req)

    # Prepend the system instruction
    # oa_messages = [{"role": "system", "content": [{"type": "text", "text": system_instruction}]}] + oa_messages

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=oa_messages,
            stream=True,
            temperature=temperature,
            max_tokens=context_length,
        )

        for chunk in stream:
            try:
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None)
                if text:
                    yield text.encode("utf-8")
            except Exception:
                # Skip malformed chunk
                continue

    except OpenAIError as e:
        err = f"[OpenAIError] {str(e)}"
        yield f"\n{err}".encode("utf-8")
    except Exception as e:
        err = f"[ServerError] {str(e)}"
        yield f"\n{err}".encode("utf-8")