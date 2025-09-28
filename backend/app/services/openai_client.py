import os
from typing import Optional, Dict, Any
from flask import current_app
from openai import OpenAI


model_providers_url = {
                "Gemini": "https://api.siliconflow.cn/v1",
                "OpenAI": "https://api.openai.com/v1",       
                "SiliconFlow": "https://api.siliconflow.cn/v1", 
                "OpenRouter":"https://openrouter.ai/api/v1", 
                "Local":os.environ.get("LOCAL_URL")}

def build_openai_client(api_key,model_provider_name) -> Optional[OpenAI]:
    if model_provider_name not in model_providers_url:
        current_app.logger.warning(f"Unknown model provider: {model_provider_name}")
        return None
    if model_provider_name == "Local":
        api_key = "empty"
    else:
        api_key = api_key
    base_url = model_providers_url[model_provider_name]
    if not api_key:
        return None
    kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    return OpenAI(**kwargs)