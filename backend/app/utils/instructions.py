from app.schemas.chat import ChatRequest

def build_system_instruction(req: ChatRequest) -> str:
    """
    Safer alternative to chain-of-thought: emit a tiny plan (<=3 bullets, <=40 words),
    then answer. If knowledge_base flag is present, nudge to use it; if use_web_search
    is on, ask the model to note what it'd search for (no actual browsing here).
    """
    kb_note = ""
    if req.knowledge_base:
        kb_note = f" Prioritize reliable facts from the '{req.knowledge_base}' knowledge base when relevant."

    web_note = ""
    if req.use_web_search:
        web_note = " If up-to-date facts are required, say what you would search for and proceed with best-known stable info."

    return (
        "Before answering, output a VERY BRIEF high-level plan with at most 3 bullet points and 40 words total. "
        "After that, provide the final user-facing answer. "
        "Do NOT reveal internal chain-of-thought or token-by-token reasoning."
        + kb_note
        + web_note
    )