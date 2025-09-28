# Flask Blueprint Chat Backend (with Supabase)

A Flask + Blueprints refactor of your FastAPI streaming chat backend.
Keeps Pydantic for robust request validation and uses the OpenAI SDK (>=1.0) with streaming.
Adds Supabase integration to persist conversations and messages.

## Project Layout

```text
app/
  __init__.py            # app factory, CORS, /healthz
  config.py              # env-driven config
  extensions.py          # CORS instance
  routes/
    chat.py              # /stream-chat endpoint (Blueprint)
  services/
    openai_client.py     # OpenAI client builder
    streaming.py         # streaming generator
    supabase_client.py   # Supabase client builder
    chat_store.py        # persistence helpers
  schemas/
    chat.py              # Pydantic models + role mapping + message conversion
  utils/
    instructions.py      # system instruction builder
wsgi.py                  # entry for `python wsgi.py` or WSGI servers
requirements.txt
.env.example
supabase_schema.sql
```

## Quickstart

1) Create a virtualenv and install deps:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2) Copy and edit environment variables:

```bash
cp .env.example .env
# edit .env to set OPENAI_API_KEY, OPENAI_BASE_URL (optional), OPENAI_MODEL, SUPABASE_*, etc.
export $(grep -v '^#' .env | xargs)  # or use a dotenv loader
```

3) Run the server:

```bash
python wsgi.py
# -> serves on http://127.0.0.1:8000
```

4) Endpoint parity with your FastAPI version:

- `GET /healthz` → `{ "status": "ok" }`
- `POST /stream-chat` → streaming text/plain (first a mini plan, then `---RESPONSE---`, then the final answer)

---

## Supabase: Store Chat History

This project can persist conversations/messages to Supabase.

### 1) Enable & Configure
Set environment variables (see `.env.example`):
- `CHAT_STORE_ENABLED=true`
- `SUPABASE_URL=...`
- `SUPABASE_SERVICE_ROLE_KEY=...` (recommended on the server), or `SUPABASE_ANON_KEY`

Optionally customize table names:
- `SUPABASE_TABLE_CONVERSATIONS=conversations`
- `SUPABASE_TABLE_MESSAGES=messages`

### 2) Create Tables & Policies
Open Supabase SQL editor and run: `supabase_schema.sql`.

By default, RLS is enabled with example policies that scope rows to `auth.uid()`.
If you use the service role key in the backend, it bypasses RLS (server-side trusted inserts).

### 3) Frontend Payload Extensions
You can pass optional fields to help grouping:
```jsonc
{
  "conversationId": "uuid-optional",
  "userId": "uuid-optional"
}
```
If `conversationId` is absent, the server creates one using the first 40 chars of the latest user prompt as title.

### 4) What Gets Stored
- All incoming `messages` from the request are inserted as rows.
- The assistant reply is appended after streaming finishes.
- The tiny high-level plan (before `---RESPONSE---`) is saved into the assistant message `metadata.plan`.

### 5) Error Handling
Persistence is best-effort and will not break streaming. Failures are swallowed after logging (you can add logging hooks).