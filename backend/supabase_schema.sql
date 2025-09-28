-- Supabase schema for chat history (run in SQL editor)

-- Conversations table
create table if not exists public.conversations (
  id uuid primary key default gen_random_uuid(),
  title text not null,
  user_id uuid null,
  created_at timestamptz not null default now()
);

-- Messages table
create table if not exists public.messages (
  id uuid primary key default gen_random_uuid(),
  conversation_id uuid not null references public.conversations(id) on delete cascade,
  role text not null check (role in ('user','assistant','system')),
  content text,          -- or change to jsonb if you prefer structured content
  metadata jsonb default '{}'::jsonb,
  created_at timestamptz not null default now()
);

-- If you plan to expose to clients with anon key, enable RLS and add policies.
-- For server-side ONLY with service role, RLS can remain but the service role bypasses it.
alter table public.conversations enable row level security;
alter table public.messages enable row level security;

-- Example policies (scoped by user_id via auth.uid())
create policy if not exists "Allow select own conversations"
  on public.conversations for select
  using (user_id = auth.uid());

create policy if not exists "Allow insert own conversations"
  on public.conversations for insert
  with check (user_id = auth.uid());

create policy if not exists "Allow select messages by conversation"
  on public.messages for select
  using (
    exists (
      select 1 from public.conversations c
      where c.id = messages.conversation_id and c.user_id = auth.uid()
    )
  );

create policy if not exists "Allow insert messages for own conversations"
  on public.messages for insert
  with check (
    exists (
      select 1 from public.conversations c
      where c.id = messages.conversation_id and c.user_id = auth.uid()
    )
  );