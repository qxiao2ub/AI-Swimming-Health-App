-- Persistent visitor counter for the Kevin Sun AI Swimming Health App.
-- Run this script in the Supabase SQL editor.

create table if not exists public.app_counters (
    slug text primary key,
    visit_count bigint not null default 0,
    updated_at timestamptz not null default now()
);

alter table public.app_counters enable row level security;

create or replace function public.increment_app_counter(counter_slug text)
returns bigint
language plpgsql
security definer
set search_path = public
as $$
declare
    new_count bigint;
begin
    insert into public.app_counters (slug, visit_count, updated_at)
    values (counter_slug, 1, now())
    on conflict (slug)
    do update set
        visit_count = public.app_counters.visit_count + 1,
        updated_at = now()
    returning visit_count into new_count;

    return new_count;
end;
$$;

revoke all on function public.increment_app_counter(text) from public;
grant execute on function public.increment_app_counter(text) to anon, authenticated;
