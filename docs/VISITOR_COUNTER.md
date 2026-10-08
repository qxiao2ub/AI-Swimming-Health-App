# Visitor counter: local fallback vs persistent history

The app counts **browser sessions**, not unique people; one session is counted once across Streamlit reruns. A refresh/reconnect can create a new session, so the displayed count should not be described as verified unique visitors.

## Without Supabase (ready by default)

The app stores its counter in SQLite inside the running server's `/tmp`. It persists over reruns and may persist over restarts of the Python process on the same machine, but **can reset on Streamlit Community Cloud instance recreation or redeployment**. It is not a lifetime cloud counter.

## Persistent counter across redeploys: Supabase

1. Create a Supabase project at https://supabase.com (free tier subject to service limits).
2. In the Supabase SQL editor, run [`supabase_visitor_counter.sql`](supabase_visitor_counter.sql). It creates `app_counters` and an atomic `increment_app_counter(counter_slug)` RPC.
3. In the Streamlit app's **Settings → Secrets**, supply these values (do not commit them to GitHub):

   ```toml
   SUPABASE_URL = "https://YOUR-PROJECT.supabase.co"
   SUPABASE_ANON_KEY = "YOUR-ANON-PUBLIC-KEY"
   COUNTER_SLUG = "kevin-sun-ai-swimming-health"
   ```

4. Reboot once. The sidebar counter should show **Storage: Supabase**.
5. If Supabase is unavailable, local SQLite fallback is shown transparently; these separate stores are not automatically synchronized.

No user names, email addresses, IP addresses or login identities are stored by this counter. The counter RPC is publicly callable and can be abused to inflate the count; it is only suitable for a non-critical educational metric, not audited visitor statistics.
