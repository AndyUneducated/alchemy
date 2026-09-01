# Login PRD v1 (summary)

Email + password; bcrypt; JWT 7d + refresh 30d; ≤5 sessions per account; 5 failures / 15min lock (IP + email). Lessons: auth POST requires `X-CSRF-Token`; JWT blacklist after password change; refresh idempotency against multi-tab races.
