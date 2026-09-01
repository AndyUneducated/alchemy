# Email signup PRD (example · short)

Email + password signup; no SSO. Password ≥8 with letters and digits; double-end email validation; already registered → clear prompt to log in. Activation email after signup (link 24h); `unverified` cannot log in; resend activation, ≤1 per email per 5min.

NFR: signup P95 <300ms@100RPS; password not in log/trace; i18n; accessible errors.

Edge cases: duplicate signup for same email within 5min is idempotent; email >254 → 400; email down → enqueue first, copy says "sending email".
