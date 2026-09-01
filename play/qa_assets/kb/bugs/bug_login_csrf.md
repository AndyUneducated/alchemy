# BUG-2025-05-12 CSRF (P0)

`/api/auth/*` state-changing requests did not validate CSRF. Fix: POST requires `X-CSRF-Token` dual-checked with cookie; TC-LOGIN-005 P0.
