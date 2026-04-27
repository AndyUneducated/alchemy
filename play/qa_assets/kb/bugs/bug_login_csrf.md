# BUG-2025-04-12 CSRF（P0）

`/api/auth/*` 状态变更未验 CSRF。修：POST 须 `X-CSRF-Token` 与 cookie 双检；TC-LOGIN-005 P0。
