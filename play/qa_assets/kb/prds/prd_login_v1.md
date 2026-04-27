# 登录 PRD v1（摘要）

邮箱+密码；bcrypt；JWT 7d + refresh 30d；同账号 ≤5 session；失败 5 次/15min 锁（IP+邮箱）。教训：鉴权 POST 须 `X-CSRF-Token`；改密码后 JWT 黑名单；refresh 幂等防多 tab 竞态。
