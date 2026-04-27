# 登录 / 鉴权模块 PRD (历史 v1, 已上线)

支撑 web + 移动端的统一身份系统, v1 仅邮箱登录. 用户提交邮箱 + 密码, 服务端 bcrypt 比对, 成功签发 JWT (7 天) + refresh token (30 天).

## 关键约束

- 同账号最多 5 个活跃 session, 超过淘汰最旧
- 登录失败 5 次 / 15 min 锁定 (按 IP + 邮箱)
- JWT 过期前 24h 内可 refresh; 过期后必须重新登录
- 异地登录 (city 级 IP 变化) 触发邮件提醒

## 上线后教训

1. **CSRF 漏洞**: 初版未启 CSRF token, 上线 3 周后白帽报告. 修复: 所有 POST /api/auth/* 必须验 `X-CSRF-Token` (双 cookie 模式).
2. **JWT 撤销**: 改密码后旧 token 仍能用 7 天. 修复: 改密码立即把 active session 加 Redis 黑名单 (TTL = JWT 剩余).
3. **refresh race**: 多 tab 并发 refresh, 老的失效用户被踢. 修复: refresh 接口幂等 + 5s 内重复请求返同一 token.
