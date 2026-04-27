# 鉴权类需求测试 checklist

适用: 任何涉及用户身份 / session / token / 密码 的需求.

## P0 必测项

### 接口安全
- CSRF token 校验 (双 cookie 或同步 token 模式)
- HTTPS 强制 (HTTP 自动 301)
- CORS 白名单 (只允许自家域名)

### 身份凭据
- 密码字段不进 log / trace / Sentry / 监控 metric
- 密码用 bcrypt / argon2 hash, 不存明文
- JWT secret rotation 后旧 token 立即失效 (黑名单)
- refresh token 单次有效 (用过即作废)

### 防爆破 / 异常会话
- 登录失败 5 次 / 15 min 锁定 (按 IP + 邮箱)
- 注册 / 重发邮件每邮箱 5 min 最多 1 次
- 改密码后所有 session 立即失效
- 异地登录 (city 级 IP 变化) 邮件提醒

## 历史教训

| 事故 | 教训 |
|---|---|
| BUG-2025-04-12 CSRF | 鉴权状态变更接口必须验 CSRF token |
| BUG-2025-06-03 注册并发 | "先查再写" 必须 SETNX / 锁串行化 |
| BUG-2024-11-08 密码进 log | logger filter 屏蔽 password / token / secret 字段 |
