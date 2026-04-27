# 鉴权类需求测试 checklist

适用范围: 任何"涉及用户身份 / session / token / 密码"的需求都套这一份.

## P0 必测项

### 接口安全
- [ ] CSRF token 校验 (双 cookie 或 同步 token 模式)
- [ ] HTTPS 强制 (HTTP 自动 301 到 HTTPS)
- [ ] CORS 白名单 (只允许应用自家域名)
- [ ] 接口验签 (敏感接口加 HMAC)

### 身份凭据
- [ ] 密码字段不进任何 log / trace / Sentry / 监控 metric
- [ ] 密码用 bcrypt / argon2 hash, 不存明文
- [ ] JWT secret rotation 后旧 token 立即失效 (黑名单)
- [ ] refresh token 单次有效 (用过即作废)

### 防爆破
- [ ] 登录失败 5 次 / 15 分钟内锁定 (按 IP + 邮箱)
- [ ] 注册 / 重发邮件每邮箱 5 分钟最多 1 次
- [ ] 验证码图形 / 短信限频 (按手机号 / 邮箱 + IP)

### 异常会话
- [ ] 改密码后所有 session 立即失效
- [ ] 异地登录邮件提醒 (按 IP 地理库 city 级别变化)
- [ ] 多端在线上限 (5 / 10 个 session 淘汰最旧)

## P1 推荐项

- [ ] 2FA 流程 (TOTP / SMS / Email)
- [ ] 暴力探测告警 (Sentry / Datadog 报警, 同邮箱 60min 内 > 50 次失败)
- [ ] 弱密码字典对比 (top 10000 弱密码拒绝)
- [ ] 时间常量比较 (避免 timing attack 探测密码)

## P2 加分项

- [ ] WebAuthn / passkey 支持
- [ ] 行为风控 (登录时间 / 设备指纹异常)
- [ ] OAuth 第三方登录账号联动
- [ ] 账号删除 GDPR 流程

## 历史教训速查

| 事故 | 教训 |
|---|---|
| BUG-2025-04-12 (CSRF) | 鉴权状态变更接口必须验 CSRF token |
| BUG-2025-06-03 (注册并发) | "先查再写" 必须 SETNX / 锁串行化 |
| BUG-2024-11-08 (密码进 log) | 用 logger filter 屏蔽 password / token / secret 字段 |
