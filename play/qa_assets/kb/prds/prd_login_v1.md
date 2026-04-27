# 登录 / 鉴权模块 PRD (历史 v1, 已上线)

## 背景

支撑 web + 移动端的统一身份系统. 第一版只接邮箱登录, 不带 SSO.

## 核心流程

1. 用户输入邮箱 + 密码 → 客户端校验格式 → 提交 POST /api/login
2. 服务端 bcrypt 比对密码 → 成功签发 JWT (有效期 7 天) + refresh token (30 天)
3. 客户端把 token 存入 secure storage (mobile keychain / web httpOnly cookie)

## 关键约束

- 同一账号同一时间最多 5 个活跃 session, 超过淘汰最旧的
- 登录失败 5 次后锁定 15 分钟 (按 IP + 邮箱组合计数, 防撞库)
- JWT 续期窗口: 过期前 24h 内可用 refresh token 续期, 过期后需重新登录
- 异地登录需邮件提醒 (从 IP 地理库判断 city 级别变化)

## 教训 (来自上线后 review)

1. **CSRF token 漏洞**: 初版未启用 CSRF token 防护, 上线 3 周后被白帽报告.
   修复后 P0 任何鉴权状态变更接口必须验 CSRF token.
2. **JWT 撤销**: 初版没有 token 黑名单机制, 用户改密码后旧 token 仍能用 7 天.
   修复: 改密码立即把所有 active session 加入黑名单 (Redis SET, TTL = JWT 剩余时间).
3. **并发 refresh race condition**: 客户端多 tab 同时 refresh, 服务端发了多个新 refresh
   token, 老的失效引发用户被踢. 修复: refresh 接口幂等 + 5s 内重复请求返回同一 token.

## 测试覆盖要点

- 登录成功 / 密码错 / 邮箱不存在 / 账号被锁
- session 上限淘汰
- token 过期 / refresh 流程
- 改密码后 session 黑名单生效
- CSRF token 缺失或错误 → 403
- 异地登录邮件触发条件
