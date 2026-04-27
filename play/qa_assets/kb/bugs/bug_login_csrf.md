# BUG-2025-04-12: 登录接口 CSRF token 缺失 (P0, 已修复)

白帽报告 `/api/auth/change_password / change_email / disable_2fa / logout_all_sessions` 等所有鉴权状态变更接口未校验 CSRF token, 攻击者构造跨站请求即可改用户密码. 影响全量 ~140 万已登录用户; 实际利用为 0.

## 根因

初版鉴权框架仅校验 JWT cookie, 未校验 same-origin / CSRF token; JWT 走 cookie 天然支持 cross-site auto-submit.

## 修复

1. POST /api/auth/* 强制要求 `X-CSRF-Token` header, 与 cookie 中的 `csrfToken` 双 cookie 模式比对.
2. CSRF token 由登录响应一并下发, 24h 有效.
3. 测试用例 TC-LOGIN-005 升 P0 入回归套件.

## 教训

- 任何鉴权状态变更接口默认 P0, **必须**测 CSRF token 校验.
- 双 cookie + header 是行业标准, 别自创 token 派发逻辑.
- 安全测试用例随业务用例一起进 CI.
