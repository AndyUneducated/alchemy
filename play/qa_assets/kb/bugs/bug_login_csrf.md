# BUG-2025-04-12: 登录接口 CSRF token 缺失 (P0, 已修复)

## 简述

白帽研究员发现 `/api/auth/change_password` 等鉴权状态变更接口未校验 CSRF token,
攻击者可构造跨站请求让登录用户在不知情情况下修改密码.

## 影响范围

- 影响接口: 所有 POST /api/auth/* 状态变更接口 (change_password / change_email
  / disable_2fa / logout_all_sessions)
- 影响用户: 全量已登录用户 (~140 万)
- 实际利用: 0 (上报方案是 PoC, 无 in-the-wild 利用)

## 根因

初版鉴权框架仅校验 JWT cookie, 未校验 same-origin / CSRF token. JWT 走 cookie
导致天然支持 cross-site auto-submit.

## 修复

1. 全部 POST /api/auth/* 强制要求 `X-CSRF-Token` header, 与 cookie 中的
   `csrfToken` (双 cookie 模式) 比对.
2. CSRF token 由登录响应一并下发, 24h 有效.
3. 测试用例: TC-LOGIN-005 标记为 P0, 入回归套件.

## 教训

- 任何鉴权状态变更接口默认 P0, **必须**测 CSRF token 校验.
- 双 cookie + header 比对是行业标准, 别自创 token 派发逻辑.
- 安全测试用例须随业务用例一起进 CI, 不能只靠定期渗透.
