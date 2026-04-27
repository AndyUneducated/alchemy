# 测试用例集: 邮箱登录 (历史, 全部通过)

关联需求: 登录 / 鉴权 v1; 上线日期 2025-09.

### TC-LOGIN-001 [P0][functional] 正常登录
- 给定已激活邮箱 + 正确密码, 当提交登录, 那么返回 200 + JWT + refresh token.

### TC-LOGIN-003 [P0][boundary] 连续 5 次密码错误锁定
- 给定 5 次登录失败 (同邮箱 + 同 IP), 当第 6 次提交, 那么返回 423 "锁定 15 分钟"; 15 min 后自动解锁.

### TC-LOGIN-005 [P0][security] CSRF token 校验
- 给定用户已登录 + 攻击者伪造 POST /api/auth/*, 当缺失 CSRF token 或不匹配, 那么返回 403, 不执行状态变更.

### TC-LOGIN-004 [P1][edge] 改密码后旧 token 立即失效
- 给定用户登录获得 JWT-A, 当改密码后用 JWT-A 调鉴权接口, 那么返回 401 "session expired", 即使 JWT 未过期.

### TC-LOGIN-008 [P1][edge] refresh token race condition
- 给定 2 tab 同时 JWT 即将过期, 当并发调 /refresh, 那么后端返回同一新 JWT, 旧 refresh token 5s 内继续可用.
