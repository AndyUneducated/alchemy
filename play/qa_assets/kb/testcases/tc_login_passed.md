# 测试用例集: 邮箱登录 (历史, 已通过)

## 元信息

- 关联需求: 登录 / 鉴权 v1 (REQ-LOGIN-001)
- 测试人: QA-Charlie
- 状态: 全部通过 (上线前最后一轮)
- 上线日期: 2025-09

## 用例

### TC-LOGIN-001 [P0][functional] 正常登录
- 给定: 已注册激活的邮箱 + 正确密码
- 当: 提交登录请求
- 那么: 返回 200 + JWT + refresh token, 写入用户 secure storage

### TC-LOGIN-002 [P0][functional] 密码错误
- 给定: 已注册邮箱 + 错误密码
- 当: 提交登录请求
- 那么: 返回 401, 不暴露"是邮箱错还是密码错"(防探测)

### TC-LOGIN-003 [P0][boundary] 连续 5 次密码错误锁定
- 给定: 5 次登录失败 (同邮箱 + 同 IP)
- 当: 第 6 次提交
- 那么: 返回 423, 提示"账号已锁定 15 分钟"; 15 分钟后自动解锁

### TC-LOGIN-004 [P1][edge] 改密码后旧 token 立即失效
- 给定: 用户登录后获得 JWT-A
- 当: 用户改密码后, 用 JWT-A 调任何鉴权接口
- 那么: 返回 401 "session expired", 即使 JWT 未过期

### TC-LOGIN-005 [P0][security] CSRF token 校验
- 给定: 用户已登录, 攻击者伪造 POST /api/auth/* 请求
- 当: 缺失 CSRF token 或 token 不匹配
- 那么: 返回 403, 不执行任何状态变更

### TC-LOGIN-006 [P1][functional] session 上限淘汰
- 给定: 同账号已有 5 个活跃 session
- 当: 第 6 个客户端登录
- 那么: 最早登录的那个 session 被踢 (token 进黑名单)

### TC-LOGIN-007 [P2][functional] 异地登录邮件提醒
- 给定: 用户上次登录在杭州, IP 地理库识别本次登录在新加坡
- 当: 登录成功
- 那么: 5 分钟内发送"异地登录"邮件, 含登录时间 + 城市 + IP

### TC-LOGIN-008 [P1][edge] refresh token race condition
- 给定: 用户在 2 个 tab 同时打开应用, JWT 即将过期
- 当: 两个 tab 同时调 /refresh
- 那么: 后端返回同一个新 JWT, 旧 refresh token 继续可用 5 秒
