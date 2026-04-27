# 测试用例集: 邮箱注册 (旧 v1, 部分用例曾失败后修复)

关联需求: REQ-SIGNUP-001 (已合并到 v2); 上线 2025-06; TC-003 / 005 / 007 上线前发现 BUG, 修复后才通过.

### TC-SIGNUP-003 [P0][edge] 同邮箱并发注册 (历史 BUG)
- 给定同邮箱 < 100ms 内 2 客户端同时提交, 当 DB unique 触发, 那么一个返 200 一个返 409.
- 历史问题: service 层无锁导致 race condition, 两条 unverified 都插入. 修复: `redis.SETNX(f"signup:{email}", 60s)` 串行化.

### TC-SIGNUP-005 [P1][security] SQL 注入注册 (历史 BUG)
- 给定邮箱含 SQL 注入 payload (`' OR '1'='1`), 当提交注册, 那么 ORM 参数化转义, 邮箱按字面字符串处理, 返回 400 格式拒绝.
- 历史问题: 初版用 string 拼接 SQL 写 audit log, 注入到日志库. 修复: ORM 全栈 + lint 禁拼接 SQL.

### TC-SIGNUP-007 [P2][edge] 邮件队列阻塞 (历史 BUG)
- 给定邮件服务暂不可达, 当用户注册, 那么请求接受 + 邮件入持久队列 + 用户看"稍候", worker 异步发送, 最长等 30 min.
- 历史问题: 初版同步调邮件 SDK, 卡 30s 超时, 注册返 500 但账户已创建. 修复: 注册立即返回, 邮件走异步队列.

### TC-SIGNUP-008 [P2][a11y] 错误提示无障碍
- 给定表单字段错误, 当屏幕阅读器读取, 那么错误文本带 ARIA role="alert", 颜色不是唯一信号 (有图标 + 文字).
