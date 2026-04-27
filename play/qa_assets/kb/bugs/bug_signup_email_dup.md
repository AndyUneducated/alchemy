# BUG-2025-06-03: 邮箱注册并发竞争 (P1, 已修复)

## 简述

同一邮箱在 100ms 窗口内并发提交注册请求, 偶尔出现两条记录同时进入 service 层
的 "create user" 流程, 数据库 unique 约束触发后, 一条入 audit log 显示成功,
另一条触发 500 retry, 用户看到"注册失败"但 audit log 显示账户已创建.

## 影响范围

- 触发频次: 灰度阶段一周内 7 例
- 影响用户: 7 个用户首次注册时报错, 但邮箱已被自己占住, 重新注册返回 "已被注册"
- 客服工单: 12 张 (含 5 张转技术排查)

## 根因

注册 service 层先 SELECT email 是否存在 → 不存在则 INSERT. 这两步之间无锁,
导致并发请求都通过 SELECT 阶段, 同时 INSERT 时只有一条命中 unique 约束.

## 修复

```python
# Redis SETNX 做 60s 注册保护
if not redis.set(f"signup:{email}", "1", nx=True, ex=60):
    return Response(409, "请求过于频繁, 请稍后再试")
try:
    if user_exists(email):
        return Response(409, "邮箱已注册")
    create_user(email, ...)
finally:
    redis.delete(f"signup:{email}")
```

## 教训

- 任何 "先查再写" 流程必须**显式**串行化 (锁 / SETNX / DB unique + 重试).
- 邮件注册 / 改邮箱 / 改用户名等 "唯一键变更" 都要测**并发场景**, 仅靠 DB unique
  约束不够 (业务逻辑可能在约束触发前先 commit 一些副作用).
- 测试用例 TC-SIGNUP-003 标记为 P0 (从 P2 升级), 必须用并发模拟工具 (locust /
  k6) 跑 100ms 窗口的同邮箱压测.
