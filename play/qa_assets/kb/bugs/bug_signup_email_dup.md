# BUG-2025-06-03: 邮箱注册并发竞争 (P1, 已修复)

同邮箱 100ms 窗口内并发提交注册, 两条记录同时进 service 层 "create user", DB unique 约束触发后一条入 audit log 显示成功, 另一条触发 500 retry, 用户看到"注册失败"但账户已创建. 灰度阶段 1 周 7 例 / 12 张客服工单.

## 根因

service 层先 SELECT email 是否存在 → 不存在则 INSERT. 两步之间无锁, 并发请求都通过 SELECT, 同时 INSERT 时只有一条命中 unique.

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
- 邮箱 / 用户名等"唯一键变更"必须测**并发场景**, 仅靠 DB unique 不够.
- TC-SIGNUP-003 升级 P0, 用 locust / k6 跑 100ms 窗口同邮箱压测.
