# 性能 / SLO 测试 checklist

适用: 所有对外接口 (Web / Mobile / API).

## SLO baseline (公司级)

| 接口类别 | P50 | P95 | P99 | 错误率 |
|---|---|---|---|---|
| 鉴权类 (login / refresh) | < 100ms | < 200ms | < 500ms | < 0.1% |
| 写操作 (订单 / 支付) | < 200ms | < 500ms | < 1s | < 0.5% |
| 查询类 (列表 / 详情) | < 80ms | < 200ms | < 400ms | < 0.05% |
| 异步回调 | N/A | < 5s | < 30s | < 1% |

注册接口 P95 < 300ms (并发 100 RPS) 是默认 baseline.

## 必测项

- 单实例 RPS 上限 (爬升至错误率 > 1% 或延迟 > P99 SLO)
- 全链路压测 (含下游 DB / 缓存 / MQ)
- 下游缓存全挂 → 接口能 fallback 到 DB / graceful degrade
- 第三方 API 超时 → 熔断后降级 (circuit breaker)

## 历史教训

| 故障 | 教训 |
|---|---|
| 2025-08 大促雪崩 | DB 连接池上限 100, 雪崩时只放 80 给主请求, 20 给监控/降级路径 |
| 2025-09 邮件队列阻塞 | 同步调外部 SDK 是反模式; 外部依赖必须异步化 + 队列 |
| 2025-11 密码进 trace | OpenTelemetry 默认全字段抓 trace; auth 路径必须显式 attribute_filter 屏蔽 |
