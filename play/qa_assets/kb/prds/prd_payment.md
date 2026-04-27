# 支付 PRD v2（摘要）

Stripe/微信/支付宝回调；HMAC 验签；order 幂等；金额与订单一致。教训：回调加 nonce+时间窗；金额单位在 SDK 层统一到元；支付/退款 order 级分布式锁。
