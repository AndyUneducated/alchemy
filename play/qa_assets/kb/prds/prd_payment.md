# Payment PRD v2 (summary)

Stripe/WeChat/Alipay callbacks; HMAC signature verification; order idempotency; amount must match order. Lessons: callback nonce + time window; amount units normalized to yuan at SDK layer; distributed lock per order for pay/refund.
