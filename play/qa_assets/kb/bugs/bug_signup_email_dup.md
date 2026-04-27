# BUG-2025-06-03 注册并发（P1）

同邮箱并发「先查再写」双插。修：`redis SETNX signup:{email} 60s` 串行化。教训：唯一键须锁/SETNX；TC-SIGNUP-003。
