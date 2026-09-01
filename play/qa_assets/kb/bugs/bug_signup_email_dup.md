# BUG-2025-06-03 Signup concurrency (P1)

Concurrent signup for the same email used "check then insert" and double-inserted. Fix: `redis SETNX signup:{email} 60s` serializes. Lesson: unique keys need lock/SETNX; TC-SIGNUP-003.
