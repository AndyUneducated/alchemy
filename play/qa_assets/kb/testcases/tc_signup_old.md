# Signup test cases (legacy, summary)

TC-SIGNUP-003: same-email concurrency → 409/serialization. TC-SIGNUP-005: SQLi in email → 400. TC-SIGNUP-007: email down → async queue, do not block signup.
