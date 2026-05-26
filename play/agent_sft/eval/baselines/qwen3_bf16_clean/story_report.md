# qwen3 bf16 clean - story report

## Runs

|model|seed_runs|overall_nudge_fire_rate_mean|in_distribution_mean|held_out_mean|
|---|---:|---:|---:|---:|
|`ollama:agent-sft-qwen-3`|3|0.3833|0.4872|0.1905|
|`ollama:qwen3.5:9b`|3|0.3500|0.4359|0.1905|

## Per Run Snapshot

|run_id|model|overall|example|panel|tool_chain|code_review|
|---|---|---:|---:|---:|---:|---:|
|`20260526-105121-6a57475e`|`ollama:qwen3.5:9b@seed=0`|0.3500|0.3333|0.0000|0.4000|0.5000|
|`20260526-110816-60ea3168`|`ollama:qwen3.5:9b@seed=1`|0.3500|0.6667|0.0000|0.2000|0.5000|
|`20260526-112404-a81ef8c1`|`ollama:qwen3.5:9b@seed=2`|0.3500|0.3333|0.0000|0.6000|0.3750|
|`20260526-113758-126aa821`|`ollama:agent-sft-qwen-3@seed=0`|0.4000|0.3333|0.0000|0.4000|0.6250|
|`20260526-115406-92f23dc4`|`ollama:agent-sft-qwen-3@seed=1`|0.4000|0.3333|0.0000|0.4000|0.6250|
|`20260526-121026-6d84444e`|`ollama:agent-sft-qwen-3@seed=2`|0.3500|0.3333|0.2500|0.4000|0.3750|

- in-distribution = weighted(`tool_chain`,`code_review`) by require_tool turns (5,8).
- held-out = weighted(`example`,`panel`) by require_tool turns (3,4).
- Note: current `agent-sft-qwen-3` is placeholder tag (`FROM qwen3.5:9b`) because bf16 GGUF still garbles in ollama.
