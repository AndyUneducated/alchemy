import os

# All settings are env-driven with code-level fallback defaults (plan choice A).
# Override any value via the listed env var without editing this file.

BACKEND = os.environ.get("AGENT_ENGINE_BACKEND", "ollama")  # "ollama" | "openai" | "anthropic" | "gemini"

# Ollama: REST via urllib; OLLAMA_BASE_URL for debugging.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
# AGENT_ENGINE_MODEL env override — agent_sft Phase 2 pins 7B for triple mining
# without touching scenario YAMLs (plan §Decisions); other workflows still get qwen3.6:27b.
DEFAULT_MODEL = os.environ.get("AGENT_ENGINE_MODEL", "qwen3.6:27b")

# OpenAI-compatible SDK (LM Studio, vLLM, etc. via base_url).
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# Anthropic: official API only (no custom base_url in SDK).
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

# Gemini: google-genai SDK.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Only let BACKEND drive DEFAULT_MODEL when AGENT_ENGINE_MODEL was NOT explicitly set.
# This keeps agent_sft's `AGENT_ENGINE_MODEL=qwen3.5:9b` (BACKEND still "ollama") working.
if "AGENT_ENGINE_MODEL" not in os.environ:
    if BACKEND == "openai":
        DEFAULT_MODEL = OPENAI_MODEL
    elif BACKEND == "anthropic":
        DEFAULT_MODEL = ANTHROPIC_MODEL
    elif BACKEND == "gemini":
        DEFAULT_MODEL = GEMINI_MODEL

TEMPERATURE = float(os.environ.get("AGENT_ENGINE_TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.environ.get("AGENT_ENGINE_MAX_TOKENS", "512"))

SUMMARY_MODEL = os.environ.get("AGENT_ENGINE_SUMMARY_MODEL", DEFAULT_MODEL)
SUMMARY_MAX_TOKENS = int(os.environ.get("AGENT_ENGINE_SUMMARY_MAX_TOKENS", "400"))
SUMMARY_TEMPERATURE = float(os.environ.get("AGENT_ENGINE_SUMMARY_TEMPERATURE", "0.2"))
