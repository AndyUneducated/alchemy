BACKEND = "ollama"  # "ollama" | "openai" | "anthropic" | "gemini"

# Ollama: REST via urllib; OLLAMA_BASE_URL for debugging.
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:32b"

# OpenAI-compatible SDK (LM Studio, vLLM, etc. via base_url).
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_API_KEY = ""
OPENAI_MODEL = "gpt-4o-mini"

# Anthropic: official API only (no custom base_url in SDK).
ANTHROPIC_API_KEY = ""
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

# Gemini: google-genai SDK.
GEMINI_API_KEY = ""
GEMINI_MODEL = "gemini-2.5-flash"

if BACKEND == "openai":
    DEFAULT_MODEL = OPENAI_MODEL
elif BACKEND == "anthropic":
    DEFAULT_MODEL = ANTHROPIC_MODEL
elif BACKEND == "gemini":
    DEFAULT_MODEL = GEMINI_MODEL

TEMPERATURE = 0.7
MAX_TOKENS = 512

SUMMARY_MODEL = DEFAULT_MODEL
SUMMARY_MAX_TOKENS = 400
SUMMARY_TEMPERATURE = 0.2
