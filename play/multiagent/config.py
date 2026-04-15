BACKEND = "ollama"  # "ollama" | "openai" | "anthropic"

# Ollama
"""
Calls the REST API directly via urllib (no ollama SDK),
exposing BASE_URL for easy debugging and learning.
"""
BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "mistral:latest"

# OpenAI
"""
Uses the openai SDK, which accepts a base_url parameter,
so it can also point to any OpenAI-compatible service (LM Studio, vLLM, etc.).
"""
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_API_KEY = ""
OPENAI_MODEL = "gpt-4o-mini"

# Anthropic
"""
Uses the anthropic SDK, which has no custom base_url option;
always targets the official Anthropic API, so only key and model are needed.
"""
ANTHROPIC_API_KEY = ""
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

if BACKEND == "openai":
    DEFAULT_MODEL = OPENAI_MODEL
elif BACKEND == "anthropic":
    DEFAULT_MODEL = ANTHROPIC_MODEL

TEMPERATURE = 0.7
MAX_TOKENS = 512
