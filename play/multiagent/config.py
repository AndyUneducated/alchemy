BACKEND = "ollama"  # "ollama" | "openai" | "anthropic"

# Ollama
BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "mistral:latest"

# OpenAI-compatible — BASE_URL can point to any OpenAI-protocol service (LM Studio, vLLM, etc.)
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_API_KEY = ""
OPENAI_MODEL = "gpt-4o-mini"

# Anthropic
ANTHROPIC_API_KEY = ""
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

if BACKEND == "openai":
    DEFAULT_MODEL = OPENAI_MODEL
elif BACKEND == "anthropic":
    DEFAULT_MODEL = ANTHROPIC_MODEL

TEMPERATURE = 0.7
MAX_TOKENS = 512
