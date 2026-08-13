"""Central configuration. All settings read from environment variables with sane defaults."""

import os

USE_HUGGINGFACE = os.getenv("USE_HF", "false").lower() in ("true", "1", "yes")
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))
MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "4"))
MAX_DOC_CHUNKS = int(os.getenv("MAX_DOC_CHUNKS", "50"))

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, friendly assistant. Answer concisely and clearly. "
    "If you don't know something, say so."
)