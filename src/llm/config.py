"""Configuration for LLM clients."""

from pydantic import Field
from pydantic_settings import BaseSettings


class LLMConfig(BaseSettings):
    """Configuration for LLM client.

    Settings can be provided via environment variables with OLLAMA_ prefix.
    """

    host: str = Field(default="http://localhost:11434", description="Ollama host URL")
    model: str = Field(default="qwen2.5:7b", description="Model name")
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )
    timeout: int = Field(default=120, gt=0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=1, description="Maximum retry attempts")

    # Cache settings
    cache_enabled: bool = Field(default=True, description="Enable response caching")
    cache_dir: str = Field(default=".cache/llm", description="Cache directory")

    model_config = {
        "env_prefix": "OLLAMA_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",  # Ignore extra environment variables
    }


# Available models for comparison experiments
AVAILABLE_MODELS = [
    "qwen2.5:7b",  # Principal - fast, good reasoning
    "qwen2.5:14b",  # Larger Qwen - better quality, slower
    "llama3.1:8b",  # Meta's Llama - comparison
    "gemma2:9b",  # Google's Gemma - comparison
]
