"""LLM client for Ollama and OpenAI integration.

Provides a unified interface for querying LLMs via Ollama or OpenAI,
with support for caching, retries, and response parsing.
"""

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from src.llm.config import LLMConfig

# Type alias for interaction callback
LLMInteractionCallback = Callable[
    [str, "LLMResponse", str | None, dict | None], None
] | None


class BudgetExhaustedError(Exception):
    """Raised when LLM queries are blocked due to budget exhaustion."""
    pass


class LLMResponse(BaseModel):
    """Response from LLM query."""

    content: str
    model: str
    latency_ms: float
    tokens_generated: int = 0  # Completion tokens (for backward compatibility)
    prompt_tokens: int = 0  # Input tokens
    from_cache: bool = False
    raw_response: dict[str, Any] = Field(default_factory=dict)


class LLMStats(BaseModel):
    """Statistics for LLM client."""

    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_tokens: int = 0  # Total tokens (prompt + completion)
    prompt_tokens: int = 0  # Input tokens
    completion_tokens: int = 0  # Output tokens
    total_latency_ms: float = 0.0
    errors: int = 0

    @property
    def cache_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Get average latency."""
        return self.total_latency_ms / self.total_queries if self.total_queries > 0 else 0.0


class OllamaClient:
    """Client for Ollama local LLM.

    Features:
    - SQLite-based response caching
    - Automatic retries with exponential backoff
    - Response statistics tracking
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        interaction_callback: LLMInteractionCallback = None,
    ) -> None:
        """Initialize Ollama client.

        Args:
            config: LLM configuration. If None, uses defaults from environment.
            interaction_callback: Optional callback for logging LLM interactions.
                Signature: (prompt, response, agent_name, context) -> None
        """
        self.config = config or LLMConfig()
        self.stats = LLMStats()
        self._cache_db: sqlite3.Connection | None = None
        self._interaction_callback = interaction_callback

        if self.config.cache_enabled:
            self._init_cache()

    def _init_cache(self) -> None:
        """Initialize SQLite cache database."""
        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        db_path = cache_dir / "response_cache.db"
        self._cache_db = sqlite3.connect(str(db_path), check_same_thread=False)

        # Create cache table
        self._cache_db.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                response TEXT,
                model TEXT,
                timestamp REAL
            )
        """)
        self._cache_db.commit()

    def _get_cache_key(self, prompt: str, model: str) -> str:
        """Generate cache key from prompt and model.

        Args:
            prompt: The prompt text
            model: Model name

        Returns:
            Cache key hash
        """
        content = f"{model}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _check_cache(self, key: str) -> LLMResponse | None:
        """Check if response is cached.

        Args:
            key: Cache key

        Returns:
            Cached response or None
        """
        if not self._cache_db:
            return None

        cursor = self._cache_db.execute(
            "SELECT response, model FROM cache WHERE key = ?", (key,)
        )
        row = cursor.fetchone()

        if row:
            self.stats.cache_hits += 1
            return LLMResponse(
                content=row[0],
                model=row[1],
                latency_ms=0.0,
                from_cache=True,
            )

        self.stats.cache_misses += 1
        return None

    def _save_cache(self, key: str, response: LLMResponse) -> None:
        """Save response to cache.

        Args:
            key: Cache key
            response: Response to cache
        """
        if not self._cache_db:
            return

        self._cache_db.execute(
            "INSERT OR REPLACE INTO cache (key, response, model, timestamp) VALUES (?, ?, ?, ?)",
            (key, response.content, response.model, time.time()),
        )
        self._cache_db.commit()

    def query(
        self,
        prompt: str,
        temperature: float | None = None,
        use_cache: bool = True,
        json_schema: dict | None = None,
        agent_name: str | None = None,
        context: dict | None = None,
    ) -> LLMResponse:
        """Query the LLM.

        Args:
            prompt: The prompt to send
            temperature: Override temperature (optional)
            use_cache: Whether to use caching
            json_schema: Optional JSON schema for structured output (Ollama format parameter)
            agent_name: Name of the calling agent (for logging)
            context: Additional context for logging

        Returns:
            LLM response
        """
        self.stats.total_queries += 1

        # Check cache first (include schema in cache key if provided)
        cache_key_extra = json.dumps(json_schema, sort_keys=True) if json_schema else ""
        if use_cache and self.config.cache_enabled:
            cache_key = self._get_cache_key(prompt + cache_key_extra, self.config.model)
            cached = self._check_cache(cache_key)
            if cached:
                # Log cached response if callback provided
                if self._interaction_callback:
                    self._interaction_callback(prompt, cached, agent_name, context)
                return cached

        # Query Ollama with retries
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = self._query_ollama(prompt, temperature, json_schema)

                # Save to cache
                if use_cache and self.config.cache_enabled:
                    self._save_cache(cache_key, response)

                # Update stats
                self.stats.prompt_tokens += response.prompt_tokens
                self.stats.completion_tokens += response.tokens_generated
                self.stats.total_tokens += response.prompt_tokens + response.tokens_generated
                self.stats.total_latency_ms += response.latency_ms

                # Log interaction if callback provided
                if self._interaction_callback:
                    self._interaction_callback(prompt, response, agent_name, context)

                return response

            except Exception as e:
                last_error = e
                self.stats.errors += 1

                # Exponential backoff
                if attempt < self.config.max_retries - 1:
                    wait_time = 2**attempt
                    time.sleep(wait_time)

        # All retries failed
        raise RuntimeError(f"LLM query failed after {self.config.max_retries} attempts: {last_error}")

    def _query_ollama(
        self,
        prompt: str,
        temperature: float | None = None,
        json_schema: dict | None = None,
    ) -> LLMResponse:
        """Execute query to Ollama.

        Args:
            prompt: The prompt
            temperature: Override temperature
            json_schema: Optional JSON schema for structured output

        Returns:
            LLM response
        """
        import ollama

        client = ollama.Client(host=self.config.host)
        temp = temperature if temperature is not None else self.config.temperature

        start_time = time.time()

        # Build request kwargs
        kwargs = {
            "model": self.config.model,
            "prompt": prompt,
            "options": {"temperature": temp},
        }

        # Add structured output format if schema provided
        if json_schema:
            kwargs["format"] = json_schema

        response = client.generate(**kwargs)

        latency_ms = (time.time() - start_time) * 1000

        # Handle both dict and object response types
        if hasattr(response, "response"):
            content = response.response
            eval_count = getattr(response, "eval_count", 0)
            raw = {"model": self.config.model, "response": content}
        else:
            content = response.get("response", "")
            eval_count = response.get("eval_count", 0)
            raw = dict(response) if isinstance(response, dict) else {}

        return LLMResponse(
            content=content,
            model=self.config.model,
            latency_ms=latency_ms,
            tokens_generated=eval_count,
            raw_response=raw,
        )

    def check_connection(self) -> bool:
        """Check if Ollama is available.

        Returns:
            True if connection successful
        """
        try:
            import ollama

            client = ollama.Client(host=self.config.host)
            client.list()
            return True
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """List available models.

        Returns:
            List of model names
        """
        try:
            import ollama

            client = ollama.Client(host=self.config.host)
            models = client.list()
            return [m.get("name", "") for m in models.get("models", [])]
        except Exception:
            return []

    def get_stats(self) -> LLMStats:
        """Get client statistics.

        Returns:
            Statistics object
        """
        return self.stats

    def clear_cache(self) -> None:
        """Clear the response cache."""
        if self._cache_db:
            self._cache_db.execute("DELETE FROM cache")
            self._cache_db.commit()

    def close(self) -> None:
        """Close the client and release resources."""
        if self._cache_db:
            self._cache_db.close()
            self._cache_db = None


class OpenAIClient:
    """Client for OpenAI API (GPT-4o, GPT-5, etc).

    Features:
    - SQLite-based response caching
    - Automatic retries with exponential backoff
    - Response statistics tracking
    - GPT-5 reasoning_effort support

    GPT-5 Models:
    - gpt-5, gpt-5.1, gpt-5.2: Main models with reasoning
    - gpt-5-mini: Smaller, faster model
    - gpt-5.2-codex: Optimized for coding tasks
    """

    # GPT-5 models that support reasoning_effort parameter
    GPT5_MODELS = {"gpt-5", "gpt-5.1", "gpt-5.2", "gpt-5-mini", "gpt-5.2-codex"}

    # Codex models that ONLY work with the Responses API (v1/responses)
    RESPONSES_API_MODELS = {"gpt-5.1-codex", "gpt-5.1-codex-mini", "gpt-5.1-codex-max", "gpt-5.2-codex"}

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        cache_enabled: bool = True,
        cache_dir: str = ".cache/llm",
        interaction_callback: LLMInteractionCallback = None,
    ) -> None:
        """Initialize OpenAI client.

        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            model: Model name (default: gpt-4o). GPT-5 models: gpt-5, gpt-5.1, gpt-5.2, gpt-5-mini
            temperature: Sampling temperature
            reasoning_effort: For GPT-5 models only. Controls reasoning depth.
                Options: none, minimal, low, medium, high, xhigh (xhigh only for gpt-5.2)
                Default: none for gpt-5.1/5.2, medium for gpt-5
            cache_enabled: Whether to enable caching
            cache_dir: Cache directory path
            interaction_callback: Optional callback for logging LLM interactions.
                Signature: (prompt, response, agent_name, context) -> None
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.cache_enabled = cache_enabled
        self.cache_dir = cache_dir
        self.stats = LLMStats()
        self._cache_db: sqlite3.Connection | None = None
        self._interaction_callback = interaction_callback

        # Create OpenAI client once (reuses HTTP connection pool)
        from openai import OpenAI
        self._client = OpenAI(api_key=self.api_key, timeout=120.0)

        # Budget exhaustion flag - when True, all queries raise BudgetExhaustedError
        self._budget_exhausted = False

        if self.cache_enabled:
            self._init_cache()

    def stop_all_queries(self) -> None:
        """Stop all future LLM queries by setting budget exhaustion flag.

        Once called, all subsequent query() calls will raise BudgetExhaustedError.
        This is used to enforce strict token budget limits.
        """
        self._budget_exhausted = True

    def resume_queries(self) -> None:
        """Resume LLM queries by clearing the budget exhaustion flag."""
        self._budget_exhausted = False

    def _init_cache(self) -> None:
        """Initialize SQLite cache database."""
        cache_dir = Path(self.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        db_path = cache_dir / "openai_cache.db"
        self._cache_db = sqlite3.connect(str(db_path), check_same_thread=False)

        self._cache_db.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                response TEXT,
                model TEXT,
                timestamp REAL
            )
        """)
        self._cache_db.commit()

    def _get_cache_key(self, prompt: str, model: str) -> str:
        """Generate cache key from prompt and model."""
        content = f"{model}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _check_cache(self, key: str) -> LLMResponse | None:
        """Check if response is cached."""
        if not self._cache_db:
            return None

        cursor = self._cache_db.execute(
            "SELECT response, model FROM cache WHERE key = ?", (key,)
        )
        row = cursor.fetchone()

        if row:
            self.stats.cache_hits += 1
            return LLMResponse(
                content=row[0],
                model=row[1],
                latency_ms=0.0,
                from_cache=True,
            )

        self.stats.cache_misses += 1
        return None

    def _save_cache(self, key: str, response: LLMResponse) -> None:
        """Save response to cache."""
        if not self._cache_db:
            return

        self._cache_db.execute(
            "INSERT OR REPLACE INTO cache (key, response, model, timestamp) VALUES (?, ?, ?, ?)",
            (key, response.content, response.model, time.time()),
        )
        self._cache_db.commit()

    def query(
        self,
        prompt: str,
        temperature: float | None = None,
        use_cache: bool = True,
        json_schema: dict | None = None,
        agent_name: str | None = None,
        context: dict | None = None,
    ) -> LLMResponse:
        """Query the LLM.

        Args:
            prompt: The prompt to send
            temperature: Override temperature (optional)
            use_cache: Whether to use caching
            json_schema: Optional JSON schema (ignored for OpenAI, use response_format)
            agent_name: Name of the calling agent (for logging)
            context: Additional context for logging

        Returns:
            LLM response

        Raises:
            BudgetExhaustedError: If budget is exhausted and stop_all_queries() was called.
        """
        # Check budget exhaustion flag first (before any API calls)
        if self._budget_exhausted:
            raise BudgetExhaustedError(
                "LLM queries blocked: token budget exhausted. "
                f"Agent: {agent_name or 'unknown'}"
            )

        self.stats.total_queries += 1

        # Check cache first
        if use_cache and self.cache_enabled:
            cache_key = self._get_cache_key(prompt, self.model)
            cached = self._check_cache(cache_key)
            if cached:
                # Log cached response if callback provided
                if self._interaction_callback:
                    self._interaction_callback(prompt, cached, agent_name, context)
                return cached

        # Query OpenAI with retries
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            try:
                response = self._query_openai(prompt, temperature)

                # Save to cache
                if use_cache and self.cache_enabled:
                    self._save_cache(cache_key, response)

                # Update stats
                self.stats.prompt_tokens += response.prompt_tokens
                self.stats.completion_tokens += response.tokens_generated
                self.stats.total_tokens += response.prompt_tokens + response.tokens_generated
                self.stats.total_latency_ms += response.latency_ms

                # Log interaction if callback provided
                if self._interaction_callback:
                    self._interaction_callback(prompt, response, agent_name, context)

                return response

            except Exception as e:
                last_error = e
                self.stats.errors += 1

                # Exponential backoff
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    time.sleep(wait_time)

        raise RuntimeError(f"OpenAI query failed after {max_retries} attempts: {last_error}")

    def _is_gpt5_model(self) -> bool:
        """Check if current model is a GPT-5 variant."""
        model_lower = self.model.lower()
        return any(model_lower.startswith(m) for m in self.GPT5_MODELS)

    def _requires_responses_api(self) -> bool:
        """Check if current model requires the Responses API (v1/responses)."""
        model_lower = self.model.lower()
        return any(model_lower.startswith(m) for m in self.RESPONSES_API_MODELS)

    def _query_openai(
        self,
        prompt: str,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Execute query to OpenAI.

        For Codex models (gpt-5.1-codex-*): uses Responses API (v1/responses).
        For GPT-5 models: uses reasoning_effort parameter with Chat Completions.
        For GPT-4 models: uses standard chat completions.
        """
        # Reuse the client created in __init__ (connection pooling)
        client = self._client
        temp = temperature if temperature is not None else self.temperature

        start_time = time.time()

        # Codex models require the Responses API
        if self._requires_responses_api():
            return self._query_responses_api(client, prompt, start_time)

        # Build request parameters for Chat Completions API
        request_params = {
            "model": self.model,
            "messages": [
                {"role": "developer", "content": "You are an expert algorithm designer."},
                {"role": "user", "content": prompt},
            ],
        }

        # GPT-5 models: use reasoning_effort instead of temperature
        if self._is_gpt5_model():
            if self.reasoning_effort:
                request_params["reasoning_effort"] = self.reasoning_effort
            # GPT-5 models only support temperature=1.0
            request_params["temperature"] = 1.0
        else:
            # GPT-4 models: use temperature
            request_params["temperature"] = temp

        response = client.chat.completions.create(**request_params)

        latency_ms = (time.time() - start_time) * 1000

        content = response.choices[0].message.content or ""
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0

        return LLMResponse(
            content=content,
            model=self.model,
            latency_ms=latency_ms,
            tokens_generated=completion_tokens,
            prompt_tokens=prompt_tokens,
            raw_response={"id": response.id},
        )

    def _query_responses_api(
        self,
        client,
        prompt: str,
        start_time: float,
    ) -> LLMResponse:
        """Execute query using the Responses API (v1/responses).

        Required for Codex models (gpt-5.1-codex-*, gpt-5.2-codex).
        """
        # Build request for Responses API
        request_params = {
            "model": self.model,
            "input": [
                {"role": "developer", "content": "You are an expert algorithm designer."},
                {"role": "user", "content": prompt},
            ],
        }

        # Add reasoning_effort if specified
        if self.reasoning_effort:
            request_params["reasoning"] = {"effort": self.reasoning_effort}

        response = client.responses.create(**request_params)

        latency_ms = (time.time() - start_time) * 1000

        # Extract content from response
        content = ""
        if response.output:
            for item in response.output:
                if hasattr(item, "content") and item.content:
                    for content_item in item.content:
                        if hasattr(content_item, "text"):
                            content += content_item.text

        # Get usage stats
        completion_tokens = response.usage.output_tokens if response.usage else 0
        prompt_tokens = response.usage.input_tokens if response.usage else 0

        return LLMResponse(
            content=content,
            model=self.model,
            latency_ms=latency_ms,
            tokens_generated=completion_tokens,
            prompt_tokens=prompt_tokens,
            raw_response={"id": response.id},
        )

    def get_stats(self) -> LLMStats:
        """Get client statistics."""
        return self.stats

    def clear_cache(self) -> None:
        """Clear the response cache."""
        if self._cache_db:
            self._cache_db.execute("DELETE FROM cache")
            self._cache_db.commit()

    def close(self) -> None:
        """Close the client and release resources."""
        if self._cache_db:
            self._cache_db.close()
            self._cache_db = None
