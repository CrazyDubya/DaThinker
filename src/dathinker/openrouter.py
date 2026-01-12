"""OpenRouter API client for LLM interactions.

Security notes:
- API keys are never logged or printed
- HTTP headers are never logged (contain Authorization)
- Use ~/.config/dathinker/config.toml for secure key storage
"""

import asyncio
import os
import ssl
import httpx
import logging
from typing import AsyncIterator
from pydantic import BaseModel


# Configure logging - NEVER log headers or API keys
logger = logging.getLogger(__name__)


class Message(BaseModel):
    """A chat message."""
    role: str  # "system", "user", or "assistant"
    content: str


class OpenRouterClient:
    """Async client for OpenRouter API.

    Security: This client handles sensitive API keys.
    - Never log self.headers (contains Authorization)
    - Never print self.api_key
    - Use get_api_key_preview() for debugging
    """

    BASE_URL = "https://openrouter.ai/api/v1"

    # Cost-effective models - cheap but good quality (paid tier)
    MODELS = {
        # Fast: For quick meta-decisions, agent selection (~$0.10/M tokens)
        "fast": "google/gemma-2-9b-it",
        # Balanced: Good quality for agent responses (~$0.07/M input, $0.07/M output)
        "balanced": "meta-llama/llama-3.1-8b-instruct",
        # Reasoning: Better model for complex thinking (~$0.15/M input, $0.60/M output)
        "reasoning": "openai/gpt-4o-mini",
    }

    MAX_RETRIES = 4
    RETRY_DELAYS = [2, 4, 8, 16]  # Exponential backoff

    def __init__(self, api_key: str | None = None):
        # Try to load from config first, then environment
        if api_key is None:
            try:
                from .config import get_api_key
                api_key = get_api_key()
            except ImportError:
                pass

        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found. Set it via:\n"
                "  1. ~/.config/dathinker/config.toml (recommended)\n"
                "  2. OPENROUTER_API_KEY environment variable"
            )

        # SECURITY: Never log these headers - they contain the API key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/dathinker",
            "X-Title": "DaThinker",
        }

    def get_api_key_preview(self) -> str:
        """Get a safe preview of the API key for debugging (never log the full key)."""
        if not self.api_key:
            return "(not set)"
        if len(self.api_key) > 8:
            return f"{self.api_key[:4]}...{self.api_key[-4:]}"
        return "****"

    def _create_client(self) -> httpx.AsyncClient:
        """Create an httpx client with SSL verification disabled for environments with cert issues."""
        return httpx.AsyncClient(
            timeout=90.0,
            verify=False,  # Disable SSL verification for cert issues in some environments
        )

    async def chat(
        self,
        messages: list[Message],
        model: str = "balanced",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Send a chat completion request with retry logic."""

        model_id = self.MODELS.get(model, model)

        payload = {
            "model": model_id,
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                async with self._create_client() as client:
                    response = await client.post(
                        f"{self.BASE_URL}/chat/completions",
                        headers=self.headers,
                        json=payload,
                    )

                    # Handle rate limiting and service unavailable
                    if response.status_code in (429, 503):
                        try:
                            data = response.json()
                            error_msg = data.get("error", {}).get("message", f"HTTP {response.status_code}")
                        except Exception:
                            error_msg = f"HTTP {response.status_code}"
                        if attempt < self.MAX_RETRIES - 1:
                            await asyncio.sleep(self.RETRY_DELAYS[attempt])
                            continue
                        raise RuntimeError(f"Service error after {self.MAX_RETRIES} attempts: {error_msg}")

                    response.raise_for_status()
                    data = response.json()

                    return data["choices"][0]["message"]["content"]

            except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ProxyError) as e:
                last_error = e
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RETRY_DELAYS[attempt])
                    continue
                raise RuntimeError(f"Connection failed after {self.MAX_RETRIES} attempts: {e}")

        raise last_error or RuntimeError("Unknown error")

    async def chat_stream(
        self,
        messages: list[Message],
        model: str = "balanced",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """Stream a chat completion response."""

        model_id = self.MODELS.get(model, model)

        payload = {
            "model": model_id,
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        async with self._create_client() as client:
            async with client.stream(
                "POST",
                f"{self.BASE_URL}/chat/completions",
                headers=self.headers,
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            import json
                            chunk = json.loads(data)
                            if delta := chunk["choices"][0].get("delta", {}).get("content"):
                                yield delta
                        except (json.JSONDecodeError, KeyError):
                            continue
