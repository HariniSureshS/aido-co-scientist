"""Claude API client wrapper for the agent framework.

Provides a thin wrapper around the Anthropic Python SDK with:
- Cost tracking per call
- Budget enforcement (refuse calls when budget exhausted)
- Structured JSON output parsing
- Retry with exponential backoff on transient failures
- Graceful degradation (returns None on failure, never crashes the pipeline)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from co_scientist.llm.cost import CostTracker
from co_scientist.llm.parser import extract_json

logger = logging.getLogger(__name__)

# Default model — Sonnet for cost-efficiency; Opus only if explicitly requested
DEFAULT_MODEL = "claude-sonnet-4-20250514"

# Retry settings
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds — doubles each retry: 2, 4, 8


class ClaudeClient:
    """Wrapper around the Anthropic API for agent LLM calls.

    Usage:
        client = ClaudeClient(cost_tracker=tracker)
        response = client.ask(
            system_prompt="You are a data analyst...",
            user_message="Given this dataset profile: ...",
            agent_name="data_analyst",
        )
        # response is a dict (parsed JSON) or None on failure
    """

    def __init__(
        self,
        cost_tracker: CostTracker | None = None,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        request_timeout: float = 60.0,
    ):
        self.model = model
        self.cost_tracker = cost_tracker or CostTracker()
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._request_timeout = request_timeout
        self._client = None  # lazy init

    @property
    def available(self) -> bool:
        """Check if the API is configured (key present)."""
        return bool(self._api_key)

    def _get_client(self):
        """Lazy-initialize the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(
                    api_key=self._api_key,
                    timeout=self._request_timeout,
                )
            except ImportError:
                logger.warning("anthropic package not installed — LLM calls disabled")
                return None
            except Exception as e:
                logger.warning("Failed to initialize Anthropic client: %s", e)
                return None
        return self._client

    def ask(
        self,
        system_prompt: str,
        user_message: str,
        agent_name: str = "unknown",
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> dict[str, Any] | None:
        """Send a message to Claude and return parsed JSON response.

        Returns None if:
        - API key not configured
        - Budget exhausted
        - API call fails
        - Response can't be parsed as JSON
        """
        if not self.available:
            logger.debug("No API key — skipping LLM call for %s", agent_name)
            return None

        if not self.cost_tracker.can_afford():
            logger.warning("LLM budget exhausted — skipping call for %s", agent_name)
            return None

        client = self._get_client()
        if client is None:
            return None

        response = self._call_with_retry(
            client, system_prompt, user_message, agent_name, max_tokens, temperature
        )
        if response is None:
            return None

        # Extract text
        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        # Parse JSON
        parsed = extract_json(text)
        if parsed is None:
            logger.warning(
                "Could not parse JSON from %s response: %s",
                agent_name,
                text[:200],
            )
            # Return raw text wrapped in a dict so callers can still use it
            return {"raw_text": text, "_parse_failed": True}

        return parsed

    def ask_text(
        self,
        system_prompt: str,
        user_message: str,
        agent_name: str = "unknown",
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str | None:
        """Send a message and return raw text (not JSON).

        Used for report sections, interpretations, etc.
        """
        if not self.available:
            return None

        if not self.cost_tracker.can_afford():
            logger.warning("LLM budget exhausted — skipping text call for %s", agent_name)
            return None

        client = self._get_client()
        if client is None:
            return None

        response = self._call_with_retry(
            client, system_prompt, user_message, agent_name, max_tokens, temperature
        )
        if response is None:
            return None

        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        return text if text else None

    def _call_with_retry(
        self,
        client: Any,
        system_prompt: str,
        user_message: str,
        agent_name: str,
        max_tokens: int,
        temperature: float,
    ) -> Any:
        """Call the API with retry + exponential backoff on transient failures."""
        last_error = None
        for attempt in range(MAX_RETRIES):
            start = time.time()
            try:
                response = client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                )
                latency = time.time() - start

                # Track cost
                self.cost_tracker.record(
                    model=self.model,
                    agent=agent_name,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    latency_seconds=latency,
                )
                return response

            except Exception as e:
                last_error = e
                latency = time.time() - start
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "LLM call failed for %s (attempt %d/%d, %.1fs): %s — retrying in %.0fs",
                        agent_name, attempt + 1, MAX_RETRIES, latency, e, delay,
                    )
                    time.sleep(delay)
                else:
                    logger.warning(
                        "LLM call failed for %s after %d attempts: %s",
                        agent_name, MAX_RETRIES, e,
                    )

        return None
