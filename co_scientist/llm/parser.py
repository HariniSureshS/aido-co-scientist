"""Parse structured JSON from LLM responses."""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json(text: str) -> dict[str, Any] | None:
    """Extract a JSON object from LLM response text.

    Handles common LLM output patterns:
    - Pure JSON
    - JSON wrapped in ```json ... ``` code blocks
    - JSON embedded in surrounding text
    """
    if not text or not text.strip():
        return None

    text = text.strip()

    # Try 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try 2: Extract from ```json ... ``` code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try 3: Find first { ... } block (greedy from first { to last })
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace : last_brace + 1])
        except json.JSONDecodeError:
            pass

    return None


def parse_decision(text: str) -> dict[str, Any]:
    """Parse an LLM response into a Decision-compatible dict.

    Expected JSON format:
    {
        "action": "select_model",
        "parameters": {"model_type": "xgboost", ...},
        "reasoning": "XGBoost is well-suited for ...",
        "confidence": 0.85,
        "fallback": "random_forest"
    }

    If parsing fails, returns a safe fallback decision.
    """
    parsed = extract_json(text)
    if parsed and "action" in parsed:
        return {
            "action": parsed["action"],
            "parameters": parsed.get("parameters", {}),
            "reasoning": parsed.get("reasoning", ""),
            "confidence": float(parsed.get("confidence", 0.5)),
            "fallback": parsed.get("fallback", ""),
        }

    # Fallback: could not parse — return a no-op
    return {
        "action": "no_op",
        "parameters": {},
        "reasoning": f"Failed to parse LLM response: {text[:200]}",
        "confidence": 0.0,
        "fallback": "",
    }


def parse_list(text: str) -> list[str]:
    """Extract a list of strings from LLM response.

    Handles JSON arrays, numbered lists, and bullet lists.
    """
    parsed = extract_json(text)
    if isinstance(parsed, list):
        return [str(item) for item in parsed]

    # Try parsing as JSON array directly
    text = text.strip()
    if text.startswith("["):
        try:
            items = json.loads(text)
            if isinstance(items, list):
                return [str(i) for i in items]
        except json.JSONDecodeError:
            pass

    # Fall back to line-based parsing
    lines = text.strip().split("\n")
    items = []
    for line in lines:
        line = line.strip()
        # Remove bullet/number prefixes
        line = re.sub(r"^[\d]+[.)]\s*", "", line)
        line = re.sub(r"^[-*•]\s*", "", line)
        line = line.strip()
        if line:
            items.append(line)
    return items
