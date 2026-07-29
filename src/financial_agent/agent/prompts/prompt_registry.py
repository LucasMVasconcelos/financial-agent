"""Prompt version registry.

A tiny indirection so "which system prompt is active" is a one-line change
(and trivially A/B-testable or rollback-able) instead of a hunt-and-replace
across the codebase. Add `system_prompt_v2.py` alongside v1, register it
below, and flip `ACTIVE_SYSTEM_PROMPT_VERSION` when it's ready to ship.
"""

from __future__ import annotations

from financial_agent.agent.prompts.system_prompt_v1 import SYSTEM_PROMPT_V1

_REGISTRY: dict[str, str] = {
    "v1": SYSTEM_PROMPT_V1,
}

ACTIVE_SYSTEM_PROMPT_VERSION = "v1"


def get_active_system_prompt() -> str:
    return _REGISTRY[ACTIVE_SYSTEM_PROMPT_VERSION]
