"""Conditional 2signal imports. Falls back to no-op if twosignal is not installed."""

from typing import Any

try:
    from twosignal import observe as _observe
    from twosignal.types import SpanType

    def observe(**kwargs: Any):
        return _observe(**kwargs)

except ImportError:
    import enum

    class SpanType(enum.Enum):
        AGENT = "AGENT"
        TOOL = "TOOL"
        LLM = "LLM"
        CHAIN = "CHAIN"
        RETRIEVAL = "RETRIEVAL"
        DELEGATION = "DELEGATION"
        GUARDRAIL = "GUARDRAIL"
        VOICE = "VOICE"
        HUMAN_HANDOFF = "HUMAN_HANDOFF"
        CUSTOM = "CUSTOM"

    def observe(**kwargs: Any):
        """No-op decorator when twosignal is not installed."""
        def decorator(func):
            return func
        return decorator
