"""Stage executor package: dispatch ``stage["type"]`` to the right module."""

from . import agent, deterministic

__all__ = ["agent", "deterministic"]
