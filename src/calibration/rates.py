"""Fail-closed RBI rate policy helpers."""

from __future__ import annotations

import json
from pathlib import Path


def load_rate_policy(path: str | Path = "configs/rate_policy.json") -> dict:
    policy = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"approved", "decimal_scale", "as_of_alignment", "day_count"}
    missing = required.difference(policy)
    if missing:
        raise ValueError(f"rate policy missing fields: {sorted(missing)}")
    if policy["day_count"] <= 0 or policy["decimal_scale"] <= 0:
        raise ValueError("rate policy scale and day_count must be positive")
    return policy


def source_value_to_decimal(source_value: float, policy: dict) -> float:
    """Convert a source value only after the policy has been explicitly approved."""
    if not policy.get("approved", False):
        raise RuntimeError("RBI rate policy is not approved; refusing to create a model rate")
    return float(source_value) * float(policy["decimal_scale"])
