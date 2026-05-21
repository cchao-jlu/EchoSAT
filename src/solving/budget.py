from __future__ import annotations

from typing import Any


def apply_rollout_budget(
    params: dict[str, Any],
    budget_type: str = "cpu_time",
    cpu_lim: float | int | None = None,
    conflicts: int | None = None,
) -> dict[str, Any]:
    params = dict(params)
    if cpu_lim is not None:
        params["cpu-lim"] = cpu_lim
    if budget_type == "cpu_time":
        return params
    if budget_type == "conflicts":
        if conflicts is None or int(conflicts) <= 0:
            raise ValueError("conflict rollout budget requires a positive conflicts value")
        params["conf-lim"] = int(conflicts)
        return params
    raise ValueError(f"Unknown rollout budget type {budget_type}")
