from __future__ import annotations

from copy import copy
import math

import pandas as pd
import torch
from torch_geometric.data import HeteroData


GLOBAL_STATE_DIM = 6
EVENT_VAR_STATE_DIM = 5
EVENT_VAR_STATE_DIM_ENHANCED = 20


def result_to_code(result: str | None) -> float:
    if result == "SATISFIABLE":
        return 1.0
    if result == "UNSATISFIABLE":
        return -1.0
    return 0.0


def safe_log1p_stat(stats: dict, key: str) -> float:
    value = stats.get(key, 0.0)
    if value is None or pd.isna(value):
        value = 0.0
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        value = 0.0
    return float(torch.log1p(torch.tensor(value)).item())


def solver_stats_to_global_state(stats: dict | pd.Series) -> torch.Tensor:
    if isinstance(stats, pd.Series):
        stats = stats.to_dict()

    return torch.tensor(
        [
            safe_log1p_stat(stats, "decisions"),
            safe_log1p_stat(stats, "conflicts"),
            safe_log1p_stat(stats, "propagations"),
            safe_log1p_stat(stats, "restarts"),
            safe_log1p_stat(stats, "CPU time"),
            result_to_code(stats.get("Result")),
        ],
        dtype=torch.float32,
    )


def attach_global_state(
    data: HeteroData,
    stats: dict | pd.Series,
    global_state_dim: int,
) -> HeteroData:
    data = copy(data)
    global_state = solver_stats_to_global_state(stats)
    if global_state_dim != global_state.shape[0]:
        raise ValueError(
            f"feedback refinement expected global_state_dim={global_state.shape[0]}, "
            f"got model.global_state_dim={global_state_dim}"
        )
    data.global_state = global_state
    return data


def attach_global_state_batch(
    data_list: list[HeteroData],
    solver_stats: pd.DataFrame,
    global_state_dim: int,
) -> list[HeteroData]:
    stats_by_cnf = {
        int(cnf_id): group.iloc[0]
        for cnf_id, group in solver_stats.groupby("cnf_id", sort=False)
    }

    updated = []
    for data in data_list:
        cnf_id = int(data.cnf_id.item())
        stats = stats_by_cnf.get(cnf_id)
        if stats is None:
            updated.append(copy(data))
            continue
        updated.append(attach_global_state(data, stats=stats, global_state_dim=global_state_dim))
    return updated


def _event_values(stats: dict | pd.Series, key: str, num_vars: int) -> torch.Tensor:
    if isinstance(stats, pd.Series):
        stats = stats.to_dict()
    values = stats.get(key)
    if values is None or (isinstance(values, float) and pd.isna(values)):
        return torch.zeros(num_vars, dtype=torch.float32)
    values = torch.tensor(values, dtype=torch.float32)
    if values.numel() < num_vars:
        values = torch.cat([values, torch.zeros(num_vars - values.numel(), dtype=torch.float32)])
    elif values.numel() > num_vars:
        values = values[:num_vars]
    values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).clamp_min_(0.0)
    return torch.nan_to_num(torch.log1p(values), nan=0.0, posinf=20.0, neginf=0.0).clamp_(0.0, 20.0)


def event_state_dim(feature_mode: str = "legacy") -> int:
    if feature_mode == "legacy":
        return EVENT_VAR_STATE_DIM
    if feature_mode == "enhanced":
        return EVENT_VAR_STATE_DIM_ENHANCED
    raise ValueError(f"Unknown event state feature mode {feature_mode}")


def _event_raw_values(stats: dict | pd.Series, key: str, num_vars: int) -> torch.Tensor:
    if isinstance(stats, pd.Series):
        stats = stats.to_dict()
    values = stats.get(key)
    if values is None or (isinstance(values, float) and pd.isna(values)):
        return torch.zeros(num_vars, dtype=torch.float32)
    values = torch.tensor(values, dtype=torch.float32)
    if values.numel() < num_vars:
        values = torch.cat([values, torch.zeros(num_vars - values.numel(), dtype=torch.float32)])
    elif values.numel() > num_vars:
        values = values[:num_vars]
    return torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).clamp_min_(0.0)


def _rate_values(values: torch.Tensor) -> torch.Tensor:
    total = values.sum().clamp_min(1.0)
    return (values / total).clamp_(0.0, 1.0)


def _rank_values(values: torch.Tensor) -> torch.Tensor:
    if values.numel() <= 1 or torch.all(values == values[0]):
        return torch.zeros_like(values)
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(values)
    ranks[order] = torch.arange(values.numel(), dtype=torch.float32)
    return ranks / float(values.numel() - 1)


def solver_stats_to_var_event_state(
    stats: dict | pd.Series,
    num_vars: int,
    feature_mode: str = "legacy",
) -> torch.Tensor:
    """Encode solver event traces as one state vector per SAT variable."""
    keys = [
        "event_var_decisions",
        "event_var_propagations",
        "event_var_conflict_lits",
        "event_var_learnt_lits",
        "event_var_activity",
    ]
    if feature_mode == "legacy":
        return torch.stack(
            [_event_values(stats, key, num_vars) for key in keys],
            dim=1,
        )
    if feature_mode != "enhanced":
        raise ValueError(f"Unknown event state feature mode {feature_mode}")

    raw_values = [_event_raw_values(stats, key, num_vars) for key in keys]
    log_values = [torch.log1p(values).clamp_(0.0, 20.0) for values in raw_values]
    delta_log_values = log_values
    rate_values = [_rate_values(values) for values in raw_values]
    rank_values = [_rank_values(values) for values in raw_values]
    return torch.stack(
        [
            *log_values,
            *delta_log_values,
            *rate_values,
            *rank_values,
        ],
        dim=1,
    )


def attach_var_event_state(
    data: HeteroData,
    stats: dict | pd.Series,
    var_state_dim: int,
    momentum: float = 0.0,
    feature_mode: str = "legacy",
) -> HeteroData:
    data = copy(data)
    num_vars = int(data["lit"].num_nodes // 2)
    var_state = solver_stats_to_var_event_state(stats, num_vars=num_vars, feature_mode=feature_mode)
    if var_state_dim != var_state.shape[1]:
        raise ValueError(
            f"event feedback expected var_state_dim={var_state.shape[1]}, "
            f"got model.var_state_dim={var_state_dim}"
        )
    if hasattr(data["var"], "event_memory") and momentum > 0.0:
        prev_state = data["var"].event_memory.to(dtype=torch.float32)
        if prev_state.shape == var_state.shape:
            var_state = momentum * prev_state + (1.0 - momentum) * var_state
    elif hasattr(data["var"], "event_state") and momentum > 0.0:
        prev_state = data["var"].event_state.to(dtype=torch.float32)
        if prev_state.shape == var_state.shape:
            var_state = momentum * prev_state + (1.0 - momentum) * var_state
    var_state = torch.nan_to_num(var_state, nan=0.0, posinf=20.0, neginf=0.0).clamp_(0.0, 20.0)
    data["var"].num_nodes = num_vars
    data["var"].event_state = var_state
    data["var"].event_memory = var_state
    return data


def attach_var_event_state_batch(
    data_list: list[HeteroData],
    solver_stats: pd.DataFrame,
    var_state_dim: int,
    momentum: float = 0.0,
    feature_mode: str = "legacy",
) -> list[HeteroData]:
    stats_by_cnf = {
        int(cnf_id): group.iloc[0]
        for cnf_id, group in solver_stats.groupby("cnf_id", sort=False)
    }

    updated = []
    for data in data_list:
        cnf_id = int(data.cnf_id.item())
        stats = stats_by_cnf.get(cnf_id)
        if stats is None:
            updated.append(copy(data))
            continue
        updated.append(
            attach_var_event_state(
                data,
                stats=stats,
                var_state_dim=var_state_dim,
                momentum=momentum,
                feature_mode=feature_mode,
            )
        )
    return updated
