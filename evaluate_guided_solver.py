import os

from omegaconf import DictConfig, OmegaConf
import hydra
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from src.data.dataset import DimacsCNFDataset
from src.model.model import load_checkpoint
from src.policy import policy
from src.policy.evaluate import sample_var_params, compute_solver_stats, var_params_from_target_prediction
from src.solving.budget import apply_rollout_budget
from src.solving.state import attach_global_state_batch, attach_var_event_state_batch


def print_solver_metrics(solver_stats: pd.DataFrame) -> None:
    keys = ["decisions", "conflicts", "propagations", "restarts", "CPU time", "GPU time", "time"]
    metrics = {key: solver_stats[key].mean() for key in keys if key in solver_stats.columns}
    print(
        f"Metrics: \n"
        + "\n".join(f"{key}: {val:.2f}" for key, val in metrics.items())
    )


def _cnf_id_value(data) -> int:
    cnf_id = data.cnf_id
    return int(cnf_id.item() if hasattr(cnf_id, "item") else cnf_id)


def _gpu_time_by_cnf(data_list: list) -> pd.DataFrame:
    rows = []
    for data in data_list:
        rows.append(
            {
                "cnf_id": _cnf_id_value(data),
                "GPU time": float(getattr(data, "gpu_time", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def add_total_time_columns(
    solver_stats: pd.DataFrame,
    data_list: list,
    refinement_stats_list: list[pd.DataFrame] | None = None,
    refinement_data_lists: list[list] | None = None,
) -> pd.DataFrame:
    solver_stats = solver_stats.copy()
    gpu_time = _gpu_time_by_cnf(data_list).rename(columns={"GPU time": "final GPU time"})
    solver_stats = solver_stats.merge(gpu_time, on="cnf_id", how="left")
    solver_stats["final GPU time"] = solver_stats["final GPU time"].fillna(0.0)
    solver_stats["refinement CPU time"] = 0.0
    solver_stats["refinement GPU time"] = 0.0

    for refinement_stats in refinement_stats_list or []:
        if "CPU time" not in refinement_stats.columns:
            continue
        cpu_time = (
            refinement_stats[["cnf_id", "CPU time"]]
            .groupby("cnf_id", sort=False)["CPU time"]
            .sum()
            .rename("round CPU time")
            .reset_index()
        )
        solver_stats = solver_stats.merge(cpu_time, on="cnf_id", how="left")
        solver_stats["refinement CPU time"] += solver_stats["round CPU time"].fillna(0.0)
        solver_stats = solver_stats.drop(columns=["round CPU time"])

    for refinement_data in refinement_data_lists or []:
        gpu_round = (
            _gpu_time_by_cnf(refinement_data)
            .groupby("cnf_id", sort=False)["GPU time"]
            .sum()
            .rename("round GPU time")
            .reset_index()
        )
        solver_stats = solver_stats.merge(gpu_round, on="cnf_id", how="left")
        solver_stats["refinement GPU time"] += solver_stats["round GPU time"].fillna(0.0)
        solver_stats = solver_stats.drop(columns=["round GPU time"])

    solver_stats["GPU time"] = solver_stats["final GPU time"] + solver_stats["refinement GPU time"]
    solver_stats["time"] = (
        solver_stats["CPU time"]
        + solver_stats["refinement CPU time"]
        + solver_stats["GPU time"]
    )
    return solver_stats


@hydra.main(version_base=None, config_path="configs", config_name="config_eval_guided_solver")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)

    if not cfg.is_supervised:
        model, transform, model_cfg = load_checkpoint(cfg.checkpoint, var_output=True)
    else:
        model, transform, model_cfg = load_checkpoint(cfg.checkpoint, var_output=False)

    dataset = DimacsCNFDataset(
        path=cfg.dataset.eval_path,
        transform=transform,
        lazy=bool(cfg.dataset.lazy) if "lazy" in cfg.dataset else False,
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.loader.batch_size,
        num_workers=cfg.loader.num_workers,
        shuffle=False,
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    global_state_dim = getattr(model, "global_state_dim", 0)
    var_state_dim = getattr(model, "var_state_dim", 0)
    feedback_state_type = cfg.feedback_refinement.state_type if "state_type" in cfg.feedback_refinement else "global"
    if cfg.feedback_refinement.enabled and feedback_state_type == "global" and global_state_dim <= 0:
        raise ValueError("global feedback_refinement requires a checkpoint trained with model.global_state_dim > 0")
    if cfg.feedback_refinement.enabled and feedback_state_type == "event_var" and var_state_dim <= 0:
        raise ValueError("event_var feedback_refinement requires a checkpoint trained with model.var_state_dim > 0")
    event_state_feature_mode = (
        cfg.feedback_refinement.event_state_features
        if "event_state_features" in cfg.feedback_refinement
        else "legacy"
    )
    use_event_adapter_cache = bool(getattr(model, "event_adapter_enabled", False))

    # warmup GPU for accurate time measurements
    warmup_steps = 4
    with torch.no_grad():
        for i, data in enumerate(loader):
            data.to(device)
            y_var = model(data)
            if not cfg.is_supervised:
                _ = policy.mode(y_var)
            if i >= warmup_steps:
                break

    current_loader = loader
    refinement_stats_list = []
    refinement_data_lists = []
    if cfg.feedback_refinement.enabled:
        rollout_budget_type = (
            cfg.feedback_refinement.rollout_budget_type
            if "rollout_budget_type" in cfg.feedback_refinement
            else "cpu_time"
        )
        rollout_conflicts = (
            int(cfg.feedback_refinement.rollout_conflicts)
            if "rollout_conflicts" in cfg.feedback_refinement
            else None
        )
        warmup_params = apply_rollout_budget(
            dict(cfg.solver.params),
            budget_type=rollout_budget_type,
            cpu_lim=cfg.feedback_refinement.warmup_cpu_lim,
            conflicts=rollout_conflicts,
        )
        num_rounds = int(cfg.feedback_refinement.num_rounds) if "num_rounds" in cfg.feedback_refinement else 1
        state_momentum = float(cfg.feedback_refinement.state_momentum) if "state_momentum" in cfg.feedback_refinement else 0.0
        if feedback_state_type == "event_var":
            warmup_params["collect-events"] = True

        for _ in range(num_rounds):
            if not cfg.is_supervised:
                warmup_data_list = sample_var_params(
                    model=model,
                    loader=current_loader,
                    device=device,
                    use_mode=True,
                    num_samples=1,
                    scale_sigma=model_cfg.scale_sigma,
                    add_timing=True,
                    cache_var_features=use_event_adapter_cache,
                )
            else:
                warmup_data_list = var_params_from_target_prediction(
                    model=model,
                    loader=current_loader,
                    device=device,
                    target=model_cfg.dataset.target,
                    pred_scale=cfg.pred_scale,
                    add_timing=True,
                )

            warmup_stats = compute_solver_stats(
                dataset=dataset,
                data_list=warmup_data_list,
                num_workers=cfg.solver.num_workers,
                solver=model_cfg.solver.solver if cfg.solver.solver is None else cfg.solver.solver,
                **warmup_params,
            )
            refinement_stats_list.append(warmup_stats)
            refinement_data_lists.append(warmup_data_list)
            if feedback_state_type == "event_var":
                refined_graphs = attach_var_event_state_batch(
                    warmup_data_list,
                    warmup_stats,
                    var_state_dim=var_state_dim,
                    momentum=state_momentum,
                    feature_mode=event_state_feature_mode,
                )
            else:
                refined_graphs = attach_global_state_batch(
                    warmup_data_list,
                    warmup_stats,
                    global_state_dim=global_state_dim,
                )
            current_loader = DataLoader(
                dataset=refined_graphs,
                batch_size=cfg.loader.batch_size,
                num_workers=cfg.loader.num_workers,
                shuffle=False,
            )

    if not cfg.is_supervised:
        data_list = sample_var_params(
            model=model,
            loader=current_loader,
            device=device,
            use_mode=True,
            num_samples=1,
            scale_sigma=model_cfg.scale_sigma,
            add_timing=True,
            cache_var_features=use_event_adapter_cache,
        )
    else:
        data_list = var_params_from_target_prediction(
            model=model,
            loader=current_loader,
            device=device,
            target=model_cfg.dataset.target,
            pred_scale=cfg.pred_scale,
            add_timing=True,
        )

    solver = model_cfg.solver.solver if cfg.solver.solver is None else cfg.solver.solver
    solver_params = cfg.solver.params
    print(solver_params)

    solver_stats = compute_solver_stats(
        dataset=dataset,
        data_list=data_list,
        num_workers=cfg.solver.num_workers,
        solver=solver,
        **solver_params,
    )

    solver_stats = add_total_time_columns(
        solver_stats,
        data_list,
        refinement_stats_list=refinement_stats_list,
        refinement_data_lists=refinement_data_lists,
    )

    print_solver_metrics(solver_stats)

    if cfg.save_file is not None:
        save_file = os.path.join(os.path.dirname(cfg.checkpoint), cfg.save_file)
        solver_stats.to_csv(save_file)


if __name__ == '__main__':
    main()
