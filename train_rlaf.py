import os
import signal
import sys
import time

import hydra
import numpy as np
import pandas as pd
import torch

import wandb
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import DataLoader
from torch_geometric.seed import seed_everything

from src.data.dataset import DimacsCNFDataset, RLTrainingDataset
from src.policy.evaluate import compute_solver_stats, sample_random_var_params, sample_var_params
from src.model.model import GNN, init_model, init_transform
from src.solving.budget import apply_rollout_budget
from src.solving.state import attach_global_state_batch, attach_var_event_state_batch

from src.training.dpo import train_dpo
from src.training.grpo import train_grpo, get_grpo_advantage

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def _memory_snapshot() -> str:
    try:
        with open("/proc/self/status") as f:
            rows = {
                line.split(":", 1)[0]: line.split(":", 1)[1].strip()
                for line in f
                if line.startswith(("VmRSS:", "VmHWM:", "VmSize:", "Threads:"))
            }
        return ", ".join(f"{key}={value}" for key, value in rows.items())
    except OSError:
        return "memory snapshot unavailable"


def _install_signal_diagnostics() -> None:
    def _handler(signum, frame):
        print(f"Received signal {signum}; {_memory_snapshot()}", flush=True)
        sys.exit(128 + signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, _handler)


def log_solver_metrics(
        solver_stats: pd.DataFrame,
        iteration: int,
        global_step: int,
        prefix: str = "train",
        add_target_histogram: bool = False,
        target_stat: str = "decisions",
) -> None:
    keys = ["decisions", "conflicts", "propagations", "restarts", "CPU time"]
    metrics = {f"{prefix}/{key}": solver_stats[key].mean() for key in keys if key in solver_stats.columns}

    print(
        f"Solver metrics at iteration {iteration} ({prefix}): \n"
        + "\n".join(f"{key}: {val:.2f}" for key, val in metrics.items())
    )

    metrics[f"iteration"] = iteration
    metrics[f"global_step"] = global_step

    for key in keys:
        if key in metrics:
            metrics[f"{prefix}/{key}_histogram"] = wandb.Histogram(solver_stats[key])

    if add_target_histogram:
        grouped = solver_stats[["cnf_id", target_stat]].groupby("cnf_id")
        target_mean = grouped.mean().loc[solver_stats["cnf_id"]]
        target_mean = target_mean[target_stat].to_numpy()
        if not np.any(np.isnan(target_mean)):
            metrics[f"{prefix}/{target_stat}_histogram_mean"] = wandb.Histogram(target_mean)
        target_std = grouped.std().loc[solver_stats["cnf_id"]]
        target_std = target_std[target_stat].to_numpy()
        if not np.any(np.isnan(target_std)):
            metrics[f"{prefix}/{target_stat}_histogram_std"] = wandb.Histogram(target_std)

    wandb.log(metrics, step=global_step)


def add_composite_target(solver_stats: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    solver_stats = solver_stats.copy()
    if cfg.training.target_stat != "composite":
        return solver_stats

    def stat_tensor(column: str) -> torch.Tensor:
        if column not in solver_stats.columns:
            return torch.zeros(len(solver_stats), dtype=torch.float64)
        values = solver_stats[column]
        fill_value = values.max()
        if pd.isna(fill_value):
            fill_value = 0.0
        return torch.tensor(values.fillna(fill_value).to_numpy(), dtype=torch.float64)

    composite = torch.zeros(len(solver_stats), dtype=torch.float64)
    if cfg.training.composite_cpu_weight != 0.0:
        cpu_time = stat_tensor("CPU time")
        composite += cfg.training.composite_cpu_weight * torch.log1p(cpu_time)
    if cfg.training.composite_conflicts_weight != 0.0:
        conflicts = stat_tensor("conflicts")
        composite += cfg.training.composite_conflicts_weight * torch.log1p(conflicts)
    if cfg.training.composite_decisions_weight != 0.0:
        decisions = stat_tensor("decisions")
        composite += cfg.training.composite_decisions_weight * torch.log1p(decisions)

    solver_stats["composite"] = composite.numpy()
    return solver_stats


def save_model(model: GNN, cfg: DictConfig, checkpoint_name: str = "last") -> None:
    model_dir = cfg.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    cfg_path = os.path.join(model_dir, "config.yaml")
    with open(cfg_path, "w") as f:
        OmegaConf.save(cfg, f)
    ckpt_path = os.path.join(model_dir, f"{checkpoint_name}.pt")
    state_dict = model.state_dict()
    torch.save(state_dict, ckpt_path)


def load_compatible_checkpoint(model: torch.nn.Module, ckpt_path: str) -> tuple[list[str], list[str]]:
    checkpoint_state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model_state = model.state_dict()
    compatible_state = {}
    skipped = []
    for key, value in checkpoint_state.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible_state[key] = value
        else:
            skipped.append(key)
    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    skipped.extend(unexpected)
    return sorted(compatible_state), sorted(set(skipped + list(missing)))


def build_feedback_state_loader(
        model: GNN,
        dataset: DimacsCNFDataset,
        loader: DataLoader,
        cfg: DictConfig,
        device: torch.device | str,
        max_num_batches: int = -1,
        mode: str | None = None,
        shuffle: bool = True,
) -> DataLoader:
    mode = cfg.training.feedback_input_mode if mode is None else mode
    if mode == "none":
        return loader

    global_state_dim = getattr(model, "global_state_dim", 0)
    var_state_dim = getattr(model, "var_state_dim", 0)
    feedback_state_type = cfg.training.feedback_state_type if "feedback_state_type" in cfg.training else "global"
    state_momentum = float(cfg.training.feedback_state_momentum) if "feedback_state_momentum" in cfg.training else 0.0
    event_state_feature_mode = (
        cfg.training.feedback_event_state_features
        if "feedback_event_state_features" in cfg.training
        else "legacy"
    )
    use_event_adapter_cache = bool(getattr(model, "event_adapter_enabled", False))
    if feedback_state_type == "global" and global_state_dim <= 0:
        raise ValueError("global feedback requires model.global_state_dim > 0")
    if feedback_state_type == "event_var" and var_state_dim <= 0:
        raise ValueError("event_var feedback requires model.var_state_dim > 0")

    rollout_budget_type = (
        cfg.training.feedback_warmup_budget_type
        if "feedback_warmup_budget_type" in cfg.training
        else "cpu_time"
    )
    rollout_conflicts = (
        int(cfg.training.feedback_warmup_conflicts)
        if "feedback_warmup_conflicts" in cfg.training
        else None
    )
    warmup_params = apply_rollout_budget(
        dict(cfg.solver.params),
        budget_type=rollout_budget_type,
        cpu_lim=cfg.training.feedback_warmup_cpu_lim,
        conflicts=rollout_conflicts,
    )
    if feedback_state_type == "event_var":
        warmup_params["collect-events"] = True

    if mode == "mixed_warmup":
        mode = "model_warmup" if np.random.random() < cfg.training.feedback_model_warmup_prob else "random_warmup"

    if mode == "random_warmup":
        warmup_data_list = sample_random_var_params(
            loader=loader,
            num_samples=cfg.training.feedback_warmup_num_samples,
            max_num_batches=max_num_batches,
            weight_scale=cfg.training.feedback_random_weight,
        )
    elif mode == "model_warmup":
        warmup_data_list = sample_var_params(
            model=model,
            loader=loader,
            num_samples=cfg.training.feedback_warmup_num_samples,
            max_num_batches=max_num_batches,
            device=device,
            use_mode=True,
            scale_sigma=cfg.scale_sigma,
            cache_var_features=use_event_adapter_cache,
        )
    else:
        raise ValueError(f"Unknown feedback_input_mode {mode}")

    warmup_stats = compute_solver_stats(
        dataset=dataset,
        data_list=warmup_data_list,
        num_workers=cfg.solver.num_workers,
        solver=cfg.solver.solver,
        **warmup_params,
    )
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
    return DataLoader(
        dataset=refined_graphs,
        batch_size=cfg.loader.batch_size,
        num_workers=cfg.loader.num_workers,
        shuffle=shuffle,
    )


@hydra.main(version_base=None, config_path="configs", config_name="config_train_rlaf")
def main(cfg: DictConfig):
    _install_signal_diagnostics()
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)

    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        config=OmegaConf.to_container(cfg),
        mode=cfg.wandb.mode if "mode" in cfg.wandb else None,
    )

    transform = init_transform(cfg)
    model = init_model(cfg, transform)
    if cfg.from_checkpoint is not None:
        loaded_keys, skipped_keys = load_compatible_checkpoint(model, cfg.from_checkpoint)
        print(
            f"Loaded {len(loaded_keys)} compatible tensors from {cfg.from_checkpoint}; "
            f"skipped {len(skipped_keys)} tensors with missing or mismatched shapes."
        )

    dataset_train = DimacsCNFDataset(
        path=cfg.dataset.train_path,
        transform=transform,
        lazy=bool(cfg.dataset.lazy) if "lazy" in cfg.dataset else False,
    )
    loader_train = DataLoader(
        dataset=dataset_train,
        batch_size=cfg.loader.batch_size,
        num_workers=cfg.loader.num_workers,
        shuffle=True,
    )

    dataset_val = DimacsCNFDataset(
        path=cfg.dataset.val_path,
        transform=transform,
        lazy=bool(cfg.dataset.lazy_val) if "lazy_val" in cfg.dataset else False,
    )
    loader_val = DataLoader(
        dataset=dataset_val,
        batch_size=cfg.loader.batch_size,
        num_workers=cfg.loader.num_workers,
        shuffle=False,
    )

    assert cfg.training.cnf_per_iter % cfg.loader.batch_size == 0
    train_sample_num_batches = cfg.training.cnf_per_iter // cfg.loader.batch_size

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device {device}")

    optim = torch.optim.AdamW(
        params=model.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        maximize=True,
    )

    warmup_iterations = 5
    def lr_lambda(step):
        if step < warmup_iterations:
            # Warmup from 0 to 1.0
            return float(step + 1) / float(warmup_iterations)
        else:
            return 1.0
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    if cfg.ckpt_interval is not None:
        print(f"Saving checkpoint at iteration 0")
        save_model(model, cfg, f"iter=0")

    best_score = np.inf

    global_step = 0
    for iteration in range(cfg.training.iterations):
        print(f" ----------------------- {'GRPO' if cfg.method == 'grpo' else 'DPO'} Iteration {iteration} ----------------------- ")
        print(f"[diag] iteration {iteration} start: {_memory_snapshot()}", flush=True)

        # validate if necessary
        do_validation = cfg.val_interval is not None and cfg.val_interval > 0 and iteration % cfg.val_interval == 0
        if iteration == 0 and bool(cfg.skip_initial_val):
            do_validation = False
        if do_validation:
            loader_val_current = build_feedback_state_loader(
                model=model,
                dataset=dataset_val,
                loader=loader_val,
                cfg=cfg,
                device=device,
                mode=cfg.training.feedback_val_mode,
                shuffle=False,
            )
            data_list_val = sample_var_params(
                model=model,
                loader=loader_val_current,
                num_samples=1,
                device=device,
                use_mode=True,
                scale_sigma=cfg.scale_sigma,
            )

            solver_stats_val = compute_solver_stats(
                dataset=dataset_val,
                data_list=data_list_val,
                num_workers=cfg.solver.num_workers,
                solver=cfg.solver.solver,
                **cfg.solver.params,
            )
            solver_stats_val = add_composite_target(solver_stats_val, cfg)

            log_solver_metrics(
                solver_stats=solver_stats_val,
                iteration=iteration,
                global_step=global_step,
                prefix="val",
                add_target_histogram=True,
                target_stat=cfg.training.target_stat
            )

            score = solver_stats_val[cfg.training.target_stat].mean()
            if score < best_score:
                print("Saving new best checkpoint")
                save_model(model, cfg, "best")
                best_score = score

        loader_train_current = build_feedback_state_loader(
            model=model,
            dataset=dataset_train,
            loader=loader_train,
            cfg=cfg,
            device=device,
            max_num_batches=train_sample_num_batches,
            shuffle=True,
        )
        print(f"[diag] iteration {iteration} after feedback loader: {_memory_snapshot()}", flush=True)

        data_list_train = sample_var_params(
            model=model,
            loader=loader_train_current,
            num_samples=cfg.training.num_samples,
            max_num_batches=train_sample_num_batches,
            device=device,
            scale_sigma=cfg.scale_sigma,
        )
        print(f"[diag] iteration {iteration} after policy sampling: {_memory_snapshot()}", flush=True)

        solver_stats = compute_solver_stats(
            dataset=dataset_train,
            data_list=data_list_train,
            num_workers=cfg.solver.num_workers,
            solver=cfg.solver.solver,
            **cfg.solver.params,
        )
        print(f"[diag] iteration {iteration} after solver stats: {_memory_snapshot()}", flush=True)
        solver_stats = add_composite_target(solver_stats, cfg)

        log_solver_metrics(
            solver_stats=solver_stats,
            iteration=iteration,
            global_step=global_step,
            prefix="train",
            add_target_histogram=True,
            target_stat=cfg.training.target_stat,
        )

        if cfg.method == "grpo":
            solver_stats["advantage"] = get_grpo_advantage(solver_stats, cfg.training.target_stat)
            iteration_dataset = RLTrainingDataset(
                data_list=data_list_train,
                solver_stats=solver_stats,
                target_stat="advantage",
                objective="maximize",
            )
        else:
            iteration_dataset = RLTrainingDataset(
                data_list=data_list_train,
                solver_stats=solver_stats,
                target_stat=cfg.training.target_stat
            )

        iteration_loader = DataLoader(
            dataset=iteration_dataset,
            batch_size=cfg.loader.batch_size,
            num_workers=cfg.loader.num_workers,
            shuffle=True
        )

        if cfg.method == "grpo":
            global_step = train_grpo(
                model=model,
                optim=optim,
                sched=sched,
                loader=iteration_loader,
                steps=cfg.training.steps_per_iter,
                clip_ratio=cfg.training.clip_ratio,
                kl_penalty=cfg.training.kl_penalty,
                global_step=global_step,
                device=device,
                use_amp=cfg.training.use_amp,
                scale_sigma=cfg.scale_sigma,
            )
        else:
            global_step = train_dpo(
                model=model,
                optim=optim,
                sched=sched,
                loader=iteration_loader,
                steps=cfg.training.steps_per_iter,
                beta=cfg.training.beta,
                kl_penalty=cfg.training.kl_penalty,
                global_step=global_step,
                device=device,
                use_amp=cfg.training.use_amp,
                scale_sigma=cfg.scale_sigma,
            )

        if cfg.ckpt_interval is not None and iteration % cfg.ckpt_interval == 0:
            print(f"Saving checkpoint at iteration {iteration}")
            save_model(model, cfg, f"iter={iteration}")
        if cfg.last_ckpt_interval is not None and iteration % cfg.last_ckpt_interval == 0:
            save_model(model, cfg, "last")

    save_model(model, cfg, "last")
    wandb.finish()


if __name__ == '__main__':
    main()
