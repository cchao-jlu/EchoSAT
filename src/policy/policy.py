import torch
from torch import Tensor
from torch.distributions import Distribution
from torch_scatter import scatter_sum


def _distribution_params(y_var: Tensor, scale_sigma: float) -> tuple[Tensor, Tensor, Tensor]:
    y_var = torch.nan_to_num(y_var, nan=0.0, posinf=8.0, neginf=-8.0).clamp(-20.0, 20.0)
    if y_var.shape[-1] == 2:
        rho, mu = y_var[:, 0], y_var[:, 1]
        sigma = torch.full_like(mu, fill_value=scale_sigma)
    elif y_var.shape[-1] == 3:
        rho, mu, raw_sigma = y_var[:, 0], y_var[:, 1], y_var[:, 2]
        sigma = scale_sigma * torch.exp(raw_sigma.clamp(-4.0, 4.0))
    else:
        raise ValueError(f"Expected policy output dimension 2 or 3, got {y_var.shape[-1]}")

    return rho, mu, sigma


def distributions(y_var: Tensor, scale_sigma: float = 0.1) -> tuple[Distribution, Distribution]:
    rho, mu, sigma = _distribution_params(y_var, scale_sigma)

    # clamp rho at |rho| == 8
    rho = rho.clamp(-8, 8)

    phase_dist = torch.distributions.Binomial(logits=rho, total_count=1)
    scale_dist = torch.distributions.LogNormal(mu, sigma)
    return phase_dist, scale_dist


def sample(y_var: Tensor, num_samples: int = 1, scale_sigma: float = 0.1) -> Tensor:
    phase_dist, scale_dist = distributions(y_var, scale_sigma=scale_sigma)
    sample_shape = torch.Size((num_samples,))
    phase = phase_dist.sample(sample_shape=sample_shape)
    scale = scale_dist.sample(sample_shape=sample_shape)
    var_params = torch.stack([phase, scale], dim=-1)
    return var_params


def mode(y_var: Tensor, scale_sigma: float = 0.1) -> Tensor:
    phase_dist, scale_dist = distributions(y_var, scale_sigma=scale_sigma)
    phase_mode = phase_dist.mode
    scale_mode = scale_dist.mode
    var_params = torch.stack([phase_mode, scale_mode], dim=-1).unsqueeze(0)
    return var_params


def log_prob(y_var: Tensor, var_params: Tensor, var_batch: Tensor | None = None, scale_sigma: float = 0.1) -> Tensor:
    phase_dist, scale_dist = distributions(y_var, scale_sigma=scale_sigma)
    phase_log_prob = phase_dist.log_prob(var_params[:, :, 0])
    scale_log_prob = scale_dist.log_prob(var_params[:, :, 1])

    if var_batch is not None:
        log_prob = phase_log_prob + scale_log_prob
        log_prob = scatter_sum(log_prob, var_batch, dim=1)
    else:
        log_prob = torch.stack([phase_log_prob, scale_log_prob], dim=-1)

    return log_prob


def kl_div(y_var: Tensor, y_lit_ref: Tensor, var_batch: Tensor | None = None, scale_sigma: float = 0.1) -> Tensor:
    phase_dist, scale_dist = distributions(y_var, scale_sigma=scale_sigma)
    phase_dist_ref, scale_dist_ref = distributions(y_lit_ref, scale_sigma=scale_sigma)

    phase_kl_div = torch.distributions.kl_divergence(phase_dist, phase_dist_ref)
    scale_kl_div = torch.distributions.kl_divergence(scale_dist, scale_dist_ref)

    kl_div = phase_kl_div + scale_kl_div

    if var_batch is None:
        kl_div = kl_div.sum()
    else:
        kl_div = scatter_sum(kl_div, var_batch, dim=0)

    return kl_div


def entropy(y_var: Tensor, var_batch: Tensor | None = None, scale_sigma: float = 0.1) -> Tensor:
    phase_dist, scale_dist = distributions(y_var, scale_sigma=scale_sigma)

    phase_entropy = phase_dist.entropy()
    scale_entropy = scale_dist.entropy()
    entropy = phase_entropy + scale_entropy

    if var_batch is None:
        entropy = entropy.sum()
    else:
        entropy = scatter_sum(entropy, var_batch, dim=0).mean()

    return entropy
