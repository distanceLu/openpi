from __future__ import annotations

import torch


def make_denoise_timesteps(num_steps: int, device: torch.device) -> torch.Tensor:
    """Return K+1 timesteps from 1 -> 0, matching RLinf's pi0.5 schedule."""

    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    timesteps = torch.linspace(1.0, 1.0 / num_steps, num_steps, device=device)
    return torch.cat([timesteps, torch.zeros(1, device=device)])


def normal_logprob(sample: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Elementwise log N(sample | mean, std^2), safe for deterministic ODE steps."""

    deterministic = std <= eps
    safe_std = torch.where(deterministic, torch.ones_like(std), std.clamp_min(eps))
    logprob = -torch.log(safe_std) - 0.5 * torch.log(2 * torch.pi * torch.ones_like(sample))
    logprob = logprob - 0.5 * ((sample - mean) / safe_std).pow(2)
    return torch.where(deterministic, torch.zeros_like(logprob), logprob)


def normal_entropy(std: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    deterministic = std <= eps
    safe_std = torch.where(deterministic, torch.ones_like(std), std.clamp_min(eps))
    entropy = 0.5 * torch.log(2 * torch.pi * torch.e * safe_std.pow(2))
    return torch.where(deterministic, torch.zeros_like(entropy), entropy)


def pi05_flow_step_mean_std(
    x_t: torch.Tensor,
    velocity: torch.Tensor,
    timestep: torch.Tensor,
    delta: torch.Tensor,
    method: str = "flow_ode",
    noise_level: float | torch.Tensor = 0.0,
    learned_std: torch.Tensor | None = None,
    timesteps: torch.Tensor | None = None,
    step_index: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute one pi0.5 denoising transition distribution.

    The model predicts velocity v_theta(o, x_t, t).  Following the RLinf
    implementation, derive x0/x1 predictions and then the transition mean/std
    for ODE, SDE, CPS, or learned-noise sampling.
    """

    while timestep.dim() < x_t.dim():
        timestep = timestep.unsqueeze(-1)
    while delta.dim() < x_t.dim():
        delta = delta.unsqueeze(-1)

    x0_pred = x_t - velocity * timestep
    x1_pred = x_t + velocity * (1.0 - timestep)
    next_timestep = timestep - delta

    if method == "flow_ode":
        x0_weight = 1.0 - next_timestep
        x1_weight = next_timestep
        std = torch.zeros_like(x_t)
    elif method == "flow_sde":
        if timesteps is None or step_index is None:
            raise ValueError("flow_sde requires the complete timesteps schedule and step_index")
        schedule = timesteps.to(device=x_t.device, dtype=x_t.dtype)
        indices = step_index.to(device=x_t.device, dtype=torch.long)
        denom_timesteps = torch.where(schedule == 1, schedule[1], schedule)
        sigma_ratio = schedule / (1.0 - denom_timesteps)
        sigmas = torch.as_tensor(noise_level, device=x_t.device, dtype=x_t.dtype) * torch.sqrt(sigma_ratio)[:-1]
        sigma = sigmas[indices]
        while sigma.dim() < x_t.dim():
            sigma = sigma.unsqueeze(-1)
        sigma = sigma.expand_as(x_t)
        x0_weight = 1.0 - next_timestep
        x1_weight = next_timestep - sigma.pow(2) * delta / (2.0 * timestep)
        std = torch.sqrt(delta) * sigma
    elif method == "flow_cps":
        noise_level_t = torch.as_tensor(noise_level, device=x_t.device, dtype=x_t.dtype)
        cos_term = torch.cos(torch.pi * noise_level_t / 2.0)
        sin_term = torch.sin(torch.pi * noise_level_t / 2.0)
        x0_weight = 1.0 - next_timestep
        x1_weight = next_timestep * cos_term
        std = next_timestep * sin_term
    elif method == "flow_noise":
        if learned_std is None:
            raise ValueError("learned_std is required when method='flow_noise'")
        x0_weight = 1.0 - next_timestep
        x1_weight = next_timestep
        std = learned_std
    else:
        raise ValueError(f"Unknown pi0.5 denoise method: {method}")

    mean = x0_pred * x0_weight + x1_pred * x1_weight
    return mean, std.expand_as(x_t)
