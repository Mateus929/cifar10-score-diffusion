import torch
import math

import torch


@torch.no_grad()
def general_langevin_sampler(
        score_model,
        x_init,
        schedule,  # List of (t, sigma) tuples
        n_steps_per_sigma=10,
        step_size_factor=2e-5
):
    """
    Optimized Langevin Sampler for 2026 Score-Based Models.
    :param score_model: The trained U-Net s(x, t)
    :param x_init: Initial noise tensor (Batch, C, H, W)
    :param schedule: Iterable of (t_val, sigma_val) pre-computed pairs
    :param n_steps_per_sigma: Inner loop steps for Langevin refinement
    :param step_size_factor: Tuning parameter for step magnitude
    """
    x = x_init.clone()
    device = x.device
    batch_size = x.shape[0]

    for t_val, sigma_val in schedule:
        t = torch.full((batch_size, 1, 1, 1), t_val, device=device)
        step_size = step_size_factor * (sigma_val ** 2)
        for _ in range(n_steps_per_sigma):
            score = score_model(x, t)
            z = torch.randn_like(x)
            x = x + step_size * score + torch.sqrt(2 * step_size) * z
    return x
