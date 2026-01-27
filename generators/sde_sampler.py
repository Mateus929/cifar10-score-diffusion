import numpy as np
import torch

from utils.diffusion_utils import marginal_prob_std, diffusion_coeff


def sde_sampler(score_model, config,
                eps=1e-3):
    """Generate samples from score-based models with Predictor-Corrector method.

    Args:
      score_model: A PyTorch model that represents the time-dependent score-based model.
      num_steps: The number of sampling steps.
        Equivalent to the number of discretized time steps.
      device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
      eps: The smallest time step for numerical stability.

    Returns:
      Samples.
    """
    batch_size = config.get("batch_size", 128)
    num_steps = config.get("num_steps", 500)
    device = config.get("device", 'cuda')
    snr = config.get("snr", 0.16)
    sigma = config.get("sigma", 25.0)

    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, 3, 32, 32, device=device) * marginal_prob_std(t, sigma)[:, None, None, None]
    time_steps = np.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    with torch.no_grad():
        for time_step in time_steps:
            batch_time_step = torch.ones(batch_size, device=device) * time_step

            grad = score_model(x, batch_time_step)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            grad_norm = grad_norm.clamp(min=1e-10) #OPTIONAL
            noise_norm = np.sqrt(np.prod(x.shape[1:]))
            langevin_step_size = 2 * (snr * noise_norm / grad_norm) ** 2
            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)

            g = diffusion_coeff(batch_time_step, sigma, device)
            x_mean = x + (g ** 2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
            x = x_mean + torch.sqrt(g ** 2 * step_size)[:, None, None, None] * torch.randn_like(x)

        return x_mean


def sample_many_dsm(score_model, config):
    total = config.get("total", 100)
    cur_total = 0
    batch_size = config.get("batch_size", 128)

    fake_images = []

    while cur_total < total:
        cur_batch_size = min(batch_size, total - cur_total)
        config["batch_size"] = cur_batch_size

        fake_images.append(sde_sampler(score_model, config))
    config["batch_size"] = batch_size
    return torch.concat(fake_images)