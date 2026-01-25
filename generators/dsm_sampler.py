import torch
from tqdm import tqdm


@torch.no_grad()
def dsm_sampler(model, config, device):
    """
    Annealed Langevin sampler consistent with DSM training.

    Assumes training corruption:
        x_t = x + sigma(t) * epsilon

    Model learns:
        s_theta(x, t) ≈ ∇_x log p_sigma(x)

    Config requirements:
    --------------------
    sigma_min : float
    sigma_max : float
    T         : float
    num_eval_images : int
    batch_size : int
    n_steps_per_sigma : int
    step_size_factor : float
    num_slices: int
    """

    model.eval()

    sigma_min = config["sigma_min"]
    sigma_max = config["sigma_max"]
    T = config.get("T", 1.0)

    num_images = config.get("num_eval_images", 10000)
    batch_size = config.get("batch_size", 128)

    n_steps_per_sigma = config.get("n_steps_per_sigma", 100)
    step_size_factor = config.get("step_size_factor", 0.01)

    timesteps = torch.linspace(T, 1e-3, config.get("num_slices", 50), device=device)

    sigmas = sigma_min * torch.sqrt(
        (sigma_max / sigma_min) ** (2 * timesteps / T) - 1.0
    )

    fake_images = []

    print(f"Generating {num_images} images using DSM Langevin...")

    for _ in tqdm(range(0, num_images, batch_size)):
        cur_bs = min(batch_size, num_images - len(fake_images) * batch_size)

        x = torch.randn(cur_bs, 3, 32, 32, device=device) * sigmas[0]

        for t_val, sigma in zip(timesteps, sigmas):
            t = torch.full((cur_bs, 1, 1, 1), t_val, device=device)
            step_size = step_size_factor * sigma**2

            for _ in range(n_steps_per_sigma):
                score = model(x, t)
                noise = torch.randn_like(x)

                x = x + step_size * score + torch.sqrt(2 * step_size) * noise
        x = torch.clamp(x, -1.0, 1.0)
        fake_images.append(x.cpu())

    return torch.cat(fake_images, dim=0)
