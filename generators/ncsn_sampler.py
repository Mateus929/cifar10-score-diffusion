import torch
from tqdm import tqdm
import numpy as np


@torch.no_grad()
def nscn_sampler(model, config, device):
    model.eval()

    sigma_min = config["sigma_min"]
    sigma_max = config["sigma_max"]
    n_steps_per_sigma = config["n_steps_per_sigma"]
    L = config.get("L", 10)
    step_size_factor = config.get("step_size_factor", 2e-5)

    sigmas = torch.tensor(
        sigma_max * (sigma_min / sigma_max) ** (np.arange(L) / (L - 1)),
        dtype=torch.float32,
        device=device
    )

    num_images = config.get("num_eval_images", 10000)
    batch_size = config.get("batch_size", 128)

    fake_images = []

    print(f"Generating {num_images} images using DSM Langevin...")

    for _ in tqdm(range(0, num_images, batch_size)):
        cur_bs = min(batch_size, num_images - len(fake_images) * batch_size)

        x = torch.randn(cur_bs, 3, 32, 32, device=device) * sigmas[0]

        for level in range(0, L):
            step_size = step_size_factor * ((sigmas[level] / sigma_min) ** 2)
            for _ in range(n_steps_per_sigma):
                sigma_level = sigmas[level].expand(cur_bs)
                score = model(x, sigma_level)
                z = torch.randn_like(x)
                x = x + step_size * score + torch.sqrt(2 * step_size) * z

        fake_images.append(x)

    return torch.cat(fake_images, dim=0)
