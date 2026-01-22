from generators.samplers import general_langevin_sampler
from tqdm import tqdm
import torch

def dsm_sampler(model, config, device):
    """
    Iterative sampler that wraps the Annealed Langevin Dynamics.

    Config Requirements:
    - num_eval_images (int): Total images to generate. Default: 10000.
    - batch_size (int): Sampler batch size. Default: 128.
    - sigma_min (float): Minimum noise level from training.
    - sigma_max (float): Maximum noise level from training.
    - T (float): Time horizon from training.
    - n_steps_per_sigma (int): Langevin inner-loop steps. Default: 5.
    - step_size_factor (float): Multiplier for the step size. Default: 2e-5.
    """

    T = config.get("T", 1)
    sigma_min, sigma_max = config["sigma_min"], config["sigma_max"]
    timesteps = torch.linspace(T, 1e-3, 100)
    sigmas = [sigma_min * torch.sqrt((sigma_max / sigma_min) ** (2 * t / T) - 1) for t in timesteps]
    schedule = list(zip(timesteps, sigmas))

    num_fake_images = config.get("num_eval_images", 10000)
    batch_size = config.get("batch_size", 128)
    fake_images_list = []

    print(f"Generating {num_fake_images} fake images...")

    for _ in tqdm(range(0, num_fake_images, batch_size)):
        cur_batch_size = min(batch_size, num_fake_images - len(fake_images_list) * batch_size)
        x_init = torch.randn((cur_batch_size, 3, 32, 32), device=device) * sigma_max

        samples = general_langevin_sampler(
            model,
            x_init,
            schedule,
            n_steps_per_sigma=config.get("n_steps_per_sigma", 5),
            step_size_factor=config.get("step_size_factor", 2e-5)
        )

        samples = torch.clamp(samples, -1.0, 1.0)
        fake_images_list.append(samples.cpu())

    return torch.cat(fake_images_list, dim=0)