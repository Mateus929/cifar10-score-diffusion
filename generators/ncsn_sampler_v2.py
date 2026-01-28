from tqdm import tqdm
import torch

from utils.ncsn_utils import linear_noise_scale


@torch.no_grad()
def langevin_step(model, x, step_size, noise_idx):
    score = model(x, noise_idx)
    noise = torch.randn_like(x)
    x = x + step_size * score + torch.sqrt(2 * step_size) * noise
    return x

@torch.no_grad()
def langevin_dynamics(model, x, step_size, noise_idx, n_steps=100):
    for i in range(n_steps):
        x = langevin_step(model, x, step_size, noise_idx)
    return x

@torch.no_grad()
def annealed_langevin_dynamics(model, x, noise_scales, n_steps=100, eps=2e-5):
    batch_size = x.size(0)
    for i in tqdm(range(len(noise_scales))):
        noise_idx = torch.full(
            (batch_size,),
            i,
            dtype=torch.long,
            device=x.device
        )
        step_size = eps * (noise_scales[i] / noise_scales[-1]) ** 2
        x = langevin_dynamics(model, x, step_size, noise_idx, n_steps)
        x = x.clamp(-1, 1)
    return x


@torch.no_grad()
def sample(model, shape, noise_scales, device, n_steps=100, eps=2e-5):
    model.eval()
    # x = torch.rand(shape).to(device
    x = torch.randn(shape, device=device) * noise_scales[0]
    x = annealed_langevin_dynamics(model, x, noise_scales, n_steps, eps)
    return x

def sample_many_ncsn(score_model, config):
    score_model.eval()
    total = config.get("total", 128)
    cur_total = 0
    batch_size = config.get("batch_size", 128)

    fake_images = []

    sigma_max = config.get("sigma_max", 1)
    sigma_min = config.get("sigma_min", 0.01)
    L = config.get("L", 10)
    n_steps = config.get("n_steps", 100)
    eps = config.get("eps", 1e-5)

    device = torch.device(config.get('device', 'cuda'))
    noise_scales = linear_noise_scale(start=sigma_max, end=sigma_min, length=L).to(device)

    while cur_total < total:
        cur_batch_size = min(batch_size, total - cur_total)
        print(f"Generating {cur_batch_size} fake images...")
        cur_total += cur_batch_size

        shape = (cur_batch_size, 3, 32, 32)
        fake_images.append(sample(score_model, shape, noise_scales, device, n_steps, eps))

    return torch.concat(fake_images)