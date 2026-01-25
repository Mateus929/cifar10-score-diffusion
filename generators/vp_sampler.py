import torch

@torch.no_grad()
def vp_sampler(model, config, device):
    model.eval()

    num_images = config.get("num_eval_images", 5000)
    num_steps = config.get("num_steps", 1000)

    beta_min = config.get("beta_min", 0.1)
    beta_max = config.get("beta_max", 20.0)

    x = torch.randn(num_images, 3, 32, 32, device=device)

    dt = 1.0 / num_steps
    t = torch.ones(num_images, device=device)

    for _ in range(num_steps):
        t_ = t.view(-1, 1, 1, 1)

        beta = beta_min + t_ * (beta_max - beta_min)

        score = model(x, t)

        drift = -0.5 * beta * x - beta * score
        diffusion = torch.sqrt(beta * dt)

        x = x + drift * dt + diffusion * torch.randn_like(x)

        t = torch.clamp(t - dt, min=1e-4)

    return x
