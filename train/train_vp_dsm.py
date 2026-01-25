import os
import uuid
import torch
import wandb

from models.simple_unet import ScoreUNet
from utils.data_loader import load_cifar10
from utils.eval import get_metric_scores
from utils.losses import vp_score_loss
from utils.checkpoint_manager import CheckpointManager
from generators.vp_sampler import vp_sampler

BASE_WORK_DIR = os.environ.get("BASE_WORK_DIR")
CHECKPOINT_DIR = os.path.join(BASE_WORK_DIR, "checkpoints")

def beta_t(t, beta_min, beta_max):
    return beta_min + t * (beta_max - beta_min)

def int_beta_t(t, beta_min, beta_max):
    return beta_min * t + 0.5 * (beta_max - beta_min) * t ** 2

def marginal_alpha(t, beta_min, beta_max):
    return torch.exp(-0.5 * int_beta_t(t, beta_min, beta_max))

def marginal_sigma(t, beta_min, beta_max):
    return torch.sqrt(1.0 - marginal_alpha(t, beta_min, beta_max) ** 2)

def train_vp(config):
    run_id = config.get("run_id", str(uuid.uuid4().hex[:8]))

    wandb.init(
        project="diffusion-score-matching",
        name=config["run_name"],
        config=config,
        id=run_id,
        resume="allow"
    )

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = "cpu"
    beta_min = config.get("beta_min", 0.1)
    beta_max = config.get("beta_max", 20.0)

    model = ScoreUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    train_loader, test_loader = load_cifar10(
        batch_size=config.get("batch_size", 128),
    )

    checkpoint_manager = CheckpointManager(
        base_dir=CHECKPOINT_DIR,
        run_id=run_id,
        max_checkpoints=3
    )

    start_epoch = 0
    latest = checkpoint_manager.get_latest_checkpoint()
    if latest and config.get("resume_training", False):
        start_epoch, _ = checkpoint_manager.load_checkpoint(
            latest,
            {"model": model},
            {"optimizer": optimizer}
        )
        start_epoch += 1

    print(f"Starting VP training from epoch {start_epoch}")


    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        epoch_loss = 0.0

        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()

            B = x.size(0)
            t = torch.rand(B, device=device) * 0.999 + 1e-4
            t_ = t.view(B, 1, 1, 1)

            noise = torch.randn_like(x)

            alpha = marginal_alpha(t_, beta_min, beta_max)
            sigma = marginal_sigma(t_, beta_min, beta_max)

            x_noisy = alpha * x + sigma * noise

            score = model(x_noisy, t)

            loss = vp_score_loss(score, noise, sigma)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)

        wandb.log({
            "epoch/epoch": epoch,
            "epoch/loss": epoch_loss
        })

        print(f"[VP] Epoch {epoch} | Loss: {epoch_loss:.4f}")

        if (epoch + 1) % config.get("save_every", 10) == 0:
            checkpoint_manager.save_checkpoint(
                epoch,
                {"model": model},
                {"optimizer": optimizer},
                {}
            )

    if config.get("eval", False):
        evaluate_vp(model, config, train_loader, test_loader)

    wandb.finish()
    return model

@torch.no_grad()
def evaluate_vp(model, config, train_loader, test_loader):
    model.eval()
    device = next(model.parameters()).device

    fake_images = vp_sampler(model, config, device)

    print("Computing FID / IS...")

    test_metrics = get_metric_scores(test_loader, fake_images, "test")
    train_metrics = get_metric_scores(train_loader, fake_images, "train")

    wandb.log({
        "fid/test": test_metrics["FID"],
        "is/test_mean": test_metrics["IS_MEAN"],
        "fid/train": train_metrics["FID"]
    })

    samples = fake_images[:4]
    samples = ((samples + 1.0) * 0.5 * 255).clamp(0, 255).byte()

    for i, img in enumerate(samples):
        img = img.permute(1, 2, 0).cpu().numpy()
        wandb.log({f"images/sample_image_{i}": wandb.Image(img)})

    print("Train:", train_metrics)
    print("Test:", test_metrics)

    return train_metrics, test_metrics

