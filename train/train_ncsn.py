import os
import torch
import uuid

from generators.ncsn_sampler import nscn_sampler
from models.NSCNnet import NSCNModel
from utils.data_loader import load_cifar10_01
from utils.eval import get_metric_scores
from utils.losses import ncsn_loss
import wandb
from utils.checkpoint_manager import CheckpointManager
import numpy as np

BASE_WORK_DIR = os.environ.get("BASE_WORK_DIR")
CHECKPOINT_DIR = os.path.join(BASE_WORK_DIR, "checkpoints")

def train_ncsn(config):

    run_id = config.get("run_id", str(uuid.uuid4().hex[:8]))
    print("Current run id: ", run_id)
    wandb.init(
        project="diffusion-score-matching",
        name=config["run_name"],
        config=config,
        id=run_id,
        resume="allow"
    )

    sigma_min = config.get("sigma_min", 0.01)
    sigma_max = config.get("sigma_max", 1.0)
    L = config.get("L", 10)

    total_epochs = config["epochs"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sigmas = torch.tensor(
        sigma_max * (sigma_min / sigma_max) ** (np.arange(L) / (L - 1)),
        dtype=torch.float32,
        device=device
    )

    s = NSCNModel().to(device)
    optimizer = torch.optim.Adam(s.parameters(), lr=config["lr"])
    train_loader, test_loader = load_cifar10_01(batch_size=config.get("batch_size", 128))

    checkpoint_manager = CheckpointManager(
        base_dir=CHECKPOINT_DIR,
        run_id=run_id,
        max_checkpoints=3
    )
    start_epoch = 0
    latest_checkpoint = checkpoint_manager.get_latest_checkpoint()
    if latest_checkpoint and config.get('resume_training', False):
        models = {'s': s}
        optimizers = {"optimizer": optimizer}
        start_epoch, _ = checkpoint_manager.load_checkpoint(latest_checkpoint, models, optimizers)
        start_epoch += 1

    print("Starting training from epoch ", start_epoch)

    for epoch in range(start_epoch, total_epochs):
        s.train()

        epoch_loss = 0

        for x, _ in train_loader:
            optimizer.zero_grad()
            x = x.to(device)

            indices = torch.randint(0, L, (x.size(0),), device=x.device)
            sigma = sigmas[indices].view(-1, 1, 1, 1)

            epsilon = torch.randn_like(x)
            x_noisy = x + epsilon * sigma

            y = s(x_noisy, sigma)

            loss = ncsn_loss(y, epsilon, sigma)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        wandb.log(
            {
                "epoch/epoch": epoch,
                "epoch/loss": epoch_loss / len(train_loader),
            }
        )

        print(f"{epoch}/{total_epochs}, loss: {epoch_loss / len(train_loader)}")

        if (epoch + 1) % config.get("save_every", 10) == 0:
            models = {'s': s}
            optimizers = {"optimizer": optimizer}
            checkpoint_manager.save_checkpoint(epoch, models, optimizers, {})

    if config.get("eval", False):
        evaluate_ncsn(s, config, train_loader, test_loader, True)

    wandb.finish()

    return s


def evaluate_ncsn(model, config, train_loader=None, test_loader=None, log_wanbd=False):
    """
    Generates fake images and calculates FID/IS metrics against real data.
    This function calls `dsm_sampler`, which requires further Langevin-specific parameters.
    """
    model.eval()
    device = next(model.parameters()).device
    fake_images = nscn_sampler(model, config, device)

    print("Calculating FID and IS scores...")

    test_metrics = get_metric_scores(test_loader, fake_images, "test")
    train_metrics = get_metric_scores(train_loader, fake_images, "train")

    if log_wanbd:
        wandb.log({
            "metrics/fid_test": test_metrics['FID'],
            "metrics/is_test_mean": test_metrics['IS_MEAN'],
            "metrics/is_test_std": test_metrics['IS_STD'],
            "metrics/fid_train": train_metrics['FID'],
            "metrics/is_train_mean": train_metrics['IS_MEAN'],
            "metrics/is_train_std": train_metrics['IS_STD'],
        })

    samples = fake_images[:4]
    samples = ((samples + 1.0) * 0.5 * 255).clamp(0, 255).byte()

    for i, img in enumerate(samples):
        img = img.permute(1, 2, 0).cpu().numpy()
        if log_wanbd:
            wandb.log({f"images/sample_image_{i}": wandb.Image(img)})

    print(f"Train: {train_metrics}")
    print(f"Test: {test_metrics}")
    return train_metrics, test_metrics
