import uuid

import wandb
from tqdm import tqdm
import torch

import os

from generators.ncsn_sampler_v2 import sample_many_ncsn
from models.refinet import RefineNet

from utils.checkpoint_manager import CheckpointManager
from utils.data_loader import load_cifar10
from utils.eval import get_metric_scores
from utils.ncsn_utils import linear_noise_scale, score_matching_loss

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

    batch_size = config.get("batch_size", 128)
    epochs = config.get("epochs", 100)
    device = torch.device(config.get('device', 'cuda'))
    lr = config.get("lr", 0.001)
    sigma_max = config.get("sigma_max", 1)
    sigma_min = config.get("sigma_min", 0.01)
    L = config.get("L", 10)

    train_dataloader, test_dataloader = load_cifar10(batch_size=batch_size)

    model = RefineNet(
        in_channels=3,
        hidden_channels=(128, 256, 512, 1024),
        n_noise_scale=L
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    noise_scales = linear_noise_scale(start=sigma_max, end=sigma_min, length=L).to(device)

    checkpoint_manager = CheckpointManager(
        base_dir=CHECKPOINT_DIR,
        run_id=run_id,
        max_checkpoints=3
    )
    start_epoch = 0
    latest_checkpoint = checkpoint_manager.get_latest_checkpoint()
    if latest_checkpoint and config.get('resume_training', False):
        models = {'s': model}
        optimizers = {"optimizer": optimizer}
        start_epoch, _ = checkpoint_manager.load_checkpoint(latest_checkpoint, models, optimizers)
        start_epoch += 1

    print("Starting training from epoch ", start_epoch)

    model.train()

    for epoch in range(start_epoch, epochs):
        print(f'Epoch {epoch}')
        epoch_loss = 0.
        for i, (x, _) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            x = x.to(device)
            loss = score_matching_loss(model, x, noise_scales)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch loss: {epoch_loss / len(train_dataloader)}')
        wandb.log(
            {
                "epoch/epoch": epoch,
                "epoch/loss": epoch_loss / len(train_dataloader),
            }
        )
        if (epoch + 1) % config.get("save_every", 5) == 0:
            models = {'s': model}
            optimizers = {"optimizer": optimizer}
            checkpoint_manager.save_checkpoint(epoch, models, optimizers, {})
    if config.get('eval', False):
        evaluate_ncsn(model, config, train_dataloader, test_dataloader, log_wandb=True)

def evaluate_ncsn(model, config, train_loader=None, test_loader=None, log_wandb=False):
    """
    Generates fake images and calculates FID/IS metrics against real data.
    This function calls `dsm_sampler`, which requires further Langevin-specific parameters.
    """
    model.eval()
    print("Generating fake images...")
    fake_images = sample_many_ncsn(model, config)

    print("Calculating FID and IS scores...")

    test_metrics = get_metric_scores(test_loader, fake_images, "test")
    train_metrics = get_metric_scores(train_loader, fake_images, "train")

    if log_wandb:
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
        if log_wandb:
            wandb.log({f"images/sample_image_{i}": wandb.Image(img)})

    print(f"Train: {train_metrics}")
    print(f"Test: {test_metrics}")
    return train_metrics, test_metrics