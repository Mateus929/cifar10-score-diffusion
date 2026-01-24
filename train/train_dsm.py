import os
import torch
import uuid


from generators.dsm_sampler import dsm_sampler
from models.unet import DiffusionUNet
from models.simple_score_net import ScoreNet
from utils.data_loader import load_cifar10
from utils.eval import get_metric_scores
from utils.losses import dsm_loss
import wandb
from utils.checkpoint_manager import CheckpointManager

BASE_WORK_DIR = os.environ.get("BASE_WORK_DIR")
CHECKPOINT_DIR = os.path.join(BASE_WORK_DIR, "checkpoints")

model_mappings = {
    "score-net": ScoreNet(),
    "unet": DiffusionUNet(),
}


def train_dsm(config):
    """
    Main training loop for Denoising Score Matching using a VE-SDE.

    Config Requirements:
    - run_name (str): Unique name for the W&B run.
    - epochs (int): Total training epochs.
    - lr (float): Learning rate for Adam optimizer.
    - sigma_min (float): Minimum noise level. Default: 0.01.
    - sigma_max (float): Maximum noise level. Default: 50.0 (recommended for CIFAR).
    - T (float): Max time horizon. Default: 1.
    - batch_size (int): Data batch size. Default: 128.
    - save_every (int): Epoch interval to save checkpoints. Default: 10.
    - resume_training (bool): If True, loads latest checkpoint. Default: False.
    - eval (bool): Whether to run evaluation after training. Default: False.
    - run_id (str, optional): Specific ID to resume a W&B run.
    - model_name (str, optional): Architecture that will be used. Default: unet.

    Remarks:
    - In case of eval = True, refer to `dsm_sampler` for configuration parameters used during the evaluation phase.
    """
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
    sigma_max = config.get("sigma_max", 50.0)
    T = config.get("T", 1)

    total_epochs = config["epochs"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = config.get("model_name", "unet")
    s = model_mappings[model_name].to(device)
    optimizer = torch.optim.Adam(s.parameters(), lr=config["lr"])
    train_loader, test_loader = load_cifar10(batch_size=config.get("batch_size", 128))

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

        for batch_index, (x, _) in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.to(device)

            cur_batch_size = x.size(0)
            t = (torch.rand(cur_batch_size, 1, 1, 1, device=device) * 0.999 + 0.001) * T
            sigma = sigma_min * torch.sqrt((sigma_max / sigma_min) ** (2 * t / T) - 1)
            epsilon = torch.randn_like(x)
            x_noisy = x + epsilon * sigma
            y = s(x_noisy, t)

            lambda_t = sigma ** 2
            loss = dsm_loss(y, epsilon, sigma, lambda_t)
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
        evaluate_dsm(s, config, train_loader, test_loader, True)

    wandb.finish()

    return s


def evaluate_dsm(model, config, train_loader=None, test_loader=None, log_wanbd=False):
    """
    Generates fake images and calculates FID/IS metrics against real data.
    This function calls `dsm_sampler`, which requires further Langevin-specific parameters.
    """
    model.eval()
    device = next(model.parameters()).device
    fake_images = dsm_sampler(model, config, device)

    print("Calculating FID and IS scores...")

    test_metrics = get_metric_scores(test_loader, fake_images)
    train_metrics = get_metric_scores(train_loader, fake_images)

    if log_wanbd:
        wandb.log({
            "metrics/fid_test": test_metrics['FID'],
            "metrics/is_test_mean": test_metrics['IS_MEAN'],
            "metrics/is_test_std": test_metrics['IS_STD'],
            "metrics/fid_train": train_metrics['FID'],
            "metrics/is_train_mean": train_metrics['IS_MEAN'],
            "metrics/is_train_std": train_metrics['IS_STD'],
        })

    print(f"Train: {train_metrics}")
    print(f"Test: {test_metrics}")
    return train_metrics, test_metrics
