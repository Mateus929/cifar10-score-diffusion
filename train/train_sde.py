import os
import uuid
from functools import partial

import torch

import wandb
from torch.optim import Adam
import tqdm

from generators.sde_sampler import sample_many_dsm
from models.diffusion_scorenet import ScoreNet
from utils.checkpoint_manager import CheckpointManager
from utils.data_loader import load_cifar10
from utils.diffusion_utils import marginal_prob_std, loss_diffusion
from utils.eval import get_metric_scores

BASE_WORK_DIR = os.environ.get("BASE_WORK_DIR")
CHECKPOINT_DIR = os.path.join(BASE_WORK_DIR, "checkpoints")


def train_sde(config):

    run_id = config.get("run_id", str(uuid.uuid4().hex[:8]))
    print("Current run id: ", run_id)
    wandb.init(
        project="diffusion-score-matching",
        name=config["run_name"],
        config=config,
        id=run_id,
        resume="allow"
    )

    sigma = config.get("sigma", 25.0)
    marginal_prob_std_fn = partial(marginal_prob_std, sigma=sigma)
    score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
    device = config.get("device", "cuda")
    score_model = score_model.to(device)

    n_epochs = config.get("epochs", 100)
    batch_size = config.get("batch_size", 128)
    lr = config.get("lr", 0.001)

    data_loader, test_loader = load_cifar10(batch_size=batch_size)

    optimizer = Adam(score_model.parameters(), lr=lr)

    checkpoint_manager = CheckpointManager(
        base_dir=CHECKPOINT_DIR,
        run_id=run_id,
        max_checkpoints=3
    )
    start_epoch = 0
    latest_checkpoint = checkpoint_manager.get_latest_checkpoint()
    if latest_checkpoint and config.get('resume_training', False):
        models = {'s': score_model}
        optimizers = {"optimizer": optimizer}
        start_epoch, _ = checkpoint_manager.load_checkpoint(latest_checkpoint, models, optimizers)
        start_epoch += 1

    print("Starting training from epoch ", start_epoch)

    tqdm_epoch = tqdm.notebook.trange(start_epoch, n_epochs)
    for epoch in tqdm_epoch:
      avg_loss = 0.
      num_items = 0
      for x, _ in data_loader:
        x = x.to(device)
        loss = loss_diffusion(score_model, x, marginal_prob_std_fn)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]

      wandb.log(
          {
              "epoch/epoch": epoch,
              "epoch/loss": avg_loss / num_items,
          }
      )

      tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))

      if (epoch + 1) % config.get("save_every", 5) == 0:
          models = {'s': score_model}
          optimizers = {"optimizer": optimizer}
          checkpoint_manager.save_checkpoint(epoch, models, optimizers, {})

    if config.get('eval', False):
        evaluate_sde(score_model, config, data_loader, test_loader, log_wandb=True)

def evaluate_sde(model, config, train_loader=None, test_loader=None, log_wandb=False):
    """
    Generates fake images and calculates FID/IS metrics against real data.
    This function calls `dsm_sampler`, which requires further Langevin-specific parameters.
    """
    model.eval()
    device = next(model.parameters()).device
    fake_images = sample_many_dsm(model, config)

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