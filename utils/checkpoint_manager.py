import os
import torch

class CheckpointManager:
    def __init__(self, base_dir, run_id, max_checkpoints=3):
        self.checkpoint_dir : str | bytes = os.path.join(base_dir, run_id)
        self.max_checkpoints = max_checkpoints
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, epoch, models_dict, optimizers_dict, metrics):
        """
        Save checkpoint into the run-specific folder (base_dir/run_id).
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')

        checkpoint = {
            'epoch': epoch,
            'metrics': metrics
        }

        for name, model in models_dict.items():
            checkpoint[f'{name}_state_dict'] = model.state_dict()

        for name, optimizer in optimizers_dict.items():
            checkpoint[f'{name}_state_dict'] = optimizer.state_dict()

        torch.save(checkpoint, checkpoint_path)
        self._cleanup_old_checkpoints()

    def load_checkpoint(self, checkpoint_path, models_dict, optimizers_dict):
        """
        Load checkpoint from a specific path.
        """
        checkpoint = torch.load(checkpoint_path)
        for name, model in models_dict.items():
            model.load_state_dict(checkpoint[f'{name}_state_dict'])
        for name, optimizer in optimizers_dict.items():
            optimizer.load_state_dict(checkpoint[f'{name}_state_dict'])
        return checkpoint['epoch'], checkpoint['metrics']

    def get_latest_checkpoint(self):
        """Find the most recent *regular* checkpoint."""
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('checkpoint_epoch_')]
        if not checkpoints:
            return None
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        return os.path.join(self.checkpoint_dir, checkpoints[-1])

    def _cleanup_old_checkpoints(self):
        """Keep only the most recent *regular* checkpoints."""
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('checkpoint_epoch_')]
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

        for old_checkpoint in checkpoints[:-self.max_checkpoints]:
            os.remove(os.path.join(self.checkpoint_dir, old_checkpoint))