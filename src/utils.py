import yaml
import math
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt


def read_yaml(file_path: str) -> dict:
    with open(file_path, "r") as f:
        return yaml.safe_load(f)
    

class ConstantCosineLR(_LRScheduler):
    """
    Constant learning rate followed by CosineAnnealing.
    """
    def __init__(
        self, 
        optimizer,
        total_steps, 
        pct_cosine, 
        last_epoch=-1,
        ):
        self.total_steps = total_steps
        self.milestone = int(total_steps * (1 - pct_cosine))
        self.cosine_steps = max(total_steps - self.milestone, 1)
        self.min_lr = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.milestone:
            factor = 1.0
        else:
            s = step - self.milestone
            factor = 0.5 * (1 + math.cos(math.pi * s / self.cosine_steps))
        return [lr * factor for lr in self.base_lrs]
    

def plot_curves(train_losses: list[float], val_losses: list[float], save_path: str | None = None) -> None:
    """
    Plots training and validation loss curves.
    
    :param train_losses: List of training losses.
    :param val_losses: List of validation losses.
    :param save_path: Optional path to save the plot. If None, the plot will be shown.
    """
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()

    save_path = save_path or "loss_curves.png"
    plt.savefig(save_path, bbox_inches='tight')