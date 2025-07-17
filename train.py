import lightning as L
from tqdm import tqdm
import torch
import pandas as pd

from src.data import TransitDataModule
from src.model import TransitModel
from src.loss import GaussianLogLikelihoodLoss
from src.utils import read_yaml, ConstantCosineLR, plot_curves


def make_dataloaders(config: dict, fabric: L.Fabric) -> tuple[L.LightningDataModule, L.LightningDataModule]:
    dm = TransitDataModule(**config["data_module"], random_state=config["seed"])
    dm.setup()
    train_loader = fabric.setup_dataloaders(dm.train_dataloader())
    val_loader = fabric.setup_dataloaders(dm.val_dataloader())
    return train_loader, val_loader

def calc_naive_stats() -> tuple[float, float]:
    gt = pd.read_csv("ariel-data-challenge-2025/train.csv")
    values = gt.iloc[:, 1:].values
    mean = values.mean()
    std = values.std()
    return mean, std


@torch.no_grad()
def validate_step(
        val_loader: torch.utils.data.DataLoader, 
        model: torch.nn.Module, 
        criterion: torch.nn.Module, 
        fabric: L.Fabric,
        best_val_loss: float
        ) -> torch.Tensor:
    
    model.eval()
    total_outputs = []

    for i in range(10):
        loader_outputs, loader_targets = [], []
        
        for batch in val_loader:
            inputs, targets = batch
            outputs = model(inputs)
            if torch.isnan(outputs).any():
                raise ValueError("NAN!")
            
            loader_outputs.append(outputs)
            loader_targets.append(targets)

        loader_outputs = torch.cat(loader_outputs, dim=0)
        loader_targets = torch.cat(loader_targets, dim=0)
        total_outputs.append(loader_outputs)
    
    total_outputs = sum(total_outputs) / 10
    val_loss = criterion(total_outputs, loader_targets).item()
    fabric.print(f"Validation Loss: {val_loss}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        fabric.print(f"New best validation loss: {best_val_loss}, saving model...")
        fabric.save("best_model.pth", model.state_dict())

    return val_loss


def train_step(
        train_loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        fabric: L.Fabric,
        epoch: int
        ) -> torch.Tensor:
    model.train()

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
        inputs, targets = batch

        optimizer.zero_grad()
        outputs = model(inputs)
        if torch.isnan(outputs).any():
            raise ValueError("NAN!")
        
        loss = criterion(outputs, targets)

        # clip gradients
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        fabric.backward(loss)
        optimizer.step()

    fabric.print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
    
    # save last model
    fabric.save("last_model.pth", model.state_dict())
    
    return loss.item()

class CustomLoss(torch.nn.Module):
    def __init__(self, naive_mean: torch.Tensor, naive_std: torch.Tensor, mse_weight: float = 0.0):
        super().__init__()
        self.mod1 = GaussianLogLikelihoodLoss(naive_mean=naive_mean, naive_std=naive_std)
        self.mod2 = torch.nn.MSELoss()
        self.mse_weight = mse_weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.mod1(y_pred, y_true) + self.mod2(y_pred[:, :283], y_true) * self.mse_weight

def main() -> None:
    config = read_yaml("config.yaml")
    torch.set_float32_matmul_precision("highest")

    fabric = L.Fabric(**config["fabric"])
    fabric.seed_everything(config["seed"], workers=True)
    fabric.launch()

    train_loader, val_loader = make_dataloaders(config, fabric)

    model = TransitModel(**config["model"])

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fabric.print(f"Number of trainable parameters: {num_params / 1e6:.4f}M")

    optimizer = torch.optim.AdamW(model.parameters(), **config["optimizer"])
    scheduler = ConstantCosineLR(
        optimizer,
        total_steps=config["train"]["max_epochs"] * len(train_loader),
        **config["scheduler"]
    )
    
    naive_mean, naive_std = calc_naive_stats()
    criterion = CustomLoss(naive_mean, naive_std, mse_weight=10.0)
    val_criterion = CustomLoss(naive_mean, naive_std, mse_weight=0.0)

    model, optimizer = fabric.setup(model, optimizer)

    # xavier initialization
    model.apply(lambda x: torch.nn.init.xavier_normal_(x.weight) if isinstance(x, torch.nn.Linear) else None)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(config["train"]["max_epochs"]):

        # Train step
        train_loss = train_step(
            train_loader, model, optimizer, criterion, fabric, epoch
        )
        train_losses.append(train_loss)
        
        # Validation step
        val_loss = validate_step(
            val_loader, model, val_criterion, fabric, best_val_loss
        )
        best_val_loss = min(best_val_loss, val_loss)

        val_losses.append(val_loss)

        # # print lr
        # fabric.print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        plot_curves(train_losses, val_losses, save_path="loss_curves.png")
        fabric.print("")
        scheduler.step()



if __name__ == "__main__":
    main()