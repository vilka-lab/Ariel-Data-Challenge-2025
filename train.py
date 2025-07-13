import lightning as L
from tqdm import tqdm
import torch
import pandas as pd

from src.data import TransitDataModule
from src.model import ConvModel
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

def correct_shape(a: torch.Tensor) -> torch.Tensor:
    shape = list(a.shape)
    mid_size = shape.pop(1)
    shape[0] = shape[0] * mid_size
    return a.reshape(shape)

def validate_step(
        val_loader: torch.utils.data.DataLoader, 
        model: torch.nn.Module, 
        criterion: torch.nn.Module, 
        fabric: L.Fabric,
        best_val_loss: float
        ) -> torch.Tensor:
    
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for batch in tqdm(val_loader, desc="Validation"):
            inputs, targets, _ = batch
            outputs = model(inputs)
            if torch.isnan(outputs).any():
                raise ValueError("NAN!")
            
            val_loss += criterion(outputs, targets).item()

        val_loss /= len(val_loader)
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
        epoch: int,
        scheduler: torch.nn.Module
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
        # scheduler.step()

    fabric.print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
    
    # save last model
    fabric.save("last_model.pth", model.state_dict())
    
    return loss.item()

class CustomLoss(torch.nn.Module):
    def __init__(self, naive_mean: torch.Tensor, naive_std: torch.Tensor):
        super().__init__()
        self.mod1 = GaussianLogLikelihoodLoss(naive_mean=naive_mean, naive_std=naive_std)
        self.mod2 = torch.nn.MSELoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.mod1(y_pred, y_true) # + self.mod2(y_pred[:, :283], y_true)

def main() -> None:
    config = read_yaml("config.yaml")
    torch.set_float32_matmul_precision("high")

    fabric = L.Fabric(**config["fabric"])
    fabric.launch()

    L.seed_everything(config["seed"])
    train_loader, val_loader = make_dataloaders(config, fabric)

    model = ConvModel(**config["model"])
    optimizer = torch.optim.Adam(model.parameters(), **config["optimizer"])
    scheduler = ConstantCosineLR(
        optimizer,
        total_steps=config["train"]["max_epochs"] * len(train_loader),
        **config["scheduler"]
    )
    
    naive_mean, naive_std = calc_naive_stats()
    # criterion = GaussianLogLikelihoodLoss(naive_mean=naive_mean, naive_std=naive_std)
    # criterion = torch.nn.MSELoss()
    criterion = CustomLoss(naive_mean, naive_std)

    model, optimizer = fabric.setup(model, optimizer)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(config["train"]["max_epochs"]):
        # Train step
        train_loss = train_step(
            train_loader, model, optimizer, criterion, fabric, epoch, scheduler
        )
        train_losses.append(train_loss)
        
        # Validation step
        # val_loss = validate_step(
        #     val_loader, model, criterion, fabric, best_val_loss
        # )
        # best_val_loss = min(best_val_loss, val_loss)

        # val_losses.append(val_loss)

        # # print lr
        # fabric.print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # # plot_curves(train_losses, val_losses, save_path="loss_curves.png")
        # fabric.print("")



if __name__ == "__main__":
    main()