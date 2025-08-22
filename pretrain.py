import lightning as L
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import joblib
from pathlib import Path

from src.data import TransitDataModule, TransitDataset
from src.model import TransitModel
from src.loss import GaussianLogLikelihoodLoss
from src.utils import read_yaml, ConstantCosineLR, plot_curves
from src.generator import GeneratedDataProcessor




def make_dataloaders(config: dict, fabric: L.Fabric) -> tuple[L.LightningDataModule, L.LightningDataModule]:
    df = pd.read_csv("generated_data.csv")
    wavelengths = pd.read_csv("ariel-data-challenge-2025/wavelengths.csv")
    processor = GeneratedDataProcessor(df, cache_folder="cached_generated_data", wavelengths=wavelengths)

    ref_gt = pd.read_csv("ariel-data-challenge-2025/train.csv")
    ref_meta = pd.read_csv("ariel-data-challenge-2025/train_star_info.csv")
    gt = df[ref_gt.columns]
    meta = df[ref_meta.columns]

    dataset = TransitDataset(processor, gt=gt, meta=meta, output_stats=None)
    joblib.dump(dataset.output_stats, "stats_pretrain.joblib")

    train_loader = DataLoader(
        dataset, 
        batch_size=config["data_module"]["batch_size"], 
        shuffle=True, 
        drop_last=True, 
        num_workers=config["data_module"]["num_workers"]
        )
    
    
    config["data_module"]["full_train"] = True
    dm = TransitDataModule(**config["data_module"], random_state=config["seed"])
    dm.setup(load_stats=True)
    val_loader = dm.val_dataloader()
    val_loader.dataset.output_stats = dataset.output_stats

    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)
    return train_loader, val_loader

def calc_naive_stats() -> tuple[float, float]:
    gt = pd.read_csv("ariel-data-challenge-2025/train.csv")
    values = gt.iloc[:, 1:].values
    mean = values.mean()
    std = values.std()
    return mean, std

EPOCH_DIV = 50


@torch.no_grad()
def validate_step(
        val_loader: torch.utils.data.DataLoader, 
        model: torch.nn.Module, 
        criterion: torch.nn.Module, 
        fabric: L.Fabric,
        best_val_loss: float,
        epoch: int,
        fold: int
        ) -> torch.Tensor:
    
    model.eval()

    loader_outputs, loader_targets = [], []
    
    for batch in val_loader:
        outputs = model(batch)
        if torch.isnan(outputs).any():
            raise ValueError("NAN!")
        
        outputs[:, :283] = val_loader.dataset.denorm(outputs[:, :283], "targets")
        
        loader_outputs.append(outputs)
        loader_targets.append(val_loader.dataset.denorm(batch["targets"], "targets"))

    loader_outputs = torch.cat(loader_outputs, dim=0)
    loader_targets = torch.cat(loader_targets, dim=0)
    
    val_loss = criterion(loader_outputs, loader_targets).item()

    if epoch % EPOCH_DIV == 0:
        fabric.print(f"Validation Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        fabric.print(f"New best validation loss: {best_val_loss:.4f}, saving model...")
        fabric.save(f"best_model_{fold}.pth", model.state_dict())

    return val_loss


def train_step(
        train_loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        fabric: L.Fabric,
        epoch: int,
        fold: int
        ) -> torch.Tensor:
    model.train()

    for batch in train_loader:

        optimizer.zero_grad()
        outputs = model(batch)
        if torch.isnan(outputs).any():
            raise ValueError("NAN!")
        
        outputs[:, :283] = train_loader.dataset.denorm(outputs[:, :283], "targets")
        
        loss = criterion(outputs, train_loader.dataset.denorm(batch["targets"], "targets"))

        # clip gradients
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        fabric.backward(loss)
        optimizer.step()
    
    if epoch % EPOCH_DIV == 0:
        fabric.print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
    
    # save last model
    fabric.save(f"last_model_{fold}.pth", model.state_dict())
    
    return loss.item()

class CustomLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mod = torch.nn.MSELoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.mod(y_pred, y_true) * 1e6

def main(config: dict) -> float:
    torch.set_float32_matmul_precision("high")
    # torch.autograd.set_detect_anomaly(True)

    fabric = L.Fabric(**config["fabric"])
    fabric.seed_everything(config["seed"], workers=True)
    fabric.launch()

    train_loader, val_loader = make_dataloaders(config, fabric)

    model = TransitModel()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fabric.print(f"Number of trainable parameters: {num_params / 1e6:.4f}M")

    optimizer = torch.optim.Adam(model.parameters(), **config["optimizer"])
    scheduler = ConstantCosineLR(
        optimizer,
        total_steps=config["train"]["max_epochs"] * len(train_loader),
        **config["scheduler"]
    )
    
    criterion = GaussianLogLikelihoodLoss(*calc_naive_stats(), fgs_weight=57.846)

    model, optimizer = fabric.setup(model, optimizer)

    # xavier initialization
    model.apply(lambda x: torch.nn.init.xavier_normal_(x.weight) if isinstance(x, torch.nn.Linear) else None)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    fold = "pretrain"
    for epoch in range(config["train"]["max_epochs"]):
        # Train step
        train_loss = train_step(
            train_loader, model, optimizer, criterion, fabric, epoch, fold=fold
        )
        train_losses.append(np.clip(train_loss, a_min=None, a_max=1.0))
        
        # Validation step
        val_loss = validate_step(
            val_loader, model, criterion, fabric, best_val_loss, epoch, fold=fold
        )
        best_val_loss = min(best_val_loss, val_loss)

        val_losses.append(np.clip(val_loss, a_min=None, a_max=1.0))
        
        if epoch % EPOCH_DIV == 0:
            plot_curves(train_losses, val_losses, save_path=f"loss_curves_{fold}.png")
            fabric.print("")

        scheduler.step()

    print(f"Done for {fold = }, best val loss: {best_val_loss:.4f}")
    return best_val_loss


if __name__ == "__main__":
    config = read_yaml("config_pretrain.yaml")
    loss_val = main(config)