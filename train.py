import lightning as L
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np

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
    total_loss = 0
    with tqdm(train_loader, desc=f"Epoch {epoch + 1}", disable=True) as pbar:
        for batch in train_loader:

            optimizer.zero_grad()
            outputs = model(batch)
            if torch.isnan(outputs).any():
                raise ValueError("NAN!")
            
            outputs[:, :283] = train_loader.dataset.denorm(outputs[:, :283], "targets")
            
            loss = criterion(outputs, train_loader.dataset.denorm(batch["targets"], "targets"))

            # clip gradients
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            total_loss += loss.item()

            fabric.backward(loss)
            optimizer.step()
            pbar.update(1)
            pbar.set_postfix({"loss": loss.item()})

    loss = total_loss / len(train_loader)
    
    if epoch % EPOCH_DIV == 0:
        fabric.print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")
    
    # save last model
    fabric.save(f"last_model_{fold}.pth", model.state_dict())
    
    return loss

class CustomLoss(torch.nn.Module):
    def __init__(self, coef: float):
        super().__init__()
        self.mod1 = torch.nn.MSELoss()
        self.mod2 = GaussianLogLikelihoodLoss(*calc_naive_stats(), fgs_weight=57.846)
        self.coef = coef

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return (self.mod1(y_pred[:, :283], y_true) * 1e6) * self.coef + self.mod2(y_pred, y_true) * (1 - self.coef)

def main(config: dict) -> float:
    torch.set_float32_matmul_precision("high")
    # torch.autograd.set_detect_anomaly(True)

    fabric = L.Fabric(**config["fabric"])
    fabric.seed_everything(config["seed"], workers=True)
    fabric.launch()

    train_loader, val_loader = make_dataloaders(config, fabric)

    model = TransitModel(pretrained=True)
    if config["model"]["checkpoint"] is not None:
        model.load_state_dict(torch.load(config["model"]["checkpoint"]))
        fabric.print(f"Loaded model from {config['model']['checkpoint']}")

    # xavier initialization
    # model.apply(lambda x: torch.nn.init.xavier_normal_(x.weight) if isinstance(x, torch.nn.Linear) else None)

    # model = torch.compile(model, mode="default")

    optimizer = torch.optim.AdamW(model.parameters(), **config["optimizer"])
    scheduler = ConstantCosineLR(
        optimizer,
        total_steps=config["train"]["max_epochs"] * len(train_loader),
        **config["scheduler"]
    )
    
    if config["gauss_loss"]:
        train_criterion = CustomLoss(coef=0.0)
        val_criterion = CustomLoss(coef=0.0)
        # model.freeze_backbone()
    else:
        train_criterion = CustomLoss(coef=0.9)
        val_criterion = CustomLoss(coef=0.0)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fabric.print(f"Number of trainable parameters: {num_params / 1e6:.4f}M")


    model, optimizer = fabric.setup(model, optimizer)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    fold = config["data_module"]["fold"]
    for epoch in range(config["train"]["max_epochs"]):
        # Train step
        train_loss = train_step(
            train_loader, model, optimizer, train_criterion, fabric, epoch, fold=fold
        )

        # Validation step
        val_loss = validate_step(
            val_loader, model, val_criterion, fabric, best_val_loss, epoch, fold=fold
        )
        best_val_loss = min(best_val_loss, val_loss)
        
        if config["gauss_loss"]:
            val_loss = np.clip(val_loss, a_min=None, a_max=1.0)
            train_loss = np.clip(train_loss, a_min=None, a_max=1.0)
        else:
            val_loss = np.clip(val_loss, a_min=None, a_max=2.0)
            train_loss = np.clip(train_loss, a_min=None, a_max=2.0)
        
        val_losses.append(val_loss)
        train_losses.append(train_loss)

        # # print lr
        # fabric.print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        if epoch % EPOCH_DIV == 0:
            plot_curves(train_losses, val_losses, save_path=f"loss_curves_{fold}.png")
            fabric.print("")

        scheduler.step()

    print(f"Done for {fold = }, best val loss: {best_val_loss:.4f}")
    return best_val_loss


if __name__ == "__main__":
    config = read_yaml("config.yaml")
    
    losses = []
    for fold in range(5):
        print(f"Fold {fold}")
        config["data_module"]["fold"] = fold
        config["gauss_loss"] = False

        loss_val = main(config)

        # config["gauss_loss"] = True
        # config["model"]["checkpoint"] = f"best_model_{fold}.pth"
        
        # loss_val = main(config)
        losses.append(loss_val)

    print(f"Losses: {losses}")
    print(f"Mean loss: {np.mean(losses):.4f}")