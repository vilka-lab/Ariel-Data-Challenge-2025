import lightning as L
from tqdm import tqdm
import torch
import pandas as pd

from src.data import TransitDataModule
from src.model import TransitModel, UncertaintyModel
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
        pred_model: torch.nn.Module,
        unc_model: torch.nn.Module, 
        criterion: torch.nn.Module, 
        fabric: L.Fabric,
        best_val_loss: float,
        epoch: int
        ) -> torch.Tensor:
    
    unc_model.eval()

    loader_outputs, loader_targets = [], []
    
    for batch in val_loader:
        outputs = pred_model(batch)

        unc_input = torch.cat([outputs, batch["meta"]], dim=1)
        unc_outputs = unc_model(unc_input)

        if torch.isnan(unc_outputs).any():
            raise ValueError("NAN!")
        
        outputs = val_loader.dataset.denorm(outputs, "targets")
        outputs = torch.cat([outputs, unc_outputs], dim=1)
        
        loader_outputs.append(outputs)
        loader_targets.append(val_loader.dataset.denorm(batch["targets"], "targets"))

    loader_outputs = torch.cat(loader_outputs, dim=0)
    loader_targets = torch.cat(loader_targets, dim=0)
    
    val_loss = criterion(loader_outputs, loader_targets).item()

    if epoch % EPOCH_DIV == 0:
        fabric.print(f"Validation Loss: {val_loss}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        fabric.print(f"New best validation loss: {best_val_loss}, saving model...")
        fabric.save("best_unc_model.pth", unc_model.state_dict())

    return val_loss


def train_step(
        train_loader: torch.utils.data.DataLoader,
        pred_model: torch.nn.Module,
        unc_model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        fabric: L.Fabric,
        epoch: int
        ) -> torch.Tensor:
    unc_model.train()

    for batch in train_loader:
        optimizer.zero_grad()
        
        outputs = pred_model(batch)

        unc_input = torch.cat([outputs, batch["meta"]], dim=1)
        unc_outputs = unc_model(unc_input)

        if torch.isnan(unc_outputs).any():
            raise ValueError("NAN!")
        
        outputs = train_loader.dataset.denorm(outputs, "targets")
        outputs = torch.cat([outputs, unc_outputs], dim=1)
        
        loss = criterion(outputs, train_loader.dataset.denorm(batch["targets"], "targets"))

        # clip gradients
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        fabric.backward(loss)
        optimizer.step()
    
    if epoch % EPOCH_DIV == 0:
        fabric.print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
    
    # save last model
    fabric.save("last_unc_model.pth", unc_model.state_dict())
    
    return loss.item()

def load_pred_model() -> TransitModel:
    model = TransitModel()
    model.load_state_dict(torch.load("best_model.pth", map_location=torch.device("cpu")))
    model.eval()

    # freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    return model


def main() -> None:
    config = read_yaml("config_unc.yaml")
    torch.set_float32_matmul_precision("high")

    fabric = L.Fabric(**config["fabric"])
    fabric.seed_everything(config["seed"], workers=True)
    fabric.launch()

    train_loader, val_loader = make_dataloaders(config, fabric)
    
    pred_model = load_pred_model()
    pred_model = fabric.setup(pred_model)
    
    unc_model = UncertaintyModel()

    num_params = sum(p.numel() for p in unc_model.parameters() if p.requires_grad)
    fabric.print(f"Number of trainable parameters: {num_params / 1e6:.4f}M")

    optimizer = torch.optim.Adam(unc_model.parameters(), **config["optimizer"])
    scheduler = ConstantCosineLR(
        optimizer,
        total_steps=config["train"]["max_epochs"] * len(train_loader),
        **config["scheduler"]
    )
    
    criterion = GaussianLogLikelihoodLoss(*calc_naive_stats(), fgs_weight=57.846)

    unc_model, optimizer = fabric.setup(unc_model, optimizer)

    # xavier initialization
    unc_model.apply(lambda x: torch.nn.init.xavier_normal_(x.weight) if isinstance(x, torch.nn.Linear) else None)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(config["train"]["max_epochs"]):

        # Train step
        train_loss = train_step(
            train_loader, pred_model, unc_model, optimizer, criterion, fabric, epoch
        )
        train_losses.append(train_loss)
        
        # Validation step
        val_loss = validate_step(
            val_loader, pred_model, unc_model, criterion, fabric, best_val_loss, epoch
        )
        best_val_loss = min(best_val_loss, val_loss)

        val_losses.append(val_loss)

        # # print lr
        # fabric.print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        if epoch % EPOCH_DIV == 0:
            plot_curves(train_losses, val_losses, save_path="unc_loss_curves.png")
            fabric.print("")

        scheduler.step()

    print(f"Done, best val loss: {best_val_loss}")


if __name__ == "__main__":
    main()