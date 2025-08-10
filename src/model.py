import torch
import torch.nn as nn
import torch.nn.functional as F

class MeanTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_channels = 283
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(8),

            nn.Conv1d(8, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.linear = nn.Sequential(
            nn.Linear(576, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 1)
        )
        self.out = nn.Linear(self.num_channels, self.num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = x.reshape(B * self.num_channels, 1, -1)
        x = self.conv(x)

        x = x.view(x.size(0), -1)
        x = self.linear(x)

        x = x.view(B, -1)
        x = self.out(x)
        return x
    

class UncertaintyModel(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.weights = nn.Sequential(
            nn.BatchNorm1d(7 + 283),
            nn.Dropout(0.5),
            nn.Linear(7 + 283, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 283)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.weights(x) - 8)
    

def print_stats(a: torch.Tensor, name: str) -> None:
    print(f"{name} mean: {a.mean()}")
    print(f"{name} std: {a.std()}")
    print(f"{name} min: {a.min()}")
    print(f"{name} max: {a.max()}")
    print(f"{name} shape: {a.shape}")
    print("=" * 20)

class TransitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tower = MeanTower()
        self.unc_model = UncertaintyModel()
        self.static_coef = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        v = torch.transpose(x["curves"], 1, 2) # [N, 283, 187]

        spectre = self.tower(v)

        unc_input = torch.cat([spectre, x["meta"]], dim=1)
        unc_out = self.unc_model(unc_input)
        
        spectre = spectre + x["static_component"] * self.static_coef
        out = torch.cat([spectre, unc_out], dim=1)
        return out
    