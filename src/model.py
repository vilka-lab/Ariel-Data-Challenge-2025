import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class MeanTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(32),

            nn.Conv1d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),

            nn.Linear(29440, 500),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(100, 1)
        )
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights(x)
    

class TransitTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(3, 1), padding="same"),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 1)),
                nn.BatchNorm2d(32),

                nn.Conv2d(32, 64, kernel_size=(3, 1), padding="same"),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 1)),

                nn.Conv2d(64, 128, kernel_size=(3, 1), padding="same"),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 1)),

                nn.Conv2d(128, 256, kernel_size=(3, 1), padding="same"),
                nn.ReLU(),

                nn.Conv2d(256, 32, kernel_size=(1, 3), padding='same'),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 2)),
                nn.BatchNorm2d(32),

                nn.Conv2d(32, 64, kernel_size=(1, 3), padding='same'),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 2)),

                nn.Conv2d(64, 128, kernel_size=(1, 3), padding='same'),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 2)),

                nn.Conv2d(128, 256, kernel_size=(1, 3), padding='same'),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 2)),

                nn.Flatten(),
                
                nn.Linear(21760, 700),  # Adjust input size based on pooling
                nn.ReLU(),

                nn.Dropout(0.2),
                nn.Linear(700, 283)
            )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights(x)
    

class UncertaintyModel(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.weights = nn.Sequential(
            nn.BatchNorm1d(7 + 283),
            nn.Dropout(0.2),
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

# class TransitModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.transit_tower = TransitTower()
#         self.unc_model = UncertaintyModel()
#         self.static_coef = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, requires_grad=True))


#     def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
#         transit = self.transit_tower(x["transit_map"])
#         spectre = transit 

#         unc_input = torch.cat([spectre, x["meta"]], dim=1)
#         unc_out = self.unc_model(unc_input)
        
#         spectre = spectre + x["static_component"] * self.static_coef
#         out = torch.cat([spectre, unc_out], dim=1)
#         return out
    

class TransitModel(nn.Module):
    def __init__(self, hidden_dim: int = 512, in_features: int = 293):
        super().__init__()
        self.weights = nn.Sequential(
            nn.Dropout(0.1), 
            nn.Linear(in_features, hidden_dim),

            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),

            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 283*2),
        )
        self.static_coef = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, requires_grad=True))


    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.cat([batch["features"], batch["meta"], batch["static_component"]], dim=1)
        out = self.weights(x)
        out[:, 283:] = torch.exp(out[:, 283:] - 8)
        out[:, :283] += batch["static_component"] * self.static_coef
        return out