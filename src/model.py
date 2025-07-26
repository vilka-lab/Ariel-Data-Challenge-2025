import torch
import torch.nn as nn
import torch.nn.functional as F

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

            nn.Linear(2304, 500),
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
    

class TransitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean_tower = MeanTower()
        self.transit_tower = TransitTower()

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        mean = self.mean_tower(x["white_curve"])
        transit = self.transit_tower(x["transit_map"])
        return mean + transit