import pandas as pd
from src.data import DataProcessor, TransitTrainDataset, TransitTestDataset
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchcnnbuilder.models import ForecasterBase
import torch.nn.functional as F
from torchvision.models import resnet18



class Conv2DBlock(nn.Module):
    def __init__(self, n_input_channels, k_output_channels):
        super(Conv2DBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=n_input_channels,
                              out_channels=k_output_channels,
                              kernel_size=(32, 1),
                              stride=1,
                              padding=(16, 0))  # Padding to maintain spatial dimensions
        self.batch_norm = nn.BatchNorm2d(k_output_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x
    
class ProjectBlock(nn.Module):
    def __init__(self, n_input_channels, k_output_channels):
        super(ProjectBlock, self).__init__()
        self.linear = nn.Linear(in_features=n_input_channels, out_features=k_output_channels)
        self.norm = nn.LayerNorm(k_output_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        x = self.relu(x)
        return x
    

class AirsTower(nn.Module):
    def __init__(self, num_layers: int = 5, hidden_channels: int = 16, project_coef: int = 128) -> None:
        super().__init__()
        self.tower = nn.Sequential(
            Conv2DBlock(n_input_channels=3, k_output_channels=hidden_channels),
            *[Conv2DBlock(n_input_channels=hidden_channels, k_output_channels=hidden_channels) for _ in range(num_layers-1)]
        )
        self.pooling = nn.MaxPool2d(kernel_size=(32, 1))
        self.project = ProjectBlock(n_input_channels=hidden_channels * 356, k_output_channels=hidden_channels * project_coef)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tower(x)
        x = self.pooling(x)
        x = self.project(x.view(x.size(0), -1))
        return x
    

class FgsTower(nn.Module):
    def __init__(self, output_features: int = 64) -> None:
        super().__init__()
        self.backbone = resnet18()
        self.backbone.layer4 = nn.Identity()
        self.backbone.fc = nn.Linear(256, output_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return x
    
class MetaTower(nn.Module):
    def __init__(self, output_features: int = 32) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features=7, out_features=output_features)
        self.norm = nn.LayerNorm(output_features)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class TransitModel(nn.Module):
    def __init__(
        self, 
        num_layers: int = 5, 
        hidden_channels: int = 16,
        airs_project_coef: int = 128,
        fgs_output_features: int = 32,
        meta_output_features: int = 32
        ) -> None:
        super().__init__()
        self.fgs_tower = FgsTower(output_features=fgs_output_features)
        self.airs_tower = AirsTower(num_layers=num_layers, hidden_channels=hidden_channels, project_coef=airs_project_coef)
        self.meta_tower = MetaTower(output_features=meta_output_features)
        self.linear = nn.Linear(
            in_features=hidden_channels * airs_project_coef + fgs_output_features + meta_output_features, 
            out_features=283*2
            )

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        fgs_out = self.fgs_tower(x['fgs'])
        airs_out = self.airs_tower(x['airs'])
        meta_out = self.meta_tower(x['meta'])
        out = torch.cat([fgs_out, airs_out, meta_out], dim=1)

        out = self.linear(out)
        out[:, :283] /= 100
        out[:, 283:] = F.sigmoid(out[:, 283:])
                
        return out

# class ConvModel(nn.Module):
#     def __init__(
#         self, 
#         num_layers: int = 5, 
#         hidden_channels: int = 16
#         ) -> None:
#         super().__init__()
#         self.airs_tower = nn.Sequential(
#             Conv2DBlock(n_input_channels=3, k_output_channels=hidden_channels),
#             *[Conv2DBlock(n_input_channels=hidden_channels, k_output_channels=hidden_channels) for _ in range(num_layers-1)]
#         )
#         self.pooling = nn.MaxPool2d(kernel_size=(32, 1))
#         self.project = ProjectBlock(n_input_channels=hidden_channels * 356, k_output_channels=hidden_channels * 140)
#         self.linear = nn.Linear(in_features=hidden_channels * 140, out_features=283*2)

#     def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
#         airs_out = self.airs_tower(x["airs"])
#         airs_out = self.pooling(airs_out)
#         airs_out = self.project(airs_out.view(airs_out.size(0), -1))
#         airs_out = self.linear(airs_out)

#         airs_out[:, :283] /= 100
#         airs_out[:, 283:] = F.sigmoid(airs_out[:, 283:])
                
#         return airs_out