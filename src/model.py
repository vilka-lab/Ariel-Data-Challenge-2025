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

class ConvModel(nn.Module):
    def __init__(
        self, 
        num_layers: int = 5, 
        hidden_channels: int = 16
        ) -> None:
        super().__init__()
        self.airs_tower = nn.Sequential(
            Conv2DBlock(n_input_channels=3, k_output_channels=hidden_channels),
            *[Conv2DBlock(n_input_channels=hidden_channels, k_output_channels=hidden_channels) for _ in range(num_layers-1)]
        )
        self.pooling = nn.MaxPool2d(kernel_size=(32, 1))
        self.project = nn.Linear(in_features=hidden_channels * 356, out_features=hidden_channels * 283)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_features=hidden_channels * 283, out_features=283*2)

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        airs_out = self.airs_tower(x["airs"])
        airs_out = self.pooling(airs_out)
        airs_out = self.relu(self.project(airs_out.view(airs_out.size(0), -1)))
        airs_out = self.linear(airs_out)

        airs_out[:, :283] /= 100
        airs_out[:, 283:] = F.sigmoid(airs_out[:, 283:])
                
        return airs_out