import pandas as pd
from src.data import DataProcessor, TransitTrainDataset, TransitTestDataset
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchcnnbuilder.models import ForecasterBase



class ConvModel(nn.Module):
    def __init__(
        self, 
        num_layers: int = 5, 
        hidden_channels: int = 16
        ) -> None:
        super().__init__()
        self.airs_tower = nn.Linear(32*356, 1024)
        self.fgs_tower = nn.Linear(32*32, 128)
        self.linear = nn.Linear(1024 + 128, 283*2)

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        airs_out = self.airs_tower(x["airs"].flatten(start_dim=1))
        fgs_out = self.fgs_tower(x["fgs"].flatten(start_dim=1))

        out = torch.cat([airs_out, fgs_out], dim=1)
        out = self.linear(out)
        # out[:, :283] *= 147
        
        return out