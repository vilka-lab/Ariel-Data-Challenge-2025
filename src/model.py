import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch3dunet.unet3d.model import UNet3D
# from segmentation_models_pytorch import Unet
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
    

class HorizontalZip(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.weights = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding="same"),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights(x)
    

class VerticalZip(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.weights = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding="same"),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights(x)
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.weights = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=(3, 3), padding="same"),
            nn.SiLU(),
            nn.BatchNorm2d(in_channels // 2),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=(3, 3), padding="same"),
            nn.SiLU(),
            nn.BatchNorm2d(in_channels),
        )
        self.resid = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), padding="same")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights(x) + self.resid(x)
    
    

class TransitTower(nn.Module):
    def __init__(self):
        super().__init__()
        # self.weights = nn.Sequential(
        #     nn.Conv2d(1, 4, kernel_size=(5, 5), padding="same"),

        #     HorizontalZip(4, 8),
        #     # ConvBlock(8),

        #     VerticalZip(8, 16),
        #     # ConvBlock(16),

        #     HorizontalZip(16, 32),
        #     # ConvBlock(32),

        #     VerticalZip(32, 64),
        #     # ConvBlock(64),

        #     HorizontalZip(64, 128),
        #     # ConvBlock(128),

        #     # ConvBlock(128),
        #     # ConvBlock(128),

        #     VerticalZip(128, 128),
        #     HorizontalZip(128, 128),
            
        #     # ConvBlock(128),
        #     # ConvBlock(128),

        #     VerticalZip(128, 128),
        #     HorizontalZip(128, 128),
            
        #     # ConvBlock(128),
        #     # ConvBlock(128),

        #     nn.Flatten(),
        #     nn.SiLU(),
        #     nn.Linear(11264, 1024),
        # )
        self.weights = timm.create_model('vit_base_patch32_224.sam_in1k', pretrained=True, in_chans=1, img_size=(375, 318))
        self.weights.head = nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights(x)
    

class UncertaintyModel(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_features: int = 768):
        super().__init__()
        self.weights = nn.Sequential(
            nn.BatchNorm1d(7 + num_features),
            nn.Dropout(0.2),
            nn.Linear(7 + num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 283)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.weights(x) - 8)
    

class TransitModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.mean_tower = MeanTower()
        self.transit_tower = TransitTower()
        num_features = 768

        self.unc_model = UncertaintyModel(num_features=num_features)
        self.linear = nn.Linear(num_features + 7, 283)
        # self.gate = nn.Sequential(
        #     nn.Linear(7 + num_features + 1, 1),
        #     nn.Sigmoid()
        # )


    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        features = self.transit_tower(x["transit_map"])
        
        transit = self.linear(torch.cat([features, x["meta"]], dim=1))
        
        unc_input = torch.cat([features, x["meta"]], dim=1)
        unc_out = self.unc_model(unc_input)

        # gate = self.gate(torch.cat([features, x["meta"], x["static_component"]], dim=1)) * 2
        # transit = transit + gate * x["static_component"]
        
        out = torch.cat([transit, unc_out], dim=1)
        return out
    
    def freeze_backbone(self):
        models = [self.transit_tower, self.linear]
        
        for model in models:
            for param in model.parameters():
                param.requires_grad = False

    def freeze_uncertainty(self):
        for param in self.unc_model.parameters():
            param.requires_grad = False
