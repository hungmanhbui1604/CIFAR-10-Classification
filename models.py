import torch
import torch.nn as nn
from torch import Tensor
from typing import List

class CNN(nn.Module):
    def __init__(self, out_dim: int, batch_norm: bool, dropouts: List):
        super().__init__()

        block1 = self.make_block(in_channels=3, out_channels=64, batch_norm=batch_norm, downsample=True, dropout=dropouts[0])
        block2 = self.make_block(64, 128, batch_norm, True, dropouts[1])
        block3 = self.make_block(128, 256, batch_norm, True, dropouts[2])
        self.backbone = nn.Sequential(block1, block2, block3, nn.Flatten())

        layers = [
            nn.Linear(256*4*4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropouts[3]) if dropouts[3] else None,
            nn.Linear(512, out_dim)
        ]
        fc = [layer for layer in layers if layer is not None]
        self.fc = nn.Sequential(*fc)

    def make_block(self, in_channels: int, out_channels: int, batch_norm: bool, downsample: bool, dropout: float) -> nn.Sequential:
        bias = False if batch_norm else True        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels) if batch_norm else None,
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels) if batch_norm else None,
            nn.ReLU(inplace=True)
        ]
        block = [layer for layer in layers if layer is not None]
        if downsample:
            block.append(nn.MaxPool2d(2, 2))
        if dropout:
            block.append(nn.Dropout2d(dropout))
        return nn.Sequential(*block)

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)
        out = self.fc(features)
        return out

