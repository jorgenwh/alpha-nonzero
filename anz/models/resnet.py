import torch
import torch.nn as nn
import torch.nn.functional as F
from ..constants import BOARD_CONV_CHANNELS, N_BLOCKS


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=BOARD_CONV_CHANNELS, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )
        self.residual_tower = nn.Sequential(*[Block() for _ in range(N_BLOCKS)])

        # value head
        self.v_conv_bn = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=BOARD_CONV_CHANNELS, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=BOARD_CONV_CHANNELS)
        )
        self.v_fc = nn.Linear(8*8*BOARD_CONV_CHANNELS, 1024)
        self.v = nn.Linear(1024, 1)

    def forward(self, x):
        N, _, _, _ = x.shape

        # residual tower forward
        r = self.conv_block(x)
        r = self.residual_tower(r)

        # value head forward
        v = self.v_conv_bn(r)
        v = F.relu(v)
        v = v.view(N, 8*8*BOARD_CONV_CHANNELS)
        v = self.v_fc(v)
        v = F.relu(v)
        v = self.v(v)
        v = torch.tanh(v)

        return v


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=256)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=256)

    def forward(self, x):
        residual = x
        r = self.conv1(x)
        r = self.bn1(r)
        r = F.relu(r)
        r = self.conv2(r)
        r = self.bn2(r)
        r += residual
        r = F.relu(r)
        return r

