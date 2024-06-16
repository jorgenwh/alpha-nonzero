import torch
import torch.nn as nn
import torch.nn.functional as F
from ..constants import BOARD_CONV_CHANNELS, N_BLOCKS, POLICY_SIZE


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=BOARD_CONV_CHANNELS, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )
        self.residual_tower = nn.Sequential(*[Block() for _ in range(N_BLOCKS)])

        # policy head
        self.pi_conv_bn = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=2)
        )
        self.pi = nn.Linear(2 * 8 * 8, POLICY_SIZE)

        # value head
        self.v_conv_bn = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=1)
        )
        self.v_fc = nn.Linear(8 * 8, 256)
        self.v = nn.Linear(256, 1)

    def forward(self, x):
        N, _, _, _ = x.shape

        # residual tower forward
        r = self.conv_block(x)
        r = self.residual_tower(r)

        # policy head forward
        pi = self.pi_conv_bn(r)
        pi = F.relu(pi)
        pi = pi.view(N, 2 * 8 * 8)
        pi = self.pi(pi)

        # value head forward
        v = self.v_conv_bn(r)
        v = F.relu(v)
        v = v.view(N, 8 * 8)
        v = self.v_fc(v)
        v = F.relu(v)
        v = self.v(v)
        v = torch.tanh(v)

        return pi, v


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

