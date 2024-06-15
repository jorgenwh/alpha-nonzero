import torch
from pytorch_model_summary import summary

from anz.constants import BOARD_CONV_CHANNELS, BATCH_SIZE
from anz.models import ResNet

if __name__ == '__main__':
    print(f"BOARD_CONV_CHANNELS={BOARD_CONV_CHANNELS}\nBATCH_SIZE={BATCH_SIZE}")
    x = torch.randn(size=(BATCH_SIZE, BOARD_CONV_CHANNELS, 8, 8))
    model = ResNet()

    print("Forwarding input of shape: " + str(list(x.shape)))
    p, v = model(x)
    print("Forward pass successful")
    print(f"Policy shape: {p.shape}")
    print(f"Value shape: {v.shape}")

    print(summary(model, x))
