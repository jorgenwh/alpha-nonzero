import time
import torch
from pytorch_model_summary import summary

from anz.constants import BOARD_CONV_CHANNELS, BATCH_SIZE, DEVICE
from anz.models import ResNet
from anz.helpers import get_torch_model_size

if __name__ == '__main__':
    print(f"BOARD_CONV_CHANNELS={BOARD_CONV_CHANNELS}\nBATCH_SIZE={BATCH_SIZE}")
    x = torch.randn(size=(BATCH_SIZE, BOARD_CONV_CHANNELS, 8, 8), dtype=torch.float32).to(DEVICE)
    model = ResNet().to(DEVICE)

    print("Forwarding input of shape: " + str(list(x.shape)))
    t0 = time.time()
    pi, v = model(x)
    t1 = time.time()
    print("Forward pass successful - took {:.2f} seconds".format(t1 - t0))
    print(f"Policy shape: {pi.shape}")
    print(f"Value shape: {v.shape}")

    print(summary(model, x))

    model_size = get_torch_model_size(model)
    print(f"Model size: {round(model_size / 1e9, 3)} GB")

