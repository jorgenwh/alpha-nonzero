import torch
from pytorch_model_summary import summary

from anz.constants import VOCAB_SIZE, BLOCK_SIZE, BATCH_SIZE
from anz.models import Transformer

if __name__ == '__main__':
    print(f"VOCAB_SIZE={VOCAB_SIZE}\nBLOCK_SIZE={BLOCK_SIZE}\nBATCH_SIZE={BATCH_SIZE}")
    x = torch.randint(VOCAB_SIZE, size=(BATCH_SIZE, BLOCK_SIZE), dtype=torch.int64)
    model = Transformer()

    print("Forwarding input of shape: " + str(list(x.shape)))
    pi, v = model(x)
    print("Forward pass successful")
    print(f"Policy shape: {pi.shape}")
    print(f"Value shape: {v.shape}")

    print(summary(model, x))

