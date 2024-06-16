import time
import torch
from pytorch_model_summary import summary

from anz.constants import VOCAB_SIZE, BLOCK_SIZE, BATCH_SIZE
from anz.models import Transformer

if __name__ == '__main__':
    print(f"VOCAB_SIZE={VOCAB_SIZE}\nBLOCK_SIZE={BLOCK_SIZE}\nBATCH_SIZE={BATCH_SIZE}")
    x = torch.randint(VOCAB_SIZE, size=(BATCH_SIZE, BLOCK_SIZE), dtype=torch.int64)
    model = Transformer()

    print("Forwarding input of shape: " + str(list(x.shape)))
    t0 = time.time()
    pi, v = model(x)
    t1 = time.time()
    print("Forward pass successful - took {:.2f} seconds".format(t1 - t0))
    print(f"Policy shape: {pi.shape}")
    print(f"Value shape: {v.shape}")

    print(summary(model, x))

