import os
import torch
from tqdm import tqdm

from model import Transformer
from helpers import fetch_training_data, AverageMeter
from constants import (
    OUTPUT_DIR,
    MAX_DATA_POINTS,
    LEARNING_RATE,
    BATCH_SIZE,
    TRAINING_ITERS,
    VOCAB_SIZE,
    BLOCK_SIZE,
    NUM_BINS,
    D_MODEL,
    N_HEADS,
    N_LAYERS,
    DEVICE,
)


if __name__ == "__main__":
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    xdata, ydata = fetch_training_data(fn="data/training_data.pkl", num_bins=NUM_BINS, max_data_points=MAX_DATA_POINTS)

    model = Transformer(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        output_size=NUM_BINS,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    epoch_loss = AverageMeter()
    bar = tqdm(range(TRAINING_ITERS), desc="training", bar_format="{l_bar}{bar}| update: {n_fmt}/{total_fmt} - {unit} - elapsed: {elapsed}")
    for it in bar:
        ix = torch.randint(0, xdata.size(0), (BATCH_SIZE,)) # (b)
        x = torch.stack([xdata[i] for i in ix])
        y = torch.stack([ydata[i] for i in ix])
        assert x.shape == (BATCH_SIZE, BLOCK_SIZE)
        assert y.shape == (BATCH_SIZE, NUM_BINS)

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        output = model(x)
        assert output.shape == (BATCH_SIZE, NUM_BINS), f"Output shape {output.shape} is not correct"

        loss = loss_fn(output, y)
        epoch_loss.update(loss.item(), BATCH_SIZE)
        bar.unit = f"loss: {epoch_loss}"

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % 1000 == 0:
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/model_checkpoint_{it}.pt")

