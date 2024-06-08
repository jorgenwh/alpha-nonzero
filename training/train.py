import os
import torch
from tqdm import tqdm

from model import Transformer
from utils import prepare_training_data, AverageMeter
from utils import VOCAB_SIZE, BLOCK_SIZE, OUTPUT_SIZE


D_MODEL        = 512
N_HEADS        = 8
N_BLOCKS       = 6
DEVICE         = "cuda:0"
BATCH_SIZE     = 64
LEARNING_RATE  = 0.001
TRAINING_ITERS = 2000


OUTPUT_DIR     = "models"
START_BATCH_NUMBER = 1
END_BATCH_NUMBER   = 100


if __name__ == "__main__":
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    xdata, ydata = prepare_training_data(START_BATCH_NUMBER, END_BATCH_NUMBER, max_data_points=5000)

    model = Transformer(
        vocab_size=VOCAB_SIZE,
        output_size=OUTPUT_SIZE,
        block_size=BLOCK_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_blocks=N_BLOCKS,
        device=DEVICE,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    #loss_fn = torch.nn.BCELoss()
    loss_fn = torch.nn.MSELoss()

    model.train()
    epoch_loss = AverageMeter()
    bar = tqdm(range(TRAINING_ITERS), desc="training", bar_format="{l_bar}{bar}| update: {n_fmt}/{total_fmt} - {unit} - elapsed: {elapsed}")
    for it in bar:
        ix = torch.randint(0, xdata.size(0), (BATCH_SIZE,)) # (b)
        x = torch.stack([xdata[i] for i in ix])
        y = torch.stack([ydata[i] for i in ix])
        assert x.shape == (BATCH_SIZE, BLOCK_SIZE)
        assert y.shape == (BATCH_SIZE, 1)

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        output = model(x)
        assert output.shape == (BATCH_SIZE, 1), f"Output shape {output.shape} is not correct"

        loss = loss_fn(output, y)
        epoch_loss.update(loss.item(), BATCH_SIZE)
        bar.unit = f"loss: {epoch_loss}"

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % 1000 == 0:
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/model_checkpoint_{it}.pt")

