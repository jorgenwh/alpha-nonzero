import torch
from tqdm import tqdm

from .models import Transformer, ResNet
from .helpers import AverageMeter
from .data_loader import get_data_loader
from .constants import EPOCHS, DEVICE

def mean_squared_error_loss(target: torch.Tensor, out: torch.Tensor, size: int) -> torch.Tensor:
    return torch.sum((target - out.reshape(-1)) ** 2) / size

def train_loop(model, optimizer, data_loader, model_type, output_dir):
    model.train()
    loss_fn = torch.nn.MSELoss()
    for epoch in range(EPOCHS):
        print(f"Epoch: {epoch+1}/{EPOCHS}")
        epoch_loss = AverageMeter()
        bar = tqdm(data_loader, desc="training", bar_format="{l_bar}{bar}| update: {n_fmt}/{total_fmt} - {unit} - elapsed: {elapsed}")
        for i, (positions, vs) in enumerate(bar):

            positions = positions.to(DEVICE)
            vs = vs.to(DEVICE)

            v = model(positions)

            #loss = mean_squared_error_loss(vs, v, vs.shape[0])
            loss = loss_fn(v, vs)

            epoch_loss.update(loss.item(), positions.shape[0])
            bar.unit = f"loss: {epoch_loss}"

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                torch.save(model.state_dict(), f"{output_dir}/{model_type}_checkpoint_epoch{epoch}_iter{i}.pth")

def train(data_fn,model_type, output_dir, max_datapoints):
    if model_type == "transformer":
        model = Transformer().to(DEVICE)
    elif model_type == "resnet":
        model = ResNet().to(DEVICE)
    else:
        assert False, f"Invalid model_type: '{model_type}'"

    optimizer = torch.optim.Adam(model.parameters())
    data_loader = get_data_loader(data_fn, model_type, max_datapoints)

    train_loop(model, optimizer, data_loader, model_type, output_dir)
