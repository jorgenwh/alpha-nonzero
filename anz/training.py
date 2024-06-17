import torch
from tqdm import tqdm

from .models import Transformer, ResNet
from .helpers import AverageMeter
from .data_loader import get_data_loader
from .constants import EPOCHS, DEVICE

def cross_entropy_loss(target: torch.Tensor, out: torch.Tensor, size: int) -> torch.Tensor:
    return -torch.sum(target * out) / size

def mean_squared_error_loss(target: torch.Tensor, out: torch.Tensor, size: int) -> torch.Tensor:
    return torch.sum((target - out.reshape(-1)) ** 2) / size

def train_loop(model, optimizer, data_loader, model_type, output_dir):
    model.train()
    for epoch in range(EPOCHS):
        print(f"Epoch: {epoch+1}/{EPOCHS}")
        epoch_loss = AverageMeter()
        epoch_pi_loss = AverageMeter()
        epoch_v_loss = AverageMeter()
        bar = tqdm(data_loader, desc="training", bar_format="{l_bar}{bar}| update: {n_fmt}/{total_fmt} - {unit} - elapsed: {elapsed}")
        for i, (positions, pis, vs) in enumerate(bar):

            positions = positions.to(DEVICE)
            pis = pis.to(DEVICE)
            vs = vs.to(DEVICE)

            pi, v = model(positions)

            pi_loss = cross_entropy_loss(pis, torch.log_softmax(pi, dim=1), pis.shape[0])
            v_loss = mean_squared_error_loss(vs, v, vs.shape[0])
            loss = pi_loss + v_loss

            epoch_pi_loss.update(pi_loss.item(), positions.shape[0])
            epoch_v_loss.update(v_loss.item(), positions.shape[0])
            epoch_loss.update(loss.item(), positions.shape[0])
            bar.unit = f"pi_loss: {epoch_pi_loss} v_loss: {epoch_v_loss} loss: {epoch_loss}"

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
