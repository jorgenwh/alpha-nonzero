from typing import Union
import torch
from tqdm import tqdm

from .models import Transformer, ResNet
from .helpers import AverageMeter
from .data_loader import get_data_loader
from .constants import EPOCHS, DEVICE

def train_loop(model, optimizer, data_loader, model_type, output_dir):
    pi_loss_fn = torch.nn.CrossEntropyLoss()
    v_loss_fn = torch.nn.MSELoss()
    model.train()
    print("FP16 disabled")

    for epoch in range(EPOCHS):
        print(f"Epoch: {epoch+1}/{EPOCHS}")

        epoch_loss = AverageMeter()
        epoch_pi_loss = AverageMeter()
        epoch_v_loss = AverageMeter()

        bar = tqdm(
            data_loader, 
            desc="training", 
            bar_format="{l_bar}{bar}| update: {n_fmt}/{total_fmt} - {unit} - elapsed: {elapsed}"
        )
        for i, (positions, pis, vs) in enumerate(bar):
            optimizer.zero_grad()

            positions = positions.squeeze().to(DEVICE)
            pis = pis.squeeze().to(DEVICE)
            vs = vs.reshape(-1, 1).to(DEVICE)

            pi, v = model(positions)

            pi_loss = pi_loss_fn(pi, pis)
            v_loss = v_loss_fn(v, vs)
            loss = pi_loss + v_loss

            epoch_pi_loss.update(pi_loss.item(), positions.shape[0])
            epoch_v_loss.update(v_loss.item(), positions.shape[0])
            epoch_loss.update(loss.item(), positions.shape[0])
            bar.unit = f"pi_loss: {epoch_pi_loss} v_loss: {epoch_v_loss} loss: {epoch_loss}"

            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                torch.save(model.state_dict(), f"{output_dir}/{model_type}_checkpoint_epoch{epoch}_iter{i}.pth")

def train_loop_fp16(model, optimizer, data_loader, model_type, output_dir):
    pi_loss_fn = torch.nn.CrossEntropyLoss()
    v_loss_fn = torch.nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()
    model.train()
    print("FP16 enabled")

    for epoch in range(EPOCHS):
        print(f"Epoch: {epoch+1}/{EPOCHS}")

        epoch_loss = AverageMeter()
        epoch_pi_loss = AverageMeter()
        epoch_v_loss = AverageMeter()

        bar = tqdm(
            data_loader, 
            desc="training", 
            bar_format="{l_bar}{bar}| update: {n_fmt}/{total_fmt} - {unit} - elapsed: {elapsed}"
        )
        for i, (positions, pis, vs) in enumerate(bar):
            optimizer.zero_grad()

            positions = positions.squeeze().to(DEVICE)
            pis = pis.squeeze().to(DEVICE)
            vs = vs.reshape(-1, 1).to(DEVICE)

            with torch.cuda.amp.autocast():
                pi, v = model(positions)

                pi_loss = pi_loss_fn(pi, pis)
                v_loss = v_loss_fn(v, vs)
                loss = pi_loss + v_loss

            epoch_pi_loss.update(pi_loss.item(), positions.shape[0])
            epoch_v_loss.update(v_loss.item(), positions.shape[0])
            epoch_loss.update(loss.item(), positions.shape[0])
            bar.unit = f"pi_loss: {epoch_pi_loss} v_loss: {epoch_v_loss} loss: {epoch_loss}"

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if i % 1000 == 0:
                torch.save(model.state_dict(), f"{output_dir}/{model_type}_checkpoint_epoch{epoch}_iter{i}.pth")

def train(data_fn: str, model_type: str, output_dir: str, max_datapoints: Union[int, None], use_fp16: bool):
    if model_type == "transformer":
        model = Transformer().to(DEVICE)
    elif model_type == "resnet":
        model = ResNet().to(DEVICE)
    else:
        assert False, f"Invalid model_type: '{model_type}'"

    optimizer = torch.optim.Adam(model.parameters())
    data_loader = get_data_loader(data_fn, model_type, max_datapoints, use_fp16)

    if use_fp16:
        train_loop_fp16(model, optimizer, data_loader, model_type, output_dir)
    else:
        train_loop(model, optimizer, data_loader, model_type, output_dir)
