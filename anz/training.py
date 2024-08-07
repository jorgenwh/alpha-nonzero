from typing import Union
import torch
from tqdm import tqdm

from .models import Transformer, ResNet
from .helpers import AverageMeter
from .data_loader import get_data_loader
from .constants import EPOCHS, DEVICE

def train_loop(
        model, 
        optimizer, 
        data_loader, 
        model_type, 
        output_dir, 
        use_fp16: bool = False, 
        compile_model: bool = False
) -> None:
    pi_loss_fn = torch.nn.CrossEntropyLoss()
    v_loss_fn = torch.nn.MSELoss()
    model.train()

    print("fp16:", use_fp16)
    print("compile_model:", compile_model)
    print("device:", DEVICE)

    if compile_model:
        model = torch.compile(model)

    i = 0
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
        for positions, pis, vs in bar:
            optimizer.zero_grad()

            positions = positions.squeeze().to(DEVICE)
            pis = pis.squeeze().to(DEVICE)
            vs = vs.view(-1, 1).to(DEVICE)

            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16 if use_fp16 else torch.float32):
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

            i += 1
            if i % 5000 == 0:
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "epoch_loss": epoch_loss.avg,
                    "updates": i
                }
                torch.save(checkpoint, f"{output_dir}/{model_type}_checkpoint_iter{i}.pth")

def train(
        data_fn: str, 
        model_type: str, 
        output_dir: str, 
        max_datapoints: Union[int, None], 
        use_fp16: bool, 
        compile_model: bool = False
):
    if model_type == "transformer":
        model = Transformer().to(DEVICE)
    elif model_type == "resnet":
        model = ResNet().to(DEVICE)
    else:
        assert False, f"Invalid model_type: '{model_type}'"

    optimizer = torch.optim.Adam(model.parameters())
    data_loader = get_data_loader(data_fn, model_type, max_datapoints, use_fp16)

    train_loop(model, optimizer, data_loader, model_type, output_dir, use_fp16, compile_model)
