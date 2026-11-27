import argparse
import json
import math
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from burgers_phase1_data import CONFIG, ROOT
from darcy_day15 import arrays, write_json
from darcy_fno import atomic_checkpoint


class FourierLayer(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.modes = modes
        self.weights = nn.Parameter(torch.randn(width, width, modes, dtype=torch.cfloat)/width)
        self.local = nn.Conv1d(width, width, 1)

    def forward(self, x):
        z = torch.fft.rfft(x)
        low = torch.einsum("bik,iok->bok", z[..., :self.modes], self.weights)
        y = F.pad(low, (0, z.shape[-1]-self.modes))
        return F.gelu(torch.fft.irfft(y, n=x.shape[-1])+self.local(x))


class TrajectoryFNO(nn.Module):
    def __init__(self, config):
        super().__init__()
        model = config["model"]
        width = model["width"]
        self.register_buffer("log_nu_mid", torch.tensor(np.log(config["viscosity_id"]).mean(), dtype=torch.float32))
        self.register_buffer("log_nu_scale", torch.tensor(np.ptp(np.log(config["viscosity_id"]))/2, dtype=torch.float32))
        self.lift = nn.Conv1d(2, width, 1)
        self.layers = nn.Sequential(*[FourierLayer(width, model["modes"]) for _ in range(model["depth"])])
        self.readout = nn.Sequential(nn.Conv1d(width, width, 1), nn.GELU(), nn.Conv1d(width, config["frames"], 1))

    def forward(self, u0, nu):
        log_nu = ((nu.log()-self.log_nu_mid)/self.log_nu_scale)[:, None].expand_as(u0)
        trajectory = self.readout(self.layers(self.lift(torch.stack((u0, log_nu), dim=1))))
        # The known initial condition and conserved mean are enforced at every grid.
        trajectory = trajectory-trajectory.mean(dim=-1, keepdim=True)+u0.mean(dim=-1)[:, None, None]
        return torch.cat((u0[:, None], trajectory), dim=1)


def torch_device():
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(config, device):
    if (ROOT/"fno.pt").exists():
        raise RuntimeError("Checkpoint already exists. Run lock/eval with the frozen model.")
    torch.manual_seed(config["seed"])
    model = TrajectoryFNO(config).to(device)
    settings = config["model"]
    train_data, val_data = (arrays(ROOT/f"{name}.npz") for name in ("train", "val"))
    training, validation = [tuple(torch.from_numpy(data[key]).float().to(device) for key in ("u64", "nu"))
                            for data in (train_data, val_data)]
    labels, nu = training
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=1e-4)
    steps = math.ceil(len(labels)/settings["batch"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=settings["lr"],
        epochs=settings["epochs"], steps_per_epoch=steps, pct_start=.18)
    history = []
    best = float("inf")
    best_checkpoint = None
    for epoch in range(1, settings["epochs"]+1):
        start = perf_counter()
        model.train()
        order = torch.randperm(len(labels), device=device)
        for ids in order.split(settings["batch"]):
            y = labels[ids]
            prediction = model(y[:, 0], nu[ids])
            loss = F.mse_loss(prediction[:, 1:], y[:, 1:])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
        model.eval()
        with torch.no_grad():
            y, visc = validation
            pred = torch.cat([model(yy[:, 0], vv) for yy, vv in zip(y.split(32), visc.split(32))])
            relative = ((pred-y).flatten(1).norm(dim=1)/y.flatten(1).norm(dim=1)).mean().item()
        history.append({"epoch": epoch, "val_relative_l2": relative, "seconds": perf_counter()-start})
        if relative < best:
            best = relative
            best_checkpoint = {"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                               "epoch": epoch}
        if epoch % 10 == 0 or epoch == settings["epochs"]:
            print(f"Burgers epoch {epoch}: Val relative L2={relative:.5f}, {history[-1]['seconds']:.2f}s", flush=True)
    atomic_checkpoint(ROOT / "fno.pt", best_checkpoint)
    write_json(ROOT / "train_log.json", {"history": history, "selected_epoch": best_checkpoint["epoch"],
               "parameters": sum(x.numel() for x in model.parameters()), "torch": torch.__version__,
               "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None})


def model_from_checkpoint(config, device):
    checkpoint = torch.load(ROOT/"fno.pt", map_location=device, weights_only=True)
    model = TrajectoryFNO(config).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval(), checkpoint["epoch"]


@torch.inference_mode()
def infer(model, device, initial, viscosity, batch=32):
    u0 = torch.as_tensor(initial, device=device, dtype=torch.float32)
    nu = torch.as_tensor(viscosity, device=device, dtype=torch.float32)
    return torch.cat([model(u, v).cpu() for u, v in zip(u0.split(batch), nu.split(batch))]).numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("train",))
    parser.parse_args()
    train(json.loads(CONFIG.read_text(encoding="utf-8")), torch_device())
