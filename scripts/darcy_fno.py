import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SpectralBlock(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.modes = modes
        shape = (width, width, modes, modes, 2)
        self.positive = nn.Parameter(torch.randn(shape) / width)
        self.negative = nn.Parameter(torch.randn(shape) / width)
        self.local = nn.Conv2d(width, width, 1)

    def forward(self, x):
        m = self.modes
        nx, ny = x.shape[-2:]
        coeff = torch.fft.rfft2(x, norm="ortho")
        transformed = coeff.new_zeros(coeff.shape)
        transformed[:, :, :m, :m] = torch.einsum(
            "bixy,ioxy->boxy", coeff[:, :, :m, :m], torch.view_as_complex(self.positive))
        transformed[:, :, -m:, :m] = torch.einsum(
            "bixy,ioxy->boxy", coeff[:, :, -m:, :m], torch.view_as_complex(self.negative))
        # C2R requires the ky=0 column to be Hermitian along x. The
        # independently learned positive and negative x multipliers need this
        # projection. The ky=Nyquist column is zero by mode truncation.
        zero_column = transformed[..., :1]
        mirrored = torch.cat((zero_column[:, :, :1], zero_column[:, :, 1:].flip((2,))), dim=2)
        transformed = torch.cat(((zero_column + mirrored.conj()) * 0.5, transformed[..., 1:]), dim=3)
        return torch.fft.irfft2(transformed, s=(nx, ny), norm="ortho") + self.local(x)


class FNO(nn.Module):
    def __init__(self, spec):
        super().__init__()
        width = spec["width"]
        self.lift = nn.Conv2d(3, width, 1)
        self.blocks = nn.ModuleList(SpectralBlock(width, spec["modes"]) for _ in range(spec["layers"]))
        self.project = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, 1, 1))
        self.register_buffer("input_std", torch.ones(()))
        self.register_buffer("output_scale", torch.ones(()))

    def forward(self, a):
        batch, n, m = a.shape
        coord_x = (torch.arange(n, device=a.device, dtype=a.dtype) + 0.5) / n
        coord_y = (torch.arange(m, device=a.device, dtype=a.dtype) + 0.5) / m
        x, y = torch.meshgrid(coord_x, coord_y, indexing="ij")
        xy = torch.stack((x, y)).expand(batch, 2, n, m)
        z = self.lift(torch.cat(((a.log() / self.input_std)[:, None], xy), dim=1))
        for block in self.blocks:
            z = F.gelu(block(z))
        mask = 16 * x * (1-x) * y * (1-y)
        return self.project(z)[:, 0] * mask * self.output_scale


def relative_l2(prediction, truth):
    return ((prediction - truth).flatten(1).norm(dim=1)
            / truth.flatten(1).norm(dim=1)).mean()


def load_arrays(path, device):
    with np.load(path) as dataset:
        return tuple(torch.from_numpy(dataset[k]).to(device) for k in ("a", "u"))


def atomic_checkpoint(path, checkpoint):
    temp = path.with_suffix(".tmp")
    torch.save(checkpoint, temp)
    temp.replace(path)


@torch.inference_mode()
def predict(model, a, batch_size):
    return torch.cat([model(part).cpu() for part in a.split(batch_size)]).numpy()


def train(config, seed, root, device):
    spec = config["model"]
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    model = FNO(spec).to(device)
    train_a, train_u = load_arrays(root / "data/train.npz", device)
    val_a, val_u = load_arrays(root / "data/val.npz", device)
    model.input_std.copy_(train_a.log().std())
    model.output_scale.copy_(train_u.square().mean().sqrt())
    opt = torch.optim.AdamW(model.parameters(), lr=spec["learning_rate"], weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, spec["epochs"], eta_min=5e-5)
    order_rng = torch.Generator(device="cpu").manual_seed(seed)
    best, start = float("inf"), 0
    history = []
    latest = root / "latest.pt"
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["schedule"])
        order_rng.set_state(state["rng"])
        best, start, history = state["best"], state["epoch"], state["history"]
    print(f"Training seed {seed} on {device}: {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}", flush=True)
    for epoch in range(start, spec["epochs"]):
        tick = time.perf_counter()
        model.train()
        losses = []
        for inds in torch.randperm(len(train_a), generator=order_rng).split(spec["batch_size"]):
            opt.zero_grad(set_to_none=True)
            loss = relative_l2(model(train_a[inds]), train_u[inds])
            loss.backward()
            opt.step()
            losses.append(loss.detach())
        schedule.step()
        model.eval()
        with torch.inference_mode():
            validation = float(torch.stack([
                relative_l2(model(part), target) * len(part)
                for part, target in zip(val_a.split(spec["batch_size"]), val_u.split(spec["batch_size"]))
            ]).sum() / len(val_a))
        train_loss = float(torch.stack(losses).mean())
        elapsed = time.perf_counter() - tick
        history.append({"epoch": epoch+1, "train_relative_l2": train_loss,
                        "val_relative_l2": validation, "seconds": elapsed})
        if validation < best:
            best = validation
            atomic_checkpoint(root / "fno.pt", {"model": model.state_dict(), "epoch": epoch+1,
                                               "val_relative_l2": best})
        atomic_checkpoint(latest, {"model": model.state_dict(), "optimizer": opt.state_dict(),
                                   "schedule": schedule.state_dict(), "rng": order_rng.get_state(),
                                   "epoch": epoch+1, "best": best, "history": history})
        print(f"epoch {epoch+1}/{spec['epochs']} train {train_loss:.4f} val {validation:.4f} {elapsed:.2f}s", flush=True)
    information = {"seed": seed, "device": str(device), "torch": str(torch.__version__),
                   "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   "parameters": sum(p.numel() for p in model.parameters()), "history": history}
    (root / "training.json").write_text(json.dumps(information, indent=2), encoding="utf-8")


def infer(config, root, device, include_ood):
    model = FNO(config["model"]).to(device)
    checkpoint = torch.load(root / "fno.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    splits = ["val", "cal", "id_test", "paired_test"]
    if include_ood:
        splits += ["ood_a", "ood_b"]
    for split in splits:
        with np.load(root / f"data/{split}.npz") as dataset:
            values = {"pred": predict(model, torch.from_numpy(dataset["a"]).to(device), config["model"]["batch_size"])}
            if "a_f" in dataset:
                values["pred_f"] = predict(model, torch.from_numpy(dataset["a_f"]).to(device), 4)
        np.savez(root / f"{split}_pred.npz", **values)
        print(f"inferred {split} with validation checkpoint epoch {checkpoint['epoch']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["train", "infer"])
    parser.add_argument("--config", default="configs/phase0_darcy.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ood", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    root = Path("results") / f"seed_{args.seed}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.stage == "train":
        train(config, args.seed, root, device)
    else:
        infer(config, root, device, args.ood)
