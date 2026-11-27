import argparse
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.ndimage import map_coordinates
from torch import nn
from torch.nn import functional as F

from allen_cahn_horizon_audit import samples
from allen_cahn_phase2_probe import OP, StepFNO, chunks, reconstruct, rollout, scale, scales, torch_device
from darcy_day15 import arrays, band_summary, comparison, digest, evaluate_bands, value_at_coverage, write_json
from darcy_day15_ops import timed, timing_summary
from darcy_fno import atomic_checkpoint
from darcy_phase0_eval import boot_interval, quantile


CONFIG = Path("configs/phase2_ac_formal.json")
MODELS = ("FNO", "U-Net")
TARGETS = (("id", 64), ("paired", 64), ("paired", 128), ("ood", 64))
DIFFUSION = "Diffusion-only OpRes-CP"


class StepUNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        def block(source, target):
            return nn.Sequential(
                nn.Conv2d(source, target, 3, padding=1, padding_mode="circular"), nn.GELU(),
                nn.Conv2d(target, target, 3, padding=1, padding_mode="circular"), nn.GELU(),
            )

        self.first = block(3, 32)
        self.second = block(32, 64)
        self.middle = block(64, 128)
        self.decode2 = block(192, 64)
        self.decode1 = block(96, 32)
        self.out = nn.Conv2d(32, 1, 1)
        self.dt = config["dt"]

    def forward(self, u, nu, rho):
        nu = ((nu-.04)/.02)[:, None, None].expand_as(u)
        rho = ((rho-.8)/.3)[:, None, None].expand_as(u)
        x = self.first(torch.stack((u, nu, rho), dim=1))
        y = self.second(F.avg_pool2d(x, 2))
        z = self.middle(F.avg_pool2d(y, 2))
        y = self.decode2(torch.cat((F.interpolate(z, scale_factor=2, mode="nearest"), y), dim=1))
        x = self.decode1(torch.cat((F.interpolate(y, scale_factor=2, mode="nearest"), x), dim=1))
        return u + self.dt*self.out(x)[:, 0]


def cache_data(config, split, root):
    path = root/f"{split}.npz"
    frames = config["frames"]//2 if split in ("train", "val", "cal") else config["frames"]
    if not path.is_file():
        temp = path.with_suffix(".tmp.npz")
        np.savez(temp, **samples(config | {"frames": frames}, split))
        temp.replace(path)
    expected_ids = config["streams"][split]*10000 + np.arange(config["counts"][split])
    with np.load(path) as data:
        if not np.array_equal(data["id"], expected_ids) or data["u64"].shape[1] != frames + 1:
            raise RuntimeError(f"Cached {split} trajectories differ from the formal configuration.")
    return path


def train(config, kind, train_data, val, device, folder, signature, *, initialization=None, factory=None):
    init = config["seed"] if initialization is None else initialization
    torch.manual_seed(init)
    constructor = factory or (StepFNO if kind == "FNO" else StepUNet)
    model = constructor(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epochs"], eta_min=.00015)
    u, nu, rho = (torch.as_tensor(train_data[key], device=device, dtype=torch.float32)
                  for key in ("u64", "nu", "rho"))
    last, best = folder/"last.pt", folder/"best.pt"
    history, minimum = [], float("inf")
    if last.is_file():
        saved = torch.load(last, map_location=device, weights_only=True)
        if saved["signature"] != signature:
            raise RuntimeError("Checkpoint does not match the locked training data and settings.")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        history, minimum = saved["history"], saved["minimum"]
    for epoch in range(len(history), config["epochs"]):
        started = perf_counter()
        if initialization is not None:
            torch.manual_seed(init+epoch*100000)
        rng = np.random.default_rng(np.random.SeedSequence([config["seed"], epoch]))
        model.train()
        for _ in range(config["train_steps_per_epoch"]):
            ix = rng.integers(len(u), size=config["batch"])
            t = rng.integers(u.shape[1]-1, size=config["batch"])
            ix, t = (torch.as_tensor(x, device=device) for x in (ix, t))
            loss = F.mse_loss(model(u[ix, t], nu[ix], rho[ix]), u[ix, t+1])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        predicted = rollout(model, val, 64, config["frames"]//2, device, config["batch"])
        error = float(np.linalg.norm(predicted[:, 1:]-val["u64"][:, 1:]) /
                      np.linalg.norm(val["u64"][:, 1:]))
        if not np.isfinite(error):
            raise RuntimeError(f"{kind} produced a non-finite validation rollout.")
        if error < minimum:
            minimum = error
            atomic_checkpoint(best, {"model": model.state_dict(), "epoch": epoch+1})
        history.append({"val_relative_l2": error, "elapsed_s": perf_counter()-started})
        atomic_checkpoint(last, {"signature": signature, "model": model.state_dict(), "history": history,
                                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                                 "minimum": minimum})
        print(f"Allen-Cahn {kind} epoch {epoch+1}: Val {error:.6f}", flush=True)
    chosen = torch.load(best, map_location=device, weights_only=True)
    model.load_state_dict(chosen["model"])
    return model.eval(), {"history": history, "epoch": chosen["epoch"], "weight_sha256": digest(best)}


def fit_scales(data, config, model, device):
    error_sum = np.zeros(data["u64"].shape[-2:])
    residual_medians = []
    for ix, prediction, _, residual in chunks(data, 64, config, model, device, config["inverse"]):
        error_sum += abs(prediction[:, 1:]-data["u64"][ix, 1:]).sum(axis=(0, 1))
        residual_medians.extend(np.median(abs(residual), axis=(1, 2, 3)))
    fixed = error_sum/(len(data["id"])*(data["u64"].shape[1]-1))
    return np.maximum(fixed, .05*fixed.mean()), 1/np.median(residual_medians)


def calibrate(data, config, model, device, fitted):
    scores = defaultdict(list)
    for ix, prediction, lifted, residual in chunks(data, 64, config, model, device, config["inverse"]):
        error = abs(prediction[:, 1:]-data["u64"][ix, 1:])
        diffused, _ = reconstruct(prediction,data["nu"][ix],data["rho"][ix],config,"diffusion")
        maps = scales(lifted,residual,fitted[0],config,fitted[1]) | {DIFFUSION:scale(diffused,config)}
        for method, sigma in maps.items():
            scores[method].extend((error/sigma).max(axis=(1, 2, 3)))
    return {str(alpha): {method: quantile(np.asarray(value), alpha) for method, value in scores.items()}
            for alpha in config["alpha_levels"]}


def evaluate(data, n, config, model, device, fitted, thresholds):
    fixed, pi = fitted
    if n != fixed.shape[0]:
        grid = np.arange(n)*fixed.shape[0]/n
        fixed = map_coordinates(fixed, np.meshgrid(grid, grid, indexing="ij"), order=1, mode="grid-wrap")
    rows, relative_errors = defaultdict(lambda: defaultdict(list)), defaultdict(list)
    stability = {"prediction_max": 0., "lift_max": 0.}
    for ix, prediction, lifted, residual in chunks(data, n, config, model, device, config["inverse"]):
        truth = data[f"u{n}"][ix, 1:]
        diffused, _ = reconstruct(prediction,data["nu"][ix],data["rho"][ix],config,"diffusion")
        stability["prediction_max"] = max(stability["prediction_max"], float(abs(prediction).max()))
        stability["lift_max"] = max(stability["lift_max"], float(abs(lifted).max()))
        for frames in (config["frames"]//2, config["frames"]):
            target = truth[:, :frames]
            error = (abs(prediction[:, 1:frames+1]-target)).reshape(len(truth), -1)
            relative_errors[frames].extend(np.linalg.norm(error, axis=1) /
                                           np.linalg.norm(target.reshape(len(truth), -1), axis=1))
            dx = np.roll(target, 1, axis=-1)-np.roll(target, -1, axis=-1)
            dy = np.roll(target, 1, axis=-2)-np.roll(target, -1, axis=-2)
            gradients = (dx*dx+dy*dy).reshape(error.shape)
            k = int(np.ceil(.10*gradients.shape[1]))
            top = np.argpartition(gradients, -k, axis=1)[:, -k:]
            maps = scales(lifted[:, :frames],residual[:, :frames],fixed,config,pi)
            maps[DIFFUSION] = scale(diffused[:, :frames],config)
            for method, sigma in maps.items():
                sigma = sigma.reshape(error.shape)
                ratios = error/sigma
                for alpha, qs in thresholds.items():
                    row = evaluate_bands(ratios, sigma, qs[method], top)
                    for col, value in row.items():
                        rows[alpha, frames, method][col].append(value)
    raw, summaries = defaultdict(dict), defaultdict(dict)
    for (alpha,frames,method), columns in rows.items():
        values = {col: np.concatenate(chunks) for col,chunks in columns.items()}
        raw[alpha,frames][method] = values
        q = value_at_coverage(values["score"], 1-float(alpha))
        summaries[alpha,frames][method] = band_summary(values) | {
            "matched_nominal_width": float(2*q*values["scale_mean"].mean()),
        }
    output = {}
    for (alpha,frames), methods in summaries.items():
        values = raw[alpha,frames]
        comparisons = {method: comparison(values[OP], row, config, config | {"alpha": float(alpha)})
                       for method, row in values.items() if method != OP}
        for result in comparisons.values():
            if "matched90_width_ratio_ci95" in result:
                result["matched_nominal_width_ratio_ci95"] = result.pop("matched90_width_ratio_ci95")
        output.setdefault(alpha,{})[str(frames)] = {
            "methods": methods, "point_mean_relative_l2": float(np.mean(relative_errors[frames])),
            "comparisons": comparisons, "stability": stability,
        }
    return output, raw


def paired_change(base, fine, config, base_frames=None):
    output = {}
    for (alpha,frames), comparison_rows in fine.items():
        methods = base[alpha, frames if base_frames is None else base_frames]
        changes = {}
        for method, row in methods.items():
            other = comparison_rows[method]
            delta = other["coverage"].astype(float)-row["coverage"]
            width_ratio = None
            if not (row["infinite"].any() or other["infinite"].any()):
                width_ratio = float(other["width"].mean()/row["width"].mean())
            changes[method] = {"coverage_delta": float(delta.mean()),
                               "coverage_delta_ci95": boot_interval(delta,config["bootstrap_resamples"],
                                                                    config["bootstrap_seed"]),
                               "width_ratio": width_ratio}
        output.setdefault(alpha,{})[str(frames)] = changes
    return output


def timing(data, n, config, model, device):
    elapsed = []
    for i in range(24):
        sample = {key:value[i:i+1] for key,value in data.items()}
        prediction, forward = timed(lambda: rollout(model,sample,n,config["frames"],device,1))
        _, post = timed(lambda: scale(reconstruct(prediction,sample["nu"],sample["rho"],
                                                   config,config["inverse"])[0],config))
        if i>=4:
            elapsed.append((forward,post,forward+post))
    return {key:timing_summary(values) for key,values in zip(("forward","post","total"),zip(*elapsed))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    plan = json.loads(CONFIG.read_text(encoding="utf-8"))
    if args.seed not in plan["seeds"]:
        parser.error("Seed not registered.")
    config = json.loads(Path(plan["base"]).read_text(encoding="utf-8")) | plan | {"seed":args.seed}
    device = torch_device()
    if args.smoke:
        config |= {"counts":dict.fromkeys(config["counts"],12),"epochs":1,"train_steps_per_epoch":2,
                   "batch":4,"bootstrap_resamples":50}
        config["streams"] = {split:stream+100 for split,stream in config["streams"].items()}
    elif device.type!="cuda":
        raise RuntimeError("The formal Allen-Cahn benchmark requires CUDA.")
    root = Path("results")/("phase2_ac_smoke" if args.smoke else "phase2_allen_cahn")/f"seed_{args.seed}"
    root.mkdir(parents=True, exist_ok=True)
    paths = {split:cache_data(config,split,root) for split in ("train","val")}
    signatures = {split:digest(path) for split,path in paths.items()}
    train_data, val = (arrays(paths[split]) for split in ("train","val"))
    models = {}
    for kind in MODELS:
        folder = root/kind
        folder.mkdir(exist_ok=True)
        model, log = train(config,kind,train_data,val,device,folder,
                           {"config":config,"data_sha256":signatures,"kind":kind,"source":digest(Path(__file__))})
        models[kind] = (model,log,fit_scales(val,config,model,device))
    del train_data, val
    lock = {}
    for kind,(_,log,fit) in models.items():
        fixed_path = root/kind/"fixed.npy"
        np.save(fixed_path,fit[0])
        lock[kind] = {"weight_sha256":log["weight_sha256"],"pi":float(fit[1]),
                      "fixed_sha256":digest(fixed_path)}
    write_json(root/"lock.json", {"config":config,"models":lock,"data_sha256":signatures})
    for split in ("cal","id","paired","ood"):
        paths[split] = cache_data(config,split,root)
    cal = arrays(paths["cal"])
    results = {}
    for kind,(model,log,fit) in models.items():
        thresholds = calibrate(cal,config,model,device,fit)
        groups, raw, latency = {}, {}, {}
        for split,n in TARGETS:
            key = f"{split}_{n}"
            data = arrays(paths[split])
            groups[key], raw[key] = evaluate(data,n,config,model,device,fit,thresholds)
            if not args.smoke and split=="paired":
                latency[str(n)] = timing(data,n,config,model,device)
            print(f"Allen-Cahn {kind} evaluated {key}",flush=True)
        horizons = {}
        for key,windows in raw.items():
            long = {index:rows for index,rows in windows.items() if index[1]==config["frames"]}
            horizons[key] = paired_change(windows,long,config,base_frames=config["frames"]//2)
        np.savez_compressed(root/kind/"field_metrics.npz", **{
            f"{key}__{alpha}__{frames}__{method}__{column}":value
            for key, group in raw.items() for (alpha,frames), method_rows in group.items()
            for method, columns in method_rows.items() for column, value in columns.items()})
        reported_thresholds = {
            alpha:{method:q if np.isfinite(q) else "inf" for method,q in values.items()}
            for alpha,values in thresholds.items()
        }
        results[kind] = {"training":log,"thresholds":reported_thresholds,"results":groups,"latency_ms":latency,
                         "horizon_change":horizons,
                         "paired_grid_change":paired_change(raw["paired_64"],raw["paired_128"],config)}
    write_json(root/"metrics.json", {"seed":args.seed,"config":config,"results":results,
                                    "smoke":args.smoke,"gpu":torch.cuda.get_device_name() if device.type=="cuda" else None,
                                    "data_sha256":{split:digest(path) for split,path in paths.items()}})


if __name__=="__main__":
    main()
