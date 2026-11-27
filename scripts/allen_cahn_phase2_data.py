import argparse
import json
from pathlib import Path

import numpy as np
from scipy.fft import irfft2, rfft2

from burgers_phase1_data import phi


CONFIG = Path("configs/phase2_ac_probe.json")
ROOT = Path("results/phase2_ac_probe")


def initial(config, split):
    rng = np.random.default_rng(np.random.SeedSequence([config["seed"], config["streams"][split]]))
    count = config["counts"][split]
    rho = rng.uniform(*(config["ood_rho"] if split == "ood" else config["rho"]), count)
    nu = rng.uniform(*config["nu"], count)
    amp = rng.uniform(*config["rms"], count)
    white = rng.normal(size=(count, 16, 16))
    small = rfft2(white, norm="forward")
    kx, ky = np.fft.fftfreq(16)*16, np.arange(9)
    radial = kx[:, None]**2+ky**2
    small *= (radial <= 16)/(1+radial)**0.65
    norms = np.sqrt(np.mean(irfft2(small, norm="forward")**2, axis=(-2, -1)))
    small *= (amp/norms)[:, None, None]
    return nu, rho, small


def spectrum(n):
    kx, ky = np.fft.fftfreq(n)*n, np.arange(n//2+1)
    return kx[:, None]**2+ky**2, (abs(kx)[:, None] < n/4) & (ky[None] < n/4)


def project(z, n):
    if n == z.shape[-2]:
        return irfft2(z, s=(n,n), norm="forward")
    half = n//2
    low = np.concatenate((z[:, :half, :half+1], z[:, -half:, :half+1]), axis=-2)
    low[:, half, :] = 0
    low[..., -1] = 0
    return irfft2(low, s=(n,n), norm="forward")


def solve(coeff, nu, rho, config, grids, frames):
    n = config["reference"]
    k2, mask = spectrum(n)
    dt = config["dt"]/config["substeps"]
    expo = -dt*nu[:, None, None]*k2
    ex = np.exp(expo)
    p1, p2 = (dt*v for v in phi(expo))
    z = np.zeros((len(nu), n, n//2+1), complex)
    z[:, :8, :9], z[:, -8:, :9] = coeff[:, :8], coeff[:, 8:]
    result = {g: [project(z, g).astype(np.float32)] for g in grids}

    def reaction(v):
        u = irfft2(v, s=(n,n), norm="forward")
        return rho[:, None, None]*mask*rfft2(u-u**3, norm="forward")

    for _ in range(frames):
        for __ in range(config["substeps"]):
            f = reaction(z)
            stage = ex*z+p1*f
            z = stage+p2*(reaction(stage)-f)
        for g in grids:
            result[g].append(project(z, g).astype(np.float32))
    return {g: np.stack(parts, axis=1) for g, parts in result.items()}


def generate(config, split):
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT/f"{split}.npz"
    if path.exists():
        return
    nu, rho, coeff = initial(config, split)
    grids = config["grids"] if split == "paired" else [64]
    frames = config["frames"]//2 if split in ("train", "val", "cal") else config["frames"]
    output = {g: [] for g in grids}
    for i in range(0, len(nu), 12):
        sl = slice(i, i+12)
        block = solve(coeff[sl], nu[sl], rho[sl], config, grids, frames)
        for g in grids:
            output[g].append(block[g])
    payload = {f"u{g}": np.concatenate(x) for g,x in output.items()}
    if not all(np.isfinite(x).all() for x in payload.values()):
        raise RuntimeError(f"Nonfinite Allen–Cahn solution for {split}.")
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, nu=nu, rho=rho, id=config["streams"][split]*10000+np.arange(len(nu)), **payload)
    tmp.replace(path)
    print(f"Allen–Cahn {split}: {len(nu)} fields, {grids}, {frames} steps", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("splits", nargs="+", choices=("train","val","cal","id","paired","ood"))
    args = p.parse_args()
    if set(args.splits)-{"train","val"} and not (ROOT/"lock.json").exists():
        p.error("Generate evaluation data after the Val lock.")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for name in args.splits:
        generate(config, name)
