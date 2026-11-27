import argparse
import json
from pathlib import Path

import numpy as np
from scipy.fft import irfft, rfft


CONFIG = Path("configs/phase1_burgers.json")
ROOT = Path("results/phase1_burgers")


def phi(z):
    """ETDRK2 coefficients phi_1 and phi_2, including the zero mode."""
    nonzero = z != 0
    p1 = np.divide(np.expm1(z), z, out=np.ones_like(z), where=nonzero)
    p2 = np.divide(p1-1, z, out=np.full_like(z, 0.5), where=nonzero)
    return p1, p2


def initial_draw(config, split):
    rng = np.random.default_rng(np.random.SeedSequence([config["seed"], config["streams"][split]]))
    n = config["counts"][split]
    viscosity = config["viscosity_ood" if split == "ood" else "viscosity_id"]
    nu = np.exp(rng.uniform(*np.log(viscosity), size=n))
    modes = np.arange(1, 9)
    coeff = (rng.normal(size=(n, len(modes))) + 1j*rng.normal(size=(n, len(modes)))) / modes**1.35
    coeff *= rng.uniform(0.55, 1.25, size=(n, 1)) / np.sqrt(2*np.sum(abs(coeff)**2, axis=1, keepdims=True))
    mean = rng.uniform(-0.4, 0.4, size=(n, 1))
    return nu, np.concatenate([mean, coeff], axis=1)


def spectral_restriction(v, n):
    low = v[:, :n//2+1].copy()
    low[:, -1] = 0
    return irfft(low, n=n, norm="forward")


def solve(coeff, nu, config, grids, audit=False):
    n = config["reference_grid"]
    k = np.arange(n//2+1)
    ik = 1j*k*(k <= n//3)
    substeps = config["steps_per_frame"]
    dt = config["frame_dt"]/substeps
    z = -dt*nu[:, None]*k*k
    p1, p2 = phi(z)
    exp_z = np.exp(z)
    state = np.zeros((len(nu), len(k)), dtype=complex)
    state[:, :coeff.shape[1]] = coeff

    def flux(v):
        u = irfft(v, n=n, norm="forward")
        return -0.5*ik*rfft(u*u, norm="forward")

    trajectories = {g: [spectral_restriction(state, g)] for g in grids}
    defects = []
    mean_initial = state[:, 0].copy()
    for _ in range(config["frames"]):
        for _ in range(substeps):
            f = flux(state)
            provisional = exp_z*state + dt*p1*f
            updated = provisional + dt*p2*(flux(provisional)-f)
            if audit:
                d_t = (updated-state)/dt
                midpoint = (state+updated)*0.5
                differential = flux(midpoint) - nu[:, None]*k*k*midpoint
                defects.append(float(np.sqrt(np.mean(irfft(d_t-differential, n=n, norm="forward")**2))))
            state = updated
        for g in grids:
            trajectories[g].append(spectral_restriction(state, g))
    result = {g: np.stack(t, axis=1) for g, t in trajectories.items()}
    return result, {"internal_dt": dt, "internal_midpoint_residual_rms": float(np.mean(defects)) if defects else None,
                    "mean_drift_max": float(np.max(abs(state[:, 0]-mean_initial)))}


def generate(config, split):
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / f"{split}.npz"
    if path.exists():
        return
    grids = config["grids"] if split == "paired" else config["grids"][:1]
    nu, coeff = initial_draw(config, split)
    parts = {g: [] for g in grids}
    for offset in range(0, len(nu), 32):
        sl = slice(offset, offset+32)
        values, _ = solve(coeff[sl], nu[sl], config, grids)
        for g in grids:
            parts[g].append(values[g].astype(np.float32))
    payload = {f"u{g}": np.concatenate(parts[g]) for g in grids}
    if not all(np.isfinite(u).all() for u in payload.values()):
        raise RuntimeError(f"Non-finite ETDRK2 solution for {split}.")
    temp = path.with_suffix(".tmp.npz")
    np.savez(temp, nu=nu, id=config["streams"][split]*10000+np.arange(len(nu)), **payload)
    temp.replace(path)
    print(f"Burgers data {split}: {len(nu)} fields on {grids}", flush=True)


def generator_audit(config):
    output = {}
    for split in ("val", "ood"):
        nu, coeff = initial_draw(config, split)
        grids = config["grids"]
        ref, stats = solve(coeff[:12], nu[:12], config, grids, audit=True)
        refinement = {**config, "reference_grid": 1024, "steps_per_frame": config["steps_per_frame"]*2}
        fine, stats_fine = solve(coeff[:12], nu[:12], refinement, grids, audit=True)
        relative = {str(g): float(np.linalg.norm(ref[g]-fine[g])/np.linalg.norm(fine[g])) for g in grids}
        output[split] = {"coarse": stats, "refined": stats_fine, "relative_l2_to_refined": relative}
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("splits", nargs="+", choices=("train", "val", "cal", "id", "paired", "ood"))
    args = parser.parse_args()
    spec = json.loads(CONFIG.read_text(encoding="utf-8"))
    if set(args.splits) & {"cal", "id", "paired", "ood"} and not (ROOT / "lock.json").is_file():
        parser.error("Generate evaluation labels after the Val lock.")
    for name in args.splits:
        generate(spec, name)
