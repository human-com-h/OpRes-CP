import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.stats import truncnorm


@lru_cache
def basis(n, modes):
    x = (np.arange(n) + 0.5) / n
    phi = np.cos(np.pi * x[:, None] * np.arange(modes))
    phi[:, 1:] *= np.sqrt(2.0)
    return phi


def coefficient(n, sigma, draw, modes, length, block):
    rng = np.random.default_rng(draw)
    if block:
        levels = rng.choice([-2.0, -0.5, 1.0, 2.5], (8, 8))
        levels -= levels.mean()
        levels /= max(levels.std(), 1e-12)
        index = ((np.arange(n) + 0.5) * 8 / n).astype(int)
        log_a = sigma * levels[np.ix_(index, index)]
    else:
        k = np.arange(modes, dtype=float)
        spectrum = (1 + (np.pi * length)**2 * (k[:, None]**2 + k[None, :]**2))**-1.25
        weights = rng.standard_normal((modes, modes)) * spectrum
        weights[0, 0] = 0
        phi = basis(n, modes)
        log_a = (sigma / np.linalg.norm(weights)) * (phi @ weights @ phi.T)
    return np.exp(log_a)


def faces(a):
    horizontal = 2 * a[..., :, :-1] * a[..., :, 1:] / (a[..., :, :-1] + a[..., :, 1:])
    vertical = 2 * a[..., :-1, :] * a[..., 1:, :] / (a[..., :-1, :] + a[..., 1:, :])
    return horizontal, vertical


def boundary_diagonal(a):
    edge = np.zeros_like(a)
    edge[..., 0, :] += 2 * a[..., 0, :]
    edge[..., -1, :] += 2 * a[..., -1, :]
    edge[..., :, 0] += 2 * a[..., :, 0]
    edge[..., :, -1] += 2 * a[..., :, -1]
    return edge


def apply_operator(a, u):
    horizontal, vertical = faces(a)
    result = boundary_diagonal(a) * u
    flux = horizontal * (u[..., :, :-1] - u[..., :, 1:])
    result[..., :, :-1] += flux
    result[..., :, 1:] -= flux
    flux = vertical * (u[..., :-1, :] - u[..., 1:, :])
    result[..., :-1, :] += flux
    result[..., 1:, :] -= flux
    return result * a.shape[-1]**2


@lru_cache
def matrix_indices(n):
    g = np.arange(n*n).reshape(n, n)
    i = np.r_[g[:, :-1].ravel(), g[:-1, :].ravel()]
    j = np.r_[g[:, 1:].ravel(), g[1:, :].ravel()]
    ids = g.ravel()
    return np.r_[ids, i, j], np.r_[ids, j, i]


def solve(a):
    n = a.shape[0]
    horizontal, vertical = faces(a)
    diagonal = boundary_diagonal(a)
    diagonal[:, :-1] += horizontal
    diagonal[:, 1:] += horizontal
    diagonal[:-1, :] += vertical
    diagonal[1:, :] += vertical
    links = -np.r_[horizontal.ravel(), vertical.ravel()]
    matrix = coo_matrix((np.r_[diagonal.ravel(), links, links], matrix_indices(n)),
                        shape=(n*n, n*n)).tocsc()
    return spsolve(matrix, np.full(n*n, 1 / n**2)).reshape(n, n)


def generate_one(job):
    field_id, split, sigma, draw, n, fine, modes, length = job
    a = coefficient(n, sigma, draw, modes, length, split == "ood_b")
    u = solve(a)
    residual = float(np.max(np.abs(apply_operator(a, u) - 1)))
    result = (a.astype(np.float32), u.astype(np.float32))
    if split == "paired_test":
        a_f = coefficient(fine, sigma, draw, modes, length, False)
        u_f = solve(a_f)
        residual = max(residual, float(np.max(np.abs(apply_operator(a_f, u_f) - 1))))
        result += (a_f.astype(np.float32), u_f.astype(np.float32))
    return field_id, sigma, residual, result


def generate(config, seed, directory, workers, include_ood, id_offset=0):
    directory.mkdir(parents=True, exist_ok=True)
    spec = config["sigma"]
    rng = np.random.default_rng(seed)
    start = id_offset
    from contextlib import nullcontext
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else nullcontext(None)
    with executor as pool:
        for split, count in config["counts"].items():
            mean = spec["ood_a_mean"] if split == "ood_a" else spec["id_mean"]
            std = spec["std"]
            bounds = (np.array(spec["bounds"]) - mean) / std
            sigmas = (rng.uniform(*spec["ood_b_bounds"], count) if split == "ood_b"
                      else truncnorm.rvs(*bounds, loc=mean, scale=std, size=count, random_state=rng))
            draws = rng.integers(0, 2**63, size=count)
            jobs = [(start + i, split, sigmas[i], draws[i], config["coarse"], config["fine"],
                     config["field_modes"], config["correlation_length"]) for i in range(count)]
            start += count
            path = directory / f"{split}.npz"
            if (split.startswith("ood") and not include_ood) or path.exists():
                continue
            entries = list(pool.map(generate_one, jobs, chunksize=8) if pool else map(generate_one, jobs))
            field_ids, sigma, residual, arrays = zip(*entries)
            fields = dict(zip(("a", "u", "a_f", "u_f"), map(np.stack, zip(*arrays))))
            temp = path.with_suffix(".tmp.npz")
            np.savez(temp, id=field_ids, sigma=sigma, reference_residual=residual, **fields)
            temp.replace(path)
            print(f"{split}: {count}, max reference residual {max(residual):.3g}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase0_darcy.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ood", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    generate(config, args.seed, Path("results") / f"seed_{args.seed}" / "data", args.workers, args.ood)
