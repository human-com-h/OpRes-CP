"""P2-A: Non-rectangular L-shaped 2D Darcy benchmark with Galerkin AMG lifting.

Demonstrates OpRes-CP on an irregular re-entrant domain Omega_L = (0,1)^2 \\ [0.5,1)^2
where FFT/DST spectral inverses do not apply. Replaces spectral lifting with O(M)
cell-agglomeration Galerkin algebraic multigrid (AMG-V2 / AMG-PCG2) and mask-normalized
physical Gaussian envelopes across the five formal seeds (42, 137, 314, 271, 811).

Strict Train/Val/Cal/Test isolation; zero modifications to sealed Phase 2D/3 archives.
Outputs written under results/phase3_p2/lshape_darcy/.
"""

import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from time import perf_counter, perf_counter_ns

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.stats import truncnorm
import torch
from torch import nn
from torch.nn import functional as F

from darcy_data import coefficient
from darcy_day15 import value_at_coverage
from darcy_fno import SpectralBlock, atomic_checkpoint, relative_l2
from darcy_phase0_eval import boot_interval, quantile
from phase3_summarize import SEEDS, stat


OUT = Path("results/phase3_p2/lshape_darcy")
FIG_OUT = Path("results/phase3_p2/figures")
PAPER_FIG = Path("paper/figures") if Path("paper").is_dir() else Path("outputs/figures")

# Locked protocol mirroring configs/phase0_darcy.json + phase2_darcy.json
CFG = {
    "seeds": list(SEEDS),
    "coarse": 64,
    "fine": 128,
    "counts": {
        "train": 800,
        "val": 200,
        "cal": 300,
        "id_test": 300,
        "paired_test": 300,
        "ood_a": 300,
    },
    "streams": {
        "train": 8201,
        "val": 8202,
        "cal": 8203,
        "id_test": 8204,
        "paired_test": 8205,
        "ood_a": 8206,
    },
    "sigma": {"id_mean": 1.0, "ood_a_mean": 1.35, "std": 0.35, "bounds": [0.4, 2.0]},
    "field_modes": 32,
    "correlation_length": 0.08,
    "model": {"modes": 12, "width": 32, "layers": 4, "epochs": 30, "batch_size": 16, "learning_rate": 1.5e-3},
    "ensemble_init_offsets": [0, 10000, 20000, 30000, 40000],
    "physical_radius": 0.04,
    "relative_floor": 0.1,
    "lambda": 0.85,
    "alpha": 0.1,
    "bootstrap_seed": 6187,
    "bootstrap_resamples": 2000,
}


@lru_cache
def lshape_mask(n):
    """Active cell mask for Omega_L = (0,1)^2 \\ [0.5, 1)^2."""
    x = (np.arange(n) + 0.5) / n
    xx, yy = np.meshgrid(x, x, indexing="ij")
    return ~((xx > 0.5) & (yy > 0.5))


@lru_cache
def lshape_boundary_weight(n):
    """Exact piecewise-linear distance to partial Omega_L, normalized to [0, 1]."""
    x = (np.arange(n) + 0.5) / n
    xx, yy = np.meshgrid(x, x, indexing="ij")
    mask = lshape_mask(n)
    d_outer = np.minimum.reduce((xx, 1.0 - xx, yy, 1.0 - yy))
    d_reent = np.maximum(0.5 - xx, 0.5 - yy)
    d = np.where(mask, np.minimum(d_outer, np.maximum(d_reent, 0.0)), 0.0)
    return (d / max(float(d.max()), 1e-12)).astype(np.float32)


@lru_cache
def lshape_corner_mask(n, radius=0.15):
    """Active cells within Euclidean distance `radius` of the re-entrant corner (0.5, 0.5)."""
    x = (np.arange(n) + 0.5) / n
    xx, yy = np.meshgrid(x, x, indexing="ij")
    return lshape_mask(n) & (((xx - 0.5) ** 2 + (yy - 0.5) ** 2) <= radius ** 2)


def build_fvm_level0(a):
    """Construct flux-form 5-point FVM stencil (tx, ty, bdiag, diag) on Omega_L.

    The PDE operator is A_h u = n^2 * A_flux u, so A_flux u = r / n^2 solves A_h u = r.
    """
    n = a.shape[-1]
    m = lshape_mask(n)
    # Interior active-active harmonic face transmissibilities
    tx = (2.0 * a[:, :-1] * a[:, 1:] / (a[:, :-1] + a[:, 1:])) * (m[:, :-1] & m[:, 1:])
    ty = (2.0 * a[:-1, :] * a[1:, :] / (a[:-1, :] + a[1:, :])) * (m[:-1, :] & m[1:, :])
    # Dirichlet faces: any side of an active cell adjacent to domain exterior or cutout
    bdiag = np.zeros_like(a)
    # Left / right faces
    left_ext = np.ones((n, n), dtype=bool)
    left_ext[:, 1:] = ~m[:, :-1]
    right_ext = np.ones((n, n), dtype=bool)
    right_ext[:, :-1] = ~m[:, 1:]
    # Up / down faces
    up_ext = np.ones((n, n), dtype=bool)
    up_ext[1:, :] = ~m[:-1, :]
    down_ext = np.ones((n, n), dtype=bool)
    down_ext[:-1, :] = ~m[1:, :]
    n_dirichlet_faces = (left_ext.astype(int) + right_ext.astype(int) +
                         up_ext.astype(int) + down_ext.astype(int))
    bdiag = np.where(m, 2.0 * a * n_dirichlet_faces, 0.0)
    diag = bdiag.copy()
    diag[:, :-1] += tx
    diag[:, 1:] += tx
    diag[:-1, :] += ty
    diag[1:, :] += ty
    inv_diag = np.where(m, 1.0 / np.maximum(diag, 1e-30), 0.0)
    return {"n": n, "mask": m, "tx": tx, "ty": ty, "bdiag": bdiag, "diag": diag, "inv_diag": inv_diag}


def apply_flux(lvl, u):
    """Apply flux-form stencil A_flux to field u (single field, shape (n, n))."""
    y = lvl["diag"] * u
    y[:, :-1] -= lvl["tx"] * u[:, 1:]
    y[:, 1:] -= lvl["tx"] * u[:, :-1]
    y[:-1, :] -= lvl["ty"] * u[1:, :]
    y[1:, :] -= lvl["ty"] * u[:-1, :]
    return y


def apply_pde(a, u):
    """Apply A_h u = n^2 * A_flux u to a single field or batch."""
    if a.ndim == 2:
        lvl = build_fvm_level0(a)
        return apply_flux(lvl, u) * (a.shape[-1] ** 2)
    return np.stack([apply_pde(aa, uu) for aa, uu in zip(a, u)])


def assemble_sparse_flux(lvl):
    """Assemble exact symmetric positive definite CSR/CSC matrix on active cells."""
    n = lvl["n"]
    m = lvl["mask"]
    idx = np.full((n, n), -1, dtype=np.int32)
    active = np.flatnonzero(m)
    idx.ravel()[active] = np.arange(len(active), dtype=np.int32)
    rows, cols, vals = [], [], []
    # Diagonal
    d_idx = np.arange(len(active), dtype=np.int32)
    rows.append(d_idx)
    cols.append(d_idx)
    vals.append(lvl["diag"][m])
    # Horizontal edges
    h_mask = m[:, :-1] & m[:, 1:]
    hi, hj = idx[:, :-1][h_mask], idx[:, 1:][h_mask]
    hv = -lvl["tx"][h_mask]
    rows.extend([hi, hj])
    cols.extend([hj, hi])
    vals.extend([hv, hv])
    # Vertical edges
    v_mask = m[:-1, :] & m[1:, :]
    vi, vj = idx[:-1, :][v_mask], idx[1:, :][v_mask]
    vv = -lvl["ty"][v_mask]
    rows.extend([vi, vj])
    cols.extend([vj, vi])
    vals.extend([vv, vv])
    mat = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                     shape=(len(active), len(active))).tocsc()
    return mat, active


def solve_lshape(a, rhs_pde=None):
    """Exact sparse solve of A_h u = rhs_pde (default rhs_pde = 1 on Omega_L)."""
    n = a.shape[-1]
    lvl = build_fvm_level0(a)
    mat, active = assemble_sparse_flux(lvl)
    if rhs_pde is None:
        b = np.full(len(active), 1.0 / (n * n), dtype=np.float64)
    else:
        b = rhs_pde.ravel()[active] / (n * n)
    sol = spsolve(mat, b)
    u = np.zeros((n, n), dtype=np.float64)
    u.ravel()[active] = sol
    return u


def build_amg_hierarchy(a, min_n=4):
    """Build exact 2x2 cell-agglomeration Galerkin AMG hierarchy A_{l+1} = P_l^T A_l P_l."""
    levels = [build_fvm_level0(a)]
    while levels[-1]["n"] > min_n:
        f = levels[-1]
        nc = f["n"] // 2
        m_c = f["mask"].reshape(nc, 2, nc, 2).any(axis=(1, 3))
        tx_c = f["tx"][:, 1::2].reshape(nc, 2, nc - 1).sum(axis=1)
        ty_c = f["ty"][1::2, :].reshape(nc - 1, nc, 2).sum(axis=2)
        bdiag_c = f["bdiag"].reshape(nc, 2, nc, 2).sum(axis=(1, 3))
        diag_c = bdiag_c.copy()
        diag_c[:, :-1] += tx_c
        diag_c[:, 1:] += tx_c
        diag_c[:-1, :] += ty_c
        diag_c[1:, :] += ty_c
        inv_diag_c = np.where(m_c, 1.0 / np.maximum(diag_c, 1e-30), 0.0)
        levels.append({"n": nc, "mask": m_c, "tx": tx_c, "ty": ty_c,
                       "bdiag": bdiag_c, "diag": diag_c, "inv_diag": inv_diag_c})
    # Exact coarsest solve matrix (12 active unknowns at n=4)
    cmat, cactive = assemble_sparse_flux(levels[-1])
    levels[-1]["coarse_mat"] = cmat
    levels[-1]["coarse_active"] = cactive
    # Precompute red-black checkerboard masks per level
    for lvl in levels:
        n = lvl["n"]
        ii, jj = np.indices((n, n))
        lvl["red"] = ((ii + jj) & 1) == 0
        lvl["black"] = ~lvl["red"]
    return levels


def _rb_gs_step(lvl, u, b, color):
    """One vectorized half-sweep of Red-Black Gauss-Seidel on masked 5-point stencil."""
    rhs = b.copy()
    rhs[:, :-1] += lvl["tx"] * u[:, 1:]
    rhs[:, 1:] += lvl["tx"] * u[:, :-1]
    rhs[:-1, :] += lvl["ty"] * u[1:, :]
    rhs[1:, :] += lvl["ty"] * u[:-1, :]
    c = lvl[color]
    u[c] = rhs[c] * lvl["inv_diag"][c]


def amg_vcycle(levels, b, idx=0, nu=2, omega_coarse=1.0):
    """Symmetric Galerkin V-cycle (RB pre-smooth, exact coarsest, BR post-smooth)."""
    lvl = levels[idx]
    if idx == len(levels) - 1:
        u = np.zeros_like(b)
        act = lvl["coarse_active"]
        u.ravel()[act] = spsolve(lvl["coarse_mat"], b.ravel()[act])
        return u
    u = np.zeros_like(b)
    for _ in range(nu):
        _rb_gs_step(lvl, u, b, "red")
        _rb_gs_step(lvl, u, b, "black")
    res = b - apply_flux(lvl, u)
    nc = lvl["n"] // 2
    r_c = res.reshape(nc, 2, nc, 2).sum(axis=(1, 3))
    e_c = amg_vcycle(levels, r_c, idx=idx + 1, nu=nu, omega_coarse=omega_coarse)
    e_f = np.repeat(np.repeat(e_c, 2, axis=0), 2, axis=1)
    u += omega_coarse * np.where(lvl["mask"], e_f, 0.0)
    for _ in range(nu):
        _rb_gs_step(lvl, u, b, "black")
        _rb_gs_step(lvl, u, b, "red")
    return u


def lift_amg_v2(levels, r_pde, cycles=2, nu=2, omega_coarse=1.6):
    """Standalone 2-V-cycle Galerkin AMG lift (strictly O(M), over-relaxed UA-AMG)."""
    n = levels[0]["n"]
    b = r_pde / (n * n)
    x = np.zeros_like(b)
    for _ in range(cycles):
        res = b - apply_flux(levels[0], x)
        x += amg_vcycle(levels, res, idx=0, nu=nu, omega_coarse=omega_coarse)
    return x


def lift_amg_pcg(levels, r_pde, steps=2, nu=2):
    """PCG preconditioned by symmetric Galerkin V-cycle (strictly SPD, O(M))."""
    n = levels[0]["n"]
    lvl0 = levels[0]
    b = r_pde / (n * n)
    x = np.zeros_like(b)
    res = b.copy()
    z = amg_vcycle(levels, res, idx=0, nu=nu, omega_coarse=1.0)
    p = z.copy()
    rz = float(np.sum(res * z))
    for i in range(steps):
        if rz <= 1e-30:
            break
        ap = apply_flux(lvl0, p)
        denom = float(np.sum(p * ap))
        if denom <= 1e-30:
            break
        alpha = rz / denom
        x += alpha * p
        res -= alpha * ap
        if i + 1 < steps:
            z = amg_vcycle(levels, res, idx=0, nu=nu, omega_coarse=1.0)
            next_rz = float(np.sum(res * z))
            p = z + (next_rz / rz) * p
            rz = next_rz
    return x


def lift_jacobi_pcg(levels, r_pde, steps=5):
    """Jacobi-preconditioned CG baseline (no multigrid hierarchy)."""
    n = levels[0]["n"]
    lvl0 = levels[0]
    b = r_pde / (n * n)
    x = np.zeros_like(b)
    res = b.copy()
    z = lvl0["inv_diag"] * res
    p = z.copy()
    rz = float(np.sum(res * z))
    for i in range(steps):
        if rz <= 1e-30:
            break
        ap = apply_flux(lvl0, p)
        denom = float(np.sum(p * ap))
        if denom <= 1e-30:
            break
        alpha = rz / denom
        x += alpha * p
        res -= alpha * ap
        if i + 1 < steps:
            z = lvl0["inv_diag"] * res
            next_rz = float(np.sum(res * z))
            p = z + (next_rz / rz) * p
            rz = next_rz
    return x


def verify_amg_unit_tests():
    """Verify exact Galerkin identity, SPD symmetry of V-cycle, and contraction rate."""
    a = coefficient(64, 1.0, 98765, 32, 0.08, False)
    levels = build_amg_hierarchy(a)
    rng = np.random.default_rng(123)
    # 1. Exact Galerkin check: u_c^T A_{l+1} v_c == (P u_c)^T A_l (P v_c)
    max_gal_err = 0.0
    for idx in range(len(levels) - 1):
        lf, lc = levels[idx], levels[idx + 1]
        uc = rng.normal(size=(lc["n"], lc["n"])) * lc["mask"]
        vc = rng.normal(size=(lc["n"], lc["n"])) * lc["mask"]
        uf = np.repeat(np.repeat(uc, 2, axis=0), 2, axis=1) * lf["mask"]
        vf = np.repeat(np.repeat(vc, 2, axis=0), 2, axis=1) * lf["mask"]
        lhs = float(np.sum(uc * apply_flux(lc, vc)))
        rhs = float(np.sum(uf * apply_flux(lf, vf)))
        max_gal_err = max(max_gal_err, abs(lhs - rhs) / max(abs(rhs), 1.0))
    assert max_gal_err < 1e-13, f"Galerkin check failed: {max_gal_err}"
    # 2. V-cycle symmetry check: x^T B y == y^T B x
    x = rng.normal(size=(64, 64)) * levels[0]["mask"]
    y = rng.normal(size=(64, 64)) * levels[0]["mask"]
    bx = amg_vcycle(levels, x, idx=0, nu=2, omega_coarse=1.0)
    by = amg_vcycle(levels, y, idx=0, nu=2, omega_coarse=1.0)
    sym_err = abs(float(np.sum(x * by)) - float(np.sum(y * bx))) / max(abs(float(np.sum(x * by))), 1e-12)
    assert sym_err < 1e-12, f"V-cycle symmetry check failed: {sym_err}"
    # 3. Exact solve residual + AMG contraction
    u_true = solve_lshape(a)
    m = levels[0]["mask"]
    ref_res = float(np.max(np.abs((apply_pde(a, u_true) - 1.0)[m])))
    assert ref_res < 1e-11, f"L-shape reference solver residual too large: {ref_res}"
    r_test = np.where(m, 1.0, 0.0)
    z_v2 = lift_amg_v2(levels, r_test)
    z_pcg2 = lift_amg_pcg(levels, r_test, steps=2)
    z_j5 = lift_jacobi_pcg(levels, r_test, steps=5)
    err_v2 = float(np.linalg.norm((z_v2 - u_true)[m]) / np.linalg.norm(u_true[m]))
    err_pcg2 = float(np.linalg.norm((z_pcg2 - u_true)[m]) / np.linalg.norm(u_true[m]))
    err_j5 = float(np.linalg.norm((z_j5 - u_true)[m]) / np.linalg.norm(u_true[m]))
    out = {
        "galerkin_max_rel_error": max_gal_err,
        "vcycle_symmetry_rel_error": sym_err,
        "reference_solver_max_residual": ref_res,
        "unit_rhs_rel_l2_error": {"amg_v2": err_v2, "amg_pcg2": err_pcg2, "jacobi_pcg5": err_j5},
    }
    print(f"[P2-A Unit Check] Galerkin={max_gal_err:.2e}, Sym={sym_err:.2e}, RefRes={ref_res:.2e}, "
          f"RelErr: V2={err_v2:.3f}, PCG2={err_pcg2:.3f}, J5={err_j5:.3f}", flush=True)
    return out


def masked_physical_scale(z, cfg):
    """Mask-normalized physical Gaussian envelope on Omega_L + relative RMS floor + lambda mix.

    Implements G_{ell,h}(z^2) = (K_ell * (z^2 1_Omega)) / (K_ell * 1_Omega), matching Prop. 1.
    """
    n = z.shape[-1]
    m = lshape_mask(n)
    m_f = m.astype(np.float64)
    radius = cfg["physical_radius"] * n
    denom = np.maximum(gaussian_filter(m_f, (radius, radius), mode="constant", cval=0.0), 1e-12)
    z2 = np.where(m, z * z, 0.0)
    if z.ndim == 2:
        num = gaussian_filter(z2, (radius, radius), mode="constant", cval=0.0)
        rms_z = math.sqrt(float(np.mean(z2[m])))
        env = np.where(m, np.sqrt(num / denom + (cfg["relative_floor"] * rms_z) ** 2), 0.0)
        rms_s = math.sqrt(float(np.mean((env * env)[m])))
        lam = cfg["lambda"]
        return np.where(m, np.sqrt((1.0 - lam) * rms_s * rms_s + lam * env * env), 0.0)
    out = np.empty_like(z, dtype=np.float64)
    for i in range(len(z)):
        num = gaussian_filter(z2[i], (radius, radius), mode="constant", cval=0.0)
        rms_z = math.sqrt(float(np.mean(z2[i][m])))
        env = np.where(m, np.sqrt(num / denom + (cfg["relative_floor"] * rms_z) ** 2), 0.0)
        rms_s = math.sqrt(float(np.mean((env * env)[m])))
        lam = cfg["lambda"]
        out[i] = np.where(m, np.sqrt((1.0 - lam) * rms_s * rms_s + lam * env * env), 0.0)
    return out


def _gen_one_lshape(job):
    field_id, split, sigma, draw, n, fine, modes, length = job
    a = coefficient(n, sigma, draw, modes, length, False)
    u = solve_lshape(a)
    m = lshape_mask(n)
    res = float(np.max(np.abs((apply_pde(a, u) - 1.0)[m])))
    out = (a.astype(np.float32), u.astype(np.float32))
    if split == "paired_test":
        a_f = coefficient(fine, sigma, draw, modes, length, False)
        u_f = solve_lshape(a_f)
        mf = lshape_mask(fine)
        res = max(res, float(np.max(np.abs((apply_pde(a_f, u_f) - 1.0)[mf]))))
        out += (a_f.astype(np.float32), u_f.astype(np.float32))
    return field_id, sigma, res, out


def ensure_lshape_data(seed, workers=4):
    root = OUT / f"seed_{seed}" / "data"
    root.mkdir(parents=True, exist_ok=True)
    spec = CFG["sigma"]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for split, count in CFG["counts"].items():
            path = root / f"{split}.npz"
            if path.exists():
                continue
            stream = CFG["streams"][split]
            rng = np.random.default_rng(np.random.SeedSequence([seed, stream]))
            mean = spec["ood_a_mean"] if split == "ood_a" else spec["id_mean"]
            std = spec["std"]
            bounds = (np.array(spec["bounds"]) - mean) / std
            sigmas = truncnorm.rvs(*bounds, loc=mean, scale=std, size=count, random_state=rng)
            draws = rng.integers(0, 2 ** 63, size=count)
            id_offset = stream * 10000
            jobs = [(id_offset + i, split, float(sigmas[i]), int(draws[i]),
                     CFG["coarse"], CFG["fine"], CFG["field_modes"], CFG["correlation_length"])
                    for i in range(count)]
            entries = list(pool.map(_gen_one_lshape, jobs, chunksize=8))
            fids, sigs, ress, arrs = zip(*entries)
            fields = dict(zip(("a", "u", "a_f", "u_f"), map(np.stack, zip(*arrs))))
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, id=np.array(fids), sigma=np.array(sigs), reference_residual=np.array(ress), **fields)
            tmp.replace(path)
            print(f"[P2-A Data] seed={seed} {split}: n={count}, max_ref_res={max(ress):.2e}", flush=True)
    return root


class LShapeFNO(nn.Module):
    """Mask-aware 2D FNO enforcing homogeneous Dirichlet BCs on partial Omega_L."""

    def __init__(self, spec):
        super().__init__()
        width = spec["width"]
        self.lift = nn.Conv2d(4, width, 1)
        self.blocks = nn.ModuleList(SpectralBlock(width, spec["modes"]) for _ in range(spec["layers"]))
        self.project = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, 1, 1))
        self.register_buffer("input_std", torch.ones(()))
        self.register_buffer("output_scale", torch.ones(()))

    def forward(self, a):
        batch, n, m = a.shape
        coord_x = (torch.arange(n, device=a.device, dtype=a.dtype) + 0.5) / n
        coord_y = (torch.arange(m, device=a.device, dtype=a.dtype) + 0.5) / m
        x, y = torch.meshgrid(coord_x, coord_y, indexing="ij")
        mask = torch.as_tensor(lshape_mask(n), device=a.device, dtype=a.dtype)
        bweight = torch.as_tensor(lshape_boundary_weight(n), device=a.device, dtype=a.dtype)
        log_a = (a.log() / self.input_std) * mask
        feats = torch.cat((
            log_a[:, None],
            mask[None, None].expand(batch, 1, n, m),
            x[None, None].expand(batch, 1, n, m),
            y[None, None].expand(batch, 1, n, m),
        ), dim=1)
        z = self.lift(feats)
        for block in self.blocks:
            z = F.gelu(block(z))
        return self.project(z)[:, 0] * bweight * self.output_scale


def masked_rel_l2(pred, truth, mask_t):
    diff = ((pred - truth) * mask_t).flatten(1).norm(dim=1)
    denom = (truth * mask_t).flatten(1).norm(dim=1)
    return (diff / denom).mean()


def train_lshape_member(seed, init_seed, data_root, ckpt_path, device):
    spec = CFG["model"]
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=True)
        model = LShapeFNO(spec).to(device)
        model.load_state_dict(ck["model"])
        return model.eval(), float(ck["train_seconds"])
    with np.load(data_root / "train.npz") as tr, np.load(data_root / "val.npz") as va:
        tr_a, tr_u = torch.from_numpy(tr["a"]).to(device), torch.from_numpy(tr["u"]).to(device)
        va_a, va_u = torch.from_numpy(va["a"]).to(device), torch.from_numpy(va["u"]).to(device)
    mask_t = torch.as_tensor(lshape_mask(CFG["coarse"]), device=device, dtype=torch.float32)
    torch.manual_seed(init_seed)
    model = LShapeFNO(spec).to(device)
    model.input_std.copy_(tr_a.log()[:, lshape_mask(CFG["coarse"])].std())
    model.output_scale.copy_(tr_u[:, lshape_mask(CFG["coarse"])].square().mean().sqrt())
    opt = torch.optim.AdamW(model.parameters(), lr=spec["learning_rate"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, spec["epochs"], eta_min=5e-5)
    rng = torch.Generator(device="cpu").manual_seed(init_seed)
    best, best_state = float("inf"), None
    t0 = perf_counter()
    for _ in range(spec["epochs"]):
        model.train()
        for ids in torch.randperm(len(tr_a), generator=rng).split(spec["batch_size"]):
            opt.zero_grad(set_to_none=True)
            loss = masked_rel_l2(model(tr_a[ids]), tr_u[ids], mask_t)
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.inference_mode():
            vloss = float(sum(masked_rel_l2(model(pa), pu, mask_t) * len(pa)
                              for pa, pu in zip(va_a.split(32), va_u.split(32))) / len(va_a))
        if vloss < best:
            best = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    elapsed = perf_counter() - t0
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_checkpoint(ckpt_path, {"model": best_state, "val_rel_l2": best, "train_seconds": elapsed})
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model.eval(), elapsed


@torch.inference_mode()
def predict_lshape(model, a_np, device, batch=16):
    parts = []
    for i in range(0, len(a_np), batch):
        x = torch.from_numpy(a_np[i:i + batch]).to(device)
        parts.append(model(x).cpu().numpy())
    return np.concatenate(parts, axis=0)


def bilinear_lshape(field_64, n_target):
    """Map Val-mean spatial error map from 64^2 to 128^2 via nearest/zoom on Omega_L."""
    if field_64.shape[-1] == n_target:
        return field_64
    scale_factor = n_target // field_64.shape[-1]
    up = np.repeat(np.repeat(field_64, scale_factor, axis=0), scale_factor, axis=1)
    return np.where(lshape_mask(n_target), up, 0.0)


def eval_field_metrics(pred, truth, scale, q, n):
    """Evaluate simultaneous, pointwise, and re-entrant corner coverage and widths."""
    m = lshape_mask(n)
    cm = lshape_corner_mask(n, radius=0.15)
    err = np.abs(pred - truth)[:, m]
    sc = np.maximum(scale[:, m], 1e-12)
    ratio = err / sc
    scores = ratio.max(axis=1)
    mean_s = sc.mean(axis=1)
    cov = scores <= q
    act_w = 2.0 * q * mean_s
    pt_cov = (ratio <= q).mean(axis=1)
    # Re-entrant corner metrics
    c_err = np.abs(pred - truth)[:, cm]
    c_sc = np.maximum(scale[:, cm], 1e-12)
    c_ratio = c_err / c_sc
    corner_sim_cov = (c_ratio.max(axis=1) <= q)
    corner_pt_cov = (c_ratio <= q).mean(axis=1)
    corner_w = 2.0 * q * c_sc.mean(axis=1)
    q90_emp = value_at_coverage(scores, 0.9)
    matched_w = 2.0 * q90_emp * float(mean_s.mean())
    rel_l2 = (np.linalg.norm(err, axis=1) / np.linalg.norm(truth[:, m], axis=1))
    return {
        "scores": scores,
        "scale_mean": mean_s,
        "cov_vec": cov,
        "coverage": float(cov.mean()),
        "actual_width": float(act_w.mean()),
        "matched_width": float(matched_w),
        "point_cov": float(pt_cov.mean()),
        "corner_sim_cov": float(corner_sim_cov.mean()),
        "corner_pt_cov": float(corner_pt_cov.mean()),
        "corner_width": float(corner_w.mean()),
        "point_rel_l2": float(rel_l2.mean()),
    }


def _lift_one_field(args):
    a, p = args
    n = a.shape[-1]
    m = lshape_mask(n)
    levels = build_amg_hierarchy(a)
    r = np.where(m, apply_flux(levels[0], p) * (n * n) - 1.0, 0.0)
    z_v2 = lift_amg_v2(levels, r, cycles=2, nu=2, omega_coarse=1.6)
    z_pcg2 = lift_amg_pcg(levels, r, steps=2, nu=2)
    z_unsigned = lift_amg_v2(levels, np.abs(r), cycles=2, nu=2, omega_coarse=1.6)
    z_j5 = lift_jacobi_pcg(levels, r, steps=5)
    z_exact = solve_lshape(a, rhs_pde=r)
    return r, z_v2, z_pcg2, z_unsigned, z_j5, z_exact


def compute_physics_scales(a_arr, pred_arr, fixed_map, pi_lam, member_preds, ens_floor, workers=4):
    """Compute all compared conformal scales on Omega_L (parallelized across CPU workers)."""
    n = a_arr.shape[-1]
    m = lshape_mask(n)
    k = len(a_arr)
    jobs = [(a_arr[i].astype(np.float64), pred_arr[i].astype(np.float64)) for i in range(k)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_lift_one_field, jobs, chunksize=8))
    residuals, z_v2, z_pcg2, z_unsigned, z_j5, z_exact = (np.stack(arr) for arr in zip(*results))

    ens_std = member_preds.std(axis=0, ddof=1, dtype=np.float64)
    ens_scale = np.where(m, np.hypot(ens_std, ens_floor), 0.0)
    fixed_n = bilinear_lshape(fixed_map, n)

    scales = {
        "Uniform-CP": np.broadcast_to(m.astype(np.float64), (k, n, n)).copy(),
        "Choi CP-B": np.broadcast_to(fixed_n, (k, n, n)).copy(),
        "Full-Field PI-CP": np.where(m, 1.0 + pi_lam * np.abs(residuals), 0.0),
        "Ensemble (same center)": ens_scale,
        "Ensemble (mean center)": ens_scale,
        "Unsigned AMG-V2": masked_physical_scale(z_unsigned, CFG),
        "Jacobi-PCG5": masked_physical_scale(z_j5, CFG),
        "OpRes-CP (AMG-V2)": masked_physical_scale(z_v2, CFG),
        "OpRes-CP (AMG-PCG2)": masked_physical_scale(z_pcg2, CFG),
        "Exact Sparse Lift (oracle)": masked_physical_scale(z_exact, CFG),
    }
    return scales, residuals, z_pcg2


def benchmark_timings(models, data_id, data_paired, device):
    """Benchmark single-case P50/P90 timings on 64^2 and 128^2 L-shape grids."""
    res = {}
    for label, a_arr in (("64", data_id["a"][:30]), ("128", data_paired["a_f"][:30])):
        n = a_arr.shape[-1]
        m = lshape_mask(n)
        # Warmup
        _ = predict_lshape(models[0], a_arr[:1], device, batch=1)
        lv0 = build_amg_hierarchy(a_arr[0].astype(np.float64))
        r0 = np.where(m, apply_flux(lv0[0], np.zeros((n, n))) * (n * n) - 1.0, 0.0)
        _ = lift_amg_v2(lv0, r0)
        _ = lift_amg_pcg(lv0, r0, steps=2)
        _ = lift_jacobi_pcg(lv0, r0, steps=5)
        _ = solve_lshape(a_arr[0].astype(np.float64), rhs_pde=r0)
        torch.cuda.synchronize(device)

        t_fno, t_ens, t_setup, t_v2, t_pcg2, t_j5, t_ex, t_env = ([] for _ in range(8))
        for i in range(5, 25):
            ai = a_arr[i:i + 1]
            torch.cuda.synchronize(device)
            t0 = perf_counter_ns()
            pi = predict_lshape(models[0], ai, device, batch=1)[0].astype(np.float64)
            torch.cuda.synchronize(device)
            t_fno.append((perf_counter_ns() - t0) / 1e6)

            torch.cuda.synchronize(device)
            t0 = perf_counter_ns()
            for mdl in models:
                _ = predict_lshape(mdl, ai, device, batch=1)
            torch.cuda.synchronize(device)
            t_ens.append((perf_counter_ns() - t0) / 1e6)

            a64 = ai[0].astype(np.float64)
            t0 = perf_counter_ns()
            lv = build_amg_hierarchy(a64)
            r = np.where(m, apply_flux(lv[0], pi) * (n * n) - 1.0, 0.0)
            t_setup.append((perf_counter_ns() - t0) / 1e6)

            t0 = perf_counter_ns()
            z = lift_amg_v2(lv, r, cycles=2, nu=2, omega_coarse=1.6)
            t_v2.append((perf_counter_ns() - t0) / 1e6)

            t0 = perf_counter_ns()
            z2 = lift_amg_pcg(lv, r, steps=2, nu=2)
            t_pcg2.append((perf_counter_ns() - t0) / 1e6)

            t0 = perf_counter_ns()
            _ = lift_jacobi_pcg(lv, r, steps=5)
            t_j5.append((perf_counter_ns() - t0) / 1e6)

            t0 = perf_counter_ns()
            _ = solve_lshape(a64, rhs_pde=r)
            t_ex.append((perf_counter_ns() - t0) / 1e6)

            t0 = perf_counter_ns()
            _ = masked_physical_scale(z2, CFG)
            t_env.append((perf_counter_ns() - t0) / 1e6)

        def q50_90(v):
            return {"p50_ms": float(np.median(v)), "p90_ms": float(np.quantile(v, 0.9))}

        res[label] = {
            "fno_single": q50_90(t_fno),
            "ensemble_5": q50_90(t_ens),
            "amg_setup_plus_res": q50_90(t_setup),
            "amg_v2_solve": q50_90(t_v2),
            "amg_pcg2_solve": q50_90(t_pcg2),
            "jacobi_pcg5_solve": q50_90(t_j5),
            "exact_oracle_solve": q50_90(t_ex),
            "masked_envelope": q50_90(t_env),
            "opres_v2_total_p50_ms": float(np.median(np.array(t_fno) + np.array(t_setup) + np.array(t_v2) + np.array(t_env))),
            "opres_pcg2_total_p50_ms": float(np.median(np.array(t_fno) + np.array(t_setup) + np.array(t_pcg2) + np.array(t_env))),
            "exact_oracle_total_p50_ms": float(np.median(np.array(t_fno) + np.array(t_ex) + np.array(t_env))),
        }
    return res


def plot_lshape_figure(example, summary):
    """Generate publication-quality 2x4 figure for L-shaped Darcy + AMG lifting."""
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 8.5, "axes.titlesize": 9.0, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(2, 4, figsize=(11.2, 5.2), dpi=300)
    m = lshape_mask(64)

    def mask_nan(arr):
        out = arr.astype(float).copy()
        out[~m] = np.nan
        return out

    cmap_field = plt.cm.viridis.copy()
    cmap_field.set_bad("#e8e8e8")
    cmap_err = plt.cm.magma.copy()
    cmap_err.set_bad("#e8e8e8")

    panels = [
        (mask_nan(np.log(example["a"])), r"Log-permeability $\log a$ on $\Omega_L$", cmap_field, None),
        (mask_nan(example["truth"]), r"Reference pressure $u$ ($64^2$)", cmap_field, None),
        (mask_nan(np.abs(example["pred"] - example["truth"])), r"Pointwise error $|u - \hat{u}|$", cmap_err, None),
        (mask_nan(example["width_pcg2"]), r"OpRes-CP (AMG-PCG2) width", cmap_err, None),
    ]
    for ax, (img, title, cmap, vmax) in zip(axes[0], panels):
        im = ax.imshow(img.T, origin="lower", extent=[0, 1, 0, 1], cmap=cmap, vmax=vmax, rasterized=True)
        ax.plot([0.5, 0.5, 1.0], [1.0, 0.5, 0.5], color="#333333", lw=1.0)
        ax.scatter([0.5], [0.5], s=18, facecolors="none", edgecolors="#00d4ff", lw=1.2, zorder=5)
        ax.set_title(title)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yticks([0, 0.5, 1.0])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=6.8)

    # Bottom row: quantitative comparisons
    # (B1) ID matched 90% width
    ax = axes[1, 0]
    m_order = ["Uniform-CP", "Full-Field PI-CP", "Ensemble (mean center)", "Unsigned AMG-V2",
               "Jacobi-PCG5", "OpRes-CP (AMG-V2)", "OpRes-CP (AMG-PCG2)", "Exact Sparse Lift (oracle)"]
    labels = ["Uniform", "PI-CP", "5-Ens.", "Unsign V2", "Jac-PCG5", "AMG-V2", "AMG-PCG2", "Oracle"]
    vals = [summary["splits"]["id_64"][k]["matched_width"]["mean"] for k in m_order]
    sds = [summary["splits"]["id_64"][k]["matched_width"]["sd"] for k in m_order]
    colors = ["#888888", "#999999", "#d95f02", "#b2abd2", "#80b1d3", "#1b9e77", "#006837", "#444444"]
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, vals, xerr=sds, color=colors, height=0.68, capsize=2, edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Matched 90% full width")
    ax.set_title(r"ID $64^2$ sharpness (5 seeds)")

    # (B2) Paired 64^2 -> 128^2 transfer coverage
    ax = axes[1, 1]
    t_keys = ["Uniform-CP", "Ensemble (mean center)", "Jacobi-PCG5", "OpRes-CP (AMG-V2)", "OpRes-CP (AMG-PCG2)", "Exact Sparse Lift (oracle)"]
    t_labs = ["Uniform", "5-Ens.", "Jac-PCG5", "AMG-V2", "AMG-PCG2", "Oracle"]
    c64 = [100 * summary["splits"]["paired_64"][k]["coverage"]["mean"] for k in t_keys]
    c128 = [100 * summary["splits"]["paired_128"][k]["coverage"]["mean"] for k in t_keys]
    x_pos = np.arange(len(t_labs))
    ax.bar(x_pos - 0.19, c64, width=0.36, label=r"$64^2$", color="#6baed6")
    ax.bar(x_pos + 0.19, c128, width=0.36, label=r"$128^2$", color="#08519c")
    ax.axhline(90.0, ls="--", lw=0.9, color="#333333")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(t_labs, rotation=24, ha="right")
    ax.set_ylim(0, 108)
    ax.set_ylabel("Simultaneous cov. (%)")
    ax.set_title(r"Paired $64^2 \to 128^2$ transfer")
    ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=7.0, handlelength=1.1, columnspacing=0.8)

    # (B3) Contrast OOD coverage vs matched width
    ax = axes[1, 2]
    o_keys = ["Uniform-CP", "Full-Field PI-CP", "Ensemble (mean center)", "Jacobi-PCG5", "OpRes-CP (AMG-V2)", "OpRes-CP (AMG-PCG2)"]
    o_labs = ["Uniform", "PI-CP", "5-Ens.", "Jac-PCG5", "AMG-V2", "AMG-PCG2"]
    o_cols = ["#888888", "#999999", "#d95f02", "#80b1d3", "#1b9e77", "#006837"]
    offsets = {
        "AMG-PCG2": (0.018, 0.5),
        "AMG-V2": (0.018, -3.2),
        "Jac-PCG5": (0.018, -1.5),
        "5-Ens.": (0.018, 1.0),
        "Uniform": (0.018, 0.8),
        "PI-CP": (-0.085, 2.5),
    }
    for k, lab, col in zip(o_keys, o_labs, o_cols):
        w = summary["splits"]["ood_a_64"][k]["matched_width"]["mean"]
        c = 100 * summary["splits"]["ood_a_64"][k]["coverage"]["mean"]
        ax.scatter([w], [c], s=48, color=col, zorder=4)
        dx, dy = offsets[lab]
        ax.annotate(lab, (w, c), xytext=(w + dx, c + dy), fontsize=6.8)
    ax.axhline(90.0, ls="--", lw=0.9, color="#333333")
    ax.set_xlim(0.0, 0.48)
    ax.set_ylim(65.0, 101.0)
    ax.set_xlabel("OOD matched 90% width")
    ax.set_ylabel("OOD coverage (%)")
    ax.set_title("Contrast OOD (locked threshold)")

    # (B4) Single-field lift latency: AMG vs exact sparse oracle across 64^2 & 128^2
    ax = axes[1, 3]
    tm = summary["timing_single_field"]
    grids = [r"$64^2$ (3k)", r"$128^2$ (12k)"]
    xg = np.arange(2)
    v2_t = [tm[g]["amg_setup_plus_res"]["p50_ms"] + tm[g]["amg_v2_solve"]["p50_ms"] for g in ("64", "128")]
    pcg2_t = [tm[g]["amg_setup_plus_res"]["p50_ms"] + tm[g]["amg_pcg2_solve"]["p50_ms"] for g in ("64", "128")]
    ex_t = [tm[g]["exact_oracle_solve"]["p50_ms"] for g in ("64", "128")]
    ax.bar(xg - 0.25, v2_t, width=0.24, label="AMG-V2", color="#1b9e77")
    ax.bar(xg, pcg2_t, width=0.24, label="AMG-PCG2", color="#006837")
    ax.bar(xg + 0.25, ex_t, width=0.24, label="Exact spsolve", color="#555555")
    ax.set_xticks(xg)
    ax.set_xticklabels(grids)
    ax.set_ylabel("CPU lift P50 (ms)")
    ax.set_title(r"$O(M)$ AMG lift vs. exact solve")
    ax.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    for base in (FIG_OUT, PAPER_FIG):
        fig.savefig(base / "fig6_lshape_amg.pdf", bbox_inches="tight")
        fig.savefig(base / "fig6_lshape_amg.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"[P2-A] Running L-shape Darcy AMG 5-seed benchmark on {device}", flush=True)

    unit = verify_amg_unit_tests()
    per_seed = {}
    timing_res = None
    plot_example = None

    methods = [
        "Uniform-CP", "Choi CP-B", "Full-Field PI-CP",
        "Ensemble (same center)", "Ensemble (mean center)",
        "Unsigned AMG-V2", "Jacobi-PCG5",
        "OpRes-CP (AMG-V2)", "OpRes-CP (AMG-PCG2)", "Exact Sparse Lift (oracle)",
    ]

    for seed in SEEDS:
        t_seed = perf_counter()
        data_root = ensure_lshape_data(seed)
        models = []
        train_secs = []
        for idx, offset in enumerate(CFG["ensemble_init_offsets"]):
            ckpt = OUT / f"seed_{seed}" / f"fno_member_{idx}.pt"
            mdl, secs = train_lshape_member(seed, seed + offset, data_root, ckpt, device)
            models.append(mdl)
            train_secs.append(secs)
        single = models[0]
        print(f"[P2-A] seed={seed} trained 5 models in {sum(train_secs):.1f}s (single={train_secs[0]:.1f}s)", flush=True)

        # Load splits
        splits = {name: dict(np.load(data_root / f"{name}.npz"))
                  for name in ("val", "cal", "id_test", "paired_test", "ood_a")}

        # Val fit for CP-B, PI-CP, ensemble floor (strictly Val only!)
        val_members = np.stack([predict_lshape(m, splits["val"]["a"], device) for m in models])
        val_pred = val_members[0]
        m64 = lshape_mask(64)
        val_err = np.abs(val_pred - splits["val"]["u"])
        fixed_map = val_err.mean(axis=0, dtype=np.float64)
        fixed_map = np.where(m64, np.maximum(fixed_map, 0.05 * fixed_map[m64].mean()), 0.0)
        val_res = apply_pde(splits["val"]["a"].astype(np.float64), val_pred.astype(np.float64)) - 1.0
        pi_lam = float(1.0 / max(np.median(np.abs(val_res[:, m64])), 1e-12))
        ens_floor = 0.10 * float(np.sqrt(np.mean(val_members.std(axis=0, ddof=1, dtype=np.float64)[:, m64] ** 2)))

        # Calibrate thresholds on Cal ONLY
        cal_members = np.stack([predict_lshape(m, splits["cal"]["a"], device) for m in models])
        cal_scales, _, _ = compute_physics_scales(
            splits["cal"]["a"], cal_members[0], fixed_map, pi_lam, cal_members, ens_floor
        )
        cal_truth = splits["cal"]["u"].astype(np.float64)
        qs = {}
        for name in methods:
            center = cal_members.mean(axis=0) if name == "Ensemble (mean center)" else cal_members[0]
            err = np.abs(center - cal_truth)[:, m64]
            sc = np.maximum(cal_scales[name][:, m64], 1e-12)
            scores = (err / sc).max(axis=1)
            qs[name] = quantile(scores, CFG["alpha"])

        # Evaluate on ID 64^2, Paired 64^2, Paired 128^2, Contrast OOD 64^2
        eval_specs = [
            ("id_64", splits["id_test"]["a"], splits["id_test"]["u"], 64),
            ("paired_64", splits["paired_test"]["a"], splits["paired_test"]["u"], 64),
            ("paired_128", splits["paired_test"]["a_f"], splits["paired_test"]["u_f"], 128),
            ("ood_a_64", splits["ood_a"]["a"], splits["ood_a"]["u"], 64),
        ]
        seed_out = {"train_seconds_single": train_secs[0], "train_seconds_ens5": sum(train_secs),
                    "thresholds": qs, "splits": {}, "paired_transfer_diff": {}}
        raw_vecs = {}
        for label, a_arr, u_arr, n in eval_specs:
            mem = np.stack([predict_lshape(m, a_arr, device, batch=8 if n == 128 else 16) for m in models])
            scales, _, z_pcg2 = compute_physics_scales(a_arr, mem[0], fixed_map, pi_lam, mem, ens_floor)
            seed_out["splits"][label] = {}
            raw_vecs[label] = {}
            for name in methods:
                center = mem.mean(axis=0) if name == "Ensemble (mean center)" else mem[0]
                met = eval_field_metrics(center.astype(np.float64), u_arr.astype(np.float64), scales[name], qs[name], n)
                raw_vecs[label][name] = met
                seed_out["splits"][label][name] = {k: v for k, v in met.items()
                                                   if not isinstance(v, np.ndarray)}
            if seed == SEEDS[0] and label == "id_64":
                plot_example = {
                    "a": a_arr[0], "truth": u_arr[0], "pred": mem[0][0],
                    "width_pcg2": 2.0 * qs["OpRes-CP (AMG-PCG2)"] * scales["OpRes-CP (AMG-PCG2)"][0],
                    "z_pcg2": z_pcg2[0],
                }

        # Paired 64->128 drift per seed
        for name in methods:
            d_cov = raw_vecs["paired_128"][name]["cov_vec"].astype(float) - raw_vecs["paired_64"][name]["cov_vec"].astype(float)
            seed_out["paired_transfer_diff"][name] = {
                "cov_delta_pp": float(100.0 * d_cov.mean()),
                "width_ratio": float(seed_out["splits"]["paired_128"][name]["actual_width"] /
                                     seed_out["splits"]["paired_64"][name]["actual_width"]),
            }

        if seed == SEEDS[0]:
            timing_res = benchmark_timings(models, splits["id_test"], splits["paired_test"], device)

        per_seed[str(seed)] = seed_out
        id_op = seed_out["splits"]["id_64"]["OpRes-CP (AMG-PCG2)"]
        id_ens = seed_out["splits"]["id_64"]["Ensemble (mean center)"]
        p128_op = seed_out["splits"]["paired_128"]["OpRes-CP (AMG-PCG2)"]
        print(f"[P2-A] seed={seed} done in {perf_counter()-t_seed:.1f}s | "
              f"ID cov={100*id_op['coverage']:.1f}%, width={id_op['matched_width']:.5f} vs Ens {id_ens['matched_width']:.5f} "
              f"(-{100*(1-id_op['matched_width']/id_ens['matched_width']):.1f}%) | "
              f"128^2 cov={100*p128_op['coverage']:.1f}%", flush=True)

    # Aggregate 5 seeds
    agg_splits = {}
    for label in ("id_64", "paired_64", "paired_128", "ood_a_64"):
        agg_splits[label] = {}
        for name in methods:
            agg_splits[label][name] = {}
            for key in ("coverage", "actual_width", "matched_width", "point_cov",
                        "corner_sim_cov", "corner_pt_cov", "corner_width", "point_rel_l2"):
                vals = [per_seed[str(s)]["splits"][label][name][key] for s in SEEDS]
                _, avg, sd, lo, hi = stat(vals)
                agg_splits[label][name][key] = {"mean": avg, "sd": sd, "ci95": [lo, hi]}

    agg_transfer = {}
    for name in methods:
        dvals = [per_seed[str(s)]["paired_transfer_diff"][name]["cov_delta_pp"] for s in SEEDS]
        wvals = [per_seed[str(s)]["paired_transfer_diff"][name]["width_ratio"] for s in SEEDS]
        _, d_avg, d_sd, d_lo, d_hi = stat(dvals)
        _, w_avg, w_sd, w_lo, w_hi = stat(wvals)
        agg_transfer[name] = {
            "cov_delta_pp": {"mean": d_avg, "sd": d_sd, "ci95": [d_lo, d_hi]},
            "width_ratio": {"mean": w_avg, "sd": w_sd, "ci95": [w_lo, w_hi]},
        }

    # Width reductions vs Ensemble mean and Uniform
    shrink_vs_ens = [1.0 - per_seed[str(s)]["splits"]["id_64"]["OpRes-CP (AMG-PCG2)"]["matched_width"] /
                     per_seed[str(s)]["splits"]["id_64"]["Ensemble (mean center)"]["matched_width"] for s in SEEDS]
    shrink_v2_vs_ens = [1.0 - per_seed[str(s)]["splits"]["id_64"]["OpRes-CP (AMG-V2)"]["matched_width"] /
                        per_seed[str(s)]["splits"]["id_64"]["Ensemble (mean center)"]["matched_width"] for s in SEEDS]
    shrink_vs_unif = [1.0 - per_seed[str(s)]["splits"]["id_64"]["OpRes-CP (AMG-PCG2)"]["matched_width"] /
                      per_seed[str(s)]["splits"]["id_64"]["Uniform-CP"]["matched_width"] for s in SEEDS]

    summary = {
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "config": CFG,
        "unit_checks": unit,
        "splits": agg_splits,
        "paired_transfer": agg_transfer,
        "id_width_reduction": {
            "amg_pcg2_vs_ensemble_mean": dict(zip(("n", "mean", "sd", "lo", "hi"), stat(shrink_vs_ens))),
            "amg_v2_vs_ensemble_mean": dict(zip(("n", "mean", "sd", "lo", "hi"), stat(shrink_v2_vs_ens))),
            "amg_pcg2_vs_uniform": dict(zip(("n", "mean", "sd", "lo", "hi"), stat(shrink_vs_unif))),
        },
        "train_seconds": {
            "single_mean": float(np.mean([per_seed[str(s)]["train_seconds_single"] for s in SEEDS])),
            "ens5_mean": float(np.mean([per_seed[str(s)]["train_seconds_ens5"] for s in SEEDS])),
        },
        "timing_single_field": timing_res,
        "per_seed": per_seed,
    }
    (OUT / "lshape_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot_lshape_figure(plot_example, summary)
    print(f"[P2-A] COMPLETE! Summary saved to {OUT / 'lshape_summary.json'}", flush=True)
    print(f"[P2-A] ID AMG-PCG2 vs Ensemble shrinkage: {100*summary['id_width_reduction']['amg_pcg2_vs_ensemble_mean']['mean']:.2f}% "
          f"(AMG-V2: {100*summary['id_width_reduction']['amg_v2_vs_ensemble_mean']['mean']:.2f}%)", flush=True)


if __name__ == "__main__":
    main()
