"""Revision v4 Task 2 (P0):
  Part A: Exact 3-term nonlinear remainder decomposition on Burgers (FNO & U-Net 64/256)
          and Allen-Cahn (FNO T=1/T=2 & U-Net T=2 64^2/128^2).
  Part B: Five-seed temporal discretization convergence:
          - Burgers K_sub in {1, 2, 4, 8, 16};
          - Allen-Cahn Strang internal substeps K_str in {1, 2, 4} and frame step dt in {0.05, 0.10, 0.20}.
  Part C: SPD-preconditioned PCG-2 (SPD-AMG-PCG2) theoretical verification via Power Iteration:
          exact condition number kappa(M_h A_h), Chebyshev energy contraction bound delta_PCG^(2),
          100% empirical compliance check, and exact degree-1 positive homogeneity (< 1e-13).
  Part D: Extended Jha-style (CMAME 2024) variational point-corrector comparisons on Darcy 64^2 and Burgers 64.

Strictly zero backbone retraining; formal seeds [42, 137, 314, 271, 811].
Outputs written to results/phase3_revision_v4/task2_nonlinear_pcg_jha.json.
"""

from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter, perf_counter_ns

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import allen_cahn_phase2_data as ac_data
import allen_cahn_phase2_formal as ac_formal
import burgers_phase1_ops as burg_ops
import burgers_phase2 as burg_p2
from darcy_data import apply_operator as darcy_apply_op, solve as darcy_solve
from darcy_day15 import arrays, value_at_coverage
from darcy_day15_ops import lift as darcy_lift, mixture as darcy_mix, physical_scale as darcy_phys_scale
from darcy_phase0_eval import quantile
from phase2d import batches, take
from phase2d_data import Protocol
from phase2d_statistics import relative_l2
from phase3_allen_cahn import jacobian as ac_jacobian, midpoints as ac_midpoints, strang_cpu as ac_strang_cpu
from phase3_common import finite_json, mix as p3_mix
import phase3_p2_lshape_amg as lsh
from phase3_summarize import DOWNLOADS, SEEDS, STAGE, stat
from rev2_task1_fidelity import build_rect_amg_hierarchy, compute_signed_recovery
from rev2_task5_ac_amg import ensure_ac_t2_split, strang_and_scale_gpu_fp64
from rev3_task1_opnorm_gamma import _power_iter_energy_defect


OUT = Path("results/phase3_revision_v4")
V4_CACHE = OUT / "cache"
V3_CACHE = Path("results/phase3_revision_v3/cache")
REV_CACHE = Path("results/phase3_revision/cache")
LSHAPE_ROOT = Path("results/phase3_p2/lshape_darcy")
FROZEN_ROOT = STAGE / "phase2d_frozen"


def _seed_stats(values):
    _, avg, sd, lo, hi = stat(values)
    return {
        "mean": float(avg),
        "sd": float(sd),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "per_seed": {str(s): float(v) for s, v in zip(SEEDS, values)},
    }


# ---------- Part A: Exact 3-Term Nonlinear Decomposition ----------

def _dealiased_prod_deriv(a, b):
    """Compute d_x(a * b) using exact 3n/2 dealiasing matching burgers_phase1_ops.divergence_of_square."""
    n = a.shape[-1]
    fa = np.fft.rfft(a, axis=-1, norm="forward")
    fb = np.fft.rfft(b, axis=-1, norm="forward")
    fa[..., -1] = 0.0
    fb[..., -1] = 0.0
    pad_n = (3 * n) // 2
    pa = np.fft.irfft(fa, n=pad_n, axis=-1, norm="forward")
    pb = np.fft.irfft(fb, n=pad_n, axis=-1, norm="forward")
    flux = np.fft.rfft(pa * pb, axis=-1, norm="forward")[..., :n // 2 + 1]
    flux[..., -1] = 0.0
    return np.fft.irfft(1j * burg_ops.modes(n) * flux, n=n, axis=-1, norm="forward")


def _burgers_3term_decompose(full_hat, full_true, nu, cfg):
    """Verify exact identity R_j(u_hat) = L_j(e) + N_j(e) + tau_j and lift each piece."""
    dt = float(cfg["frame_dt"])
    n = full_hat.shape[-1]
    e_full = full_hat - full_true
    e_mid = 0.5 * (e_full[:, 1:] + e_full[:, :-1])
    uhat_mid = 0.5 * (full_hat[:, 1:] + full_hat[:, :-1])

    r_total = burg_ops.trajectory_residual(full_hat, nu, cfg)
    tau = burg_ops.trajectory_residual(full_true, nu, cfg)
    lap_e = np.fft.irfft(-(burg_ops.modes(n) ** 2) * np.fft.rfft(e_mid, axis=-1), n=n, axis=-1)
    l_e = (e_full[:, 1:] - e_full[:, :-1]) / dt + _dealiased_prod_deriv(uhat_mid, e_mid) - nu[:, None, None] * lap_e
    n_e = -0.5 * _dealiased_prod_deriv(e_mid, e_mid)

    decomp_err = float(np.max(np.abs(r_total - (l_e + n_e + tau))))
    assert decomp_err < 1e-11, f"Burgers 3-term identity violation: {decomp_err:.3e}"

    z_tot = burg_ops.lift("adv_diff", full_hat, r_total, nu, cfg)
    z_le = burg_ops.lift("adv_diff", full_hat, l_e, nu, cfg)
    z_ne = burg_ops.lift("adv_diff", full_hat, n_e, nu, cfg)
    z_tau = burg_ops.lift("adv_diff", full_hat, tau, nu, cfg)

    lin_err = float(np.max(np.abs(z_tot - (z_le + z_ne + z_tau))))
    assert lin_err < 1e-11, f"Burgers lift linearity violation: {lin_err:.3e}"

    e_eval = e_full[:, 1:]
    nf = len(e_eval)
    norm_e = np.maximum(np.linalg.norm(e_eval.reshape(nf, -1), axis=1), 1e-30)
    norm_zne = np.linalg.norm(z_ne.reshape(nf, -1), axis=1)
    norm_ztau = np.linalg.norm(z_tau.reshape(nf, -1), axis=1)
    norm_lin = np.linalg.norm((z_le - e_eval).reshape(nf, -1), axis=1)
    norm_tot = np.linalg.norm((z_tot - e_eval).reshape(nf, -1), axis=1)

    # Fit log-log slope of ||B_h N_h||_2 vs ||e_h||_2 (expect quadratic ~ 2.0)
    mask = (norm_e > 1e-12) & (norm_zne > 1e-15)
    slope = float(np.polyfit(np.log(norm_e[mask]), np.log(norm_zne[mask]), 1)[0]) if mask.sum() >= 10 else 0.0

    return {
        "decomp_max_err": decomp_err,
        "lin_defect_ratio": float(np.mean(norm_lin / norm_e)),
        "nonlinear_ratio": float(np.mean(norm_zne / norm_e)),
        "tau_ratio": float(np.mean(norm_ztau / norm_e)),
        "total_rel_err": float(np.mean(norm_tot / norm_e)),
        "nonlinear_loglog_slope": slope,
        "point_rel_l2": float(relative_l2(full_hat[:, 1:], full_true[:, 1:]).mean()),
    }


@torch.inference_mode()
def _solve_ac_gpu_grid(coeff_np, nu_np, rho_np, cfg, frames, target_grid, device):
    """Exact GPU FP32 Allen-Cahn ETDRK2 solver matching allen_cahn_phase2_data.solve for grid=64 or 128."""
    n = cfg["reference"]  # 128
    k2_np, mask_np = ac_data.spectrum(n)
    k2 = torch.as_tensor(k2_np, dtype=torch.float32, device=device)
    mask = torch.as_tensor(mask_np, dtype=torch.bool, device=device)
    nu = torch.as_tensor(nu_np, dtype=torch.float32, device=device)
    rho = torch.as_tensor(rho_np, dtype=torch.float32, device=device)

    dt = float(cfg["dt"] / cfg["substeps"])
    expo64 = -dt * nu_np[:, None, None].astype(np.float64) * k2_np[None].astype(np.float64)
    ex_np = np.exp(expo64)
    nz = expo64 != 0.0
    p1_np = np.divide(np.expm1(expo64), expo64, out=np.ones_like(expo64), where=nz)
    p2_np = np.divide(p1_np - 1.0, expo64, out=np.full_like(expo64, 0.5), where=nz)
    ex = torch.as_tensor(ex_np, dtype=torch.float32, device=device)
    p1 = torch.as_tensor(dt * p1_np, dtype=torch.float32, device=device)
    p2 = torch.as_tensor(dt * p2_np, dtype=torch.float32, device=device)

    coeff = torch.as_tensor(coeff_np, dtype=torch.complex64, device=device)
    z = torch.zeros((len(nu_np), n, n // 2 + 1), dtype=torch.complex64, device=device)
    z[:, :8, :9] = coeff[:, :8]
    z[:, -8:, :9] = coeff[:, 8:]

    def _proj(zv):
        if target_grid == n:
            return torch.fft.irfft2(zv, s=(n, n), norm="forward").cpu().numpy()
        half = target_grid // 2
        low = torch.cat((zv[:, :half, :half + 1], zv[:, -half:, :half + 1]), dim=-2).clone()
        low[:, half, :] = 0.0
        low[..., -1] = 0.0
        return torch.fft.irfft2(low, s=(target_grid, target_grid), norm="forward").cpu().numpy()

    def _react(zv):
        u = torch.fft.irfft2(zv, s=(n, n), norm="forward")
        return rho[:, None, None] * mask[None] * torch.fft.rfft2(u - u * u * u, norm="forward")

    out = [_proj(z)]
    for _ in range(frames):
        for __ in range(cfg["substeps"]):
            f = _react(z)
            stage = ex * z + p1 * f
            z = stage + p2 * (_react(stage) - f)
        out.append(_proj(z))
    return np.stack(out, axis=1)


def _ensure_ac_t2_paired128(seed, cfg, device, count=100):
    root = V4_CACHE / "ac_t2_128" / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "paired_t2.npz"
    if path.exists():
        return arrays(path)
    from phase2d_ac import STREAMS as AC_T2_STREAMS
    c = cfg | {"streams": AC_T2_STREAMS, "seed": seed, "counts": cfg["counts"] | {"paired": count}}
    nu, rho, coeff = ac_data.initial(c, "paired")
    if seed == SEEDS[0]:
        ref_cpu = ac_data.solve(coeff[:2], nu[:2], rho[:2], c, [128], 40)[128]
        ref_gpu = _solve_ac_gpu_grid(coeff[:2], nu[:2], rho[:2], c, 40, 128, device)
        md = float(np.max(np.abs(ref_cpu - ref_gpu)))
        assert md < 5e-5, f"AC 128 GPU FP32 mismatch: {md:.3e}"
    parts64, parts128 = [], []
    for i in range(0, len(nu), 25):
        sl = slice(i, i + 25)
        parts64.append(_solve_ac_gpu_grid(coeff[sl], nu[sl], rho[sl], c, 40, 64, device))
        parts128.append(_solve_ac_gpu_grid(coeff[sl], nu[sl], rho[sl], c, 40, 128, device))
    u64 = np.concatenate(parts64, axis=0).astype(np.float32)
    u128 = np.concatenate(parts128, axis=0).astype(np.float32)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, nu=nu, rho=rho, u64=u64, u128=u128)
    tmp.replace(path)
    return arrays(path)


@torch.inference_mode()
def _ac_4term_decompose(full_hat, full_true, nu_np, rho_np, cfg, device):
    """Exact 4-term decomposition:
       R_j(u_hat) = L_j^guard(e) + G_j(e) + N_j(e) + tau_j
    and Strang lifting on GPU FP32 (linear in forcing with fixed guarded Jacobian).
    Check identity in float64 CPU to < 1e-11 first.
    """
    dt = float(cfg["dt"])
    n = full_hat.shape[-1]
    k2_np, mask_np = ac_data.spectrum(n)
    k2 = torch.as_tensor(k2_np, dtype=torch.float64, device=device)
    mask = torch.as_tensor(mask_np, dtype=torch.bool, device=device)

    e_full = full_hat - full_true
    mid_h = 0.5 * (full_hat[:, 1:] + full_hat[:, :-1])
    mid_u = 0.5 * (full_true[:, 1:] + full_true[:, :-1])
    mid_e = 0.5 * (e_full[:, 1:] + e_full[:, :-1])

    def _de(arr_np):
        t = torch.as_tensor(arr_np, dtype=torch.float64, device=device)
        return torch.fft.irfft2(mask[None, None] * torch.fft.rfft2(t), s=(n, n))

    def _lap(arr_np):
        t = torch.as_tensor(arr_np, dtype=torch.float64, device=device)
        return torch.fft.irfft2((-k2[None, None]) * torch.fft.rfft2(t), s=(n, n))

    nu = torch.as_tensor(nu_np, dtype=torch.float64, device=device)[:, None, None, None]
    rho = torch.as_tensor(rho_np, dtype=torch.float64, device=device)[:, None, None, None]

    # Compute exact terms in float64 on GPU (or chunked)
    mh_t = torch.as_tensor(mid_h, dtype=torch.float64, device=device)
    mu_t = torch.as_tensor(mid_u, dtype=torch.float64, device=device)
    me_t = torch.as_tensor(mid_e, dtype=torch.float64, device=device)

    dh = torch.as_tensor((full_hat[:, 1:] - full_hat[:, :-1]) / dt, dtype=torch.float64, device=device)
    du = torch.as_tensor((full_true[:, 1:] - full_true[:, :-1]) / dt, dtype=torch.float64, device=device)
    de = dh - du

    r_tot = dh - nu * _lap(mid_h) - rho * _de(mid_h - mid_h ** 3)
    tau = du - nu * _lap(mid_u) - rho * _de(mid_u - mid_u ** 3)

    jac_exact = rho * (1.0 - 3.0 * mh_t * mh_t)
    jac_guard = jac_exact.clamp(-float(cfg["reaction_clip"]), float(cfg["reaction_clip"]))
    jac_guard = (jac_guard - float(cfg["positive_damping"]) * jac_guard.clamp_min(0.0)).clamp_max(float(cfg["max_growth"]))

    l_guard = de - nu * _lap(mid_e) - _de_tensor(mask, jac_guard * me_t, n)
    g_e = _de_tensor(mask, (jac_guard - jac_exact) * me_t, n)
    n_e = -rho * _de_tensor(mask, 3.0 * mh_t * me_t * me_t - me_t * me_t * me_t, n)

    decomp_err = float((r_tot - (l_guard + g_e + n_e + tau)).abs().max().item())
    assert decomp_err < 1e-11, f"Allen-Cahn 4-term identity violation: {decomp_err:.3e}"

    def _strang_lift(forcing_t):
        diff = -nu[:, 0] * k2[None]
        half = torch.exp(0.5 * dt * (diff - float(cfg["gamma"])))
        state = torch.zeros_like(forcing_t[:, 0])
        zs = []
        for j in range(forcing_t.shape[1]):
            f = 0.5 * dt * forcing_t[:, j]
            state = torch.fft.irfft2(half * torch.fft.rfft2(state + f), s=(n, n))
            state = state * torch.exp(dt * jac_guard[:, j])
            state = torch.fft.irfft2(half * torch.fft.rfft2(state), s=(n, n)) + f
            zs.append(state)
        return torch.stack(zs, dim=1)

    z_tot = _strang_lift(r_tot)
    z_lg = _strang_lift(l_guard)
    z_ge = _strang_lift(g_e)
    z_ne = _strang_lift(n_e)
    z_tau = _strang_lift(tau)

    e_eval = torch.as_tensor(e_full[:, 1:], dtype=torch.float64, device=device)
    nb = len(e_eval)
    ne_norm = e_eval.reshape(nb, -1).norm(dim=1).clamp_min(1e-30)
    nz_ne = z_ne.reshape(nb, -1).norm(dim=1)
    nz_ge = z_ge.reshape(nb, -1).norm(dim=1)
    nz_tau = z_tau.reshape(nb, -1).norm(dim=1)
    n_lin = (z_lg - e_eval).reshape(nb, -1).norm(dim=1)
    n_tot = (z_tot - e_eval).reshape(nb, -1).norm(dim=1)

    m_np = ne_norm.cpu().numpy()
    nz_np = nz_ne.cpu().numpy()
    ok = (m_np > 1e-12) & (nz_np > 1e-15)
    slope = float(np.polyfit(np.log(m_np[ok]), np.log(nz_np[ok]), 1)[0]) if ok.sum() >= 10 else 0.0

    return {
        "decomp_max_err": decomp_err,
        "lin_guard_defect_ratio": float((n_lin / ne_norm).mean().item()),
        "guard_bias_ratio": float((nz_ge / ne_norm).mean().item()),
        "nonlinear_ratio": float((nz_ne / ne_norm).mean().item()),
        "tau_ratio": float((nz_tau / ne_norm).mean().item()),
        "total_rel_err": float((n_tot / ne_norm).mean().item()),
        "nonlinear_loglog_slope": slope,
        "point_rel_l2": float(relative_l2(full_hat[:, 1:], full_true[:, 1:]).mean()),
    }


def _de_tensor(mask, tensor, n):
    return torch.fft.irfft2(mask[None, None] * torch.fft.rfft2(tensor), s=(n, n))


def run_part_a_nonlinear(device):
    print("[Task 2A] Exact nonlinear remainder decomposition (Burgers FNO/U-Net & AC FNO/U-Net)...", flush=True)
    burg_splits = ("id", "ood", "ood_high", "unet_64", "unet_256")
    burg_ps = {s: {} for s in map(str, SEEDS)}
    for seed in SEEDS:
        root = REV_CACHE / "burgers" / f"seed_{seed}"
        v3_root = V3_CACHE / "burgers" / f"seed_{seed}"
        p = Protocol("burgers", seed, 64, root, FROZEN_ROOT / f"seed_{seed}", device)
        fno = p.open_frozen()
        cfg = p.config

        # Load frozen 2b U-Net checkpoint
        unet_ck = DOWNLOADS / f"2b/seed_{seed}/results/phase2_burgers/seed_{seed}/U-Net/fno.pt"
        unet = burg_p2.make_model("U-Net", cfg).to(device)
        unet.load_state_dict(torch.load(unet_ck, map_location=device, weights_only=True)["model"])
        unet.eval()

        for sp_key, fname, mdl, grid, res_k in (
            ("id", root / "data/id.npz", fno, 64, "id"),
            ("ood", root / "data/ood.npz", fno, 64, "ood"),
            ("ood_high", root / "data/ood_high.npz", fno, 64, "ood_high"),
            ("paired", v3_root / "data/paired.npz", unet, 64, "unet_64"),
            ("paired", v3_root / "data/paired.npz", unet, 256, "unet_256"),
        ):
            d = arrays(fname)
            inp, truth = p.split(d, grid, p.frames)
            preds = [p.predict(mdl, take(inp, sl), grid, p.frames) for sl in batches(len(truth), 64)]
            pred = np.concatenate(preds, axis=0).astype(np.float64)
            full_hat = np.concatenate((inp[0][:, None].astype(np.float64), pred), axis=1)
            full_true = d[f"u{grid}"].astype(np.float64)
            burg_ps[str(seed)][res_k] = _burgers_3term_decompose(full_hat, full_true, inp[1].astype(np.float64), cfg)

    burg_sum = {
        sp: {k: _seed_stats([burg_ps[str(s)][sp][k] for s in SEEDS]) for k in burg_ps[str(SEEDS[0])][sp]}
        for sp in burg_splits
    }
    for sp in burg_splits:
        s = burg_sum[sp]
        print(f"  Burgers {sp:9s}: nl={s['nonlinear_ratio']['mean']:.3f}, tau={s['tau_ratio']['mean']:.3f}, "
              f"lin={s['lin_defect_ratio']['mean']:.3f}, slope={s['nonlinear_loglog_slope']['mean']:.2f}, ptL2={100*s['point_rel_l2']['mean']:.2f}%", flush=True)

    # Allen-Cahn FNO (T=1 ID, T=1 OOD, T=2 ID) & U-Net (T=2 64^2, T=2 128^2)
    ac_splits = ("fno_t1_id", "fno_t1_ood", "fno_t2_id", "unet_t2_64", "unet_t2_128")
    ac_ps = {s: {} for s in map(str, SEEDS)}
    for seed in SEEDS:
        root = REV_CACHE / "allen_cahn" / f"seed_{seed}"
        p = Protocol("allen_cahn", seed, 64, root, FROZEN_ROOT / f"seed_{seed}", device)
        fno = p.open_frozen(unet=False)
        unet = p.open_frozen(unet=True)
        cfg = p.config

        d_t1_id = arrays(root / "data/id.npz")
        d_t1_ood = arrays(root / "data/ood.npz")
        d_t2_id = ensure_ac_t2_split(seed, "id", cfg, device)
        d_t2_128 = _ensure_ac_t2_paired128(seed, cfg, device, count=100)

        for sp_k, data, mdl, grid, frames, max_n in (
            ("fno_t1_id", d_t1_id, fno, 64, 20, 200),
            ("fno_t1_ood", d_t1_ood, fno, 64, 20, 200),
            ("fno_t2_id", d_t2_id, fno, 64, 40, 200),
            ("unet_t2_64", d_t2_128, unet, 64, 40, 100),
            ("unet_t2_128", d_t2_128, unet, 128, 40, 100),
        ):
            inp, truth = p.split(data, grid, frames)
            inp_sub = take(inp, slice(0, max_n))
            pred = np.concatenate([p.predict(mdl, take(inp_sub, sl), grid, frames) for sl in batches(max_n, 16)], axis=0).astype(np.float64)
            full_hat = np.concatenate((inp_sub[0][:, None].astype(np.float64), pred), axis=1)
            full_true = data[f"u{grid}"][:max_n, :frames + 1].astype(np.float64)
            # Chunk in batches of 50 to fit GPU float64 nicely
            parts = []
            for sl in batches(max_n, 50):
                parts.append(_ac_4term_decompose(full_hat[sl], full_true[sl], inp_sub[1][sl].astype(np.float64),
                                                 inp_sub[2][sl].astype(np.float64), cfg, device))
            ac_ps[str(seed)][sp_k] = {
                k: float(np.mean([pt[k] for pt in parts])) if k != "decomp_max_err" else float(max(pt[k] for pt in parts))
                for k in parts[0]
            }

    ac_sum = {
        sp: {k: _seed_stats([ac_ps[str(s)][sp][k] for s in SEEDS]) for k in ac_ps[str(SEEDS[0])][sp]}
        for sp in ac_splits
    }
    for sp in ac_splits:
        s = ac_sum[sp]
        print(f"  AC {sp:12s}: nl={s['nonlinear_ratio']['mean']:.3f}, guard={s['guard_bias_ratio']['mean']:.3f}, "
              f"tau={s['tau_ratio']['mean']:.3f}, linG={s['lin_guard_defect_ratio']['mean']:.3f}, "
              f"slope={s['nonlinear_loglog_slope']['mean']:.2f}, ptL2={100*s['point_rel_l2']['mean']:.2f}%", flush=True)

    return {"burgers": {"summary": burg_sum, "per_seed": burg_ps}, "allen_cahn": {"summary": ac_sum, "per_seed": ac_ps}}


# ---------- Part B: Five-Seed Temporal Discretization Convergence ----------

def _clamped_face_departures(velocity, dt):
    """Same as burg_ops.face_departures, also tracking if trajectory folded under large dt."""
    n = velocity.shape[-1]
    faces = (velocity + np.roll(velocity, 1, axis=-1)) * 0.5
    departure = np.arange(n)[None] - dt * faces * n / (2.0 * math.pi)
    jacobian = np.roll(departure, -1, axis=-1) - departure
    jacobian[:, -1] += n
    folded = np.any(jacobian <= 0.0, axis=1)
    cell = np.floor(departure).astype(int)
    return cell % n, cell // n, departure - cell, folded


def _burgers_lift_safe(full, residual, nu, cfg, substeps):
    n = residual.shape[-1]
    dt = float(cfg["frame_dt"]) / substeps
    exponent = -dt * (nu[:, None] * burg_ops.modes(n) ** 2 + float(cfg["gamma"]))
    exp_v = np.exp(exponent)
    force_coef = dt * burg_ops.phi(exponent)[0]
    forcing = np.fft.rfft(residual, axis=-1) * force_coef[:, None]
    z = np.zeros_like(residual[:, 0])
    states = []
    ever_folded = np.zeros(len(residual), dtype=bool)
    for frame in range(residual.shape[1]):
        mid = 0.5 * (full[:, frame] + full[:, frame + 1])
        c, per, fr, fmask = _clamped_face_departures(mid, dt)
        ever_folded |= fmask
        for _ in range(substeps):
            adv = burg_ops.transport(z, (c, per, fr))
            z = np.fft.irfft(exp_v * np.fft.rfft(adv) + forcing[:, frame], n=n)
        states.append(z)
    return np.stack(states, axis=1), ever_folded


def run_part_b_time_convergence(device):
    print("[Task 2B] Five-seed temporal convergence (Burgers K_sub & Allen-Cahn Strang K_str / dt)...", flush=True)
    # 1) Burgers K_sub in {1, 2, 4, 8, 16}
    k_subs = (1, 2, 4, 8, 16)
    burg_ps = {}
    for seed in SEEDS:
        root = REV_CACHE / "burgers" / f"seed_{seed}"
        p = Protocol("burgers", seed, 64, root, FROZEN_ROOT / f"seed_{seed}", device)
        model = p.open_frozen()
        cfg = p.config
        cal_d = arrays(root / "data/cal.npz")
        id_d = arrays(root / "data/id.npz")

        def _prep(d):
            inp, y = p.split(d, 64, p.frames)
            pred = np.concatenate([p.predict(model, take(inp, sl), 64, p.frames) for sl in batches(len(y), 64)], axis=0).astype(np.float64)
            full = np.concatenate((inp[0][:, None].astype(np.float64), pred), axis=1)
            nu = inp[1].astype(np.float64)
            r = burg_ops.trajectory_residual(full, nu, cfg)
            return full, nu, pred, y.astype(np.float64), r

        fc, nuc, pc, yc, rc = _prep(cal_d)
        fi, nui, pi, yi, ri = _prep(id_d)
        ei = pi - yi
        ec = pc - yc
        seed_out = {}
        for ks in k_subs:
            zc, fold_c = _burgers_lift_safe(fc, rc, nuc, cfg, ks)
            zi, fold_i = _burgers_lift_safe(fi, ri, nui, cfg, ks)
            # Timing per trajectory (single trajectory P50 ms)
            tms = []
            for rep in range(25):
                t0 = perf_counter_ns()
                _ = _burgers_lift_safe(fi[rep:rep + 1], ri[rep:rep + 1], nui[rep:rep + 1], cfg, ks)
                if rep >= 5:
                    tms.append((perf_counter_ns() - t0) / 1e6)

            non_fold = ~fold_i
            sc_c = np.maximum(p3_mix(burg_ops.envelope(zc, cfg), 0.85), 1e-12)
            sc_i = np.maximum(p3_mix(burg_ops.envelope(zi, cfg), 0.85), 1e-12)
            q90 = quantile((np.abs(ec).reshape(len(ec), -1) / sc_c.reshape(len(ec), -1)).max(axis=1), 0.10)
            scores = (np.abs(ei).reshape(len(ei), -1) / sc_i.reshape(len(ei), -1)).max(axis=1)
            q_emp = value_at_coverage(scores, 0.90)

            rec = compute_signed_recovery(zi[non_fold].reshape(non_fold.sum(), -1), ei[non_fold].reshape(non_fold.sum(), -1))
            seed_out[str(ks)] = {
                "non_folded_rate": float(non_fold.mean()),
                "cos": rec["cos_mean"],
                "rel_err": rec["rel_l2_err_mean"],
                "coverage": float((scores <= q90).mean()),
                "matched90_width": float(2.0 * q_emp * sc_i.mean()),
                "lift_p50_ms": float(np.median(tms)),
            }
        burg_ps[str(seed)] = seed_out

    burg_sum = {
        str(ks): {mk: _seed_stats([burg_ps[str(s)][str(ks)][mk] for s in SEEDS]) for mk in burg_ps[str(SEEDS[0])][str(ks)]}
        for ks in k_subs
    }
    for ks in k_subs:
        s = burg_sum[str(ks)]
        print(f"  Burgers K_sub={ks:2d}: nonFold={100*s['non_folded_rate']['mean']:.1f}%, "
              f"cos={s['cos']['mean']:.4f}, relErr={s['rel_err']['mean']:.4f}, "
              f"cov={100*s['coverage']['mean']:.2f}%, m90w={s['matched90_width']['mean']:.4f}, ms={s['lift_p50_ms']['mean']:.2f}", flush=True)

    # 2) Allen-Cahn Strang K_str in {1, 2, 4} & dt in {0.05, 0.10, 0.20}
    k_strs = (1, 2, 4)
    strides = (1, 2, 4)  # dt = 0.05, 0.10, 0.20
    ac_kstr_ps = {}
    ac_dt_ps = {}
    for seed in SEEDS:
        root = REV_CACHE / "allen_cahn" / f"seed_{seed}"
        p = Protocol("allen_cahn", seed, 64, root, FROZEN_ROOT / f"seed_{seed}", device)
        model = p.open_frozen()
        cfg = p.config
        d_t1_cal = arrays(root / "data/cal.npz")
        d_t1_id = arrays(root / "data/id.npz")
        d_t2_cal = ensure_ac_t2_split(seed, "cal", cfg, device)
        d_t2_id = ensure_ac_t2_split(seed, "id", cfg, device)

        def _prep_ac(d, frames):
            inp, y = p.split(d, 64, frames)
            pred = np.concatenate([p.predict(model, take(inp, sl), 64, frames) for sl in batches(len(y), 16)], axis=0).astype(np.float64)
            full_hat = np.concatenate((inp[0][:, None].astype(np.float64), pred), axis=1)
            full_true = d["u64"][:, :frames + 1].astype(np.float64)
            return full_hat, full_true, inp[1].astype(np.float64), inp[2].astype(np.float64)

        @torch.inference_mode()
        def _strang_kstr(full_np, nu_np, rho_np, cfg_u, kstr):
            dt_f = float(cfg_u["dt"])
            dt_s = dt_f / kstr
            n = full_np.shape[-1]
            k2_np, mask_np = ac_data.spectrum(n)
            k2 = torch.as_tensor(k2_np, dtype=torch.float32, device=device)
            mask = torch.as_tensor(mask_np, dtype=torch.bool, device=device)
            full = torch.as_tensor(full_np, dtype=torch.float32, device=device)
            nu = torch.as_tensor(nu_np, dtype=torch.float32, device=device)
            rho = torch.as_tensor(rho_np, dtype=torch.float32, device=device)
            diff = -nu[:, None, None] * k2[None]
            half = torch.exp(0.5 * dt_s * (diff - float(cfg_u["gamma"])))
            state = torch.zeros_like(full[:, 0])
            z_list = []
            for j in range(1, full.shape[1]):
                left, right = full[:, j - 1], full[:, j]
                mid = 0.5 * (left + right)
                nonlin = torch.fft.irfft2(mask[None] * torch.fft.rfft2(mid - mid * mid * mid), s=(n, n))
                r = (right - left) / dt_f - torch.fft.irfft2(diff * torch.fft.rfft2(mid), s=(n, n)) - rho[:, None, None] * nonlin
                jac = (rho[:, None, None] * (1.0 - 3.0 * mid * mid)).clamp(-cfg_u["reaction_clip"], cfg_u["reaction_clip"])
                jac = (jac - cfg_u["positive_damping"] * jac.clamp_min(0.0)).clamp_max(cfg_u["max_growth"])
                f_sub = 0.5 * dt_s * r
                exp_j = torch.exp(dt_s * jac)
                for _ in range(kstr):
                    state = torch.fft.irfft2(half * torch.fft.rfft2(state + f_sub), s=(n, n))
                    state = state * exp_j
                    state = torch.fft.irfft2(half * torch.fft.rfft2(state), s=(n, n)) + f_sub
                z_list.append(state)
            z = torch.stack(z_list, dim=1)
            # GPU scale
            sigma = float(cfg_u["spatial_radius"]) * n / (2.0 * math.pi)
            rad = int(4.0 * sigma + 0.5)
            grid = torch.arange(-rad, rad + 1, dtype=torch.float32, device=device)
            kern = torch.exp(-0.5 * (grid / sigma) ** 2)
            kern = kern / kern.sum()
            sq = z * z
            avg = sq.mean(dim=(1, 2, 3), keepdim=True)
            b, t, h, w = sq.shape
            x = torch.nn.functional.pad(sq.reshape(b * t, 1, h, w), (rad, rad, 0, 0), mode="circular")
            x = torch.nn.functional.conv2d(x, kern.view(1, 1, 1, -1))
            x = torch.nn.functional.pad(x, (0, 0, rad, rad), mode="circular")
            x = torch.nn.functional.conv2d(x, kern.view(1, 1, -1, 1))
            smooth = x.reshape(b, t, h, w) + 0.01 * avg
            lam = float(cfg_u["lambda"])
            sig = torch.sqrt((1.0 - lam) * 1.01 * avg + lam * smooth).clamp_min(1e-12)
            return z.cpu().numpy().astype(np.float64), sig.cpu().numpy().astype(np.float64)

        # Evaluate K_str on T=1 & T=2
        ac_kstr_ps[str(seed)] = {}
        for h_lbl, d_c, d_i, fr in (("T1", d_t1_cal, d_t1_id, 20), ("T2", d_t2_cal, d_t2_id, 40)):
            fc, tc, nuc, rhoc = _prep_ac(d_c, fr)
            fi, ti, nui, rhoi = _prep_ac(d_i, fr)
            ei = (fi[:, 1:] - ti[:, 1:])
            ec = (fc[:, 1:] - tc[:, 1:])
            ac_kstr_ps[str(seed)][h_lbl] = {}
            for ks in k_strs:
                zc, sc_c = _strang_kstr(fc[:200], nuc[:200], rhoc[:200], cfg, ks)
                zi, sc_i = _strang_kstr(fi[:200], nui[:200], rhoi[:200], cfg, ks)
                q90 = quantile((np.abs(ec[:200]).reshape(200, -1) / sc_c.reshape(200, -1)).max(axis=1), 0.10)
                scores = (np.abs(ei[:200]).reshape(200, -1) / sc_i.reshape(200, -1)).max(axis=1)
                q_emp = value_at_coverage(scores, 0.90)
                rec = compute_signed_recovery(zi.reshape(200, -1), ei[:200].reshape(200, -1))
                ac_kstr_ps[str(seed)][h_lbl][str(ks)] = {
                    "cos": rec["cos_mean"],
                    "rel_err": rec["rel_l2_err_mean"],
                    "coverage": float((scores <= q90).mean()),
                    "matched90_width": float(2.0 * q_emp * sc_i.mean()),
                }

        # Evaluate dt in {0.05, 0.10, 0.20} on T=1 & T=2 (both FNO and true-tau error)
        ac_dt_ps[str(seed)] = {}
        for h_lbl, d_i, fr in (("T1", d_t1_id, 20), ("T2", d_t2_id, 40)):
            fi, ti, nui, rhoi = _prep_ac(d_i, fr)
            ac_dt_ps[str(seed)][h_lbl] = {}
            for st in strides:
                cfg_s = cfg | {"dt": float(cfg["dt"]) * st}
                fh_s = fi[:, ::st]
                ft_s = ti[:, ::st]
                e_s = fh_s[:, 1:] - ft_s[:, 1:]
                zh, sh = _strang_kstr(fh_s[:200], nui[:200], rhoi[:200], cfg_s, 1)
                rec = compute_signed_recovery(zh.reshape(200, -1), e_s[:200].reshape(200, -1))
                # Measure pure reference midpoint truncation ||R(U)|| and lifted ||B_h tau||
                zt, _ = _strang_kstr(ft_s[:200], nui[:200], rhoi[:200], cfg_s, 1)
                ne = np.maximum(np.linalg.norm(e_s[:200].reshape(200, -1), axis=1), 1e-30)
                ntau = np.linalg.norm(zt.reshape(200, -1), axis=1)
                n_ti = np.maximum(np.linalg.norm(ft_s[:200, 1:].reshape(200, -1), axis=1), 1e-30)
                ac_dt_ps[str(seed)][h_lbl][f"{cfg_s['dt']:.2f}"] = {
                    "cos": rec["cos_mean"],
                    "rel_err": rec["rel_l2_err_mean"],
                    "tau_lift_over_e": float(np.mean(ntau / ne)),
                    "tau_lift_over_u": float(np.mean(ntau / n_ti)),
                }

    ac_kstr_sum = {
        h: {str(ks): {mk: _seed_stats([ac_kstr_ps[str(s)][h][str(ks)][mk] for s in SEEDS]) for mk in ac_kstr_ps[str(SEEDS[0])][h][str(ks)]}
            for ks in k_strs}
        for h in ("T1", "T2")
    }
    ac_dt_sum = {
        h: {dtk: {mk: _seed_stats([ac_dt_ps[str(s)][h][dtk][mk] for s in SEEDS]) for mk in ac_dt_ps[str(SEEDS[0])][h][dtk]}
            for dtk in ("0.05", "0.10", "0.20")}
        for h in ("T1", "T2")
    }
    for h in ("T1", "T2"):
        for dtk in ("0.05", "0.10", "0.20"):
            s = ac_dt_sum[h][dtk]
            print(f"  AC {h} dt={dtk}: cos={s['cos']['mean']:.4f}, relErr={s['rel_err']['mean']:.4f}, "
                  f"||B*tau||/||u||={s['tau_lift_over_u']['mean']:.4f}", flush=True)
    return {
        "burgers_ksub": {"summary": burg_sum, "per_seed": burg_ps},
        "ac_kstr": {"summary": ac_kstr_sum, "per_seed": ac_kstr_ps},
        "ac_dt": {"summary": ac_dt_sum, "per_seed": ac_dt_ps},
    }


# ---------- Part C: SPD-AMG-PCG2 Condition Number, Chebyshev Bound & Positive Homogeneity ----------

def _power_iter_ma_max(apply_a, apply_m, mask, init_vec, max_iter=60, tol=1e-5):
    """Estimate lambda_max(M A) in A-inner product for SPD M, A."""
    v = np.where(mask, init_vec, 0.0)
    av = apply_a(v)
    v = v / math.sqrt(max(float(np.sum(v[mask] * av[mask])), 1e-30))
    av = apply_a(v)
    prev = -1.0
    val = 1.0
    for it in range(max_iter):
        w = apply_m(apply_a(v))
        aw = apply_a(w)
        num = float(np.sum(w[mask] * av[mask]))
        # Rayleigh quotient in A-inner product: <w, A v>_? Wait: <M A v, A v> / <v, A v>
        # Since <v, A v> = 1, <M A v, A v> = sum(w * av)
        val = num
        nw = math.sqrt(max(float(np.sum(w[mask] * aw[mask])), 1e-30))
        v = np.where(mask, w / nw, 0.0)
        av = apply_a(v)
        if it >= 3 and abs(val - prev) <= tol * max(abs(val), 1e-12):
            break
        prev = val
    return float(val)


def _spd_pcg_worker(job):
    dom, a, e, draw_seed = job
    n = a.shape[-1]
    rng = np.random.default_rng(draw_seed)
    init = rng.normal(size=(n, n))

    if dom == "darcy_rect":
        mask = np.ones((n, n), dtype=bool)
        lv = build_rect_amg_hierarchy(a, min_n=4)
        app_a = lambda u: darcy_apply_op(a, u)
        app_m = lambda r: lsh.amg_vcycle(lv, r / (n * n), idx=0, nu=2, omega_coarse=1.0)
        app_pcg2 = lambda r: lsh.lift_amg_pcg(lv, r, steps=2, nu=2)
    else:
        mask = lsh.lshape_mask(n)
        lv = lsh.build_amg_hierarchy(a, min_n=4)
        app_a = lambda u: np.where(mask, lsh.apply_flux(lv[0], u) * (n * n), 0.0)
        app_m = lambda r: np.where(mask, lsh.amg_vcycle(lv, r / (n * n), idx=0, nu=2, omega_coarse=1.0), 0.0)
        app_pcg2 = lambda r: np.where(mask, lsh.lift_amg_pcg(lv, r, steps=2, nu=2), 0.0)

    # 1) ||I - M A||_A via power iteration
    e1_norm = _power_iter_energy_defect(app_a, app_m, mask, init, max_iter=80, tol=1e-5)
    lmax = _power_iter_ma_max(app_a, app_m, mask, init, max_iter=60, tol=1e-5)
    lmin = max(1.0 - e1_norm, 1e-6)
    kappa = max(lmax / lmin, 1.0)
    tau = (math.sqrt(kappa) - 1.0) / (math.sqrt(kappa) + 1.0)
    cheb_exact = (2.0 * (tau ** 2)) / (1.0 + tau ** 4)
    cheb_upper = 2.0 * (tau ** 2)

    # 2) Empirical SPD-AMG-PCG2 directional energy defect and L2 defect on e
    ae = app_a(e)
    z = app_pcg2(ae)
    diff = z - e
    a_diff = app_a(diff)
    ne_l2 = max(float(np.linalg.norm(e[mask])), 1e-30)
    ne_a = math.sqrt(max(float(np.sum(e[mask] * ae[mask])), 1e-30))
    dir_l2 = float(np.linalg.norm(diff[mask]) / ne_l2)
    dir_en = math.sqrt(max(float(np.sum(diff[mask] * a_diff[mask])), 0.0)) / ne_a

    # 3) Degree-1 positive homogeneity check K_2(c r) == c K_2(r) to < 1e-13
    hom_err = 0.0
    z_r = app_pcg2(ae)
    denom = max(float(np.max(np.abs(z_r[mask]))), 1e-30)
    for c in (0.01, 3.7, 1000.0):
        zc = app_pcg2(c * ae)
        hom_err = max(hom_err, float(np.max(np.abs(zc[mask] - c * z_r[mask])) / (c * denom)))

    assert hom_err < 1e-13, f"SPD-AMG-PCG2 positive homogeneity failed: {hom_err:.3e}"
    assert dir_en <= cheb_upper + 1e-6, f"Chebyshev bound violated: {dir_en} > {cheb_upper}"

    return {
        "e1_norm": float(e1_norm),
        "lmin": float(lmin),
        "lmax": float(lmax),
        "kappa": float(kappa),
        "cheb_exact": float(cheb_exact),
        "cheb_upper": float(cheb_upper),
        "dir_l2": float(dir_l2),
        "dir_energy": float(dir_en),
        "cheb_compliant": float(dir_en <= cheb_exact + 1e-6),
        "homogeneity_err": float(hom_err),
    }


def run_part_c_spd_pcg(device, workers=6, n_fields=20):
    print(f"[Task 2C] SPD-AMG-PCG2 Chebyshev bound & positive homogeneity verification ({n_fields} fields/seed x 5 seeds)...", flush=True)
    results = {}
    for dom, splits in (("darcy_rect", (("id_test", "ID"), ("ood_a", "OOD-A"), ("ood_b", "OOD-B"))),
                        ("darcy_lshape", (("id_test", "ID"), ("ood_a", "OOD-A")))):
        ps = {str(s): {} for s in SEEDS}
        for seed in SEEDS:
            if dom == "darcy_rect":
                root = REV_CACHE / "darcy" / f"seed_{seed}"
                p = Protocol("darcy", seed, 64, root, FROZEN_ROOT / f"seed_{seed}", device)
                model = p.open_frozen()
            else:
                sroot = LSHAPE_ROOT / f"seed_{seed}"
                ck = torch.load(sroot / "fno_member_0.pt", map_location=device, weights_only=True)
                model = lsh.LShapeFNO(lsh.CFG["model"]).to(device)
                model.load_state_dict(ck["model"])
                model.eval()
            for sp_k, sp_l in splits:
                d = arrays((REV_CACHE / "darcy" if dom == "darcy_rect" else LSHAPE_ROOT) / f"seed_{seed}/data/{sp_k}.npz")
                a = d["a"][:n_fields].astype(np.float64)
                u = d["u"][:n_fields].astype(np.float64)
                if dom == "darcy_rect":
                    pred = p.predict(model, (d["a"][:n_fields],), 64, None).astype(np.float64)
                else:
                    pred = lsh.predict_lshape(model, d["a"][:n_fields], device, batch=32).astype(np.float64)
                err = pred - u
                jobs = [(dom, a[i], err[i], int(seed * 100000 + i * 31 + 3)) for i in range(n_fields)]
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    rows = list(pool.map(_spd_pcg_worker, jobs, chunksize=4))
                agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
                agg["homogeneity_max_err"] = float(max(r["homogeneity_err"] for r in rows))
                ps[str(seed)][sp_l] = agg
        summary = {}
        for _, sp_l in splits:
            summary[sp_l] = {
                mk: _seed_stats([ps[str(s)][sp_l][mk] for s in SEEDS])
                for mk in ps[str(SEEDS[0])][sp_l]
            }
            s = summary[sp_l]
            print(f"  {dom:12s} {sp_l:5s} | dir_l2={s['dir_l2']['mean']:.3f}, dir_en={s['dir_energy']['mean']:.3f}, "
                  f"kappa={s['kappa']['mean']:.2f}, cheb_exact={s['cheb_exact']['mean']:.3f} (2tau^2={s['cheb_upper']['mean']:.3f}), "
                  f"comp={100*s['cheb_compliant']['mean']:.1f}%, homMax={s['homogeneity_max_err']['mean']:.2e}", flush=True)
        results[dom] = {"summary": summary, "per_seed": ps}
    return results


# ---------- Part D: Extended Jha-Style Variational Point Corrector Comparisons ----------

def _darcy_jha_worker(job):
    a, pred = job
    r = darcy_apply_op(a, pred) - 1.0
    z_liouv = darcy_lift("liouville", a, r)
    lv = build_rect_amg_hierarchy(a, min_n=4)
    z_amg = lsh.lift_amg_v2(lv, r, cycles=2, nu=2, omega_coarse=1.6)
    z_ex = darcy_solve(a, ) if False else None
    # Exact sparse correction solve(A_h z = r) using matrix from assemble_sparse_flux
    from scipy.sparse.linalg import spsolve as _sp
    mat, act = lsh.assemble_sparse_flux(lv[0])
    n = a.shape[-1]
    z_ex = _sp(mat, r.ravel()[act] / (n * n)).reshape(n, n)
    # Orthogonal second residual lift after Jha AMG-V2 correction
    u_jha = pred - z_amg
    r2 = darcy_apply_op(a, u_jha) - 1.0
    z_jha2 = lsh.lift_amg_v2(lv, r2, cycles=2, nu=2, omega_coarse=1.6)
    return z_liouv, z_amg, z_ex, z_jha2


def run_part_d_jha_comparisons(device, workers=6):
    print("[Task 2D] Jha-style variational point corrector vs OpRes-CP & orthogonal combinations (Darcy & Burgers)...", flush=True)
    # 1) Darcy 64^2 (ID, OOD-A, OOD-B)
    d_methods = (
        "OpRes-CP (Liouville)",
        "OpRes-CP (AMG-V2)",
        "Liouville Corr. + Unif-CP",
        "Jha AMG-V2 Corr. + Unif-CP",
        "Jha Exact spsolve + Unif-CP",
        "Jha AMG-V2 + OpRes-CP",
    )
    darcy_ps = {}
    for seed in SEEDS:
        root = REV_CACHE / "darcy" / f"seed_{seed}"
        p = Protocol("darcy", seed, 64, root, FROZEN_ROOT / f"seed_{seed}", device)
        model = p.open_frozen()
        cfg, lam = p.config, p.plan["lambda"]

        def _prep_d(sp_k):
            d = arrays(root / f"data/{sp_k}.npz")
            a = d["a"].astype(np.float64)
            u = d["u"].astype(np.float64)
            pred = p.predict(model, (d["a"],), 64, None).astype(np.float64)
            with ProcessPoolExecutor(max_workers=workers) as pool:
                outs = list(pool.map(_darcy_jha_worker, [(a[i], pred[i]) for i in range(len(a))], chunksize=16))
            zl, za, ze, z2 = (np.stack(x) for x in zip(*outs))
            sl = np.maximum(darcy_mix(darcy_phys_scale(zl, cfg), lam), 1e-12)
            sa = np.maximum(darcy_mix(darcy_phys_scale(za, cfg), lam), 1e-12)
            s2 = np.maximum(darcy_mix(darcy_phys_scale(z2, cfg), lam), 1e-12)
            ones = np.ones_like(pred)
            arms = {
                "OpRes-CP (Liouville)": (pred, sl),
                "OpRes-CP (AMG-V2)": (pred, sa),
                "Liouville Corr. + Unif-CP": (pred - zl, ones),
                "Jha AMG-V2 Corr. + Unif-CP": (pred - za, ones),
                "Jha Exact spsolve + Unif-CP": (pred - ze, ones),
                "Jha AMG-V2 + OpRes-CP": (pred - za, s2),
            }
            return u, arms

        u_c, arms_c = _prep_d("cal")
        qs = {}
        for m in d_methods:
            c, s = arms_c[m]
            sc = (np.abs(c - u_c).reshape(len(u_c), -1) / s.reshape(len(u_c), -1)).max(axis=1)
            qs[m] = quantile(sc, 0.10)

        darcy_ps[str(seed)] = {}
        for sp_k, sp_l in (("id_test", "ID"), ("ood_a", "OOD-A"), ("ood_b", "OOD-B")):
            u_t, arms_t = _prep_d(sp_k)
            sp_out = {}
            for m in d_methods:
                c, s = arms_t[m]
                n = len(u_t)
                sc = (np.abs(c - u_t).reshape(n, -1) / s.reshape(n, -1)).max(axis=1)
                q_emp = value_at_coverage(sc, 0.90)
                sp_out[m] = {
                    "point_rel_l2": float(relative_l2(c, u_t).mean()),
                    "coverage": float((sc <= qs[m]).mean()),
                    "actual_width": float((2.0 * qs[m] * s.reshape(n, -1).mean(axis=1)).mean()),
                    "matched90_width": float(2.0 * q_emp * s.mean()),
                }
            darcy_ps[str(seed)][sp_l] = sp_out

    darcy_sum = {
        sp: {m: {mk: _seed_stats([darcy_ps[str(s)][sp][m][mk] for s in SEEDS]) for mk in darcy_ps[str(SEEDS[0])][sp][m]}
             for m in d_methods}
        for sp in ("ID", "OOD-A", "OOD-B")
    }
    # Measure single-field P50 CPU+GPU inference latency for Darcy Jha vs OpRes arms (seed 42, 20 cases)
    d_timings = {}
    root42 = REV_CACHE / "darcy" / "seed_42"
    p42 = Protocol("darcy", 42, 64, root42, FROZEN_ROOT / "seed_42", device)
    m42 = p42.open_frozen()
    cfg42 = p42.config
    d_id42 = arrays(root42 / "data/id_test.npz")
    a_42 = d_id42["a"][:24].astype(np.float64)
    _ = p42.predict(m42, (d_id42["a"][:1],), 64, None)
    torch.cuda.synchronize(device)
    t_fwd, t_liouv, t_amg, t_ex, t_jha2 = [], [], [], [], []
    for i in range(4, 24):
        torch.cuda.synchronize(device)
        t0 = perf_counter_ns()
        pr = p42.predict(m42, (d_id42["a"][i:i + 1],), 64, None)[0].astype(np.float64)
        torch.cuda.synchronize(device)
        tf = (perf_counter_ns() - t0) / 1e6
        t_fwd.append(tf)

        ai = a_42[i]
        t0 = perf_counter_ns()
        r = darcy_apply_op(ai, pr) - 1.0
        zl = darcy_lift("liouville", ai, r)
        _ = darcy_mix(darcy_phys_scale(zl, cfg42), p42.plan["lambda"])
        t_liouv.append(tf + (perf_counter_ns() - t0) / 1e6)

        t0 = perf_counter_ns()
        r = darcy_apply_op(ai, pr) - 1.0
        lv = build_rect_amg_hierarchy(ai, min_n=4)
        za = lsh.lift_amg_v2(lv, r, cycles=2, nu=2, omega_coarse=1.6)
        _ = darcy_mix(darcy_phys_scale(za, cfg42), p42.plan["lambda"])
        ta = (perf_counter_ns() - t0) / 1e6
        t_amg.append(tf + ta)

        t0 = perf_counter_ns()
        r = darcy_apply_op(ai, pr) - 1.0
        lv = build_rect_amg_hierarchy(ai, min_n=4)
        mat, act = lsh.assemble_sparse_flux(lv[0])
        from scipy.sparse.linalg import spsolve as _sp
        _ = _sp(mat, r.ravel()[act] / (ai.shape[-1] ** 2))
        t_ex.append(tf + (perf_counter_ns() - t0) / 1e6)

        t0 = perf_counter_ns()
        u_j = pr - za
        r2 = darcy_apply_op(ai, u_j) - 1.0
        z2 = lsh.lift_amg_v2(lv, r2, cycles=2, nu=2, omega_coarse=1.6)
        _ = darcy_mix(darcy_phys_scale(z2, cfg42), p42.plan["lambda"])
        t_jha2.append(tf + ta + (perf_counter_ns() - t0) / 1e6)

    d_timings = {
        "OpRes-CP (Liouville)": float(np.median(t_liouv)),
        "OpRes-CP (AMG-V2)": float(np.median(t_amg)),
        "Liouville Corr. + Unif-CP": float(np.median(t_liouv)),
        "Jha AMG-V2 Corr. + Unif-CP": float(np.median(t_amg)),
        "Jha Exact spsolve + Unif-CP": float(np.median(t_ex)),
        "Jha AMG-V2 + OpRes-CP": float(np.median(t_jha2)),
    }

    for sp in ("ID", "OOD-B"):
        print(f"  Darcy {sp:5s} Jha vs OpRes:",
              {m: (f"{100*darcy_sum[sp][m]['point_rel_l2']['mean']:.2f}%",
                   f"{100*darcy_sum[sp][m]['coverage']['mean']:.1f}%",
                   f"{darcy_sum[sp][m]['matched90_width']['mean']:.4f}") for m in d_methods}, flush=True)

    # 2) Burgers 64 (ID, Visc-OOD, High-OOD)
    b_methods = (
        "OpRes-CP (frozen FNO)",
        "Jha 1-step Corr. + Unif-CP",
        "Jha 1-step Corr. + OpRes-CP",
    )
    burg_ps = {}
    for seed in SEEDS:
        root = REV_CACHE / "burgers" / f"seed_{seed}"
        p = Protocol("burgers", seed, 64, root, FROZEN_ROOT / f"seed_{seed}", device)
        model = p.open_frozen()
        cfg = p.config

        def _prep_b(sp_k):
            d = arrays(root / f"data/{sp_k}.npz")
            inp, y = p.split(d, 64, p.frames)
            nu = inp[1].astype(np.float64)
            pred = np.concatenate([p.predict(model, take(inp, sl), 64, p.frames) for sl in batches(len(y), 64)], axis=0).astype(np.float64)
            full = np.concatenate((inp[0][:, None].astype(np.float64), pred), axis=1)
            r = burg_ops.trajectory_residual(full, nu, cfg)
            z = burg_ops.lift("adv_diff", full, r, nu, cfg)
            s1 = np.maximum(p3_mix(burg_ops.envelope(z, cfg), 0.85), 1e-12)
            pred_jha = pred - z
            full_jha = np.concatenate((inp[0][:, None].astype(np.float64), pred_jha), axis=1)
            r2 = burg_ops.trajectory_residual(full_jha, nu, cfg)
            z2 = burg_ops.lift("adv_diff", full_jha, r2, nu, cfg)
            s2 = np.maximum(p3_mix(burg_ops.envelope(z2, cfg), 0.85), 1e-12)
            arms = {
                "OpRes-CP (frozen FNO)": (pred, s1),
                "Jha 1-step Corr. + Unif-CP": (pred_jha, np.ones_like(pred)),
                "Jha 1-step Corr. + OpRes-CP": (pred_jha, s2),
            }
            return y.astype(np.float64), arms

        yc, ac = _prep_b("cal")
        qb = {}
        for m in b_methods:
            c, s = ac[m]
            qb[m] = quantile((np.abs(c - yc).reshape(len(yc), -1) / s.reshape(len(yc), -1)).max(axis=1), 0.10)

        burg_ps[str(seed)] = {}
        for sp_k, sp_l in (("id", "ID"), ("ood", "Visc-OOD"), ("ood_high", "High-OOD")):
            yt, at = _prep_b(sp_k)
            sp_out = {}
            for m in b_methods:
                c, s = at[m]
                n = len(yt)
                sc = (np.abs(c - yt).reshape(n, -1) / s.reshape(n, -1)).max(axis=1)
                q_emp = value_at_coverage(sc, 0.90)
                sp_out[m] = {
                    "point_rel_l2": float(relative_l2(c, yt).mean()),
                    "coverage": float((sc <= qb[m]).mean()),
                    "actual_width": float((2.0 * qb[m] * s.reshape(n, -1).mean(axis=1)).mean()),
                    "matched90_width": float(2.0 * q_emp * s.mean()),
                }
            burg_ps[str(seed)][sp_l] = sp_out

    burg_sum = {
        sp: {m: {mk: _seed_stats([burg_ps[str(s)][sp][m][mk] for s in SEEDS]) for mk in burg_ps[str(SEEDS[0])][sp][m]}
             for m in b_methods}
        for sp in ("ID", "Visc-OOD", "High-OOD")
    }
    # Measure single-trajectory P50 CPU+GPU latency for Burgers arms (seed 42, 20 cases)
    broot42 = REV_CACHE / "burgers" / "seed_42"
    bp42 = Protocol("burgers", 42, 64, broot42, FROZEN_ROOT / "seed_42", device)
    bm42 = bp42.open_frozen()
    bcfg42 = bp42.config
    bid42 = arrays(broot42 / "data/id.npz")
    binp42, _ = bp42.split(bid42, 64, bp42.frames)
    _ = bp42.predict(bm42, take(binp42, slice(0, 1)), 64, bp42.frames)
    torch.cuda.synchronize(device)
    tb_op, tb_jc, tb_j2 = [], [], []
    for i in range(4, 24):
        xi = take(binp42, slice(i, i + 1))
        nui = xi[1].astype(np.float64)
        torch.cuda.synchronize(device)
        t0 = perf_counter_ns()
        pr = bp42.predict(bm42, xi, 64, bp42.frames).astype(np.float64)
        torch.cuda.synchronize(device)
        tf = (perf_counter_ns() - t0) / 1e6

        t0 = perf_counter_ns()
        full = np.concatenate((xi[0][:, None].astype(np.float64), pr), axis=1)
        r1 = burg_ops.trajectory_residual(full, nui, bcfg42)
        z1 = burg_ops.lift("adv_diff", full, r1, nui, bcfg42)
        t_lift = (perf_counter_ns() - t0) / 1e6

        t0 = perf_counter_ns()
        _ = p3_mix(burg_ops.envelope(z1, bcfg42), 0.85)
        t_env = (perf_counter_ns() - t0) / 1e6
        tb_op.append(tf + t_lift + t_env)
        tb_jc.append(tf + t_lift)

        t0 = perf_counter_ns()
        fj = np.concatenate((xi[0][:, None].astype(np.float64), pr - z1), axis=1)
        r2 = burg_ops.trajectory_residual(fj, nui, bcfg42)
        z2 = burg_ops.lift("adv_diff", fj, r2, nui, bcfg42)
        _ = p3_mix(burg_ops.envelope(z2, bcfg42), 0.85)
        tb_j2.append(tf + t_lift + (perf_counter_ns() - t0) / 1e6)

    b_timings = {
        "OpRes-CP (frozen FNO)": float(np.median(tb_op)),
        "Jha 1-step Corr. + Unif-CP": float(np.median(tb_jc)),
        "Jha 1-step Corr. + OpRes-CP": float(np.median(tb_j2)),
    }
    for sp in ("ID", "Visc-OOD", "High-OOD"):
        print(f"  Burgers {sp:8s} Jha vs OpRes:",
              {m: (f"{100*burg_sum[sp][m]['point_rel_l2']['mean']:.2f}%",
                   f"{100*burg_sum[sp][m]['coverage']['mean']:.1f}%",
                   f"{burg_sum[sp][m]['matched90_width']['mean']:.4f}") for m in b_methods}, flush=True)

    return {
        "darcy": {"summary": darcy_sum, "per_seed": darcy_ps, "latency_p50_ms": d_timings},
        "burgers": {"summary": burg_sum, "per_seed": burg_ps, "latency_p50_ms": b_timings},
    }


def run_task2(device=None, workers=6):
    OUT.mkdir(parents=True, exist_ok=True)
    V4_CACHE.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = perf_counter()

    part_a = run_part_a_nonlinear(device)
    part_b = run_part_b_time_convergence(device)
    part_c = run_part_c_spd_pcg(device, workers=workers, n_fields=20)
    part_d = run_part_d_jha_comparisons(device, workers=workers)

    out = {
        "seeds": list(SEEDS),
        "elapsed_s": perf_counter() - t0,
        "part_a_nonlinear_decomp": part_a,
        "part_b_time_convergence": part_b,
        "part_c_spd_pcg": part_c,
        "part_d_jha_comparisons": part_d,
    }
    path = OUT / "task2_nonlinear_pcg_jha.json"
    path.write_text(json.dumps(finite_json(out), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[Rev4 Task 2] Done in {out['elapsed_s']:.1f}s -> {path}", flush=True)
    return out


if __name__ == "__main__":
    run_task2()
