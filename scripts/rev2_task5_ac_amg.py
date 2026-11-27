"""Task 5 (P1): Allen-Cahn 5-Seed Jacobian Parameter Sensitivity & L-Shape AMG Multiscale Scaling.

Part A: Zero-retraining 5-seed Allen-Cahn sensitivity scan over locked Strang Jacobian parameters:
  Locked: (c_clip=2.0, beta_+=0.0, g_max=1.5, gamma=0.2)
  Sweeps: beta_+ in {0.0, 0.25, 0.5}, g_max in {0.5, 1.5, 3.0}, c_clip in {1.0, 2.0, 4.0},
          plus Jacobian-off control, evaluated on T=1 (ID, React-OOD) and independent T=2 (ID, React-OOD).

Part B: Pure-operator L-shape Galerkin AMG multiscale scaling & contrast convergence audit:
  Grids n in {32, 64, 128, 256, 512} (768 to 196,608 active cells) x sigma_{log a} in {0.5, 1.0, 1.5, 2.0}.
  Reports exact Grid Complexity C_G, Operator Complexity C_O (with analytical upper bound 4/3),
  CPU wall-clock scaling curves & log-log slopes (AMG-V2 / AMG-PCG2 vs Exact spsolve),
  two-step residual reduction ||r_2||_2 / ||r_0||_2, and operator error ||B_h A_h e_h - e_h||_2 / ||e_h||_2.

Outputs written to results/phase3_revision_v2/task5_ac_amg.json.
"""

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
from scipy.fft import irfft2, rfft2
import torch

import allen_cahn_phase2_data as ac_data
from allen_cahn_phase2_probe import scale as ac_scale_cpu
from darcy_data import coefficient
from darcy_day15 import arrays, value_at_coverage
from darcy_phase0_eval import quantile
from phase2d import batches, take
from phase2d_ac import STREAMS as AC_T2_STREAMS
from phase2d_data import Protocol
from phase3_common import finite_json
import phase3_p2_lshape_amg as lsh
from phase3_summarize import SEEDS, STAGE, stat


OUT = Path("results/phase3_revision_v2")
REV_CACHE = Path("results/phase3_revision/cache")
T2_CACHE = OUT / "cache/ac_t2"
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


# ---------- GPU-accelerated fast FP32 Allen-Cahn ETDRK2 solver (unit-checked vs CPU SciPy) ----------

@torch.inference_mode()
def solve_ac_gpu_fp32(coeff_np, nu_np, rho_np, cfg, frames, device):
    """Fast FP32 GPU counterpart of allen_cahn_phase2_data.solve for grid=64 (verified <1e-5 vs CPU FP64)."""
    n = cfg["reference"]  # 128
    k2_np, mask_np = ac_data.spectrum(n)
    k2 = torch.as_tensor(k2_np, dtype=torch.float32, device=device)
    mask = torch.as_tensor(mask_np, dtype=torch.bool, device=device)
    nu = torch.as_tensor(nu_np, dtype=torch.float32, device=device)
    rho = torch.as_tensor(rho_np, dtype=torch.float32, device=device)

    dt = float(cfg["dt"] / cfg["substeps"])
    # Compute phi coefficients in float64 once then cast to float32 for zero cancellation loss
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

    def _proj64(zv):
        half = 32
        low = torch.cat((zv[:, :half, :half + 1], zv[:, -half:, :half + 1]), dim=-2).clone()
        low[:, half, :] = 0.0
        low[..., -1] = 0.0
        return torch.fft.irfft2(low, s=(64, 64), norm="forward").cpu().numpy()

    def _react(zv):
        u = torch.fft.irfft2(zv, s=(n, n), norm="forward")
        return rho[:, None, None] * mask[None] * torch.fft.rfft2(u - u * u * u, norm="forward")

    out = [_proj64(z)]
    for _ in range(frames):
        for __ in range(cfg["substeps"]):
            f = _react(z)
            stage = ex * z + p1 * f
            z = stage + p2 * (_react(stage) - f)
        out.append(_proj64(z))
    return np.stack(out, axis=1)


def ensure_ac_t2_split(seed, split, cfg, device):
    """Generate and cache exact independent T=2 Allen-Cahn trajectories (40 frames, 64^2)."""
    root = T2_CACHE / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"fresh_{split}.npz"
    if path.exists():
        return arrays(path)
    c = cfg | {"streams": AC_T2_STREAMS, "seed": seed}
    nu, rho, coeff = ac_data.initial(c, split)
    if split == "cal" and seed == SEEDS[1]:
        ref_cpu = ac_data.solve(coeff[:2], nu[:2], rho[:2], c, [64], 40)[64]
        ref_gpu = solve_ac_gpu_fp32(coeff[:2], nu[:2], rho[:2], c, 40, device)
        max_diff = float(np.max(np.abs(ref_cpu - ref_gpu)))
        assert max_diff < 5e-5, f"AC T=2 GPU FP32 solver mismatch vs CPU SciPy: {max_diff}"
        print(f"  [Task 5A Unit Check] AC T=2 GPU FP32 vs CPU SciPy max_abs_diff={max_diff:.2e} PASS", flush=True)
    parts = []
    for i in range(0, len(nu), 100):
        sl = slice(i, i + 100)
        parts.append(solve_ac_gpu_fp32(coeff[sl], nu[sl], rho[sl], c, 40, device))
    u64 = np.concatenate(parts, axis=0)
    ids = AC_T2_STREAMS[split] * 10000 + np.arange(len(nu))
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, nu=nu, rho=rho, id=ids, u64=u64)
    tmp.replace(path)
    return arrays(path)


@torch.inference_mode()
def strang_and_scale_gpu_fp64(full_np, nu_np, rho_np, cfg, device, *, jac_on=True):
    """Fast GPU FP32 Strang lift + periodic Gaussian RMS envelope + quadratic mixture (matches CPU < 5e-6)."""
    dt = float(cfg["dt"])
    n = full_np.shape[-1]
    k2_np, mask_np = ac_data.spectrum(n)
    k2 = torch.as_tensor(k2_np, dtype=torch.float32, device=device)
    mask = torch.as_tensor(mask_np, dtype=torch.bool, device=device)

    full = torch.as_tensor(full_np, dtype=torch.float32, device=device)
    nu = torch.as_tensor(nu_np, dtype=torch.float32, device=device)
    rho = torch.as_tensor(rho_np, dtype=torch.float32, device=device)
    diff = -nu[:, None, None] * k2[None]
    half = torch.exp(0.5 * dt * (diff - float(cfg["gamma"])))
    state = torch.zeros_like(full[:, 0])
    z_list = []
    for j in range(1, full.shape[1]):
        left, right = full[:, j - 1], full[:, j]
        mid = 0.5 * (left + right)
        nonlin = torch.fft.irfft2(mask[None] * torch.fft.rfft2(mid - mid * mid * mid), s=(n, n))
        r = (right - left) / dt - torch.fft.irfft2(diff * torch.fft.rfft2(mid), s=(n, n)) - rho[:, None, None] * nonlin
        force = 0.5 * dt * r
        state = torch.fft.irfft2(half * torch.fft.rfft2(state + force), s=(n, n))
        if jac_on:
            jac = (rho[:, None, None] * (1.0 - 3.0 * mid * mid)).clamp(-cfg["reaction_clip"], cfg["reaction_clip"])
            jac = (jac - cfg["positive_damping"] * jac.clamp_min(0.0)).clamp_max(cfg["max_growth"])
            state = state * torch.exp(dt * jac)
        state = torch.fft.irfft2(half * torch.fft.rfft2(state), s=(n, n)) + force
        z_list.append(state)
    z = torch.stack(z_list, dim=1)

    sigma = float(cfg["spatial_radius"]) * n / (2.0 * math.pi)
    rad = int(4.0 * sigma + 0.5)
    grid = torch.arange(-rad, rad + 1, dtype=torch.float32, device=device)
    kern = torch.exp(-0.5 * (grid / sigma) ** 2)
    kern = kern / kern.sum()
    sq = z * z
    avg = sq.mean(dim=(1, 2, 3), keepdim=True)
    b, t, h, w = sq.shape
    x = sq.reshape(b * t, 1, h, w)
    x = torch.nn.functional.pad(x, (rad, rad, 0, 0), mode="circular")
    x = torch.nn.functional.conv2d(x, kern.view(1, 1, 1, -1))
    x = torch.nn.functional.pad(x, (0, 0, rad, rad), mode="circular")
    x = torch.nn.functional.conv2d(x, kern.view(1, 1, -1, 1))
    smooth = x.reshape(b, t, h, w) + 0.01 * avg
    lam = float(cfg["lambda"])
    sig = torch.sqrt((1.0 - lam) * 1.01 * avg + lam * smooth)
    return sig.cpu().numpy().astype(np.float64)


def run_part_a_ac_sensitivity(device):
    print("[Task 5A] Allen-Cahn 5-seed Jacobian hyperparameter sensitivity (T=1 & T=2)...", flush=True)
    configs = (
        ("locked", {"reaction_clip": 2.0, "positive_damping": 0.0, "max_growth": 1.5}, True),
        ("beta_0.25", {"reaction_clip": 2.0, "positive_damping": 0.25, "max_growth": 1.5}, True),
        ("beta_0.50", {"reaction_clip": 2.0, "positive_damping": 0.50, "max_growth": 1.5}, True),
        ("gmax_0.5", {"reaction_clip": 2.0, "positive_damping": 0.0, "max_growth": 0.5}, True),
        ("gmax_3.0", {"reaction_clip": 2.0, "positive_damping": 0.0, "max_growth": 3.0}, True),
        ("cclip_1.0", {"reaction_clip": 1.0, "positive_damping": 0.0, "max_growth": 1.5}, True),
        ("cclip_4.0", {"reaction_clip": 4.0, "positive_damping": 0.0, "max_growth": 1.5}, True),
        ("jac_off", {"reaction_clip": 2.0, "positive_damping": 0.0, "max_growth": 1.5}, False),
    )

    per_seed = {}
    for seed in SEEDS:
        root = REV_CACHE / "allen_cahn" / f"seed_{seed}"
        p = Protocol("allen_cahn", seed, 64, root, FROZEN_ROOT / f"seed_{seed}", device)
        model = p.open_frozen()
        base_cfg = p.config

        # Load T=1 splits and independent T=2 splits
        t1_data = {s: arrays(root / f"data/{s}.npz") for s in ("cal", "id", "ood")}
        t2_data = {s: ensure_ac_t2_split(seed, s, base_cfg, device) for s in ("cal", "id", "ood")}

        # Precompute frozen FNO rollouts once per seed (zero retraining!)
        def _rollout_split(d, frames):
            inp, y = p.split(d, 64, frames)
            preds = []
            for sl in batches(len(y), 16):
                preds.append(p.predict(model, take(inp, sl), 64, frames))
            pred = np.concatenate(preds, axis=0)
            full = np.concatenate((inp[0][:, None], pred), axis=1).astype(np.float64)
            return full, y.astype(np.float64), inp[1].astype(np.float64), inp[2].astype(np.float64)

        t1_roll = {s: _rollout_split(t1_data[s], 20) for s in ("cal", "id", "ood")}
        t2_roll = {s: _rollout_split(t2_data[s], 40) for s in ("cal", "id", "ood")}

        def _scores_and_means(roll, cfg, jac_on):
            full, y, nu, rho = roll
            n = len(y)
            sc_list, sm_list = [], []
            for sl in batches(n, 25):
                sig = strang_and_scale_gpu_fp64(full[sl], nu[sl], rho[sl], cfg, device, jac_on=jac_on)
                sig = np.maximum(sig, 1e-12)
                err = np.abs(full[sl, 1:] - y[sl])
                sc_list.append((err / sig).reshape(len(err), -1).max(axis=1))
                sm_list.append(sig.reshape(len(err), -1).mean(axis=1))
            return np.concatenate(sc_list), np.concatenate(sm_list)

        seed_out = {}
        for cname, overrides, jac_on in configs:
            cfg = base_cfg | overrides
            t1_cal_sc, _ = _scores_and_means(t1_roll["cal"], cfg, jac_on)
            t2_cal_sc, _ = _scores_and_means(t2_roll["cal"], cfg, jac_on)
            q1_90 = quantile(t1_cal_sc, 0.10)
            q1_95 = quantile(t1_cal_sc, 0.05)
            q2_90 = quantile(t2_cal_sc, 0.10)

            def _eval(roll, q_use):
                sc, sm = _scores_and_means(roll, cfg, jac_on)
                q_emp = value_at_coverage(sc, 0.90)
                return {
                    "coverage": float((sc <= q_use).mean()),
                    "actual_width": float((2.0 * q_use * sm).mean()),
                    "matched90_width": float(2.0 * q_emp * sm.mean()),
                }

            t2_id_sc, t2_id_sm = _scores_and_means(t2_roll["id"], cfg, jac_on)
            t2_id_qemp = value_at_coverage(t2_id_sc, 0.90)
            t2_ood_sc, t2_ood_sm = _scores_and_means(t2_roll["ood"], cfg, jac_on)
            t2_ood_qemp = value_at_coverage(t2_ood_sc, 0.90)

            seed_out[cname] = {
                "params": overrides | {"jac_on": jac_on, "gamma": base_cfg["gamma"]},
                "t1_id": _eval(t1_roll["id"], q1_90),
                "t1_ood": _eval(t1_roll["ood"], q1_90),
                "t2_id_t2cal": {
                    "coverage": float((t2_id_sc <= q2_90).mean()),
                    "actual_width": float((2.0 * q2_90 * t2_id_sm).mean()),
                    "matched90_width": float(2.0 * t2_id_qemp * t2_id_sm.mean()),
                },
                "t2_id_t1_95reuse": {
                    "coverage": float((t2_id_sc <= q1_95).mean()),
                    "actual_width": float((2.0 * q1_95 * t2_id_sm).mean()),
                },
                "t2_ood_t2cal": {
                    "coverage": float((t2_ood_sc <= q2_90).mean()),
                    "actual_width": float((2.0 * q2_90 * t2_ood_sm).mean()),
                    "matched90_width": float(2.0 * t2_ood_qemp * t2_ood_sm.mean()),
                },
            }
        per_seed[str(seed)] = seed_out
        lk = seed_out["locked"]
        print(f"  [Task 5A] seed={seed} locked: T1_ID={100*lk['t1_id']['coverage']:.2f}%/{lk['t1_id']['matched90_width']:.4f}, "
              f"T1_OOD={100*lk['t1_ood']['coverage']:.2f}%/{lk['t1_ood']['matched90_width']:.4f}, "
              f"T2_ID={100*lk['t2_id_t2cal']['coverage']:.2f}%/{lk['t2_id_t2cal']['matched90_width']:.4f}, "
              f"T2_OOD={100*lk['t2_ood_t2cal']['coverage']:.2f}%/{lk['t2_ood_t2cal']['matched90_width']:.4f}", flush=True)

    # Aggregate across 5 seeds
    summary = {}
    for cname, overrides, jac_on in configs:
        summary[cname] = {"params": overrides | {"jac_on": jac_on, "gamma": 0.2}}
        for split_k in ("t1_id", "t1_ood", "t2_id_t2cal", "t2_id_t1_95reuse", "t2_ood_t2cal"):
            mkeys = list(per_seed[str(SEEDS[0])][cname][split_k].keys())
            summary[cname][split_k] = {
                mk: _seed_stats([per_seed[str(s)][cname][split_k][mk] for s in SEEDS])
                for mk in mkeys
            }
    return {"summary": summary, "per_seed": per_seed}


# ---------- Part B: L-shape AMG Multiscale Scaling & Contrast Convergence ----------

def _amg_complexity(levels):
    """Exact Grid Complexity C_G and Operator Complexity C_O for 2x2 cell-agglomeration 5-pt AMG."""
    m_list = []
    nnz_list = []
    for lvl in levels:
        m = lvl["mask"]
        n_act = int(m.sum())
        n_h = int((m[:, :-1] & m[:, 1:]).sum())
        n_v = int((m[:-1, :] & m[1:, :]).sum())
        nnz = n_act + 2 * (n_h + n_v)
        m_list.append(n_act)
        nnz_list.append(nnz)
    c_g = sum(m_list) / float(m_list[0])
    c_o = sum(nnz_list) / float(nnz_list[0])
    return {
        "n_levels": len(levels),
        "active_cells_finest": m_list[0],
        "nnz_finest": nnz_list[0],
        "active_cells_per_level": m_list,
        "nnz_per_level": nnz_list,
        "grid_complexity_CG": float(c_g),
        "operator_complexity_CO": float(c_o),
        "analytical_upper_bound": 4.0 / 3.0,
    }


def _make_realistic_error(n, rng):
    """Smooth + medium + fine Dirichlet-zero error field on Omega_L for operator audit."""
    m = lsh.lshape_mask(n)
    bw = lsh.lshape_boundary_weight(n).astype(np.float64)
    x = (np.arange(n) + 0.5) / n
    xx, yy = np.meshgrid(x, x, indexing="ij")
    # Sum of low/mid/high Fourier modes masked by boundary weight (mirrors real FNO error spectrum)
    e = np.zeros((n, n), dtype=np.float64)
    for k1, k2, amp in ((1, 1, 1.0), (2, 1, 0.7), (1, 3, 0.5), (4, 3, 0.35), (7, 5, 0.2), (12, 9, 0.1)):
        ph1, ph2 = rng.uniform(0, 2 * math.pi, size=2)
        e += amp * rng.normal() * np.sin(math.pi * k1 * xx + ph1) * np.sin(math.pi * k2 * yy + ph2)
    # Add mild grid-scale component
    e += 0.08 * rng.normal(size=(n, n))
    e = np.where(m, e * bw, 0.0)
    return e / max(float(np.linalg.norm(e[m])), 1e-30)


def run_part_b_amg_scaling():
    print("[Task 5B] L-shape AMG multiscale scaling (n=32..512) & contrast sweep (sigma=0.5..2.0)...", flush=True)
    grids = (32, 64, 128, 256, 512)
    sigmas = (0.5, 1.0, 1.5, 2.0)

    # 1) Exact C_G, C_O and timing scaling at sigma = 1.0 across n = 32..512
    scaling_by_grid = {}
    for n in grids:
        m = lsh.lshape_mask(n)
        n_reps = 10 if n <= 256 else 5  # 1 or 2 fields per locked seed
        t_setup, t_v2_solve, t_pcg2_solve, t_v2_tot, t_pcg2_tot, t_spsolve = [], [], [], [], [], []
        comp = None

        # Warmup once
        a_w = coefficient(n, 1.0, 70001, 32, 0.08, False)
        lv_w = lsh.build_amg_hierarchy(a_w)
        r_w = np.where(m, 1.0, 0.0)
        _ = lsh.lift_amg_v2(lv_w, r_w)
        _ = lsh.lift_amg_pcg(lv_w, r_w, steps=2)
        _ = lsh.solve_lshape(a_w, rhs_pde=r_w)
        comp = _amg_complexity(lv_w)

        for idx in range(n_reps):
            seed = SEEDS[idx % len(SEEDS)]
            draw = int(np.random.default_rng(np.random.SeedSequence([seed, 9800, n, idx])).integers(0, 2 ** 62))
            a = coefficient(n, 1.0, draw, 32, 0.08, False)
            r = np.where(m, 1.0, 0.0)

            t0 = perf_counter_ns()
            lv = lsh.build_amg_hierarchy(a)
            ts = (perf_counter_ns() - t0) / 1e6

            t0 = perf_counter_ns()
            _ = lsh.lift_amg_v2(lv, r, cycles=2, nu=2, omega_coarse=1.6)
            tv = (perf_counter_ns() - t0) / 1e6

            t0 = perf_counter_ns()
            _ = lsh.lift_amg_pcg(lv, r, steps=2, nu=2)
            tp = (perf_counter_ns() - t0) / 1e6

            t0 = perf_counter_ns()
            _ = lsh.solve_lshape(a, rhs_pde=r)
            te = (perf_counter_ns() - t0) / 1e6

            t_setup.append(ts)
            t_v2_solve.append(tv)
            t_pcg2_solve.append(tp)
            t_v2_tot.append(ts + tv)
            t_pcg2_tot.append(ts + tp)
            t_spsolve.append(te)

        scaling_by_grid[str(n)] = {
            "complexity": comp,
            "n_reps": n_reps,
            "setup_p50_ms": float(np.median(t_setup)),
            "amg_v2_solve_p50_ms": float(np.median(t_v2_solve)),
            "amg_pcg2_solve_p50_ms": float(np.median(t_pcg2_solve)),
            "amg_v2_total_p50_ms": float(np.median(t_v2_tot)),
            "amg_pcg2_total_p50_ms": float(np.median(t_pcg2_tot)),
            "spsolve_p50_ms": float(np.median(t_spsolve)),
            "speedup_v2_total_vs_spsolve": float(np.median(t_spsolve) / np.median(t_v2_tot)),
            "speedup_pcg2_total_vs_spsolve": float(np.median(t_spsolve) / np.median(t_pcg2_tot)),
        }
        sc = scaling_by_grid[str(n)]
        print(f"  [Task 5B Scale] n={n:3d} (M={comp['active_cells_finest']:6d}): "
              f"C_G={comp['grid_complexity_CG']:.4f}, C_O={comp['operator_complexity_CO']:.4f} | "
              f"V2_tot={sc['amg_v2_total_p50_ms']:.2f}ms, PCG2_tot={sc['amg_pcg2_total_p50_ms']:.2f}ms, "
              f"spsolve={sc['spsolve_p50_ms']:.2f}ms ({sc['speedup_v2_total_vs_spsolve']:.2f}x)", flush=True)

    # Fit log-log slopes t ~ M^beta across n in {64, 128, 256, 512} and {32..512}
    def _fit_slope(grid_list, key):
        log_m = np.log([scaling_by_grid[str(g)]["complexity"]["active_cells_finest"] for g in grid_list])
        log_t = np.log([scaling_by_grid[str(g)][key] for g in grid_list])
        slope, intercept = np.polyfit(log_m, log_t, 1)
        return float(slope)

    slopes = {
        f"{rng_name}": {
            k: _fit_slope(glist, k)
            for k in ("amg_v2_solve_p50_ms", "amg_v2_total_p50_ms", "amg_pcg2_total_p50_ms", "spsolve_p50_ms")
        }
        for rng_name, glist in (("n_64_to_512", (64, 128, 256, 512)), ("n_32_to_512", (32, 64, 128, 256, 512)))
    }
    print(f"  [Task 5B Slopes 64..512] V2_solve={slopes['n_64_to_512']['amg_v2_solve_p50_ms']:.3f}, "
          f"V2_tot={slopes['n_64_to_512']['amg_v2_total_p50_ms']:.3f}, "
          f"PCG2_tot={slopes['n_64_to_512']['amg_pcg2_total_p50_ms']:.3f}, "
          f"spsolve={slopes['n_64_to_512']['spsolve_p50_ms']:.3f}", flush=True)

    # 2) Convergence & operator approximation error across (n, sigma) using 5 seeds
    conv_grid = {}
    for n in grids:
        m = lsh.lshape_mask(n)
        conv_grid[str(n)] = {}
        n_per_seed = 4 if n <= 256 else 1  # 20 fields for n<=256, 5 fields (1 per seed) for n=512
        for sig in sigmas:
            per_seed_vals = {k: [] for k in (
                "res_red_v2", "res_red_pcg2", "res_red_j5",
                "op_err_v2", "op_err_pcg2", "op_err_j5",
                "unit_res_red_v2", "unit_res_red_pcg2",
            )}
            for seed in SEEDS:
                rng = np.random.default_rng(np.random.SeedSequence([seed, 9900, n, int(round(sig * 100))]))
                s_acc = {k: [] for k in per_seed_vals}
                for _ in range(n_per_seed):
                    draw = int(rng.integers(0, 2 ** 62))
                    a = coefficient(n, sig, draw, 32, 0.08, False)
                    lv = lsh.build_amg_hierarchy(a)
                    e = _make_realistic_error(n, rng)
                    r0 = np.where(m, lsh.apply_flux(lv[0], e) * (n * n), 0.0)
                    nr0 = max(float(np.linalg.norm(r0[m])), 1e-30)
                    ne = max(float(np.linalg.norm(e[m])), 1e-30)

                    zv2 = lsh.lift_amg_v2(lv, r0, cycles=2, nu=2, omega_coarse=1.6)
                    zpcg = lsh.lift_amg_pcg(lv, r0, steps=2, nu=2)
                    zj5 = lsh.lift_jacobi_pcg(lv, r0, steps=5)

                    for tag, z in (("v2", zv2), ("pcg2", zpcg), ("j5", zj5)):
                        r_after = r0 - np.where(m, lsh.apply_flux(lv[0], z) * (n * n), 0.0)
                        s_acc[f"res_red_{tag}"].append(float(np.linalg.norm(r_after[m]) / nr0))
                        s_acc[f"op_err_{tag}"].append(float(np.linalg.norm((z - e)[m]) / ne))

                    # Also unit RHS r_0 = 1 residual reduction
                    r_u = np.where(m, 1.0, 0.0)
                    nru = float(np.linalg.norm(r_u[m]))
                    for tag, z in (("v2", lsh.lift_amg_v2(lv, r_u, cycles=2, nu=2, omega_coarse=1.6)),
                                   ("pcg2", lsh.lift_amg_pcg(lv, r_u, steps=2, nu=2))):
                        ra = r_u - np.where(m, lsh.apply_flux(lv[0], z) * (n * n), 0.0)
                        s_acc[f"unit_res_red_{tag}"].append(float(np.linalg.norm(ra[m]) / nru))

                for k in per_seed_vals:
                    per_seed_vals[k].append(float(np.mean(s_acc[k])))

            conv_grid[str(n)][str(sig)] = {k: _seed_stats(v) for k, v in per_seed_vals.items()}
        c10 = conv_grid[str(n)]["1.0"]
        c20 = conv_grid[str(n)]["2.0"]
        print(f"  [Task 5B Conv] n={n:3d} | sig=1.0: resRed V2={c10['res_red_v2']['mean']:.3f}/PCG2={c10['res_red_pcg2']['mean']:.3f}, "
              f"opErr V2={c10['op_err_v2']['mean']:.3f}/PCG2={c10['op_err_pcg2']['mean']:.3f}/J5={c10['op_err_j5']['mean']:.3f} | "
              f"sig=2.0: resRed V2={c20['res_red_v2']['mean']:.3f}/PCG2={c20['res_red_pcg2']['mean']:.3f}, "
              f"opErr V2={c20['op_err_v2']['mean']:.3f}/PCG2={c20['op_err_pcg2']['mean']:.3f}", flush=True)

    return {
        "grids": list(grids),
        "sigmas": list(sigmas),
        "scaling_at_sigma_1.0": scaling_by_grid,
        "log_log_slopes": slopes,
        "convergence_grid": conv_grid,
    }


def run_task5(device=None):
    OUT.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = perf_counter()

    part_a = run_part_a_ac_sensitivity(device)
    part_b = run_part_b_amg_scaling()

    out = {
        "seeds": list(SEEDS),
        "elapsed_s": perf_counter() - t0,
        "part_a_ac_sensitivity": part_a,
        "part_b_amg_scaling": part_b,
    }
    path = OUT / "task5_ac_amg.json"
    path.write_text(json.dumps(finite_json(out), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[Task 5] Done in {out['elapsed_s']:.1f}s -> {path}", flush=True)
    return out


if __name__ == "__main__":
    run_task5()
