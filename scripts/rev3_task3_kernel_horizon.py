"""Revision v3 Task 3 (P0):
  Part A: Continuous kernel-resolution transition scan l_{pix,train} in
          {0.64, 1.0, 1.28, 1.5, 2.0, 2.56, 3.0, 4.0} on Darcy 32^2 -> 64^2/128^2
          and Burgers 64 -> 128/256 (comparing fixed-physical-radius vs. fixed-cell-footprint).
  Part B: Allen-Cahn T=1 -> T=2 Stiller-style temporal state-reweighted WCP comparison
          (Full-Grid 4096-D state u(T) vs. Compact 5-D temporal state summary, both
          Transductive and Split-Fit 250/250, with zero target labels) vs. OpRes-CP.
  Part C: Millard-style (ICLR 2026) scalar radius resolution extrapolation baseline on
          Darcy 32^2 -> 64^2 -> 128^2 (Uniform-CP + Millard vs. OpRes-CP vs. OpRes + Millard).

Strictly read-only against results/phase2_kaggle_private/.
Formal seeds locked to [42, 137, 314, 271, 811].
Outputs written to results/phase3_revision_v3/task3_kernel_horizon.json.
"""

import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import burgers_phase1_ops as burg_ops
from darcy_data import apply_operator as darcy_apply_op, generate as darcy_gen
from darcy_day15 import arrays, estimated_weights, value_at_coverage, weighted_quantile
from darcy_day15_ops import lift as darcy_lift, mixture as darcy_mix, physical_scale as darcy_phys_scale
from darcy_phase0_eval import quantile
from phase2d import batches, take
from phase2d_data import Protocol
from phase3_common import finite_json, mix as p3_mix
from phase3_summarize import SEEDS, STAGE, stat
from rev2_task5_ac_amg import ensure_ac_t2_split, strang_and_scale_gpu_fp64


OUT = Path("results/phase3_revision_v3")
V3_CACHE = OUT / "cache"
REV_CACHE = Path("results/phase3_revision/cache")
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


def _ensure_darcy32_splits(seed, device, workers=6):
    root = V3_CACHE / "darcy32" / f"seed_{seed}"
    p = Protocol("darcy", seed, 32, root, FROZEN_ROOT / f"seed_{seed}", device)
    # Fast parallel pre-generation for cal (32) and paired_test (32->128, 32->64)
    for split, sub, fine in (("cal", "data", None), ("paired_test", "data", None), ("paired_test", "intermediate_data", 64)):
        d_dir = root / sub
        if not (d_dir / f"{split}.npz").exists():
            stream = p.plan["streams"][split]
            seq = np.random.SeedSequence([seed, stream])
            cfg = p.config | ({"fine": fine} if fine else {}) | {"counts": {split: p.plan["counts"][split]}}
            darcy_gen(cfg, seq, d_dir, workers, True, id_offset=stream * 10000)
    p.data_path("cal")
    p.data_path("paired_test", 128)
    p.data_path("paired_test", 64)
    return p


# ---------- Part A: Continuous Kernel-Resolution Transition Scan ----------

def run_part_a_kernel_scan(device, workers=6):
    print("[Task 3A] Continuous kernel-resolution scan l_{pix,train} in [0.64, 4.0] (Darcy 32->64/128, Burgers 64->128/256)...", flush=True)
    lpix_grid = (0.64, 1.0, 1.28, 1.5, 2.0, 2.56, 3.0, 4.0)

    # 1) Darcy 32^2 -> 64^2 / 128^2 (5 seeds)
    darcy_per_seed = {}
    for seed in SEEDS:
        p = _ensure_darcy32_splits(seed, device, workers=workers)
        model = p.open_frozen()
        cfg, inv, lam = p.config, p.plan["inverse"], p.plan["lambda"]

        def _prep_darcy(split, grid):
            d = arrays(p.data_path(split, grid))
            inp, truth = p.split(d, grid, None)
            pred = p.predict(model, inp, grid, None).astype(np.float64)
            a = inp[0].astype(np.float64)
            r = darcy_apply_op(a, pred) - 1.0
            z = darcy_lift(inv, a, r)
            e = np.abs(pred - truth.astype(np.float64))
            return z, e

        z_cal, e_cal = _prep_darcy("cal", 32)
        z_32, e_32 = _prep_darcy("paired_test", 32)
        z_64, e_64 = _prep_darcy("paired_test", 64)
        z_128, e_128 = _prep_darcy("paired_test", 128)

        seed_out = {}
        for lpix in lpix_grid:
            lphys = lpix / 32.0
            c_cal = cfg | {"physical_radius": lphys}
            s_cal = np.maximum(darcy_mix(darcy_phys_scale(z_cal, c_cal), lam), 1e-12)
            q90 = quantile((e_cal.reshape(len(e_cal), -1) / s_cal.reshape(len(e_cal), -1)).max(axis=1), 0.10)

            def _eval_d(z_t, e_t, grid, policy):
                r_use = lphys if policy == "fixed_physical" else (lpix / float(grid))
                sc = np.maximum(darcy_mix(darcy_phys_scale(z_t, cfg | {"physical_radius": r_use}), lam), 1e-12)
                n = len(e_t)
                scores = (e_t.reshape(n, -1) / sc.reshape(n, -1)).max(axis=1)
                q_emp = value_at_coverage(scores, 0.90)
                return {
                    "coverage": float((scores <= q90).mean()),
                    "actual_width": float((2.0 * q90 * sc.reshape(n, -1).mean(axis=1)).mean()),
                    "matched90_width": float(2.0 * q_emp * sc.reshape(n, -1).mean()),
                }

            r32 = _eval_d(z_32, e_32, 32, "fixed_physical")
            r64_phys = _eval_d(z_64, e_64, 64, "fixed_physical")
            r64_cell = _eval_d(z_64, e_64, 64, "fixed_cell")
            r128_phys = _eval_d(z_128, e_128, 128, "fixed_physical")
            r128_cell = _eval_d(z_128, e_128, 128, "fixed_cell")
            seed_out[f"{lpix:g}"] = {
                "lphys": lphys,
                "g32": r32,
                "g64_fixed_phys": r64_phys | {"drift_pp": 100.0 * (r64_phys["coverage"] - r32["coverage"]),
                                              "width_ratio": r64_phys["actual_width"] / r32["actual_width"]},
                "g64_fixed_cell": r64_cell | {"drift_pp": 100.0 * (r64_cell["coverage"] - r32["coverage"]),
                                              "width_ratio": r64_cell["actual_width"] / r32["actual_width"]},
                "g128_fixed_phys": r128_phys | {"drift_pp": 100.0 * (r128_phys["coverage"] - r32["coverage"]),
                                                "width_ratio": r128_phys["actual_width"] / r32["actual_width"]},
                "g128_fixed_cell": r128_cell | {"drift_pp": 100.0 * (r128_cell["coverage"] - r32["coverage"]),
                                                "width_ratio": r128_cell["actual_width"] / r32["actual_width"]},
            }
        darcy_per_seed[str(seed)] = seed_out
        print(f"  Darcy32->128 seed={seed} | fixed_phys cov(0.64/1.28/2.0/4.0)="
              f"{100*seed_out['0.64']['g128_fixed_phys']['coverage']:.1f}%/"
              f"{100*seed_out['1.28']['g128_fixed_phys']['coverage']:.1f}%/"
              f"{100*seed_out['2']['g128_fixed_phys']['coverage']:.1f}%/"
              f"{100*seed_out['4']['g128_fixed_phys']['coverage']:.1f}% | "
              f"fixed_cell(1.28)={100*seed_out['1.28']['g128_fixed_cell']['coverage']:.1f}%", flush=True)

    # 2) Burgers 64 -> 128 / 256 (5 seeds)
    burg_per_seed = {}
    for seed in SEEDS:
        rev_root = REV_CACHE / "burgers" / f"seed_{seed}"
        v3_root = V3_CACHE / "burgers" / f"seed_{seed}"
        p_cal = Protocol("burgers", seed, 64, rev_root, FROZEN_ROOT / f"seed_{seed}", device)
        p_pair = Protocol("burgers", seed, 64, v3_root, FROZEN_ROOT / f"seed_{seed}", device)
        p_pair.data_path("paired")
        model = p_cal.open_frozen()
        cfg = p_cal.config

        def _prep_burg(d, grid):
            inp, truth = p_cal.split(d, grid, p_cal.frames)
            preds = [p_cal.predict(model, take(inp, sl), grid, p_cal.frames) for sl in batches(len(truth), 64)]
            pred = np.concatenate(preds, axis=0).astype(np.float64)
            full = np.concatenate((inp[0][:, None].astype(np.float64), pred), axis=1)
            nu = inp[1].astype(np.float64)
            r = burg_ops.trajectory_residual(full, nu, cfg)
            z = burg_ops.lift("adv_diff", full, r, nu, cfg)
            e = np.abs(pred - truth.astype(np.float64))
            return z, e

        z_cal, e_cal = _prep_burg(arrays(rev_root / "data/cal.npz"), 64)
        paired_d = arrays(v3_root / "data/paired.npz")
        z_64, e_64 = _prep_burg(paired_d, 64)
        z_128, e_128 = _prep_burg(paired_d, 128)
        z_256, e_256 = _prep_burg(paired_d, 256)

        seed_out = {}
        for lpix in lpix_grid:
            lphys = lpix * (2.0 * math.pi) / 64.0
            s_cal = np.maximum(p3_mix(burg_ops.envelope(z_cal, cfg | {"spatial_radius": lphys}), 0.85), 1e-12)
            q90 = quantile((e_cal.reshape(len(e_cal), -1) / s_cal.reshape(len(e_cal), -1)).max(axis=1), 0.10)

            def _eval_b(z_t, e_t, grid, policy):
                r_use = lphys if policy == "fixed_physical" else (lpix * (2.0 * math.pi) / float(grid))
                sc = np.maximum(p3_mix(burg_ops.envelope(z_t, cfg | {"spatial_radius": r_use}), 0.85), 1e-12)
                n = len(e_t)
                scores = (e_t.reshape(n, -1) / sc.reshape(n, -1)).max(axis=1)
                q_emp = value_at_coverage(scores, 0.90)
                return {
                    "coverage": float((scores <= q90).mean()),
                    "actual_width": float((2.0 * q90 * sc.reshape(n, -1).mean(axis=1)).mean()),
                    "matched90_width": float(2.0 * q_emp * sc.reshape(n, -1).mean()),
                }

            r64 = _eval_b(z_64, e_64, 64, "fixed_physical")
            r128_phys = _eval_b(z_128, e_128, 128, "fixed_physical")
            r128_cell = _eval_b(z_128, e_128, 128, "fixed_cell")
            r256_phys = _eval_b(z_256, e_256, 256, "fixed_physical")
            r256_cell = _eval_b(z_256, e_256, 256, "fixed_cell")
            seed_out[f"{lpix:g}"] = {
                "lphys": lphys,
                "g64": r64,
                "g128_fixed_phys": r128_phys | {"drift_pp": 100.0 * (r128_phys["coverage"] - r64["coverage"]),
                                                "width_ratio": r128_phys["actual_width"] / r64["actual_width"]},
                "g128_fixed_cell": r128_cell | {"drift_pp": 100.0 * (r128_cell["coverage"] - r64["coverage"]),
                                                "width_ratio": r128_cell["actual_width"] / r64["actual_width"]},
                "g256_fixed_phys": r256_phys | {"drift_pp": 100.0 * (r256_phys["coverage"] - r64["coverage"]),
                                                "width_ratio": r256_phys["actual_width"] / r64["actual_width"]},
                "g256_fixed_cell": r256_cell | {"drift_pp": 100.0 * (r256_cell["coverage"] - r64["coverage"]),
                                                "width_ratio": r256_cell["actual_width"] / r64["actual_width"]},
            }
        burg_per_seed[str(seed)] = seed_out
        print(f"  Burgers64->256 seed={seed} | fixed_phys cov(0.64/1.0/2.0/4.0)="
              f"{100*seed_out['0.64']['g256_fixed_phys']['coverage']:.1f}%/"
              f"{100*seed_out['1']['g256_fixed_phys']['coverage']:.1f}%/"
              f"{100*seed_out['2']['g256_fixed_phys']['coverage']:.1f}%/"
              f"{100*seed_out['4']['g256_fixed_phys']['coverage']:.1f}%", flush=True)

    def _agg_scan(pseed):
        out = {}
        for lk in pseed[str(SEEDS[0])]:
            out[lk] = {"lphys": pseed[str(SEEDS[0])][lk]["lphys"]}
            for gk in pseed[str(SEEDS[0])][lk]:
                if gk == "lphys":
                    continue
                mkeys = list(pseed[str(SEEDS[0])][lk][gk].keys())
                out[lk][gk] = {
                    mk: _seed_stats([pseed[str(s)][lk][gk][mk] for s in SEEDS])
                    for mk in mkeys
                }
        return out

    return {
        "lpix_grid": list(lpix_grid),
        "darcy_32_to_128": {"summary": _agg_scan(darcy_per_seed), "per_seed": darcy_per_seed},
        "burgers_64_to_256": {"summary": _agg_scan(burg_per_seed), "per_seed": burg_per_seed},
    }


# ---------- Part B: Stiller-style Temporal State-Reweighted WCP (Allen-Cahn T=1 -> T=2) ----------

def _temporal_features(end_pred, nu, rho):
    """Unlabeled temporal state features at horizon endpoint t=T (zero target truth):
      - Full-Grid: flattened u_hat(T) (4096-D) + (nu, rho) -> 4098-D
      - Compact Summary (5-D): [RMS(u_hat(T)), RMS(|grad u_hat(T)|), interface_frac(|u_hat|<0.5), nu, rho]
    """
    n = len(end_pred)
    dx = np.roll(end_pred, 1, axis=-1) - np.roll(end_pred, -1, axis=-1)
    dy = np.roll(end_pred, 1, axis=-2) - np.roll(end_pred, -1, axis=-2)
    rms_u = np.sqrt(np.mean(end_pred * end_pred, axis=(-2, -1)))
    rms_g = np.sqrt(np.mean(dx * dx + dy * dy, axis=(-2, -1)))
    iface = (np.abs(end_pred) < 0.5).mean(axis=(-2, -1))
    compact = np.column_stack((rms_u, rms_g, iface, nu, rho))
    full = np.column_stack((end_pred.reshape(n, -1), nu, rho))
    return compact.astype(np.float32), full.astype(np.float32)


def run_part_b_temporal_wcp(device):
    print("[Task 3B] Allen-Cahn T=1 -> T=2 Stiller-style temporal state-reweighted WCP comparison...", flush=True)
    per_seed = {}
    for seed in SEEDS:
        root = REV_CACHE / "allen_cahn" / f"seed_{seed}"
        p = Protocol("allen_cahn", seed, 64, root, FROZEN_ROOT / f"seed_{seed}", device)
        model = p.open_frozen()
        cfg = p.config

        d_t1 = arrays(root / "data/cal.npz")
        d_t2 = ensure_ac_t2_split(seed, "id", cfg, device)

        def _scores_and_endstate(d, frames):
            inp, y = p.split(d, 64, frames)
            pred = np.concatenate([p.predict(model, take(inp, sl), 64, frames) for sl in batches(len(y), 16)], axis=0)
            full = np.concatenate((inp[0][:, None], pred), axis=1).astype(np.float64)
            nu = inp[1].astype(np.float64)
            rho = inp[2].astype(np.float64)
            sigs = []
            for sl in batches(len(y), 25):
                sigs.append(strang_and_scale_gpu_fp64(full[sl], nu[sl], rho[sl], cfg, device, jac_on=True))
            sig = np.maximum(np.concatenate(sigs, axis=0), 1e-12)
            n = len(y)
            sc = (np.abs(pred - y.astype(np.float64)).reshape(n, -1) / sig.reshape(n, -1)).max(axis=1)
            sm = sig.reshape(n, -1).mean(axis=1)
            comp, full_f = _temporal_features(pred[:, -1].astype(np.float64), nu, rho)
            return sc, sm, comp, full_f

        sc_t1, _, comp_t1, full_t1 = _scores_and_endstate(d_t1, 20)
        sc_t2, sm_t2, comp_t2, full_t2 = _scores_and_endstate(d_t2, 40)

        # Unweighted T1 90% and 95% reuse baselines
        q1_90 = quantile(sc_t1, 0.10)
        q1_95 = quantile(sc_t1, 0.05)

        def _run_wcp(xc, xt):
            # 1) Transductive (500 Cal vs 500 Test unlabeled features)
            wc_tr, wt_tr = estimated_weights(xc, xt, {"base_weight_clip": (0.02, 50.0), "weight_logistic_c": 1.0})
            q_tr, p_tr, ess_tr = weighted_quantile(sc_t1, wc_tr, wt_tr, 0.10)
            fin_tr = np.isfinite(q_tr)
            cov_tr = np.where(fin_tr, sc_t2 <= q_tr, True)
            tr_res = {
                "coverage": float(cov_tr.mean()),
                "infinite_rate": float((~fin_tr).mean()),
                "finite_n": int(fin_tr.sum()),
                "finite_coverage": float(cov_tr[fin_tr].mean()) if fin_tr.any() else None,
                "finite_width": float((2.0 * q_tr[fin_tr] * sm_t2[fin_tr]).mean()) if fin_tr.any() else None,
                "ess": float(ess_tr),
                "p_test_median": float(np.median(p_tr)),
            }

            # 2) Independent Split-Fit (250 Cal + 250 held-out Test fit; disjoint 250 Cal + 250 eval Test)
            n_c, n_t = len(xc), len(xt)
            half_c, half_t = n_c // 2, n_t // 2
            q_sf = np.empty(n_t, dtype=np.float64)
            ess_list = []
            for test_sl, hold_sl in ((slice(0, half_t), slice(half_t, n_t)), (slice(half_t, n_t), slice(0, half_t))):
                # Fit classifier on Cal[:250] vs held-out Test half
                from sklearn.linear_model import LogisticRegression
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler
                x_fit = np.vstack((xc[:half_c], xt[hold_sl]))
                y_fit = np.r_[np.zeros(half_c, dtype=int), np.ones(len(xt[hold_sl]), dtype=int)]
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
                clf.fit(x_fit, y_fit)
                prior = len(xt[hold_sl]) / half_c

                def _w(arr):
                    prob = clf.predict_proba(arr)[:, 1]
                    prob = np.clip(prob, 1e-6, 1.0 - 1e-6)
                    return np.clip(prob / (1.0 - prob) / prior, 0.02, 50.0)

                wc_eval = _w(xc[half_c:])
                wt_eval = _w(xt[test_sl])
                qt, _, ess_h = weighted_quantile(sc_t1[half_c:], wc_eval, wt_eval, 0.10)
                q_sf[test_sl] = qt
                ess_list.append(float(ess_h))
            fin_sf = np.isfinite(q_sf)
            cov_sf = np.where(fin_sf, sc_t2 <= q_sf, True)
            sf_res = {
                "coverage": float(cov_sf.mean()),
                "infinite_rate": float((~fin_sf).mean()),
                "finite_n": int(fin_sf.sum()),
                "finite_coverage": float(cov_sf[fin_sf].mean()) if fin_sf.any() else None,
                "finite_width": float((2.0 * q_sf[fin_sf] * sm_t2[fin_sf]).mean()) if fin_sf.any() else None,
                "ess": float(np.mean(ess_list)),
            }
            return {"transductive": tr_res, "split_fit": sf_res}

        wcp_comp = _run_wcp(comp_t1, comp_t2)
        wcp_full = _run_wcp(full_t1, full_t2)
        per_seed[str(seed)] = {
            "unweighted_t1_90": {
                "coverage": float((sc_t2 <= q1_90).mean()),
                "actual_width": float((2.0 * q1_90 * sm_t2).mean()),
            },
            "unweighted_t1_95": {
                "coverage": float((sc_t2 <= q1_95).mean()),
                "actual_width": float((2.0 * q1_95 * sm_t2).mean()),
            },
            "compact_summary_wcp": wcp_comp,
            "full_grid_state_wcp": wcp_full,
        }
        print(f"  AC Temporal WCP seed={seed} | T1_90={100*per_seed[str(seed)]['unweighted_t1_90']['coverage']:.1f}% | "
              f"Compact(Tr/SF) ESS={wcp_comp['transductive']['ess']:.1f}/{wcp_comp['split_fit']['ess']:.1f} "
              f"Inf={100*wcp_comp['transductive']['infinite_rate']:.1f}%/{100*wcp_comp['split_fit']['infinite_rate']:.1f}% "
              f"Cov={100*wcp_comp['transductive']['coverage']:.1f}%/{100*wcp_comp['split_fit']['coverage']:.1f}% | "
              f"Full(Tr/SF) ESS={wcp_full['transductive']['ess']:.1f}/{wcp_full['split_fit']['ess']:.1f} "
              f"Inf={100*wcp_full['transductive']['infinite_rate']:.1f}%/{100*wcp_full['split_fit']['infinite_rate']:.1f}%",
              flush=True)

    # Aggregate 5 seeds
    def _agg_leaf(getter):
        vals = []
        for s in SEEDS:
            v = getter(per_seed[str(s)])
            if v is not None and isinstance(v, (int, float)) and math.isfinite(v):
                vals.append(float(v))
        if len(vals) == len(SEEDS):
            return _seed_stats(vals)
        return {"mean": float(np.mean(vals)) if vals else None,
                "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                "n_defined": len(vals)}

    summary = {
        "unweighted_t1_90": {
            k: _agg_leaf(lambda d, kk=k: d["unweighted_t1_90"][kk]) for k in ("coverage", "actual_width")
        },
        "unweighted_t1_95": {
            k: _agg_leaf(lambda d, kk=k: d["unweighted_t1_95"][kk]) for k in ("coverage", "actual_width")
        },
    }
    for mkey in ("compact_summary_wcp", "full_grid_state_wcp"):
        summary[mkey] = {}
        for proto in ("transductive", "split_fit"):
            mnames = list(per_seed[str(SEEDS[0])][mkey][proto].keys())
            summary[mkey][proto] = {
                mk: _agg_leaf(lambda d, mm=mkey, pp=proto, kk=mk: d[mm][pp][kk])
                for mk in mnames
            }
    return {"summary": summary, "per_seed": per_seed}


# ---------- Part C: Millard-style (ICLR 2026) Scalar Radius Extrapolation Comparison ----------

def run_part_c_millard_extrapolation(device, workers=6):
    print("[Task 3C] Millard-style scalar radius extrapolation comparison on Darcy 32^2 -> 64^2 -> 128^2...", flush=True)
    per_seed = {}
    for seed in SEEDS:
        p = _ensure_darcy32_splits(seed, device, workers=workers)
        model = p.open_frozen()
        cfg, inv, lam = p.config, p.plan["inverse"], p.plan["lambda"]

        # Load paired 32, 64, 128; split 0:250 as two-level calibration stream for q(h) fit,
        # and 250:500 as strictly held-out paired test evaluation!
        def _scores_scales(grid):
            d = arrays(p.data_path("paired_test", grid))
            inp, truth = p.split(d, grid, None)
            pred = p.predict(model, inp, grid, None).astype(np.float64)
            a = inp[0].astype(np.float64)
            r = darcy_apply_op(a, pred) - 1.0
            z = darcy_lift(inv, a, r)
            e = np.abs(pred - truth.astype(np.float64))
            n = len(e)
            ef = e.reshape(n, -1)
            unif_sc = ef.max(axis=1)

            # OpRes physical radius (0.04) and fixed-cell (1.28 / grid)
            s_phys = np.maximum(darcy_mix(darcy_phys_scale(z, cfg), lam), 1e-12).reshape(n, -1)
            s_cell = np.maximum(darcy_mix(darcy_phys_scale(z, cfg | {"physical_radius": 1.28 / grid}), lam), 1e-12).reshape(n, -1)
            return {
                "unif_sc": unif_sc,
                "phys_sc": (ef / s_phys).max(axis=1),
                "phys_w": s_phys.mean(axis=1),
                "cell_sc": (ef / s_cell).max(axis=1),
                "cell_w": s_cell.mean(axis=1),
            }

        g32 = _scores_scales(32)
        g64 = _scores_scales(64)
        g128 = _scores_scales(128)

        # Also get standard 32^2 Cal (500 fields) for zero-shot OpRes fixed-cell baseline
        d_cal = arrays(p.data_path("cal", 32))
        inp_c, tr_c = p.split(d_cal, 32, None)
        pred_c = p.predict(model, inp_c, 32, None).astype(np.float64)
        a_c = inp_c[0].astype(np.float64)
        z_c = darcy_lift(inv, a_c, darcy_apply_op(a_c, pred_c) - 1.0)
        s_c = np.maximum(darcy_mix(darcy_phys_scale(z_c, cfg), lam), 1e-12).reshape(len(a_c), -1)
        q_op_cal32 = quantile((np.abs(pred_c - tr_c.astype(np.float64)).reshape(len(a_c), -1) / s_c).max(axis=1), 0.10)
        q_unif_cal32 = quantile(np.abs(pred_c - tr_c.astype(np.float64)).reshape(len(a_c), -1).max(axis=1), 0.10)

        fit_sl = slice(0, 250)
        ev_sl = slice(250, 500)
        h32, h64, h128 = 1.0 / 32.0, 1.0 / 64.0, 1.0 / 128.0
        extrap_factor = (h128 - h64) / (h64 - h32)  # = 0.5

        def _fit_extrap(sc32, sc64):
            q32 = quantile(sc32[fit_sl], 0.10)
            q64 = quantile(sc64[fit_sl], 0.10)
            q128_lin = max(q64 + extrap_factor * (q64 - q32), 1e-12)
            # Also quadratic in h: q(h) = q0 + c * h^2 -> (h128^2 - h64^2)/(h64^2 - h32^2) = 0.25
            q128_h2 = max(q64 + 0.25 * (q64 - q32), 1e-12)
            return q32, q64, q128_lin, q128_h2

        uq32, uq64, uq128_lin, uq128_h2 = _fit_extrap(g32["unif_sc"], g64["unif_sc"])
        cq32, cq64, cq128_lin, _ = _fit_extrap(g32["cell_sc"], g64["cell_sc"])
        pq32, pq64, pq128_lin, _ = _fit_extrap(g32["phys_sc"], g64["phys_sc"])

        ev128_u = g128["unif_sc"][ev_sl]
        ev128_c = g128["cell_sc"][ev_sl]
        ev128_cw = g128["cell_w"][ev_sl]
        ev128_p = g128["phys_sc"][ev_sl]
        ev128_pw = g128["phys_w"][ev_sl]

        per_seed[str(seed)] = {
            "uniform_zeroshot_cal32": {
                "coverage": float((ev128_u <= q_unif_cal32).mean()),
                "actual_width": float(2.0 * q_unif_cal32),
            },
            "uniform_millard_lin_h": {
                "coverage": float((ev128_u <= uq128_lin).mean()),
                "actual_width": float(2.0 * uq128_lin),
            },
            "uniform_millard_h2": {
                "coverage": float((ev128_u <= uq128_h2).mean()),
                "actual_width": float(2.0 * uq128_h2),
            },
            "opres_zeroshot_fixed_phys_cal32": {
                "coverage": float((ev128_p <= q_op_cal32).mean()),
                "actual_width": float((2.0 * q_op_cal32 * ev128_pw).mean()),
            },
            "opres_zeroshot_fixed_cell_cal32": {
                "coverage": float((ev128_c <= q_op_cal32).mean()),
                "actual_width": float((2.0 * q_op_cal32 * ev128_cw).mean()),
            },
            "opres_fixed_cell_plus_millard_lin_h": {
                "coverage": float((ev128_c <= cq128_lin).mean()),
                "actual_width": float((2.0 * cq128_lin * ev128_cw).mean()),
            },
            "opres_fixed_phys_plus_millard_lin_h": {
                "coverage": float((ev128_p <= pq128_lin).mean()),
                "actual_width": float((2.0 * pq128_lin * ev128_pw).mean()),
            },
        }
        ps = per_seed[str(seed)]
        print(f"  Millard extrap seed={seed} (128^2 held-out) | Unif(ZS/M-h)={100*ps['uniform_zeroshot_cal32']['coverage']:.1f}%/"
              f"{100*ps['uniform_millard_lin_h']['coverage']:.1f}% (w={ps['uniform_millard_lin_h']['actual_width']:.4f}) | "
              f"OpRes-Cell(ZS/M-h)={100*ps['opres_zeroshot_fixed_cell_cal32']['coverage']:.1f}%/"
              f"{100*ps['opres_fixed_cell_plus_millard_lin_h']['coverage']:.1f}% (w={ps['opres_zeroshot_fixed_cell_cal32']['actual_width']:.4f})",
              flush=True)

    summary = {}
    for mname in per_seed[str(SEEDS[0])]:
        summary[mname] = {
            mk: _seed_stats([per_seed[str(s)][mname][mk] for s in SEEDS])
            for mk in ("coverage", "actual_width")
        }
    return {"summary": summary, "per_seed": per_seed}


def run_task3(device=None, workers=6):
    OUT.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = perf_counter()

    part_a = run_part_a_kernel_scan(device, workers=workers)
    part_b = run_part_b_temporal_wcp(device)
    part_c = run_part_c_millard_extrapolation(device, workers=workers)

    out = {
        "seeds": list(SEEDS),
        "elapsed_s": perf_counter() - t0,
        "part_a_kernel_scan": part_a,
        "part_b_temporal_wcp": part_b,
        "part_c_millard_extrapolation": part_c,
    }
    path = OUT / "task3_kernel_horizon.json"
    path.write_text(json.dumps(finite_json(out), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[Task 3] Done in {out['elapsed_s']:.1f}s -> {path}", flush=True)
    return out


if __name__ == "__main__":
    run_task3()
