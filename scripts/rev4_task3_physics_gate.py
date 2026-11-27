"""Revision v4 Task 3 (P1):
  Part A: Enriched Cal-only physical residual fingerprints (RMS(Z_h), effective wavenumber
          k_eff = ||R_h||_2 / ||Z_h||_2, residual high-frequency ratio ||R_h^{k>8}||_2 / ||R_h||_2,
          early-window peak-to-RMS ratio max_{t in [0.2,0.5]} ||R_h||_inf / RMS(Z_h))
          on Burgers ID-only noise and ambient noise (eta_noise in {0.0, 0.02, 0.05, 0.10, 0.20}).
  Part B: Finite-sample exchangeable physics-adaptive global-local mixture lambda(A) in [lam_min, 0.85]
          whose annealing rule depends deterministically ONLY on Val quantiles + test input A & frozen FNO,
          re-calibrated on Cal to preserve Theorem 1 exact ID validity while probing High-OOD recovery
          across alpha_hf in {0.0, 0.05, 0.10, 0.25, 0.50, 1.0} and eta_noise = 0.05.

Outputs written to results/phase3_revision_v4/task3_physics_gate.json.
"""

import json
import os
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from burgers_phase1_data import solve as burgers_solve
import burgers_phase1_ops as burg_ops
from darcy_day15 import arrays, value_at_coverage
from darcy_phase0_eval import quantile
from phase2d import batches, take
from phase2d_burgers_diagnosis import features as burg_features
from phase2d_data import Protocol
from phase2d_statistics import relative_l2
from phase3_common import finite_json, mix as p3_mix, rms as p3_rms
from phase3_summarize import SEEDS, STAGE, stat
from rev2_task3_burgers import ALPHAS_HF, ETAS_NOISE, _compute_gate_metrics, _noisy_burgers_split


OUT = Path("results/phase3_revision_v4")
REV_CACHE = Path("results/phase3_revision/cache")
V3_CACHE = Path("results/phase3_revision_v3/cache")
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


def _solve_burgers_batch(coeff, nu, cfg):
    parts = []
    for i in range(0, len(nu), 64):
        sl = slice(i, i + 64)
        u, _ = burgers_solve(coeff[sl], nu[sl], cfg, [64])
        parts.append(u[64].astype(np.float32))
    return np.concatenate(parts, axis=0)


def _extract_physics_fingerprints(u0, nu, y_true, model, p, cfg):
    preds = []
    for sl in batches(len(u0), 64):
        preds.append(p.predict(model, (u0[sl].astype(np.float32), nu[sl].astype(np.float32)), 64, p.frames))
    pred = np.concatenate(preds, axis=0).astype(np.float64)
    full = np.concatenate((u0[:, None].astype(np.float64), pred), axis=1)
    r = burg_ops.trajectory_residual(full, nu, cfg)
    z = burg_ops.lift("adv_diff", full, r, nu, cfg)
    env = burg_ops.envelope(z, cfg)  # before lambda mix
    sc_85 = np.maximum(p3_mix(env, 0.85), 1e-12)

    n = len(u0)
    rf = r.reshape(n, -1)
    zf = z.reshape(n, -1)
    nr = np.linalg.norm(rf, axis=1)
    nz = np.linalg.norm(zf, axis=1)
    rms_z = p3_rms(z).ravel()
    k_eff = nr / np.maximum(nz, 1e-12)

    # Residual high-frequency energy fraction (modes k > 8)
    r_spec = np.abs(np.fft.rfft(r, axis=-1)) ** 2
    r_spec[..., 1:-1] *= 2.0
    r_hf = np.sqrt(np.maximum(r_spec[..., 9:].sum(axis=(1, 2)), 0.0) / np.maximum(r_spec.sum(axis=(1, 2)), 1e-30))

    # Early-window (t in [0.2, 0.5], frame indices 1..4) residual peak-to-RMS(Z_h)
    early_peak = np.abs(r[:, 1:5, :]).reshape(n, -1).max(axis=1) / np.maximum(rms_z, 1e-12)

    _, spec_feat = burg_features(u0.astype(np.float64), nu.astype(np.float64))
    hf_in = spec_feat[:, -1]

    out = {
        "pred": pred,
        "env": env,
        "sc_85": sc_85,
        "rms_z": rms_z,
        "k_eff": k_eff,
        "r_hf": r_hf,
        "early_peak": early_peak,
        "hf_in": hf_in,
    }
    if y_true is not None:
        scores_85 = (np.abs(pred - y_true) / sc_85).reshape(n, -1).max(axis=1)
        rel_err = relative_l2(pred, y_true)
        out["scores_85"] = scores_85
        out["rel_err"] = rel_err
    return out


def _cal_zscore_max(cal_arrs, test_arrs):
    """Cal-only standardized max log-feature gate (zero target leakage, zero batch transductive fitting)."""
    c_cols, t_cols = [], []
    for ca, ta in zip(cal_arrs, test_arrs):
        lc = np.log(np.maximum(ca, 1e-12))
        lt = np.log(np.maximum(ta, 1e-12))
        mu = float(lc.mean())
        sd = max(float(lc.std()), 1e-8)
        c_cols.append((lc - mu) / sd)
        t_cols.append((lt - mu) / sd)
    return np.max(np.column_stack(c_cols), axis=1), np.max(np.column_stack(t_cols), axis=1)


def _adaptive_lambda_scale(env, lam_vec, c_vec=1.0):
    """Compute per-trajectory sigma_h(x) using instance-specific deterministic Val multiplier and lambda(A)."""
    rms2 = np.mean(env * env, axis=(-2, -1), keepdims=True)
    lam_b = lam_vec[:, None, None]
    c_b = np.asarray(c_vec, dtype=np.float64)[:, None, None] if np.ndim(c_vec) > 0 else float(c_vec)
    return np.maximum(c_b * np.sqrt((1.0 - lam_b) * rms2 + lam_b * (env * env)), 1e-12)


def run_task3_physics_gate(device=None):
    OUT.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = perf_counter()

    # ---------- Part A: Enriched Cal-only Physical Fingerprints on Noisy Overlap ----------
    print("[Task 3A] Enriched Cal-only physical residual fingerprints on noisy spectral overlap...", flush=True)
    part_a_ps = {"id_only_noise": {}, "ambient_noise": {}}

    for seed in SEEDS:
        root = REV_CACHE / "burgers" / f"seed_{seed}"
        p = Protocol("burgers", seed, 64, root, FROZEN_ROOT / f"seed_{seed}", device)
        model = p.open_frozen()
        cfg = p.config

        cal_clean = arrays(root / "data/cal.npz")
        id_clean = arrays(root / "data/id.npz")
        hi_clean = arrays(root / "data/ood_high.npz")
        hi_ext_clean = _extract_physics_fingerprints(hi_clean["u64"][:, 0], hi_clean["nu"],
                                                     hi_clean["u64"][:, 1:].astype(np.float64), model, p, cfg)

        seed_id, seed_amb = {}, {}
        for e_idx, eta in enumerate(ETAS_NOISE):
            ek = str(eta)
            if eta == 0.0:
                cal_u64, id_u64, hi_amb_u64 = cal_clean["u64"], id_clean["u64"], hi_clean["u64"]
            else:
                rng_c = np.random.default_rng(np.random.SeedSequence([seed, 6950 + e_idx * 3 + 0]))
                rng_i = np.random.default_rng(np.random.SeedSequence([seed, 6950 + e_idx * 3 + 1]))
                rng_h = np.random.default_rng(np.random.SeedSequence([seed, 6950 + e_idx * 3 + 2]))
                cal_u64 = _noisy_burgers_split(cal_clean["u64"][:, 0], cal_clean["nu"], eta, cfg, rng_c)
                id_u64 = _noisy_burgers_split(id_clean["u64"][:, 0], id_clean["nu"], eta, cfg, rng_i)
                hi_amb_u64 = _noisy_burgers_split(hi_clean["u64"][:, 0], hi_clean["nu"], eta, cfg, rng_h)

            cal_ext = _extract_physics_fingerprints(cal_u64[:, 0], cal_clean["nu"], cal_u64[:, 1:].astype(np.float64), model, p, cfg)
            id_ext = _extract_physics_fingerprints(id_u64[:, 0], id_clean["nu"], id_u64[:, 1:].astype(np.float64), model, p, cfg)
            hi_amb_ext = hi_ext_clean if eta == 0.0 else _extract_physics_fingerprints(
                hi_amb_u64[:, 0], hi_clean["nu"], hi_amb_u64[:, 1:].astype(np.float64), model, p, cfg)
            q90_cal = quantile(cal_ext["scores_85"], 0.10)

            def _eval(hi_ext):
                n_id, n_hi = len(id_ext["scores_85"]), len(hi_ext["scores_85"])
                is_ood = np.concatenate((np.zeros(n_id, dtype=bool), np.ones(n_hi, dtype=bool)))
                scores_mix = np.concatenate((id_ext["scores_85"], hi_ext["scores_85"]))
                rel_mix = np.concatenate((id_ext["rel_err"], hi_ext["rel_err"]))
                pf = rel_mix > 0.15
                bh = scores_mix <= q90_cal

                gates = {}
                for gk in ("rms_z", "k_eff", "r_hf", "early_peak", "hf_in"):
                    gates[gk] = _compute_gate_metrics(
                        cal_ext[gk], np.concatenate((id_ext[gk], hi_ext[gk])), is_ood, pf, bh,
                    )
                sc_pc, sc_pt = _cal_zscore_max(
                    [cal_ext["rms_z"], cal_ext["k_eff"], cal_ext["r_hf"], cal_ext["early_peak"]],
                    [np.concatenate((id_ext[k], hi_ext[k])) for k in ("rms_z", "k_eff", "r_hf", "early_peak")],
                )
                gates["phys_combo_cal"] = _compute_gate_metrics(sc_pc, sc_pt, is_ood, pf, bh)

                sc_jc, sc_jt = _cal_zscore_max(
                    [cal_ext["hf_in"], cal_ext["rms_z"], cal_ext["k_eff"], cal_ext["r_hf"]],
                    [np.concatenate((id_ext[k], hi_ext[k])) for k in ("hf_in", "rms_z", "k_eff", "r_hf")],
                )
                gates["input_phys_combo_cal"] = _compute_gate_metrics(sc_jc, sc_jt, is_ood, pf, bh)
                return gates

            seed_id[ek] = _eval(hi_ext_clean)
            seed_amb[ek] = _eval(hi_amb_ext)

        part_a_ps["id_only_noise"][str(seed)] = seed_id
        part_a_ps["ambient_noise"][str(seed)] = seed_amb
        print(f"  [Task 3A] seed={seed} eta=0.10 Ambient AUROC: k_eff={seed_amb['0.1']['k_eff']['auroc']:.3f}, "
              f"r_hf={seed_amb['0.1']['r_hf']['auroc']:.3f}, physCombo={seed_amb['0.1']['phys_combo_cal']['auroc']:.3f}, "
              f"in+phys={seed_amb['0.1']['input_phys_combo_cal']['auroc']:.3f} | "
              f"ID-only in+phys={seed_id['0.1']['input_phys_combo_cal']['auroc']:.3f}", flush=True)

    def _agg_nested(ps_dict):
        first = ps_dict[str(SEEDS[0])]
        def _rec(node, path):
            if isinstance(node, dict):
                return {k: _rec(v, path + [k]) for k, v in node.items()}
            vals = []
            for s in SEEDS:
                cur = ps_dict[str(s)]
                for k in path:
                    cur = cur[k]
                if cur is not None:
                    vals.append(float(cur))
            return _seed_stats(vals) if len(vals) == len(SEEDS) else {"mean": float(np.mean(vals)) if vals else None}
        return _rec(first, [])

    part_a_sum = {
        "id_only_noise": _agg_nested(part_a_ps["id_only_noise"]),
        "ambient_noise": _agg_nested(part_a_ps["ambient_noise"]),
    }

    # ---------- Part B: Finite-Sample Exchangeable Self-Gated Mixture lambda(A) ----------
    print("[Task 3B] Exchangeable Val-anchored physics-adaptive mixture lambda(A) across alpha_hf in [0, 1]...", flush=True)
    part_b_ps = {}
    policies = (
        "locked_0.85",
        "phys_gate_lam0.0",
        "phys_gate_lam0.25",
        "spec_phys_gate_lam0.0",
        "spec_phys_gate_lam0.25",
    )

    for seed in SEEDS:
        root = REV_CACHE / "burgers" / f"seed_{seed}"
        v3_root = V3_CACHE / "burgers" / f"seed_{seed}"
        p_val = Protocol("burgers", seed, 64, v3_root, FROZEN_ROOT / f"seed_{seed}", device)
        val_path = p_val.data_path("val")
        p = Protocol("burgers", seed, 64, root, FROZEN_ROOT / f"seed_{seed}", device)
        model = p.open_frozen()
        cfg = p.config

        val_d = arrays(val_path)
        cal_d = arrays(root / "data/cal.npz")
        id_d = arrays(root / "data/id.npz")
        visc_d = arrays(root / "data/ood.npz")

        val_y = val_d["u64"][:, 1:].astype(np.float64)
        val_ext = _extract_physics_fingerprints(val_d["u64"][:, 0], val_d["nu"], val_y, model, p, cfg)
        cal_ext = _extract_physics_fingerprints(cal_d["u64"][:, 0], cal_d["nu"], cal_d["u64"][:, 1:].astype(np.float64), model, p, cfg)
        id_ext = _extract_physics_fingerprints(id_d["u64"][:, 0], id_d["nu"], id_d["u64"][:, 1:].astype(np.float64), model, p, cfg)
        visc_ext = _extract_physics_fingerprints(visc_d["u64"][:, 0], visc_d["nu"], visc_d["u64"][:, 1:].astype(np.float64), model, p, cfg)

        # Lock threshold & scale-multiplier anchors strictly on Val (zero Cal/Test/OOD leakage)
        def _val_stats(key):
            lv = np.log(np.maximum(val_ext[key], 1e-12))
            q95 = float(np.quantile(lv, 0.95))
            iqr = max(float(np.quantile(lv, 0.75) - np.quantile(lv, 0.25)), 1e-4)
            return q95, iqr

        rk_q95, rk_iqr = _val_stats("k_eff")
        rh_q95, rh_iqr = _val_stats("r_hf")
        rm_q95, rm_iqr = _val_stats("rms_z")
        in_q95, in_iqr = _val_stats("hf_in")

        lam_grid = np.array([0.0, 0.25, 0.50, 0.85], dtype=np.float64)
        q_vg = []
        n_val = len(val_y)
        for lg in lam_grid:
            sc_vg = _adaptive_lambda_scale(val_ext["env"], np.full(n_val, lg, dtype=np.float64), 1.0)
            s_vg = (np.abs(val_ext["pred"] - val_y) / sc_vg).reshape(n_val, -1).max(axis=1)
            q_vg.append(float(np.quantile(s_vg, 0.90)))
        c_grid = np.array(q_vg, dtype=np.float64) / max(float(q_vg[-1]), 1e-12)

        def _c_val_fn(lam_vec):
            return np.interp(lam_vec, lam_grid, c_grid)

        def _compute_lambda(ext, pol):
            n = len(ext["k_eff"])
            if pol == "locked_0.85":
                return np.full(n, 0.85, dtype=np.float64)
            z_k = (np.log(np.maximum(ext["k_eff"], 1e-12)) - rk_q95) / (1.5 * rk_iqr)
            z_r = (np.log(np.maximum(ext["r_hf"], 1e-12)) - rh_q95) / (1.5 * rh_iqr)
            z_rm = (np.log(np.maximum(ext["rms_z"], 1e-12)) - rm_q95) / (1.5 * rm_iqr)
            z_phys = np.maximum(0.0, np.maximum(z_rm, np.maximum(z_k, z_r)))
            if pol.startswith("spec_phys"):
                z_in = np.maximum(0.0, (np.log(np.maximum(ext["hf_in"], 1e-12)) - in_q95) / (1.5 * in_iqr))
                z_act = np.maximum(z_phys, z_in)
            else:
                z_act = z_phys
            lam_min = 0.25 if pol.endswith("0.25") else 0.0
            # Deterministic smooth annealing: z_act <= 0 -> 0.85; z_act > 0 -> smoothly drops toward lam_min
            gate_w = np.exp(-(z_act ** 2))
            return lam_min + (0.85 - lam_min) * gate_w

        # Calibrate q90 on Cal for each policy
        q_pol = {}
        cal_y = cal_d["u64"][:, 1:].astype(np.float64)
        for pol in policies:
            lam_c = _compute_lambda(cal_ext, pol)
            sc_c = _adaptive_lambda_scale(cal_ext["env"], lam_c, _c_val_fn(lam_c))
            scores_c = (np.abs(cal_ext["pred"] - cal_y) / sc_c).reshape(len(cal_y), -1).max(axis=1)
            q_pol[pol] = quantile(scores_c, 0.10)

        def _eval_pol(ext, y_true, pol):
            lam_t = _compute_lambda(ext, pol)
            sc_t = _adaptive_lambda_scale(ext["env"], lam_t, _c_val_fn(lam_t))
            n = len(y_true)
            scores_t = (np.abs(ext["pred"] - y_true) / sc_t).reshape(n, -1).max(axis=1)
            q_emp = value_at_coverage(scores_t, 0.90)
            return {
                "mean_lambda": float(lam_t.mean()),
                "coverage": float((scores_t <= q_pol[pol]).mean()),
                "actual_width": float((2.0 * q_pol[pol] * sc_t.reshape(n, -1).mean(axis=1)).mean()),
                "matched90_width": float(2.0 * q_emp * sc_t.mean()),
            }

        seed_b = {"splits": {}, "alpha_hf_sweep": {}}
        for sp_l, ext_s, y_s in (("ID", id_ext, id_d["u64"][:, 1:].astype(np.float64)),
                                 ("Visc-OOD", visc_ext, visc_d["u64"][:, 1:].astype(np.float64))):
            seed_b["splits"][sp_l] = {pol: _eval_pol(ext_s, y_s, pol) for pol in policies}

        # Continuous alpha_hf sweep using same disjoint stream 6901 as rev2_task3_burgers.py
        n_test = 500
        rng = np.random.default_rng(np.random.SeedSequence([seed, 6901]))
        vr = cfg["viscosity_id"]
        nu_base = np.exp(rng.uniform(*np.log(vr), size=n_test))
        modes16 = np.arange(1, 17, dtype=np.float64)
        raw_c = (rng.normal(size=(n_test, 16)) + 1j * rng.normal(size=(n_test, 16))) / (modes16 ** 1.35)
        amp_base = rng.uniform(0.55, 1.25, size=(n_test, 1))
        mean_base = rng.uniform(-0.4, 0.4, size=(n_test, 1))

        for ahf in ALPHAS_HF:
            c = raw_c.copy()
            c[:, 8:] *= ahf
            c *= amp_base / np.sqrt(2.0 * np.sum(np.abs(c) ** 2, axis=1, keepdims=True))
            coeff = np.concatenate((mean_base, c), axis=1)
            u64 = _solve_burgers_batch(coeff, nu_base, cfg)
            ext_a = _extract_physics_fingerprints(u64[:, 0], nu_base, u64[:, 1:].astype(np.float64), model, p, cfg)
            seed_b["alpha_hf_sweep"][str(ahf)] = {
                pol: _eval_pol(ext_a, u64[:, 1:].astype(np.float64), pol) for pol in policies
            }

        # Also test noisy ambient High-OOD at eta=0.05
        hi_clean = arrays(root / "data/ood_high.npz")
        rng_h5 = np.random.default_rng(np.random.SeedSequence([seed, 6950 + 2 * 3 + 2]))
        hi_n5 = _noisy_burgers_split(hi_clean["u64"][:, 0], hi_clean["nu"], 0.05, cfg, rng_h5)
        ext_n5 = _extract_physics_fingerprints(hi_n5[:, 0], hi_clean["nu"], hi_n5[:, 1:].astype(np.float64), model, p, cfg)
        seed_b["splits"]["High_eta0.05"] = {pol: _eval_pol(ext_n5, hi_n5[:, 1:].astype(np.float64), pol) for pol in policies}

        part_b_ps[str(seed)] = seed_b
        print(f"  [Task 3B] seed={seed} | ID cov/w: locked={100*seed_b['splits']['ID']['locked_0.85']['coverage']:.1f}%/{seed_b['splits']['ID']['locked_0.85']['actual_width']:.4f}, "
              f"phys0={100*seed_b['splits']['ID']['phys_gate_lam0.0']['coverage']:.1f}%/{seed_b['splits']['ID']['phys_gate_lam0.0']['actual_width']:.4f} | "
              f"ahf=1.0 cov: locked={100*seed_b['alpha_hf_sweep']['1.0']['locked_0.85']['coverage']:.1f}%, "
              f"phys0={100*seed_b['alpha_hf_sweep']['1.0']['phys_gate_lam0.0']['coverage']:.1f}%, "
              f"spec+phys0={100*seed_b['alpha_hf_sweep']['1.0']['spec_phys_gate_lam0.0']['coverage']:.1f}%", flush=True)

    part_b_sum = _agg_nested(part_b_ps)

    out = {
        "seeds": list(SEEDS),
        "elapsed_s": perf_counter() - t0,
        "etas_noise": list(ETAS_NOISE),
        "alphas_hf": list(ALPHAS_HF),
        "part_a_fingerprints": {"summary": part_a_sum, "per_seed": part_a_ps},
        "part_b_adaptive_lambda": {"summary": part_b_sum, "per_seed": part_b_ps},
    }
    path = OUT / "task3_physics_gate.json"
    path.write_text(json.dumps(finite_json(out), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[Rev4 Task 3] Done in {out['elapsed_s']:.1f}s -> {path}", flush=True)
    return out


if __name__ == "__main__":
    run_task3_physics_gate()
