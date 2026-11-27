"""Revision v3 Task 2 (P0): Leakage-Free Val-Fit Exact RLCP-Band & Consolidated Fair LSCI Audit.

Fixes the in-sample denominator leakage of naive Cal-fit LSCI bands (which yielded
85.92% / 80.84% / 82.80% undercoverage in rev2) by fitting the locally weighted
coordinatewise scale function sigma_LSCI^val(x) strictly on the independent Val stream
(250 fields/trajectories per seed, hash-verified against frozen reference manifests),
then scoring the independent 500-case Cal and 500-case Test streams and applying the
exact RLCP locally weighted max-conformal quantile (Hore & Barber 2025).

Also merges the already-verified LOO RLCP-Band, Native-32, OpRes Box->32, and
OpRes Cal-32 results from results/phase3_revision_v2/task2_lsci_fair.json into a
single clean JSON artifact: results/phase3_revision_v3/task2_lsci_exact.json.

Strictly read-only against results/phase2_kaggle_private/.
Formal seeds locked to [42, 137, 314, 271, 811].
"""

import json
import math
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.spatial.distance import cdist
import torch

from darcy_day15 import arrays, value_at_coverage, weighted_quantile
from phase2d import batches, take
from phase2d_data import Protocol
from phase2d_statistics import LocalSlicedCP, flat
from phase3_common import finite_json
from phase3_summarize import PHASE2_OP, SEEDS, STAGE, phase2_metrics, stat


OUT = Path("results/phase3_revision_v3")
V3_CACHE = OUT / "cache"
REV_CACHE = Path("results/phase3_revision/cache")
REV2_TASK2 = Path("results/phase3_revision_v2/task2_lsci_fair.json")
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


def _predict_all(protocol, model, data, grid, frames, batch=16):
    inputs, truth = protocol.split(data, grid, frames)
    preds = []
    for sl in batches(len(truth), batch):
        preds.append(protocol.predict(model, take(inputs, sl), grid, frames))
    return inputs, truth, np.concatenate(preds, axis=0)


class ValFittedLSCIScale:
    """Fixed scale rule sigma_LSCI^val(x) fitted strictly on independent Val residuals."""

    def __init__(self, r_val_flat, x_val, rho_rel=0.10):
        self.r_val_sq = (r_val_flat.astype(np.float64)) ** 2
        self.rho_rel = float(rho_rel)
        self.center = x_val.mean(axis=0)
        sd = x_val.std(axis=0)
        self.spread = np.where(sd > 1e-8, sd, 1.0)
        self.ref_x = (x_val - self.center) / self.spread
        dists = cdist(self.ref_x, self.ref_x, "sqeuclidean")
        pos = dists[dists > 0]
        self.bandwidth = float(np.median(pos)) if len(pos) else 1.0

    def compute_scales(self, x_query, device=None):
        """Return (scale_flat, scale_means) of shape (n_q, d) and (n_q,)."""
        xq = (x_query - self.center) / self.spread
        dists = cdist(xq, self.ref_x, "sqeuclidean")
        w = np.exp(-(dists - dists.min(axis=1, keepdims=True)) / self.bandwidth)
        p = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-30)

        n_q = len(xq)
        d = self.r_val_sq.shape[1]
        use_gpu = (device is not None and device.type == "cuda" and d > 4096)
        if use_gpu:
            r2_t = torch.as_tensor(self.r_val_sq, dtype=torch.float32, device=device)
            p_t = torch.as_tensor(p, dtype=torch.float32, device=device)
            with torch.inference_mode():
                s0_sq = p_t @ r2_t
                rms_sq = s0_sq.mean(dim=1, keepdim=True)
                sig_t = torch.sqrt(s0_sq + (self.rho_rel ** 2) * rms_sq).clamp_min(1e-12)
                return sig_t, sig_t.mean(dim=1).cpu().numpy().astype(np.float64)

        s0_sq = p @ self.r_val_sq
        rms_sq = s0_sq.mean(axis=1, keepdims=True)
        sig = np.maximum(np.sqrt(s0_sq + (self.rho_rel ** 2) * rms_sq), 1e-12)
        return sig, sig.mean(axis=1)


def _eval_valfit_rlcp_band(val_scale_rule, x_cal, r_cal_flat, weights, x_test, r_test_flat, alpha=0.10, device=None):
    """Evaluate exact RLCP locally weighted max-conformal band using fixed Val-fitted scale."""
    sig_cal, _ = val_scale_rule.compute_scales(x_cal, device=device)
    sig_test, test_scale_means = val_scale_rule.compute_scales(x_test, device=device)

    if isinstance(sig_cal, torch.Tensor):
        with torch.inference_mode():
            abs_cal_t = torch.as_tensor(np.abs(r_cal_flat), dtype=torch.float32, device=device)
            abs_test_t = torch.as_tensor(np.abs(r_test_flat), dtype=torch.float32, device=device)
            s_cal = (abs_cal_t / sig_cal).amax(dim=1).cpu().numpy().astype(np.float64)
            s_test = (abs_test_t / sig_test).amax(dim=1).cpu().numpy().astype(np.float64)
    else:
        s_cal = (np.abs(r_cal_flat).astype(np.float64) / sig_cal).max(axis=1)
        s_test = (np.abs(r_test_flat).astype(np.float64) / sig_test).max(axis=1)

    n_test = len(s_test)
    cov = np.zeros(n_test, dtype=bool)
    is_inf = np.zeros(n_test, dtype=bool)
    actual_width = np.full(n_test, np.inf, dtype=np.float64)

    for i, w in enumerate(weights):
        wc = w[:-1]
        wt = float(w[-1])
        cm = float(wc.sum())
        if cm <= 0.0 or wt > alpha * (cm + wt) + 1e-12:
            is_inf[i] = True
            cov[i] = True
            continue
        q_arr, _, _ = weighted_quantile(s_cal, wc, np.array([wt]), alpha)
        q = float(q_arr[0])
        if not np.isfinite(q):
            is_inf[i] = True
            cov[i] = True
        else:
            cov[i] = s_test[i] <= q
            actual_width[i] = 2.0 * q * float(test_scale_means[i])

    fin = ~is_inf
    q90_emp = value_at_coverage(s_test, 1.0 - alpha)
    matched90_width = float(2.0 * q90_emp * test_scale_means.mean())
    return {
        "coverage": float(cov.mean()),
        "infinite_rate": float(is_inf.mean()),
        "finite_n": int(fin.sum()),
        "finite_coverage": float(cov[fin].mean()) if fin.any() else None,
        "actual_width": float(actual_width.mean()) if fin.all() else "inf",
        "finite_actual_width": float(actual_width[fin].mean()) if fin.any() else None,
        "matched90_width": matched90_width,
    }


def run_synthetic_unit_test(n_trials=10):
    """Verify Val-Fit Exact RLCP-Band achieves ~90% i.i.d. marginal simultaneous coverage."""
    covs, infs = [], []
    for t in range(n_trials):
        rng = np.random.default_rng(2027 + t)
        n_val, n_cal, n_test, d, p = 250, 500, 600, 64, 5
        x_val = rng.normal(size=(n_val, p))
        x_cal = rng.normal(size=(n_cal, p))
        x_test = rng.normal(size=(n_test, p))
        r_val = rng.normal(size=(n_val, d)) * (0.5 + np.abs(x_val[:, :1]))
        r_cal = rng.normal(size=(n_cal, d)) * (0.5 + np.abs(x_cal[:, :1]))
        r_test = rng.normal(size=(n_test, d)) * (0.5 + np.abs(x_test[:, :1]))

        val_rule = ValFittedLSCIScale(r_val, x_val, rho_rel=0.10)
        fit = LocalSlicedCP(r_cal, x_cal, seed=54321 + t)
        w = fit.prepare(x_test)
        res = _eval_valfit_rlcp_band(val_rule, x_cal, r_cal, w, x_test, r_test, alpha=0.10)
        covs.append(res["coverage"])
        infs.append(res["infinite_rate"])
    mean_cov = float(np.mean(covs))
    sd_cov = float(np.std(covs, ddof=1))
    mean_inf = float(np.mean(infs))
    print(f"[Task 2 Unit Check] {n_trials}-trial i.i.d. Val-Fit Exact RLCP-Band cov={100*mean_cov:.2f}% +/- {100*sd_cov:.2f}% "
          f"(inf={100*mean_inf:.2f}%)", flush=True)
    assert 0.88 <= mean_cov <= 0.92, f"Val-Fit RLCP-Band mean coverage out of [0.88, 0.92]: {mean_cov}"
    return {
        "n_trials": n_trials,
        "iid_valfit_rlcp_band_cov90_mean": mean_cov,
        "iid_valfit_rlcp_band_cov90_sd": sd_cov,
        "iid_valfit_rlcp_band_inf90_mean": mean_inf,
    }


def run_task2(device=None):
    OUT.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = perf_counter()

    unit = run_synthetic_unit_test()
    rev2_data = json.loads(REV2_TASK2.read_text(encoding="utf-8"))

    specs = (
        ("darcy", None, (("id_test", "id_test_64"), ("ood_a", "ood_a_64"), ("ood_b", "ood_b_64"))),
        ("burgers", 16, (("id", "id_64"), ("ood", "ood_64"), ("ood_high", "ood_high_64"))),
        ("allen_cahn", 20, (("id", "id_64_20"), ("ood", "ood_64_20"))),
    )

    per_pde = {}
    for pde, _, split_pairs in specs:
        per_seed = {}
        for seed in SEEDS:
            rev_root = REV_CACHE / pde / f"seed_{seed}"
            v3_root = V3_CACHE / pde / f"seed_{seed}"
            # Use v3_root only for generating/caching Val (Protocol.data_path verifies exact stream IDs)
            p_val = Protocol(pde, seed, 64, v3_root, FROZEN_ROOT / f"seed_{seed}", device)
            val_path = p_val.data_path("val")

            p = Protocol(pde, seed, 64, rev_root, FROZEN_ROOT / f"seed_{seed}", device)
            model = p.open_frozen()
            cal_frames = p.cal_frames if pde == "allen_cahn" else p.frames
            batch = 16 if pde != "allen_cahn" else 8

            val_data = arrays(val_path)
            xval, yval, pval = _predict_all(p, model, val_data, 64, cal_frames, batch=batch)
            summary_val = p.summaries(xval)
            r_val_flat = flat(yval - pval).astype(np.float32)
            val_rule = ValFittedLSCIScale(r_val_flat, summary_val, rho_rel=0.10)

            cal_data = arrays(rev_root / "data/cal.npz")
            xcal, ycal, pcal = _predict_all(p, model, cal_data, 64, cal_frames, batch=batch)
            summary_cal = p.summaries(xcal)
            r_cal = (ycal - pcal).astype(np.float32)
            r_cal_flat = flat(r_cal)

            seed_splits = {}
            for sp_key, p2d_label in split_pairs:
                data = arrays(rev_root / f"data/{sp_key}.npz")
                xt, yt, pt = _predict_all(p, model, data, 64, cal_frames, batch=batch)
                cov_x = p.summaries(xt)
                r_test_flat = flat(yt - pt).astype(np.float64)

                # Identical RLCP kernel weights & seed as rev2_task2_lsci.py
                fit = LocalSlicedCP(r_cal, summary_cal, seed + 9029)
                w = fit.prepare(cov_x)

                vf_band = _eval_valfit_rlcp_band(
                    val_rule, summary_cal, r_cal_flat, w, cov_x, r_test_flat, alpha=0.10, device=device
                )
                r2_sp = rev2_data["pdes"][pde]["per_seed"][str(seed)][p2d_label]
                op_mw = r2_sp["opres_whole_field"]["matched90_width"]
                seed_splits[p2d_label] = {
                    "valfit_rlcp_band": vf_band,
                    "width_shrink_pct_vs_valfit_matched90": 100.0 * (1.0 - op_mw / vf_band["matched90_width"]),
                    "loo_rlcp_band": {
                        "coverage": r2_sp["lsci_cal_band"]["loo_coverage"],
                        "infinite_rate": r2_sp["lsci_cal_band"]["infinite_rate"],
                        "finite_n": r2_sp["lsci_cal_band"]["finite_n"],
                        "finite_actual_width": r2_sp["lsci_cal_band"]["loo_finite_actual_width"],
                        "matched90_width": r2_sp["lsci_cal_band"]["matched90_width"],
                    },
                    "opres_whole_field": r2_sp["opres_whole_field"],
                    "proj32": r2_sp["proj32"],
                }
                fw = vf_band["finite_actual_width"]
                fw_s = f"{fw:.4f}" if fw is not None else "inf"
                print(f"  [Task 2] {pde:10s} seed={seed} {p2d_label:12s} | ValFit cov={100*vf_band['coverage']:.2f}% "
                      f"(inf={100*vf_band['infinite_rate']:.2f}%, fw={fw_s}, mw={vf_band['matched90_width']:.4f}) | "
                      f"LOO cov={100*r2_sp['lsci_cal_band']['loo_coverage']:.2f}% | "
                      f"OpRes mw={op_mw:.4f}", flush=True)
            per_seed[str(seed)] = seed_splits

        summary = {}
        for _, p2d_label in split_pairs:
            recs = [per_seed[str(s)][p2d_label] for s in SEEDS]

            def _agg(keys):
                vals = []
                for r in recs:
                    cur = r
                    for k in keys:
                        cur = cur[k]
                    if cur is not None and isinstance(cur, (int, float)) and math.isfinite(cur):
                        vals.append(float(cur))
                if len(vals) == len(SEEDS):
                    return _seed_stats(vals)
                return {
                    "mean": float(np.mean(vals)) if vals else None,
                    "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                    "n_defined": len(vals),
                }

            summary[p2d_label] = {
                "valfit_rlcp_band": {
                    k: _agg(["valfit_rlcp_band", k])
                    for k in ("coverage", "infinite_rate", "finite_n", "finite_coverage",
                              "finite_actual_width", "matched90_width")
                },
                "width_shrink_pct_vs_valfit_matched90": _agg(["width_shrink_pct_vs_valfit_matched90"]),
                "loo_rlcp_band": {
                    k: _agg(["loo_rlcp_band", k])
                    for k in ("coverage", "infinite_rate", "finite_n", "finite_actual_width", "matched90_width")
                },
                "opres_whole_field": {
                    k: _agg(["opres_whole_field", k])
                    for k in ("coverage", "actual_width", "matched90_width")
                },
                "proj32": rev2_data["pdes"][pde]["summary"][p2d_label]["proj32"],
            }
        per_pde[pde] = {"summary": summary, "per_seed": per_seed}

    out = {
        "seeds": list(SEEDS),
        "elapsed_s": perf_counter() - t0,
        "unit_checks": unit,
        "pdes": per_pde,
    }
    path = OUT / "task2_lsci_exact.json"
    path.write_text(json.dumps(finite_json(out), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[Task 2] Done in {out['elapsed_s']:.1f}s -> {path}", flush=True)
    return out


if __name__ == "__main__":
    run_task2()
