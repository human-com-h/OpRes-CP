"""Task 4 (P0): Theorem 2 Quantitative Verification of Density-Ratio Error delta_w.

Computes empirical delta_w = 0.5 * E_{P}[|w_tilde - w^*|] across all 5 formal seeds
[42, 137, 314, 271, 811] comparing Compact Physical Summaries vs. Full-Grid features
under both Transductive and strict Split-Fit (Theorem 2) protocols, using exact
closed-form generator density ratios w^*:
  1. Darcy Contrast OOD-A (full support overlap [0.4, 2.0]; exact w^*(sigma) = p_Q(sigma)/p_P(sigma))
  2. Burgers Viscosity OOD (overlap-conditioned Q|_{nu >= 0.02}, exact delta_suf = 0 on overlap;
     plus exact analytical out-of-support mass Q(nu < 0.02) = 70.36%)
  3. Allen-Cahn Reaction OOD (overlap-conditioned Q|_{rho <= 1.1}, exact delta_suf = 0 on overlap;
     plus exact analytical out-of-support mass Q(rho > 1.1) = 66.67%)

Outputs written to results/phase3_revision_v2/task4_delta_w.json.
"""

import json
import math
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from darcy_day15 import arrays, oracle_weights, weighted_quantile
from phase2d_data import Protocol
from phase3_common import OP, finite_json, summarize
from phase3_summarize import SEEDS, STAGE, result_root, stat


OUT = Path("results/phase3_revision_v2")
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


def _fit_ratio_fn(x_src, x_tgt, c_reg=1.0, clip=(0.02, 50.0)):
    est = make_pipeline(StandardScaler(), LogisticRegression(C=c_reg, max_iter=1000))
    x = np.concatenate((x_src, x_tgt), axis=0)
    y = np.concatenate((np.zeros(len(x_src), dtype=int), np.ones(len(x_tgt), dtype=int)))
    est.fit(x, y)
    prior = len(x_src) / float(len(x_tgt))
    return lambda arr: np.clip(np.exp(est.decision_function(arr)) * prior, clip[0], clip[1])


def _delta_w_tv(w_hat_p, w_star_p):
    """Exact empirical 0.5 * E_P[|w_tilde - w^*|] with w_tilde = w_hat / E_P[w_hat]."""
    w_tilde = w_hat_p / max(float(np.mean(w_hat_p)), 1e-30)
    return float(0.5 * np.mean(np.abs(w_tilde - w_star_p)))


def _eval_wcp_and_delta_w(xc, xt, x_id, cal_op, tgt_op, w_star_cal, w_star_id):
    """Evaluate Transductive and Strict Split-Fit WCP + delta_w on held-out P (id_test) and eval Cal."""
    # 1) Transductive: fit on (Cal[500], Target[500])
    wf_tr = _fit_ratio_fn(xc, xt)
    wc_tr = wf_tr(xc)
    wt_tr = wf_tr(xt)
    wid_tr = wf_tr(x_id)
    q_tr, pt_tr, ess_tr = weighted_quantile(cal_op[:, 0], wc_tr, wt_tr, 0.10)
    row_tr = summarize(tgt_op, q_tr, 0.10)
    tr_out = {
        "delta_w_heldout_P": _delta_w_tv(wid_tr, w_star_id),
        "delta_w_insample_cal": _delta_w_tv(wc_tr, w_star_cal),
        "coverage": row_tr["coverage"],
        "infinite_rate": row_tr["infinite_rate"],
        "finite_n": row_tr["finite_n"],
        "finite_coverage": row_tr["finite_coverage"],
        "finite_width": row_tr["finite_width"],
        "ess": ess_tr,
        "p_test_p50": float(np.median(pt_tr)),
        "w_hat_id_mean": float(np.mean(wid_tr)),
        "w_star_id_mean": float(np.mean(w_star_id)),
    }

    # 2) Strict Split-Fit (Theorem 2 premise): fit on (Cal[:250], Target_other_half),
    #    evaluate delta_w on disjoint Cal[250:] and held-out ID[500], and WCP on Target_this_half.
    half = len(xc) // 2
    th = len(xt) // 2
    q_sp = np.empty(len(xt), dtype=float)
    pt_sp = np.empty(len(xt), dtype=float)
    ess_folds, dw_cal_folds, dw_id_folds = [], [], []
    for fit_sl, eval_sl in ((slice(th, None), slice(0, th)), (slice(0, th), slice(th, None))):
        wf = _fit_ratio_fn(xc[:half], xt[fit_sl])
        wc_eval = wf(xc[half:])
        wt_eval = wf(xt[eval_sl])
        wid_eval = wf(x_id)
        dw_cal_folds.append(_delta_w_tv(wc_eval, w_star_cal[half:]))
        dw_id_folds.append(_delta_w_tv(wid_eval, w_star_id))
        qf, pf, essf = weighted_quantile(cal_op[half:, 0], wc_eval, wt_eval, 0.10)
        q_sp[eval_sl] = qf
        pt_sp[eval_sl] = pf
        ess_folds.append(essf)
    row_sp = summarize(tgt_op, q_sp, 0.10)
    sp_out = {
        "delta_w_heldout_P": float(np.mean(dw_id_folds)),
        "delta_w_disjoint_cal": float(np.mean(dw_cal_folds)),
        "coverage": row_sp["coverage"],
        "infinite_rate": row_sp["infinite_rate"],
        "finite_n": row_sp["finite_n"],
        "finite_coverage": row_sp["finite_coverage"],
        "finite_width": row_sp["finite_width"],
        "ess": float(np.mean(ess_folds)),
        "p_test_p50": float(np.median(pt_sp)),
        "thm2_lower_bound_no_suf": max(0.0, 0.90 - float(np.mean(dw_id_folds))),
    }
    return {"transductive": tr_out, "split_fit": sp_out}


def run_task4():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = perf_counter()
    results = {"darcy_ood_a": {}, "burgers_visc_overlap": {}, "allen_cahn_react_overlap": {}}

    # ---------- 1. Darcy Contrast OOD-A (Full overlap, exact w*(sigma)) ----------
    print("[Task 4] Darcy Contrast OOD-A (full overlap [0.4, 2.0])...", flush=True)
    for feat_name, feat_key in (("Summary (p=5)", "_input"), ("Full-Grid (d=4096)", "_full"), ("Oracle 1D sigma", "_oracle_1d")):
        per_seed = {}
        for seed in SEEDS:
            root3 = result_root("darcy", seed)
            c_root = REV_CACHE / f"darcy/seed_{seed}/data"
            cal_d = arrays(c_root / "cal.npz")
            id_d = arrays(c_root / "id_test.npz")
            ood_d = arrays(c_root / "ood_a.npz")
            cfg = json.loads(Path("configs/phase0_darcy.json").read_text(encoding="utf-8"))

            w_star_cal = oracle_weights(cal_d["sigma"], cfg)
            w_star_id = oracle_weights(id_d["sigma"], cfg)

            with np.load(root3 / "fields_grid64_cal.npz") as zc, \
                 np.load(root3 / "fields_grid64_id_test_64.npz") as zi, \
                 np.load(root3 / "fields_grid64_ood_a_64.npz") as zt:
                cal_op, tgt_op = zc[OP], zt[OP]
                if feat_key == "_oracle_1d":
                    xc = cal_d["sigma"][:, None]
                    x_id = id_d["sigma"][:, None]
                    xt = ood_d["sigma"][:, None]
                elif feat_key == "_full":
                    xc = zc["_full"].astype(np.float32)
                    x_id = zi["_full"].astype(np.float32)
                    xt = zt["_full"].astype(np.float32)
                else:
                    xc = zc[feat_key]
                    x_id = zi[feat_key]
                    xt = zt[feat_key]

            per_seed[str(seed)] = _eval_wcp_and_delta_w(xc, xt, x_id, cal_op, tgt_op, w_star_cal, w_star_id)
        results["darcy_ood_a"][feat_name] = per_seed

    # ---------- 2. Burgers Viscosity OOD (Overlap-conditioned Q|_{nu >= 0.02}, exact delta_suf = 0) ----------
    print("[Task 4] Burgers Viscosity OOD (overlap nu in [0.02, 0.028], exact delta_suf=0)...", flush=True)
    # Exact analytical out-of-support mass Q(nu < 0.02)
    burg_oos_mass = math.log(0.02 / 0.009) / math.log(0.028 / 0.009)
    burg_ov_scale = math.log(0.07 / 0.02) / math.log(0.028 / 0.02)
    for feat_name, feat_kind in (("Summary (p=3)", "summary"), ("Full-Grid (d=65)", "full")):
        per_seed = {}
        for seed in SEEDS:
            root3 = result_root("burgers", seed)
            c_root = REV_CACHE / f"burgers/seed_{seed}/data"
            cal_d = arrays(c_root / "cal.npz")
            id_d = arrays(c_root / "id.npz")
            ood_d = arrays(c_root / "ood.npz")
            ov_mask = ood_d["nu"] >= 0.02

            w_star_cal = np.where(cal_d["nu"] <= 0.028, burg_ov_scale, 0.0)
            w_star_id = np.where(id_d["nu"] <= 0.028, burg_ov_scale, 0.0)

            with np.load(root3 / "fields_cal.npz") as zc, \
                 np.load(root3 / "fields_id.npz") as zi, \
                 np.load(root3 / "fields_ood.npz") as zt:
                cal_op = zc[OP]
                tgt_op = zt[OP][ov_mask]
                if feat_kind == "summary":
                    xc = zc["_input"]
                    x_id = zi["_input"]
                    xt = zt["_input"][ov_mask]
                else:
                    xc = np.column_stack((cal_d["u64"][:, 0], cal_d["nu"])).astype(np.float32)
                    x_id = np.column_stack((id_d["u64"][:, 0], id_d["nu"])).astype(np.float32)
                    xt = np.column_stack((ood_d["u64"][ov_mask, 0], ood_d["nu"][ov_mask])).astype(np.float32)

            per_seed[str(seed)] = _eval_wcp_and_delta_w(xc, xt, x_id, cal_op, tgt_op, w_star_cal, w_star_id)
        results["burgers_visc_overlap"][feat_name] = per_seed

    # ---------- 3. Allen-Cahn Reaction OOD (Overlap-conditioned Q|_{rho <= 1.1}, exact delta_suf = 0) ----------
    print("[Task 4] Allen-Cahn Reaction OOD (overlap rho in [0.9, 1.1], exact delta_suf=0)...", flush=True)
    ac_oos_mass = (1.5 - 1.1) / (1.5 - 0.9)  # 2/3
    ac_ov_scale = (1.1 - 0.5) / (1.1 - 0.9)  # 3.0
    for feat_name, feat_key in (("Summary (p=4)", "_input"), ("Full-Grid (d=4098)", "_full")):
        per_seed = {}
        for seed in SEEDS:
            root3 = result_root("allen_cahn", seed)
            c_root = REV_CACHE / f"allen_cahn/seed_{seed}/data"
            cal_d = arrays(c_root / "cal.npz")
            id_d = arrays(c_root / "id.npz")
            ood_d = arrays(c_root / "ood.npz")
            ov_mask = ood_d["rho"] <= 1.1

            w_star_cal = np.where(cal_d["rho"] >= 0.9, ac_ov_scale, 0.0)
            w_star_id = np.where(id_d["rho"] >= 0.9, ac_ov_scale, 0.0)

            with np.load(root3 / "fields_short_cal.npz") as zc, \
                 np.load(root3 / "fields_short_id.npz") as zi, \
                 np.load(root3 / "fields_short_ood.npz") as zt:
                cal_op = zc[OP]
                tgt_op = zt[OP][ov_mask]
                xc = zc[feat_key].astype(np.float32) if feat_key == "_full" else zc[feat_key]
                x_id = zi[feat_key].astype(np.float32) if feat_key == "_full" else zi[feat_key]
                xt = (zt[feat_key].astype(np.float32) if feat_key == "_full" else zt[feat_key])[ov_mask]

            per_seed[str(seed)] = _eval_wcp_and_delta_w(xc, xt, x_id, cal_op, tgt_op, w_star_cal, w_star_id)
        results["allen_cahn_react_overlap"][feat_name] = per_seed

    # Aggregate all across 5 seeds
    summary = {}
    for bench, feats in results.items():
        summary[bench] = {}
        for fname, ps in feats.items():
            summary[bench][fname] = {}
            for proto in ("transductive", "split_fit"):
                keys = list(ps[str(SEEDS[0])][proto].keys())
                agg = {}
                for k in keys:
                    vals = [ps[str(s)][proto][k] for s in SEEDS if ps[str(s)][proto][k] is not None]
                    agg[k] = _seed_stats(vals) if len(vals) == len(SEEDS) else {"mean": float(np.mean(vals)) if vals else None}
                summary[bench][fname][proto] = agg
            tr = summary[bench][fname]["transductive"]
            sp = summary[bench][fname]["split_fit"]
            print(f"  [Task 4] {bench:24s} | {fname:18s} | "
                  f"Transd dw_P={tr['delta_w_heldout_P']['mean']:.4f} (inCal={tr['delta_w_insample_cal']['mean']:.4f}) ESS={tr['ess']['mean']:.1f} inf={100*tr['infinite_rate']['mean']:.1f}% | "
                  f"Split dw_P={sp['delta_w_heldout_P']['mean']:.4f} ESS={sp['ess']['mean']:.1f} cov={100*sp['coverage']['mean']:.2f}%",
                  flush=True)

    out = {
        "seeds": list(SEEDS),
        "elapsed_s": perf_counter() - t0,
        "analytical_support": {
            "darcy_ood_a_oos_mass": 0.0,
            "burgers_visc_oos_mass": burg_oos_mass,
            "allen_cahn_react_oos_mass": ac_oos_mass,
            "exact_delta_suf_zero_on_overlap": ["burgers_visc_overlap", "allen_cahn_react_overlap"],
        },
        "summary": summary,
        "per_seed": results,
    }
    path = OUT / "task4_delta_w.json"
    path.write_text(json.dumps(finite_json(out), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[Task 4] Done in {out['elapsed_s']:.1f}s -> {path}", flush=True)
    return out


if __name__ == "__main__":
    run_task4()
