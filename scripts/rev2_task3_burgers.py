"""Task 3 (P0): Burgers Continuous High-Frequency Shift & Noisy Spectral Overlap Stress Test.

Zero backbone retraining; strictly read-only against results/phase2_kaggle_private/.
Evaluates across all 5 formal seeds [42, 137, 314, 271, 811]:
  Part A: Continuous high-frequency injection alpha_hf in {0.0, 0.05, 0.10, 0.25, 0.50, 1.0}
          comparing OpRes-CP (lam=0.85, ell=0.08), OpRes-CP (lam=0.25, ell=0.32),
          Physics-Scalar (lam=0), and 5-Member Ensemble (mean).
  Part B: Spectrally overlapping noisy stream with relative RMS noise
          eta_noise in {0.0, 0.02, 0.05, 0.10, 0.20} injected into modes 9-16:
          comparing (i) Pure input spectral gate vs (ii) Input + physics lifted-scale joint gate
          on AUROC, FPR@95%TPR, TPR@1%FPR, TPR@5%FPR, point-failure AURC,
          miscoverage AURC, and accepted-set coverage.
          Reports BOTH Protocol A (noise on ID Cal+Test only, literal to prompt) and
          Protocol B (common ambient background noise on Cal+ID+OOD-high).

Outputs written to results/phase3_revision_v2/task3_burgers_stress.json.
"""

import json
import math
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import torch

from burgers_phase1_data import solve as burgers_solve
import burgers_phase1_ops as burg_ops
from darcy_day15 import arrays, estimated_weights, value_at_coverage
from darcy_phase0_eval import quantile
from phase2d import batches, take
from phase2d_burgers_diagnosis import features as burg_features
from phase2d_data import Protocol
from phase2d_statistics import relative_l2
from phase3_common import finite_json, mix as p3_mix, rms as p3_rms
from phase3_summarize import DOWNLOADS, SEEDS, STAGE, stat


OUT = Path("results/phase3_revision_v2")
REV_CACHE = Path("results/phase3_revision/cache")
FROZEN_ROOT = STAGE / "phase2d_frozen"
ALPHAS_HF = (0.0, 0.05, 0.10, 0.25, 0.50, 1.0)
ETAS_NOISE = (0.0, 0.02, 0.05, 0.10, 0.20)


def _seed_stats(values):
    _, avg, sd, lo, hi = stat(values)
    return {
        "mean": float(avg),
        "sd": float(sd),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "per_seed": {str(s): float(v) for s, v in zip(SEEDS, values)},
    }


def _load_burgers_ensemble(seed, protocol, device):
    models = [protocol.open_frozen()]
    for i in range(1, 5):
        ck = DOWNLOADS / f"2db/seed_{seed}/results/phase2d/burgers/seed_{seed}/grid64/ensemble_{i}/FNO/fno.pt"
        m = protocol.make_model().to(device)
        m.load_state_dict(torch.load(ck, map_location=device, weights_only=True)["model"])
        models.append(m.eval())
    m_json = json.loads((DOWNLOADS / f"2db/seed_{seed}/results/phase2d/burgers/seed_{seed}/grid64/metrics.json").read_text(encoding="utf-8"))
    ens_floor = float(m_json["variance_floor"]["ensemble"])
    return models, ens_floor


def _predict_ens(protocol, models, u0, nu, batch=64):
    n = len(u0)
    p0_l, mean_l, std_l = [], [], []
    for sl in batches(n, batch):
        inp = (u0[sl].astype(np.float32), nu[sl].astype(np.float32))
        stack = np.stack([protocol.predict(m, inp, 64, protocol.frames).astype(np.float64) for m in models], axis=0)
        p0_l.append(stack[0])
        mean_l.append(stack.mean(axis=0))
        std_l.append(stack.std(axis=0, ddof=1))
    return np.concatenate(p0_l), np.concatenate(mean_l), np.concatenate(std_l)


def _solve_burgers_batch(coeff, nu, cfg):
    parts = []
    for i in range(0, len(nu), 64):
        sl = slice(i, i + 64)
        u, _ = burgers_solve(coeff[sl], nu[sl], cfg, [64])
        parts.append(u[64].astype(np.float32))
    return np.concatenate(parts, axis=0)


def _eval_band(center, truth, scale, q90):
    sc = np.maximum(scale, 1e-12)
    err = np.abs(center - truth)
    scores = (err / sc).reshape(len(truth), -1).max(axis=1)
    mean_s = sc.reshape(len(truth), -1).mean(axis=1)
    cov = float((scores <= q90).mean())
    act_w = float((2.0 * q90 * mean_s).mean())
    q_emp = value_at_coverage(scores, 0.90)
    matched_w = float(2.0 * q_emp * mean_s.mean())
    rel_err = float(relative_l2(center, truth).mean())
    return {
        "coverage": cov,
        "actual_width": act_w,
        "matched90_width": matched_w,
        "point_rel_l2": rel_err,
        "scores": scores,
        "scale_mean": mean_s,
    }


def _compute_gate_metrics(score_cal, score_test, is_ood, point_fail, band_hit, cal_alphas=(0.01, 0.05)):
    """Compute AUROC, FPR@95%TPR, TPR@1%/5%FPR, AURC (point-failure & miscoverage),
    and Cal-rank accepted-set coverage."""
    auroc = float(roc_auc_score(is_ood, score_test))
    fpr, tpr, _ = roc_curve(is_ood, score_test, drop_intermediate=False)
    fpr_at_95 = float(fpr[tpr >= 0.95].min()) if np.any(tpr >= 0.95) else 1.0
    tpr_at_01 = float(tpr[fpr <= 0.01].max()) if np.any(fpr <= 0.01) else 0.0
    tpr_at_05 = float(tpr[fpr <= 0.05].max()) if np.any(fpr <= 0.05) else 0.0

    # Risk-Coverage AURC (lower score = more confident / accepted first)
    order = np.argsort(score_test, kind="stable")
    n = len(score_test)
    pf_sorted = point_fail[order].astype(np.float64)
    mc_sorted = (~band_hit[order]).astype(np.float64)
    ks = np.arange(1, n + 1, dtype=np.float64)
    point_aurc = float(np.mean(np.cumsum(pf_sorted) / ks))
    miscov_aurc = float(np.mean(np.cumsum(mc_sorted) / ks))

    ops = {}
    for a in cal_alphas:
        cut = quantile(score_cal, a)
        keep = score_test <= cut
        ops[str(a)] = {
            "id_reject": float((~keep[~is_ood]).mean()),
            "ood_reject": float((~keep[is_ood]).mean()),
            "accept_rate": float(keep.mean()),
            "accepted_n": int(keep.sum()),
            "accepted_coverage": float(band_hit[keep].mean()) if keep.any() else None,
            "accepted_point_failure": float(point_fail[keep].mean()) if keep.any() else None,
        }
    return {
        "auroc": auroc,
        "fpr_at_95_tpr": fpr_at_95,
        "tpr_at_1_fpr": tpr_at_01,
        "tpr_at_5_fpr": tpr_at_05,
        "point_aurc": point_aurc,
        "miscov_aurc": miscov_aurc,
        "ungated_point_failure": float(point_fail.mean()),
        "ungated_coverage": float(band_hit.mean()),
        "ops": ops,
    }


def _noisy_burgers_split(u0_clean, nu, eta, cfg, rng):
    """Inject independent modes 9-16 noise with relative AC RMS = eta into initial condition u0,
    then evolve via the exact 512-point ETDRK2 Burgers solver."""
    n = len(u0_clean)
    # Extract exact complex Fourier coefficients (modes 0..16) from 64-pt grid
    spec = np.fft.rfft(u0_clean.astype(np.float64), axis=-1, norm="forward")
    coeff = spec[:, :17].copy()
    if eta > 0.0:
        ac_rms = np.sqrt(np.maximum(2.0 * np.sum(np.abs(coeff[:, 1:]) ** 2, axis=1, keepdims=True), 1e-30))
        # White complex Gaussian excitation on modes 9..16 (indices 9..16 inclusive)
        noise = rng.normal(size=(n, 8)) + 1j * rng.normal(size=(n, 8))
        noise_rms = np.sqrt(np.maximum(2.0 * np.sum(np.abs(noise) ** 2, axis=1, keepdims=True), 1e-30))
        coeff[:, 9:17] += noise * (eta * ac_rms / noise_rms)
    u64 = _solve_burgers_batch(coeff, nu, cfg)
    return u64


def run_task3(device=None):
    OUT.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = perf_counter()

    # ---------- Part A: Continuous alpha_hf curve ----------
    print("[Task 3A] Continuous high-frequency injection sweep (5 seeds x 6 alpha_hf)...", flush=True)
    part_a_per_seed = {}
    methods_a = ("opres_locked", "opres_robust", "physics_scalar", "ensemble_mean")

    for seed in SEEDS:
        root = REV_CACHE / "burgers" / f"seed_{seed}"
        p = Protocol("burgers", seed, 64, root, FROZEN_ROOT / f"seed_{seed}", device)
        models, ens_floor = _load_burgers_ensemble(seed, p, device)
        cfg = p.config

        # Calibrate all 4 methods on locked cal.npz
        cal_data = arrays(root / "data/cal.npz")
        cal_u0, cal_nu, cal_y = cal_data["u64"][:, 0], cal_data["nu"], cal_data["u64"][:, 1:].astype(np.float64)
        cal_p0, cal_em, cal_es = _predict_ens(p, models, cal_u0, cal_nu)
        cal_full = np.concatenate((cal_u0[:, None].astype(np.float64), cal_p0), axis=1)
        cal_r = burg_ops.trajectory_residual(cal_full, cal_nu, cfg)
        cal_z = burg_ops.lift("adv_diff", cal_full, cal_r, cal_nu, cfg)

        cal_scales = {
            "opres_locked": p3_mix(burg_ops.envelope(cal_z, cfg), 0.85),
            "opres_robust": p3_mix(burg_ops.envelope(cal_z, cfg | {"spatial_radius": 0.32}), 0.25),
            "physics_scalar": p3_mix(burg_ops.envelope(cal_z, cfg), 0.0),
            "ensemble_mean": np.hypot(cal_es, ens_floor),
        }
        q90 = {}
        for mname, sc in cal_scales.items():
            center = cal_em if mname == "ensemble_mean" else cal_p0
            scores = (np.abs(center - cal_y) / np.maximum(sc, 1e-12)).reshape(len(cal_y), -1).max(axis=1)
            q90[mname] = quantile(scores, 0.10)

        # Draw paired base 16-mode spectrum on a disjoint stream (6901) per seed
        n_test = 500
        rng = np.random.default_rng(np.random.SeedSequence([seed, 6901]))
        vr = cfg["viscosity_id"]
        nu_base = np.exp(rng.uniform(*np.log(vr), size=n_test))
        modes16 = np.arange(1, 17, dtype=np.float64)
        raw_c = (rng.normal(size=(n_test, 16)) + 1j * rng.normal(size=(n_test, 16))) / (modes16 ** 1.35)
        amp_base = rng.uniform(0.55, 1.25, size=(n_test, 1))
        mean_base = rng.uniform(-0.4, 0.4, size=(n_test, 1))

        seed_a = {}
        for ahf in ALPHAS_HF:
            c = raw_c.copy()
            c[:, 8:] *= ahf
            c *= amp_base / np.sqrt(2.0 * np.sum(np.abs(c) ** 2, axis=1, keepdims=True))
            coeff = np.concatenate((mean_base, c), axis=1)
            u64 = _solve_burgers_batch(coeff, nu_base, cfg)
            u0 = u64[:, 0]
            y_true = u64[:, 1:].astype(np.float64)

            p0, em, es = _predict_ens(p, models, u0, nu_base)
            full = np.concatenate((u0[:, None].astype(np.float64), p0), axis=1)
            r = burg_ops.trajectory_residual(full, nu_base, cfg)
            z = burg_ops.lift("adv_diff", full, r, nu_base, cfg)

            t_scales = {
                "opres_locked": p3_mix(burg_ops.envelope(z, cfg), 0.85),
                "opres_robust": p3_mix(burg_ops.envelope(z, cfg | {"spatial_radius": 0.32}), 0.25),
                "physics_scalar": p3_mix(burg_ops.envelope(z, cfg), 0.0),
                "ensemble_mean": np.hypot(es, ens_floor),
            }
            res_a = {}
            for mname in methods_a:
                center = em if mname == "ensemble_mean" else p0
                ev = _eval_band(center, y_true, t_scales[mname], q90[mname])
                res_a[mname] = {k: ev[k] for k in ("coverage", "actual_width", "matched90_width", "point_rel_l2")}
            # Also track mean high-mode energy fraction
            sp_u0 = np.abs(np.fft.rfft(u0.astype(np.float64), axis=-1)) ** 2
            res_a["mean_hf_energy_ratio"] = float((sp_u0[:, 9:].sum(axis=-1) / np.maximum(sp_u0.sum(axis=-1), 1e-30)).mean())
            seed_a[str(ahf)] = res_a
        part_a_per_seed[str(seed)] = seed_a
        print(f"  [Task 3A] seed={seed} | ahf=0.0: OpRes={100*seed_a['0.0']['opres_locked']['coverage']:.1f}% | "
              f"0.10: {100*seed_a['0.1']['opres_locked']['coverage']:.1f}% | "
              f"0.25: {100*seed_a['0.25']['opres_locked']['coverage']:.1f}% | "
              f"0.50: {100*seed_a['0.5']['opres_locked']['coverage']:.1f}% | "
              f"1.0: OpRes={100*seed_a['1.0']['opres_locked']['coverage']:.1f}% / "
              f"Robust={100*seed_a['1.0']['opres_robust']['coverage']:.1f}% / "
              f"Scalar={100*seed_a['1.0']['physics_scalar']['coverage']:.1f}% / "
              f"Ens={100*seed_a['1.0']['ensemble_mean']['coverage']:.1f}%", flush=True)

    # Aggregate Part A
    part_a_summary = {}
    for ahf in ALPHAS_HF:
        ak = str(ahf)
        part_a_summary[ak] = {
            "mean_hf_energy_ratio": _seed_stats([part_a_per_seed[str(s)][ak]["mean_hf_energy_ratio"] for s in SEEDS]),
            "methods": {
                mname: {
                    k: _seed_stats([part_a_per_seed[str(s)][ak][mname][k] for s in SEEDS])
                    for k in ("coverage", "actual_width", "matched90_width", "point_rel_l2")
                }
                for mname in methods_a
            },
        }

    # ---------- Part B: Spectrally Overlapping Noisy Stream ----------
    print("[Task 3B] Noisy spectral overlap gate stress test (5 seeds x 5 eta_noise)...", flush=True)
    part_b_per_seed = {"id_only_noise": {}, "ambient_noise": {}}

    for seed in SEEDS:
        root = REV_CACHE / "burgers" / f"seed_{seed}"
        p = Protocol("burgers", seed, 64, root, FROZEN_ROOT / f"seed_{seed}", device)
        model = p.open_frozen()
        cfg = p.config

        cal_clean = arrays(root / "data/cal.npz")
        id_clean = arrays(root / "data/id.npz")
        hi_clean = arrays(root / "data/ood_high.npz")

        def _extract_stream(u64, nu):
            u0 = u64[:, 0]
            y_true = u64[:, 1:].astype(np.float64)
            preds = []
            for sl in batches(len(u0), 64):
                preds.append(p.predict(model, (u0[sl].astype(np.float32), nu[sl].astype(np.float32)), 64, p.frames))
            pred = np.concatenate(preds, axis=0).astype(np.float64)
            full = np.concatenate((u0[:, None].astype(np.float64), pred), axis=1)
            r = burg_ops.trajectory_residual(full, nu, cfg)
            z = burg_ops.lift("adv_diff", full, r, nu, cfg)
            sc = p3_mix(burg_ops.envelope(z, cfg), 0.85)
            scores = (np.abs(pred - y_true) / np.maximum(sc, 1e-12)).reshape(len(y_true), -1).max(axis=1)
            rel_err = relative_l2(pred, y_true)
            _, spec_feat = burg_features(u0.astype(np.float64), nu.astype(np.float64))
            # Physics features (label-free): rms(r), rms(z), max(|z|), mean(sigma_h)
            phys_feat = np.column_stack((
                p3_rms(r).ravel(),
                p3_rms(z).ravel(),
                np.abs(z).reshape(len(z), -1).max(axis=1),
                sc.reshape(len(sc), -1).mean(axis=1),
            ))
            return {
                "scores": scores,
                "rel_err": rel_err,
                "spec_feat": spec_feat,          # (n, 4): [log_nu, max|u0'|, rms(u0), E_{k>8}/E_tot]
                "hf_ratio": spec_feat[:, -1],
                "phys_feat": phys_feat,
                "joint_feat": np.column_stack((spec_feat, phys_feat)),
                "rms_z": phys_feat[:, 1],
            }

        # Precompute clean OOD-high once per seed
        hi_clean_ext = _extract_stream(hi_clean["u64"], hi_clean["nu"])

        seed_id_only = {}
        seed_ambient = {}

        for e_idx, eta in enumerate(ETAS_NOISE):
            ek = str(eta)
            if eta == 0.0:
                cal_u64 = cal_clean["u64"]
                id_u64 = id_clean["u64"]
                hi_amb_u64 = hi_clean["u64"]
            else:
                rng_c = np.random.default_rng(np.random.SeedSequence([seed, 6950 + e_idx * 3 + 0]))
                rng_i = np.random.default_rng(np.random.SeedSequence([seed, 6950 + e_idx * 3 + 1]))
                rng_h = np.random.default_rng(np.random.SeedSequence([seed, 6950 + e_idx * 3 + 2]))
                cal_u64 = _noisy_burgers_split(cal_clean["u64"][:, 0], cal_clean["nu"], eta, cfg, rng_c)
                id_u64 = _noisy_burgers_split(id_clean["u64"][:, 0], id_clean["nu"], eta, cfg, rng_i)
                hi_amb_u64 = _noisy_burgers_split(hi_clean["u64"][:, 0], hi_clean["nu"], eta, cfg, rng_h)

            cal_ext = _extract_stream(cal_u64, cal_clean["nu"])
            id_ext = _extract_stream(id_u64, id_clean["nu"])
            hi_amb_ext = hi_clean_ext if eta == 0.0 else _extract_stream(hi_amb_u64, hi_clean["nu"])

            q90_cal = quantile(cal_ext["scores"], 0.10)

            def _eval_gates(hi_ext):
                n_id = len(id_ext["scores"])
                n_hi = len(hi_ext["scores"])
                is_ood = np.concatenate((np.zeros(n_id, dtype=bool), np.ones(n_hi, dtype=bool)))
                scores_mix = np.concatenate((id_ext["scores"], hi_ext["scores"]))
                rel_mix = np.concatenate((id_ext["rel_err"], hi_ext["rel_err"]))
                point_fail = rel_mix > 0.15
                band_hit = scores_mix <= q90_cal

                # Gate 1a: Pure input scalar spectral energy E_{k>8}/E_tot (Cal-only, no target fit)
                g_spec_scalar = _compute_gate_metrics(
                    cal_ext["hf_ratio"],
                    np.concatenate((id_ext["hf_ratio"], hi_ext["hf_ratio"])),
                    is_ood, point_fail, band_hit,
                )

                # Gate 1b: Pure input spectral logistic (half-Cal source vs half-Cal+mix, matching phase3_burgers.py)
                ref = len(cal_ext["spec_feat"]) // 2
                xc_in = cal_ext["spec_feat"]
                xt_in = np.concatenate((id_ext["spec_feat"], hi_ext["spec_feat"]), axis=0)
                _, w_in = estimated_weights(xc_in[:ref], np.concatenate((xc_in[ref:], xt_in), axis=0),
                                            {"base_weight_clip": [0.02, 50.0], "weight_logistic_c": 1.0})
                g_spec_logistic = _compute_gate_metrics(
                    w_in[:len(xc_in) - ref], w_in[len(xc_in) - ref:],
                    is_ood, point_fail, band_hit,
                )

                # Gate 2a: Pure physics lifted-scale RMS(Z_h) (Cal-only, zero target fit!)
                g_phys_only = _compute_gate_metrics(
                    cal_ext["rms_z"],
                    np.concatenate((id_ext["rms_z"], hi_ext["rms_z"])),
                    is_ood, point_fail, band_hit,
                )

                # Gate 2b: Input + physics lifted-scale joint logistic (half-Cal source vs half-Cal+mix)
                xc_jt = cal_ext["joint_feat"]
                xt_jt = np.concatenate((id_ext["joint_feat"], hi_ext["joint_feat"]), axis=0)
                _, w_jt = estimated_weights(xc_jt[:ref], np.concatenate((xc_jt[ref:], xt_jt), axis=0),
                                            {"base_weight_clip": [0.02, 50.0], "weight_logistic_c": 1.0})
                g_joint_logistic = _compute_gate_metrics(
                    w_jt[:len(xc_jt) - ref], w_jt[len(xc_jt) - ref:],
                    is_ood, point_fail, band_hit,
                )

                # Gate 2c: Cal-only standardized sum of max-norm spectral + lifted scale (strictly inductive, no target batch!)
                mu_s, sd_s = float(cal_ext["hf_ratio"].mean()), max(float(cal_ext["hf_ratio"].std()), 1e-8)
                mu_z, sd_z = float(cal_ext["rms_z"].mean()), max(float(cal_ext["rms_z"].std()), 1e-8)
                # Use log-scale for heavy-tailed spectral energy & lifted RMS
                lc_s = np.log(np.maximum(cal_ext["hf_ratio"], 1e-12))
                lc_z = np.log(np.maximum(cal_ext["rms_z"], 1e-12))
                lt_s = np.log(np.maximum(np.concatenate((id_ext["hf_ratio"], hi_ext["hf_ratio"])), 1e-12))
                lt_z = np.log(np.maximum(np.concatenate((id_ext["rms_z"], hi_ext["rms_z"])), 1e-12))
                sc_cal_ind = np.maximum(
                    (lc_s - lc_s.mean()) / max(lc_s.std(), 1e-8),
                    (lc_z - lc_z.mean()) / max(lc_z.std(), 1e-8),
                )
                sc_tst_ind = np.maximum(
                    (lt_s - lc_s.mean()) / max(lc_s.std(), 1e-8),
                    (lt_z - lc_z.mean()) / max(lc_z.std(), 1e-8),
                )
                g_joint_cal_only = _compute_gate_metrics(sc_cal_ind, sc_tst_ind, is_ood, point_fail, band_hit)

                return {
                    "id_cov_unweighted": float((id_ext["scores"] <= q90_cal).mean()),
                    "ood_cov_unweighted": float((hi_ext["scores"] <= q90_cal).mean()),
                    "id_rel_l2": float(id_ext["rel_err"].mean()),
                    "ood_rel_l2": float(hi_ext["rel_err"].mean()),
                    "pure_spectral_scalar": g_spec_scalar,
                    "pure_spectral_logistic": g_spec_logistic,
                    "physics_rms_z_cal_only": g_phys_only,
                    "joint_input_physics_logistic": g_joint_logistic,
                    "joint_input_physics_cal_only": g_joint_cal_only,
                }

            seed_id_only[ek] = _eval_gates(hi_clean_ext)
            seed_ambient[ek] = _eval_gates(hi_amb_ext)

        part_b_per_seed["id_only_noise"][str(seed)] = seed_id_only
        part_b_per_seed["ambient_noise"][str(seed)] = seed_ambient
        for ptag, sdic in (("ID-only", seed_id_only), ("Ambient", seed_ambient)):
            r10 = sdic["0.1"]
            print(f"  [Task 3B {ptag:7s}] seed={seed} eta=0.10 | "
                  f"SpecScalar AUROC={r10['pure_spectral_scalar']['auroc']:.3f} AURC={r10['pure_spectral_scalar']['point_aurc']:.4f} | "
                  f"SpecLog AUROC={r10['pure_spectral_logistic']['auroc']:.3f} | "
                  f"JointLog AUROC={r10['joint_input_physics_logistic']['auroc']:.3f} AURC={r10['joint_input_physics_logistic']['point_aurc']:.4f} | "
                  f"PhysOnly AUROC={r10['physics_rms_z_cal_only']['auroc']:.3f} AURC={r10['physics_rms_z_cal_only']['point_aurc']:.4f}",
                  flush=True)

    # Aggregate Part B
    def _agg_b(proto_ps):
        first = proto_ps[str(SEEDS[0])]

        def _rec(node, path):
            if isinstance(node, dict):
                return {k: _rec(v, path + [k]) for k, v in node.items()}
            vals = []
            for s in SEEDS:
                cur = proto_ps[str(s)]
                for k in path:
                    cur = cur[k]
                if cur is not None:
                    vals.append(float(cur))
            if len(vals) == len(SEEDS):
                return _seed_stats(vals)
            return {"mean": float(np.mean(vals)) if vals else None, "n_defined": len(vals)}

        return _rec(first, [])

    out = {
        "seeds": list(SEEDS),
        "elapsed_s": perf_counter() - t0,
        "alphas_hf": list(ALPHAS_HF),
        "etas_noise": list(ETAS_NOISE),
        "part_a_continuous_shift": {
            "summary": part_a_summary,
            "per_seed": part_a_per_seed,
        },
        "part_b_noisy_spectral_overlap": {
            "summary": {
                "id_only_noise": _agg_b(part_b_per_seed["id_only_noise"]),
                "ambient_noise": _agg_b(part_b_per_seed["ambient_noise"]),
            },
            "per_seed": part_b_per_seed,
        },
    }
    path = OUT / "task3_burgers_stress.json"
    path.write_text(json.dumps(finite_json(out), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[Task 3] Done in {out['elapsed_s']:.1f}s -> {path}", flush=True)
    return out


if __name__ == "__main__":
    run_task3()
