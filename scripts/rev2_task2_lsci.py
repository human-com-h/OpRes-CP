"""Task 2 (P0): Fair Same-Event LSCI Comparison across 5 seeds x 3 PDEs.

Preserves original tab:opres-lsci audit numbers untouched and adds two strictly
apples-to-apples comparisons using the exact same LocalSlicedCP RLCP kernel
weights w and 32 random projections {phi_m}_{m=1}^{32}:
  1) LSCI Calibrated Simultaneous Band (same rectangular band, same 90% whole-field event):
     Locally weighted coordinatewise residual RMS using ALL calibration residuals
     + OpRes-CP relative floor (rho_rel = 0.10), calibrated via exact RLCP
     locally weighted max-standardized conformal quantile.
  2) Native 32-Projection Width Comparison (same 32-projection event):
     Exact mean projection interval width of LSCI's native depth set along the
     32 slices {phi_m} vs. OpRes-CP's induced support-function width
     2 * q * sum_j |phi_{m,j}| * sigma_h(x_j) (and both coverages on the 32-slice event).

Outputs written to results/phase3_revision_v2/task2_lsci_fair.json.
"""

import json
import math
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from darcy_day15 import arrays, value_at_coverage, weighted_quantile
from phase2d import batches, take
from phase2d_data import Protocol
from phase2d_statistics import ALPHAS, LocalSlicedCP, _weighted_depth_quantile, flat
from phase3_common import finite_json
from phase3_summarize import PHASE2_OP, SEEDS, STAGE, phase2_metrics, stat


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


def _predict_all(protocol, model, data, grid, frames, batch=16):
    inputs, truth = protocol.split(data, grid, frames)
    preds = []
    for sl in batches(len(truth), batch):
        preds.append(protocol.predict(model, take(inputs, sl), grid, frames))
    return inputs, truth, np.concatenate(preds, axis=0)


def _exact_phi_abs_l1_weights(fit):
    """Build exact |phi_m| sparse weight representation accounting for duplicate coordinate indices.

    Returns (slices, d) or list of (unique_idx, abs_coeff) per slice so
    sum_j |phi_{m,j}| * sigma(x_j) is exact even if random coordinates collide.
    """
    per_slice = []
    for m in range(fit.coordinates.shape[0]):
        idx = fit.coordinates[m]
        coef = fit.coefficients[m]
        uniq, inv = np.unique(idx, return_inverse=True)
        summed = np.zeros(len(uniq), dtype=np.float64)
        np.add.at(summed, inv, coef)
        per_slice.append((uniq, np.abs(summed)))
    return per_slice


def _opres_projection_support_halfwidth(sigma_flat, phi_abs_list):
    """Exact half-width sum_j |phi_{m,j}| * sigma_i(x_j) for all cases i and slices m.
    Returns array of shape (n_test, n_slices)."""
    n_test = sigma_flat.shape[0]
    n_slices = len(phi_abs_list)
    out = np.empty((n_test, n_slices), dtype=np.float64)
    for m, (uniq, abs_c) in enumerate(phi_abs_list):
        out[:, m] = sigma_flat[:, uniq] @ abs_c
    return out


def _lsci_exact_slice_interval_widths(fit, weights, target_residuals, alpha=0.1):
    """Compute exact 32-projection interval widths for LSCI's native projected-depth set.

    For each test case i with RLCP weights w = (w_1, ..., w_n, w_test):
      - Evaluates the exact label-free calibration-weighted depth threshold tau_cal
        from fit._region(w), where the 32-slice projected-depth set is the intersection
        of the 32 slabs {u : L_m <= <phi_m, u> <= U_m} with
        d_m^{cal}(v) >= tau_cal (exact [tau_cal/2, 1 - tau_cal/2] weighted quantiles
        along each slice m).
      - Also verifies exact symmetric (n+1)-point native set membership (matching tab:opres-lsci)
        and computes the exact slab coverage on the 32 projections.
    """
    residuals = flat(target_residuals).astype(np.float64)
    projections = fit.project(residuals)  # (n_test, 32)
    n_test, n_slices = projections.shape
    j_idx = np.arange(n_slices)

    native_hit = np.zeros(n_test, dtype=bool)
    native_inf = np.zeros(n_test, dtype=bool)
    slab_hit = np.zeros(n_test, dtype=bool)
    slab_inf = np.zeros(n_test, dtype=bool)
    slab_mean_width = np.full(n_test, np.inf, dtype=np.float64)
    slab_median_width = np.full(n_test, np.inf, dtype=np.float64)
    # Also compute exact (n+1)-point inclusive interval widths (conservative outer bound)
    sym_inf = np.zeros(n_test, dtype=bool)
    sym_mean_width = np.full(n_test, np.inf, dtype=np.float64)

    for i, w in enumerate(weights):
        w_cal = w[:-1]
        w_test = float(w[-1])
        cal_mass = float(w_cal.sum())
        cumulative = np.c_[np.zeros(n_slices), np.cumsum(w_cal[fit.order], axis=1)]

        # 1) Label-free calibration-weighted depth threshold
        if cal_mass > 0.0:
            lo_d = np.take_along_axis(cumulative, fit.right, axis=1) / cal_mass
            hi_d = (cal_mass - np.take_along_axis(cumulative, fit.left, axis=1)) / cal_mass
            depths_cal = np.take_along_axis(2.0 * np.minimum(lo_d, hi_d), fit.inverse, axis=1).min(axis=0)
        else:
            depths_cal = np.zeros(len(w_cal), dtype=np.float64)
        tau_cal, is_inf_cal = _weighted_depth_quantile(depths_cal, w_cal, w_test, alpha)

        # 2) Exact symmetric (n+1)-point native membership (identical to phase2d_statistics.py)
        proj = projections[i]
        le_cal = np.take_along_axis(cumulative, fit.right, axis=1)
        lt_cal = np.take_along_axis(cumulative, fit.left, axis=1)
        le_tot = le_cal + w_test * (proj[:, None] <= fit.sorted)
        ge_tot = 1.0 - (lt_cal + w_test * (proj[:, None] < fit.sorted))
        cal_depths_full = np.take_along_axis(2.0 * np.minimum(le_tot, ge_tot), fit.inverse, axis=1).min(axis=0)
        left = [np.searchsorted(a, v, side="left") for a, v in zip(fit.sorted, proj)]
        right = [np.searchsorted(a, v, side="right") for a, v in zip(fit.sorted, proj)]
        test_depth = float(np.min(2.0 * np.minimum(cumulative[j_idx, right] + w_test, 1.0 - cumulative[j_idx, left])))
        q_native, is_inf_native = _weighted_depth_quantile(cal_depths_full, w_cal, w_test, alpha)
        native_inf[i] = is_inf_native
        native_hit[i] = True if is_inf_native else (test_depth >= q_native - 1e-12)

        # Exact 1D slice intervals [L_m, U_m] for label-free depth set {v : d_m^{cal}(v) >= tau_cal}
        if is_inf_cal or cal_mass <= 0.0:
            slab_inf[i] = True
            slab_hit[i] = True
        else:
            half_tau = 0.5 * tau_cal * cal_mass
            # Smallest index k with cum[k+1] >= half_tau  =>  cum[:, 1:] >= half_tau
            # Largest index k with cal_mass - cum[k] >= half_tau  =>  cum[:, :-1] <= cal_mass - half_tau
            widths_m = np.empty(n_slices, dtype=np.float64)
            inside_all = True
            for m in range(n_slices):
                c_row = cumulative[m]  # length n+1, c_row[0]=0, c_row[n]=cal_mass
                z_row = fit.sorted[m]
                if half_tau <= 1e-15:
                    lo_v, hi_v = z_row[0], z_row[-1]
                else:
                    lo_k = int(np.searchsorted(c_row[1:], half_tau - 1e-12, side="left"))
                    lo_k = min(lo_k, len(z_row) - 1)
                    # Find largest k with c_row[k] <= cal_mass - half_tau
                    hi_k = int(np.searchsorted(c_row[:-1], cal_mass - half_tau + 1e-12, side="right")) - 1
                    hi_k = max(0, min(hi_k, len(z_row) - 1))
                    lo_v, hi_v = z_row[lo_k], z_row[hi_k]
                widths_m[m] = max(0.0, hi_v - lo_v)
                if not (lo_v - 1e-12 <= proj[m] <= hi_v + 1e-12):
                    inside_all = False
            slab_hit[i] = inside_all
            slab_mean_width[i] = float(np.mean(widths_m))
            slab_median_width[i] = float(np.median(widths_m))

        # Exact (n+1)-point slice intervals for d_m^{(n+1)}(v) >= tau_cal (unbounded if tau_cal <= 2*w_test)
        if is_inf_cal or cal_mass <= 0.0 or (0.5 * tau_cal <= w_test + 1e-12):
            sym_inf[i] = True
        else:
            # W_<=(v) + w_test >= tau/2 => W_<=(v) >= tau/2 - w_test
            # 1 - W_<(v) >= tau/2 => W_<(v) <= 1 - tau/2 (note total mass = 1)
            lo_req = 0.5 * tau_cal - w_test
            hi_lim = 1.0 - 0.5 * tau_cal
            w_sym = np.empty(n_slices, dtype=np.float64)
            for m in range(n_slices):
                c_row = cumulative[m]
                z_row = fit.sorted[m]
                lo_k = int(np.searchsorted(c_row[1:], lo_req - 1e-12, side="left"))
                lo_k = min(lo_k, len(z_row) - 1)
                hi_k = int(np.searchsorted(c_row[:-1], hi_lim + 1e-12, side="right")) - 1
                hi_k = max(0, min(hi_k, len(z_row) - 1))
                w_sym[m] = max(0.0, z_row[hi_k] - z_row[lo_k])
            sym_mean_width[i] = float(np.mean(w_sym))

    return {
        "native_hit": native_hit,
        "native_inf": native_inf,
        "slab_hit": slab_hit,
        "slab_inf": slab_inf,
        "slab_mean_width": slab_mean_width,
        "slab_median_width": slab_median_width,
        "sym_inf": sym_inf,
        "sym_mean_width": sym_mean_width,
    }


def _lsci_calibrated_simultaneous_band(r_cal_flat, weights, r_test_flat, rho_rel=0.10, alpha=0.10, device=None):
    """Part 1: LSCI Calibrated Simultaneous Band (same rectangular band, same 90% whole-field event).

    Uses ALL n calibration residuals with the EXACT RLCP kernel weights w = (w_1,...,w_n,w_test):
      - Coordinatewise weighted RMS: s0(x) = sqrt(sum_i p_i r_i(x)^2), p_i = w_i / sum_j w_j
      - Relative floor matching OpRes-CP: sigma_LSCI(x) = sqrt(s0(x)^2 + (rho_rel * RMS(s0))^2)
      - Calibrate max-standardized score S_i = max_x |r_i(x)| / sigma_LSCI(x) using exact
        RLCP locally weighted conformal quantile (test mass w_test at +inf).

    Uses GPU tensor reduction when d > 4096 and CUDA is available (exact float32/float64 parity).
    """
    n_cal, d = r_cal_flat.shape
    n_test = r_test_flat.shape[0]
    use_gpu = (device is not None and device.type == "cuda" and d > 4096)

    abs_cal = np.abs(r_cal_flat).astype(np.float64)
    r2_cal = (r_cal_flat.astype(np.float64)) ** 2
    abs_test = np.abs(r_test_flat).astype(np.float64)

    if use_gpu:
        abs_cal_t = torch.as_tensor(np.abs(r_cal_flat), dtype=torch.float32, device=device)
        r2_cal_t = abs_cal_t * abs_cal_t
        abs_test_t = torch.as_tensor(np.abs(r_test_flat), dtype=torch.float32, device=device)

    cov = np.zeros(n_test, dtype=bool)
    is_inf = np.zeros(n_test, dtype=bool)
    actual_width = np.full(n_test, np.inf, dtype=np.float64)
    test_scores = np.empty(n_test, dtype=np.float64)
    scale_means = np.empty(n_test, dtype=np.float64)
    cov_loo = np.zeros(n_test, dtype=bool)
    actual_width_loo = np.full(n_test, np.inf, dtype=np.float64)

    chunk = 50 if use_gpu else (25 if d > 10000 else 100)
    for i0 in range(0, n_test, chunk):
        i1 = min(i0 + chunk, n_test)
        w_blk = weights[i0:i1]
        w_cal_blk = w_blk[:, :-1]
        w_test_blk = w_blk[:, -1]
        cal_mass_blk = w_cal_blk.sum(axis=1)
        p_blk = np.divide(w_cal_blk, np.maximum(cal_mass_blk[:, None], 1e-30))

        if use_gpu:
            with torch.inference_mode():
                p_t = torch.as_tensor(p_blk, dtype=torch.float32, device=device)
                s0_sq_t = p_t @ r2_cal_t                                  # (b, d)
                rms_sq_t = s0_sq_t.mean(dim=1, keepdim=True)
                sigma_t = torch.sqrt(s0_sq_t + (rho_rel ** 2) * rms_sq_t).clamp_min(1e-12)
                inv_sig_t = 1.0 / sigma_t                                 # (b, d)
                s_means_blk = sigma_t.mean(dim=1).cpu().numpy().astype(np.float64)
                s_test_blk = (abs_test_t[i0:i1] * inv_sig_t).amax(dim=1).cpu().numpy().astype(np.float64)

            for loc in range(i1 - i0):
                idx = i0 + loc
                wc = w_cal_blk[loc]
                wt = float(w_test_blk[loc])
                cm = float(cal_mass_blk[loc])
                s_mean = float(s_means_blk[loc])
                s_test = float(s_test_blk[loc])
                scale_means[idx] = s_mean
                test_scores[idx] = s_test
                if cm <= 0.0 or wt > alpha * (cm + wt) + 1e-12:
                    is_inf[idx] = True
                    cov[idx] = True
                    cov_loo[idx] = True
                    continue
                with torch.inference_mode():
                    s_cal_t = (abs_cal_t * inv_sig_t[loc].unsqueeze(0)).amax(dim=1)
                    s_cal = s_cal_t.cpu().numpy().astype(np.float64)
                q_arr, _, _ = weighted_quantile(s_cal, wc, np.array([wt]), alpha)
                q = float(q_arr[0])
                if not np.isfinite(q):
                    is_inf[idx] = True
                    cov[idx] = True
                    cov_loo[idx] = True
                else:
                    cov[idx] = s_test <= q
                    actual_width[idx] = 2.0 * q * s_mean

                # LOO adjustment on GPU for non-negligible weights
                p_i = p_blk[loc]
                if p_i.max() < 0.99:
                    s_cal_loo = s_cal.copy()
                    big = np.flatnonzero(p_i > 1e-3)
                    if len(big):
                        with torch.inference_mode():
                            big_t = torch.as_tensor(big, dtype=torch.long, device=device)
                            pb_t = p_t[loc, big_t].unsqueeze(1)
                            s0_b = ((s0_sq_t[loc].unsqueeze(0) - pb_t * r2_cal_t[big_t]) / (1.0 - pb_t).clamp_min(1e-12)).clamp_min(0.0)
                            rms_b = s0_b.mean(dim=1, keepdim=True)
                            sig_b = torch.sqrt(s0_b + (rho_rel ** 2) * rms_b).clamp_min(1e-12)
                            s_cal_loo[big] = (abs_cal_t[big_t] / sig_b).amax(dim=1).cpu().numpy().astype(np.float64)
                    q_loo_arr, _, _ = weighted_quantile(s_cal_loo, wc, np.array([wt]), alpha)
                    q_loo = float(q_loo_arr[0])
                    if not np.isfinite(q_loo):
                        cov_loo[idx] = True
                    else:
                        cov_loo[idx] = s_test <= q_loo
                        actual_width_loo[idx] = 2.0 * q_loo * s_mean
            continue

        s0_sq = p_blk @ r2_cal
        rms_s0_sq = s0_sq.mean(axis=1, keepdims=True)
        sigma_blk = np.maximum(np.sqrt(s0_sq + (rho_rel ** 2) * rms_s0_sq), 1e-12)

        for loc in range(i1 - i0):
            idx = i0 + loc
            sig = sigma_blk[loc]
            wc = w_cal_blk[loc]
            wt = float(w_test_blk[loc])
            cm = float(cal_mass_blk[loc])
            inv_sig = 1.0 / sig
            s_mean = float(sig.mean())
            scale_means[idx] = s_mean
            s_test = float(np.max(abs_test[idx] * inv_sig))
            test_scores[idx] = s_test

            if cm <= 0.0 or wt > alpha * (cm + wt) + 1e-12:
                is_inf[idx] = True
                cov[idx] = True
                cov_loo[idx] = True
                continue

            s_cal = (abs_cal * inv_sig[None, :]).max(axis=1)
            q_arr, _, _ = weighted_quantile(s_cal, wc, np.array([wt]), alpha)
            q = float(q_arr[0])
            if not np.isfinite(q):
                is_inf[idx] = True
                cov[idx] = True
                cov_loo[idx] = True
            else:
                cov[idx] = s_test <= q
                actual_width[idx] = 2.0 * q * s_mean

            p_i = p_blk[loc]
            if p_i.max() < 0.99:
                denom_loo = np.maximum(1.0 - p_i[:, None], 1e-12)
                s0_sq_loo = np.maximum((s0_sq[loc][None, :] - p_i[:, None] * r2_cal) / denom_loo, 0.0)
                rms_loo_sq = s0_sq_loo.mean(axis=1, keepdims=True)
                sig_loo = np.maximum(np.sqrt(s0_sq_loo + (rho_rel ** 2) * rms_loo_sq), 1e-12)
                s_cal_loo = (abs_cal / sig_loo).max(axis=1)
                q_loo_arr, _, _ = weighted_quantile(s_cal_loo, wc, np.array([wt]), alpha)
                q_loo = float(q_loo_arr[0])
                if not np.isfinite(q_loo):
                    cov_loo[idx] = True
                else:
                    cov_loo[idx] = s_test <= q_loo
                    actual_width_loo[idx] = 2.0 * q_loo * s_mean

    fin = ~is_inf
    q90_emp = value_at_coverage(test_scores, 1.0 - alpha)
    matched90_width = float(2.0 * q90_emp * scale_means.mean())

    return {
        "coverage": float(cov.mean()),
        "infinite_rate": float(is_inf.mean()),
        "finite_n": int(fin.sum()),
        "finite_coverage": float(cov[fin].mean()) if fin.any() else None,
        "actual_width": float(actual_width.mean()) if fin.all() else "inf",
        "finite_actual_width": float(actual_width[fin].mean()) if fin.any() else None,
        "matched90_width": matched90_width,
        "loo_coverage": float(cov_loo.mean()),
        "loo_finite_actual_width": float(actual_width_loo[fin].mean()) if fin.any() else None,
    }


def run_synthetic_unit_test():
    """Verify LSCI Calibrated Simultaneous Band attains ~90% simultaneous coverage on i.i.d. data."""
    rng = np.random.default_rng(2026)
    n_cal, n_test, d, p = 500, 600, 64, 5
    x_cal = rng.normal(size=(n_cal, p))
    x_test = rng.normal(size=(n_test, p))
    sc_cal = 0.5 + np.abs(x_cal[:, :1])
    sc_test = 0.5 + np.abs(x_test[:, :1])
    r_cal = rng.normal(size=(n_cal, d)) * sc_cal
    r_test = rng.normal(size=(n_test, d)) * sc_test
    fit = LocalSlicedCP(r_cal, x_cal, seed=12345)
    w = fit.prepare(x_test)
    res = _lsci_calibrated_simultaneous_band(r_cal, w, r_test, rho_rel=0.10, alpha=0.10)
    proj = _lsci_exact_slice_interval_widths(fit, w, r_test, alpha=0.10)
    print(f"[Task 2 Unit Check] i.i.d. Calibrated-Band cov={100*res['coverage']:.2f}% (LOO={100*res['loo_coverage']:.2f}%), "
          f"Native32 cov={100*proj['native_hit'].mean():.2f}%, Slab32 cov={100*proj['slab_hit'].mean():.2f}%", flush=True)
    assert 0.85 <= res["coverage"] <= 0.95, f"Calibrated band synthetic cov out of range: {res['coverage']}"
    assert 0.86 <= res["loo_coverage"] <= 0.95, f"LOO band synthetic cov out of range: {res['loo_coverage']}"
    return {
        "iid_cal_band_cov90": res["coverage"],
        "iid_cal_band_loo_cov90": res["loo_coverage"],
        "iid_native32_cov90": float(proj["native_hit"].mean()),
        "iid_slab32_cov90": float(proj["slab_hit"].mean()),
    }


def run_task2(device=None):
    OUT.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = perf_counter()

    unit = run_synthetic_unit_test()

    specs = (
        ("darcy", None, (("id_test", "id_test_64"), ("ood_a", "ood_a_64"), ("ood_b", "ood_b_64"))),
        ("burgers", 16, (("id", "id_64"), ("ood", "ood_64"), ("ood_high", "ood_high_64"))),
        ("allen_cahn", 20, (("id", "id_64_20"), ("ood", "ood_64_20"))),
    )

    per_pde = {}
    for pde, _, split_pairs in specs:
        per_seed = {}
        for seed in SEEDS:
            root = REV_CACHE / pde / f"seed_{seed}"
            p = Protocol(pde, seed, 64, root, FROZEN_ROOT / f"seed_{seed}", device)
            model = p.open_frozen()
            cal_frames = p.cal_frames if pde == "allen_cahn" else p.frames
            batch = 16 if pde != "allen_cahn" else 8

            cal_data = arrays(root / "data/cal.npz")
            xcal, ycal, pcal = _predict_all(p, model, cal_data, 64, cal_frames, batch=batch)
            summary_cal = p.summaries(xcal)
            r_cal = (ycal - pcal).astype(np.float32)
            r_cal_flat = flat(r_cal)

            # Locked OpRes-CP calibration threshold (verify exact match with Phase 2D)
            s_cal_op = np.maximum(p.op_scale(pcal, xcal), 1e-12)
            s_cal_op_flat = flat(s_cal_op).astype(np.float64)
            op_cal_scores = (np.abs(flat(pcal - ycal)) / s_cal_op_flat).max(axis=1)
            q90_op = float(np.sort(op_cal_scores)[math.ceil(0.9 * (len(op_cal_scores) + 1)) - 1])

            seed_splits = {}
            for sp_key, p2d_label in split_pairs:
                data = arrays(root / f"data/{sp_key}.npz")
                xt, yt, pt = _predict_all(p, model, data, 64, cal_frames, batch=batch)
                cov_x = p.summaries(xt)
                r_test = (yt - pt).astype(np.float64)
                r_test_flat = flat(r_test)

                # Exact same LocalSlicedCP seed & RLCP weights as scripts/phase3_revision.py
                fit = LocalSlicedCP(r_cal, summary_cal, seed + 9029)
                w = fit.prepare(cov_x)

                # Part 1: LSCI Calibrated Simultaneous Band (same rectangular band, 90% whole-field event)
                lsci_band = _lsci_calibrated_simultaneous_band(r_cal_flat, w, r_test_flat, rho_rel=0.10, alpha=0.10, device=device)

                # OpRes-CP whole-field simultaneous band metrics on exact same split
                s_op = np.maximum(p.op_scale(pt, xt), 1e-12)
                s_op_flat = flat(s_op).astype(np.float64)
                op_scores = (np.abs(r_test_flat) / s_op_flat).max(axis=1)
                op_cov = float((op_scores <= q90_op).mean())
                op_act_w = float((2.0 * q90_op * s_op_flat.mean(axis=1)).mean())
                op_q90_emp = value_at_coverage(op_scores, 0.90)
                op_matched_w = float(2.0 * op_q90_emp * s_op_flat.mean())

                if sp_key in ("id_test", "id"):
                    ref_cov = phase2_metrics(pde, seed, 64)["results"][p2d_label][PHASE2_OP]["0.1"]["coverage"]
                    assert abs(op_cov - ref_cov) <= 1e-12, f"OpRes mismatch {pde} {seed}: {op_cov} vs {ref_cov}"

                # Part 2: Exact 32-projection width comparison on the SAME 32 projections {phi_m}
                phi_abs = _exact_phi_abs_l1_weights(fit)
                lsci_proj = _lsci_exact_slice_interval_widths(fit, w, r_test_flat, alpha=0.10)

                # OpRes-CP induced support-function width 2 * q * sum_j |phi_{m,j}| * sigma_h(x_j)
                op_hw = _opres_projection_support_halfwidth(s_op_flat, phi_abs)  # (n_test, 32)
                op_proj_w_fullq = 2.0 * q90_op * op_hw                           # (n_test, 32)
                test_proj = fit.project(r_test_flat)                             # (n_test, 32)
                op_proj_cov_fullq = float((np.abs(test_proj) <= q90_op * op_hw + 1e-12).all(axis=1).mean())

                # Also compute OpRes-CP calibrated DIRECTLY on the 32-projection event (for matched-event comparison!)
                op_cal_hw = _opres_projection_support_halfwidth(s_cal_op_flat, phi_abs)
                cal_proj = fit.project(r_cal_flat.astype(np.float64))
                op_cal_proj_scores = (np.abs(cal_proj) / np.maximum(op_cal_hw, 1e-30)).max(axis=1)
                q90_op_proj = float(np.sort(op_cal_proj_scores)[math.ceil(0.9 * (len(op_cal_proj_scores) + 1)) - 1])
                op_test_proj_scores = (np.abs(test_proj) / np.maximum(op_hw, 1e-30)).max(axis=1)
                op_proj_cov_cal32 = float((op_test_proj_scores <= q90_op_proj).mean())
                op_proj_w_cal32 = float((2.0 * q90_op_proj * op_hw).mean())
                op_q90_proj_emp = value_at_coverage(op_test_proj_scores, 0.90)
                op_proj_w_matched90 = float(2.0 * op_q90_proj_emp * op_hw.mean())

                fin_slab = ~lsci_proj["slab_inf"]
                fin_sym = ~lsci_proj["sym_inf"]
                proj_comp = {
                    "lsci_native32_cov": float(lsci_proj["native_hit"].mean()),
                    "lsci_native32_inf_rate": float(lsci_proj["native_inf"].mean()),
                    "lsci_slab32_cov": float(lsci_proj["slab_hit"].mean()),
                    "lsci_slab32_inf_rate": float(lsci_proj["slab_inf"].mean()),
                    "lsci_slab32_finite_n": int(fin_slab.sum()),
                    "lsci_slab32_finite_width": float(lsci_proj["slab_mean_width"][fin_slab].mean()) if fin_slab.any() else None,
                    "lsci_sym32_inf_rate": float(lsci_proj["sym_inf"].mean()),
                    "lsci_sym32_finite_width": float(lsci_proj["sym_mean_width"][fin_sym].mean()) if fin_sym.any() else None,
                    "opres_full_band_induced_32proj_cov": op_proj_cov_fullq,
                    "opres_full_band_induced_32proj_width": float(op_proj_w_fullq.mean()),
                    "opres_cal32_cov": op_proj_cov_cal32,
                    "opres_cal32_actual_width": op_proj_w_cal32,
                    "opres_cal32_matched90_width": op_proj_w_matched90,
                }

                seed_splits[p2d_label] = {
                    "lsci_cal_band": lsci_band,
                    "opres_whole_field": {
                        "coverage": op_cov,
                        "actual_width": op_act_w,
                        "matched90_width": op_matched_w,
                    },
                    "width_ratio_opres_over_lsci_cal_band_matched90": op_matched_w / lsci_band["matched90_width"],
                    "width_shrink_pct_vs_lsci_cal_band_matched90": 100.0 * (1.0 - op_matched_w / lsci_band["matched90_width"]),
                    "proj32": proj_comp,
                }
                sw = proj_comp["lsci_slab32_finite_width"]
                sw_s = f"{sw:.4f}" if sw is not None else "inf"
                print(f"  [Task 2] {pde:10s} seed={seed} {p2d_label:12s} | "
                      f"Band90 cov: LSCI={100*lsci_band['coverage']:.2f}% (mw={lsci_band['matched90_width']:.4f}) "
                      f"vs OpRes={100*op_cov:.2f}% (mw={op_matched_w:.4f}) | "
                      f"Proj32 w: LSCI={sw_s} vs OpRes(fullQ)={proj_comp['opres_full_band_induced_32proj_width']:.4f} "
                      f"(cov={100*op_proj_cov_fullq:.1f}%) / OpRes(cal32)={op_proj_w_matched90:.4f}", flush=True)
            per_seed[str(seed)] = seed_splits

        # Aggregate across 5 seeds
        summary = {}
        for _, p2d_label in split_pairs:
            recs = [per_seed[str(s)][p2d_label] for s in SEEDS]

            def _agg_path(keys):
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
                "lsci_cal_band": {
                    k: _agg_path(["lsci_cal_band", k])
                    for k in ("coverage", "infinite_rate", "finite_n", "finite_coverage",
                              "finite_actual_width", "matched90_width", "loo_coverage", "loo_finite_actual_width")
                },
                "opres_whole_field": {
                    k: _agg_path(["opres_whole_field", k])
                    for k in ("coverage", "actual_width", "matched90_width")
                },
                "width_shrink_pct_vs_lsci_cal_band_matched90": _agg_path(["width_shrink_pct_vs_lsci_cal_band_matched90"]),
                "proj32": {
                    k: _agg_path(["proj32", k])
                    for k in ("lsci_native32_cov", "lsci_native32_inf_rate", "lsci_slab32_cov",
                              "lsci_slab32_inf_rate", "lsci_slab32_finite_n", "lsci_slab32_finite_width",
                              "lsci_sym32_inf_rate", "lsci_sym32_finite_width",
                              "opres_full_band_induced_32proj_cov", "opres_full_band_induced_32proj_width",
                              "opres_cal32_cov", "opres_cal32_actual_width", "opres_cal32_matched90_width")
                },
            }
        per_pde[pde] = {"summary": summary, "per_seed": per_seed}

    out = {
        "seeds": list(SEEDS),
        "elapsed_s": perf_counter() - t0,
        "unit_checks": unit,
        "pdes": per_pde,
    }
    path = OUT / "task2_lsci_fair.json"
    path.write_text(json.dumps(finite_json(out), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[Task 2] Done in {out['elapsed_s']:.1f}s -> {path}", flush=True)
    return out


if __name__ == "__main__":
    run_task2()
