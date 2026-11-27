"""Task 1 (P0): Five-seed Scale-Error Spatial Fidelity & Operator Approximation Audit.

Zero backbone retraining; strictly read-only against results/phase2_kaggle_private/.
Evaluates across all 5 formal seeds [42, 137, 314, 271, 811] on:
  - Darcy 64^2 (ID, OOD-A, OOD-B)
  - Burgers 64 (ID, Visc-OOD, High-OOD)
  - Allen-Cahn 64^2 T=1 (ID, React-OOD)
  - L-shape Darcy 64^2 (ID, OOD-A)

Outputs written to results/phase3_revision_v2/task1_fidelity.json.
"""

from concurrent.futures import ProcessPoolExecutor
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
from scipy.fft import dstn, idstn
from scipy.linalg import eigh
from scipy.sparse.linalg import spsolve
from scipy.stats import rankdata, spearmanr
import torch

from allen_cahn_phase2_probe import reconstruct as ac_reconstruct, scale as ac_scale
import burgers_phase1_ops as burg_ops
from darcy_data import apply_operator as darcy_apply_op
from darcy_day15 import arrays
from darcy_day15_ops import lift as darcy_lift, mixture as darcy_mix, physical_scale as darcy_phys_scale
from phase2d import batches, take
from phase2d_data import Protocol
from phase3_allen_cahn import jacobian as ac_jacobian, midpoints as ac_midpoints, strang_cpu as ac_strang_cpu
from phase3_common import envelope as p3_envelope, finite_json, mix as p3_mix
import phase3_p2_lshape_amg as lsh
from phase3_summarize import DOWNLOADS, SEEDS, STAGE, stat


OUT = Path("results/phase3_revision_v2")
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


def _row_pearson(x, y):
    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(xc, axis=1) * np.linalg.norm(yc, axis=1)
    out = np.zeros(len(x), dtype=np.float64)
    good = denom > 1e-30
    out[good] = np.sum(xc[good] * yc[good], axis=1) / denom[good]
    return out


def _row_spearman(x, y):
    rx = rankdata(x, axis=1)
    ry = rankdata(y, axis=1)
    return _row_pearson(rx, ry)


def compute_scale_metrics(abs_err_flat, scale_flat):
    """Per-field spatial localization, peak recall/capture, normalized peak scale,
    Theorem 1 effective concentration rho_i, and cross-sample Spearman."""
    n, m = abs_err_flat.shape
    sc = np.maximum(scale_flat.astype(np.float64), 1e-30)
    ae = abs_err_flat.astype(np.float64)

    k10 = math.ceil(0.10 * m)
    k05 = math.ceil(0.05 * m)

    # Chunked for low peak RAM on Allen-Cahn (m = 81920)
    chunk = 50 if m > 10000 else n
    sp_list, pe_list, pr10_list, pr05_list, pc05_list, pc10_list = [], [], [], [], [], []
    sigma_star_list, rho_eff_list, mean_s_list, linf_list = [], [], [], []

    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        a_blk = ae[i0:i1]
        s_blk = sc[i0:i1]
        b = i1 - i0

        sp_list.append(_row_spearman(s_blk, a_blk))
        pe_list.append(_row_pearson(s_blk, a_blk))

        # Top-10% and top-5% thresholds via partition
        t10_err = np.partition(a_blk, m - k10, axis=1)[:, m - k10:m - k10 + 1]
        t10_sc = np.partition(s_blk, m - k10, axis=1)[:, m - k10:m - k10 + 1]
        # Exact top-k index sets using argpartition to avoid tie inflation
        idx_err10 = np.argpartition(a_blk, m - k10, axis=1)[:, m - k10:]
        idx_sc10 = np.argpartition(s_blk, m - k10, axis=1)[:, m - k10:]
        idx_err05 = np.argpartition(a_blk, m - k05, axis=1)[:, m - k05:]
        idx_sc05 = np.argpartition(s_blk, m - k05, axis=1)[:, m - k05:]

        err10_mask = np.zeros((b, m), dtype=bool)
        sc10_mask = np.zeros((b, m), dtype=bool)
        err05_mask = np.zeros((b, m), dtype=bool)
        sc05_mask = np.zeros((b, m), dtype=bool)
        rows = np.arange(b)[:, None]
        err10_mask[rows, idx_err10] = True
        sc10_mask[rows, idx_sc10] = True
        err05_mask[rows, idx_err05] = True
        sc05_mask[rows, idx_sc05] = True

        pr10_list.append((err10_mask & sc10_mask).sum(axis=1) / float(k10))
        pr05_list.append((err05_mask & sc05_mask).sum(axis=1) / float(k05))

        x_star = np.argmax(a_blk, axis=1)
        pc05_list.append(sc05_mask[np.arange(b), x_star].astype(np.float64))
        pc10_list.append(sc10_mask[np.arange(b), x_star].astype(np.float64))

        s_mean = s_blk.mean(axis=1)
        s_tilde = s_blk / np.maximum(s_mean[:, None], 1e-30)
        sigma_star_list.append(s_tilde[np.arange(b), x_star])

        linf = a_blk[np.arange(b), x_star]
        max_std_ratio = (a_blk / np.maximum(s_tilde, 1e-30)).max(axis=1)
        rho_eff = np.where(max_std_ratio > 1e-30, linf / max_std_ratio, 0.0)
        rho_eff_list.append(rho_eff)

        mean_s_list.append(s_mean)
        linf_list.append(linf)

    mean_s = np.concatenate(mean_s_list)
    linf = np.concatenate(linf_list)
    rho_cross = float(spearmanr(mean_s, linf).statistic)
    rho_eff_all = np.concatenate(rho_eff_list)
    sigma_star_all = np.concatenate(sigma_star_list)

    return {
        "spearman": float(np.mean(np.concatenate(sp_list))),
        "pearson": float(np.mean(np.concatenate(pe_list))),
        "peak_recall_10": float(np.mean(np.concatenate(pr10_list))),
        "peak_recall_5": float(np.mean(np.concatenate(pr05_list))),
        "peak_capture_5": float(np.mean(np.concatenate(pc05_list))),
        "peak_capture_10": float(np.mean(np.concatenate(pc10_list))),
        "sigma_tilde_xstar_mean": float(np.mean(sigma_star_all)),
        "sigma_tilde_xstar_median": float(np.median(sigma_star_all)),
        "rho_eff_mean": float(np.mean(rho_eff_all)),
        "rho_eff_p10": float(np.quantile(rho_eff_all, 0.10)),
        "rho_cross": rho_cross,
    }


def compute_signed_recovery(z_flat, e_flat):
    """Per-field signed error recovery: cos(Z_h, e_h), ||Z_h - e_h||_2/||e_h||_2,
    and scale-invariant LS residual min_alpha ||alpha Z_h - e_h||_2/||e_h||_2."""
    z = z_flat.astype(np.float64)
    e = e_flat.astype(np.float64)
    nz = np.linalg.norm(z, axis=1)
    ne = np.linalg.norm(e, axis=1)
    dot = np.sum(z * e, axis=1)
    cos = np.where((nz > 1e-30) & (ne > 1e-30), dot / (nz * ne), 0.0)
    rel_err = np.where(ne > 1e-30, np.linalg.norm(z - e, axis=1) / ne, 0.0)
    alpha_ls = np.where(nz > 1e-30, dot / (nz * nz), 0.0)
    ls_res = np.where(ne > 1e-30, np.linalg.norm(alpha_ls[:, None] * z - e, axis=1) / ne, 0.0)
    alpha_pos = np.maximum(alpha_ls, 0.0)
    ls_res_pos = np.where(ne > 1e-30, np.linalg.norm(alpha_pos[:, None] * z - e, axis=1) / ne, 0.0)
    return {
        "cos_mean": float(np.mean(cos)),
        "cos_median": float(np.median(cos)),
        "rel_l2_err_mean": float(np.mean(rel_err)),
        "rel_l2_err_median": float(np.median(rel_err)),
        "ls_residual_mean": float(np.mean(ls_res)),
        "ls_residual_pos_mean": float(np.mean(ls_res_pos)),
        "alpha_ls_median": float(np.median(alpha_ls)),
    }


# ---------- Rectangular FVM AMG (identical 2x2 Galerkin AMG on [0,1]^2) ----------

def build_rect_fvm_level0(a):
    n = a.shape[-1]
    m = np.ones((n, n), dtype=bool)
    tx = 2.0 * a[:, :-1] * a[:, 1:] / (a[:, :-1] + a[:, 1:])
    ty = 2.0 * a[:-1, :] * a[1:, :] / (a[:-1, :] + a[1:, :])
    bdiag = np.zeros_like(a)
    bdiag[0, :] += 2.0 * a[0, :]
    bdiag[-1, :] += 2.0 * a[-1, :]
    bdiag[:, 0] += 2.0 * a[:, 0]
    bdiag[:, -1] += 2.0 * a[:, -1]
    diag = bdiag.copy()
    diag[:, :-1] += tx
    diag[:, 1:] += tx
    diag[:-1, :] += ty
    diag[1:, :] += ty
    inv_diag = 1.0 / np.maximum(diag, 1e-30)
    return {"n": n, "mask": m, "tx": tx, "ty": ty, "bdiag": bdiag, "diag": diag, "inv_diag": inv_diag}


def build_rect_amg_hierarchy(a, min_n=4):
    levels = [build_rect_fvm_level0(a)]
    while levels[-1]["n"] > min_n:
        f = levels[-1]
        nc = f["n"] // 2
        m_c = np.ones((nc, nc), dtype=bool)
        tx_c = f["tx"][:, 1::2].reshape(nc, 2, nc - 1).sum(axis=1)
        ty_c = f["ty"][1::2, :].reshape(nc - 1, nc, 2).sum(axis=2)
        bdiag_c = f["bdiag"].reshape(nc, 2, nc, 2).sum(axis=(1, 3))
        diag_c = bdiag_c.copy()
        diag_c[:, :-1] += tx_c
        diag_c[:, 1:] += tx_c
        diag_c[:-1, :] += ty_c
        diag_c[1:, :] += ty_c
        inv_diag_c = 1.0 / np.maximum(diag_c, 1e-30)
        levels.append({"n": nc, "mask": m_c, "tx": tx_c, "ty": ty_c,
                       "bdiag": bdiag_c, "diag": diag_c, "inv_diag": inv_diag_c})
    cmat, cactive = lsh.assemble_sparse_flux(levels[-1])
    levels[-1]["coarse_mat"] = cmat
    levels[-1]["coarse_active"] = cactive
    for lvl in levels:
        nn = lvl["n"]
        ii, jj = np.indices((nn, nn))
        lvl["red"] = ((ii + jj) & 1) == 0
        lvl["black"] = ~lvl["red"]
    return levels


def _darcy_rect_op_worker(job):
    """Worker for Darcy 64^2 exact operator approximation & DST-II band decomposition."""
    a, e = job
    n = a.shape[-1]
    ae_full = darcy_apply_op(a, e)
    levels = build_rect_amg_hierarchy(a)

    # Full-field B_h A_h e_h
    z_liouv = darcy_lift("liouville", a, ae_full)
    z_v2 = lsh.lift_amg_v2(levels, ae_full, cycles=2, nu=2, omega_coarse=1.6)
    z_pcg2 = lsh.lift_amg_pcg(levels, ae_full, steps=2, nu=2)
    z_j5 = lsh.lift_jacobi_pcg(levels, ae_full, steps=5)

    ne = max(float(np.linalg.norm(e)), 1e-30)
    full_err = [
        float(np.linalg.norm(z_liouv - e) / ne),
        float(np.linalg.norm(z_v2 - e) / ne),
        float(np.linalg.norm(z_pcg2 - e) / ne),
        float(np.linalg.norm(z_j5 - e) / ne),
    ]

    # DST-II radial frequency bands: Low (k<=8), Mid (8<k<=24), High (k>24)
    k = np.arange(1, n + 1, dtype=np.float64)
    kr = np.hypot(k[:, None], k[None, :])
    masks = (kr <= 8.0, (kr > 8.0) & (kr <= 24.0), kr > 24.0)
    e_hat = dstn(e, type=2, norm="ortho")
    z_hat = dstn(z_liouv, type=2, norm="ortho")

    band_energy = []
    band_op_err = []  # 3 bands x 4 inverses
    band_z_cos = []
    band_e_abs = []
    for bm in masks:
        eb = idstn(np.where(bm, e_hat, 0.0), type=2, norm="ortho")
        zb_proj = idstn(np.where(bm, z_hat, 0.0), type=2, norm="ortho")
        neb = float(np.linalg.norm(eb))
        band_energy.append((neb / ne) ** 2)
        nzb = float(np.linalg.norm(zb_proj))
        band_z_cos.append(float(np.sum(zb_proj * eb) / (nzb * neb)) if (nzb > 1e-30 and neb > 1e-30) else 0.0)
        band_e_abs.append(np.abs(eb).astype(np.float32))
        if neb > 1e-30:
            aeb = darcy_apply_op(a, eb)
            b_liouv = darcy_lift("liouville", a, aeb)
            b_v2 = lsh.lift_amg_v2(levels, aeb, cycles=2, nu=2, omega_coarse=1.6)
            b_pcg2 = lsh.lift_amg_pcg(levels, aeb, steps=2, nu=2)
            b_j5 = lsh.lift_jacobi_pcg(levels, aeb, steps=5)
            band_op_err.append([
                float(np.linalg.norm(b_liouv - eb) / neb),
                float(np.linalg.norm(b_v2 - eb) / neb),
                float(np.linalg.norm(b_pcg2 - eb) / neb),
                float(np.linalg.norm(b_j5 - eb) / neb),
            ])
        else:
            band_op_err.append([0.0, 0.0, 0.0, 0.0])

    return full_err, band_energy, band_op_err, band_z_cos, band_e_abs


def _lshape_op_worker(job):
    """Worker for L-shape 64^2 exact operator approximation."""
    a, e = job
    n = a.shape[-1]
    m = lsh.lshape_mask(n)
    levels = lsh.build_amg_hierarchy(a)
    ae = np.where(m, lsh.apply_flux(levels[0], e) * (n * n), 0.0)
    z_v2 = lsh.lift_amg_v2(levels, ae, cycles=2, nu=2, omega_coarse=1.6)
    z_pcg2 = lsh.lift_amg_pcg(levels, ae, steps=2, nu=2)
    z_j5 = lsh.lift_jacobi_pcg(levels, ae, steps=5)
    ne = max(float(np.linalg.norm(e[m])), 1e-30)
    return (
        float(np.linalg.norm((z_v2 - e)[m]) / ne),
        float(np.linalg.norm((z_pcg2 - e)[m]) / ne),
        float(np.linalg.norm((z_j5 - e)[m]) / ne),
    )


def _lshape_laplacian_bands(n=64):
    """Exact discrete Dirichlet Laplacian eigenbasis on Omega_L partitioned into Low/Mid/High bands."""
    lvl = lsh.build_fvm_level0(np.ones((n, n), dtype=np.float64))
    mat, active = lsh.assemble_sparse_flux(lvl)
    dense = (mat.toarray()) * (n * n)
    evals, evecs = eigh(dense)
    # Physical effective radial wavenumber k_eff = sqrt(lambda) / pi
    k_eff = np.sqrt(np.maximum(evals, 0.0)) / math.pi
    masks = (k_eff <= 8.0, (k_eff > 8.0) & (k_eff <= 24.0), k_eff > 24.0)
    return active, evecs, masks


def _predict_all(protocol, model, data, grid, frames, batch=16):
    inputs, truth = protocol.split(data, grid, frames)
    preds = []
    for sl in batches(len(truth), batch):
        preds.append(protocol.predict(model, take(inputs, sl), grid, frames))
    return inputs, truth, np.concatenate(preds, axis=0)


def _load_ensemble(pde, seed, protocol, device):
    """Load all 5 frozen FNO ensemble checkpoints for a given PDE & seed."""
    primary = protocol.open_frozen()
    models = [primary]
    if pde == "darcy":
        for i in range(1, 5):
            ck_path = DOWNLOADS / f"2a/seed_{seed}/results/phase2_darcy/seed_{seed}/grid64/FNO/member{i}/fno.pt"
            m = protocol.make_model().to(device)
            m.load_state_dict(torch.load(ck_path, map_location=device, weights_only=True)["model"])
            models.append(m.eval())
    elif pde == "burgers":
        for i in range(1, 5):
            ck_path = DOWNLOADS / f"2db/seed_{seed}/results/phase2d/burgers/seed_{seed}/grid64/ensemble_{i}/FNO/fno.pt"
            m = protocol.make_model().to(device)
            m.load_state_dict(torch.load(ck_path, map_location=device, weights_only=True)["model"])
            models.append(m.eval())
    elif pde == "allen_cahn":
        for i in range(1, 5):
            ck_path = DOWNLOADS / f"2dc/seed_{seed}/results/phase2d/allen_cahn/seed_{seed}/grid64/ensemble_{i}/best.pt"
            m = protocol.make_model().to(device)
            m.load_state_dict(torch.load(ck_path, map_location=device, weights_only=True)["model"])
            models.append(m.eval())
    else:
        raise ValueError(pde)
    return models


def _ensemble_std(protocol, models, data, grid, frames, batch=16):
    """Return (primary_pred, ens_mean, ens_std) using exact ddof=1 across 5 members."""
    inputs, truth = protocol.split(data, grid, frames)
    n = len(truth)
    p0_list, mean_list, std_list = [], [], []
    for sl in batches(n, batch):
        x = take(inputs, sl)
        stack = np.stack([protocol.predict(m, x, grid, frames).astype(np.float64) for m in models], axis=0)
        p0_list.append(stack[0])
        mean_list.append(stack.mean(axis=0))
        std_list.append(stack.std(axis=0, ddof=1))
    return inputs, truth.astype(np.float64), np.concatenate(p0_list), np.concatenate(mean_list), np.concatenate(std_list)


def _get_ens_floor(pde, seed):
    if pde == "burgers":
        m = json.loads((DOWNLOADS / f"2db/seed_{seed}/results/phase2d/burgers/seed_{seed}/grid64/metrics.json").read_text(encoding="utf-8"))
        return float(m["variance_floor"]["ensemble"])
    if pde == "allen_cahn":
        m = json.loads((DOWNLOADS / f"2dc/seed_{seed}/results/phase2d/allen_cahn/seed_{seed}/grid64/metrics.json").read_text(encoding="utf-8"))
        return float(m["variance_floor"]["ensemble"])
    return None


def audit_darcy(device, workers):
    print("[Task 1] Auditing Darcy 64^2 (ID, OOD-A, OOD-B)...", flush=True)
    # Unit-check rectangular FVM operator equivalence
    a0 = arrays(REV_CACHE / "darcy/seed_42/data/id_test.npz")["a"][0].astype(np.float64)
    rng = np.random.default_rng(0)
    u0 = rng.normal(size=(64, 64))
    lv0 = build_rect_fvm_level0(a0)
    diff = np.max(np.abs(lsh.apply_flux(lv0, u0) * (64 ** 2) - darcy_apply_op(a0, u0)))
    assert diff < 1e-10, f"Rectangular FVM stencil mismatch: {diff}"

    splits = (("id_test", "ID"), ("ood_a", "OOD-A"), ("ood_b", "OOD-B"))
    per_seed = {}
    for seed in SEEDS:
        root = REV_CACHE / "darcy" / f"seed_{seed}"
        p = Protocol("darcy", seed, 64, root, FROZEN_ROOT / f"seed_{seed}", device)
        models = _load_ensemble("darcy", seed, p, device)
        cfg, inv, lam = p.config, p.plan["inverse"], p.plan["lambda"]
        # Compute Cal ensemble floor (approx same scale as Val floor) for secondary floored check
        cal_data = arrays(root / "data/cal.npz")
        _, _, _, _, cal_std = _ensemble_std(p, models, cal_data, 64, None, batch=32)
        ens_floor = 0.10 * float(np.sqrt(np.mean(cal_std * cal_std)))

        seed_out = {}
        for sp_key, sp_label in splits:
            data = arrays(root / f"data/{sp_key}.npz")
            inputs, truth, pred, ens_mean, ens_std = _ensemble_std(p, models, data, 64, None, batch=32)
            a = inputs[0].astype(np.float64)
            e = pred - truth
            ae = np.abs(e)
            r = darcy_apply_op(a, pred) - 1.0
            z = darcy_lift(inv, a, r)
            u_z = darcy_lift(inv, a, np.abs(r))

            # Five canonical scales (matching Phase 3/2D conventions + pure variants)
            s_raw_abs = np.abs(r)
            s_raw_p3 = darcy_mix(p3_envelope(r, 0, 2, "reflect", cfg["relative_floor"]), lam)
            s_unsigned = darcy_mix(darcy_phys_scale(u_z, cfg), lam)
            s_signed_raw = np.abs(z)
            s_signed_p3 = darcy_mix(p3_envelope(z, 0, 2, "reflect", cfg["relative_floor"]), lam)
            s_opres = darcy_mix(darcy_phys_scale(z, cfg), lam)
            s_ens_raw = ens_std
            s_ens_floored = np.hypot(ens_std, ens_floor)

            n = len(a)
            ae_flat = ae.reshape(n, -1)
            scales_dict = {
                "raw_residual": s_raw_abs.reshape(n, -1),
                "raw_residual_p3": s_raw_p3.reshape(n, -1),
                "unsigned_inv_env": s_unsigned.reshape(n, -1),
                "signed_z_no_env_raw": s_signed_raw.reshape(n, -1),
                "signed_z_no_env": s_signed_p3.reshape(n, -1),
                "opres_sigma": s_opres.reshape(n, -1),
                "ensemble_std": s_ens_raw.reshape(n, -1),
                "ensemble_std_floored": s_ens_floored.reshape(n, -1),
            }
            scale_res = {k: compute_scale_metrics(ae_flat, v) for k, v in scales_dict.items()}
            # Also evaluate ensemble std against ensemble-mean error
            scale_res["ensemble_std_vs_ens_mean"] = compute_scale_metrics(
                np.abs(ens_mean - truth).reshape(n, -1), s_ens_raw.reshape(n, -1)
            )
            signed_rec = compute_signed_recovery(z.reshape(n, -1), e.reshape(n, -1))

            # Exact operator approximation residual + frequency-band decomposition
            jobs = [(a[i], e[i]) for i in range(n)]
            with ProcessPoolExecutor(max_workers=workers) as pool:
                op_out = list(pool.map(_darcy_rect_op_worker, jobs, chunksize=16))
            full_errs = np.array([x[0] for x in op_out])         # (n, 4)
            band_energy = np.array([x[1] for x in op_out])       # (n, 3)
            band_op_errs = np.array([x[2] for x in op_out])      # (n, 3, 4)
            band_z_cos = np.array([x[3] for x in op_out])        # (n, 3)

            inv_names = ("liouville", "amg_v2", "amg_pcg2", "jacobi_pcg5")
            band_names = ("low_k_le_8", "mid_8_to_24", "high_k_gt_24")
            op_audit = {
                "full_rel_err_mean": {nm: float(full_errs[:, j].mean()) for j, nm in enumerate(inv_names)},
                "full_rel_err_median": {nm: float(np.median(full_errs[:, j])) for j, nm in enumerate(inv_names)},
                "bands": {},
            }
            for bi, bname in enumerate(band_names):
                eb_abs_flat = np.stack([x[4][bi] for x in op_out]).reshape(n, -1).astype(np.float64)
                op_audit["bands"][bname] = {
                    "error_energy_fraction": float(band_energy[:, bi].mean()),
                    "z_band_cos": float(band_z_cos[:, bi].mean()),
                    "op_rel_err_mean": {nm: float(band_op_errs[:, bi, j].mean()) for j, nm in enumerate(inv_names)},
                    "scale_spearman": {
                        "raw_residual": float(_row_spearman(s_raw_abs.reshape(n, -1), eb_abs_flat).mean()),
                        "unsigned_inv_env": float(_row_spearman(s_unsigned.reshape(n, -1), eb_abs_flat).mean()),
                        "opres_sigma": float(_row_spearman(s_opres.reshape(n, -1), eb_abs_flat).mean()),
                        "ensemble_std": float(_row_spearman(s_ens_raw.reshape(n, -1), eb_abs_flat).mean()),
                    },
                }

            seed_out[sp_label] = {
                "signed_recovery": signed_rec,
                "scales": scale_res,
                "operator_audit": op_audit,
            }
            print(f"  Darcy seed={seed} {sp_label:5s}: cos={signed_rec['cos_mean']:.3f}, "
                  f"rho(Raw)={scale_res['raw_residual']['spearman']:.3f}, "
                  f"rho(OpRes)={scale_res['opres_sigma']['spearman']:.3f}, "
                  f"rho(Ens)={scale_res['ensemble_std']['spearman']:.3f}, "
                  f"opErr(Liouv)={op_audit['full_rel_err_mean']['liouville']:.3f}", flush=True)
        per_seed[str(seed)] = seed_out
    return per_seed


def audit_burgers(device):
    print("[Task 1] Auditing Burgers 64 (ID, Visc-OOD, High-OOD)...", flush=True)
    splits = (("id", "ID"), ("ood", "Visc-OOD"), ("ood_high", "High-OOD"))
    per_seed = {}
    for seed in SEEDS:
        root = REV_CACHE / "burgers" / f"seed_{seed}"
        p = Protocol("burgers", seed, 64, root, FROZEN_ROOT / f"seed_{seed}", device)
        models = _load_ensemble("burgers", seed, p, device)
        cfg = p.config
        ens_floor = _get_ens_floor("burgers", seed)
        radius = cfg["spatial_radius"] * 64 / (2.0 * math.pi)
        floor = cfg["envelope_relative_floor"]

        seed_out = {}
        for sp_key, sp_label in splits:
            data = arrays(root / f"data/{sp_key}.npz")
            inputs, truth, pred, ens_mean, ens_std = _ensemble_std(p, models, data, 64, p.frames, batch=32)
            initial, nu = inputs
            full = np.concatenate((initial[:, None], pred), axis=1)
            e = pred - truth
            ae = np.abs(e)
            r = burg_ops.trajectory_residual(full, nu, cfg)
            z = burg_ops.lift("adv_diff", full, r, nu, cfg)
            u_z = burg_ops.lift("adv_diff", full, np.abs(r), nu, cfg)

            s_raw_abs = np.abs(r)
            s_raw_p3 = p3_mix(p3_envelope(r, 0, 1, "wrap", floor), 0.85)
            s_unsigned = p3_mix(p3_envelope(u_z, radius, 1, "wrap", floor), 0.85)
            s_signed_raw = np.abs(z)
            s_signed_p3 = p3_mix(p3_envelope(z, 0, 1, "wrap", floor), 0.85)
            s_opres = p3_mix(burg_ops.envelope(z, cfg), 0.85)
            s_ens_raw = ens_std
            s_ens_floored = np.hypot(ens_std, ens_floor)

            n = len(truth)
            ae_flat = ae.reshape(n, -1)
            scales_dict = {
                "raw_residual": s_raw_abs.reshape(n, -1),
                "raw_residual_p3": s_raw_p3.reshape(n, -1),
                "unsigned_inv_env": s_unsigned.reshape(n, -1),
                "signed_z_no_env_raw": s_signed_raw.reshape(n, -1),
                "signed_z_no_env": s_signed_p3.reshape(n, -1),
                "opres_sigma": s_opres.reshape(n, -1),
                "ensemble_std": s_ens_raw.reshape(n, -1),
                "ensemble_std_floored": s_ens_floored.reshape(n, -1),
            }
            scale_res = {k: compute_scale_metrics(ae_flat, v) for k, v in scales_dict.items()}
            scale_res["ensemble_std_vs_ens_mean"] = compute_scale_metrics(
                np.abs(ens_mean - truth).reshape(n, -1), s_ens_raw.reshape(n, -1)
            )
            signed_rec = compute_signed_recovery(z.reshape(n, -1), e.reshape(n, -1))
            seed_out[sp_label] = {"signed_recovery": signed_rec, "scales": scale_res}
            print(f"  Burgers seed={seed} {sp_label:8s}: cos={signed_rec['cos_mean']:.3f}, "
                  f"rho(Raw)={scale_res['raw_residual']['spearman']:.3f}, "
                  f"rho(OpRes)={scale_res['opres_sigma']['spearman']:.3f}, "
                  f"rho(Ens)={scale_res['ensemble_std']['spearman']:.3f}", flush=True)
        per_seed[str(seed)] = seed_out
    return per_seed


def audit_allen_cahn(device):
    print("[Task 1] Auditing Allen-Cahn 64^2 T=1 (ID, React-OOD)...", flush=True)
    splits = (("id", "ID"), ("ood", "React-OOD"))
    per_seed = {}
    for seed in SEEDS:
        root = REV_CACHE / "allen_cahn" / f"seed_{seed}"
        p = Protocol("allen_cahn", seed, 64, root, FROZEN_ROOT / f"seed_{seed}", device)
        models = _load_ensemble("allen_cahn", seed, p, device)
        cfg = p.config
        ens_floor = _get_ens_floor("allen_cahn", seed)

        seed_out = {}
        for sp_key, sp_label in splits:
            data = arrays(root / f"data/{sp_key}.npz")
            inputs, truth = p.split(data, 64, p.cal_frames)
            n = len(truth)
            # Process in batches of 25 to keep memory small
            m_dim = int(np.prod(truth.shape[1:]))
            scale_names = (
                "raw_residual", "raw_residual_p3", "unsigned_inv_env",
                "signed_z_no_env_raw", "signed_z_no_env", "opres_sigma",
                "ensemble_std", "ensemble_std_floored", "ensemble_std_vs_ens_mean",
            )
            acc = {k: {"sp": [], "pe": [], "pr10": [], "pr05": [], "pc05": [], "pc10": [],
                       "sstar": [], "rho_eff": [], "mean_s": [], "linf": []} for k in scale_names}
            cos_all, rel_all, ls_all, ls_pos_all, alpha_all = [], [], [], [], []

            k10 = math.ceil(0.10 * m_dim)
            k05 = math.ceil(0.05 * m_dim)

            for sl in batches(n, 25):
                x = take(inputs, sl)
                y = truth[sl].astype(np.float64)
                stack = np.stack([p.predict(mdl, x, 64, p.cal_frames).astype(np.float64) for mdl in models], axis=0)
                pred = stack[0]
                ens_mean = stack.mean(axis=0)
                ens_std = stack.std(axis=0, ddof=1)

                initial, nu, rho = x
                full = np.concatenate((initial[:, None].astype(np.float64), pred), axis=1)
                z, r = ac_reconstruct(full, nu, rho, cfg, "reaction_strang")
                jac = ac_jacobian(ac_midpoints(full), rho, cfg)
                u_z = ac_strang_cpu(np.abs(r), nu, cfg, jac)

                e = pred - y
                b = len(y)
                e_flat = e.reshape(b, -1)
                z_flat = z.reshape(b, -1)
                nz = np.linalg.norm(z_flat, axis=1)
                ne = np.linalg.norm(e_flat, axis=1)
                dot = np.sum(z_flat * e_flat, axis=1)
                cos_all.append(np.where((nz > 1e-30) & (ne > 1e-30), dot / (nz * ne), 0.0))
                rel_all.append(np.where(ne > 1e-30, np.linalg.norm(z_flat - e_flat, axis=1) / ne, 0.0))
                als = np.where(nz > 1e-30, dot / (nz * nz), 0.0)
                alpha_all.append(als)
                ls_all.append(np.where(ne > 1e-30, np.linalg.norm(als[:, None] * z_flat - e_flat, axis=1) / ne, 0.0))
                apos = np.maximum(als, 0.0)
                ls_pos_all.append(np.where(ne > 1e-30, np.linalg.norm(apos[:, None] * z_flat - e_flat, axis=1) / ne, 0.0))

                ae_flat = np.abs(e_flat)
                ae_ens_flat = np.abs((ens_mean - y).reshape(b, -1))
                s_map = {
                    "raw_residual": np.abs(r).reshape(b, -1),
                    "raw_residual_p3": p3_mix(p3_envelope(r, 0, 2, "wrap", 0.1), cfg["lambda"]).reshape(b, -1),
                    "unsigned_inv_env": ac_scale(u_z, cfg).reshape(b, -1),
                    "signed_z_no_env_raw": np.abs(z).reshape(b, -1),
                    "signed_z_no_env": p3_mix(p3_envelope(z, 0, 2, "wrap", 0.1), cfg["lambda"]).reshape(b, -1),
                    "opres_sigma": ac_scale(z, cfg).reshape(b, -1),
                    "ensemble_std": ens_std.reshape(b, -1),
                    "ensemble_std_floored": np.hypot(ens_std, ens_floor).reshape(b, -1),
                    "ensemble_std_vs_ens_mean": ens_std.reshape(b, -1),
                }

                # Precompute error masks for primary and ens_mean
                def _err_info(err_f):
                    r_err = rankdata(err_f, axis=1)
                    i10 = np.argpartition(err_f, m_dim - k10, axis=1)[:, m_dim - k10:]
                    i05 = np.argpartition(err_f, m_dim - k05, axis=1)[:, m_dim - k05:]
                    m10 = np.zeros((b, m_dim), dtype=bool)
                    m05 = np.zeros((b, m_dim), dtype=bool)
                    rr = np.arange(b)[:, None]
                    m10[rr, i10] = True
                    m05[rr, i05] = True
                    xs = np.argmax(err_f, axis=1)
                    li = err_f[np.arange(b), xs]
                    return r_err, m10, m05, xs, li

                e_info_prim = _err_info(ae_flat)
                e_info_ens = _err_info(ae_ens_flat)
                rr = np.arange(b)[:, None]

                for sname, s_blk in s_map.items():
                    err_f = ae_ens_flat if sname == "ensemble_std_vs_ens_mean" else ae_flat
                    r_err, e10, e05, xs, li = e_info_ens if sname == "ensemble_std_vs_ens_mean" else e_info_prim
                    s_blk = np.maximum(s_blk, 1e-30)
                    r_sc = rankdata(s_blk, axis=1)
                    acc[sname]["sp"].append(_row_pearson(r_sc, r_err))
                    acc[sname]["pe"].append(_row_pearson(s_blk, err_f))

                    si10 = np.argpartition(s_blk, m_dim - k10, axis=1)[:, m_dim - k10:]
                    si05 = np.argpartition(s_blk, m_dim - k05, axis=1)[:, m_dim - k05:]
                    sm10 = np.zeros((b, m_dim), dtype=bool)
                    sm05 = np.zeros((b, m_dim), dtype=bool)
                    sm10[rr, si10] = True
                    sm05[rr, si05] = True

                    acc[sname]["pr10"].append((e10 & sm10).sum(axis=1) / float(k10))
                    acc[sname]["pr05"].append((e05 & sm05).sum(axis=1) / float(k05))
                    acc[sname]["pc05"].append(sm05[np.arange(b), xs].astype(np.float64))
                    acc[sname]["pc10"].append(sm10[np.arange(b), xs].astype(np.float64))

                    sm = s_blk.mean(axis=1)
                    st = s_blk / np.maximum(sm[:, None], 1e-30)
                    acc[sname]["sstar"].append(st[np.arange(b), xs])
                    msr = (err_f / np.maximum(st, 1e-30)).max(axis=1)
                    acc[sname]["rho_eff"].append(np.where(msr > 1e-30, li / msr, 0.0))
                    acc[sname]["mean_s"].append(sm)
                    acc[sname]["linf"].append(li)

            scale_res = {}
            for sname in scale_names:
                d = acc[sname]
                ms = np.concatenate(d["mean_s"])
                li = np.concatenate(d["linf"])
                ss = np.concatenate(d["sstar"])
                re = np.concatenate(d["rho_eff"])
                scale_res[sname] = {
                    "spearman": float(np.mean(np.concatenate(d["sp"]))),
                    "pearson": float(np.mean(np.concatenate(d["pe"]))),
                    "peak_recall_10": float(np.mean(np.concatenate(d["pr10"]))),
                    "peak_recall_5": float(np.mean(np.concatenate(d["pr05"]))),
                    "peak_capture_5": float(np.mean(np.concatenate(d["pc05"]))),
                    "peak_capture_10": float(np.mean(np.concatenate(d["pc10"]))),
                    "sigma_tilde_xstar_mean": float(np.mean(ss)),
                    "sigma_tilde_xstar_median": float(np.median(ss)),
                    "rho_eff_mean": float(np.mean(re)),
                    "rho_eff_p10": float(np.quantile(re, 0.10)),
                    "rho_cross": float(spearmanr(ms, li).statistic),
                }

            cos_v = np.concatenate(cos_all)
            rel_v = np.concatenate(rel_all)
            signed_rec = {
                "cos_mean": float(np.mean(cos_v)),
                "cos_median": float(np.median(cos_v)),
                "rel_l2_err_mean": float(np.mean(rel_v)),
                "rel_l2_err_median": float(np.median(rel_v)),
                "ls_residual_mean": float(np.mean(np.concatenate(ls_all))),
                "ls_residual_pos_mean": float(np.mean(np.concatenate(ls_pos_all))),
                "alpha_ls_median": float(np.median(np.concatenate(alpha_all))),
            }
            seed_out[sp_label] = {"signed_recovery": signed_rec, "scales": scale_res}
            print(f"  AC T=1 seed={seed} {sp_label:9s}: cos={signed_rec['cos_mean']:.3f}, "
                  f"rho(Raw)={scale_res['raw_residual']['spearman']:.3f}, "
                  f"rho(OpRes)={scale_res['opres_sigma']['spearman']:.3f}, "
                  f"rho(Ens)={scale_res['ensemble_std']['spearman']:.3f}", flush=True)
        per_seed[str(seed)] = seed_out
    return per_seed


def audit_lshape(device, workers):
    print("[Task 1] Auditing L-shape Darcy 64^2 (ID, OOD-A)...", flush=True)
    m = lsh.lshape_mask(64)
    active, evecs, band_masks = _lshape_laplacian_bands(64)
    splits = (("id_test", "ID"), ("ood_a", "OOD-A"))
    per_seed = {}
    for seed in SEEDS:
        sroot = LSHAPE_ROOT / f"seed_{seed}"
        models = []
        for i in range(5):
            ck = torch.load(sroot / f"fno_member_{i}.pt", map_location=device, weights_only=True)
            mdl = lsh.LShapeFNO(lsh.CFG["model"]).to(device)
            mdl.load_state_dict(ck["model"])
            models.append(mdl.eval())
        # Val variance floor
        val_d = arrays(sroot / "data/val.npz")
        val_stack = np.stack([lsh.predict_lshape(mdl, val_d["a"], device, batch=32) for mdl in models], axis=0)
        val_std = val_stack.std(axis=0, ddof=1, dtype=np.float64)
        ens_floor = 0.10 * float(np.sqrt(np.mean((val_std * val_std)[:, m])))

        seed_out = {}
        for sp_key, sp_label in splits:
            data = arrays(sroot / f"data/{sp_key}.npz")
            a = data["a"].astype(np.float64)
            truth = data["u"].astype(np.float64)
            stack = np.stack([lsh.predict_lshape(mdl, data["a"], device, batch=32).astype(np.float64) for mdl in models], axis=0)
            pred = stack[0]
            ens_mean = stack.mean(axis=0)
            ens_std = stack.std(axis=0, ddof=1)
            e = pred - truth
            n = len(a)

            jobs = [(a[i], pred[i]) for i in range(n)]
            with ProcessPoolExecutor(max_workers=workers) as pool:
                lifted = list(pool.map(lsh._lift_one_field, jobs, chunksize=8))
            residuals, z_v2, z_pcg2, z_unsigned, z_j5, _ = (np.stack(arr) for arr in zip(*lifted))

            # Masked no-env floor+mix helper matching Phase 3 convention
            def _no_env_scale(val):
                out = np.zeros_like(val)
                for i in range(n):
                    v2 = (val[i] * val[i])[m]
                    rv = math.sqrt(float(np.mean(v2)))
                    env = np.sqrt(v2 + (lsh.CFG["relative_floor"] * rv) ** 2)
                    rs = math.sqrt(float(np.mean(env * env)))
                    lam = lsh.CFG["lambda"]
                    out[i, m] = np.sqrt((1.0 - lam) * rs * rs + lam * env * env)
                return out

            s_raw_abs = np.abs(residuals)[:, m]
            s_raw_p3 = _no_env_scale(residuals)[:, m]
            s_unsigned = lsh.masked_physical_scale(z_unsigned, lsh.CFG)[:, m]
            s_signed_raw = np.abs(z_pcg2)[:, m]
            s_signed_p3 = _no_env_scale(z_pcg2)[:, m]
            s_opres_pcg2 = lsh.masked_physical_scale(z_pcg2, lsh.CFG)[:, m]
            s_opres_v2 = lsh.masked_physical_scale(z_v2, lsh.CFG)[:, m]
            s_ens_raw = ens_std[:, m]
            s_ens_floored = np.hypot(ens_std, ens_floor)[:, m]

            ae_m = np.abs(e)[:, m]
            scales_dict = {
                "raw_residual": s_raw_abs,
                "raw_residual_p3": s_raw_p3,
                "unsigned_inv_env": s_unsigned,
                "signed_z_no_env_raw": s_signed_raw,
                "signed_z_no_env": s_signed_p3,
                "opres_sigma": s_opres_pcg2,
                "opres_sigma_v2": s_opres_v2,
                "ensemble_std": s_ens_raw,
                "ensemble_std_floored": s_ens_floored,
            }
            scale_res = {k: compute_scale_metrics(ae_m, v) for k, v in scales_dict.items()}
            scale_res["ensemble_std_vs_ens_mean"] = compute_scale_metrics(
                np.abs(ens_mean - truth)[:, m], s_ens_raw
            )
            signed_rec_pcg2 = compute_signed_recovery(z_pcg2[:, m], e[:, m])
            signed_rec_v2 = compute_signed_recovery(z_v2[:, m], e[:, m])

            # Operator error ||B_h A_h e_h - e_h|| / ||e_h||
            op_jobs = [(a[i], e[i]) for i in range(n)]
            with ProcessPoolExecutor(max_workers=workers) as pool:
                op_errs = np.array(list(pool.map(_lshape_op_worker, op_jobs, chunksize=8)))
            inv_names = ("amg_v2", "amg_pcg2", "jacobi_pcg5")

            # Laplacian eigenmode frequency-band decomposition on active cells
            e_act = e[:, m]                 # (n, 3072)
            z_act = z_pcg2[:, m]            # (n, 3072)
            e_coeff = e_act @ evecs         # (n, 3072)
            z_coeff = z_act @ evecs         # (n, 3072)
            ne2 = np.sum(e_act * e_act, axis=1)
            band_names = ("low_k_le_8", "mid_8_to_24", "high_k_gt_24")
            bands_out = {}
            for bm, bname in zip(band_masks, band_names):
                eb = (e_coeff[:, bm]) @ evecs[:, bm].T
                zb = (z_coeff[:, bm]) @ evecs[:, bm].T
                neb = np.linalg.norm(eb, axis=1)
                nzb = np.linalg.norm(zb, axis=1)
                efrac = float(np.mean((neb * neb) / np.maximum(ne2, 1e-30)))
                zcos = float(np.mean(np.where((neb > 1e-30) & (nzb > 1e-30), np.sum(zb * eb, axis=1) / (neb * nzb), 0.0)))
                eb_abs = np.abs(eb)
                bands_out[bname] = {
                    "error_energy_fraction": efrac,
                    "z_band_cos": zcos,
                    "scale_spearman": {
                        "raw_residual": float(_row_spearman(s_raw_abs, eb_abs).mean()),
                        "unsigned_inv_env": float(_row_spearman(s_unsigned, eb_abs).mean()),
                        "opres_sigma": float(_row_spearman(s_opres_pcg2, eb_abs).mean()),
                        "ensemble_std": float(_row_spearman(s_ens_raw, eb_abs).mean()),
                    },
                }

            seed_out[sp_label] = {
                "signed_recovery": signed_rec_pcg2,
                "signed_recovery_v2": signed_rec_v2,
                "scales": scale_res,
                "operator_audit": {
                    "full_rel_err_mean": {nm: float(op_errs[:, j].mean()) for j, nm in enumerate(inv_names)},
                    "full_rel_err_median": {nm: float(np.median(op_errs[:, j])) for j, nm in enumerate(inv_names)},
                    "bands": bands_out,
                },
            }
            print(f"  L-shape seed={seed} {sp_label:5s}: cos(PCG2)={signed_rec_pcg2['cos_mean']:.3f}, "
                  f"rho(Raw)={scale_res['raw_residual']['spearman']:.3f}, "
                  f"rho(OpRes)={scale_res['opres_sigma']['spearman']:.3f}, "
                  f"rho(Ens)={scale_res['ensemble_std']['spearman']:.3f}, "
                  f"opErr(V2/PCG2/J5)={op_errs[:,0].mean():.3f}/{op_errs[:,1].mean():.3f}/{op_errs[:,2].mean():.3f}",
                  flush=True)
        per_seed[str(seed)] = seed_out
    return per_seed


def aggregate_nested(per_seed):
    """Aggregate arbitrary nested dicts of floats across the 5 seeds."""
    first = per_seed[str(SEEDS[0])]

    def _agg(node, path):
        if isinstance(node, dict):
            return {k: _agg(v, path + [k]) for k, v in node.items()}
        vals = []
        for s in SEEDS:
            cur = per_seed[str(s)]
            for k in path:
                cur = cur[k]
            vals.append(cur)
        return _seed_stats(vals)

    return _agg(first, [])


def run_task1(device=None):
    OUT.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    workers = min(8, os.cpu_count() or 4)
    t0 = perf_counter()

    darcy_ps = audit_darcy(device, workers)
    burgers_ps = audit_burgers(device)
    ac_ps = audit_allen_cahn(device)
    lshape_ps = audit_lshape(device, workers)

    out = {
        "seeds": list(SEEDS),
        "elapsed_s": perf_counter() - t0,
        "summary": {
            "darcy": aggregate_nested(darcy_ps),
            "burgers": aggregate_nested(burgers_ps),
            "allen_cahn": aggregate_nested(ac_ps),
            "lshape_darcy": aggregate_nested(lshape_ps),
        },
        "per_seed": {
            "darcy": darcy_ps,
            "burgers": burgers_ps,
            "allen_cahn": ac_ps,
            "lshape_darcy": lshape_ps,
        },
    }
    path = OUT / "task1_fidelity.json"
    path.write_text(json.dumps(finite_json(out), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[Task 1] Done in {out['elapsed_s']:.1f}s -> {path}", flush=True)
    return out


if __name__ == "__main__":
    run_task1()
