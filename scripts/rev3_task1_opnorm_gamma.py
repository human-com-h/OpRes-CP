"""Revision v3 Task 1 (P0):
  Part A: Global induced operator norms ||B_h A_h - I||_{2->2} and ||I - B_h A_h||_{A_h}
          via Power Iteration on Darcy 64^2 (ID, OOD-A, OOD-B) and L-shape Darcy 64^2 (ID, OOD-A),
          contrasting global induced norms against empirical directional defects d_h(e_h).
  Part B: Darcy 64^2 OOD-B spatial fidelity and conformal band evaluation of AMG-V2 & AMG-PCG2
          vs. Liouville DST-II and 5-Member Ensemble across all 5 formal seeds.
  Part C: Five-seed zero-retraining temporal damping gamma in {0.0, 0.05, 0.1, 0.2, 0.4}
          ablation on frozen Burgers (64) and Allen-Cahn (64^2, T=1 and T=2) FNOs.

Strictly read-only against results/phase2_kaggle_private/.
Formal seeds locked to [42, 137, 314, 271, 811].
Outputs written to results/phase3_revision_v3/task1_opnorm_gamma.json.
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
from scipy.linalg import eigh as dense_eigh
import torch

import burgers_phase1_ops as burg_ops
from darcy_data import apply_operator as darcy_apply_op, coefficient as darcy_coeff
from darcy_day15 import arrays, value_at_coverage
from darcy_day15_ops import lift as darcy_lift, mixture as darcy_mix, physical_scale as darcy_phys_scale
from darcy_phase0_eval import quantile
from phase2d import batches, take
from phase2d_data import Protocol
from phase3_common import finite_json, mix as p3_mix
import phase3_p2_lshape_amg as lsh
from phase3_summarize import SEEDS, STAGE, stat
from rev2_task1_fidelity import (
    _ensemble_std,
    _load_ensemble,
    build_rect_amg_hierarchy,
    build_rect_fvm_level0,
    compute_scale_metrics,
    compute_signed_recovery,
)
from rev2_task5_ac_amg import ensure_ac_t2_split


OUT = Path("results/phase3_revision_v3")
REV_CACHE = Path("results/phase3_revision/cache")
T2_CACHE = Path("results/phase3_revision_v2/cache/ac_t2")
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


# ---------- Part A: Power Iteration for Global Operator Induced Norms ----------

def _lift_jacobi_linear(levels, r_pde, steps=5, omega=2.0 / 3.0):
    """Linear symmetric damped Jacobi smoother (for linear induced-norm comparison)."""
    n = levels[0]["n"]
    lvl0 = levels[0]
    b = r_pde / (n * n)
    x = np.zeros_like(b)
    for _ in range(steps):
        res = b - lsh.apply_flux(lvl0, x)
        x += omega * lvl0["inv_diag"] * res
    return x


def _power_iter_l2_defect(apply_a, apply_b, mask, init_vec, max_iter=80, tol=1e-5):
    """Power iteration on D^T D for D = B A - I (assuming A=A^T, B=B^T)."""
    v = np.where(mask, init_vec, 0.0)
    nv = float(np.linalg.norm(v[mask]))
    v = v / max(nv, 1e-30)
    prev = -1.0
    val = 0.0
    for it in range(max_iter):
        dv = apply_b(apply_a(v)) - v
        val = float(np.linalg.norm(dv[mask]))
        dt_dv = apply_a(apply_b(dv)) - dv
        nd = float(np.linalg.norm(dt_dv[mask]))
        if nd <= 1e-30:
            val = 0.0
            break
        v = np.where(mask, dt_dv / nd, 0.0)
        if it >= 3 and abs(val - prev) <= tol * max(val, 1e-12):
            break
        prev = val
    return float(val)


def _power_iter_energy_defect(apply_a, apply_b, mask, init_vec, max_iter=80, tol=1e-5):
    """Power iteration on E^2 in the A_h inner product for E = I - B A (E is A-self-adjoint)."""
    v = np.where(mask, init_vec, 0.0)
    av = apply_a(v)
    va = float(np.sum(v[mask] * av[mask]))
    v = v / math.sqrt(max(va, 1e-30))
    prev = -1.0
    val = 0.0
    for it in range(max_iter):
        ev = v - apply_b(apply_a(v))
        aev = apply_a(ev)
        ev_a = float(np.sum(ev[mask] * aev[mask]))
        val = math.sqrt(max(ev_a, 0.0))
        # Apply E twice to avoid sign oscillation on negative eigenvalues of E
        e2v = ev - apply_b(apply_a(ev))
        ae2v = apply_a(e2v)
        e2_a = float(np.sum(e2v[mask] * ae2v[mask]))
        if e2_a <= 1e-30:
            val = 0.0
            break
        v = np.where(mask, e2v / math.sqrt(e2_a), 0.0)
        if it >= 3 and abs(val - prev) <= tol * max(val, 1e-12):
            break
        prev = val
    return float(val)


def unit_test_power_iteration():
    """Verify Power Iteration induced norms against dense SVD & generalized eigh on n=16."""
    rng = np.random.default_rng(1234)
    # 1) Rectangular 16x16
    a_rect = darcy_coeff(16, 1.3, 99, 16, 0.08, False)
    lv_rect = build_rect_amg_hierarchy(a_rect, min_n=4)
    mat_r, act_r = lsh.assemble_sparse_flux(lv_rect[0])
    A_r = mat_r.toarray() * (16 * 16)
    mask_r = np.ones((16, 16), dtype=bool)

    # 2) L-shape 16x16
    a_lsh = darcy_coeff(16, 1.3, 101, 16, 0.08, False)
    lv_lsh = lsh.build_amg_hierarchy(a_lsh, min_n=4)
    mat_l, act_l = lsh.assemble_sparse_flux(lv_lsh[0])
    A_l = mat_l.toarray() * (16 * 16)
    mask_l = lsh.lshape_mask(16)

    cases = [
        ("rect_liouville", A_r, act_r, mask_r,
         lambda u: darcy_apply_op(a_rect, u),
         lambda r: darcy_lift("liouville", a_rect, r)),
        ("rect_amg_v2_o16", A_r, act_r, mask_r,
         lambda u: darcy_apply_op(a_rect, u),
         lambda r: lsh.lift_amg_v2(lv_rect, r, cycles=2, nu=2, omega_coarse=1.6)),
        ("rect_amg_v2_spd", A_r, act_r, mask_r,
         lambda u: darcy_apply_op(a_rect, u),
         lambda r: lsh.lift_amg_v2(lv_rect, r, cycles=2, nu=2, omega_coarse=1.0)),
        ("lshape_amg_v2_o16", A_l, act_l, mask_l,
         lambda u: np.where(mask_l, lsh.apply_flux(lv_lsh[0], u) * (16 * 16), 0.0),
         lambda r: lsh.lift_amg_v2(lv_lsh, r, cycles=2, nu=2, omega_coarse=1.6)),
        ("lshape_amg_v2_spd", A_l, act_l, mask_l,
         lambda u: np.where(mask_l, lsh.apply_flux(lv_lsh[0], u) * (16 * 16), 0.0),
         lambda r: lsh.lift_amg_v2(lv_lsh, r, cycles=2, nu=2, omega_coarse=1.0)),
    ]

    summary = {}
    for cname, A, act, mask, app_a, app_b in cases:
        m = len(act)
        B = np.zeros((m, m), dtype=np.float64)
        for j in range(m):
            e_j = np.zeros((16, 16), dtype=np.float64)
            e_j.ravel()[act[j]] = 1.0
            B[:, j] = app_b(e_j).ravel()[act]
        sym_err = float(np.max(np.abs(B - B.T)))
        assert sym_err < 1e-11, f"{cname} operator asymmetry: {sym_err}"

        D = B @ A - np.eye(m)
        exact_l2 = float(np.linalg.svd(D, compute_uv=False)[0])
        # A-induced norm of E = I - B A equals max |lambda| of A^{1/2} (I - B A) A^{-1/2}
        evals = dense_eigh(A - A @ B @ A, A, eigvals_only=True)
        exact_energy = float(np.max(np.abs(evals)))

        init = rng.normal(size=(16, 16))
        pi_l2 = _power_iter_l2_defect(app_a, app_b, mask, init, max_iter=120, tol=1e-7)
        pi_en = _power_iter_energy_defect(app_a, app_b, mask, init, max_iter=120, tol=1e-7)
        err_l2 = abs(pi_l2 - exact_l2) / max(exact_l2, 1e-12)
        err_en = abs(pi_en - exact_energy) / max(exact_energy, 1e-12)
        assert err_l2 < 1e-5 and err_en < 1e-5, (
            f"Power iteration mismatch on {cname}: l2={pi_l2} vs {exact_l2}, en={pi_en} vs {exact_energy}"
        )
        summary[cname] = {
            "sym_err": sym_err,
            "exact_l2": exact_l2,
            "pi_l2": pi_l2,
            "exact_energy": exact_energy,
            "pi_energy": pi_en,
        }
    print("[Task 1 Unit Check] Power Iteration vs exact SVD/eigh on n=16 PASS:",
          {k: (round(v["pi_l2"], 4), round(v["pi_energy"], 4)) for k, v in summary.items()}, flush=True)
    return summary


def _opnorm_rect_worker(job):
    a, e, draw_seed = job
    n = a.shape[-1]
    mask = np.ones((n, n), dtype=bool)
    lv = build_rect_amg_hierarchy(a, min_n=4)
    rng = np.random.default_rng(draw_seed)
    init = rng.normal(size=(n, n))

    app_a = lambda u: darcy_apply_op(a, u)
    ops = {
        "liouville": lambda r: darcy_lift("liouville", a, r),
        "amg_v2": lambda r: lsh.lift_amg_v2(lv, r, cycles=2, nu=2, omega_coarse=1.6),
        "amg_v2_spd": lambda r: lsh.lift_amg_v2(lv, r, cycles=2, nu=2, omega_coarse=1.0),
        "jacobi5_lin": lambda r: _lift_jacobi_linear(lv, r, steps=5, omega=2.0 / 3.0),
    }

    out = {}
    ae = app_a(e)
    ne = max(float(np.linalg.norm(e)), 1e-30)
    ne_a = math.sqrt(max(float(np.sum(e * ae)), 1e-30))
    for nm, app_b in ops.items():
        d_dir = app_b(ae) - e
        l2_dir = float(np.linalg.norm(d_dir) / ne)
        en_dir = math.sqrt(max(float(np.sum(d_dir * app_a(d_dir))), 0.0)) / ne_a
        l2_glob = _power_iter_l2_defect(app_a, app_b, mask, init)
        en_glob = _power_iter_energy_defect(app_a, app_b, mask, init)
        out[nm] = {
            "dir_l2": l2_dir,
            "dir_energy": en_dir,
            "global_l2": l2_glob,
            "global_energy": en_glob,
        }

    # Nonlinear Krylov controls: directional defect on e_h + empirical max over 20 Gaussian dirs
    vecs = rng.normal(size=(20, n, n))
    for nm, app_b in (
        ("amg_pcg2", lambda r: lsh.lift_amg_pcg(lv, r, steps=2, nu=2)),
        ("jacobi_pcg5", lambda r: lsh.lift_jacobi_pcg(lv, r, steps=5)),
    ):
        d_dir = app_b(ae) - e
        l2_dir = float(np.linalg.norm(d_dir) / ne)
        en_dir = math.sqrt(max(float(np.sum(d_dir * app_a(d_dir))), 0.0)) / ne_a
        rand_l2, rand_en = [], []
        for v in vecs:
            av = app_a(v)
            dv = app_b(av) - v
            rand_l2.append(float(np.linalg.norm(dv) / max(float(np.linalg.norm(v)), 1e-30)))
            rand_en.append(math.sqrt(max(float(np.sum(dv * app_a(dv))), 0.0)) /
                           math.sqrt(max(float(np.sum(v * av)), 1e-30)))
        out[nm] = {
            "dir_l2": l2_dir,
            "dir_energy": en_dir,
            "rand20_max_l2": float(max(rand_l2)),
            "rand20_max_energy": float(max(rand_en)),
        }
    return out


def _opnorm_lshape_worker(job):
    a, e, draw_seed = job
    n = a.shape[-1]
    mask = lsh.lshape_mask(n)
    lv = lsh.build_amg_hierarchy(a, min_n=4)
    rng = np.random.default_rng(draw_seed)
    init = rng.normal(size=(n, n))

    app_a = lambda u: np.where(mask, lsh.apply_flux(lv[0], u) * (n * n), 0.0)
    ops = {
        "amg_v2": lambda r: lsh.lift_amg_v2(lv, r, cycles=2, nu=2, omega_coarse=1.6),
        "amg_v2_spd": lambda r: lsh.lift_amg_v2(lv, r, cycles=2, nu=2, omega_coarse=1.0),
        "jacobi5_lin": lambda r: _lift_jacobi_linear(lv, r, steps=5, omega=2.0 / 3.0),
    }

    out = {}
    ae = app_a(e)
    ne = max(float(np.linalg.norm(e[mask])), 1e-30)
    ne_a = math.sqrt(max(float(np.sum(e[mask] * ae[mask])), 1e-30))
    for nm, app_b in ops.items():
        d_dir = app_b(ae) - e
        l2_dir = float(np.linalg.norm(d_dir[mask]) / ne)
        en_dir = math.sqrt(max(float(np.sum(d_dir[mask] * app_a(d_dir)[mask])), 0.0)) / ne_a
        l2_glob = _power_iter_l2_defect(app_a, app_b, mask, init)
        en_glob = _power_iter_energy_defect(app_a, app_b, mask, init)
        out[nm] = {
            "dir_l2": l2_dir,
            "dir_energy": en_dir,
            "global_l2": l2_glob,
            "global_energy": en_glob,
        }

    vecs = np.where(mask[None], rng.normal(size=(20, n, n)), 0.0)
    for nm, app_b in (
        ("amg_pcg2", lambda r: lsh.lift_amg_pcg(lv, r, steps=2, nu=2)),
        ("jacobi_pcg5", lambda r: lsh.lift_jacobi_pcg(lv, r, steps=5)),
    ):
        d_dir = app_b(ae) - e
        l2_dir = float(np.linalg.norm(d_dir[mask]) / ne)
        en_dir = math.sqrt(max(float(np.sum(d_dir[mask] * app_a(d_dir)[mask])), 0.0)) / ne_a
        rand_l2, rand_en = [], []
        for v in vecs:
            av = app_a(v)
            dv = app_b(av) - v
            rand_l2.append(float(np.linalg.norm(dv[mask]) / max(float(np.linalg.norm(v[mask])), 1e-30)))
            rand_en.append(math.sqrt(max(float(np.sum(dv[mask] * app_a(dv)[mask])), 0.0)) /
                           math.sqrt(max(float(np.sum(v[mask] * av[mask])), 1e-30)))
        out[nm] = {
            "dir_l2": l2_dir,
            "dir_energy": en_dir,
            "rand20_max_l2": float(max(rand_l2)),
            "rand20_max_energy": float(max(rand_en)),
        }
    return out


def run_part_a_opnorms(device, workers, n_fields=20):
    print(f"[Task 1A] Global induced operator norms via Power Iteration ({n_fields} fields/seed x 5 seeds)...", flush=True)
    unit = unit_test_power_iteration()

    # 1) Rectangular Darcy 64^2 (ID, OOD-A, OOD-B)
    rect_splits = (("id_test", "ID"), ("ood_a", "OOD-A"), ("ood_b", "OOD-B"))
    rect_per_seed = {s: {} for s in map(str, SEEDS)}
    for seed in SEEDS:
        root = REV_CACHE / "darcy" / f"seed_{seed}"
        p = Protocol("darcy", seed, 64, root, FROZEN_ROOT / f"seed_{seed}", device)
        model = p.open_frozen()
        for sp_key, sp_label in rect_splits:
            d = arrays(root / f"data/{sp_key}.npz")
            a = d["a"][:n_fields].astype(np.float64)
            u = d["u"][:n_fields].astype(np.float64)
            pred = p.predict(model, (d["a"][:n_fields],), 64, None).astype(np.float64)
            err = pred - u
            jobs = [(a[i], err[i], int(seed * 100000 + i * 17 + 1)) for i in range(n_fields)]
            with ProcessPoolExecutor(max_workers=workers) as pool:
                rows = list(pool.map(_opnorm_rect_worker, jobs, chunksize=4))
            agg = {}
            for op_name in rows[0]:
                agg[op_name] = {
                    k: float(np.mean([r[op_name][k] for r in rows]))
                    for k in rows[0][op_name]
                }
                if "global_energy" in rows[0][op_name]:
                    vals = [r[op_name]["global_energy"] for r in rows]
                    agg[op_name]["global_energy_max"] = float(np.max(vals))
                    agg[op_name]["frac_lt_1"] = float(np.mean([v < 1.0 for v in vals]))
            rect_per_seed[str(seed)][sp_label] = agg
            print(f"  Darcy seed={seed} {sp_label:5s} | Liouv L2(dir/glob)={agg['liouville']['dir_l2']:.3f}/{agg['liouville']['global_l2']:.3f} | "
                  f"AMG-V2 L2={agg['amg_v2']['global_l2']:.3f}, En={agg['amg_v2']['global_energy']:.3f} (max={agg['amg_v2']['global_energy_max']:.3f}) | "
                  f"V2-SPD En={agg['amg_v2_spd']['global_energy']:.3f} (max={agg['amg_v2_spd']['global_energy_max']:.3f})", flush=True)

    # 2) L-shape Darcy 64^2 (ID, OOD-A)
    lsh_splits = (("id_test", "ID"), ("ood_a", "OOD-A"))
    lsh_per_seed = {s: {} for s in map(str, SEEDS)}
    for seed in SEEDS:
        sroot = LSHAPE_ROOT / f"seed_{seed}"
        ck = torch.load(sroot / "fno_member_0.pt", map_location=device, weights_only=True)
        mdl = lsh.LShapeFNO(lsh.CFG["model"]).to(device)
        mdl.load_state_dict(ck["model"])
        mdl.eval()
        for sp_key, sp_label in lsh_splits:
            d = arrays(sroot / f"data/{sp_key}.npz")
            a = d["a"][:n_fields].astype(np.float64)
            u = d["u"][:n_fields].astype(np.float64)
            pred = lsh.predict_lshape(mdl, d["a"][:n_fields], device, batch=32).astype(np.float64)
            err = pred - u
            jobs = [(a[i], err[i], int(seed * 100000 + i * 23 + 7)) for i in range(n_fields)]
            with ProcessPoolExecutor(max_workers=workers) as pool:
                rows = list(pool.map(_opnorm_lshape_worker, jobs, chunksize=4))
            agg = {}
            for op_name in rows[0]:
                agg[op_name] = {
                    k: float(np.mean([r[op_name][k] for r in rows]))
                    for k in rows[0][op_name]
                }
                if "global_energy" in rows[0][op_name]:
                    vals = [r[op_name]["global_energy"] for r in rows]
                    agg[op_name]["global_energy_max"] = float(np.max(vals))
                    agg[op_name]["frac_lt_1"] = float(np.mean([v < 1.0 for v in vals]))
            lsh_per_seed[str(seed)][sp_label] = agg
            print(f"  Lshape seed={seed} {sp_label:5s} | AMG-V2 L2={agg['amg_v2']['global_l2']:.3f}, "
                  f"En={agg['amg_v2']['global_energy']:.3f} (max={agg['amg_v2']['global_energy_max']:.3f}) | "
                  f"V2-SPD En={agg['amg_v2_spd']['global_energy']:.3f} (max={agg['amg_v2_spd']['global_energy_max']:.3f})", flush=True)

    def _agg_seeds(pseed, splits):
        out = {}
        for _, sp_label in splits:
            out[sp_label] = {}
            op_names = list(pseed[str(SEEDS[0])][sp_label].keys())
            for op_name in op_names:
                mkeys = list(pseed[str(SEEDS[0])][sp_label][op_name].keys())
                out[sp_label][op_name] = {
                    mk: _seed_stats([pseed[str(s)][sp_label][op_name][mk] for s in SEEDS])
                    for mk in mkeys
                }
        return out

    return {
        "n_fields_per_seed": n_fields,
        "unit_test_n16": unit,
        "darcy_rect": {"summary": _agg_seeds(rect_per_seed, rect_splits), "per_seed": rect_per_seed},
        "darcy_lshape": {"summary": _agg_seeds(lsh_per_seed, lsh_splits), "per_seed": lsh_per_seed},
    }


# ---------- Part B: Darcy 64^2 OOD-B AMG-V2 / AMG-PCG2 Spatial Localization & Bands ----------

def _rect_amg_lift_worker(job):
    a, r = job
    lv = build_rect_amg_hierarchy(a, min_n=4)
    z_v2 = lsh.lift_amg_v2(lv, r, cycles=2, nu=2, omega_coarse=1.6)
    z_pcg2 = lsh.lift_amg_pcg(lv, r, steps=2, nu=2)
    return z_v2, z_pcg2


def run_part_b_darcy_oodb_amg(device, workers):
    print("[Task 1B] Darcy 64^2 OOD-B AMG-V2 & AMG-PCG2 spatial fidelity & conformal bands (5 seeds x 500 fields)...", flush=True)
    per_seed = {}
    for seed in SEEDS:
        root = REV_CACHE / "darcy" / f"seed_{seed}"
        p = Protocol("darcy", seed, 64, root, FROZEN_ROOT / f"seed_{seed}", device)
        models = _load_ensemble("darcy", seed, p, device)
        cfg, lam = p.config, p.plan["lambda"]

        # Load Cal, ID, OOD-B
        data_cal = arrays(root / "data/cal.npz")
        data_id = arrays(root / "data/id_test.npz")
        data_oodb = arrays(root / "data/ood_b.npz")

        def _prep_split(d):
            inp, truth, pred, _, ens_std = _ensemble_std(p, models, d, 64, None, batch=32)
            a = inp[0].astype(np.float64)
            e = pred - truth
            r = darcy_apply_op(a, pred) - 1.0
            z_liouv = darcy_lift("liouville", a, r)
            jobs = [(a[i], r[i]) for i in range(len(a))]
            with ProcessPoolExecutor(max_workers=workers) as pool:
                out = list(pool.map(_rect_amg_lift_worker, jobs, chunksize=16))
            z_v2 = np.stack([x[0] for x in out])
            z_pcg2 = np.stack([x[1] for x in out])
            s_liouv = darcy_mix(darcy_phys_scale(z_liouv, cfg), lam)
            s_v2 = darcy_mix(darcy_phys_scale(z_v2, cfg), lam)
            s_pcg2 = darcy_mix(darcy_phys_scale(z_pcg2, cfg), lam)
            return a, truth, pred, e, z_liouv, z_v2, z_pcg2, s_liouv, s_v2, s_pcg2, ens_std

        cal_sp = _prep_split(data_cal)
        id_sp = _prep_split(data_id)
        oodb_sp = _prep_split(data_oodb)

        def _cal_q(e, s):
            n = len(e)
            sc = (np.abs(e).reshape(n, -1) / np.maximum(s.reshape(n, -1), 1e-12)).max(axis=1)
            return quantile(sc, 0.10)

        qs = {
            "liouville": _cal_q(cal_sp[3], cal_sp[7]),
            "amg_v2": _cal_q(cal_sp[3], cal_sp[8]),
            "amg_pcg2": _cal_q(cal_sp[3], cal_sp[9]),
            "ensemble_std": _cal_q(cal_sp[3], cal_sp[10]),
        }

        def _eval_band(e, s, q):
            n = len(e)
            sf = np.maximum(s.reshape(n, -1), 1e-12)
            sc = (np.abs(e).reshape(n, -1) / sf).max(axis=1)
            q_emp = value_at_coverage(sc, 0.90)
            return {
                "coverage": float((sc <= q).mean()),
                "actual_width": float((2.0 * q * sf.mean(axis=1)).mean()),
                "matched90_width": float(2.0 * q_emp * sf.mean()),
            }

        n = len(oodb_sp[0])
        ae_oodb = np.abs(oodb_sp[3]).reshape(n, -1)
        per_seed[str(seed)] = {
            "ood_b_fidelity": {
                "liouville": compute_scale_metrics(ae_oodb, oodb_sp[7].reshape(n, -1)),
                "amg_v2": compute_scale_metrics(ae_oodb, oodb_sp[8].reshape(n, -1)),
                "amg_pcg2": compute_scale_metrics(ae_oodb, oodb_sp[9].reshape(n, -1)),
                "ensemble_std": compute_scale_metrics(ae_oodb, oodb_sp[10].reshape(n, -1)),
            },
            "ood_b_signed_recovery": {
                "liouville": compute_signed_recovery(oodb_sp[4].reshape(n, -1), oodb_sp[3].reshape(n, -1)),
                "amg_v2": compute_signed_recovery(oodb_sp[5].reshape(n, -1), oodb_sp[3].reshape(n, -1)),
                "amg_pcg2": compute_signed_recovery(oodb_sp[6].reshape(n, -1), oodb_sp[3].reshape(n, -1)),
            },
            "id_bands": {
                "liouville": _eval_band(id_sp[3], id_sp[7], qs["liouville"]),
                "amg_v2": _eval_band(id_sp[3], id_sp[8], qs["amg_v2"]),
                "amg_pcg2": _eval_band(id_sp[3], id_sp[9], qs["amg_pcg2"]),
            },
            "ood_b_bands": {
                "liouville": _eval_band(oodb_sp[3], oodb_sp[7], qs["liouville"]),
                "amg_v2": _eval_band(oodb_sp[3], oodb_sp[8], qs["amg_v2"]),
                "amg_pcg2": _eval_band(oodb_sp[3], oodb_sp[9], qs["amg_pcg2"]),
            },
        }
        v2f = per_seed[str(seed)]["ood_b_fidelity"]["amg_v2"]
        v2b = per_seed[str(seed)]["ood_b_bands"]["amg_v2"]
        lf = per_seed[str(seed)]["ood_b_fidelity"]["liouville"]
        print(f"  Darcy OOD-B seed={seed} | Liouv rho={lf['spearman']:.3f}, pc5={100*lf['peak_capture_5']:.1f}% | "
              f"AMG-V2 rho={v2f['spearman']:.3f}, pc5={100*v2f['peak_capture_5']:.1f}%, "
              f"cov={100*v2b['coverage']:.2f}%, mw={v2b['matched90_width']:.4f}", flush=True)

    # Aggregate 5 seeds
    summary = {}
    for block in ("ood_b_fidelity", "ood_b_signed_recovery", "id_bands", "ood_b_bands"):
        summary[block] = {}
        methods = list(per_seed[str(SEEDS[0])][block].keys())
        for mname in methods:
            mkeys = list(per_seed[str(SEEDS[0])][block][mname].keys())
            summary[block][mname] = {
                mk: _seed_stats([per_seed[str(s)][block][mname][mk] for s in SEEDS])
                for mk in mkeys
            }
    return {"summary": summary, "per_seed": per_seed}


# ---------- Part C: Temporal Damping gamma Ablation (Burgers & Allen-Cahn) ----------

def run_part_c_gamma_ablation(device):
    print("[Task 1C] Five-seed gamma in {0.0, 0.05, 0.1, 0.2, 0.4} ablation (Burgers & Allen-Cahn)...", flush=True)
    gammas = (0.0, 0.05, 0.1, 0.2, 0.4)

    # Unit check: gamma=0.0 on Burgers phi(0)=1 and no NaN/Inf
    p1_zero, _ = burg_ops.phi(np.array([0.0, -0.1]))
    assert abs(p1_zero[0] - 1.0) < 1e-14 and np.isfinite(p1_zero).all(), f"phi(0) failed: {p1_zero}"

    # 1) Burgers 64 (5 seeds)
    burg_per_seed = {}
    for seed in SEEDS:
        root = REV_CACHE / "burgers" / f"seed_{seed}"
        p = Protocol("burgers", seed, 64, root, FROZEN_ROOT / f"seed_{seed}", device)
        model = p.open_frozen()
        base_cfg = p.config
        splits = {}
        for sp in ("cal", "id", "ood", "ood_high"):
            d = arrays(root / f"data/{sp}.npz")
            inp, truth = p.split(d, 64, p.frames)
            preds = [p.predict(model, take(inp, sl), 64, p.frames) for sl in batches(len(truth), 64)]
            pred = np.concatenate(preds, axis=0).astype(np.float64)
            full = np.concatenate((inp[0][:, None].astype(np.float64), pred), axis=1)
            r = burg_ops.trajectory_residual(full, inp[1].astype(np.float64), base_cfg)
            splits[sp] = (full, inp[1].astype(np.float64), pred, truth.astype(np.float64), r)

        seed_out = {}
        for g in gammas:
            cfg = base_cfg | {"gamma": g}
            # Calibrate
            full_c, nu_c, pred_c, y_c, r_c = splits["cal"]
            z_c = burg_ops.lift("adv_diff", full_c, r_c, nu_c, cfg)
            s_c = np.maximum(p3_mix(burg_ops.envelope(z_c, cfg), 0.85), 1e-12)
            assert np.isfinite(z_c).all() and np.isfinite(s_c).all()
            sc_c = (np.abs(pred_c - y_c).reshape(len(y_c), -1) / s_c.reshape(len(y_c), -1)).max(axis=1)
            q90 = quantile(sc_c, 0.10)

            g_res = {}
            for sp, sp_label in (("id", "ID"), ("ood", "Visc-OOD"), ("ood_high", "High-OOD")):
                full_t, nu_t, pred_t, y_t, r_t = splits[sp]
                z_t = burg_ops.lift("adv_diff", full_t, r_t, nu_t, cfg)
                s_t = np.maximum(p3_mix(burg_ops.envelope(z_t, cfg), 0.85), 1e-12)
                e_t = pred_t - y_t
                n = len(y_t)
                ef = e_t.reshape(n, -1)
                zf = z_t.reshape(n, -1)
                sf = s_t.reshape(n, -1)
                sc_t = (np.abs(ef) / sf).max(axis=1)
                q_emp = value_at_coverage(sc_t, 0.90)
                rec = compute_signed_recovery(zf, ef)
                sm = compute_scale_metrics(np.abs(ef), sf)
                g_res[sp_label] = {
                    "cos": rec["cos_mean"],
                    "spearman": sm["spearman"],
                    "rho_eff": sm["rho_eff_mean"],
                    "coverage": float((sc_t <= q90).mean()),
                    "actual_width": float((2.0 * q90 * sf.mean(axis=1)).mean()),
                    "matched90_width": float(2.0 * q_emp * sf.mean()),
                }
            seed_out[f"{g:g}"] = g_res
        burg_per_seed[str(seed)] = seed_out
        print(f"  Burgers gamma seed={seed} | ID mw(0.0/0.1/0.4)="
              f"{seed_out['0']['ID']['matched90_width']:.4f}/"
              f"{seed_out['0.1']['ID']['matched90_width']:.4f}/"
              f"{seed_out['0.4']['ID']['matched90_width']:.4f}, "
              f"Visc cov={100*seed_out['0']['Visc-OOD']['coverage']:.1f}%/"
              f"{100*seed_out['0.1']['Visc-OOD']['coverage']:.1f}%/"
              f"{100*seed_out['0.4']['Visc-OOD']['coverage']:.1f}%", flush=True)

    # 2) Allen-Cahn 64^2 T=1 & T=2 (GPU strang + scale + cosine/spearman)
    @torch.inference_mode()
    def _ac_lift_and_scale_gpu(full_np, nu_np, rho_np, cfg):
        import allen_cahn_phase2_data as ac_data
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
        sig = torch.sqrt((1.0 - lam) * 1.01 * avg + lam * smooth).clamp_min(1e-12)
        return z.cpu().numpy().astype(np.float64), sig.cpu().numpy().astype(np.float64)

    ac_per_seed = {}
    for seed in SEEDS:
        root = REV_CACHE / "allen_cahn" / f"seed_{seed}"
        p = Protocol("allen_cahn", seed, 64, root, FROZEN_ROOT / f"seed_{seed}", device)
        model = p.open_frozen()
        base_cfg = p.config

        t1_data = {s: arrays(root / f"data/{s}.npz") for s in ("cal", "id", "ood")}
        t2_data = {s: ensure_ac_t2_split(seed, s, base_cfg, device) for s in ("cal", "id", "ood")}

        def _rollout(d, frames):
            inp, y = p.split(d, 64, frames)
            pred = np.concatenate([p.predict(model, take(inp, sl), 64, frames) for sl in batches(len(y), 16)], axis=0)
            full = np.concatenate((inp[0][:, None], pred), axis=1).astype(np.float64)
            return full, y.astype(np.float64), inp[1].astype(np.float64), inp[2].astype(np.float64)

        t1_roll = {s: _rollout(t1_data[s], 20) for s in ("cal", "id", "ood")}
        t2_roll = {s: _rollout(t2_data[s], 40) for s in ("cal", "id", "ood")}

        def _eval_ac_split(roll, cfg, need_spatial=True):
            full, y, nu, rho = roll
            n = len(y)
            sc_list, sm_list, cos_list, sp_list = [], [], [], []
            for sl in batches(n, 25):
                z, sig = _ac_lift_and_scale_gpu(full[sl], nu[sl], rho[sl], cfg)
                e = full[sl, 1:] - y[sl]
                b = len(e)
                ef = e.reshape(b, -1)
                zf = z.reshape(b, -1)
                sf = sig.reshape(b, -1)
                sc_list.append((np.abs(ef) / sf).max(axis=1))
                sm_list.append(sf.mean(axis=1))
                if need_spatial:
                    nz = np.linalg.norm(zf, axis=1)
                    ne = np.linalg.norm(ef, axis=1)
                    dot = np.sum(zf * ef, axis=1)
                    cos_list.append(np.where((nz > 1e-30) & (ne > 1e-30), dot / (nz * ne), 0.0))
                    # Sample 81920 coords -> exact rankdata per trajectory is ~25ms/batch
                    from scipy.stats import rankdata
                    rs = rankdata(sf, axis=1)
                    re = rankdata(np.abs(ef), axis=1)
                    rsc = rs - rs.mean(axis=1, keepdims=True)
                    rec = re - re.mean(axis=1, keepdims=True)
                    den = np.linalg.norm(rsc, axis=1) * np.linalg.norm(rec, axis=1)
                    sp_list.append(np.where(den > 1e-30, np.sum(rsc * rec, axis=1) / den, 0.0))
            out = {"scores": np.concatenate(sc_list), "smean": np.concatenate(sm_list)}
            if need_spatial:
                out["cos"] = float(np.mean(np.concatenate(cos_list)))
                out["spearman"] = float(np.mean(np.concatenate(sp_list)))
            return out

        seed_out = {}
        for g in gammas:
            cfg = base_cfg | {"gamma": g}
            t1_cal = _eval_ac_split(t1_roll["cal"], cfg, need_spatial=False)
            t2_cal = _eval_ac_split(t2_roll["cal"], cfg, need_spatial=False)
            q1_90 = quantile(t1_cal["scores"], 0.10)
            q1_95 = quantile(t1_cal["scores"], 0.05)
            q2_90 = quantile(t2_cal["scores"], 0.10)

            def _pack(ev, q_use):
                q_emp = value_at_coverage(ev["scores"], 0.90)
                return {
                    "cos": ev["cos"],
                    "spearman": ev["spearman"],
                    "coverage": float((ev["scores"] <= q_use).mean()),
                    "actual_width": float((2.0 * q_use * ev["smean"]).mean()),
                    "matched90_width": float(2.0 * q_emp * ev["smean"].mean()),
                }

            t1_id = _eval_ac_split(t1_roll["id"], cfg, True)
            t1_ood = _eval_ac_split(t1_roll["ood"], cfg, True)
            t2_id = _eval_ac_split(t2_roll["id"], cfg, True)
            t2_ood = _eval_ac_split(t2_roll["ood"], cfg, True)

            t2_id_pack = _pack(t2_id, q2_90)
            t2_id_pack["t1_95reuse_coverage"] = float((t2_id["scores"] <= q1_95).mean())
            t2_id_pack["t1_95reuse_actual_width"] = float((2.0 * q1_95 * t2_id["smean"]).mean())

            seed_out[f"{g:g}"] = {
                "T1_ID": _pack(t1_id, q1_90),
                "T1_ReactOOD": _pack(t1_ood, q1_90),
                "T2_ID": t2_id_pack,
                "T2_ReactOOD": _pack(t2_ood, q2_90),
            }
        ac_per_seed[str(seed)] = seed_out
        print(f"  AC gamma seed={seed} | T1_ID mw(0.0/0.2/0.4)="
              f"{seed_out['0']['T1_ID']['matched90_width']:.4f}/"
              f"{seed_out['0.2']['T1_ID']['matched90_width']:.4f}/"
              f"{seed_out['0.4']['T1_ID']['matched90_width']:.4f} | "
              f"T2_ID cov(0.0/0.2/0.4)="
              f"{100*seed_out['0']['T2_ID']['coverage']:.1f}%/"
              f"{100*seed_out['0.2']['T2_ID']['coverage']:.1f}%/"
              f"{100*seed_out['0.4']['T2_ID']['coverage']:.1f}% | "
              f"T2_reuse={100*seed_out['0']['T2_ID']['t1_95reuse_coverage']:.1f}%/"
              f"{100*seed_out['0.2']['T2_ID']['t1_95reuse_coverage']:.1f}%/"
              f"{100*seed_out['0.4']['T2_ID']['t1_95reuse_coverage']:.1f}%", flush=True)

    def _agg_gamma(pseed):
        out = {}
        for gk in pseed[str(SEEDS[0])]:
            out[gk] = {}
            for sk in pseed[str(SEEDS[0])][gk]:
                mkeys = list(pseed[str(SEEDS[0])][gk][sk].keys())
                out[gk][sk] = {
                    mk: _seed_stats([pseed[str(s)][gk][sk][mk] for s in SEEDS])
                    for mk in mkeys
                }
        return out

    return {
        "gammas": list(gammas),
        "burgers": {"summary": _agg_gamma(burg_per_seed), "per_seed": burg_per_seed},
        "allen_cahn": {"summary": _agg_gamma(ac_per_seed), "per_seed": ac_per_seed},
    }


def run_task1(device=None, workers=6):
    OUT.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = perf_counter()

    part_a = run_part_a_opnorms(device, workers=workers, n_fields=20)
    part_b = run_part_b_darcy_oodb_amg(device, workers=workers)
    part_c = run_part_c_gamma_ablation(device)

    out = {
        "seeds": list(SEEDS),
        "elapsed_s": perf_counter() - t0,
        "part_a_opnorms": part_a,
        "part_b_darcy_oodb_amg": part_b,
        "part_c_gamma_ablation": part_c,
    }
    path = OUT / "task1_opnorm_gamma.json"
    path.write_text(json.dumps(finite_json(out), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[Task 1] Done in {out['elapsed_s']:.1f}s -> {path}", flush=True)
    return out


if __name__ == "__main__":
    run_task1()
