#!/usr/bin/env python3
"""Tier 1 (<30s CPU, zero weights): Core mathematical, operator, and artifact self-check.

Runs without any downloaded .pt checkpoint files or GPU. Verifies:
  1. Locked configs/*.json presence & configs/phase3.json SHA-256 integrity.
  2. 2D unstructured P1 FEM elasticity + Vector SA-AMG-V2 unit tests (--unit-only).
  3. SPD-AMG-PCG2 Proposition 2: degree-1 positive homogeneity & Chebyshev bound.
  4. Structured PDE operators smoke tests (Darcy DST-II, Burgers causal SL, AC Strang).
  5. Committed five-seed summary JSON/CSV integrity & manuscript headline consistency.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

PHASE3_SHA256 = "f241e0d197c57a348219d9756d7a7f01f794d63c7f431f926ba993bb3c37c363"


def check_configs() -> str:
    expected = [
        "phase0_darcy.json",
        "phase0_day15.json",
        "phase1_burgers.json",
        "phase2_ac_formal.json",
        "phase2_ac_probe.json",
        "phase2_burgers.json",
        "phase2_darcy.json",
        "phase3.json",
    ]
    for name in expected:
        p = ROOT / "configs" / name
        assert p.is_file(), f"Missing config: {p}"
    sha = hashlib.sha256((ROOT / "configs" / "phase3.json").read_bytes()).hexdigest()
    assert sha == PHASE3_SHA256, f"configs/phase3.json SHA-256 mismatch: {sha} != {PHASE3_SHA256}"
    return f"8 configs present; phase3.json SHA-256 verified ({sha[:12]}...)"


def check_elasticity_fem_unit() -> str:
    res = subprocess.run(
        [sys.executable, "-X", "utf8", str(ROOT / "scripts" / "rev4_task1_elasticity_fem.py"), "--unit-only"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        raise AssertionError(
            f"rev4_task1_elasticity_fem.py --unit-only failed (rc={res.returncode}):\n"
            f"{res.stdout}\n{res.stderr}"
        )
    out = res.stdout.strip().splitlines()
    last_line = out[-1] if out else "PASS"
    return f"P1 FEM Patch/Gâteaux/Vector SA-AMG-V2 PASS ({last_line[:68]})"


def check_spd_amg_pcg2() -> str:
    import darcy_data as dd
    import phase3_p2_lshape_amg as lsh
    import rev4_task2_nonlinear_pcg_jha as t2

    n = 32
    rng = np.random.default_rng(42)
    a = dd.coefficient(n, 1.0, 42, 32, 0.08, False)
    # Manufactured smooth error vanishing on outer cells
    x = (np.arange(n) + 0.5) / n
    xx, yy = np.meshgrid(x, x, indexing="ij")
    e_true = np.sin(np.pi * xx) * np.sin(2.0 * np.pi * yy) + 0.35 * np.sin(3.0 * np.pi * xx) * np.sin(np.pi * yy)

    lv = t2.build_rect_amg_hierarchy(a, min_n=4)
    mask = np.ones((n, n), dtype=bool)
    app_a = lambda u: dd.apply_operator(a, u)
    app_m = lambda r: lsh.amg_vcycle(lv, r / (n * n), idx=0, nu=2, omega_coarse=1.0)
    app_pcg2 = lambda r: lsh.lift_amg_pcg(lv, r, steps=2, nu=2)

    init = rng.normal(size=(n, n))
    e1_norm = t2._power_iter_energy_defect(app_a, app_m, mask, init, max_iter=80, tol=1e-5)
    lmax = t2._power_iter_ma_max(app_a, app_m, mask, init, max_iter=60, tol=1e-5)
    lmin = max(1.0 - e1_norm, 1e-6)
    kappa = max(lmax / lmin, 1.0)
    tau = (math.sqrt(kappa) - 1.0) / (math.sqrt(kappa) + 1.0)
    cheb_exact = (2.0 * (tau ** 2)) / (1.0 + tau ** 4)

    r = app_a(e_true)
    z = app_pcg2(r)
    diff = z - e_true
    ae = r
    adiff = app_a(diff)
    dir_en = math.sqrt(max(float(np.sum(diff * adiff)), 0.0)) / math.sqrt(max(float(np.sum(e_true * ae)), 1e-30))
    assert dir_en <= cheb_exact + 1e-6, f"Chebyshev bound violated: {dir_en} > {cheb_exact}"

    hom_err = 0.0
    denom = max(float(np.max(np.abs(z))), 1e-30)
    for c in (0.1, 3.7, 100.0):
        zc = app_pcg2(c * r)
        hom_err = max(hom_err, float(np.max(np.abs(zc - c * z)) / (c * denom)))
    assert hom_err < 1e-13, f"Positive homogeneity failed: {hom_err:.3e}"
    return f"hom_err={hom_err:.2e} (<1e-13), dir_E={dir_en:.4f} <= Cheb={cheb_exact:.4f} (kappa={kappa:.2f})"


def check_structured_pde_ops() -> str:
    import allen_cahn_phase2_probe as ac
    import burgers_phase1_ops as bg
    import darcy_data as dd
    import darcy_day15_ops as dops

    # 1) Darcy DST-II exact spectral inversion on unit permeability & positive scale on variable a
    n = 32
    x = (np.arange(n) + 0.5) / n
    xx, yy = np.meshgrid(x, x, indexing="ij")
    u_man = np.sin(np.pi * xx) * np.sin(np.pi * yy)
    a_unit = np.ones((n, n), dtype=float)
    r_unit = dd.apply_operator(a_unit, u_man)
    z_unit = dops.lift("liouville", a_unit, r_unit)
    darcy_err = float(np.linalg.norm(z_unit - u_man) / np.linalg.norm(u_man))
    assert darcy_err < 1e-12, f"Darcy DST-II exactness failed: {darcy_err:.3e}"
    a_var = dd.coefficient(n, 1.0, 137, 16, 0.08, False)
    z_var = dops.lift("liouville", a_var, dd.apply_operator(a_var, u_man))
    s_darcy = dops.mixture(dops.physical_scale(z_var, {"physical_radius": 0.04, "relative_floor": 0.1}), 0.85)
    assert np.isfinite(s_darcy).all() and float(s_darcy.min()) > 0.0

    # 2) Burgers causal semi-Lagrangian smoke check (64 points, 5 frames)
    nb, nt = 64, 5
    xb = 2.0 * np.pi * np.arange(nb) / nb
    tt = np.linspace(0.0, 0.2, nt + 1)[:, None]
    u_bg = (0.4 * np.sin(xb[None, :] - 0.5 * tt) * np.exp(-0.02 * tt))[None, :, :]
    nu_bg = np.array([0.02], dtype=float)
    cfg_bg = {"frame_dt": 0.05, "advection_substeps": 2, "gamma": 0.1, "spatial_radius": 0.24, "envelope_relative_floor": 0.1}
    res_bg = bg.trajectory_residual(u_bg, nu_bg, cfg_bg)
    z_bg = bg.lift("adv_diff", u_bg, res_bg, nu_bg, cfg_bg)
    s_bg = bg.mixed(bg.envelope(z_bg, cfg_bg), 0.85)
    assert np.isfinite(s_bg).all() and float(s_bg.min()) > 0.0

    # 3) Allen-Cahn Strang splitting smoke check (32x32, 4 frames)
    nac, nfac = 32, 4
    xa = (np.arange(nac) + 0.5) / nac * 2.0 * np.pi
    xax, xay = np.meshgrid(xa, xa, indexing="ij")
    u_ac = np.stack([(0.3 * np.sin(xax) * np.cos(xay) * np.exp(-0.05 * k)) for k in range(nfac + 1)], axis=0)[None]
    cfg_ac = {"dt": 0.05, "gamma": 0.2, "reaction_clip": 2.0, "positive_damping": 0.0, "max_growth": 1.5, "spatial_radius": 0.16, "lambda": 0.85}
    nu_ac = np.array([0.02], dtype=float)
    rho_ac = np.array([0.8], dtype=float)
    z_ac_on, _ = ac.reconstruct(u_ac, nu_ac, rho_ac, cfg_ac, "reaction_strang", reaction_jacobian=True)
    z_ac_off, _ = ac.reconstruct(u_ac, nu_ac, rho_ac, cfg_ac, "reaction_strang", reaction_jacobian=False)
    s_ac = ac.scale(z_ac_on, cfg_ac)
    assert np.isfinite(z_ac_off).all() and np.isfinite(s_ac).all() and float(s_ac.min()) > 0.0
    return f"Darcy DST-II err={darcy_err:.2e}; Burgers SL & AC Strang finite positive scales"


def check_summary_integrity() -> str:
    required = [
        "results/phase2d_summary.csv",
        "results/phase2d_paired.csv",
        "results/phase2d_mixed_gates.csv",
        "results/phase2d_acceptance.json",
        "results/phase3/phase3_audit.json",
        "results/phase3/phase3_theory_checks.json",
        "results/phase3_p2/lshape_darcy/lshape_summary.json",
        "results/phase3_p2/ac_gpu_timing/ac_gpu_benchmark.json",
        "results/phase3_revision_v2/task1_fidelity.json",
        "results/phase3_revision_v3/task1_opnorm_gamma.json",
        "results/phase3_revision_v3/task2_lsci_exact.json",
        "results/phase3_revision_v3/task3_kernel_horizon.json",
        "results/phase3_revision_v4/task1_elasticity.json",
        "results/phase3_revision_v4/task2_nonlinear_pcg_jha.json",
        "results/phase3_revision_v4/task3_physics_gate.json",
    ]
    for s in (42, 137, 314, 271, 811):
        required.append(f"results/phase2_kaggle_private/phase2d_frozen/seed_{s}/frozen_manifest.json")
    for rel in required:
        p = ROOT / rel
        assert p.is_file(), f"Missing summary artifact: {rel}"

    # Verify headline five-seed metrics match manuscript claims
    with (ROOT / "results" / "phase2d_summary.csv").open(encoding="utf-8-sig") as f:
        p2d = {(r["batch"], r["grid"], r["group"], r["method"], r["alpha"], r["metric"]): float(r["mean"])
               for r in csv.DictReader(f) if r["mean"]}

    for b, g, grp, expected_pct in [("2da", "64", "id_test_64", 67.54), ("2db", "64", "id_64", 47.40), ("2dc", "64", "id_64_20", 92.48)]:
        w_op = p2d[(b, g, grp, "OpRes-CP", "0.1", "matched_width")]
        w_en = p2d[(b, g, grp, "5-FNO Ensemble + CP | mean center", "0.1", "matched_width")]
        shrink = 100.0 * (1.0 - w_op / w_en)
        assert abs(shrink - expected_pct) < 0.4, f"{b} shrink {shrink:.2f}% != {expected_pct}%"

    ls = json.loads((ROOT / "results/phase3_p2/lshape_darcy/lshape_summary.json").read_text(encoding="utf-8"))
    ls_shrink = 100.0 * ls["id_width_reduction"]["amg_pcg2_vs_ensemble_mean"]["mean"]
    assert abs(ls_shrink - 68.61) < 0.05, f"L-shape shrink {ls_shrink:.2f}% != 68.61%"

    t1 = json.loads((ROOT / "results/phase3_revision_v4/task1_elasticity.json").read_text(encoding="utf-8"))
    el_id = t1["summary"]["splits"]["id"]
    disp_ratio = el_id["displacement"]["5-Ens. (mean center)"]["matched90_width"]["mean"] / el_id["displacement"]["OpRes-CP (Vector SA-AMG-V2)"]["matched90_width"]["mean"]
    vm_ratio = el_id["von_mises"]["5-Ens. (mean center)"]["matched90_width"]["mean"] / el_id["von_mises"]["OpRes-CP (Vector SA-AMG-V2)"]["matched90_width"]["mean"]
    assert abs(disp_ratio - 4.80) < 0.05 and abs(vm_ratio - 4.53) < 0.05, f"Elasticity ratios {disp_ratio:.2f}/{vm_ratio:.2f}"
    return f"{len(required)} artifacts verified; headline shrinks & ratios match manuscript"


def main() -> None:
    t0 = perf_counter()
    checks = [
        ("1. Locked Configs & SHA-256", check_configs),
        ("2. 2D P1 FEM & Vector SA-AMG-V2", check_elasticity_fem_unit),
        ("3. SPD-AMG-PCG2 (Proposition 2)", check_spd_amg_pcg2),
        ("4. Structured PDE Operators Smoke", check_structured_pde_ops),
        ("5. Summary JSON/CSV Consistency", check_summary_integrity),
    ]
    print("=" * 88)
    print("OpRes-CP Tier 1 Quick Verification (CPU-only, Zero Frozen Weights)")
    print("=" * 88)
    rows = []
    failed = False
    for name, fn in checks:
        ts = perf_counter()
        try:
            detail = fn()
            status = "PASS"
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            status = "FAIL"
            failed = True
        dt = perf_counter() - ts
        rows.append((name, status, f"{dt:.2f}s", detail))

    for name, status, dt, detail in rows:
        print(f"  [{status}] {name:<35} ({dt:>6}) | {detail}")
    print("=" * 88)
    total = perf_counter() - t0
    print(f"Result: {'FAIL' if failed else 'ALL 5 CHECKS PASS'} in {total:.2f}s")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
