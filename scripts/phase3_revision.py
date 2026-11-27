"""One-click reproducible offline revision pipeline (P0 A1/A2/A3/A4-Fig1 + P1 B2/B3).

Zero backbone retraining; never modifies sealed archives under
results/phase2_kaggle_private/downloads/ or phase2d_frozen/.
All outputs are written under results/phase3_revision/.
"""

from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
from time import perf_counter, perf_counter_ns

import numpy as np
import torch
from scipy.stats import t as student_t

import allen_cahn_phase2_data as ac_data
import burgers_phase2 as burg
import darcy_phase2 as darcy
from allen_cahn_phase2_probe import reconstruct as ac_reconstruct, scale as ac_scale
import burgers_phase1_ops as burg_ops
from darcy_data import apply_operator
from darcy_day15 import arrays, digest, estimated_weights, value_at_coverage, weighted_quantile
from darcy_day15_ops import lift, mixture, physical_scale
from phase2d import batches, cutoff_points, take
from phase2d_data import Protocol
from phase2d_statistics import ALPHAS, LocalSlicedCP, relative_l2
from phase3_common import OP, PLAN, calibrate, field_row, finite_json, mix, rms, save_example, summarize
from phase3_summarize import (DOWNLOADS, FORWARD, PHASE2_OP, SEEDS, STAGE,
                              horizon, phase2_metrics, read, result_root, stat)


REV = Path("results/phase3_revision")
CACHE = REV / "cache"
EXAMPLES = REV / "examples"


def _seed_stats(values):
    _, avg, sd, lo, hi = stat(values)
    return {"mean": avg, "sd": sd, "ci95_low": lo, "ci95_high": hi,
            "per_seed": {str(s): float(v) for s, v in zip(SEEDS, values)}}


def _frozen_dir(seed):
    return STAGE / "phase2d_frozen" / f"seed_{seed}"


def _gen_one_split(task):
    pde, seed, split = task
    frozen = _frozen_dir(seed)
    manifest = json.loads((frozen / "frozen_manifest.json").read_text(encoding="utf-8"))
    ref_name = manifest.get("references", {}).get(pde, f"{pde}_reference.json")
    ref = json.loads((frozen / ref_name).read_text(encoding="utf-8"))
    plan = ref["settings" if pde == "burgers" else "config"]
    root = CACHE / pde / f"seed_{seed}" / "data"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{split}.npz"

    if pde == "darcy":
        cfg = darcy.data_config(plan, 64)
        cfg["seed"] = seed
        if not path.exists():
            darcy.generate_split(plan, seed, cfg, split, root, 1)
        expected = ref["tracks"]["64"]["data_sha256"].get(f"data/{split}.npz")
    elif pde == "burgers":
        cfg = plan.copy()
        cfg["seed"] = seed
        if not path.exists():
            burg.generate(cfg, split, root)
        expected = ref["data_sha256"][split]
    elif pde == "allen_cahn":
        cfg = plan.copy()
        cfg["seed"] = seed
        if not path.exists():
            # T=1 (20 frames) is bit-for-bit identical to frames 0..20 of the 40-frame solve.
            nu, rho, coeff = ac_data.initial(cfg, split)
            frames = cfg["frames"] // 2
            parts = []
            for i in range(0, len(nu), 12):
                sl = slice(i, i + 12)
                block = ac_data.solve(coeff[sl], nu[sl], rho[sl], cfg, [64], frames)
                parts.append(block[64])
            u64 = np.concatenate(parts)
            if not np.isfinite(u64).all():
                raise RuntimeError(f"Nonfinite Allen-Cahn solution for seed {seed} {split}.")
            ids = cfg["streams"][split] * 10000 + np.arange(len(nu))
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, nu=nu, rho=rho, id=ids, u64=u64)
            tmp.replace(path)
        expected = ref["data_sha256"][split] if split == "cal" else None
    else:
        raise ValueError(pde)

    with np.load(path) as raw:
        expected_ids = plan["streams"][split] * 10000 + np.arange(plan["counts"][split])
        if not np.array_equal(raw["id"], expected_ids):
            raise ValueError(f"Wrong IDs for {pde} seed {seed} {split}")
    actual_sha = digest(path)
    return {"pde": pde, "seed": seed, "split": split, "path": str(path.as_posix()),
            "sha256": actual_sha, "reference_sha256": expected,
            "exact_hash_match": (actual_sha == expected) if expected else None}


def ensure_data_cache():
    tasks = []
    for seed in SEEDS:
        for split in ("cal", "id_test", "ood_a", "ood_b"):
            tasks.append(("darcy", seed, split))
        for split in ("cal", "id", "ood", "ood_high"):
            tasks.append(("burgers", seed, split))
        for split in ("cal", "id", "ood"):
            tasks.append(("allen_cahn", seed, split))
    workers = min(8, os.cpu_count() or 4)
    t0 = perf_counter()
    print(f"[Cache] Ensuring {len(tasks)} splits across {workers} workers...", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        records = list(ex.map(_gen_one_split, tasks))
    print(f"[Cache] Ready in {perf_counter() - t0:.1f}s.", flush=True)
    return records


def run_lsci_unit_checks():
    """Verify fixed LocalSlicedCP on synthetic i.i.d. data + target-label invariance."""
    rng = np.random.default_rng(2026)
    n_cal, n_test, d, p = 500, 1000, 64, 5
    x_cal = rng.normal(size=(n_cal, p))
    x_test = rng.normal(size=(n_test, p))
    scale_cal = 0.5 + np.abs(x_cal[:, :1])
    scale_test = 0.5 + np.abs(x_test[:, :1])
    r_cal = rng.normal(size=(n_cal, d)) * scale_cal
    r_test = rng.normal(size=(n_test, d)) * scale_test
    top = np.tile(np.arange(8), (n_test, 1))
    pe = np.zeros(n_test)

    fit = LocalSlicedCP(r_cal, x_cal, seed=12345)
    w = fit.prepare(x_test)
    out = fit.evaluate(w, r_test, top, pe)
    cov90 = float(out["0.1"]["native_hit"].mean())
    cov95 = float(out["0.05"]["native_hit"].mean())
    assert 0.87 <= cov90 <= 0.93, f"Synthetic LSCI 90% coverage out of range: {cov90}"
    assert 0.92 <= cov95 <= 0.98, f"Synthetic LSCI 95% coverage out of range: {cov95}"

    # Target-label perturbation canary for label-free weights and rectangular envelope.
    fit1 = LocalSlicedCP(r_cal, x_cal, seed=777)
    w1 = fit1.prepare(x_test[:32])
    out1 = fit1.evaluate(w1, r_test[:32], top[:32], pe[:32])
    fit2 = LocalSlicedCP(r_cal, x_cal, seed=777)
    w2 = fit2.prepare(x_test[:32])
    out2 = fit2.evaluate(w2, r_test[:32] + 100.0, top[:32], pe[:32])
    assert np.array_equal(w1, w2), "LSCI prepare weights changed under label perturbation!"
    assert np.array_equal(out1["0.1"]["width"], out2["0.1"]["width"]), "LSCI envelope width changed under label perturbation!"
    assert np.array_equal(out1["0.05"]["width"], out2["0.05"]["width"]), "LSCI 95% envelope width changed under label perturbation!"
    check = {
        "synthetic_n_cal": n_cal, "synthetic_n_test": n_test,
        "synthetic_native_cov_90": cov90, "synthetic_native_cov_95": cov95,
        "synthetic_native_inf_90": float(out["0.1"]["native_infinite"].mean()),
        "synthetic_native_inf_95": float(out["0.05"]["native_infinite"].mean()),
        "label_perturbation_invariance_passed": True,
    }
    print(f"[LSCI Check] Synthetic i.i.d. native cov: 90%->{100*cov90:.2f}%, 95%->{100*cov95:.2f}%; canary PASS.", flush=True)
    return check


def _predict_all(protocol, model, data, grid, frames, batch=16):
    inputs, truth = protocol.split(data, grid, frames)
    preds = []
    for sl in batches(len(truth), batch):
        preds.append(protocol.predict(model, take(inputs, sl), grid, frames))
    return inputs, truth, np.concatenate(preds, axis=0)


def run_darcy_boundary_revision(device):
    """Task A1: 5-seed O(M log M) spectral boundary lift vs interior-only vs oracle spsolve."""
    arms = ("interior", "spectral_liouville", "dst_local_diagnostic", "exact_oracle")
    per_seed = {}
    timings = {"spectral_liouville": [], "exact_oracle": []}
    linearity_max_err = 0.0

    for seed in SEEDS:
        root = CACHE / "darcy" / f"seed_{seed}"
        p = Protocol("darcy", seed, 64, root, _frozen_dir(seed), device)
        model = p.open_frozen()
        cfg, inv, lam = p.config, p.plan["inverse"], p.plan["lambda"]
        weight_sha = p.manifest["weights"]["darcy64"]["sha256"]

        def eval_split(split):
            data = arrays(root / "data" / f"{split}.npz")
            inputs, truth, pred_f32 = _predict_all(p, model, data, 64, None, batch=32)
            a = inputs[0].astype(float)
            pred = pred_f32.astype(float)
            # Native OpRes verification
            r_nat = apply_operator(a, pred) - 1.0
            z_nat = lift(inv, a, r_nat)
            s_nat = mixture(physical_scale(z_nat, cfg), lam)
            op_row = field_row(pred, truth, s_nat)

            # Stressed trace construction (identical to scripts/phase3_darcy.py)
            n = a.shape[-1]
            x = (np.arange(n) + 0.5) / n
            face = np.sin(np.pi * x)[None, :] * (PLAN["darcy_trace_rms_fraction"] * rms(pred)[:, 0, :])
            ext = np.sinh(np.pi * (1.0 - x))[:, None] / np.sinh(np.pi) * face[:, None, :]
            rhs = np.zeros_like(a)
            rhs[:, 0] = 2.0 * (n ** 2) * a[:, 0] * face
            stressed = pred + ext
            interior = apply_operator(a, stressed) - rhs - 1.0
            z_int = lift(inv, a, interior)
            h_spec = lift(inv, a, rhs)
            h_local = lift("local", a, rhs)
            h_exact = np.stack([lift("exact", cell, boundary) for cell, boundary in zip(a, rhs)])

            # Linearity sanity check: z_int + h_spec == lift(inv, a, apply_operator(a, stressed) - 1)
            z_full = lift(inv, a, apply_operator(a, stressed) - 1.0)
            lin_err = float(np.max(np.abs((z_int + h_spec) - z_full)))

            if seed == SEEDS[0] and split == "id_test":
                # Warmup + 50 single-field timings for spectral vs oracle boundary lift
                lift(inv, a[0], rhs[0])
                lift("exact", a[0], rhs[0])
                for i in range(50):
                    t0 = perf_counter_ns()
                    lift(inv, a[i], rhs[i])
                    timings["spectral_liouville"].append((perf_counter_ns() - t0) / 1e6)
                    t0 = perf_counter_ns()
                    lift("exact", a[i], rhs[i])
                    timings["exact_oracle"].append((perf_counter_ns() - t0) / 1e6)

            rows = {
                OP: op_row,
                "interior": field_row(stressed, truth, mixture(physical_scale(z_int, cfg), lam)),
                "spectral_liouville": field_row(stressed, truth, mixture(physical_scale(z_int + h_spec, cfg), lam)),
                "dst_local_diagnostic": field_row(stressed, truth, mixture(physical_scale(z_int + h_local, cfg), lam)),
                "exact_oracle": field_row(stressed, truth, mixture(physical_scale(z_int + h_exact, cfg), lam)),
            }
            return rows, lin_err

        cal_rows, err_c = eval_split("cal")
        id_rows, err_i = eval_split("id_test")
        linearity_max_err = max(linearity_max_err, err_c, err_i)

        thresholds = calibrate(cal_rows)
        locked_q = {str(a): p.reference["tracks"]["64"]["FNO"]["splits"]["id_test_64"][PHASE2_OP][str(a)]["q"]
                    for a in ALPHAS}
        thresholds[OP] = locked_q

        # Verify OpRes ID coverage delta == 0.0 vs archived 3a and 2da
        op_id_90 = summarize(id_rows[OP], locked_q["0.1"], 0.1)
        ref_3a = read(result_root("darcy", seed) / "metrics.json")["grid64"]["metrics"]["id_test_64"][OP]["0.1"]
        ref_2da = phase2_metrics("darcy", seed, 64)["results"]["id_test_64"][PHASE2_OP]["0.1"]
        cov_delta_3a = abs(op_id_90["coverage"] - ref_3a["coverage"])
        cov_delta_2da = abs(op_id_90["coverage"] - ref_2da["coverage"])
        width_rel_diff = abs(op_id_90["width"] / ref_3a["width"] - 1.0)
        assert cov_delta_3a == 0.0 and cov_delta_2da == 0.0, f"Darcy seed {seed} OpRes cov mismatch: {cov_delta_3a}, {cov_delta_2da}"
        assert width_rel_diff < 5e-4, f"Darcy seed {seed} width mismatch: {width_rel_diff}"

        # Also check exact_oracle against archived 3a "boundary stressed/plus harmonic"
        arch_int = read(result_root("darcy", seed) / "metrics.json")["grid64"]["metrics"]["id_test_64"]["boundary stressed/interior"]["0.1"]
        arch_ex = read(result_root("darcy", seed) / "metrics.json")["grid64"]["metrics"]["id_test_64"]["boundary stressed/plus harmonic"]["0.1"]

        metrics = {arm: {str(a): summarize(id_rows[arm], thresholds[arm][str(a)], a) for a in ALPHAS}
                   for arm in arms}
        per_seed[str(seed)] = {
            "weight_sha256": weight_sha,
            "opres_cov_delta_vs_3a": cov_delta_3a,
            "opres_cov_delta_vs_2da": cov_delta_2da,
            "opres_width_rel_diff_vs_3a": width_rel_diff,
            "oracle_cov_delta_vs_archived_3a": abs(metrics["exact_oracle"]["0.1"]["coverage"] - arch_ex["coverage"]),
            "interior_cov_delta_vs_archived_3a": abs(metrics["interior"]["0.1"]["coverage"] - arch_int["coverage"]),
            "metrics": metrics,
        }
        print(f"[A1 Darcy] seed {seed}: opres_delta=0, interior={metrics['interior']['0.1']['matched_width']:.5f}, "
              f"spectral={metrics['spectral_liouville']['0.1']['matched_width']:.5f}, "
              f"oracle={metrics['exact_oracle']['0.1']['matched_width']:.5f}", flush=True)

    summary = {}
    for arm in arms:
        summary[arm] = {}
        for a in ("0.1", "0.05"):
            summary[arm][a] = {
                k: _seed_stats([per_seed[str(s)]["metrics"][arm][a][k] for s in SEEDS])
                for k in ("coverage", "width", "matched_width", "point_rel_l2")
            }

    timing_summary = {
        k: {"p50_ms": float(np.median(v)), "p90_ms": float(np.quantile(v, 0.9)), "n": len(v)}
        for k, v in timings.items()
    }
    timing_summary["speedup_spectral_over_oracle_p50"] = (
        timing_summary["exact_oracle"]["p50_ms"] / timing_summary["spectral_liouville"]["p50_ms"]
    )
    out = {
        "linearity_max_abs_diff": linearity_max_err,
        "timing_cpu_single_field": timing_summary,
        "summary": summary,
        "per_seed": per_seed,
    }
    (REV / "darcy_boundary_revision.json").write_text(
        json.dumps(finite_json(out), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return out


def run_lsci_revision_and_fig1_examples(device, unit_check):
    """Task A3 (fixed LSCI across 3 PDEs x 5 seeds on ID + OOD) + Task A4 Fig 1 ID examples."""
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    specs = (
        ("darcy", None, (("id_test", "id_test_64"), ("ood_a", "ood_a_64"), ("ood_b", "ood_b_64"))),
        ("burgers", 16, (("id", "id_64"), ("ood", "ood_64"), ("ood_high", "ood_high_64"))),
        ("allen_cahn", 20, (("id", "id_64_20"), ("ood", "ood_64_20"))),
    )
    per_pde = {}
    fig1_examples_info = {}

    for pde, frames, split_pairs in specs:
        per_seed = {}
        for seed in SEEDS:
            root = CACHE / pde / f"seed_{seed}"
            p = Protocol(pde, seed, 64, root, _frozen_dir(seed), device)
            model = p.open_frozen()
            cal_frames = p.cal_frames if pde == "allen_cahn" else p.frames
            cal_data = arrays(root / "data" / "cal.npz")
            xcal, ycal, pcal = _predict_all(p, model, cal_data, 64, cal_frames, batch=16 if pde != "allen_cahn" else 8)
            summary_cal = p.summaries(xcal)
            residual_cal = (ycal - pcal).astype(np.float32)

            # Verify OpRes ID coverage delta == 0.0 vs Phase 2D
            s_cal = np.maximum(p.op_scale(pcal, xcal), 1e-12)
            cal_scores = (np.abs(pcal - ycal).reshape(len(ycal), -1) / s_cal.reshape(len(ycal), -1)).max(axis=1)
            q90_cal = float(np.sort(cal_scores)[math.ceil(0.9 * (len(cal_scores) + 1)) - 1])

            seed_splits = {}
            for split, p2d_label in split_pairs:
                data = arrays(root / "data" / f"{split}.npz")
                xt, yt, pt = _predict_all(p, model, data, 64, cal_frames, batch=16 if pde != "allen_cahn" else 8)

                # Save seed 42 ID FNO examples for Fig 1 (Burgers & Allen-Cahn)
                if seed == SEEDS[0] and split == "id":
                    idx = 0
                    if pde == "burgers":
                        initial, nu = xt[0][:4], xt[1][:4]
                        pred_b = pt[:4].astype(float)
                        full = np.concatenate((initial[:, None], pred_b), axis=1)
                        r_b = burg_ops.trajectory_residual(full, nu, p.config)
                        z_b = burg_ops.lift("adv_diff", full, r_b, nu, p.config)
                        sc_b = mix(burg_ops.envelope(z_b, p.config), 0.85)
                        save_example(EXAMPLES, "burgers_id", input=initial[idx], prediction=pred_b[idx],
                                     truth=yt[idx], residual=r_b[idx], lifted=z_b[idx], scale=sc_b[idx])
                        fig1_examples_info["burgers_id"] = {
                            "file": str((EXAMPLES / "example_burgers_id.npz").as_posix()),
                            "seed": seed, "split": "id_64", "backbone": "FNO", "index": idx,
                            "sample_id": int(data["id"][idx]),
                        }
                    elif pde == "allen_cahn":
                        initial, nu, rho = xt[0][:4], xt[1][:4], xt[2][:4]
                        pred_a = pt[:4].astype(float)
                        full = np.concatenate((initial[:, None], pred_a), axis=1)
                        z_a, r_a = ac_reconstruct(full, nu, rho, p.config, "reaction_strang")
                        sc_a = ac_scale(z_a, p.config)
                        save_example(EXAMPLES, "allen_cahn_id", input=initial[idx], prediction=pred_a[idx],
                                     truth=yt[idx], residual=r_a[idx], lifted=z_a[idx], scale=sc_a[idx])
                        fig1_examples_info["allen_cahn_id"] = {
                            "file": str((EXAMPLES / "example_allen_cahn_id.npz").as_posix()),
                            "seed": seed, "split": "id_64_20", "backbone": "FNO", "index": idx,
                            "sample_id": int(data["id"][idx]),
                        }

                # Check OpRes ID coverage delta on the ID split
                if split in ("id_test", "id"):
                    s_id = np.maximum(p.op_scale(pt, xt), 1e-12)
                    id_scores = (np.abs(pt - yt).reshape(len(yt), -1) / s_id.reshape(len(yt), -1)).max(axis=1)
                    cov_id = float((id_scores <= q90_cal).mean())
                    ref_cov = phase2_metrics(pde, seed, 64)["results"][p2d_label][PHASE2_OP]["0.1"]["coverage"]
                    assert abs(cov_id - ref_cov) <= 1e-12, f"{pde} seed {seed} OpRes ID cov delta: {cov_id} vs {ref_cov}"

                # Fixed LSCI (exact Phase 2D summary features & seed)
                fit = LocalSlicedCP(residual_cal, summary_cal, seed + 9029)
                cov_x = p.summaries(xt)
                w = fit.prepare(cov_x)
                top = cutoff_points(yt, pde)
                out_lsci = fit.evaluate(w, yt - pt, top, relative_l2(pt, yt))

                old_p2d = phase2_metrics(pde, seed, 64)["results"][p2d_label]["LSCI v5 sampled band"]
                split_res = {}
                for alpha in ("0.1", "0.05"):
                    r = out_lsci[alpha]
                    fin = ~r["infinite"]
                    split_res[alpha] = {
                        "n": int(len(r["coverage"])),
                        "native_slice_set_coverage": float(r["native_hit"].mean()),
                        "native_infinite_rate": float(r["native_infinite"].mean()),
                        "envelope_coverage": float(r["coverage"].mean()),
                        "envelope_pixel_coverage": float(r["pixel"].mean()),
                        "envelope_infinite_rate": float(r["infinite"].mean()),
                        "envelope_finite_n": int(fin.sum()),
                        "envelope_finite_coverage": float(r["coverage"][fin].mean()) if fin.any() else None,
                        "envelope_width": float(r["width"].mean()) if fin.all() else "inf",
                        "envelope_finite_width": float(r["width"][fin].mean()) if fin.any() else None,
                        "post_calibration_p50_ms": float(np.median(r["region_ms"])),
                        "post_calibration_p90_ms": float(np.quantile(r["region_ms"], 0.9)),
                        "old_buggy_native_cov": float(old_p2d[alpha]["native_slice_set_coverage"]),
                        "old_buggy_envelope_cov": float(old_p2d[alpha]["coverage"]),
                        "old_buggy_envelope_width": old_p2d[alpha]["width"],
                    }
                seed_splits[p2d_label] = split_res
            per_seed[str(seed)] = {
                "weight_sha256": p.manifest["weights"][p.key]["sha256"],
                "opres_id_cov_delta": 0.0,
                "splits": seed_splits,
            }
            id_lbl = split_pairs[0][1]
            r90 = seed_splits[id_lbl]["0.1"]
            print(f"[A3 LSCI] {pde:10s} seed {seed}: native90 {100*r90['old_buggy_native_cov']:.2f}% -> "
                  f"{100*r90['native_slice_set_coverage']:.2f}% (env {100*r90['envelope_coverage']:.2f}%, "
                  f"p50={r90['post_calibration_p50_ms']:.2f}ms)", flush=True)

        summary = {}
        for _, p2d_label in split_pairs:
            summary[p2d_label] = {}
            for alpha in ("0.1", "0.05"):
                recs = [per_seed[str(s)]["splits"][p2d_label][alpha] for s in SEEDS]
                agg = {}
                for k in ("native_slice_set_coverage", "native_infinite_rate",
                          "envelope_coverage", "envelope_pixel_coverage", "envelope_infinite_rate",
                          "envelope_finite_n", "post_calibration_p50_ms", "post_calibration_p90_ms",
                          "old_buggy_native_cov", "old_buggy_envelope_cov"):
                    agg[k] = _seed_stats([r[k] for r in recs])
                for k in ("envelope_finite_coverage", "envelope_finite_width"):
                    vals = [r[k] for r in recs if r[k] is not None]
                    agg[k] = {"mean": float(np.mean(vals)) if vals else None,
                              "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                              "n_finite_seeds": len(vals)}
                w_vals = [r["envelope_width"] for r in recs if isinstance(r["envelope_width"], (int, float))]
                agg["envelope_width"] = (_seed_stats(w_vals) if len(w_vals) == len(SEEDS)
                                         else {"mean": "inf", "n_finite_seeds": len(w_vals)})
                summary[p2d_label][alpha] = agg
        per_pde[pde] = {"summary": summary, "per_seed": per_seed}

    out = {"unit_checks": unit_check, "fig1_id_examples": fig1_examples_info, "pdes": per_pde}
    (REV / "lsci_revision.json").write_text(
        json.dumps(finite_json(out), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return out


def run_wcp_split_vs_transductive_and_ess_audit():
    """Task A2 (ESS float32 vs float64 audit) + Task B2 (Split-fit vs Transductive WCP)."""
    specs = (
        ("Darcy contrast", "darcy", "grid64_cal", "grid64_ood_a_64", "_input", "Summary"),
        ("Darcy contrast", "darcy", "grid64_cal", "grid64_ood_a_64", "_full", "Full"),
        ("Darcy structure", "darcy", "grid64_cal", "grid64_ood_b_64", "_input", "Summary"),
        ("Darcy structure", "darcy", "grid64_cal", "grid64_ood_b_64", "_full", "Full"),
        ("Burgers viscosity", "burgers", "cal", "ood", "_input", "Summary"),
        ("Burgers viscosity", "burgers", "cal", "ood", "_spectrum", "Spectrum"),
        ("Burgers high", "burgers", "cal", "ood_high", "_input", "Summary"),
        ("Burgers high", "burgers", "cal", "ood_high", "_spectrum", "Spectrum"),
        ("Allen--Cahn $T=1$ OOD", "allen_cahn", "short_cal", "short_ood", "_input", "Summary"),
        ("Allen--Cahn $T=1$ OOD", "allen_cahn", "short_cal", "short_ood", "_full", "Full"),
    )
    cfg = {"base_weight_clip": [0.02, 50.0], "weight_logistic_c": 1.0}
    rows_out = []
    ess_audit = {}

    for label, pde, cal_grp, tgt_grp, feat_key, feat_label in specs:
        modes = {"transductive": [], "split_fit_cal": [], "split_fit_strict": []}
        f32_ess, f64_ess, f32_pt, f64_pt = [], [], [], []
        for seed in SEEDS:
            root = result_root(pde, seed)
            with np.load(root / f"fields_{cal_grp}.npz") as zc, np.load(root / f"fields_{tgt_grp}.npz") as zt:
                cal_op, tgt_op = zc[OP], zt[OP]
                xc_raw, xt_raw = zc[feat_key], zt[feat_key]

            # Match phase2d.py float32 precision for Darcy full grid (root cause of ESS 121.43 vs 121.56)
            if pde == "darcy" and feat_key == "_full":
                xc = xc_raw.astype(np.float32)
                xt = xt_raw.astype(np.float32)
                # Also measure float64 for audit log
                wc64, wt64 = estimated_weights(xc_raw.astype(np.float64), xt_raw.astype(np.float64), cfg)
                _, p64, e64 = weighted_quantile(cal_op[:, 0], wc64, wt64, 0.1)
                f64_ess.append(e64)
                f64_pt.append(float(np.median(p64)))
            else:
                xc, xt = xc_raw, xt_raw

            # 1) Formal Transductive WCP (n_cal = 500)
            wc, wt = estimated_weights(xc, xt, cfg)
            q, p_t, ess = weighted_quantile(cal_op[:, 0], wc, wt, 0.1)
            row_t = summarize(tgt_op, q, 0.1)
            row_t.update({"ess": ess, "p_test_p50": float(np.median(p_t)), "n_fit_cal": len(xc), "n_eval_cal": len(xc)})
            modes["transductive"].append(row_t)
            if pde == "darcy" and feat_key == "_full":
                f32_ess.append(ess)
                f32_pt.append(float(np.median(p_t)))

            # 2a) Cal-split WCP: first 250 Cal + unlabeled target fit weights; last 250 Cal compute quantile
            half = len(xc) // 2
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            def _fit_ratio(x_src, x_tgt):
                est = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
                x_fit = np.concatenate((x_src, x_tgt), axis=0)
                y_fit = np.concatenate((np.zeros(len(x_src), dtype=int), np.ones(len(x_tgt), dtype=int)))
                est.fit(x_fit, y_fit)
                prior = len(x_src) / len(x_tgt)
                return lambda arr: np.clip(np.exp(est.decision_function(arr)) * prior, 0.02, 50.0)

            w_fn = _fit_ratio(xc[:half], xt)
            qs, ps, ess_s = weighted_quantile(cal_op[half:, 0], w_fn(xc[half:]), w_fn(xt), 0.1)
            row_s = summarize(tgt_op, qs, 0.1)
            row_s.update({"ess": ess_s, "p_test_p50": float(np.median(ps)), "n_fit_cal": half, "n_eval_cal": len(xc) - half})
            modes["split_fit_cal"].append(row_s)

            # 2b) Strict Theorem-2 Split-fit WCP: fit on (xc[:250], xt_other_half),
            #     evaluate on completely disjoint (xc[250:], xt_this_half) for both target halves.
            t_half = len(xt) // 2
            q_strict = np.empty(len(xt), dtype=float)
            p_strict = np.empty(len(xt), dtype=float)
            ess_folds = []
            for fit_sl, eval_sl in ((slice(t_half, None), slice(0, t_half)),
                                    (slice(0, t_half), slice(t_half, None))):
                wf = _fit_ratio(xc[:half], xt[fit_sl])
                qf, pf, essf = weighted_quantile(cal_op[half:, 0], wf(xc[half:]), wf(xt[eval_sl]), 0.1)
                q_strict[eval_sl] = qf
                p_strict[eval_sl] = pf
                ess_folds.append(essf)
            row_st = summarize(tgt_op, q_strict, 0.1)
            row_st.update({"ess": float(np.mean(ess_folds)), "p_test_p50": float(np.median(p_strict)),
                           "n_fit_cal": half, "n_eval_cal": len(xc) - half})
            modes["split_fit_strict"].append(row_st)

        if pde == "darcy" and feat_key == "_full" and tgt_grp == "grid64_ood_b_64":
            p2d_ess = [phase2_metrics("darcy", s, 64)["results"]["ood_b_64"]["Full-grid WCP"]["0.1"]["ess"] for s in SEEDS]
            ess_audit["darcy_ood_b_full_c1_cap50"] = {
                "phase2d_sealed_ess_mean": float(np.mean(p2d_ess)),
                "recomputed_float32_ess_mean": float(np.mean(f32_ess)),
                "recomputed_float64_ess_mean": float(np.mean(f64_ess)),
                "float32_vs_phase2d_max_abs_diff": float(np.max(np.abs(np.array(f32_ess) - np.array(p2d_ess)))),
                "recomputed_float32_p_test_p50_mean": float(np.mean(f32_pt)),
                "recomputed_float64_p_test_p50_mean": float(np.mean(f64_pt)),
            }

        def agg_mode(recs):
            fc = [r["finite_coverage"] for r in recs if r["finite_coverage"] is not None]
            fw = [r["finite_width"] for r in recs if r["finite_width"] is not None]
            return {
                "coverage": _seed_stats([r["coverage"] for r in recs]),
                "infinite_rate": _seed_stats([r["infinite_rate"] for r in recs]),
                "finite_n": _seed_stats([r["finite_n"] for r in recs]),
                "finite_coverage": {"mean": float(np.mean(fc)) if fc else None, "n_finite_seeds": len(fc)},
                "finite_width": {"mean": float(np.mean(fw)) if fw else None, "n_finite_seeds": len(fw)},
                "ess": _seed_stats([r["ess"] for r in recs]),
                "p_test_p50": _seed_stats([r["p_test_p50"] for r in recs]),
            }

        entry = {
            "label": label, "pde": pde, "target_group": tgt_grp, "feature": feat_label,
            "transductive": agg_mode(modes["transductive"]),
            "split_fit_cal": agg_mode(modes["split_fit_cal"]),
            "split_fit_strict": agg_mode(modes["split_fit_strict"]),
        }
        rows_out.append(entry)
        tr, sp, ss = entry["transductive"], entry["split_fit_cal"], entry["split_fit_strict"]
        print(f"[B2 WCP] {label:21s} {feat_label:8s} | Transd cov={100*tr['coverage']['mean']:5.2f}% "
              f"inf={100*tr['infinite_rate']['mean']:5.2f}% ESS={tr['ess']['mean']:6.2f} | "
              f"SplitCal cov={100*sp['coverage']['mean']:5.2f}% inf={100*sp['infinite_rate']['mean']:5.2f}% ESS={sp['ess']['mean']:6.2f} | "
              f"StrictT2 cov={100*ss['coverage']['mean']:5.2f}% inf={100*ss['infinite_rate']['mean']:5.2f}% ESS={ss['ess']['mean']:6.2f}", flush=True)

    out = {"ess_precision_audit": ess_audit, "rows": rows_out}
    (REV / "wcp_split_vs_transductive.json").write_text(
        json.dumps(finite_json(out), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return out


def run_allen_cahn_finite_sample_audit():
    """Task B3: Exact finite-sample sampling variance & 95% t-CI for Allen-Cahn T=1 ID coverage."""
    n_cal, n_test, alpha = 500, 500, 0.10
    k = math.ceil((1.0 - alpha) * (n_cal + 1))  # 451
    a_b, b_b = k, n_cal + 1 - k
    mean_p = a_b / (a_b + b_b)
    var_p = (a_b * b_b) / ((a_b + b_b) ** 2 * (a_b + b_b + 1))
    # Total variance of observed test hit fraction = E[P(1-P)]/n_test + Var(P)
    expected_p1mp = mean_p * (1.0 - mean_p) - var_p
    var_hat = expected_p1mp / n_test + var_p
    sd_single_seed = math.sqrt(var_hat)

    ac_t1_fno = [phase2_metrics("allen_cahn", s, 64)["results"]["id_64_20"][PHASE2_OP]["0.1"]["coverage"] for s in SEEDS]
    ac_t2_fno = [horizon("allen_cahn", s)["results"]["FNO"]["id_64"]["jacobian_on"]["T2_same_window"]["0.1"]["coverage"] for s in SEEDS]
    ac_t2_unet = [horizon("allen_cahn", s)["results"]["U-Net"]["id_64"]["jacobian_on"]["T2_same_window"]["0.1"]["coverage"] for s in SEEDS]
    st1 = _seed_stats(ac_t1_fno)
    t_stat = (st1["mean"] - mean_p) / (st1["sd"] / math.sqrt(len(SEEDS)))
    p_val = float(2.0 * (1.0 - student_t.cdf(abs(t_stat), df=len(SEEDS) - 1)))

    res = {
        "n_cal": n_cal, "n_test": n_test, "alpha": alpha, "k_rank": k,
        "theoretical_mean_coverage": mean_p,
        "theoretical_single_seed_sd": sd_single_seed,
        "theoretical_single_seed_sd_pp": 100.0 * sd_single_seed,
        "ac_t1_fno_id": st1,
        "ac_t1_fno_se": st1["sd"] / math.sqrt(len(SEEDS)),
        "ac_t1_fno_ci95_contains_90": bool(st1["ci95_low"] <= 0.90 <= st1["ci95_high"]),
        "ac_t1_fno_two_sided_t_pvalue_vs_nominal": p_val,
        "ac_t2_fno_id_independent": _seed_stats(ac_t2_fno),
        "ac_t2_unet_id_independent": _seed_stats(ac_t2_unet),
    }
    print(f"[B3 AC T=1] mean={100*st1['mean']:.2f}%, SD={100*st1['sd']:.2f}pp, "
          f"95% t-CI=[{100*st1['ci95_low']:.2f}%, {100*st1['ci95_high']:.2f}%], "
          f"theory single-seed SD={100*sd_single_seed:.2f}pp, p={p_val:.3f}", flush=True)
    return res


def main():
    REV.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"[Revision] Running on device={device}", flush=True)

    unit_check = run_lsci_unit_checks()
    cache_manifest = ensure_data_cache()
    a1 = run_darcy_boundary_revision(device)
    a3 = run_lsci_revision_and_fig1_examples(device, unit_check)
    b2 = run_wcp_split_vs_transductive_and_ess_audit()
    b3 = run_allen_cahn_finite_sample_audit()

    audit = {
        "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
        "seeds": SEEDS,
        "cache_manifest": cache_manifest,
        "ess_precision_audit": b2["ess_precision_audit"],
        "allen_cahn_finite_sample_audit": b3,
        "darcy_boundary_linearity_max_abs_diff": a1["linearity_max_abs_diff"],
        "darcy_boundary_timing_cpu_single_field": a1["timing_cpu_single_field"],
        "lsci_unit_checks": unit_check,
        "fig1_id_examples": a3["fig1_id_examples"],
    }
    (REV / "revision_audit.json").write_text(
        json.dumps(finite_json(audit), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"[Revision] Complete. All outputs written to {REV}.", flush=True)


if __name__ == "__main__":
    main()
