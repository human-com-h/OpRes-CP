import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.fft import rfft

from burgers_phase1_data import CONFIG, ROOT, generator_audit
from burgers_phase1_model import infer, model_from_checkpoint, torch_device
from burgers_phase1_ops import envelope, gradient, lift, mixed, rms, trajectory_residual
from darcy_day15 import (arrays, band_summary, comparison, digest, estimated_weights,
                         evaluate_bands, value_at_coverage, weighted_quantile, write_json)
from darcy_day15_ops import spatial_rho, timed, timing_summary
from darcy_phase0_eval import boot_interval, quantile


INVERSES = ("diff", "adv_diff")
UNIFORM, FIXED, PI, ABS, SCALAR, OP = METHODS = (
    "Uniform-CP", "CP-B", "Full-Field PI-CP", "Inverse-after-Abs", "Physics-Scalar CP", "OpRes-CP")


def lifted_bands(u, nu, r, choice, config):
    z = lift(choice["inverse"], u, r, nu, config)
    phys = envelope(z, config)
    return z, phys


def same_cov(error, scale, alpha):
    ratios = (error/scale).reshape(len(error), -1)
    q = value_at_coverage(ratios.max(axis=1), 1-alpha)
    return float(2*q*scale.mean()), float(np.mean(ratios.max(axis=1) <= q))


def checkpoint_hash():
    return digest(ROOT/"fno.pt")


def split_ids(data, split, config):
    ids = data["id"]
    if (np.unique(ids).size != len(ids) or len(ids) != config["counts"][split]
            or np.any(ids//10000 != config["streams"][split])):
        raise RuntimeError(f"Incorrect Burgers data stream for {split}")


def audit_lock(config, model, device, epoch):
    if (ROOT/"lock.json").exists():
        raise RuntimeError("Use the existing Burgers lock for evaluation.")
    path = ROOT/"val.npz"
    val = arrays(path)
    split_ids(val, "val", config)
    u, nu = val["u64"], val["nu"]
    pred = infer(model, device, u[:, 0], nu).astype(float)
    r = trajectory_residual(pred, nu, config)
    error = abs(pred[:, 1:]-u[:, 1:])
    rows = []
    diagnostics = {}
    for inv in INVERSES:
        z, phys = lifted_bands(pred, nu, r, {"inverse": inv}, config)
        # Changes after frame 8 must leave outputs through t_8 unchanged.
        changed = pred[:12].copy()
        changed[:, 9:] *= -0.8
        rr = trajectory_residual(changed, nu[:12], config)
        test_lift = lift(inv, changed, rr, nu[:12], config)
        scaled = envelope(lift(inv, pred[:12], -3*r[:12], nu[:12], config), config)
        diagnostics[inv] = {
            "spatial_temporal_rho": spatial_rho(phys, error),
            "max_causal_prefix_diff": float(np.max(abs(test_lift[:, :8]-z[:12, :8]))),
            "scale_homogeneity_rel": float(np.max(abs(scaled-3*phys[:12]))/np.max(3*phys[:12]))}
        for lam in config["lambdas"]:
            width, coverage = same_cov(error, mixed(phys, lam), config["alpha"])
            rows.append({"inverse": inv, "lambda": lam, "width": width, "coverage": coverage})
    choice = min(rows, key=lambda x: x["width"])
    fixed = error.mean(axis=0)
    fixed = np.maximum(fixed, .05*fixed.mean())
    np.save(ROOT/"fixed_cp_b.npy", fixed)
    sources = [*Path("scripts").glob("burgers_phase1*.py"), CONFIG]
    lock = {"config": config, "candidate": choice, "val_rows": rows, "diagnostics": diagnostics,
            "checkpoint_epoch": epoch, "checkpoint_sha256": checkpoint_hash(),
            "val_sha256": digest(path), "fixed_sha256": digest(ROOT/"fixed_cp_b.npy"),
            "pi_lambda": float(1/np.median(abs(r))), "sources": {p.as_posix(): digest(p) for p in sources}}
    write_json(ROOT/"lock.json", lock)
    print(f"Burgers Val locked: {choice}", flush=True)


def interpolate_fixed(fixed, n):
    if n == fixed.shape[-1]:
        return fixed
    old = np.arange(fixed.shape[-1])/fixed.shape[-1]
    new = np.arange(n)/n
    return np.stack([np.interp(new, old, row, period=1) for row in fixed])


def method_scales(u, nu, r, lock, fixed):
    config, choice = lock["config"], lock["candidate"]
    z, phys = lifted_bands(u, nu, r, choice, config)
    abs_phys = envelope(lift(choice["inverse"], u, abs(r), nu, config), config)
    return {
        UNIFORM: np.ones_like(r),
        FIXED: np.broadcast_to(interpolate_fixed(fixed, r.shape[-1]), r.shape),
        PI: 1+lock["pi_lambda"]*abs(r),
        ABS: abs_phys,
        SCALAR: mixed(phys, 0),
        OP: mixed(phys, choice["lambda"])
    }, z


def input_features(u, nu, r, z):
    # Use known viscosity, input steepness, and residual summaries; no truth.
    return np.column_stack([np.log(nu), np.max(abs(gradient(u[:, 0])), axis=-1),
                            (rms(z)/rms(u[:, 1:])).ravel(), rms(r).ravel()])


def infer_parts(data, split, n, config, model, device):
    split_ids(data, split, config)
    truth, nu = data[f"u{n}"], data["nu"]
    prediction = infer(model, device, truth[:, 0], nu).astype(float)
    r = trajectory_residual(prediction, nu, config)
    error = abs(prediction[:, 1:]-truth[:, 1:])
    return prediction, r, error


def calibration(data, config, lock, fixed, model, device):
    prediction, r, error = infer_parts(data, "cal", 64, config, model, device)
    scales, z = method_scales(prediction, data["nu"], r, lock, fixed)
    alpha = config["alpha"]
    scores = {}
    for name, scale in scales.items():
        ratio = (error/scale).reshape(len(error), -1)
        fields = ratio.max(axis=1)
        scores[name] = {"field_score": fields, "q": quantile(fields, alpha),
                        "q_pixel": quantile(ratio.ravel(), alpha)}
    return scores, input_features(prediction, data["nu"], r, z)


def spectrum(x):
    n = x.shape[-1]
    power = abs(rfft(x, norm="forward"))**2
    power[..., 1:-1] *= 2
    k = np.arange(n//2+1)
    energy = power.mean(axis=(0, 1))
    bands = [(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 129)]
    total = energy.sum()
    rows = [float(energy[(k >= lo) & (k < hi)].sum()/total) for lo, hi in bands]
    return {"edges": [0, 4, 8, 16, 32, 64, 129], "fractions": rows,
            "k_ge_16_fraction": sum(rows[3:]), "rms": float(np.sqrt(total))}


def timing(data, n, config, lock, model, device):
    fields = zip(data[f"u{n}"][:72, 0], data["nu"][:72])
    times = {name: [] for name in ("fno", "residual", "diff_inverse", "adv_diff_inverse",
                                   "chosen_end_to_end")}
    for i, (u0, nu) in enumerate(fields):
        p, tf = timed(lambda: infer(model, device, u0[None], np.array([nu]), batch=1).astype(float))
        rr, tr = timed(lambda: trajectory_residual(p, np.array([nu]), config))
        alts = {}
        for inverse in INVERSES:
            alts[inverse] = timed(lambda: lift(inverse, p, rr, np.array([nu]), config))
        z, tk = alts[lock["candidate"]["inverse"]]
        _, ts = timed(lambda: mixed(envelope(z, config), lock["candidate"]["lambda"]))
        if i >= 8:
            for key, elapsed in zip(times, (tf, tr, alts["diff"][1], alts["adv_diff"][1], tf+tr+tk+ts)):
                times[key].append(elapsed)
    return {name: timing_summary(x) for name, x in times.items()}


def evaluate(data, split, n, config, lock, fixed, cal_scores, cal_features, model, device):
    prediction, r, error = infer_parts(data, split, n, config, model, device)
    nu, truth = data["nu"], data[f"u{n}"]
    ss, z = method_scales(prediction, nu, r, lock, fixed)
    flat = error.reshape(len(error), -1)
    slopes = abs(gradient(truth[:, 1:].astype(float))).reshape(len(flat), -1)
    k = int(np.ceil(.10*slopes.shape[1]))
    top = np.argpartition(slopes, -k, axis=1)[:, -k:].copy()
    results, rows = {}, {}
    alpha = config["alpha"]
    if split == "ood":
        x = input_features(prediction, nu, r, z)
        (wc, wt), fit_ms = timed(lambda: estimated_weights(cal_features, x, {
            "weight_logistic_c": 1., "base_weight_clip": config["weight_clip"]}))
    for name, s in ss.items():
        scale = s.reshape(len(s), -1)
        ratios = flat/scale
        cal = cal_scores[name]
        row = evaluate_bands(ratios, scale, cal["q"], top)
        px = band_summary(evaluate_bands(ratios, scale, cal["q_pixel"], top))
        matched_q = value_at_coverage(row["score"], .90)
        summary = {**band_summary(row), "cal_q": cal["q"], "matched90_field_width": float(2*matched_q*scale.mean()),
                   "matched90_shock_width": float(2*matched_q*np.take_along_axis(scale, top, axis=1).mean()),
                   "matched90_shock_coverage": float(np.mean(np.take_along_axis(ratios, top, axis=1) <= matched_q)),
                   "pixel_calibrated": px}
        results[name], rows[name] = summary, row
        if split == "ood" and name in (UNIFORM, FIXED, SCALAR, OP):
            threshold, p, ess = weighted_quantile(cal["field_score"], wc, wt, alpha)
            weighted = evaluate_bands(ratios, scale, threshold, top)
            key = name+" / WCP"
            results[key] = {**band_summary(weighted), "ess": ess, "p_test_max": float(p.max()),
                            "fit_ms": fit_ms}
            unbounded = p > alpha
            rejected = unbounded | (ess < config["min_ess"]) | weighted["infinite"]
            results[key+" / gate"] = band_summary(weighted, rejected)
            rows[key] = {**weighted, "p_test": p, "reject": rejected}
    residual_truth = trajectory_residual(truth.astype(float), nu, config)
    diagnostics = {
        "prediction_residual": spectrum(r), "reference_snapshot_residual": spectrum(residual_truth),
        "truth_error": spectrum(prediction[:, 1:]-truth[:, 1:]),
        "mean_relative_l2": float(np.mean(rms(error)/rms(truth[:, 1:].astype(float)))),
        "mean_linf": float(flat.max(axis=1).mean()),
        "nu_mean": float(nu.mean()), "nu_below_cal_support_fraction": float(np.mean(nu < config["viscosity_id"][0])),
        "n": len(nu),
    }
    comparisons = {other: comparison(rows[OP], rows[other], config, config)
                   for other in (UNIFORM, FIXED, SCALAR)}
    return {"methods": results, "diagnostics": diagnostics, "comparisons": comparisons}, rows


def paired(all_raw, summaries, config):
    paired = {}
    coarse = summaries["paired_64"]["methods"]
    for n in (128, 256):
        key = f"paired_{n}"
        paired[key] = {}
        for method in METHODS:
            current = summaries[key]["methods"][method]
            row = all_raw[key][method]
            coarse_row = all_raw["paired_64"][method]
            delta = row["coverage"].astype(float)-coarse_row["coverage"]
            paired[key][method] = {
                "coverage_delta": float(delta.mean()),
                "coverage_delta_ci95": boot_interval(delta, config["bootstrap_resamples"], config["bootstrap_seed"]),
                "width_ratio": current["width"]/coarse[method]["width"],
                "score_ratio_median": float(np.median(row["score"]/coarse_row["score"])),
                "scale_ratio_median": float(np.median(row["scale_mean"]/coarse_row["scale_mean"]))
            }
    return paired


def audit(config, model, device, epoch):
    lock_path = ROOT/"lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (lock["config"] != config or lock["checkpoint_epoch"] != epoch
            or lock["checkpoint_sha256"] != checkpoint_hash()):
        raise RuntimeError("Checkpoint or config lock mismatch.")
    if any(digest(Path(path)) != sig for path, sig in lock["sources"].items()):
        raise RuntimeError("Evaluation code differs from the Val lock.")
    fixed_path = ROOT/"fixed_cp_b.npy"
    if digest(fixed_path) != lock["fixed_sha256"]:
        raise RuntimeError("CP-B Val scale differs from the lock.")
    fixed = np.load(fixed_path)
    cal, id_data, pairs, ood = [arrays(ROOT/f"{split}.npz") for split in ("cal", "id", "paired", "ood")]
    cal_scores, cal_features = calibration(cal, config, lock, fixed, model, device)
    summaries, raw, latencies = {}, {}, {}
    for name, data, n in (("id", id_data, 64), ("paired", pairs, 64), ("paired", pairs, 128),
                          ("paired", pairs, 256), ("ood", ood, 64)):
        label = f"{name}_{n}"
        summaries[label], raw[label] = evaluate(data, name, n, config, lock, fixed,
                                                cal_scores, cal_features, model, device)
        if name == "paired":
            latencies[str(n)] = timing(data, n, config, lock, model, device)
        print(f"Burgers evaluated {label}", flush=True)
    train = json.loads((ROOT/"train_log.json").read_text(encoding="utf-8"))
    result = {"config": config, "lock_sha256": digest(lock_path), "val_lock": lock,
        "data_sha256": {name: digest(ROOT/f"{name}.npz") for name in config["streams"]},
        "training": {**{k: train[k] for k in ("parameters", "selected_epoch", "gpu", "torch")},
                     "best_val_relative_l2": min(x["val_relative_l2"] for x in train["history"]),
                     "all_training_seconds": sum(x["seconds"] for x in train["history"])},
        "summary": summaries, "paired": paired(raw, summaries, config), "latency": latencies,
        "reference_audit": generator_audit(config)}
    write_json(ROOT/"metrics.json", result)
    np.savez_compressed(ROOT/"fields.npz", **{f"{split}__{name}__{key}": value
        for split, methods in raw.items() for name, columns in methods.items() for key, value in columns.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("lock", "eval"))
    args = parser.parse_args()
    spec = json.loads(CONFIG.read_text(encoding="utf-8"))
    target = torch_device()
    model, epoch = model_from_checkpoint(spec, target)
    if args.stage == "lock":
        audit_lock(spec, model, target, epoch)
    else:
        audit(spec, model, target, epoch)
