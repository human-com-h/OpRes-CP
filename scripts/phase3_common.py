"""Frozen replay, positive spatial scales, and field-level Phase 3 statistics."""

import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from darcy_day15 import estimated_weights, value_at_coverage, weighted_quantile
from darcy_phase0_eval import quantile
from phase2d import batches, take
from phase2d_statistics import flat, relative_l2


PLAN = json.loads(Path("configs/phase3.json").read_bytes())
ALPHAS = PLAN["alpha"]
OP = "OpRes locked"
STEPS = ("absolute residual", "inverse after absolute", "signed inverse", OP)


def rms(values):
    return np.sqrt(np.mean(flat(values)**2, axis=1)).reshape((len(values),) + (1,) * (values.ndim - 1))


def smooth_square(values, sigma, axes, mode):
    return gaussian_filter(values**2, (0,) * (values.ndim - axes) + (sigma,) * axes, mode=mode)


def envelope(values, sigma, axes, mode, floor):
    square = smooth_square(values, sigma, axes, mode) if sigma else values**2
    return np.sqrt(square + (floor * rms(values))**2)


def mix(scale, lam):
    return np.sqrt((1 - lam) * rms(scale)**2 + lam * scale**2)


def field_row(pred, truth, scale):
    scale = np.maximum(scale, 1e-12)
    score = (flat(np.abs(pred - truth)) / flat(scale)).max(axis=1)
    return np.column_stack((score, flat(scale).mean(axis=1), relative_l2(pred, truth)))


def stack_rows(chunks):
    return {name: np.concatenate(parts) for name, parts in chunks.items()}


def collect(protocol, model, data, grid, frames, calculate, *, batch=8, limit=None):
    inputs, truth = protocol.split(data, grid, frames)
    total = len(truth) if limit is None else min(len(truth), limit)
    chunks = {}
    for sl in batches(total, batch):
        x, y = take(inputs, sl), truth[sl]
        pred = protocol.predict(model, x, grid, frames).astype(float)
        for name, values in calculate(x, y, pred, sl).items():
            chunks.setdefault(name, []).append(values)
    return stack_rows(chunks)


def calibrate(rows, n=None):
    return {method: {str(a): quantile(values[:n, 0], a) for a in ALPHAS}
            for method, values in rows.items() if not method.startswith("_")}


def summarize(row, threshold, alpha):
    score, scale, rel_error = row.T
    q = np.broadcast_to(threshold, score.shape)
    finite = np.isfinite(q)
    coverage = score <= q
    widths = 2 * q * scale
    return {"n": len(score), "covered": int(coverage.sum()), "coverage": float(coverage.mean()),
            "width": float(widths.mean()) if finite.all() else "inf", "infinite_rate": float((~finite).mean()),
            "finite_n": int(finite.sum()), "finite_coverage": float(coverage[finite].mean()) if finite.any() else None,
            "finite_width": float(widths[finite].mean()) if finite.any() else None,
            "point_rel_l2": float(rel_error.mean()),
            "matched_width": float(2 * value_at_coverage(score, 1 - alpha) * scale.mean())}


def compare(rows, thresholds):
    return {method: {a: summarize(rows[method], q, float(a)) for a, q in levels.items()}
            for method, levels in thresholds.items() if method in rows}


def cal_sizes(cal, groups):
    return {str(n): {group: compare({OP: rows[OP]}, calibrate({OP: cal[OP]}, n))[OP]
                     for group, rows in groups.items()}
            for n in PLAN["calibration_counts"] if n <= len(cal[OP])}


def wcp_grid(cal, target, cal_features, target_features):
    result = {}
    for name, xc in cal_features.items():
        xt = target_features[name]
        result[name] = {}
        for c in PLAN["wcp_logistic_c"]:
            for clip in PLAN["wcp_clip"]:
                wc, wt = estimated_weights(xc, xt, {"base_weight_clip": clip, "weight_logistic_c": c})
                levels = {}
                for a in ALPHAS:
                    q, p, ess = weighted_quantile(cal[:, 0], wc, wt, a)
                    accepted = np.isfinite(q) & (ess >= 30)
                    row = summarize(target, q, a)
                    row.update({"ess": ess, "cal_weight_sum": float(wc.sum()),
                                "p_test_p50": float(np.median(p)), "p_test_p95": float(np.quantile(p, .95)),
                                "p_test_max": float(p.max()), "p_test_gt_alpha": float((p > a).mean()),
                                "rejected_without_gate": 0, "finite_ess_gate_rejected": int((~accepted).sum()),
                                "finite_ess_gate_reject_rate": float((~accepted).mean()),
                                "accepted_coverage": float((target[accepted, 0] <= q[accepted]).mean()) if accepted.any() else None})
                    levels[str(a)] = row
                result[name][f"C={c:g}, cap={clip[1]:g}"] = levels
    return result


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [finite_json(v) for v in value]
    if isinstance(value, (np.floating, float)) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf" if value < 0 else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def save(root, name, result, groups=None):
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.json"
    stage = path.with_suffix(".tmp")
    stage.write_text(json.dumps(finite_json(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    stage.replace(path)
    if groups:
        for group, data in groups.items():
            np.savez_compressed(root / f"fields_{group}.npz", **data)


def save_example(root, group, **maps):
    path = root / f"example_{group}.npz"
    np.savez_compressed(path, **{name: np.asarray(value, dtype=np.float32) for name, value in maps.items()})
