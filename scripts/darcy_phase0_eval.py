import argparse
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.fft import dstn, idstn
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata

from darcy_data import apply_operator

METHODS = ("Uniform-CP", "CP-B", "Full-Field PI-CP", "Inverse-after-Abs", "Signed Low-Pass", "OpRes-CP")
FREQUENCIES = np.array([1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192], dtype=float)


@lru_cache
def cell_spectrum(n):
    frequency = np.arange(1, n + 1)
    modes = 4 * n**2 * np.sin(np.pi * frequency / (2 * n))**2
    return modes[:, None] + modes[None, :], np.hypot(frequency[:, None], frequency[None, :])


def signed_inverse(a, residual):
    eig, _ = cell_spectrum(a.shape[-1])
    a_eff = np.exp(np.log(a).mean(axis=(-2, -1), keepdims=True))
    coeff = dstn(residual, type=2, norm="ortho", axes=(-2, -1))
    return idstn(coeff / (a_eff * eig), type=2, norm="ortho", axes=(-2, -1))


def low_pass(residual, radius):
    eig, _ = cell_spectrum(residual.shape[-1])
    coeff = dstn(residual, type=2, norm="ortho", axes=(-2, -1))
    return idstn(coeff * np.exp(-0.5 * radius**2 * eig), type=2, norm="ortho", axes=(-2, -1))


def envelope(z, config):
    radius = config["physical_radius"] * z.shape[-1]
    square = np.square(z)
    smoothed = gaussian_filter(square, (0, radius, radius), mode="reflect")
    floor = (config["relative_floor"]**2) * square.mean(axis=(-2, -1), keepdims=True)
    return np.sqrt(smoothed + floor + config["absolute_floor"]**2)


def center_coordinates(n):
    return (np.arange(n) + 0.5) / n


def bilinear(z, n):
    original = z.shape[-1]
    source = np.r_[0.0, center_coordinates(original), 1.0]
    target = center_coordinates(n)
    lower = source.searchsorted(target) - 1
    weight = (target - source[lower]) / (source[lower + 1] - source[lower])
    padded = np.pad(z, ((0, 0), (1, 1), (1, 1)), mode="edge")
    along = padded[:, lower] * (1 - weight[None, :, None]) + padded[:, lower + 1] * weight[None, :, None]
    return along[:, :, lower] * (1 - weight) + along[:, :, lower + 1] * weight


def map_prediction(z, n):
    # Zero Dirichlet values anchor the interpolant on all four physical faces.
    nodes = np.r_[0.0, center_coordinates(z.shape[-1]), 1.0]
    target = center_coordinates(n)
    lower = nodes.searchsorted(target) - 1
    weight = (target - nodes[lower]) / (nodes[lower + 1] - nodes[lower])
    padded = np.pad(z, ((0, 0), (1, 1), (1, 1)))
    along = padded[:, lower] * (1 - weight[None, :, None]) + padded[:, lower + 1] * weight[None, :, None]
    return along[:, :, lower] * (1 - weight) + along[:, :, lower + 1] * weight


def inputs(root, split):
    with np.load(root / f"data/{split}.npz") as f:
        data = dict(f)
    with np.load(root / f"{split}_pred.npz") as f:
        data.update(dict(f))
    return data


def validation_scales(val, floor):
    fixed = np.abs(val["u"] - val["pred"]).mean(axis=0, dtype=np.float64)
    fixed = np.maximum(fixed, max(0.05 * fixed.mean(), floor))
    residual = apply_operator(val["a"].astype(np.float64), val["pred"].astype(np.float64)) - 1
    return fixed, float(1 / max(np.median(np.abs(residual)), floor))


def scale_maps(a, residual, shared, lam, config):
    return (
        np.ones_like(residual),
        np.broadcast_to(shared, residual.shape),
        1 + lam * np.abs(residual),
        envelope(signed_inverse(a, np.abs(residual)), config),
        envelope(low_pass(residual, config["physical_radius"]), config),
        envelope(signed_inverse(a, residual), config),
    )


def correlation(x, y):
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    den = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    return np.divide(np.einsum("ij,ij->i", x, y), den,
                     out=np.full(len(x), np.nan), where=den > 0)


def spectrum(values):
    n = values.shape[-1]
    _, radius = cell_spectrum(n)
    power = dstn(values, type=2, norm="ortho", axes=(-2, -1))**2 / n**2
    bins = np.searchsorted(FREQUENCIES, radius, side="right") - 1
    return np.column_stack([power[:, bins == i].sum(axis=1) for i in range(len(FREQUENCIES)-1)])


def evaluate(a, truth, pred, fixed, lam, config, quantiles=None):
    method_rows = {name: {} for name in METHODS}
    common = {}
    shared = fixed if fixed.shape == a.shape[-2:] else bilinear(fixed[None], a.shape[-1])[0]
    for start in range(0, len(a), 16):
        sl = slice(start, start + 16)
        aa, y, u = (arr[sl].astype(np.float64) for arr in (a, truth, pred))
        residual = apply_operator(aa, u) - 1
        signed_error = u - y
        err = np.abs(signed_error).reshape(len(aa), -1)
        if quantiles is not None:
            raw = np.abs(residual).reshape(len(aa), -1)
            nonlocal_mask = ((raw <= np.quantile(raw, 0.5, axis=1)[:, None])
                             & (err >= np.quantile(err, 0.9, axis=1)[:, None]))
            reference_residual = np.abs(apply_operator(aa, y) - 1)
            field_records = {
                "error_linf": err.max(axis=1), "error_mean": err.mean(axis=1),
                "relative_l2": np.linalg.norm(err, axis=1) / np.linalg.norm(y.reshape(len(aa), -1), axis=1),
                "residual_rms": np.sqrt(np.square(raw).mean(axis=1)),
                "reference_cached_residual_rms": np.sqrt(np.square(reference_residual).mean(axis=(-2, -1))),
                "error_spectrum": spectrum(signed_error), "residual_spectrum": spectrum(residual),
                "nonlocal_points": nonlocal_mask.sum(axis=1),
            }
            for k, v in field_records.items():
                common.setdefault(k, []).append(v)
            error_ranks = rankdata(err, axis=1)
        for name, scale in zip(METHODS, scale_maps(aa, residual, shared, lam, config)):
            s = scale.reshape(len(aa), -1)
            standardized = err / s
            argmax = standardized.argmax(axis=1)
            i = np.arange(len(aa))
            row = {"score": standardized[i, argmax]}
            if quantiles is not None:
                q = quantiles[name]
                covered = standardized <= q
                count = nonlocal_mask.sum(axis=1)
                row.update({
                    "field_coverage": covered.all(axis=1), "pixel_coverage": covered.mean(axis=1),
                    "width": 2 * q * s.mean(axis=1), "scale_mean": s.mean(axis=1),
                    "pearson": correlation(err, s), "spearman": correlation(error_ranks, rankdata(s, axis=1)),
                    "argmax_error": err[i, argmax], "argmax_scale": s[i, argmax],
                    "nonlocal_coverage": np.divide((covered & nonlocal_mask).sum(axis=1), count,
                                                   out=np.full(len(count), np.nan), where=count > 0),
                })
            for k, v in row.items():
                method_rows[name].setdefault(k, []).append(v)
    combine = lambda records: {k: np.concatenate(v) for k, v in records.items()}
    return {key: combine(rows) for key, rows in method_rows.items()}, combine(common)


def boot_interval(values, resamples, seed):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (resamples, len(values)))].mean(axis=1)
    return [float(x) for x in np.quantile(means, (0.025, 0.975))]


def summarize(rows):
    return {name: {key: float(np.mean(value[np.isfinite(value)])) if np.isfinite(value).any() else None
                   for key, value in cols.items()} for name, cols in rows.items()}


def quantile(values, alpha):
    i = math.ceil((len(values) + 1) * (1 - alpha))
    return float(np.partition(values, i-1)[i-1]) if i <= len(values) else np.inf


def numerical_audit():
    n = 32
    x = center_coordinates(n)
    expected = (np.sin(3*np.pi*x)[:, None] * np.sin(7*np.pi*x)[None, :])[None]
    a = np.full_like(expected, 2.1)
    actual = signed_inverse(a, apply_operator(a, expected))
    return {"constant_coefficient_signed_recovery_linf": float(np.max(abs(actual - expected)))}


def main(config, seed, root):
    fixed, lam = validation_scales(inputs(root, "val"), config["absolute_floor"])
    cal = inputs(root, "cal")
    rows, _ = evaluate(cal["a"], cal["u"], cal["pred"], fixed, lam, config)
    qs = {m: quantile(rows[m]["score"], config["alpha"]) for m in METHODS}
    paired = inputs(root, "paired_test")
    test = inputs(root, "id_test")
    interpolated = map_prediction(paired["pred"], config["fine"])
    datasets = {
        "id": (test["a"], test["u"], test["pred"]),
        "paired_coarse": (paired["a"], paired["u"], paired["pred"]),
        "paired_fine_direct": (paired["a_f"], paired["u_f"], paired["pred_f"]),
        "paired_fine_interpolated": (paired["a_f"], paired["u_f"], interpolated),
    }
    all_rows, summary = {}, {}
    raw = {"paired_field_id": paired["id"], "id_field_id": test["id"], "spectrum_edges": FREQUENCIES}
    reference_restricted = paired["u_f"].reshape(len(paired["id"]), config["coarse"], 2,
                                                 config["coarse"], 2).mean(axis=(2, 4))
    reference_error = np.linalg.norm((reference_restricted - paired["u"]).reshape(len(paired["id"]), -1), axis=1)
    reference_error /= np.linalg.norm(paired["u"].reshape(len(paired["id"]), -1), axis=1)
    raw["reference_restriction_relative_l2"] = reference_error
    checks = numerical_audit()
    checks["reference_restriction_mean_relative_l2"] = float(reference_error.mean())
    for split, (a, u, p) in datasets.items():
        rows, fields = evaluate(a, u, p, fixed, lam, config, qs)
        all_rows[split] = rows
        summary[split] = {"methods": summarize(rows),
                          "field": {k: v.mean(axis=0).tolist() for k, v in fields.items()}}
        raw.update({f"{split}__{key}": v for key, v in fields.items()})
        for name, cols in rows.items():
            raw.update({f"{split}__{name}__{key}": v for key, v in cols.items()})
        print(f"evaluated {split}: {len(a)} fields", flush=True)
    pair_results = {}
    for split in ("paired_fine_direct", "paired_fine_interpolated"):
        pair_results[split] = {}
        for name in METHODS:
            coarse, fine = all_rows["paired_coarse"][name], all_rows[split][name]
            delta = fine["field_coverage"].astype(float) - coarse["field_coverage"]
            score_delta = fine["score"] - coarse["score"]
            field_error_ratio = fine["argmax_error"] / np.maximum(coarse["argmax_error"], config["absolute_floor"])
            field_scale_ratio = fine["argmax_scale"] / np.maximum(coarse["argmax_scale"], config["absolute_floor"])
            pair_results[split][name] = {
                "coverage_delta": float(delta.mean()),
                "coverage_delta_ci95": boot_interval(delta, config["bootstrap_resamples"], seed),
                "coverage_losses": int((delta < 0).sum()), "coverage_gains": int((delta > 0).sum()),
                "width_ratio": float(fine["width"].mean()/coarse["width"].mean()),
                "median_score_delta": float(np.median(score_delta)),
                "median_score_delta_over_q": float(np.median(score_delta) / qs[name]),
                "median_score_ratio": float(np.median(fine["score"] / coarse["score"])),
                "median_at_argmax_error_ratio": float(np.median(field_error_ratio)),
                "median_at_argmax_scale_ratio": float(np.median(field_scale_ratio)),
            }
            for key, values in {"coverage_delta": delta, "score_delta": score_delta,
                                "error_ratio_at_argmax": field_error_ratio,
                                "scale_ratio_at_argmax": field_scale_ratio}.items():
                raw[f"{split}__{name}__{key}"] = values
    comparisons = {}
    op = all_rows["id"]["OpRes-CP"]
    for name in METHODS[:-1]:
        other = all_rows["id"][name]
        cov_delta = op["field_coverage"].astype(float) - other["field_coverage"]
        comparison = {"coverage_delta_ci95": boot_interval(cov_delta, config["bootstrap_resamples"], seed),
                      "width_ratio": float(op["width"].mean() / other["width"].mean()),
                      "width_diff_ci95": boot_interval(op["width"]-other["width"], config["bootstrap_resamples"], seed)}
        if name != "Uniform-CP":
            spearman_gain = op["spearman"] - other["spearman"]
            comparison["spearman_delta"] = float(np.nanmean(spearman_gain))
            comparison["spearman_delta_ci95"] = boot_interval(spearman_gain, config["bootstrap_resamples"], seed)
        comparisons[name] = comparison
    output = {"seed": seed, "config": config, "pi_lambda_val_only": lam, "thresholds": qs,
              "numerics": checks, "summary": summary, "paired": pair_results,
              "id_opres_comparisons": comparisons,
              "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in Path("scripts").glob("darcy*.py")}}
    np.savez_compressed(root / "day1_fields.npz", **raw)
    (root / "day1_metrics.json").write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase0_darcy.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    main(config, args.seed, Path("results") / f"seed_{args.seed}")
