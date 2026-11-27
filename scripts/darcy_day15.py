import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import scipy
import sklearn
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.fft import dstn
from scipy.stats import truncnorm

from darcy_data import apply_operator
from darcy_day15_ops import (AXES, INVERSES, inverse_diagnostics, lift, mixture,
                            physical_scale, rms, rho, timed, timing_summary)
from darcy_fno import FNO, predict
from darcy_phase0_eval import (bilinear, boot_interval, cell_spectrum, envelope, inputs,
                              low_pass, map_prediction, quantile, validation_scales)


BASELINES = ("Uniform-CP", "CP-B", "Full-Field PI-CP", "Inverse-after-Abs",
             "Signed Low-Pass", "Original OpRes-CP")
UPGRADED = "Day1.5 OpRes-CP"
SCALAR = "Physics-Scalar CP"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def frozen_model(config, root):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint = torch.load(root / "fno.pt", map_location=device, weights_only=True)
    model = FNO(config["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval(), device, checkpoint["epoch"]


def forward(model, device, a, batch):
    return predict(model, torch.from_numpy(a).to(device), batch)


def arrays(path):
    with np.load(path) as archive:
        return dict(archive)


def per_field_scores(error, scale):
    return (error / scale).max(axis=AXES)


def value_at_coverage(values, coverage):
    flat = np.asarray(values).ravel()
    index = math.ceil(coverage * len(flat)) - 1
    return float(np.partition(flat, index)[index])


def lock_model(spec, config, root, out, model, device, epoch):
    if (out / "lock.json").exists():
        raise RuntimeError("Val lock already exists. Keep the frozen choice for this run.")
    val_path = out / "data/val.npz"
    val = arrays(val_path)
    a = val["a"].astype(float)
    prediction = forward(model, device, val["a"], config["model"]["batch_size"])
    pred = prediction.astype(float)
    r = apply_operator(a, pred) - 1
    err = pred - val["u"]
    diagnostics, lifts = inverse_diagnostics(a, r, err, config)
    rows, checks = [], {}
    for name in INVERSES:
        z = lifts[name]
        s = physical_scale(z, config)
        checks[name] = float(np.max(abs(physical_scale(lift(name, a, -3*r), config) - 3*s))
                             / np.max(3*s))
        for lam in spec["lambdas"]:
            scale = mixture(s, lam)
            score = per_field_scores(abs(err), scale)
            q = value_at_coverage(score, 1-config["alpha"])
            rows.append({"inverse": name, "lambda": lam, "coverage": float(np.mean(score <= q)),
                         "width": float(2*q*scale.mean())})
    # Choose the scalar control under the same inverse to isolate the spatial term.
    choice = min(rows, key=lambda row: row["width"])
    val_old = inputs(root, "val")
    fixed, pi_lambda = validation_scales(val_old, config["absolute_floor"])
    context = (config, fixed, pi_lambda, choice)
    for name in BASELINES:
        s = np.stack([scale_for(name, aa, rr, context) for aa, rr in zip(a, r)])
        score = per_field_scores(abs(err), s)
        q = value_at_coverage(score, 1-config["alpha"])
        rows.append({"inverse": name, "lambda": None, "coverage": float(np.mean(score <= q)),
                     "width": float(2*q*s.mean())})
    sources = {str(p.as_posix()): digest(p) for p in sorted(
        [*Path("scripts").glob("darcy*.py"), Path("configs/phase0_day15.json"),
         Path(spec["base_config"])])}
    write_json(out / "lock.json", {"spec": spec, "base_config": config, "epoch": epoch,
               "checkpoint_sha256": digest(root / "fno.pt"), "val_sha256": digest(val_path),
               "source_sha256": sources, "candidate": choice, "val_grid": rows,
               "inverse_diagnostics": diagnostics, "homogeneity_relative_error": checks})
    print("Val locked:", choice, flush=True)


def scale_for(name, a, r, context, fixed_grid=None):
    config, fixed, pi_lambda, choice = context
    if name == "Uniform-CP":
        return np.ones_like(r)
    if name == "CP-B":
        return fixed_grid if fixed_grid is not None else fixed
    if name == "Full-Field PI-CP":
        return 1 + pi_lambda*abs(r)
    if name == "Inverse-after-Abs":
        return envelope(lift("const", a, abs(r))[None], config)[0]
    if name == "Signed Low-Pass":
        return envelope(low_pass(r, config["physical_radius"])[None], config)[0]
    if name == "Original OpRes-CP":
        return envelope(lift("const", a, r)[None], config)[0]
    scale = physical_scale(lift(choice["inverse"], a, r), config)
    return mixture(scale, 0 if name == SCALAR else choice["lambda"])


def features(a, r, pred, choice):
    log = np.log(a)
    contrast = np.quantile(log, [0.1, 0.9], axis=AXES)
    z = lift(choice["inverse"], a, r)
    _, freq = cell_spectrum(a.shape[-1])
    power = dstn(r, type=2, norm="ortho", axes=AXES)**2
    centroid = (power*freq).sum(axis=AXES) / power.sum(axis=AXES)
    return np.column_stack((log.std(axis=AXES), contrast[1]-contrast[0],
                            (rms(z)/rms(pred)).ravel(), centroid))


def estimated_weights(x_cal, x_test, spec):
    # Neither this classifier nor its preprocessing has access to truth/score arrays.
    x = np.concatenate((x_cal, x_test))
    y = np.repeat([0, 1], [len(x_cal), len(x_test)])
    estimator = make_pipeline(StandardScaler(), LogisticRegression(C=spec["weight_logistic_c"],
                                                                    max_iter=1000))
    estimator.fit(x, y)
    odds = np.exp(estimator.decision_function(x)) * (len(x_cal)/len(x_test))
    weights = np.clip(odds, *spec["base_weight_clip"])
    return weights[:len(x_cal)], weights[len(x_cal):]


def oracle_weights(sigma, config):
    setup = config["sigma"]
    std = setup["std"]
    bounds = np.array(setup["bounds"])
    cal_mu, test_mu = setup["id_mean"], setup["ood_a_mean"]
    cal_bounds, test_bounds = (bounds-cal_mu)/std, (bounds-test_mu)/std
    return np.exp(truncnorm.logpdf(sigma, *test_bounds, loc=test_mu, scale=std)
                  - truncnorm.logpdf(sigma, *cal_bounds, loc=cal_mu, scale=std))


def weighted_quantile(scores, w_cal, w_test, alpha):
    order = np.argsort(scores)
    cumulative = np.cumsum(w_cal[order], dtype=float)
    target = (1-alpha)*(cumulative[-1] + w_test)
    index = np.searchsorted(cumulative, target)
    quantiles = np.r_[scores[order], np.inf][index]
    p_test = w_test / (w_cal.sum() + w_test)
    ess = float(w_cal.sum()**2 / np.sum(w_cal*w_cal))
    return quantiles, p_test, ess


def weighted_setup(a, r, pred, sigma, cal, split, config, spec, choice):
    if not split.startswith("ood_"):
        return {}
    x = features(a, r, pred, choice)
    weights, fit_ms = timed(lambda: estimated_weights(cal["features"], x, {
        **spec, "base_weight_clip": config["weight_clip"]}))
    result = {"summary-WCP": (*weights, fit_ms)}
    if split == "ood_a":
        result["oracle-1D"] = (oracle_weights(cal["sigma"], config),
                               oracle_weights(sigma, config), 0)
    return result


def calibration(config, context, root):
    cal = inputs(root, "cal")
    a, pred = cal["a"].astype(float), cal["pred"].astype(float)
    error = abs(pred - cal["u"])
    r = apply_operator(a, pred) - 1
    result = {"sigma": cal["sigma"], "features": features(a, r, pred, context[-1]), "methods": {}}
    for name in (*BASELINES, SCALAR, UPGRADED):
        scale = np.stack([scale_for(name, aa, rr, context) for aa, rr in zip(a, r)])
        ratios = error/scale
        score = ratios.max(axis=AXES)
        result["methods"][name] = {"scores": score, "field_q": quantile(score, config["alpha"]),
                                   "pixel_q": quantile(ratios.ravel(), config["alpha"])}
    return result


def evaluate_bands(ratios, scale, thresholds, top):
    q = np.broadcast_to(thresholds, (len(scale),))
    mask = ratios <= q[:, None]
    row = {
        "coverage": mask.all(axis=1), "pixel": mask.mean(axis=1),
        "width": (2*scale.mean(axis=1))*q,
        "top_coverage": np.take_along_axis(mask, top, axis=1).mean(axis=1),
        "top_width": (2*np.take_along_axis(scale, top, axis=1).mean(axis=1))*q,
        "infinite": ~np.isfinite(q), "score": ratios.max(axis=1), "scale_mean": scale.mean(axis=1)
    }
    return row


def mean_or_none(v):
    return float(np.mean(v)) if len(v) else None


def band_summary(row, abstain=None):
    finite = ~row["infinite"]
    if abstain is None:
        abstain = np.zeros(len(finite), dtype=bool)
    accept = ~abstain
    total = len(finite)
    return {
        "n": total, "finite_n": int(finite.sum()), "accepted_n": int(accept.sum()),
        "infinite_rate": float(np.mean(~finite)), "abstain": float(np.mean(abstain)),
        "coverage": mean_or_none(row["coverage"][accept]),
        "pixel": mean_or_none(row["pixel"][accept]),
        "top_coverage": mean_or_none(row["top_coverage"][accept]),
        "width": mean_or_none(row["width"][accept]) if finite[accept].all() else "inf",
        "top_width": mean_or_none(row["top_width"][accept]) if finite[accept].all() else "inf",
        "finite_width": mean_or_none(row["width"][accept & finite]),
        "finite_field_coverage": mean_or_none(row["coverage"][accept & finite]),
        "all_covered_and_accepted": float(np.mean(accept & row["coverage"])),
        "accepted_risk": mean_or_none(~row["coverage"][accept]),
    }


def risk_curve(row, ess, p_test, spec):
    result = []
    for ess_min in spec["risk_ess_thresholds"]:
        for p_max in spec["risk_p_test_thresholds"]:
            use = (ess >= ess_min) & (p_test <= p_max) & ~row["infinite"]
            result.append({"ess_min": ess_min, "p_max": p_max, "accept": float(use.mean()),
                           "accepted_risk": mean_or_none(~row["coverage"][use])})
    return result


def per_method(split, name, a, r, error, top, context, cal_row, weighted, spec):
    config, fixed, _, _ = context
    fixed_grid = fixed if a.shape[-1] == fixed.shape[-1] else bilinear(fixed[None], a.shape[-1])[0]
    scale_for(name, a[0], r[0], context, fixed_grid)
    measurements = [timed(lambda: scale_for(name, aa, rr, context, fixed_grid))
                    for aa, rr in zip(a, r)]
    scale = np.stack([s.ravel() for s, _ in measurements])
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError(f"Nonpositive or nonfinite scale for {split}/{name}")
    ratios = error / scale
    alpha = config["alpha"]
    row = evaluate_bands(ratios, scale, cal_row["field_q"], top)
    pixel = evaluate_bands(ratios, scale, cal_row["pixel_q"], top)
    matched_field_q = value_at_coverage(row["score"], 1-alpha)
    matched_pixel_q = value_at_coverage(ratios, 1-alpha)
    summary = band_summary(row)
    summary.update({"val_locked_64_field_q": cal_row["field_q"],
                    "cal_64_pixel_q": cal_row["pixel_q"],
                    "pixel_calibrated": band_summary(pixel),
                    "matched90_field_width": float(2*matched_field_q*scale.mean()),
                    "matched90_pixel_width": float(2*matched_pixel_q*scale.mean()),
                    "scale_ms": timing_summary([t for _, t in measurements])})
    methods, raw = {name: summary}, {name: row}
    for label, (w_cal, w_test, fit_ms) in weighted.items():
        (q, p_test, ess), q_ms = timed(lambda: weighted_quantile(
            cal_row["scores"], w_cal, w_test, alpha))
        w_row = evaluate_bands(ratios, scale, q, top)
        w_summary = band_summary(w_row)
        w_summary.update({"ess": ess, "p_test_p50": float(np.median(p_test)),
                          "p_test_p90": float(np.quantile(p_test, 0.9)),
                          "p_test_max": float(p_test.max()), "fit_ms": fit_ms,
                          "quantile_batch_ms": q_ms, "max_test_weight": float(w_test.max())})
        key = f"{name} / {label}"
        methods[key], raw[key] = w_summary, w_row
        if split == "ood_b":
            abstain = (p_test > alpha) | (ess < config["min_ess"]) | w_row["infinite"]
            methods[f"{key} / gate"] = {**band_summary(w_row, abstain), "risk_curve": risk_curve(
                w_row, ess, p_test, spec)}
            w_row.update({"abstain": abstain, "p_test": p_test})
    return methods, raw


def evaluate_split(split, data, pred, config, context, cal, spec):
    a, pred, truth = (data["a"].astype(float), pred.astype(float), data["u"])
    error = abs(pred - truth)
    r = apply_operator(a, pred) - 1
    flat = error.reshape(len(a), -1)
    k = math.ceil(0.1*flat.shape[1])
    top = np.argpartition(flat, -k, axis=1)[:, -k:].copy()
    weights = weighted_setup(a, r, pred, data["sigma"], cal, split, config, spec, context[-1])
    results = {}
    raw = {}
    for name in cal["methods"]:
        apply_weights = name in ("Uniform-CP", "CP-B", "Original OpRes-CP", SCALAR, UPGRADED)
        rows, fields = per_method(split, name, a, r, flat, top, context, cal["methods"][name],
                                 weights if apply_weights else {}, spec)
        results.update(rows)
        raw.update(fields)
    errors = {"linf_mean": float(flat.max(axis=1).mean()),
              "linf_median": float(np.median(flat.max(axis=1))),
              "relative_l2_mean": float(np.mean(rms(error)/rms(truth.astype(float)))),
              "sigma_sample_mean": float(data["sigma"].mean()),
              "log_a_sd_mean": float(np.log(a).std(axis=AXES).mean())}
    return {"methods": results, "base_error": errors}, raw


def comparison(a, b, spec, config):
    seed, resamples = spec["bootstrap_seed"], config["bootstrap_resamples"]
    ca, cb = a["coverage"].astype(float), b["coverage"].astype(float)
    row = {"coverage_diff": float((ca-cb).mean()),
           "coverage_diff_ci95": boot_interval(ca-cb, resamples, seed)}
    if not (a["infinite"].any() or b["infinite"].any()):
        row["width_diff_ci95"] = boot_interval(a["width"]-b["width"], resamples, seed)
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(ca), (resamples, len(ca)))
        k = math.ceil((1-config["alpha"])*len(ca))-1
        qa = np.partition(a["score"][draws], k, axis=1)[:, k]
        qb = np.partition(b["score"][draws], k, axis=1)[:, k]
        wa = qa * a["scale_mean"][draws].mean(axis=1)
        wb = qb * b["scale_mean"][draws].mean(axis=1)
        row["matched90_width_ratio_ci95"] = np.quantile(wa/wb, [0.025, 0.975]).tolist()
    return row


def comparisons(all_raw, summary, spec, config):
    groups = {}
    for split in ("id_test", "ood_a", "ood_b"):
        rows, results = all_raw[split], summary[split]["methods"]
        groups[split] = {}
        for deploy in (UPGRADED, f"{UPGRADED} / summary-WCP"):
            if deploy not in rows:
                continue
            group = {}
            for name in ("Uniform-CP", "CP-B", "Original OpRes-CP", SCALAR, UPGRADED):
                if deploy == name:
                    continue
                group[name] = comparison(rows[deploy], rows[name], spec, config)
                if deploy == UPGRADED:
                    group[name]["matched90_width_ratio"] = (
                        results[deploy]["matched90_field_width"]/results[name]["matched90_field_width"])
                weighted = f"{name} / summary-WCP"
                if deploy != weighted and weighted in rows:
                    group[weighted] = comparison(rows[deploy], rows[weighted], spec, config)
            groups[split][deploy] = group
    return groups


def paired_results(all_raw, spec, config):
    result = {}
    for split in ("paired_fine_direct", "paired_fine_interpolated"):
        result[split] = {}
        for name, coarse in all_raw["paired_coarse"].items():
            if name == "fields":
                continue
            fine = all_raw[split][name]
            delta = fine["coverage"].astype(float)-coarse["coverage"]
            result[split][name] = {
                "coverage_delta": float(delta.mean()),
                "coverage_delta_ci95": boot_interval(delta, config["bootstrap_resamples"],
                                                    spec["bootstrap_seed"]),
                "fine_coarse_width_ratio": float(fine["width"].mean()/coarse["width"].mean()),
                "median_fine_coarse_score_ratio": float(np.median(fine["score"]/coarse["score"])),
                "median_fine_coarse_scale_ratio": float(np.median(fine["scale_mean"]/coarse["scale_mean"])),
            }
    return result


def latency(model, device, a, choice, config):
    def inference(aa):
        return forward(model, device, aa[None], 1)[0]

    def post(aa, pp):
        aa, pp = aa.astype(float), pp.astype(float)
        rr = apply_operator(aa, pp) - 1
        return mixture(physical_scale(lift(choice["inverse"], aa, rr), config), choice["lambda"])

    for aa in a[:8]:
        post(aa, inference(aa))
    rows = {"fno_host_to_host": [], "upgrade_cpu": [], "end_to_end": []}
    for aa in a[:80]:
        def infer_and_scale():
            pp, infer_ms = timed(lambda: inference(aa))
            _, post_ms = timed(lambda: post(aa, pp))
            return infer_ms, post_ms
        (infer_ms, post_ms), total_ms = timed(infer_and_scale)
        for values, elapsed in zip(rows.values(), (infer_ms, post_ms, total_ms)):
            values.append(elapsed)
    return {key: timing_summary(value) for key, value in rows.items()}


def run_test(spec, config, root, out, model, device, epoch):
    lock = json.loads((out / "lock.json").read_text(encoding="utf-8"))
    if lock["spec"] != spec or lock["base_config"] != config or lock["epoch"] != epoch:
        raise RuntimeError("Run configuration differs from the Val lock.")
    if digest(root / "fno.pt") != lock["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint differs from the Val lock.")
    for path, expected in lock["source_sha256"].items():
        if digest(Path(path)) != expected:
            raise RuntimeError(f"{path} differs from the Val lock.")
    val_old = inputs(root, "val")
    fixed, pi_lambda = validation_scales(val_old, config["absolute_floor"])
    choice = lock["candidate"]
    context = (config, fixed, pi_lambda, choice)
    cal = calibration(config, context, root)
    results, raw, hashes, timings = {}, {}, {}, {}
    for split in ("id_test", "paired_test", "ood_a", "ood_b"):
        path = out / f"data/{split}.npz"
        hashes[split] = digest(path)
        # Generate weights inside evaluation from explicitly passed unlabeled features.
        data = arrays(path)
        if not np.all(data["id"] // 10000 == spec["split_streams"][split]):
            raise RuntimeError(f"Unexpected data namespace for {split}.")
        pred = forward(model, device, data["a"], config["model"]["batch_size"])
        name = "paired_coarse" if split == "paired_test" else split
        results[name], raw[name] = evaluate_split(name, data, pred, config, context, cal, spec)
        raw[name]["fields"] = {"id": data["id"], "error_linf": abs(pred-data["u"]).max(axis=AXES)}
        if split == "id_test":
            timings["64"] = latency(model, device, data["a"], choice, config)
        if split == "paired_test":
            fine = {"a": data["a_f"], "u": data["u_f"], "sigma": data["sigma"]}
            pred_f = forward(model, device, fine["a"], 4)
            for kind, pp in (("direct", pred_f), ("interpolated", map_prediction(pred, config["fine"]))):
                name = f"paired_fine_{kind}"
                results[name], raw[name] = evaluate_split(name, fine, pp, config, context, cal, spec)
                raw[name]["fields"] = {"id": data["id"], "error_linf": abs(pp-fine["u"]).max(axis=AXES)}
            timings["128"] = latency(model, device, fine["a"], choice, config)
        print(f"Evaluated {split} ({len(pred)} fields)", flush=True)
    output = {"lock_sha256": digest(out / "lock.json"), "candidate": choice, "test_sha256": hashes,
              "cal_sha256": {name: digest(root / name) for name in
                             ("data/cal.npz", "cal_pred.npz", "data/val.npz", "val_pred.npz")},
              "environment": {"gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
                              "torch": torch.__version__, "scipy": scipy.__version__,
                              "sklearn": sklearn.__version__, "cpu": platform.processor(),
                              "system": platform.platform()},
              "results": results, "latency": timings, "paired": paired_results(raw, spec, config),
              "comparisons": comparisons(raw, results, spec, config)}
    write_json(out / "metrics.json", output)
    np.savez_compressed(out / "field_metrics.npz", **{
        f"{split}__{method}__{key}": value for split, methods in raw.items()
        for method, columns in methods.items() for key, value in columns.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("lock", "test"))
    args = parser.parse_args()
    spec = json.loads(Path("configs/phase0_day15.json").read_text(encoding="utf-8"))
    config = json.loads(Path(spec["base_config"]).read_text(encoding="utf-8"))
    root = Path("results") / f"seed_{spec['model_seed']}"
    out = root / "day15"
    model, device, epoch = frozen_model(config, root)
    print(f"Frozen checkpoint epoch {epoch}, {device}, torch {torch.__version__}", flush=True)
    action = lock_model if args.stage == "lock" else run_test
    action(spec, config, root, out, model, device, epoch)
