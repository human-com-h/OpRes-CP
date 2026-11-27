import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import burgers_phase2 as phase2
from burgers_phase1_eval import OP, SCALAR, gradient, infer, lift, mixed, envelope, trajectory_residual
from darcy_day15 import arrays, band_summary, digest, estimated_weights, evaluate_bands, weighted_quantile, write_json
from darcy_phase0_eval import quantile


def spectral_ratio(initial):
    power = np.abs(np.fft.rfft(initial.astype(float), axis=-1)) ** 2
    power[:, 1:-1] *= 2
    return power[:, 9:].sum(axis=1) / np.maximum(power.sum(axis=1), 1e-30)


def features(initial, nu):
    base = np.column_stack((np.log(nu), np.abs(gradient(initial)).max(axis=-1),
                            np.sqrt(np.mean(initial.astype(float) ** 2, axis=-1))))
    return base, np.column_stack((base, spectral_ratio(initial)))


def predicted_maps(model, initial, nu, config, device):
    pred = infer(model, device, initial, nu).astype(float)
    residual = trajectory_residual(pred, nu, config)
    lifted = lift("adv_diff", pred, residual, nu, config)
    scale = envelope(lifted, config)
    return pred[:, 1:], {OP: mixed(scale, .85), SCALAR: mixed(scale, 0.)}


def location_report(error, scale, score, q, scalar_hit, initial, config):
    n, frames, grid = error.shape
    location = score.reshape(n, -1).argmax(axis=1)
    row = np.arange(n)
    frame, point = np.divmod(location, grid)
    peak = score[row, frame, point]
    miss = peak > q
    at_peak = scale[row, frame, point]
    point_rank = np.mean(scale.reshape(n, -1) <= at_peak[:, None], axis=1)
    spatial_rank = np.mean(scale[row, frame] <= at_peak[:, None], axis=1)
    centered = initial - initial.mean(axis=1, keepdims=True)
    missed = int(miss.sum())
    return {
        "n": n, "missed": missed, "missed_but_scalar_cover": int((miss & scalar_hit).sum()),
        "miss_peak_frame_counts": np.bincount(frame[miss], minlength=frames).tolist(),
        "frame_times": (np.arange(1, frames + 1) * config["frame_dt"]).tolist(),
        "miss_peak_whole_scale_rank_median": float(np.median(point_rank[miss])) if missed else None,
        "miss_peak_spatial_scale_rank_median": float(np.median(spatial_rank[miss])) if missed else None,
        "miss_peak_scale_to_mean_median": float(np.median((at_peak / scale.mean(axis=(1, 2)))[miss])) if missed else None,
        "miss_peak_abs_error_median": float(np.median(error[row, frame, point][miss])) if missed else None,
        "median_spatial_scale_p95_p05": float(np.median(
            np.quantile(scale, .95, axis=(1, 2)) / np.quantile(scale, .05, axis=(1, 2)))),
        "input_e_gt_8_over_total_median": float(np.median(spectral_ratio(initial))),
        "input_e_gt_8_over_ac_energy_median": float(np.median(spectral_ratio(centered))),
    }


def run(config, data_root, checkpoint, expected, device):
    model = phase2.make_model("FNO", config).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])
    model.eval()
    names = ("cal", "id", "ood_high")
    for name in names:
        phase2.generate(config, name, data_root)
    data = {name: arrays(data_root / f"{name}.npz") for name in names}
    for name, item in data.items():
        assert np.array_equal(item["id"], config["streams"][name] * 10000 + np.arange(len(item["id"])))
    assert len(np.unique(np.concatenate([item["id"] for item in data.values()]))) == sum(map(
        lambda item: len(item["id"]), data.values()))
    raw_features = {name: features(item["u64"][:, 0], item["nu"]) for name, item in data.items()}
    options = {"base_weight_clip": config["weight_clip"], "weight_logistic_c": 1.}
    weights, mixed_weights = {}, {}
    for idx, label in enumerate(("input", "input_plus_spectrum")):
        xc, xi, xt = (raw_features[name][idx] for name in names)
        weights[label] = estimated_weights(xc, xt, options)
        mixed_weights[label] = estimated_weights(xc, np.concatenate((xi, xt)), options)
    scores, errors, scales = {}, {}, {}
    for name, item in data.items():
        pred, sigmas = predicted_maps(model, item["u64"][:, 0], item["nu"], config, device)
        errors[name] = np.abs(pred - item["u64"][:, 1:])
        scales[name] = sigmas
        scores[name] = {key: (errors[name] / sigma).reshape(len(pred), -1).max(axis=1)
                        for key, sigma in sigmas.items()}
    alpha = .1
    q = {method: quantile(v, alpha) for method, v in scores["cal"].items()}
    target = data["ood_high"]["u64"][:, 1:]
    grad = np.abs(gradient(target.astype(float))).reshape(len(target), -1)
    n_top = int(np.ceil(.1 * grad.shape[1]))
    top = np.argpartition(grad, -n_top, axis=1)[:, -n_top:]
    scale = scales["ood_high"][OP]
    matrix = scale.reshape(len(scale), -1)
    ratios = errors["ood_high"] / scale
    bands = evaluate_bands(ratios.reshape(matrix.shape), matrix, q[OP], top)
    wcp, gate = {}, {}
    for label, (wc, wt) in weights.items():
        q_weighted, p, ess = weighted_quantile(scores["cal"][OP], wc, wt, alpha)
        weighted_rows = evaluate_bands(ratios.reshape(matrix.shape), matrix, q_weighted, top)
        wcp[label] = band_summary(weighted_rows) | {"ess": ess, "max_p_test": float(p.max())}
        wc_mix, wt_mix = mixed_weights[label]
        p_mix = wt_mix / (wc_mix.sum() + wt_mix)
        n_id = len(data["id"]["id"])
        ood = np.arange(len(p_mix)) >= n_id
        covered = np.r_[scores["id"][OP] <= q[OP], bands["coverage"]]
        width = np.r_[2 * q[OP] * scales["id"][OP].mean(axis=(1, 2)), bands["width"]]
        anchors = {}
        for threshold in config["gate_p_test_thresholds"]:
            accept = p_mix <= threshold
            anchors[str(threshold)] = {
                "n_accepted": int(accept.sum()), "id_accept": float(accept[~ood].mean()),
                "ood_reject": float((~accept[ood]).mean()),
                "accepted_coverage": float(covered[accept].mean()) if accept.any() else None,
                "accepted_width": float(width[accept].mean()) if accept.any() else None,
            }
        gate[label] = {"auroc": float(roc_auc_score(ood, p_mix)), "anchors": anchors}
    old = expected["results"]["FNO"]["0.1"]["summary"]["ood_high_64"]["methods"]
    coverage = {method: float((scores["ood_high"][method] <= q[method]).mean()) for method in q}
    return {
        "seed": config["seed"], "checkpoint_sha256": digest(checkpoint),
        "data_sha256_matches": {name: digest(data_root / f"{name}.npz") == expected["data_sha256"][name]
                                for name in names},
        "coverage": coverage, "cached_coverage": {method: old[method]["coverage"] for method in q},
        "threshold": q, "cached_threshold": {method: old[method]["cal_q"] for method in q},
        "location": location_report(errors["ood_high"], scale, ratios, q[OP],
                                    scores["ood_high"][SCALAR] <= q[SCALAR],
                                    data["ood_high"]["u64"][:, 0], config),
        "id_input_e_gt_8_over_total_median": float(np.median(raw_features["id"][1][:, -1])),
        "wcp": wcp, "gate": gate,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path("results")
    source = (root / "phase2_kaggle_private" / "downloads" / "2b" / f"seed_{args.seed}" /
              "results" / "phase2_burgers" / f"seed_{args.seed}")
    expected = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    cache = root / "phase2d_cache" / f"seed_{args.seed}" / "burgers"
    cache.mkdir(parents=True, exist_ok=True)
    result = run(expected["settings"], cache, source / "FNO" / "fno.pt", expected, phase2.torch_device())
    write_json(root / f"phase2d_burgers_diagnosis_seed{args.seed}.json", result)
    print(json.dumps({key: result[key] for key in ("coverage", "cached_coverage", "location")}, indent=2))


if __name__ == "__main__":
    main()
