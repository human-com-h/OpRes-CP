"""Frozen Burgers ablations and high-frequency diagnosis."""

import numpy as np
from scipy.signal import find_peaks
from sklearn.metrics import roc_auc_score, roc_curve

import burgers_phase1_ops as ops
from darcy_day15 import arrays, estimated_weights
from darcy_phase2 import gate_point
from darcy_phase0_eval import quantile
from phase2d_burgers_diagnosis import features, spectral_ratio
from phase3_common import (ALPHAS, OP, PLAN, STEPS, cal_sizes, calibrate, collect, compare,
                           envelope, field_row, mix, rms, save_example, wcp_grid)


def shocks(field):
    grad = -ops.gradient(field)
    n = len(grad)
    locations = find_peaks(np.tile(grad, 3), height=PLAN["burgers_shock_relative_strength"] * max(grad.max(), 0.))[0]
    return locations[(locations >= n) & (locations < 2 * n)] - n


def nearest(points, center, n):
    distance = (points - center + n // 2) % n - n // 2
    return distance[np.argmin(abs(distance))] if len(points) else np.nan


def peak_diagnostics(pred, truth, scale):
    error = np.abs(pred - truth)
    ratios = error / np.maximum(scale, 1e-12)
    n = pred.shape[-1]
    peak_time, peak_cell = np.divmod(ratios.reshape(len(pred), -1).argmax(axis=1), n)
    rows = []
    for j, (t, cell) in enumerate(zip(peak_time, peak_cell)):
        truth_frame, pred_frame = truth[j, t], pred[j, t]
        true_shocks = shocks(truth_frame)
        true_offset = nearest(true_shocks, cell, n)
        shift = nearest(shocks(pred_frame), (cell + true_offset) % n, n) if np.isfinite(true_offset) else np.nan
        if abs(shift) > PLAN["burgers_shock_search_cells"] or abs(true_offset) > PLAN["burgers_shock_search_cells"]:
            shift = np.nan
        if np.isfinite(shift):
            win = (cell + np.arange(-2, 3)) % n
            original = np.linalg.norm((pred_frame-truth_frame)[win])
            aligned = np.linalg.norm((np.roll(pred_frame, -int(shift))-truth_frame)[win])
            gain = 1-aligned/max(original, 1e-30)
        else:
            gain = np.nan
        rows.append((t + 1, cell, scale[j, t, cell] / scale[j].mean(),
                     (scale[j, t] <= scale[j, t, cell]).mean(), shift, gain,
                     len(shocks(truth[j, max(0, t - 1)])), len(true_shocks), error[j, t, cell]))
    return np.asarray(rows)


def scales(config, root, split, keep_example):
    def calculate(inputs, truth, pred, sl):
        initial, nu = inputs
        full = np.concatenate((initial[:, None], pred), axis=1)
        r = ops.trajectory_residual(full, nu, config)
        z = ops.lift("adv_diff", full, r, nu, config)
        radius = config["spatial_radius"] * pred.shape[-1] / (2 * np.pi)
        floor = config["envelope_relative_floor"]
        unsigned = ops.lift("adv_diff", full, abs(r), nu, config)
        scale_steps = [envelope(v, s, 1, "wrap", floor) for v, s in ((r, 0), (unsigned, radius), (z, 0), (z, radius))]
        rows = {name: field_row(pred, truth, mix(value, .85)) for name, value in zip(STEPS, scale_steps)}
        base_scale = mix(scale_steps[-1], .85)
        for physical in PLAN["burgers_radii"]:
            env = ops.envelope(z, config | {"spatial_radius": physical})
            for lam in PLAN["burgers_lambdas"]:
                rows[f"lambda={lam:g}, radius={physical:g}"] = field_row(pred, truth, mix(env, lam))
        rows["stride=1"] = rows[OP]
        for stride in PLAN["time_strides"][1:]:
            cfg = config | {"frame_dt": config["frame_dt"] * stride,
                            "advection_substeps": config["advection_substeps"] * stride}
            sampled = full[:, ::stride]
            rr = ops.trajectory_residual(sampled, nu, cfg)
            zz = ops.lift("adv_diff", sampled, rr, nu, cfg)
            reduced = mix(ops.envelope(zz, cfg), .85)
            rows[f"stride={stride}"] = field_row(sampled[:, 1:], truth[:, stride-1::stride], reduced)
            rows[f"stride={stride} reference evaluation grid"] = field_row(
                sampled[:, 1:], truth[:, stride-1::stride], base_scale[:, stride-1::stride])
        base, spectrum = features(initial, nu)
        defect = np.column_stack((rms(r).ravel(), rms(z).ravel()))
        rows.update({"_input": base, "_spectrum": spectrum, "_joint": np.column_stack((spectrum, defect)), "_residual": defect})
        if split == "ood_high":
            rows["_shock"] = peak_diagnostics(pred, truth, base_scale)
        if keep_example and sl.start == 0:
            idx = PLAN["fixed_plot_examples"]["burgers_trajectory"]
            save_example(root, split, input=initial[idx], prediction=pred[idx], truth=truth[idx],
                         residual=r[idx], lifted=z[idx], scale=base_scale[idx])
        return rows
    return calculate


def gate_summary(cal, id_rows, high_rows, q):
    target = np.concatenate((id_rows[OP], high_rows[OP]))
    n_id = len(id_rows[OP])
    is_high = np.arange(len(target)) >= n_id
    score, scale, point_error = target.T
    base = {"coverage": score <= q, "width": 2*q*scale, "pixel": score <= q,
            "top_coverage": score <= q, "top_width": 2*q*scale, "infinite": np.zeros(len(target), dtype=bool)}
    gate = {}
    for mode in ("spectral energy", "mix logistic"):
        xc = cal["_spectrum"]
        xt = np.concatenate((id_rows["_spectrum"], high_rows["_spectrum"]))
        if mode == "spectral energy":
            sc, st = xc[:, -1], xt[:, -1]
        else:
            ref = len(xc) // 2
            _, both = estimated_weights(xc[:ref], np.concatenate((xc[ref:], xt)),
                                        {"base_weight_clip": [.02, 50], "weight_logistic_c": 1.})
            sc, st = both[:len(xc)-ref], both[len(xc)-ref:]
        roc_fpr, roc_tpr, _ = roc_curve(is_high, st, drop_intermediate=False)
        locked_curve = gate_point(st, point_error > .15, is_high, base, {"gate_p_test_thresholds": []})
        gate[mode] = {"cal_n": len(sc), "auroc": roc_auc_score(is_high, st), "fpr": roc_fpr, "tpr": roc_tpr,
                      "point_aurc": locked_curve["point_aurc"], "risk_curve": locked_curve["curve"], "operations": {}}
        for a in PLAN["id_gate_alpha"]:
            cut = quantile(sc, a)
            use = st <= cut
            gate[mode]["operations"][str(a)] = {
                "threshold": cut, "cal_fpr_realized": (sc > cut).mean(), "id_reject": (~use[~is_high]).mean(),
                "ood_reject": (~use[is_high]).mean(), "n_accepted": use.sum(), "n": len(use),
                "accepted_coverage": (score[use] <= q).mean() if use.any() else None,
                "accepted_width": (2 * q * scale[use]).mean() if use.any() else None,
                "accepted_point_failure": (point_error[use] > .15).mean() if use.any() else None,
            }
    return gate


def run(protocol, root, limit):
    config = protocol.config
    model = protocol.open_frozen()
    groups = {}
    for name in ("cal", "id", "ood", "ood_high"):
        data = arrays(protocol.data_path(name))
        groups[name] = collect(protocol, model, data, 64, protocol.frames,
                               scales(config, root, name, protocol.seed == PLAN["seeds"][0] and name == "ood_high"), limit=limit)
        groups[name]["_id"] = data["id"][:limit]
        print(f"Burgers {protocol.seed} {name} replay finished", flush=True)
    cal = groups.pop("cal")
    thresholds = calibrate(cal)
    locked = {str(a): protocol.reference["results"]["FNO"][str(a)]["summary"]["id_64"]["methods"]["OpRes-CP"]["cal_q"] for a in ALPHAS}
    thresholds[OP] = locked
    features_for = ("input", "spectrum", "residual", "joint")
    result = {"thresholds": thresholds, "metrics": {name: compare(rows, thresholds) for name, rows in groups.items()},
              "calibration_size": cal_sizes(cal, groups), "gates": gate_summary(cal, groups["id"], groups["ood_high"], locked["0.1"]),
              "spectral_support": {"cal_max": float(cal["_spectrum"][:, -1].max()),
                                   "id_max": float(groups["id"]["_spectrum"][:, -1].max()),
                                   "high_min": float(groups["ood_high"]["_spectrum"][:, -1].min())}}
    result["wcp"] = {name: wcp_grid(cal[OP], groups[name][OP], {k: cal[f"_{k}"] for k in features_for},
                                  {k: groups[name][f"_{k}"] for k in features_for}) for name in ("ood", "ood_high")}
    return result, {"cal": cal, **groups}
