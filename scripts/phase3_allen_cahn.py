"""Frozen Allen–Cahn time-grid sensitivity and within-Strang mechanism audit."""

from time import perf_counter

import numpy as np
import torch
from scipy.fft import irfft2, rfft2

from allen_cahn_phase2_data import spectrum
from allen_cahn_phase2_probe import reconstruct, scale
from darcy_day15 import arrays, digest, value_at_coverage
from phase2d_ac import generate as generate_horizon
from phase3_common import (ALPHAS, OP, PLAN, STEPS, cal_sizes, calibrate, collect, compare,
                           envelope, field_row, mix, rms, save_example, wcp_grid)


def jacobian(mid, rho, config):
    reaction = np.clip(rho[:, None, None, None] * (1-3*mid**2), -config["reaction_clip"], config["reaction_clip"])
    return np.minimum(reaction-config["positive_damping"]*np.maximum(reaction, 0), config["max_growth"])


def midpoints(values):
    return (values[:, 1:] + values[:, :-1]) * .5


def strang_cpu(residual, nu, config, jac):
    dt = config["dt"]
    k2, _ = spectrum(residual.shape[-1])
    half = np.exp(-.5*dt*(nu[:, None, None]*k2+config["gamma"]))
    state = np.zeros_like(residual[:, 0])
    z = np.empty_like(residual)
    for j in range(residual.shape[1]):
        force = dt * residual[:, j] / 2
        state = irfft2(half*rfft2(state+force))
        if jac is not None:
            state *= np.exp(dt * jac[:, j])
        state = irfft2(half*rfft2(state)) + force
        z[:, j] = state
    return z


def strang_gpu(full, nu, rho, config):
    """Batch-vectorized over trajectories and Fourier modes, with causal recurrence in time."""
    dt = config["dt"]
    k2, mask = (torch.as_tensor(x, device=full.device) for x in spectrum(full.shape[-1]))
    diff = -nu[:, None, None] * k2.to(dtype=full.dtype)
    half = torch.exp(dt * (diff-config["gamma"]) / 2)
    state = torch.zeros_like(full[:, 0])
    z = []
    for j in range(1, full.shape[1]):
        left, right = full[:, j-1], full[:, j]
        mid = (left+right) * .5
        mid_fft = torch.fft.rfft2(mid)
        nonlin = torch.fft.irfft2(mask * torch.fft.rfft2(mid*(1-mid*mid)))
        r = (right-left)/dt - torch.fft.irfft2(diff*mid_fft) - rho[:, None, None]*nonlin
        force = dt*r/2
        jac = (rho[:, None, None]*(1-3*mid.square())).clamp(-config["reaction_clip"], config["reaction_clip"])
        multiplier = torch.exp(dt*(jac - config["positive_damping"]*jac.clamp_min(0)).clamp_max(config["max_growth"]))
        state = torch.fft.irfft2(half * torch.fft.rfft2(state+force))
        state = torch.fft.irfft2(half * torch.fft.rfft2(multiplier*state)) + force
        z.append(state)
    return torch.stack(z, dim=1)


def timing(full, nu, rho, config, device):
    if device.type != "cuda":
        return {}
    channels = (full, nu, rho)
    resident = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in channels]
    expected, _ = reconstruct(full, nu, rho, config, "reaction_strang")
    candidate = strang_gpu(*resident, config).cpu().numpy()
    error = float(np.linalg.norm((candidate-expected).ravel()) / np.linalg.norm(expected.ravel()))
    error_scale = float(np.linalg.norm((scale(candidate, config)-scale(expected, config)).ravel()) /
                        np.linalg.norm(scale(expected, config).ravel()))

    def end_to_end():
        return strang_gpu(*[torch.as_tensor(x, dtype=torch.float32, device=device) for x in channels], config).cpu()

    paths = {"cpu": lambda: reconstruct(full, nu, rho, config, "reaction_strang"),
             "gpu_resident": lambda: strang_gpu(*resident, config), "gpu_copy_in_and_out": end_to_end}
    out = {"relative_l2_lift_error": error, "relative_l2_scale_error": error_scale, "n": 20}
    for name, path in paths.items():
        times = []
        for repeat in range(23):
            torch.cuda.synchronize(device)
            t = perf_counter()
            path()
            torch.cuda.synchronize(device)
            if repeat >= 3:
                times.append(1000*(perf_counter()-t))
        out[f"{name}_p50_ms"], out[f"{name}_p90_ms"] = np.quantile(times, [.5, .9])
    return out


def short_scales(p, name):
    c = p.config

    def calculate(inputs, truth, pred, sl):
        initial, nu, rho = inputs
        full = np.concatenate((initial[:, None], pred), axis=1)
        z, r = reconstruct(full, nu, rho, c, "reaction_strang")
        jac = jacobian(midpoints(full), rho, c)
        unsigned = strang_cpu(abs(r), nu, c, jac)
        scale_steps = (mix(envelope(r, 0, 2, "wrap", .1), c["lambda"]), scale(unsigned, c),
                       mix(envelope(z, 0, 2, "wrap", .1), c["lambda"]), scale(z, c))
        rows = {method: field_row(pred, truth, value) for method, value in zip(STEPS, scale_steps)}
        rows["stride=1"] = rows[OP]
        for stride in PLAN["time_strides"][1:]:
            cfg = c | {"dt": c["dt"] * stride}
            sampled = full[:, ::stride]
            zz, _ = reconstruct(sampled, nu, rho, cfg, "reaction_strang")
            y = truth[:, stride-1::stride]
            rows[f"stride={stride}"] = field_row(sampled[:, 1:], y, scale(zz, cfg))
            rows[f"stride={stride} reference evaluation grid"] = field_row(sampled[:, 1:], y, scale_steps[-1][:, stride-1::stride])
        x = p.summaries(inputs)
        defect = np.column_stack((rms(r).ravel(), rms(z).ravel()))
        rows.update({"_input": x, "_residual": defect, "_joint": np.column_stack((x, defect)), "_full": p.full_grid(inputs)})
        return rows
    return calculate


def long_scales(p, root, grid, profile):
    c = p.config

    def calculate(inputs, truth, pred, sl):
        initial, nu, rho = inputs
        full = np.concatenate((initial[:, None], pred), axis=1)
        target = np.concatenate((initial[:, None], truth), axis=1).astype(float)
        z, r = reconstruct(full, nu, rho, c, "reaction_strang")
        jp, jt = (jacobian(midpoints(value), rho, c) for value in (full, target))
        off = strang_cpu(r, nu, c, None)
        oracle = strang_cpu(r, nu, c, jt)
        sp, so, st = (scale(value, c) for value in (z, off, oracle))
        row = {name: field_row(pred, truth, value) for name, value in (("predicted_jacobian", sp),
                        ("jacobian_off", so), ("target_jacobian_diagnostic", st))}
        mismatch = (jp < 0) != (jt < 0)
        interface = abs(abs(midpoints(target))-1/np.sqrt(3)) <= .15
        axes = tuple(range(1, target.ndim))
        row["_manifold"] = np.column_stack((mismatch.mean(axis=axes), (mismatch & interface).sum(axis=axes),
            interface.sum(axis=axes), abs(jp-jt).mean(axis=axes), rms(sp).ravel()/rms(so).ravel(),
            rms(st).ravel()/rms(so).ravel(), rms(sp).ravel()/rms(st).ravel()))
        if p.seed == PLAN["seeds"][0] and grid == 128 and sl.start == 0:
            idx = PLAN["fixed_plot_examples"]["allen_cahn_trajectory"]
            save_example(root, "paired_128", input=initial[idx], prediction=pred[idx], truth=truth[idx],
                         residual=r[idx], lifted=z[idx], scale=sp[idx], jacobian_pred=jp[idx], jacobian_target=jt[idx])
            profile.update(timing(full[idx:idx+1], nu[idx:idx+1], rho[idx:idx+1], c, p.device))
        return row
    return calculate


def run(p, root, limit):
    short, groups = p.open_frozen(), {}
    for name in ("cal", "id", "ood"):
        data = arrays(p.data_path(name))
        group = collect(p, short, data, 64, p.cal_frames, short_scales(p, name), batch=6, limit=limit)
        groups[f"short_{name}"] = group | {"_id": data["id"][:limit]}
        print(f"Allen-Cahn {p.seed} FNO short {name} replay finished", flush=True)
    cal = groups.pop("short_cal")
    thresholds = calibrate(cal)
    thresholds[OP] = {str(a): p.reference["results"]["FNO"]["thresholds"][str(a)]["OpRes-CP"] for a in ALPHAS}
    result = {"thresholds": thresholds, "metrics": {k: compare(v, thresholds) for k, v in groups.items()},
              "calibration_size": cal_sizes(cal, groups)}
    keys = ("_full", "_input", "_residual", "_joint")
    result["wcp"] = {"short_ood": wcp_grid(cal[OP], groups["short_ood"][OP], {k[1:]: cal[k] for k in keys},
                                            {k[1:]: groups["short_ood"][k] for k in keys})}
    long_model = p.open_frozen(unet=True)
    fresh_path = generate_horizon(p, "paired")
    fresh = arrays(fresh_path)
    result["long_pair_sha256"] = digest(fresh_path)
    result["manifold"], result["profile"] = {}, {}
    for grid in (64, 128):
        row = collect(p, long_model, fresh, grid, p.frames, long_scales(p, root, grid, result["profile"]), batch=4, limit=limit)
        row["_id"] = fresh["id"][:limit]
        groups[f"long_unet_{grid}"] = row
        result["manifold"][str(grid)] = {
            "matched_width": {name: {str(a): 2*value_at_coverage(values[:, 0], 1-a)*float(values[:, 1].mean()) for a in ALPHAS}
                              for name, values in row.items() if not name.startswith("_")},
            "point_rel_l2": float(row["predicted_jacobian"][:, 2].mean()),
            "jacobian_sign_mismatch": float(row["_manifold"][:, 0].mean()),
            "interface_mismatch": float(row["_manifold"][:, 1].sum()/max(1, row["_manifold"][:, 2].sum())),
            "mean_jacobian_absolute_difference": float(row["_manifold"][:, 3].mean()),
            "pred_on_over_off_rms_scale": float(row["_manifold"][:, 4].mean()),
            "target_on_over_off_rms_scale": float(row["_manifold"][:, 5].mean()),
        }
        print(f"Allen-Cahn {p.seed} U-Net long paired {grid} replay finished", flush=True)
    return result, {"short_cal": cal, **groups}
