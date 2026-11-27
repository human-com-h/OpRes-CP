"""Frozen Darcy radius, discretization, and boundary stress tests."""

import numpy as np

from darcy_data import apply_operator
from darcy_day15 import arrays
from darcy_day15_ops import lift, mixture, physical_scale
from phase2d_statistics import relative_l2
from phase3_common import (ALPHAS, OP, PLAN, STEPS, cal_sizes, calibrate, collect, compare,
                           envelope, field_row, rms, save_example, wcp_grid)


def boundary_rows(a, pred, truth, config, inv, lam):
    n = a.shape[-1]
    x = (np.arange(n) + .5) / n
    face = np.sin(np.pi * x)[None, :] * (PLAN["darcy_trace_rms_fraction"] * rms(pred)[:, 0, :])
    ext = np.sinh(np.pi * (1-x))[:, None] / np.sinh(np.pi) * face[:, None, :]
    rhs = np.zeros_like(a)
    rhs[:, 0] = 2 * n**2 * a[:, 0] * face
    spectral = lift(inv, a, rhs)
    harmonic = np.stack([lift("exact", cell, boundary) for cell, boundary in zip(a, rhs)])
    stressed = pred + ext
    interior = apply_operator(a, stressed) - rhs - 1
    z = lift(inv, a, interior)
    result = {"boundary stressed/interior": field_row(stressed, truth, mixture(physical_scale(z, config), lam)),
              "boundary stressed/plus spectral": field_row(stressed, truth, mixture(physical_scale(z+spectral, config), lam)),
              "boundary stressed/plus harmonic": field_row(stressed, truth, mixture(physical_scale(z+harmonic, config), lam))}
    result["_boundary_diagnostics"] = np.column_stack((rms(ext).ravel(), rms(spectral).ravel(),
        rms(harmonic).ravel(), rms(apply_operator(a, harmonic)-rhs).ravel(), rms(face).ravel()))
    return result


def scales(protocol, root, name, grid, keep_example=False):
    cfg, inv, lam = protocol.config, protocol.plan["inverse"], protocol.plan["lambda"]
    locked_radius = cfg["physical_radius"]

    def calculate(inputs, truth, pred, sl):
        a = inputs[0].astype(float)
        r = apply_operator(a, pred)-1
        z = lift(inv, a, r)
        unsigned = lift(inv, a, abs(r))
        values = (envelope(r, 0, 2, "reflect", cfg["relative_floor"]), physical_scale(unsigned, cfg),
                  envelope(z, 0, 2, "reflect", cfg["relative_floor"]), physical_scale(z, cfg))
        scales_ = [mixture(s, lam) for s in values]
        rows = {method: field_row(pred, truth, s) for method, s in zip(STEPS, scales_)}
        frozen_pixels = locked_radius * protocol.grid
        rows["fixed physical radius"] = rows[OP]
        pix = physical_scale(z, cfg | {"physical_radius": frozen_pixels / grid})
        rows["fixed pixel radius"] = field_row(pred, truth, mixture(pix, lam))
        if protocol.grid == 64:
            base = protocol.summaries((a,))
            defect = np.column_stack((rms(r).ravel(), rms(z).ravel()))
            rows.update({"_input": base, "_residual": defect, "_joint": np.column_stack((base, defect)),
                         "_full": protocol.full_grid((inputs[0],))})
            if name in ("cal", "id_test"):
                rows.update(boundary_rows(a, pred, truth, cfg, inv, lam))
        if name == "paired_test":
            rows["_prediction_stats"] = np.column_stack((rms(pred).ravel(), rms(truth).ravel(),
                rms(pred-truth).ravel(), rms(r).ravel(), rms(z).ravel(), rms(scales_[-1]).ravel()))
        if keep_example and sl.start == 0:
            idx = PLAN["fixed_plot_examples"]["darcy_field"]
            save_example(root, name, input=a[idx], prediction=pred[idx], truth=truth[idx], residual=r[idx],
                         lifted=z[idx], scale=scales_[-1][idx])
        return rows
    return calculate


def relative_reference(coarse, fine, coarse_coefficient):
    n = coarse.shape[-1]
    factor = fine.shape[-1] // n
    restricted = fine.astype(float).reshape(-1, n, factor, n, factor).mean((2, 4))
    r = apply_operator(coarse_coefficient.astype(float), restricted)-1
    return np.column_stack((relative_l2(coarse, restricted), rms(r).ravel(), rms(r[:, 1:-1, 1:-1]).ravel()))


def run_track(protocol, root, limit):
    model, grid = protocol.open_frozen(), protocol.grid
    cal_data = arrays(protocol.data_path("cal"))
    cal = collect(protocol, model, cal_data, grid, None, scales(protocol, root, "cal", grid), limit=limit)
    cal["_id"] = cal_data["id"][:limit]
    thresholds = calibrate(cal)
    ref = protocol.reference["tracks"][str(grid)]["FNO"]["splits"]
    label = "id_test_64" if grid == 64 else "paired_test_32"
    locked = {str(a): ref[label]["OpRes-CP"][str(a)]["q"] for a in ALPHAS}
    thresholds[OP] = thresholds["fixed physical radius"] = locked
    thresholds["fixed pixel radius"] = locked
    groups = {}
    targets = (("id_test", 64), ("ood_a", 64), ("ood_b", 64)) if grid == 64 else tuple(("paired_test", n) for n in (32, 64, 128))
    for name, n in targets:
        data = arrays(protocol.data_path(name, n))
        record = collect(protocol, model, data, n, None, scales(protocol, root, name, n,
                         grid == 64 and protocol.seed == PLAN["seeds"][0] and name == "id_test"), limit=limit)
        record["_id"] = data["id"][:limit]
        groups[f"{name}_{n}"] = record
        if grid == 32 and n == 64:
            refs_64 = data["u_f"][:limit], data["a_f"][:limit]
        if grid == 32 and n == 128:
            fine = data["u_f"][:limit]
            groups["paired_test_32"]["_reference_difference"] = relative_reference(data["u"][:limit], fine, data["a"][:limit])
            groups["paired_test_64"]["_reference_difference"] = relative_reference(refs_64[0], fine, refs_64[1])
        print(f"Darcy {protocol.seed} {grid} {name} {n} replay finished", flush=True)
    result = {"thresholds": thresholds, "metrics": {name: compare(rows, thresholds) for name, rows in groups.items()},
              "calibration_size": cal_sizes(cal, groups)}
    if grid == 64:
        keys = ("_full", "_input", "_residual", "_joint")
        result["wcp"] = {group: wcp_grid(cal[OP], groups[group][OP], {k[1:]: cal[k] for k in keys},
                                        {k[1:]: groups[group][k] for k in keys}) for group in ("ood_a_64", "ood_b_64")}
    else:
        for group in groups.values():
            if not np.array_equal(group["_id"], groups["paired_test_32"]["_id"]):
                raise ValueError("Darcy reference resolutions do not share the same physical samples")
    return result, {f"grid{grid}_{key}": value for key, value in {"cal": cal, **groups}.items()}


def run(protocol, root, limit):
    from phase2d_data import Protocol

    paired = Protocol(protocol.pde, protocol.seed, 32, root / "pressure_cache", protocol.frozen, protocol.device)
    primary, fields = run_track(protocol, root, limit)
    pressure, pressure_fields = run_track(paired, root, limit)
    protocol.data_matches.update({f"grid32/{k}": v for k, v in paired.data_matches.items()})
    return {"grid64": primary, "grid32": pressure}, fields | pressure_fields
