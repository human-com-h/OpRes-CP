from time import perf_counter_ns

import numpy as np
from scipy.fft import dstn, idstn
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.stats import rankdata, spearmanr

from darcy_data import apply_operator, boundary_diagonal, faces, matrix_indices
from darcy_phase0_eval import cell_spectrum, correlation, signed_inverse


AXES = (-2, -1)
INVERSES = ("const", "local", "liouville")
DIAGNOSTICS = (*INVERSES, "pcg2", "pcg3", "exact")


def rms(x):
    return np.sqrt(np.mean(x*x, axis=AXES, keepdims=True))


def poisson(r):
    eig, _ = cell_spectrum(r.shape[-1])
    return idstn(dstn(r, type=2, norm="ortho", axes=AXES) / eig,
                 type=2, norm="ortho", axes=AXES)


def lift(name, a, r):
    if name == "const":
        return signed_inverse(a, r)
    if name == "local":
        return poisson(r / a)
    if name == "liouville":
        inv_sqrt = 1 / np.sqrt(a)
        return inv_sqrt * poisson(inv_sqrt * r)
    if name == "exact":
        n = a.shape[-1]
        h, v = faces(a)
        diagonal = boundary_diagonal(a)
        diagonal[:, :-1] += h
        diagonal[:, 1:] += h
        diagonal[:-1, :] += v
        diagonal[1:, :] += v
        links = -np.r_[h.ravel(), v.ravel()]
        matrix = coo_matrix((np.r_[diagonal.ravel(), links, links], matrix_indices(n)),
                            shape=(n*n, n*n)).tocsc()
        return spsolve(matrix, r.ravel() / n**2).reshape(n, n)
    return pcg(a, r, int(name[-1]))


def pcg(a, r, steps):
    x = np.zeros_like(r)
    residual = r.copy()
    z = lift("liouville", a, residual)
    p = z.copy()
    rz = np.sum(residual * z)
    for i in range(steps):
        if rz == 0:
            break
        ap = apply_operator(a, p)
        step = rz / np.sum(p * ap)
        x += step * p
        residual -= step * ap
        if i + 1 < steps:
            z = lift("liouville", a, residual)
            next_rz = np.sum(residual * z)
            p = z + (next_rz / rz) * p
            rz = next_rz
    return x


def physical_scale(z, config):
    radius = config["physical_radius"] * z.shape[-1]
    spatial_sigma = (0,)*(z.ndim-2) + (radius, radius)
    return np.sqrt(gaussian_filter(z*z, spatial_sigma, mode="reflect")
                   + (config["relative_floor"] * rms(z))**2)


def mixture(s, lam):
    return np.sqrt((1-lam)*rms(s)**2 + lam*s*s)


def rho(x, y):
    return float(spearmanr(x, y).statistic)


def spatial_rho(x, y):
    x, y = (t.reshape(len(t), -1) for t in (x, y))
    return float(np.nanmean(correlation(rankdata(x, axis=1), rankdata(y, axis=1))))


def timing_summary(elapsed):
    return dict(zip(("p50_ms", "p90_ms"), np.quantile(elapsed, [0.5, 0.9]).tolist()))


def timed(call):
    begin = perf_counter_ns()
    result = call()
    return result, (perf_counter_ns() - begin) / 1e6


def inverse_diagnostics(a, r, err, config):
    result = {}
    lifts = {}
    linf = np.abs(err).max(axis=AXES)
    for name in DIAGNOSTICS:
        lift(name, a[0], r[0])
        measured = [timed(lambda: lift(name, aa, rr)) for aa, rr in zip(a, r)]
        z = np.stack([x for x, _ in measured])
        lifts[name] = z
        denom = np.linalg.norm(r.reshape(len(r), -1), axis=1)
        defect = apply_operator(a, z) - r
        result[name] = {
            "error_spatial_spearman": spatial_rho(abs(z), abs(err)),
            "envelope_spatial_spearman": spatial_rho(physical_scale(z, config), abs(err)),
            "cross_field_l2_spearman": rho(rms(z).ravel(), linf),
            "cross_field_linf_spearman": rho(np.abs(z).max(axis=AXES), linf),
            "relative_residual_l2_median": float(np.median(np.linalg.norm(
                defect.reshape(len(r), -1), axis=1) / denom)),
            "inverse_time": timing_summary([t for _, t in measured]),
        }
    result["exact"]["recovery_linf"] = float(np.max(abs(lifts["exact"] - err)))
    return result, lifts
