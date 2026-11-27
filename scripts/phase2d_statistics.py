"""Field-level metrics and feature-only calibration for Phase 2D."""

import math
from time import perf_counter

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score

from darcy_day15 import band_summary, estimated_weights, evaluate_bands, value_at_coverage, weighted_quantile
from darcy_phase0_eval import quantile


ALPHAS = (.10, .05)


def flat(value):
    return np.asarray(value).reshape(len(value), -1)


def relative_l2(center, truth):
    return np.linalg.norm(flat(center - truth), axis=1) / np.maximum(np.linalg.norm(flat(truth), axis=1), 1e-30)


def positive_scale(sigma, floor):
    return np.maximum(sigma, max(float(floor), 1e-12))


def calibrated_scores(center, scale, cal_truth):
    return (flat(np.abs(center - cal_truth)) / flat(scale)).max(axis=1)


def conformal_values(center, scale, truth, top, thresholds, point_errors=None):
    ratios = flat(np.abs(center - truth)) / flat(scale)
    point_errors = relative_l2(center, truth) if point_errors is None else point_errors
    result = {}
    for alpha, q in thresholds.items():
        result[alpha] = evaluate_bands(ratios, flat(scale), q, top) | {"point_errors":point_errors}
    return result


def summarize_values(row, alpha, matched=True):
    result = band_summary(row) | {"point_rel_l2":float(np.mean(row["point_errors"]))}
    if matched:
        matched_q = value_at_coverage(row["score"],1-float(alpha))
        result["matched_width"] = float(2 * matched_q * row["scale_mean"].mean())
    return result


def targets_for(scores, weights=None):
    if weights is None:
        thresholds = {str(a): quantile(scores,a) for a in ALPHAS}
        details = {a:{"threshold":q if np.isfinite(q) else "inf"} for a,q in thresholds.items()}
    else:
        terms = {str(a):weighted_quantile(scores,*weights,a) for a in ALPHAS}
        thresholds = {a:v[0] for a,v in terms.items()}
        details = {a:{"ess":v[2],"max_test_mass":float(v[1].max())} for a,v in terms.items()}
    return thresholds, details


def covariate_weights(x_cal, x_target, clip=(.02, 50.)):
    """The ratio estimator receives unlabeled covariates only."""
    return estimated_weights(x_cal, x_target, {"base_weight_clip": clip, "weight_logistic_c": 1.})


def mixed_gate(wc, wt, score, width, is_ood, point_error, q, anchors):
    p = wt / (wt + wc.sum())
    hit = score <= q
    if len(hit) != len(p) or not np.any(is_ood) or not np.any(~is_ood):
        raise ValueError("The mixed-flow gate requires aligned ID and OOD fields.")
    out = {}
    for threshold in anchors:
        keep = p <= threshold
        out[str(threshold)] = {
            "n": len(hit), "n_accepted": int(keep.sum()), "id_accept": float(keep[~is_ood].mean()),
            "ood_reject": float((~keep[is_ood]).mean()),
            "accepted_coverage": float(hit[keep].mean()) if keep.any() else None,
            "accepted_width": float(width[keep].mean()) if keep.any() else None,
            "accepted_point_failure": float((point_error[keep] > .15).mean()) if keep.any() else None,
        }
    return {"auroc": float(roc_auc_score(is_ood, p)), "operating_points": out}


def _weighted_depth_quantile(depths, w_cal, w_test, alpha):
    """Weighted conformal lower depth threshold with test mass w_test placed at 0."""
    total = float(np.sum(w_cal) + w_test)
    if total <= 0.0 or w_test > alpha * total + 1e-12:
        return 0.0, True
    order = np.argsort(-depths, kind="stable")
    cumulative = np.cumsum(w_cal[order])
    target = (1.0 - alpha) * total
    idx = int(np.searchsorted(cumulative, target - 1e-12, side="left"))
    if idx >= len(depths):
        return 0.0, True
    return float(depths[order[idx]]), False


class LocalSlicedCP:
    """LSCI projected-depth sets with exact RLCP weighting and a sampled envelope."""

    def __init__(self, residuals, cal_x, seed, slices=32, draws=64):
        self.residuals = flat(residuals)
        self.draws = draws
        self.rng = np.random.default_rng(seed)
        n, d = self.residuals.shape
        self.coordinates = self.rng.integers(0, d, (slices, 16))
        self.coefficients = self.rng.choice([-1., 1.], self.coordinates.shape) / 4
        self.reference = self.project(self.residuals)
        self.order = np.argsort(self.reference, axis=0, kind="stable").T
        self.sorted = np.take_along_axis(self.reference.T, self.order, axis=1)
        self.left, self.right = np.empty_like(self.order), np.empty_like(self.order)
        for i, row in enumerate(self.sorted):
            self.left[i] = np.searchsorted(row, row, side="left")
            self.right[i] = np.searchsorted(row, row, side="right")
        self.inverse = np.argsort(self.order, axis=1)
        self.center = cal_x.mean(axis=0)
        sd = cal_x.std(axis=0)
        self.spread = np.where(sd > 1e-8, sd, 1.)
        self.reference_x = (cal_x - self.center) / self.spread
        distances = cdist(self.reference_x, self.reference_x, "sqeuclidean")
        positive = distances[distances > 0]
        self.bandwidth = float(np.median(positive)) if len(positive) else 1.
        # Exact RLCP perturbation matching K(x, x_tilde) = exp(-||x - x_tilde||^2 / bandwidth).
        self.noise = math.sqrt(self.bandwidth / 2.0)

    def project(self, values):
        return np.einsum("nmk,mk->nm", values[:, self.coordinates], self.coefficients)

    def prepare(self, x_target):
        x = (x_target - self.center) / self.spread
        perturbed = x + self.rng.normal(size=x.shape) * self.noise
        cal_dist = cdist(perturbed, self.reference_x, "sqeuclidean")
        self_dist = np.sum((x - perturbed)**2, axis=1, keepdims=True)
        distance = np.concatenate((cal_dist, self_dist), axis=1)
        w = np.exp(-(distance - distance.min(axis=1, keepdims=True)) / self.bandwidth)
        return w / w.sum(axis=1, keepdims=True)

    def _region(self, weights):
        n = len(self.residuals)
        w_cal = weights[:-1]
        w_test = float(weights[-1])
        cal_mass = float(w_cal.sum())
        cumulative = np.c_[np.zeros(len(self.order)), np.cumsum(w_cal[self.order], axis=1)]
        if cal_mass > 0.0:
            lo = np.take_along_axis(cumulative, self.right, axis=1) / cal_mass
            hi = (cal_mass - np.take_along_axis(cumulative, self.left, axis=1)) / cal_mass
            depths = np.take_along_axis(2.0 * np.minimum(lo, hi), self.inverse, axis=1).min(axis=0)
        else:
            depths = np.zeros(n, dtype=float)
        levels, bounds = [], []
        for alpha in ALPHAS:
            level, is_inf = _weighted_depth_quantile(depths, w_cal, w_test, alpha)
            levels.append((level, is_inf))
            if is_inf or cal_mass <= 0.0:
                bounds.append(None)
                continue
            accepted = depths >= level - 1e-12
            probability = w_cal * accepted
            if probability.sum() <= 0.0:
                # In the zero-mass limit the sampled envelope is empty.
                bounds.append(None)
                continue
            selected = self.rng.choice(n, self.draws, replace=True, p=probability / probability.sum())
            sample = self.residuals[selected]
            bounds.append((sample.min(axis=0), sample.max(axis=0)))
        return cumulative, levels, bounds

    def evaluate(self, weights, target_residuals, top, point_errors):
        residuals = flat(target_residuals)
        projections = self.project(residuals)
        rows = [[] for _ in ALPHAS]
        native = [[] for _ in ALPHAS]
        native_inf = [[] for _ in ALPHAS]
        timing = []
        j = np.arange(len(self.order))
        for i, w in enumerate(weights):
            started = perf_counter()
            cumulative, _, bounds = self._region(w)
            timing.append((perf_counter() - started) * 1000)
            w_cal = w[:-1]
            w_test = float(w[-1])
            proj = projections[i]
            # Symmetric (n+1)-point weighted Tukey depth for exact native set membership.
            le_cal = np.take_along_axis(cumulative, self.right, axis=1)
            lt_cal = np.take_along_axis(cumulative, self.left, axis=1)
            le_tot = le_cal + w_test * (proj[:, None] <= self.sorted)
            ge_tot = 1.0 - (lt_cal + w_test * (proj[:, None] < self.sorted))
            cal_depths_full = np.take_along_axis(2.0 * np.minimum(le_tot, ge_tot), self.inverse, axis=1).min(axis=0)
            left = [np.searchsorted(a, v, side="left") for a, v in zip(self.sorted, proj)]
            right = [np.searchsorted(a, v, side="right") for a, v in zip(self.sorted, proj)]
            test_depth = float(np.min(2.0 * np.minimum(cumulative[j, right] + w_test, 1.0 - cumulative[j, left])))
            for idx, (alpha, limits) in enumerate(zip(ALPHAS, bounds)):
                q_native, is_inf_native = _weighted_depth_quantile(cal_depths_full, w_cal, w_test, alpha)
                native_inf[idx].append(is_inf_native)
                native[idx].append(True if is_inf_native else (test_depth >= q_native - 1e-12))
                if limits is None:
                    rows[idx].append((True, 1., np.inf, 1., np.inf, True))
                    continue
                lower, upper = limits
                hit = (residuals[i] >= lower) & (residuals[i] <= upper)
                width = upper - lower
                rows[idx].append((hit.all(), hit.mean(), width.mean(), hit[top[i]].mean(),
                                  width[top[i]].mean(), False))
        output = {}
        names = ("coverage", "pixel", "width", "top_coverage", "top_width", "infinite")
        for idx, alpha in enumerate(ALPHAS):
            values = {name: np.array(v) for name, v in zip(names, zip(*rows[idx]))}
            values["coverage"] = values["coverage"].astype(bool)
            values["infinite"] = values["infinite"].astype(bool)
            output[str(alpha)] = values | {
                "native_hit": np.asarray(native[idx], dtype=bool),
                "native_infinite": np.asarray(native_inf[idx], dtype=bool),
                "point_errors": point_errors,
                "region_ms": np.asarray(timing),
            }
        return output
