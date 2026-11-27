"""Validate the formal replays and combine them with the accepted Phase 2D returns."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t


RESULTS = Path("results")
STAGE = RESULTS / "phase2_kaggle_private"
DOWNLOADS = STAGE / "downloads"
OUT = RESULTS / "phase3"
PLAN_PATH = Path("configs/phase3.json")
PLAN = json.loads(PLAN_PATH.read_bytes())
SEEDS = PLAN["seeds"]
BATCHES = {"darcy": "3a", "burgers": "3b", "allen_cahn": "3c"}
FORWARD = {"darcy": "2da", "burgers": "2db", "allen_cahn": "2dc"}
PHASE2_OP = "OpRes-CP"
OP = "OpRes locked"
ENSEMBLE = "5-FNO Ensemble + CP | mean center"
SINGLE_ENSEMBLE = "5-FNO Ensemble + CP | single center"
DROPOUT = "30-draw MC-Dropout + CP"
GRID_STUDIES = (
    ("darcy", 64, "paired_test_64", "paired_test_128", "Darcy 64² → 128²"),
    ("darcy", 32, "paired_test_32", "paired_test_64", "Darcy 32² → 64²"),
    ("darcy", 32, "paired_test_32", "paired_test_128", "Darcy 32² → 128²"),
    ("burgers", 64, "paired_64", "paired_128", "Burgers 64 → 128"),
    ("burgers", 64, "paired_64", "paired_256", "Burgers 64 → 256"),
    ("allen_cahn", 64, "paired_64_20", "paired_128_20", "Allen–Cahn T=1"),
    ("allen_cahn", 64, "paired_64_40", "paired_128_40", "Allen–Cahn T=2"),
)


def read(path):
    return json.loads(path.read_bytes())


def result_root(pde, seed):
    return DOWNLOADS / BATCHES[pde] / f"seed_{seed}/results/phase3/{pde}/seed_{seed}"


def phase2_root(batch, pde, seed):
    return DOWNLOADS / batch / f"seed_{seed}/results/phase2d/{pde}/seed_{seed}"


def stat(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    avg = values.mean()
    sd = values.std(ddof=1) if n > 1 else np.nan
    half = t.ppf(.975, df=n-1) * sd / np.sqrt(n) if n > 1 else np.nan
    return n, float(avg), float(sd), float(avg-half), float(avg+half)


def horizon(pde, seed):
    return read(phase2_root("2dh", "allen_cahn_horizon", seed) / "metrics.json")


def phase2_metrics(pde, seed, grid=64):
    return read(phase2_root(FORWARD[pde], pde, seed) / f"grid{grid}/metrics.json")


def validate(pde, seed, report, root):
    def check(predicate, name):
        if not predicate:
            raise ValueError(f"{pde} seed {seed} disagrees with the locked design: {name}")

    check(report["seed"] == seed and report["pde"] == pde and report["gpu"] == "Tesla T4"
          and report["limit"] is None, "formal seed/device")
    check(report["plan_sha256"] == hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest(), "plan hash")
    frozen = read(STAGE / f"phase2d_frozen/seed_{seed}/frozen_manifest.json")["weights"]
    expected = {"darcy": ("darcy32", "darcy64"), "burgers": ("burgers",),
                "allen_cahn": ("allen_cahn", "allen_cahn_unet")}[pde]
    check(report["weights"] == {k: frozen[k]["sha256"] for k in expected}, "weight hash")
    if pde == "allen_cahn":
        check(report["long_pair_sha256"] == horizon(pde,seed)["sha256"]["paired"], "independent horizon paired hash")
    checks = report["data_hash_matches"]
    check(checks and all(checks.values()), f"reference data hashes {checks}")
    groups = {"darcy": ("grid64_cal", "grid64_id_test_64", "grid64_ood_a_64", "grid64_ood_b_64",
                        "grid32_cal", "grid32_paired_test_32", "grid32_paired_test_64", "grid32_paired_test_128"),
              "burgers": ("cal", "id", "ood", "ood_high"),
              "allen_cahn": ("short_cal", "short_id", "short_ood", "long_unet_64", "long_unet_128")}[pde]
    ids = {}
    for group in groups:
        with np.load(root / f"fields_{group}.npz") as z:
            ids[group] = z["_id"]
            check(all(z[k].shape[0] == 500 for k in z.files) and len(np.unique(ids[group])) == 500, f"sample size {group}")
    for name, i in ids.items():
        if "cal" in name:
            check(all(not np.intersect1d(i, x).size for name2, x in ids.items() if "cal" not in name2), f"Cal/Test IDs {name}")
    pairs = {"darcy": ("grid32_paired_test_32", "grid32_paired_test_64", "grid32_paired_test_128"),
             "allen_cahn": ("long_unet_64", "long_unet_128"), "burgers": ()}[pde]
    if pairs:
        check(all(np.array_equal(ids[pairs[0]], ids[g]) for g in pairs[1:]), "paired sample IDs")
    max_delta, max_ratio, comparisons = 0., 0., 0
    tracks = ((32, report["grid32"]), (64, report["grid64"])) if pde == "darcy" else ((64, report),)
    for grid, node in tracks:
        parent = phase2_metrics(pde, seed, grid)["results"]
        for name, rows in node["metrics"].items():
            ref_name = name if pde == "darcy" else name+"_64" if pde == "burgers" else name.removeprefix("short_")+"_64_20"
            for alpha, rec in rows[OP].items():
                prior = parent[ref_name][PHASE2_OP][alpha]
                max_delta = max(max_delta, abs(rec["coverage"]-prior["coverage"]))
                max_ratio = max(max_ratio, abs(rec["width"]/prior["width"]-1))
                comparisons += 1
    check(max_delta <= .002000001 and max_ratio < .005, f"frozen OpRes replay {max_delta:g}/{max_ratio:g}")
    return {"groups": len(groups), "hash_matches": len(checks), "opres_comparisons": comparisons,
            "max_coverage_difference": max_delta, "max_relative_width_difference": max_ratio}


def load(allow_partial=False):
    data, audit = defaultdict(dict), defaultdict(dict)
    for pde in BATCHES:
        for seed in SEEDS:
            root = result_root(pde, seed)
            file = root / "metrics.json"
            if not file.is_file():
                if allow_partial:
                    continue
                raise FileNotFoundError(file)
            d = read(file)
            try:
                a = validate(pde, seed, d, root)
            except FileNotFoundError:
                if allow_partial:
                    continue
                raise
            data[pde][seed], audit[pde][seed] = d, a
    return data, audit


def summaries(data):
    output = defaultdict(dict)
    for pde, seeds in data.items():
        for seed, d in seeds.items():
            def add(section, track, group, method, alpha, rows):
                for metric, val in rows.items():
                    if val is None or isinstance(val, (int, float)) or val == "inf":
                        output[pde, section, str(track), group, method, str(alpha), metric][seed] = val

            tracks = ((32, d["grid32"]), (64, d["grid64"])) if pde == "darcy" else ((64, d),)
            for grid, node in tracks:
                for group, methods in node["metrics"].items():
                    for name, levels in methods.items():
                        for alpha, row in levels.items():
                            add("mechanism", grid, group, name, alpha, row)
                for size, groups in node["calibration_size"].items():
                    for group, levels in groups.items():
                        for alpha, row in levels.items():
                            add("calibration_size", grid, group, size, alpha, row)
                for group, variants in node.get("wcp", {}).items():
                    for name, configs in variants.items():
                        for config, levels in configs.items():
                            for alpha, row in levels.items():
                                mean_band = {k: v for k, v in row.items() if k != "matched_width"}
                                add("wcp", grid, group, f"{name}, {config}", alpha, mean_band)
            if pde == "burgers":
                for method, gate in d["gates"].items():
                    scores = {k: gate[k] for k in ("auroc", "point_aurc", "cal_n")}
                    scores["fpr_at_95_tpr"] = min(x for x, y in zip(gate["fpr"], gate["tpr"]) if y >= .95)
                    add("gates", 64, "mixed", method, "", scores)
                    for alpha, vals in gate["operations"].items():
                        add("gates", 64, "mixed", method, alpha, vals)
    return output


def write_csv(series):
    with (OUT / "phase3_ablation_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("pde", "section", "base_grid", "group", "method", "alpha", "metric",
                         "n_seeds", "n_finite_seeds", "mean", "std", "ci95_low", "ci95_high", *(f"seed_{s}" for s in SEEDS)))
        for key, s in sorted(series.items()):
            vals = [v for v in s.values() if isinstance(v, (int, float))]
            if "inf" in s.values():
                result = ("inf", "", "", "")
            elif vals:
                result = tuple(x if np.isfinite(x) else "" for x in stat(vals)[1:])
            else:
                result = ("",) * 4
            writer.writerow((*key, len(s), len(vals), *result, *(s.get(seed) for seed in SEEDS)))


def diagnostics(data):
    def records(pde, group, columns):
        values = {}
        for seed in data.get(pde, {}):
            with np.load(result_root(pde,seed)/f"fields_{group}.npz") as raw:
                values[seed] = [raw[k] for k in columns]
        return values

    result = {"burgers_high": {}, "darcy_pressure": {}, "allen_cahn": {}}
    shock = records("burgers", "ood_high", (OP, "_shock", "lambda=0, radius=0.08"))
    for seed, (op, rows, scalar) in shock.items():
        q = data["burgers"][seed]["thresholds"]
        miss = op[:,0] > q[OP]["0.1"]
        early = miss & (rows[:,0]>=3) & (rows[:,0]<=5)
        matched = early & np.isfinite(rows[:,4])
        result["burgers_high"][str(seed)] = {"missed": int(miss.sum()), "early_missed": int(early.sum()),
            "matched_shock_count": int(matched.sum()), "scalar_rescues": int((miss & (scalar[:,0] <= q["lambda=0, radius=0.08"]["0.1"])).sum()),
            "early_scale_to_mean_median": float(np.median(rows[early,2])), "early_local_scale_rank_median": float(np.median(rows[early,3])),
            "matched_abs_shift_cells_median": float(np.median(abs(rows[matched,4]))),
            "matched_shift_nonzero_fraction": float((abs(rows[matched,4])>0).mean()),
            "matched_align_local_error_gain_median": float(np.median(rows[matched,5])),
            "early_shock_count_drop_fraction": float((rows[early,6]>rows[early,7]).mean())}
    refs = [records("darcy", f"grid32_paired_test_{n}", ("_reference_difference", "_prediction_stats")) for n in (32,64)]
    fine = records("darcy", "grid32_paired_test_128", ("_prediction_stats",))
    for seed, (r32, e32) in refs[0].items():
        r64, e64 = refs[1][seed]
        e128, = fine[seed]
        result["darcy_pressure"][str(seed)] = {"reference32": r32.mean(0).tolist(), "reference64": r64.mean(0).tolist(),
            "normalized_solution_defect_ratio_32_over_64": float(r32[:,0].mean()/r64[:,0].mean()),
            "prediction_stats_32": e32.mean(0).tolist(), "prediction_stats_64": e64.mean(0).tolist(),
            "prediction_stats_128": e128.mean(0).tolist(),
            "prediction_error_ratio_128_over_32": float(e128[:,2].mean()/e32[:,2].mean()),
            "scale_rms_ratio_128_over_32": float(e128[:,5].mean()/e32[:,5].mean())}
    for seed, d in data.get("allen_cahn", {}).items():
        result["allen_cahn"][str(seed)] = {"manifold": d["manifold"], "profile": d["profile"]}
    return result


def grid_gain():
    rng = np.random.default_rng(PLAN["bootstrap_seed"])
    indices = [rng.integers(0, 500, (PLAN["bootstrap_resamples"], 500)) for seed in SEEDS]
    results = []

    def field_hits(pde, grid, seed, group, method, a):
        if pde == "darcy" and grid == 64 and method in (ENSEMBLE, SINGLE_ENSEMBLE):
            name = f"seed_{seed}"
            with np.load(DOWNLOADS / f"2a/{name}/results/phase2_darcy/{name}/grid64/FNO/field_metrics.npz") as z:
                return z[f"{group}__{method}__field__{a}__coverage"].astype(float)
        root = phase2_root(FORWARD[pde],pde,seed)
        with np.load(root / f"grid{grid}/fields/{group}.npz") as z:
            return z[f"{method}__{a}__coverage"].astype(float)

    for pde,grid,core,target,label in GRID_STUDIES:
        methods = (SINGLE_ENSEMBLE,ENSEMBLE,DROPOUT) if grid==64 else (SINGLE_ENSEMBLE,ENSEMBLE)
        for a in ("0.1","0.05"):
            for comparator in methods:
                rows = []
                for seed in SEEDS:
                    op_lo,op_hi = (field_hits(pde,grid,seed,g,PHASE2_OP,a) for g in (core,target))
                    base_lo,base_hi = (field_hits(pde,grid,seed,g,comparator,a) for g in (core,target))
                    rows.append((op_hi-op_lo)-(base_hi-base_lo))
                means = [r.mean() for r in rows]
                boot = np.mean([r[ix].mean(axis=1) for r,ix in zip(rows,indices)],axis=0)
                lo,hi = np.quantile(boot,(.025,.975))
                results.append({"pde":pde,"track":label,"comparator":comparator,"alpha":a,
                    "mean_gain_pp":100*np.mean(means),"boot_low_pp":100*lo,"boot_high_pp":100*hi,
                    **{f"seed_{seed}":100*value for seed,value in zip(SEEDS,means)}})
    return results


def main():
    data, audit = load()
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(summaries(data))
    (OUT / "phase3_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (OUT / "phase3_boundary_diagnostics.json").write_text(json.dumps(diagnostics(data), indent=2, allow_nan=False)+"\n",encoding="utf-8")
    gains = grid_gain()
    with (OUT / "phase3_grid_gain.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(gains[0]))
        w.writeheader()
        w.writerows(gains)
    print({pde: len(v) for pde,v in data.items()})


if __name__ == "__main__":
    main()
