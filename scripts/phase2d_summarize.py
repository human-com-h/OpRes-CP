"""Audit and aggregate completed Phase 2D returns without contacting Kaggle."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


SEEDS = (42, 137, 314, 271, 811)
PROTOCOLS = {
    "2da": ("phase2d/darcy/seed_{seed}/grid64", "phase2d/darcy/seed_{seed}/grid32"),
    "2db": ("phase2d/burgers/seed_{seed}/grid64",),
    "2dc": ("phase2d/allen_cahn/seed_{seed}/grid64",),
    "2dh": ("phase2d/allen_cahn_horizon/seed_{seed}",),
}
REGION_METRICS = (
    "coverage", "pixel", "top_coverage", "width", "top_width", "matched_width",
    "finite_width", "finite_field_coverage", "infinite_rate", "point_rel_l2",
    "n", "finite_n", "accepted_n", "abstain", "accepted_risk",
    "ess", "max_test_mass", "weight_fit_wall_ms", "post_calibration_p50_ms",
    "native_slice_set_coverage", "paired_coverage_diff_vs_off",
)
KEYS = ("batch", "grid", "model", "group", "method", "alpha", "metric")


def validate_band(row, location):
    if row["n"] != 500 or not 0 <= row["coverage"] <= 1:
        raise ValueError(f"Incorrect evaluation denominator or coverage: {location}")
    if not 0 <= row["infinite_rate"] <= 1 or not math.isclose(
        row["finite_n"] / row["n"] + row["infinite_rate"], 1, abs_tol=1e-9
    ):
        raise ValueError(f"Incorrect finite subset denominator: {location}")
    if not 0 <= row["abstain"] <= 1 or not math.isclose(
        row["accepted_n"] / row["n"] + row["abstain"], 1, abs_tol=1e-9
    ):
        raise ValueError(f"Incorrect accepted subset denominator: {location}")


def bands(data, batch, grid, add):
    if batch == "2dh":
        if not data.get("fresh_disjoint") or data["counts"] != [500]*5:
            raise ValueError("The independent time streams did not pass separation/size checks.")
        for model, groups in data["results"].items():
            for group, variants in groups.items():
                for jacobian, strategies in variants.items():
                    for policy, alphas in strategies.items():
                        method = f"{jacobian}/{policy}"
                        if policy == "separate_block_Bonferroni":
                            for metric, value in alphas.items():
                                add(grid,model,group,method,"0.1",metric,value)
                            continue
                        for alpha, row in alphas.items():
                            validate_band(row, (model,group,method,alpha))
                            for metric in REGION_METRICS:
                                if metric in row:
                                    add(grid,model,group,method,alpha,metric,row[metric])
            for grid_key, timing in data["latency"][model].items():
                for metric, value in timing.items():
                    if metric.endswith("_ms"):
                        add(grid_key,model,"paired","profile","",metric,value)
        return
    if data.get("device") != "Tesla T4":
        raise ValueError(f"Expected a Tesla T4 return for {batch} grid {grid}.")
    for group, methods in data["results"].items():
        audit = data["cal_test_audit"][group]
        if audit["n"] != 500 or not audit["disjoint_cal"]:
            raise ValueError(f"Invalid calibration/test audit for {batch} grid {grid}, {group}.")
        for method, alphas in methods.items():
            for alpha, row in alphas.items():
                validate_band(row, (batch,grid,group,method,alpha))
                for metric in REGION_METRICS:
                    if metric in row:
                        add(grid,"FNO",group,method,alpha,metric,row[metric])
        for method, timing in data["latency"][group].items():
            for metric, value in timing.items():
                add(grid,"FNO",group,method,"",metric,value)
    for method, seconds in data["training_seconds"].items():
        add(grid,"FNO","train",method,"","seconds",seconds)
    ensemble = data["ensemble_total_training_seconds"]
    if ensemble is None:
        ensemble = data["phase2a_ensemble"].get("ensemble_total_training_seconds")
        for target_grid, timing in data["phase2a_ensemble"].get("ensemble_timing_export",{}).items():
            for metric, value in timing.items():
                if isinstance(value,(int,float)) and not isinstance(value,bool):
                    add(grid,"FNO",f"2A grid={target_grid}","ensemble reused","",metric,value)
                elif isinstance(value,dict):
                    for key, entry in value.items():
                        if isinstance(entry,(int,float)) and not isinstance(entry,bool):
                            add(grid,"FNO",f"2A grid={target_grid}","ensemble reused","",f"{metric}/{key}",entry)
    if ensemble is not None:
        add(grid,"FNO","train","ensemble total","","seconds",ensemble)


def summarize(downloads, require_complete=False):
    series = defaultdict(dict)
    completed = defaultdict(list)
    for batch, locations in PROTOCOLS.items():
        for seed in SEEDS:
            paths = [downloads/batch/f"seed_{seed}"/"results"/p.format(seed=seed)/"metrics.json"
                     for p in locations]
            if not all(p.is_file() for p in paths):
                continue
            for path in paths:
                data = json.loads(path.read_bytes())
                if data["seed"] != seed or data["smoke"]:
                    raise ValueError(f"Invalid formal return for {batch} seed {seed}.")

                def add(grid,model,group,method,alpha,metric,value):
                    key = batch,str(grid),model,group,method,alpha,metric
                    if seed in series[key]:
                        raise ValueError(f"Repeated result for seed {seed}, {key}.")
                    series[key][seed] = value

                bands(data,batch,data.get("base_grid",""),add)
            completed[batch].append(seed)
    if require_complete and any(len(completed[b]) != len(SEEDS) for b in PROTOCOLS):
        raise ValueError(f"Five-seed aggregation is incomplete: {dict(completed)}")
    rows = []
    for key, values in sorted(series.items()):
        finite = [v for v in values.values() if isinstance(v, (int,float)) and math.isfinite(v)]
        infinite = sum(v == "inf" for v in values.values())
        mean = "inf" if infinite else statistics.mean(finite) if finite else ""
        row = dict(zip(KEYS,key)) | {
            "n_seeds":len(values), "expected_seeds":len(SEEDS), "n_finite_seeds":len(finite),
            "n_infinite_seeds":infinite, "mean":mean,
            "std":statistics.stdev(finite) if not infinite and len(finite)>1 else "",
        }
        row.update({f"seed_{seed}":value for seed,value in values.items()})
        rows.append(row)
    return rows, {batch:completed[batch] for batch in PROTOCOLS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--downloads", type=Path, default=Path("results/phase2_kaggle_private/downloads"))
    parser.add_argument("--out", type=Path, default=Path("results/phase2d_summary.csv"))
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    rows, completed = summarize(args.downloads,args.require_complete)
    if not rows:
        raise ValueError("No complete Phase 2D returns were available locally.")
    columns = (*KEYS,"n_seeds","expected_seeds","n_finite_seeds","n_infinite_seeds","mean","std",
               *(f"seed_{seed}" for seed in SEEDS))
    with args.out.open("w",encoding="utf-8-sig",newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} metric series. Verified seed sets: {completed}")


if __name__ == "__main__":
    main()
