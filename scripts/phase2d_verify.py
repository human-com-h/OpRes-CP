"""Audit the frozen Phase 2D returns and recompute paired field comparisons."""

import csv
import hashlib
import json
import math
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from phase2d_summarize import PROTOCOLS, SEEDS, summarize


RESULTS = Path("results")
STAGE = RESULTS / "phase2_kaggle_private"
DOWNLOADS = STAGE / "downloads"
LEVELS = ("0.1", "0.05")
OP = "OpRes-CP"
ENSEMBLE = ("5-FNO Ensemble + CP | single center", "5-FNO Ensemble + CP | mean center")
DROPOUT = "30-draw MC-Dropout + CP"
LSCI = "LSCI v5 sampled band"
PDE = {"2da": "darcy", "2db": "burgers", "2dc": "allen_cahn", "2dh": "allen_cahn"}
GRIDS = {
    ("2da", 32): (("paired_test_32", "paired_test_64", "paired_test_128"),),
    ("2da", 64): (("paired_test_64", "paired_test_128"),),
    ("2db", 64): (("paired_64", "paired_128", "paired_256"),),
    ("2dc", 64): tuple((f"paired_64_{h}", f"paired_128_{h}") for h in (20, 40)),
}
GROUPS = {
    ("2da", 32): GRIDS["2da", 32][0],
    ("2da", 64): ("id_test_64", "ood_a_64", "ood_b_64", *GRIDS["2da", 64][0]),
    ("2db", 64): ("id_64", "ood_64", "ood_high_64", *GRIDS["2db", 64][0]),
    ("2dc", 64): ("id_64_20", "id_64_40", "ood_64_20", "ood_64_40", *sum(GRIDS["2dc", 64], ())),
}
WCP = ("Full-grid WCP", "Physical-summary WCP")


def read(path):
    return json.loads(path.read_bytes())


def check(condition, context):
    if not condition:
        raise ValueError(context)


def verify_weights(batch, seed, job):
    source = STAGE / "phase2d_frozen" / f"seed_{seed}"
    locked = read(source / "frozen_manifest.json")
    remote = read(job / f"results/phase2d_mounts/{PDE[batch]}/seed_{seed}/frozen_manifest.json")
    expected = {"darcy": {"darcy32", "darcy64"}, "burgers": {"burgers"},
                "allen_cahn": {"allen_cahn", "allen_cahn_unet"}}[PDE[batch]]
    check(remote["seed"] == locked["seed"] == seed and set(remote["weights"]) == expected, str(job))
    for name, item in remote["weights"].items():
        weight = locked["weights"][name]
        with (source / weight["file"]).open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        check(item["sha256"] == weight["sha256"] == sha, f"Frozen hash mismatch: {job}, {name}")
    return len(expected)


def verify_baseline_archives(batch, seed, job):
    if batch == "2dh":
        return 0
    root = job / f"results/phase2d/{PDE[batch]}/seed_{seed}"
    ensemble_grid = 32 if batch == "2da" else 64
    folders = [root / f"grid{ensemble_grid}/ensemble_{i}" for i in range(1, 5)]
    folders.append(root / "grid64/mc_dropout")
    names = ("best.pt", "last.pt") if batch == "2dc" else ("fno.pt",)
    for folder in folders:
        if batch == "2db":
            folder = folder / "FNO"
        if batch in ("2da", "2db"):
            history_name = "training.json" if batch == "2da" else "train_log.json"
            check(read(folder / history_name)["history"], f"Empty training history: {folder}")
        for name in names:
            path = folder / name
            check(not path.with_name(name + ".kaggle-partial").exists(), f"Partial checkpoint: {path}")
            with zipfile.ZipFile(path) as archive:
                check(archive.testzip() is None, f"Damaged checkpoint: {path}")
    return len(folders) * len(names)


def metrics_from_fields(raw):
    finite = ~raw["infinite"]
    return {**{k: float(raw[k].mean()) for k in ("coverage", "pixel", "width", "top_width", "top_coverage")},
            "point_rel_l2": float(raw["point_errors"].mean()),
            "finite_n": int(finite.sum()), "infinite_rate": float((~finite).mean()),
            "finite_field_coverage": float(raw["coverage"][finite].mean()) if finite.any() else None,
            "finite_width": float(raw["width"][finite].mean()) if finite.any() else None}


def audit_fields(path, batch, grid, data, job):
    expected = GROUPS[batch, grid]
    check(set(data["results"]) == set(expected), f"Unexpected targets: {path}")
    if grid == 64:
        check(data["dropout_draws"] == 30, f"Unexpected dropout count: {path}")
    old_fields = None
    if (batch, grid) == ("2da", 64):
        old_fields = DOWNLOADS / f"2a/{job.name}/results/phase2_darcy/{job.name}/grid64/FNO/field_metrics.npz"
    fields, ids = {}, {}
    for group in expected:
        methods = data["results"][group]
        names = {OP, *ENSEMBLE}
        if grid == 64:
            names.update((DROPOUT, LSCI))
        if group.startswith("ood"):
            names.update(WCP)
            if batch == "2db":
                names.add("Physical-summary + spectrum WCP")
        check(set(methods) == names, f"Unexpected method set: {path}, {group}")
        with np.load(path.parent / "fields" / f"{group}.npz") as src:
            ids[group] = src["id"]
            audit = data["cal_test_audit"][group]
            check(np.array_equal(ids[group], np.arange(audit["first_id"], audit["last_id"] + 1))
                  and len(ids[group]) == 500, f"Unexpected target IDs: {path}, {group}")
            for method, levels in methods.items():
                check(set(levels) == set(LEVELS), f"Missing level: {path}, {group}, {method}")
                for alpha, row in levels.items():
                    prefix = f"{method}__{alpha}__"
                    if old_fields and method in ENSEMBLE:
                        ancestor = f"{group}__{method}"
                        old_prefix = f"{ancestor}__field__{alpha}__"
                        with np.load(old_fields) as parent:
                            raw = {k.removeprefix(old_prefix): parent[k] for k in parent.files
                                   if k.startswith(old_prefix)}
                            raw["point_errors"] = parent[f"{ancestor}__point_rel_l2"]
                    else:
                        raw = {k.removeprefix(prefix): src[k] for k in src.files if k.startswith(prefix)}
                    where = f"{path}, {group}, {method}, alpha={alpha}"
                    check(raw and all(x.shape == (500,) for x in raw.values()), f"Bad field arrays: {where}")
                    values = metrics_from_fields(raw)
                    if "matched_width" in row:
                        rank = math.ceil((1 - float(alpha)) * 500) - 1
                        values["matched_width"] = 2 * float(np.partition(raw["score"], rank)[rank]) * float(raw["scale_mean"].mean())
                    if method == LSCI:
                        values["native_slice_set_coverage"] = float(raw["native_hit"].mean())
                    for metric, actual in values.items():
                        reported = row[metric]
                        good = reported == actual if actual is None else math.isclose(
                            math.inf if reported == "inf" else reported, actual, rel_tol=1e-6, abs_tol=1e-9)
                        check(good, f"{metric} disagrees with per-field arrays: {where}")
                    fields[group, method, alpha] = raw
    for shared in GRIDS[batch, grid]:
        check(all(np.array_equal(ids[shared[0]], ids[group]) for group in shared[1:]),
              f"Cross-grid ID mismatch: {path}, {shared}")
    if batch == "2dc":
        check(all(np.array_equal(ids[g], ids[g[:-2] + "40"]) for g in ids if g.endswith("20")),
              f"Cross-horizon ID mismatch: {path}")
    return fields


def add_comparisons(series, batch, grid, seed, fields, indices):
    def compare(comparison, left_key, right_key, report_width):
        left, right = fields[left_key], fields[right_key]
        delta = left["coverage"].astype(float) - right["coverage"]
        key = (batch, grid, comparison, left_key[0], right_key[0], left_key[1], right_key[1], left_key[2])
        series[(*key, "coverage_difference")][seed] = (float(delta.mean()), delta[indices].mean(axis=1))
        if report_width == "actual":
            a, b = left["width"], right["width"]
            series[(*key, "actual_width_ratio")][seed] = (float(a.mean() / b.mean()),
                a[indices].mean(axis=1) / b[indices].mean(axis=1))
        if report_width == "matched":
            rank = math.ceil((1 - float(left_key[2])) * 500) - 1

            def matched(row):
                a, b = row["score"], row["scale_mean"]
                return (np.partition(a, rank)[rank] * b.mean(),
                        np.partition(a[indices], rank, axis=1)[:, rank] * b[indices].mean(axis=1))

            a, ab = matched(left)
            b, bb = matched(right)
            series[(*key, "matched_width_ratio")][seed] = (float(a / b), ab / bb)

    for group, method, alpha in fields:
        if method in (*ENSEMBLE, DROPOUT):
            compare("methods", (group, OP, alpha), (group, method, alpha),
                    "matched" if group.startswith("id") else None)
    for core, *fine in GRIDS[batch, grid]:
        for group in fine:
            for _, method, alpha in (k for k in fields if k[0] == core and k[1] != LSCI):
                compare("grid", (group, method, alpha), (core, method, alpha), "actual")


def audit_horizon(path, job):
    data = read(path)
    check(set(data["sha256"]) == {"cal", "id", "paired", "ood"}
          and len(set(data["sha256"].values())) == 4, f"Unexpected independent streams: {path}")
    for model, groups in data["results"].items():
        check(set(groups) == {"id_64", "paired_64", "paired_128", "ood_64"}, str(path))
        for group, variants in groups.items():
            check(set(variants) == {"jacobian_on", "jacobian_off"}, str(path))
            for policies in variants.values():
                check(set(policies) == {"T1", "T1_Bonferroni", "T2_same_window", "separate_block_Bonferroni"}, str(path))
                check(all(set(policies[k]) == set(LEVELS) for k in ("T1", "T1_Bonferroni", "T2_same_window")), str(path))
            on, off = variants["jacobian_on"], variants["jacobian_off"]
            for policy in ("T1", "T1_Bonferroni", "T2_same_window"):
                for alpha in LEVELS:
                    delta = on[policy][alpha]["coverage"] - off[policy][alpha]["coverage"]
                    check(math.isclose(delta, on[policy][alpha]["paired_coverage_diff_vs_off"], abs_tol=1e-12),
                          f"Jacobian comparison mismatch: {path}, {model}, {group}, {policy}")
    check(set(data["results"]) == {"FNO", "U-Net"}, str(path))
    log = (job / f"opres-cp-phase-2dh-allen-cahn-horizon-{job.name.replace('_', '-')}.log").read_text(encoding="utf-8")
    check(all(f"seed {data['seed']} {model} {group}" in log for model, groups in data["results"].items()
              for group in groups), f"Incomplete long-window worker log: {job}")


def export_paired(series):
    keys = ("batch", "base_grid", "comparison", "left_group", "right_group", "left_method",
            "right_method", "alpha", "metric")
    with (RESULTS / "phase2d_paired.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        out = csv.writer(stream)
        out.writerow((*keys, "mean", "std", "field_bootstrap_low", "field_bootstrap_high", *(f"seed_{s}" for s in SEEDS)))
        for key, seeds in sorted(series.items()):
            check(set(seeds) == set(SEEDS), f"Missing paired comparison seeds: {key}")
            values, boot = zip(*(seeds[s] for s in SEEDS))
            ci = np.quantile(np.mean(boot, axis=0), (.025, .975))
            out.writerow((*key, statistics.mean(values), statistics.stdev(values), *ci, *values))


def summarize_mixed_gates():
    metrics = defaultdict(dict)
    for batch in ("2a", "2b"):
        for seed in SEEDS:
            folder = f"seed_{seed}"
            pde = PDE[f"2d{batch[-1]}"]
            data = read(DOWNLOADS / batch / folder / f"results/phase2_{pde}/{folder}/metrics.json")
            node = data["tracks"]["64"]["FNO"] if batch == "2a" else data["results"]["FNO"]["0.1"]
            for shift, modes in node["gates"].items():
                for mode, gate in modes.items():
                    for key in ("ood_auroc", "fpr_at_95_tpr", "point_aurc", "ungated_point_risk"):
                        metrics[batch, shift, mode, "", key][seed] = gate[key]
                    for anchor, row in gate["anchors"].items():
                        for key in ("n", "accepted_n", "finite_n", "abstain", "id_pass", "ood_recall", "coverage", "width", "point_risk"):
                            metrics[batch, shift, mode, anchor, key][seed] = row[key]
    for seed in SEEDS:
        location = PROTOCOLS["2db"][0].format(seed=seed)
        data = read(DOWNLOADS / f"2db/seed_{seed}/results/{location}/high_frequency_diagnosis.json")
        for mode, gate in data["gate"].items():
            metrics["2db", "ood_high", mode, "", "ood_auroc"][seed] = gate["auroc"]
            for anchor, row in gate["anchors"].items():
                normalized = {"accepted_n": row["n_accepted"], "n": 1000,
                              "abstain": 1 - row["n_accepted"] / 1000,
                              "id_pass": row["id_accept"], "ood_recall": row["ood_reject"],
                              "coverage": row["accepted_coverage"], "width": row["accepted_width"]}
                for key, value in normalized.items():
                    metrics["2db", "ood_high", mode, anchor, key][seed] = value
    with (RESULTS / "phase2d_mixed_gates.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        out = csv.writer(stream)
        out.writerow(("batch", "shift", "features", "p_test_max", "metric", "n_defined", "mean", "std",
                      *(f"seed_{s}" for s in SEEDS)))
        for key, seeds in sorted(metrics.items()):
            check(set(seeds) == set(SEEDS), f"Incomplete mixed-flow series: {key}")
            values = [v for v in seeds.values() if v is not None]
            out.writerow((*key, len(values), statistics.mean(values) if values else None,
                          statistics.stdev(values) if len(values) > 1 else None, *(seeds[s] for s in SEEDS)))
    return len(metrics)


def export_latex(rows):
    metrics = {(r["batch"], r["grid"], r["group"], r["method"], r["alpha"], r["metric"]): r for r in rows}
    targets = (("2da", "id_test_64", "Darcy"), ("2db", "id_64", "Burgers"),
               ("2dc", "id_64_20", "Allen--Cahn $T=1$"), ("2dc", "id_64_40", "Allen--Cahn $T=2$"))
    lines = [r"% Five-seed mean and sample SD. T=2 here reuses the T=1 calibration.",
             r"\begin{tabular}{llrrrr}", r"\hline",
             r"ID target & Method & Cov. at 90\% & Matched width & Cov. at 95\% & Matched width \\", r"\hline"]
    for batch, target, label in targets:
        for method, display in ((OP, OP), (ENSEMBLE[0], "Ensemble (single)"),
                                (ENSEMBLE[1], "Ensemble (mean)"), (DROPOUT, "MC-Dropout")):
            cells = []
            for alpha in LEVELS:
                cov = metrics[batch, "64", target, method, alpha, "coverage"]
                width = metrics[batch, "64", target, method, alpha, "matched_width"]
                cells.extend((rf"${100*cov['mean']:.2f} \pm {100*cov['std']:.2f}$",
                              rf"${width['mean']:.5f} \pm {width['std']:.5f}$"))
            lines.append(" & ".join((label, display, *cells)) + r" \\")
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    (RESULTS / "phase2_benchmark_main.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    rows, _ = summarize(DOWNLOADS, require_complete=True)
    plan = read(Path("configs/phase2_darcy.json"))
    rng = np.random.default_rng(plan["bootstrap_seed"])
    indices = {seed: rng.integers(0, 500, (plan["bootstrap_resamples"], 500)) for seed in SEEDS}
    series, audit = defaultdict(dict), {}
    for batch, protocols in PROTOCOLS.items():
        summary = dict(weight_hash_checks=0, data_hash_matches=0, field_groups=0, replay_rows=0,
                       baseline_checkpoint_archives=0, max_abs_coverage_delta=0., max_abs_width_ratio_error=0.)
        for seed in SEEDS:
            job = DOWNLOADS / batch / f"seed_{seed}"
            summary["weight_hash_checks"] += verify_weights(batch, seed, job)
            summary["baseline_checkpoint_archives"] += verify_baseline_archives(batch, seed, job)
            for protocol in protocols:
                path = job / "results" / protocol.format(seed=seed) / "metrics.json"
                if batch == "2dh":
                    audit_horizon(path, job)
                    continue
                data = read(path)
                grid = data["base_grid"]
                fields = audit_fields(path, batch, grid, data, job)
                summary["field_groups"] += len(fields)
                hashes = data["frozen_matches"]
                check(hashes and all(hashes.values()), f"Frozen data mismatch: {path}")
                summary["data_hash_matches"] += len(hashes)
                replay = [row for group in data["frozen_opres_replay"].values() for row in group.values()]
                check(len(replay) == len(GROUPS[batch, grid]) * len(LEVELS), f"Missing frozen replay: {path}")
                summary["replay_rows"] += len(replay)
                summary["max_abs_coverage_delta"] = max(summary["max_abs_coverage_delta"], *(abs(r["coverage_delta"]) for r in replay))
                summary["max_abs_width_ratio_error"] = max(summary["max_abs_width_ratio_error"], *(abs(r["width_ratio"] - 1) for r in replay))
                add_comparisons(series, batch, grid, seed, fields, indices[seed])
        audit[batch] = summary
    export_paired(series)
    export_latex(rows)
    gate_series = summarize_mixed_gates()
    (RESULTS / "phase2d_acceptance.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"{len(series)} paired comparisons; {gate_series} mixed-flow series. {audit}")


if __name__ == "__main__":
    main()
