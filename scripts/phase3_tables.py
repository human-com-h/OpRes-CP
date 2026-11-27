"""Generate publication-grade booktabs LaTeX tables for the paper."""

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from phase3_summarize import (DROPOUT, ENSEMBLE, GRID_STUDIES, OP, OUT, PHASE2_OP, SEEDS, SINGLE_ENSEMBLE,
                              grid_gain, horizon, load, phase2_metrics, stat)


REV = Path("results/phase3_revision")
REV_TABLES = REV / "tables"
PAPER_TABLES = Path("paper/tables") if Path("paper").is_dir() else Path("outputs/tables")
REPLAY_TABLES = Path("results/manuscript_replay/tables") if Path("paper").is_dir() else Path("outputs/tables")

TARGETS = (("darcy", "id_test_64", "Darcy"), ("burgers", "id_64", "Burgers"),
           ("allen_cahn", "id_64_20", "Allen--Cahn $T=1$"))
METHODS = ((PHASE2_OP, "OpRes-CP"), (SINGLE_ENSEMBLE, "Ensemble (same center)"),
           (ENSEMBLE, "Ensemble (mean)"), (DROPOUT, "MC-Dropout"))


def mean_sd(values, digits, multiplier=1):
    _, avg, sd, _, _ = stat(values)
    return rf"${multiplier*avg:.{digits}f} \pm {multiplier*sd:.{digits}f}$"


def signed(val, digits=2):
    return rf"${val:+.{digits}f}$"


def table(header, rows, alignment, caption, label, size=r"\small", tabcolsep="4pt", notes=None):
    columns = len(header)
    for row in rows:
        if row is not None and len(row) != columns:
            raise ValueError(f"Incorrect LaTeX columns for {label}: {row}")
    if len(alignment) != columns:
        raise ValueError(f"Incorrect alignment length for {label}")
    lines = [r"\begin{table*}[t]", r"\centering", size, rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}", rf"\begin{{tabular}}{{{alignment}}}",
             r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for row in rows:
        if row is None:
            lines.append(r"\addlinespace[3pt]")
        else:
            lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if notes:
        lines.append(rf"\par\smallskip\raggedright\footnotesize {notes}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def _read_p2d_csv():
    out = {}
    with Path("results/phase2d_summary.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            out[(r["batch"], r["grid"], r["group"], r["method"], r["alpha"], r["metric"])] = r
    return out


def _read_phase2_csv(path, grid=64):
    groups = defaultdict(list)
    with Path(path).open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if grid is not None and int(r["training_grid"]) != grid:
                continue
            groups[(r["model"], r["split"], r["method"], float(r["alpha"]))].append(r)
    return groups


def primary():
    bench = {pde: {s: phase2_metrics(pde, s) for s in SEEDS} for pde, *_ in TARGETS}
    rows = []
    for idx, (pde, split, name) in enumerate(TARGETS):
        if idx > 0:
            rows.append(None)
        for method, label in METHODS:
            records = [bench[pde][s]["results"][split][method] for s in SEEDS]
            columns = tuple(mean_sd([x[a][k] for x in records], 4 if k == "matched_width" else 2,
                            100 if k == "coverage" else 1) for a in ("0.1", "0.05") for k in ("coverage", "matched_width"))
            rows.append((name, label, *columns))
    blocks = [table(("Benchmark", "Method", r"Cov.\ 90\%", r"Width at 90\%", r"Cov.\ 95\%", r"Width at 95\%"), rows, "llrrrr",
             r"ID simultaneous coverage (\%) and coverage-matched full width. Mean $\pm$ SD over five seeds with 500 cases each. Width is matched empirically.", "tab:opres-id")]

    rows = []
    for pde, split, name in TARGETS:
        latency, train = [], []
        for seed in SEEDS:
            node = bench[pde][seed]
            which = "darcy64" if pde == "darcy" else pde
            old = node["phase2a_ensemble"]
            if pde == "darcy":
                a = old["ensemble_timing_export"]["64"]
                pair = [a[k]["p50_ms"] for k in ("single_plus_opres", "ensemble_total")]
            else:
                pair = [node["latency"][split][k]["p50_ms"] for k in ("OpRes end-to-end", "ensemble")]
            latency.append([*pair, node["latency"][split]["mc_dropout"]["p50_ms"]])
            ens = node["ensemble_total_training_seconds"]
            train.append([node["training_seconds"][which], old["ensemble_total_training_seconds"] if ens is None else ens,
                          node["training_seconds"]["mc_dropout"]])
        rows.append((name, *(f"{value:.2f}" for value in (*np.mean(latency, axis=0), *np.mean(train, axis=0)))))
    blocks.append(table(("Benchmark", "OpRes ms", "Ens.\\ ms", "MC ms", "Single s", "Ens.\\ s", "MC s"), rows, "lrrrrrr",
                  r"Warm T4 single-case median wall time and training time. Means over five seeds. OpRes adds no backbone training.", "tab:opres-runtime"))

    rows = []
    for pde, g, label in (("darcy", "ood_a_64", "Darcy contrast"), ("darcy", "ood_b_64", "Darcy structure"),
           ("burgers", "ood_64", "Burgers viscosity"), ("burgers", "ood_high_64", "Burgers high frequency"),
           ("allen_cahn", "ood_64_20", "Allen--Cahn reaction $T=1$"), ("allen_cahn", "ood_64_40", "Allen--Cahn reaction $T=2$")):
        entries = []
        for method in (PHASE2_OP, ENSEMBLE, DROPOUT):
            a = [bench[pde][s]["results"][g][method]["0.1"] for s in SEEDS]
            entries.append(f"{100*np.mean([x['coverage'] for x in a]):.2f} / {np.mean([x['matched_width'] for x in a]):.4f}")
        rows.append((label, *entries))
    blocks.append(table(("Target", "OpRes", "Ensemble mean", "MC-Dropout"), rows, "lrrr",
                  r"OOD coverage (\%) at the locked source threshold and width empirically matched to $90\%$ coverage, with 500 target cases per seed.", "tab:opres-ood"))

    rows = []
    gains = {(r["track"], r["comparator"]): r for r in grid_gain() if r["alpha"] == "0.1"}
    for pde, grid, source, dest, label in GRID_STUDIES:
        nodes = [phase2_metrics(pde, s, grid)["results"] for s in SEEDS]
        widths = [node[dest][PHASE2_OP]["0.1"]["width"] / node[source][PHASE2_OP]["0.1"]["width"] for node in nodes]
        diffs = [tuple(node[dest][m]["0.1"]["coverage"] - node[source][m]["0.1"]["coverage"] for node in nodes) if m in nodes[0][source] else () for m, _ in METHODS]
        g = gains[label, ENSEMBLE]
        tex_label = label.replace("²", "$^2$").replace(" → ", r" $\to$ ").replace("–", "--")
        rows.append((tex_label, *(signed(100 * np.mean(v)) if v else "--" for v in diffs),
                     f"{np.mean(widths):.3f}",
                     rf"${g['mean_gain_pp']:+.2f}\;[{g['boot_low_pp']:+.2f},\,{g['boot_high_pp']:+.2f}]$"))
    blocks.append(table(("Transfer", "OpRes", "Ens.\\ same", "Ens.\\ mean", "Dropout", "Width ratio", r"Gain vs.\ mean [95\% CI]"), rows, "lrrrrrr",
                  r"Seven paired grid transfers at $\alpha=0.10$. Coverage change and gains are in percentage points. The gain interval resamples paired fields within each of the five fixed seeds.", "tab:opres-grid",
                  size=r"\footnotesize", tabcolsep="3.5pt"))

    rows = []
    for midx, mod in enumerate(("FNO", "U-Net")):
        if midx > 0:
            rows.append(None)
        for group, label in (("id_64", r"ID $64^2$"), ("paired_128", r"Paired $128^2$"), ("ood_64", "Reaction OOD")):
            records = [horizon("allen_cahn", s)["results"][mod][group] for s in SEEDS]
            row = []
            for switch, policy in (("jacobian_on", "T1"), ("jacobian_on", "T1_Bonferroni"), ("jacobian_on", "T2_same_window"), ("jacobian_off", "T2_same_window")):
                rec = [d[switch][policy]["0.1"] for d in records]
                row.append(f"{100*np.mean([x['coverage'] for x in rec]):.2f} / {np.mean([x['width'] for x in rec]):.4f}")
            rows.append((mod, label, *row))
    blocks.append(table(("Backbone", "Target", r"T1 90\%", r"T1 95\%", r"T2 90\%", r"T2 90\%, off"), rows, "llrrrr",
                  r"Independent Allen--Cahn $T=2$ coverage (\%) / actual full width. The left three policies enable the same reaction Jacobian.", "tab:opres-horizon"))
    return blocks


def protocol_metrics(p2d_csv):
    """Auto-generate paper/tables/protocol_metrics.tex from sealed Phase 2D CSV with booktabs."""
    rows_pw = []
    for batch, grp, label in (("2da", "id_test_64", "Darcy"), ("2db", "id_64", "Burgers"), ("2dc", "id_64_20", "Allen--Cahn $T=1$")):
        def cell(metric, digits, mul=1.0):
            r = p2d_csv[(batch, "64", grp, PHASE2_OP, "0.1", metric)]
            return rf"${mul*float(r['mean']):.{digits}f} \pm {mul*float(r['std']):.{digits}f}$"
        rows_pw.append((label, cell("width", 5), cell("pixel", 4, 100), cell("top_coverage", 4, 100), cell("top_width", 5)))
    t1 = table(("Benchmark", "Full width", r"Point cov.\ (\%)", r"High-grad.\ cov.\ (\%)", "High-grad.\\ width"), rows_pw, "lrrrr",
               r"ID auxiliary metrics under the locked $90\%$ whole-field threshold. Entries are five-seed mean $\pm$ SD.", "tab:paper-pointwise")

    wcp_specs = (
        ("Darcy contrast", "Full", "2da", "ood_a_64", "Full-grid WCP"),
        ("Darcy contrast", "Summary", "2da", "ood_a_64", "Physical-summary WCP"),
        ("Darcy structure", "Full", "2da", "ood_b_64", "Full-grid WCP"),
        ("Darcy structure", "Summary", "2da", "ood_b_64", "Physical-summary WCP"),
        ("Burgers viscosity", "Full", "2db", "ood_64", "Full-grid WCP"),
        ("Burgers viscosity", "Spectrum", "2db", "ood_64", "Physical-summary + spectrum WCP"),
        ("Burgers viscosity", "Summary", "2db", "ood_64", "Physical-summary WCP"),
        ("Burgers high", "Full", "2db", "ood_high_64", "Full-grid WCP"),
        ("Burgers high", "Spectrum", "2db", "ood_high_64", "Physical-summary + spectrum WCP"),
        ("Burgers high", "Summary", "2db", "ood_high_64", "Physical-summary WCP"),
        ("Allen--Cahn $T=1$", "Full", "2dc", "ood_64_20", "Full-grid WCP"),
        ("Allen--Cahn $T=1$", "Summary", "2dc", "ood_64_20", "Physical-summary WCP"),
        ("Allen--Cahn $T=2$", "Full", "2dc", "ood_64_40", "Full-grid WCP"),
        ("Allen--Cahn $T=2$", "Summary", "2dc", "ood_64_40", "Physical-summary WCP"),
    )
    rows_wcp = []
    canonical_wcp = {}
    for target, wlabel, batch, grp, method in wcp_specs:
        def val(metric):
            s = p2d_csv[(batch, "64", grp, method, "0.1", metric)]["mean"]
            return None if s == "" else float(s)
        cov = 100 * val("coverage")
        inf = 100 * val("infinite_rate")
        fn = val("finite_n")
        fc = val("finite_field_coverage")
        fw = val("finite_width")
        ess = val("ess")
        canonical_wcp[(batch, grp, method)] = {
            "cov": cov, "inf": inf, "finite_n": fn,
            "finite_cov": None if fc is None else 100 * fc,
            "finite_width": fw, "ess": ess,
        }
        rows_wcp.append((
            target, wlabel, f"{cov:.2f}", f"{inf:.2f}", f"{fn:.1f}",
            f"{100*fc:.2f}" if fc is not None else "--",
            f"{fw:.5f}" if fw is not None else "--",
            f"{ess:.2f}",
        ))
    t2 = table(("Target", "Weight input", "Cov.", "Inf.", "Finite $n$", "Finite cov.", "Finite width", "ESS"), rows_wcp, "llrrrrrr",
               r"Formal ungated WCP at the nominal $90\%$ target. Coverage and infinite rates are percentages. All 500 cases per seed enter the unconditional rates and abstention is zero. Finite coverage and width use the finite subset. Entries are five-seed means.",
               "tab:paper-wcp-formal")
    return [t1, t2], canonical_wcp


def ablations(data, canonical_wcp, darcy_rev):
    rows = []
    step_keys = ("absolute residual", "inverse after absolute", "signed inverse", OP)
    for pidx, (pde, track, id_group, ood_group, title) in enumerate((("darcy", "grid64", "id_test_64", "ood_a_64", "Darcy"),
            ("burgers", None, "id", "ood_high", "Burgers"), ("allen_cahn", None, "short_id", "short_ood", "Allen--Cahn"))):
        if pidx > 0:
            rows.append(None)
        for method, label in zip(step_keys, (r"$|R|$", "Inverse after absolute", "Signed inverse", "Signed + envelope")):
            nodes = [(d[track] if track else d)["metrics"] for d in data[pde].values()]
            cells = tuple(mean_sd([d[g][method]["0.1"][k] for d in nodes], 4 if k == "matched_width" else 2,
                         100 if k == "coverage" else 1) for g in (id_group, ood_group) for k in ("coverage", "matched_width"))
            rows.append((title, label, *cells))
    blocks = [table(("PDE", "Frozen scale", "ID cov.", "ID width", "OOD cov.", "OOD width"), rows, "llrrrr",
                    r"Matched $90\%$ width and actual coverage (\%) across five seeds. OOD columns use Darcy contrast, Burgers high frequency, and Allen--Cahn reaction shift.", "tab:opres-mechanism")]

    rows = []
    for ridx, radius in enumerate((.04, .08, .16, .32)):
        if ridx > 0:
            rows.append(None)
        for lam in (0, .25, .5, .7, .85, 1):
            key = f"lambda={lam:g}, radius={radius:g}"
            r = [d["metrics"] for d in data["burgers"].values()]
            rows.append((f"{radius:g}", f"{lam:g}", mean_sd([x["id"][key]["0.1"]["matched_width"] for x in r], 4),
                mean_sd([x["ood_high"][key]["0.1"]["coverage"] for x in r], 2, 100)))
    blocks.append(table((r"$\ell_{\rm phys}$", r"$\lambda$", "ID matched width", r"High OOD coverage (\%)"), rows, "rrrr",
                       r"Burgers envelope radius and localization sweep. Mean $\pm$ SD over five frozen seeds.", "tab:opres-burgers-sweep"))

    rows = []
    for key, label in (("spectral energy", "Spectral energy"), ("mix logistic", "Input logistic ratio")):
        for a in ("0.01", "0.02"):
            records = [d["gates"][key]["operations"][a] for d in data["burgers"].values()]
            rates = [f"{100*np.mean([r[k] for r in records]):.2f}" for k in ("id_reject", "ood_reject", "accepted_coverage")]
            rows.append((label, f"{100*float(a):.0f}", rates[0], rates[1], f"{np.mean([v['n_accepted'] for v in records]):.1f}", rates[2]))
    blocks.append(table(("Burgers gate", "Cal target", "ID reject", "High OOD reject", "Accepted $n$", "Accepted cov."), rows, "lrrrrr",
                    r"Cal-quantile gate operating points. Rates are in percent, with 500 ID and 500 high-frequency cases per seed.", "tab:opres-gates"))

    rows = []
    for pidx, (pde, track, group, label) in enumerate((("darcy", "grid64", "id_test_64", "Darcy"), ("burgers", None, "id", "Burgers"),
                                  ("allen_cahn", None, "short_id", "Allen--Cahn"))):
        if pidx > 0:
            rows.append(None)
        nodes = [d[track] if track else d for d in data[pde].values()]
        for n in (100, 200, 400, 500):
            r = [d["calibration_size"][str(n)][group]["0.1"] for d in nodes]
            rows.append((label, str(n), mean_sd([x["coverage"] for x in r], 2, 100), mean_sd([x["width"] for x in r], 4)))
    blocks.append(table(("Benchmark", r"$n_{\rm cal}$", r"ID coverage (\%)", "Actual width"), rows, "lrrr",
                       r"Sensitivity to nested calibration prefixes, five seeds and 500 fixed ID cases per seed.", "tab:opres-calibration-size"))

    # Task A1 + A2: Darcy radius + boundary trace controls with BOTH Actual width and Matched width (5 decimals),
    # and 3 boundary arms: Interior lift only, O(M log M) spectral lift, and Oracle exact harmonic lift.
    rows = []
    for n in (32, 64, 128):
        for key, label in (("fixed physical radius", "Physical radius"), ("fixed pixel radius", "Fixed 1.28 cells")):
            records = [v["grid32"]["metrics"][f"paired_test_{n}"][key]["0.1"] for v in data["darcy"].values()]
            rows.append((f"${n}^2$", label,
                         mean_sd([v["coverage"] for v in records], 2, 100),
                         mean_sd([v["width"] for v in records], 5),
                         mean_sd([v["matched_width"] for v in records], 5)))
    rows.append(None)
    for arm, label in (
        ("interior", "Interior lift only"),
        ("spectral_liouville", r"Add $O(M\log M)$ spectral lift"),
        ("exact_oracle", "Add exact harmonic lift (oracle)"),
    ):
        recs = [darcy_rev["per_seed"][str(s)]["metrics"][arm]["0.1"] for s in SEEDS]
        rows.append((r"$64^2$ trace stress", label,
                     mean_sd([v["coverage"] for v in recs], 2, 100),
                     mean_sd([v["width"] for v in recs], 5),
                     mean_sd([v["matched_width"] for v in recs], 5)))
    blocks.append(table(("Darcy target", "Scale control", r"Coverage (\%)", "Actual width", "Matched width"), rows, "llrrr",
                       r"Darcy pressure-path and boundary-trace controls, five seeds. The radius comparison uses the same frozen center and calibration threshold. For the $64^2$ trace stress, the $O(M\log M)$ spectral boundary lift applies the same Liouville inverse to the boundary forcing, while the exact discrete harmonic lift uses a sparse solver as an oracle upper bound.",
                       "tab:opres-darcy-ablations", size=r"\footnotesize", tabcolsep="3.5pt"))

    rows = []
    for pidx, (pde, group, label) in enumerate((("burgers", "id", "Burgers"), ("allen_cahn", "short_id", "Allen--Cahn"))):
        if pidx > 0:
            rows.append(None)
        for n in (1, 2, 4):
            coarse, dense = (f"stride={n}", f"stride={n} reference evaluation grid" if n > 1 else "stride=1")
            info = [d["metrics"][group] for d in data[pde].values()]
            rows.append((label, rf"${n}\Delta t$", mean_sd([v[coarse]["0.1"]["matched_width"] for v in info], 4),
                         mean_sd([v[dense]["0.1"]["matched_width"] for v in info], 4)))
    blocks.append(table(("Benchmark", "Cadence", "Thin lift + score", "Thin score only"), rows, "llrr",
                       r"Matched $90\%$ full width. The score-only control retains the dense lift and reduces the evaluation grid.", "tab:opres-time"))

    rows = []
    for grid in ("64", "128"):
        values = [v["manifold"][grid] for v in data["allen_cahn"].values()]
        cells = (mean_sd([x["point_rel_l2"] for x in values], 2, 100), mean_sd([x["jacobian_sign_mismatch"] for x in values], 2, 100),
               *(mean_sd([x["matched_width"][m]["0.1"] for x in values], 4) for m in ("predicted_jacobian", "jacobian_off", "target_jacobian_diagnostic")))
        rows.append((f"${grid}^2$", *cells))
    blocks.append(table(("U-Net grid", r"Point $L^2$ (\%)", r"Sign mismatch (\%)", "Predicted $J$", "$J$ off", "Target $J$"), rows, "lrrrrr",
                       r"Paired independent $T=2$ trajectories. Jacobian columns are matched $90\%$ full widths, with the same forecast residual. Target $J$ is an oracle diagnosis.", "tab:opres-jacobian"))

    # Task A2: Unify canonical C=1, cap=50 rows with sealed Phase 2D formal benchmark (float32 full-grid inputs -> ESS 121.43).
    rows = []
    selections = (
        ("Darcy OOD-B", data["darcy"], "grid64", "ood_b_64", "input", "C=1, cap=50", ("2da", "ood_b_64", "Physical-summary WCP")),
        ("Darcy OOD-B", data["darcy"], "grid64", "ood_b_64", "full", "C=1, cap=50", ("2da", "ood_b_64", "Full-grid WCP")),
        ("Darcy OOD-B", data["darcy"], "grid64", "ood_b_64", "full", "C=1, cap=10", None),
        ("Burgers high", data["burgers"], None, "ood_high", "spectrum", "C=1, cap=50", ("2db", "ood_high_64", "Physical-summary + spectrum WCP")),
        ("Allen--Cahn OOD", data["allen_cahn"], None, "short_ood", "input", "C=1, cap=50", ("2dc", "ood_64_20", "Physical-summary WCP")),
    )
    for label, nodes, track, group, features, setting, canon_key in selections:
        values = [(v[track] if track else v)["wcp"][group][features][setting]["0.1"] for v in nodes.values()]
        pt_med = float(np.mean([v["p_test_p50"] for v in values]))
        if canon_key is not None:
            c = canonical_wcp[canon_key]
            cov_s, inf_s, fn_s = f"{c['cov']:.2f}", f"{c['inf']:.2f}", f"{c['finite_n']:.1f}"
            fc_s = f"{c['finite_cov']:.2f}" if c["finite_cov"] is not None else "--"
            ess_s = f"{c['ess']:.2f}"
        else:
            finite = [v["finite_coverage"] for v in values if v["finite_coverage"] is not None]
            cov_s = f"{100*np.mean([v['coverage'] for v in values]):.2f}"
            inf_s = f"{100*np.mean([v['infinite_rate'] for v in values]):.2f}"
            fn_s = f"{np.mean([v['finite_n'] for v in values]):.1f}"
            fc_s = f"{100*np.mean(finite):.2f}" if finite else "--"
            ess_s = f"{np.mean([v['ess'] for v in values]):.2f}"
        rows.append((label, features + " / " + setting, cov_s, inf_s, fn_s, fc_s, ess_s, f"{pt_med:.2f}"))
    blocks.append(table(("Shift", "Summary / clipping", "Cov.", "Inf.", "Finite $n$", "Finite cov.", "ESS", r"Med.\ $p_t$"), rows, "llrrrrrr",
                       r"WCP at the $90\%$ target. Coverage and infinite rate use percent units, with 500 target cases per seed. Canonical $C=1$, $\mathrm{cap}=50$ entries use the same float32 feature representation as Table~\ref{tab:paper-wcp-formal}; undefined finite coverage uses a dash.",
                       "tab:opres-wcp"))
    return blocks


def revision_tables(p2d_csv, canonical_wcp, lsci_rev, wcp_split):
    """Tasks A3, B1, B2: full baselines, fixed LSCI, cross-architecture, and Split-fit vs Transductive WCP."""
    g2a = _read_phase2_csv("results/phase2_2a_summary.csv", grid=64)
    g2b = _read_phase2_csv("results/phase2_2b_summary.csv", grid=64)
    g2c = _read_phase2_csv("results/phase2_2c_summary.csv", grid=64)

    def _csv_cell(tbl, model, split, method, fmt="cov_mw", suffix=""):
        rs = tbl.get((model, split, method, 0.1))
        if not rs:
            return "--"
        cov = 100.0 * float(np.mean([float(r["coverage"]) for r in rs]))
        mw = float(np.mean([float(r["matched_field_width"]) for r in rs]))
        if fmt == "cov_mw":
            return f"{cov:.2f}{suffix} / {mw:.4f}"
        return f"{cov:.2f}{suffix}"

    def _p2d_cell(batch, grp, method, fmt="cov_mw"):
        c = p2d_csv.get((batch, "64", grp, method, "0.1", "coverage"))
        m = p2d_csv.get((batch, "64", grp, method, "0.1", "matched_width"))
        if not c:
            return "--"
        cov = 100.0 * float(c["mean"])
        if fmt == "cov_mw" and m and m["mean"] != "":
            return f"{cov:.2f} / {float(m['mean']):.4f}"
        return f"{cov:.2f}"

    def _lsci_cell(pde, grp, kind):
        node = lsci_rev["pdes"][pde]["summary"][grp]["0.1"]
        if kind == "native":
            return f"{100*node['native_slice_set_coverage']['mean']:.2f}"
        ec = 100 * node["envelope_coverage"]["mean"]
        fw = node["envelope_finite_width"]["mean"]
        if fw is None:
            return f"{ec:.2f} / --"
        return f"{ec:.2f} / {fw:.3f}"

    # 1) Full same-protocol FNO baseline comparison table (B1 + A3)
    # Compact 9-col table: Method | Darcy ID (cov/mw), OOD-A, OOD-B | Burgers ID (cov/mw), Visc, High | AC T=1 ID (cov/mw), React
    baseline_rows = [
        ("Uniform-CP",
         _csv_cell(g2a, "FNO", "id_test_64", "Uniform-CP", "cov_mw"),
         _csv_cell(g2a, "FNO", "ood_a_64", "Uniform-CP", "cov"),
         _csv_cell(g2a, "FNO", "ood_b_64", "Uniform-CP", "cov"),
         _csv_cell(g2b, "FNO", "id_64", "Uniform-CP", "cov_mw"),
         _csv_cell(g2b, "FNO", "ood_64", "Uniform-CP", "cov"),
         _csv_cell(g2b, "FNO", "ood_high_64", "Uniform-CP", "cov"),
         _csv_cell(g2c, "FNO", "id_64_t1", "Uniform-CP", "cov_mw"),
         _csv_cell(g2c, "FNO", "ood_64_t1", "Uniform-CP", "cov")),
        ("Choi CP-B",
         _csv_cell(g2a, "FNO", "id_test_64", "CP-B", "cov_mw"),
         _csv_cell(g2a, "FNO", "ood_a_64", "CP-B", "cov"),
         _csv_cell(g2a, "FNO", "ood_b_64", "CP-B", "cov"),
         _csv_cell(g2b, "FNO", "id_64", "CP-B", "cov_mw"),
         _csv_cell(g2b, "FNO", "ood_64", "CP-B", "cov"),
         _csv_cell(g2b, "FNO", "ood_high_64", "CP-B", "cov"),
         _csv_cell(g2c, "FNO", "id_64_t1", "CP-B", "cov_mw"),
         _csv_cell(g2c, "FNO", "ood_64_t1", "CP-B", "cov")),
        ("Full-Field PI-CP",
         _csv_cell(g2a, "FNO", "id_test_64", "PI-CP", "cov_mw"),
         _csv_cell(g2a, "FNO", "ood_a_64", "PI-CP", "cov"),
         _csv_cell(g2a, "FNO", "ood_b_64", "PI-CP", "cov"),
         _csv_cell(g2b, "FNO", "id_64", "Full-Field PI-CP", "cov_mw"),
         _csv_cell(g2b, "FNO", "ood_64", "Full-Field PI-CP", "cov"),
         _csv_cell(g2b, "FNO", "ood_high_64", "Full-Field PI-CP", "cov"),
         _csv_cell(g2c, "FNO", "id_64_t1", "Full-Field PI-CP", "cov_mw"),
         _csv_cell(g2c, "FNO", "ood_64_t1", "Full-Field PI-CP", "cov")),
        ("Physics-Scalar CP",
         _csv_cell(g2a, "FNO", "id_test_64", "Physics-Scalar CP", "cov_mw"),
         _csv_cell(g2a, "FNO", "ood_a_64", "Physics-Scalar CP", "cov"),
         _csv_cell(g2a, "FNO", "ood_b_64", "Physics-Scalar CP", "cov"),
         _csv_cell(g2b, "FNO", "id_64", "Physics-Scalar CP", "cov_mw"),
         _csv_cell(g2b, "FNO", "ood_64", "Physics-Scalar CP", "cov"),
         _csv_cell(g2b, "FNO", "ood_high_64", "Physics-Scalar CP", "cov"),
         "--", "--"),
        ("Defect Corr.\\ + CP",
         _csv_cell(g2a, "FNO", "id_test_64", "Defect Correction + CP", "cov_mw"),
         _csv_cell(g2a, "FNO", "ood_a_64", "Defect Correction + CP", "cov"),
         _csv_cell(g2a, "FNO", "ood_b_64", "Defect Correction + CP", "cov", suffix=r"$^*$"),
         "--", "--", "--", "--", "--"),
        ("Mabtoul-adapted CW",
         _csv_cell(g2a, "FNO", "id_test_64", "Mabtoul-adapted CW", "cov_mw"),
         _csv_cell(g2a, "FNO", "ood_a_64", "Mabtoul-adapted CW", "cov"),
         _csv_cell(g2a, "FNO", "ood_b_64", "Mabtoul-adapted CW", "cov"),
         "--", "--", "--", "--", "--"),
        None,
        (r"LSCI (native set)$^\dagger$",
         _lsci_cell("darcy", "id_test_64", "native") + " / --",
         _lsci_cell("darcy", "ood_a_64", "native"),
         _lsci_cell("darcy", "ood_b_64", "native"),
         _lsci_cell("burgers", "id_64", "native") + " / --",
         _lsci_cell("burgers", "ood_64", "native"),
         _lsci_cell("burgers", "ood_high_64", "native"),
         _lsci_cell("allen_cahn", "id_64_20", "native") + " / --",
         _lsci_cell("allen_cahn", "ood_64_20", "native")),
        (r"LSCI (64-draw env.)$^\dagger$",
         _lsci_cell("darcy", "id_test_64", "env"),
         f"{100*lsci_rev['pdes']['darcy']['summary']['ood_a_64']['0.1']['envelope_coverage']['mean']:.2f}",
         f"{100*lsci_rev['pdes']['darcy']['summary']['ood_b_64']['0.1']['envelope_coverage']['mean']:.2f}",
         _lsci_cell("burgers", "id_64", "env"),
         f"{100*lsci_rev['pdes']['burgers']['summary']['ood_64']['0.1']['envelope_coverage']['mean']:.2f}",
         f"{100*lsci_rev['pdes']['burgers']['summary']['ood_high_64']['0.1']['envelope_coverage']['mean']:.2f}",
         _lsci_cell("allen_cahn", "id_64_20", "env"),
         f"{100*lsci_rev['pdes']['allen_cahn']['summary']['ood_64_20']['0.1']['envelope_coverage']['mean']:.2f}"),
        ("5-FNO Ensemble (mean)",
         _p2d_cell("2da", "id_test_64", ENSEMBLE, "cov_mw"),
         _p2d_cell("2da", "ood_a_64", ENSEMBLE, "cov"),
         _p2d_cell("2da", "ood_b_64", ENSEMBLE, "cov"),
         _p2d_cell("2db", "id_64", ENSEMBLE, "cov_mw"),
         _p2d_cell("2db", "ood_64", ENSEMBLE, "cov"),
         _p2d_cell("2db", "ood_high_64", ENSEMBLE, "cov"),
         _p2d_cell("2dc", "id_64_20", ENSEMBLE, "cov_mw"),
         _p2d_cell("2dc", "ood_64_20", ENSEMBLE, "cov")),
        ("MC-Dropout",
         _p2d_cell("2da", "id_test_64", DROPOUT, "cov_mw"),
         _p2d_cell("2da", "ood_a_64", DROPOUT, "cov"),
         _p2d_cell("2da", "ood_b_64", DROPOUT, "cov"),
         _p2d_cell("2db", "id_64", DROPOUT, "cov_mw"),
         _p2d_cell("2db", "ood_64", DROPOUT, "cov"),
         _p2d_cell("2db", "ood_high_64", DROPOUT, "cov"),
         _p2d_cell("2dc", "id_64_20", DROPOUT, "cov_mw"),
         _p2d_cell("2dc", "ood_64_20", DROPOUT, "cov")),
        ("OpRes-CP (ours)",
         _p2d_cell("2da", "id_test_64", PHASE2_OP, "cov_mw"),
         _p2d_cell("2da", "ood_a_64", PHASE2_OP, "cov"),
         _p2d_cell("2da", "ood_b_64", PHASE2_OP, "cov"),
         _p2d_cell("2db", "id_64", PHASE2_OP, "cov_mw"),
         _p2d_cell("2db", "ood_64", PHASE2_OP, "cov"),
         _p2d_cell("2db", "ood_high_64", PHASE2_OP, "cov"),
         _p2d_cell("2dc", "id_64_20", PHASE2_OP, "cov_mw"),
         _p2d_cell("2dc", "ood_64_20", PHASE2_OP, "cov")),
    ]
    t_base = table(
        ("Method", "Darcy ID", "OOD-A", "OOD-B", "Burgers ID", "Visc.", "High", r"AC $T=1$ ID", "React."),
        baseline_rows, "lrrrrrrrr",
        r"Same-protocol five-seed FNO comparison across all evaluated baselines at $\alpha=0.10$. ID columns report simultaneous coverage (\%) / matched $90\%$ full width (for LSCI 64-draw envelope, actual finite width); OOD columns report simultaneous coverage (\%) at the locked source threshold.",
        "tab:opres-all-baselines", size=r"\scriptsize", tabcolsep="2.5pt",
        notes=r"$^*$Under Darcy OOD-B, one-step Defect Correction increases point relative $L^2$ error from $17.62\%$ to $82.64\%$ ($18.30\%$ to $89.47\%$ on U-Net, $0.00\%$ coverage). Mabtoul-adapted CW drops to $0.36\%$ on FNO and $0.00\%$ on U-Net. $^\dagger$RLCP-weighted LSCI (Appendix Table~\ref{tab:opres-lsci}); native set and 64-draw rectangular envelope measure distinct coverage events."
    )

    # 2) RLCP-weighted LSCI summary table (A3)
    lsci_rows = []
    lsci_specs = (
        ("Darcy", "darcy", (("id_test_64", "ID $64^2$"), ("ood_a_64", "Contrast OOD"), ("ood_b_64", "Structure OOD"))),
        ("Burgers", "burgers", (("id_64", "ID $64$"), ("ood_64", "Viscosity OOD"), ("ood_high_64", "High-freq.\\ OOD"))),
        ("Allen--Cahn $T=1$", "allen_cahn", (("id_64_20", "ID $64^2$"), ("ood_64_20", "Reaction OOD"))),
    )
    for pidx, (pname, pde, spairs) in enumerate(lsci_specs):
        if pidx > 0:
            lsci_rows.append(None)
        for grp, slabel in spairs:
            r90 = lsci_rev["pdes"][pde]["summary"][grp]["0.1"]
            r95 = lsci_rev["pdes"][pde]["summary"][grp]["0.05"]
            fw = r90["envelope_finite_width"]["mean"]
            lsci_rows.append((
                pname, slabel,
                f"{100*r90['native_slice_set_coverage']['mean']:.2f}",
                f"{100*r95['native_slice_set_coverage']['mean']:.2f}",
                f"{100*r90['envelope_coverage']['mean']:.2f}",
                f"{fw:.4f}" if fw is not None else "--",
                f"{100*r90['native_infinite_rate']['mean']:.2f}",
                f"{r90['post_calibration_p50_ms']['mean']:.2f}",
            ))
    t_lsci = table(
        ("Benchmark", "Split", r"Nat.\ 90\%", r"Nat.\ 95\%", r"Env.\ 90\%", "Env.\\ width", r"Inf.\ (\%)", "CPU ms"),
        lsci_rows, "llrrrrrr",
        r"RLCP-weighted LSCI across five frozen FNO seeds (32 random projections, 64 envelope draws). Nat.\ 90\%/95\% evaluates inclusion in the native projected-depth set using symmetric $(n+1)$-point Tukey depth and the weighted conformal quantile with test mass $w_{n+1}$ at $0$. Env.\ 90\% and Env.\ width evaluate the label-free 64-draw coordinatewise rectangular envelope on finite cases. CPU ms is the median per-case post-calibration time.",
        "tab:opres-lsci", size=r"\footnotesize", tabcolsep="4pt",
    )

    # 3) Cross-architecture (Model-Agnostic) table (B1)
    def _paired_shrink(tbl, model, split):
        op_r = {int(r["seed"]): float(r["matched_field_width"]) for r in tbl[(model, split, PHASE2_OP, 0.1)]}
        un_r = {int(r["seed"]): float(r["matched_field_width"]) for r in tbl[(model, split, "Uniform-CP", 0.1)]}
        vals = [100.0 * (1.0 - op_r[s] / un_r[s]) for s in SEEDS]
        return f"{np.mean(vals):.2f}"

    def _mean_cov(tbl, model, split, method):
        rs = tbl.get((model, split, method, 0.1))
        return None if not rs else 100.0 * float(np.mean([float(r["coverage"]) for r in rs]))

    cross_specs = (
        ("Darcy", g2a, "FNO", "id_test_64", "ood_a_64", "ood_b_64", "paired_test_64", "paired_test_128", r"$64^2\!\to\!128^2$"),
        ("Darcy", g2a, "U-Net", "id_test_64", "ood_a_64", "ood_b_64", None, None, "--"),
        ("Darcy", g2a, "DeepONet", "id_test_64", "ood_a_64", "ood_b_64", "paired_test_64", "paired_test_128", r"$64^2\!\to\!128^2$"),
        None,
        ("Burgers", g2b, "FNO", "id_64", "ood_64", "ood_high_64", "paired_64", "paired_256", r"$64\!\to\!256$"),
        ("Burgers", g2b, "U-Net", "id_64", "ood_64", "ood_high_64", "paired_64", "paired_256", r"$64\!\to\!256$"),
        ("Burgers", g2b, "DeepONet", "id_64", "ood_64", "ood_high_64", "paired_64", "paired_256", r"$64\!\to\!256$"),
        None,
        ("Allen--Cahn $T=1$", g2c, "FNO", "id_64_t1", "ood_64_t1", None, "paired_64_t1", "paired_128_t1", r"$64^2\!\to\!128^2$"),
        ("Allen--Cahn $T=1$", g2c, "U-Net", "id_64_t1", "ood_64_t1", None, "paired_64_t1", "paired_128_t1", r"$64^2\!\to\!128^2$"),
    )
    cross_rows = []
    for item in cross_specs:
        if item is None:
            cross_rows.append(None)
            continue
        pde_lab, tbl, mod, id_s, ood1_s, ood2_s, p_src, p_dst, grid_lab = item
        id_cov = f"{_mean_cov(tbl, mod, id_s, PHASE2_OP):.2f}"
        shrink = _paired_shrink(tbl, mod, id_s)
        o1 = f"{_mean_cov(tbl, mod, ood1_s, PHASE2_OP):.2f}"
        o2 = f"{_mean_cov(tbl, mod, ood2_s, PHASE2_OP):.2f}" if ood2_s else "--"
        if p_src and p_dst:
            op_d = signed(_mean_cov(tbl, mod, p_dst, PHASE2_OP) - _mean_cov(tbl, mod, p_src, PHASE2_OP))
            un_d = signed(_mean_cov(tbl, mod, p_dst, "Uniform-CP") - _mean_cov(tbl, mod, p_src, "Uniform-CP"))
        else:
            op_d, un_d = "--", "--"
        cross_rows.append((pde_lab, mod, id_cov, rf"{shrink}\%", o1, o2, grid_lab, op_d, un_d))

    t_cross = table(
        ("PDE", "Backbone", r"ID cov.", r"Width vs.\ Unif.", "OOD-1", "OOD-2", "Transfer", r"OpRes $\Delta$", r"Unif.\ $\Delta$"),
        cross_rows, "llrrrrlrr",
        r"Model-agnostic OpRes-CP performance across architectures (five seeds, $\alpha=0.10$). Width vs.\ Unif.\ is the five-seed mean paired reduction in matched $90\%$ width relative to Uniform-CP. OOD-1 / OOD-2 are Darcy contrast / structure, Burgers viscosity / high-frequency, and Allen--Cahn reaction / --. Transfer columns show paired coverage change in percentage points ($\Delta$).",
        "tab:opres-cross-arch", size=r"\footnotesize", tabcolsep="3.2pt",
    )

    # 4) Split-fit WCP vs Transductive WCP (B2)
    canon_map = {
        ("darcy", "grid64_ood_a_64", "Summary"): ("2da", "ood_a_64", "Physical-summary WCP"),
        ("darcy", "grid64_ood_a_64", "Full"): ("2da", "ood_a_64", "Full-grid WCP"),
        ("darcy", "grid64_ood_b_64", "Summary"): ("2da", "ood_b_64", "Physical-summary WCP"),
        ("darcy", "grid64_ood_b_64", "Full"): ("2da", "ood_b_64", "Full-grid WCP"),
        ("burgers", "ood", "Summary"): ("2db", "ood_64", "Physical-summary WCP"),
        ("burgers", "ood", "Spectrum"): ("2db", "ood_64", "Physical-summary + spectrum WCP"),
        ("burgers", "ood_high", "Summary"): ("2db", "ood_high_64", "Physical-summary WCP"),
        ("burgers", "ood_high", "Spectrum"): ("2db", "ood_high_64", "Physical-summary + spectrum WCP"),
        ("allen_cahn", "short_ood", "Summary"): ("2dc", "ood_64_20", "Physical-summary WCP"),
        ("allen_cahn", "short_ood", "Full"): ("2dc", "ood_64_20", "Full-grid WCP"),
    }
    split_rows = []
    for idx, entry in enumerate(wcp_split["rows"]):
        if idx > 0:
            split_rows.append(None)
        ck = canon_map[(entry["pde"], entry["target_group"], entry["feature"])]
        c = canonical_wcp[ck]
        split_rows.append((
            entry["label"], entry["feature"], "Transductive (500)",
            f"{c['cov']:.2f}", f"{c['inf']:.2f}", f"{c['finite_n']:.1f}",
            f"{c['finite_cov']:.2f}" if c["finite_cov"] is not None else "--",
            f"{c['finite_width']:.5f}" if c["finite_width"] is not None else "--",
            f"{c['ess']:.2f}",
        ))
        m = entry["split_fit_strict"]
        fc = m["finite_coverage"]["mean"]
        fw = m["finite_width"]["mean"]
        split_rows.append((
            "", "", "Split-fit (250/250)",
            f"{100*m['coverage']['mean']:.2f}",
            f"{100*m['infinite_rate']['mean']:.2f}",
            f"{m['finite_n']['mean']:.1f}",
            f"{100*fc:.2f}" if fc is not None else "--",
            f"{fw:.5f}" if fw is not None else "--",
            f"{m['ess']['mean']:.2f}",
        ))
    t_split = table(
        ("Shift", "Features", "Protocol", "Cov.", "Inf.", "Finite $n$", "Finite cov.", "Finite width", "ESS"),
        split_rows, "lllrrrrrr",
        r"Comparison of formal transductive WCP ($n_{\rm cal}=500$, identical to Table~\ref{tab:paper-wcp-formal}) against independent split-fit WCP satisfying the premise of Theorem~\ref{thm:shift} (first $250$ Cal cases plus the held-out target half fit the density ratio; the disjoint $250$ Cal scores compute weighted quantiles for the evaluated target half, averaged over both target halves so all $500$ target cases per seed enter the denominator). All entries use $C=1$, $\mathrm{cap}=50$ at $\alpha=0.10$ across five seeds.",
        "tab:opres-wcp-split", size=r"\footnotesize", tabcolsep="3.2pt",
    )

    return [t_base, t_cross], [t_lsci, t_split]


def p2_tables():
    """Generate booktabs tables for P2-A (L-shaped Darcy AMG) and P2-B (5-seed GPU Allen-Cahn)."""
    p2_root = Path("results/phase3_p2")
    lshape_path = p2_root / "lshape_darcy" / "lshape_summary.json"
    gpu_path = p2_root / "ac_gpu_timing" / "ac_gpu_benchmark.json"
    t_lshape, t_gpu = None, None

    if lshape_path.exists():
        ls = json.loads(lshape_path.read_text(encoding="utf-8"))
        rows = []
        order = (
            ("Uniform-CP", "Uniform-CP"),
            ("Choi CP-B", "Choi CP-B"),
            ("Full-Field PI-CP", "Full-Field PI-CP"),
            ("Ensemble (same center)", "Ensemble (same center)"),
            ("Ensemble (mean center)", "Ensemble (mean center)"),
            None,
            ("Unsigned AMG-V2", "Unsigned AMG-V2"),
            ("Jacobi-PCG5", "Jacobi-PCG5"),
            ("OpRes-CP (AMG-V2)", "OpRes-CP (AMG-V2)"),
            ("OpRes-CP (AMG-PCG2)", "OpRes-CP (AMG-PCG2)"),
            ("Exact Sparse Lift (oracle)", "Exact sparse lift (oracle)"),
        )
        for item in order:
            if item is None:
                rows.append(None)
                continue
            key, lab = item
            id_m = ls["splits"]["id_64"][key]
            p128 = ls["splits"]["paired_128"][key]
            tr = ls["paired_transfer"][key]
            ood = ls["splits"]["ood_a_64"][key]
            rows.append((
                lab,
                f"{100*id_m['coverage']['mean']:.2f} / {id_m['matched_width']['mean']:.5f}",
                f"{100*id_m['corner_sim_cov']['mean']:.2f}",
                rf"{100*p128['coverage']['mean']:.2f} (${tr['cov_delta_pp']['mean']:+.2f}$)",
                f"{100*ood['coverage']['mean']:.2f} / {ood['matched_width']['mean']:.5f}",
            ))
        t_lshape = table(
            ("Method", r"ID $64^2$ cov / width", r"Corner cov.", r"Paired $128^2$ ($\Delta$ pp)", r"Contrast OOD cov / width"),
            rows, "lrrrr",
            r"Non-rectangular L-shaped 2D Darcy ($\Omega_L = (0,1)^2 \setminus [0.5,1)^2$) across five formal seeds ($\alpha=0.10$, 300 test cases per split). Widths are empirically matched at $90\%$ simultaneous coverage; Corner cov.\ evaluates simultaneous coverage within radius $0.15$ of the re-entrant corner $(0.5,0.5)$. Fast spectral FFT/DST inverses do not apply on $\Omega_L$; OpRes-CP uses $O(M)$ cell-agglomeration Galerkin AMG lifting.",
            "tab:opres-lshape", size=r"\footnotesize", tabcolsep="3.5pt",
        )

    if gpu_path.exists():
        gp = json.loads(gpu_path.read_text(encoding="utf-8"))
        rows = []
        specs = (
            ("64_T1", r"$64^2$, $T=1$"),
            ("64_T2", r"$64^2$, $T=2$"),
            ("128_T1", r"$128^2$, $T=1$"),
            ("128_T2", r"$128^2$, $T=2$"),
        )
        for key, lab in specs:
            s = gp["single_case_ms"][key]
            b8 = gp["batch8_per_case_ms"][key]
            par = gp["parity"][key]["scale_rel_l2"]["max"]
            rows.append((
                lab,
                f"{s['fno_opres_cpu_post']['mean']:.2f}",
                f"{s['fno_opres_gpu_full']['mean']:.2f}",
                f"{s['fno_ens5_gpu']['mean']:.2f}",
                f"{s['fno_mc30_gpu']['mean']:.2f}",
                f"{s['speedup_gpu_over_ens5_fno']:.2f}$\\times$",
                f"{b8['fno_opres_gpu_mean']:.2f} vs.\\ {b8['fno_ens5_gpu_mean']:.2f}",
                f"{par:.1e}",
            ))
        t_gpu = table(
            ("Setting", "CPU post", "GPU full", "Ens-5", "MC-30", "Speedup vs.\\ Ens", "Batch-8 (Op / Ens)", "Max parity"),
            rows, "lrrrrrrr",
            r"Five-seed end-to-end Allen--Cahn FNO latency (P50 ms per trajectory on RTX 4060; frozen sealed checkpoints) comparing CPU post-processing, full GPU-resident Strang lift plus periodic Gaussian envelope, five-member ensemble, and $K=30$ MC-Dropout. Max parity reports the worst-seed relative $L^2$ scale discrepancy between GPU-resident and CPU SciPy implementations.",
            "tab:opres-ac-gpu", size=r"\footnotesize", tabcolsep="3.0pt",
        )
    return t_lshape, t_gpu


def save_tables(name, blocks):
    contents = "% Auto-generated booktabs LaTeX tables. Requires booktabs.\n\n" + "\n\n".join(blocks) + "\n"
    if contents.count(r"\begin{") != contents.count(r"\end{") or contents.count("{") != contents.count("}"):
        raise ValueError(f"Unclosed LaTeX syntax in {name}")
    if r"\hline" in contents:
        raise ValueError(f"Forbidden \\hline detected in {name}")
    for directory in (REV_TABLES, PAPER_TABLES, REPLAY_TABLES):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text(contents, encoding="ascii")


def main():
    data, _ = load()
    p2d_csv = _read_p2d_csv()
    darcy_rev = json.loads((REV / "darcy_boundary_revision.json").read_text(encoding="utf-8"))
    lsci_rev = json.loads((REV / "lsci_revision.json").read_text(encoding="utf-8"))
    wcp_split = json.loads((REV / "wcp_split_vs_transductive.json").read_text(encoding="utf-8"))

    proto_blocks, canonical_wcp = protocol_metrics(p2d_csv)
    promoted_blocks, appendix_rev_blocks = revision_tables(p2d_csv, canonical_wcp, lsci_rev, wcp_split)
    save_tables("phase3_main_tables.tex", primary())
    save_tables("protocol_metrics.tex", proto_blocks)
    save_tables("phase3_ablation_tables.tex", ablations(data, canonical_wcp, darcy_rev))
    save_tables("phase3_promoted_tables.tex", promoted_blocks)
    save_tables("phase3_revision_tables.tex", appendix_rev_blocks)
    t_lshape, t_gpu = p2_tables()
    if t_lshape is not None:
        save_tables("phase3_p2_lshape_table.tex", [t_lshape])
    if t_gpu is not None:
        save_tables("phase3_p2_gpu_table.tex", [t_gpu])
    print(f"All booktabs table files written to {REV_TABLES}, {PAPER_TABLES}, and {REPLAY_TABLES}.")


if __name__ == "__main__":
    main()
