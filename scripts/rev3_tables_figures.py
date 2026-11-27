#!/usr/bin/env python3
"""Generate Revision v3 LaTeX tables, fig9_rev3_diagnostics, and CMAME Graphical Abstract."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
R3_DIR = ROOT / "results" / "phase3_revision_v3"
R2_DIR = ROOT / "results" / "phase3_revision_v2"
OUT_TBL_R3 = R3_DIR / "tables"
OUT_FIG_R3 = R3_DIR / "figures"
_HAS_CMAME = (ROOT / "paper_cmame").is_dir()
CMAME_TBL = ROOT / "paper_cmame" / "tables" if _HAS_CMAME else ROOT / "outputs" / "tables"
CMAME_FIG = ROOT / "paper_cmame" / "figures" if _HAS_CMAME else ROOT / "outputs" / "figures"
SUB_PKG = ROOT / "paper_cmame" / "submission_package" if _HAS_CMAME else ROOT / "outputs" / "submission_package"

for d in (OUT_TBL_R3, OUT_FIG_R3, CMAME_TBL, CMAME_FIG):
    d.mkdir(parents=True, exist_ok=True)
if _HAS_CMAME:
    SUB_PKG.mkdir(parents=True, exist_ok=True)

# Match scripts/phase3_figures.py visual style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 8.8,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.8,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 7.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

OP_COLOR = "#006B62"
OP_LIGHT = "#DDF0ED"
EN_COLOR = "#C45A2A"
EN_LIGHT = "#FBE9E1"
MC_COLOR = "#6256A5"
SLATE = "#2B5C8F"
GRAY = "#6B7280"
DARK = "#1F2937"


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt_ms(stat: dict, dec: int = 3, pct: bool = False) -> str:
    m = stat["mean"] * (100.0 if pct else 1.0)
    s = stat["sd"] * (100.0 if pct else 1.0)
    return f"${m:.{dec}f} \\pm {s:.{dec}f}$"


def fmt_val(stat: dict, dec: int = 4, pct: bool = False) -> str:
    v = stat["mean"] * (100.0 if pct else 1.0)
    return f"{v:.{dec}f}"


def write_both(rel_name: str, text: str) -> None:
    (OUT_TBL_R3 / rel_name).write_text(text, encoding="utf-8")
    (CMAME_TBL / rel_name).write_text(text, encoding="utf-8")


def read_base_table_tex(name: str) -> str:
    """Read clean base table from paper/tables or results/phase3_revision/tables with CMAME syntax."""
    candidates = [
        ROOT / "paper" / "tables" / name,
        ROOT / "results" / "phase3_revision" / "tables" / name,
        CMAME_TBL / name,
    ]
    src_path = next((p for p in candidates if p.is_file()), None)
    if src_path is None:
        raise FileNotFoundError(f"Base table {name} not found in {[str(c) for c in candidates]}")
    text = src_path.read_text(encoding="utf-8")
    text = text.replace("table*}", "table}").replace("table*}", "table}")
    text = re.sub(r"\\citep\b", r"\\cite", text)
    return text


def replace_table_by_label(doc: str, label: str, new_block: str) -> str:
    """Replace the exact table environment containing \\label{label}."""
    pattern = re.compile(
        r"\\begin\{table\}(?:\[[^\]]*\])?(?:(?!\\end\{table\}).)*?\\label\{"
        + re.escape(label)
        + r"\}.*?\\end\{table\}",
        re.S,
    )
    updated, count = pattern.subn(lambda _: new_block, doc, count=1)
    assert count == 1, f"Expected 1 table with label {label}, found {count}"
    return updated


# ---------------------------------------------------------------------------
# 1. Update paper_cmame/tables/phase3_rev2_main_tables.tex
# ---------------------------------------------------------------------------
def build_rev2_main_tables(t1_v2: dict, t1_v3: dict) -> None:
    orig = read_base_table_tex("phase3_rev2_main_tables.tex")

    # (a) Rebuild tab:opres-signed-recovery with Ensemble rho_i and OpRes/Ens ratio
    sr_rows = [
        ("Darcy $64^2$ ID", "darcy", "ID", "0.854 \\pm 0.007", "0.765 \\pm 0.026", "0.471 \\pm 0.009"),
        ("Darcy $64^2$ OOD-A", "darcy", "OOD-A", "0.786 \\pm 0.006", "1.153 \\pm 0.064", "0.562 \\pm 0.006"),
        ("Darcy $64^2$ OOD-B", "darcy", "OOD-B", "0.186 \\pm 0.154", "4.964 \\pm 2.952", "0.898 \\pm 0.006"),
        ("\\addlinespace[3pt]", None, None, None, None, None),
        ("Burgers $64$ ID", "burgers", "ID", "0.724 \\pm 0.011", "0.830 \\pm 0.026", "0.648 \\pm 0.011"),
        ("Burgers $64$ Visc.-OOD", "burgers", "Visc-OOD", "0.618 \\pm 0.016", "1.082 \\pm 0.024", "0.752 \\pm 0.013"),
        ("Burgers $64$ High-OOD", "burgers", "High-OOD", "0.812 \\pm 0.005", "0.583 \\pm 0.009", "0.554 \\pm 0.006"),
        ("\\addlinespace[3pt]", None, None, None, None, None),
        ("Allen--Cahn $64^2$ $T=1$ ID", "allen_cahn", "ID", "0.996 \\pm 0.000", "0.091 \\pm 0.003", "0.063 \\pm 0.002"),
        ("Allen--Cahn $64^2$ React.-OOD", "allen_cahn", "React-OOD", "0.990 \\pm 0.001", "0.131 \\pm 0.013", "0.101 \\pm 0.008"),
        ("\\addlinespace[3pt]", None, None, None, None, None),
        ("L-shape $64^2$ ID (AMG-PCG2)", "lshape_darcy", "ID", "0.954 \\pm 0.002", "0.323 \\pm 0.005", "0.287 \\pm 0.004"),
        ("L-shape $64^2$ OOD-A (AMG-PCG2)", "lshape_darcy", "OOD-A", "0.954 \\pm 0.001", "0.313 \\pm 0.006", "0.279 \\pm 0.004"),
    ]
    fid_sum = t1_v2["summary"]
    ratios = []
    sr_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\caption{Signed solution-space error recovery of $Z_h=B_h R_h$ against $e_h=\widehat U_h - U_h$ across all ten benchmark splits (five-seed mean $\pm$ SD), together with Theorem~\ref{thm:fixed}'s effective peak-concentration ratio $\overline\rho_i$ for OpRes-CP vs.\ the 5-member ensemble ($\sigma_{\rm ens}$), their paired concentration gain (OpRes/Ens.), and $\tilde\sigma(x^*)$. Positive homogeneity absorbs global mis-scaling when $\|Z_h-e_h\|_2/\|e_h\|_2 > 1$, while OpRes-CP concentrates $RATIO_MIN$--$RATIO_MAX\times$ more normalized mass at the true peak error location (computed from exact per-seed ratios).}",
        r"\label{tab:opres-signed-recovery}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Benchmark / split & $\cos(Z_h, e_h)$ & $\frac{\|Z_h-e_h\|_2}{\|e_h\|_2}$ & $\min_\alpha\frac{\|\alpha Z_h-e_h\|_2}{\|e_h\|_2}$ & OpRes $\overline\rho_i$ & Ens.\ $\overline\rho_i$ & Gain & $\tilde\sigma(x^*)$ \\",
        r"\midrule",
    ]
    for label, pde, sp, cos_s, l2_s, ls_s in sr_rows:
        if pde is None:
            sr_lines.append(label)
            continue
        op_sc = fid_sum[pde][sp]["scales"]["opres_sigma"]
        en_sc = fid_sum[pde][sp]["scales"]["ensemble_std"]
        op_r = op_sc["rho_eff_mean"]
        en_r = en_sc["rho_eff_mean"]
        op_sx = op_sc["sigma_tilde_xstar_mean"]
        # Paired ratio from exact 5-seed dict
        seeds = sorted(op_r["per_seed"].keys())
        pair_ratio = np.mean([op_r["per_seed"][s] / en_r["per_seed"][s] for s in seeds])
        ratios.append(pair_ratio)
        sr_lines.append(
            f"{label} & ${cos_s}$ & ${l2_s}$ & ${ls_s}$ & "
            f"{fmt_ms(op_r, 2)} & {fmt_ms(en_r, 2)} & ${pair_ratio:.2f}\\times$ & {fmt_ms(op_sx, 2)} \\\\"
        )
    sr_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    new_sr_block = "\n".join(sr_lines).replace("RATIO_MIN", f"{min(ratios):.2f}").replace("RATIO_MAX", f"{max(ratios):.2f}")

    # (b) Rebuild tab:opres-op-approx distinguishing directional d_h(e_h) from global Power-Iteration operator norms
    part_a = t1_v3["part_a_opnorms"]
    op_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\caption{Directional defect $d_h(e_h)=\|B_h A_h e_h - e_h\|_2/\|e_h\|_2$ vs.\ global induced operator norms $\|B_h A_h - I\|_{2\to 2}$ and $\|I - B_h A_h\|_{A_h}$ computed by Power Iteration on $A_h$ across five seeds (mean $\pm$ SD). Proposition~\ref{prop:resolution}'s global norm $\delta_{\rm inv}$ exceeds the directional defect $d_h(e_h)$; only multilevel AMG achieves strict global energy contraction $\|I - B_h A_h\|_{A_h} < 1$ across 100\% of rectangular and L-shape instances ($1.000$ frac.$<1$). Note that \texttt{AMG-PCG2} and \texttt{Jacobi-PCG5} are nonlinear Krylov preconditioners and therefore evaluated via $d_h(e_h)$.}",
        r"\label{tab:opres-op-approx}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Domain / split & Operator $B_h$ & Dir.\ defect $d_h(e_h)$ & Global $\|B_h A_h - I\|_{2\to 2}$ & Global $\|I - B_h A_h\|_{A_h}$ & Frac.\ $<1$ ($A_h$) \\",
        r"\midrule",
    ]
    specs_a = [
        ("darcy_rect", "ID", "Darcy $64^2$ ID", ["liouville", "amg_v2", "amg_v2_spd", "jacobi5_lin", "amg_pcg2", "jacobi_pcg5"]),
        ("darcy_rect", "OOD-A", "Darcy $64^2$ OOD-A", ["liouville", "amg_v2", "amg_v2_spd", "jacobi5_lin", "amg_pcg2", "jacobi_pcg5"]),
        ("darcy_rect", "OOD-B", "Darcy $64^2$ OOD-B", ["liouville", "amg_v2", "amg_v2_spd", "jacobi5_lin", "amg_pcg2", "jacobi_pcg5"]),
        ("darcy_lshape", "ID", "L-shape $64^2$ ID", ["amg_v2", "amg_v2_spd", "jacobi5_lin", "amg_pcg2", "jacobi_pcg5"]),
        ("darcy_lshape", "OOD-A", "L-shape $64^2$ OOD-A", ["amg_v2", "amg_v2_spd", "jacobi5_lin", "amg_pcg2", "jacobi_pcg5"]),
    ]
    op_names = {
        "liouville": "Liouville DST-II",
        "amg_v2": "AMG-V2 (linear)",
        "amg_v2_spd": "AMG-V2-SPD (linear)",
        "jacobi5_lin": "Jacobi5 (linear)",
        "amg_pcg2": "AMG-PCG2 (NL)",
        "jacobi_pcg5": "Jacobi-PCG5 (NL)",
    }
    for idx_g, (dom, sp, dom_lbl, ops) in enumerate(specs_a):
        sp_d = part_a[dom]["summary"][sp]
        for j, op in enumerate(ops):
            m = sp_d[op]
            lbl = dom_lbl if j == 0 else ""
            dl2 = fmt_ms(m["dir_l2"], 3)
            if "global_l2" in m:
                gl2 = fmt_ms(m["global_l2"], 3)
                gen = fmt_ms(m["global_energy"], 3)
                flt = fmt_val(m["frac_lt_1"], 3)
            else:
                gl2, gen, flt = "--", "--", "--"
            op_lines.append(f"{lbl} & {op_names[op]} & {dl2} & {gl2} & {gen} & {flt} \\\\")
        if idx_g < len(specs_a) - 1:
            op_lines.append(r"\addlinespace[2.5pt]")
    op_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    new_op_block = "\n".join(op_lines)

    # Replace both blocks safely by exact table label
    updated = replace_table_by_label(orig, "tab:opres-signed-recovery", new_sr_block)
    updated = replace_table_by_label(updated, "tab:opres-op-approx", new_op_block)
    write_both("phase3_rev2_main_tables.tex", updated)
    print(f"Updated phase3_rev2_main_tables.tex (min/max paired ratio = {min(ratios):.2f}x .. {max(ratios):.2f}x)")


# ---------------------------------------------------------------------------
# 2. Update paper_cmame/tables/phase3_revision_tables.tex (tab:opres-lsci-fair)
# ---------------------------------------------------------------------------
def build_rev_tables(t2_v3: dict) -> None:
    orig = read_base_table_tex("phase3_revision_tables.tex")

    rows_spec = [
        ("Darcy", "ID $64^2$", "darcy", "id_test_64"),
        ("", "Contr.\\ OOD", "darcy", "ood_a_64"),
        ("", "Struct.\\ OOD", "darcy", "ood_b_64"),
        ("SEP", "", "", ""),
        ("Burgers", "ID $64$", "burgers", "id_64"),
        ("", "Visc.\\ OOD", "burgers", "ood_64"),
        ("", "High OOD", "burgers", "ood_high_64"),
        ("SEP", "", "", ""),
        ("AC $T=1$", "ID $64^2$", "allen_cahn", "id_64_20"),
        ("", "React.\\ OOD", "allen_cahn", "ood_64_20"),
    ]
    pdes = t2_v3["pdes"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{1.7pt}",
        r"\caption{Leakage-free RLCP-weighted LSCI comparison across five frozen FNO seeds ($\alpha=0.10$, identical RLCP kernel weights $w$ and 32 projections $\{\phi_m\}_{m=1}^{32}$). All entries report simultaneous coverage (\%) / width (cov./act./m90 when all three appear). Left: whole-field rectangular band under two exchangeable denominators---\textbf{Val-Fit RLCP-Band} ($\sigma_{\rm LSCI}^{\rm val}$ fit on disjoint Validation residuals, relative floor $\rho_{\rm rel}=0.10$, exact RLCP weighted max-conformal quantile) and \textbf{LOO RLCP-Band} (leave-one-out denominator excluding each point before scoring)---against \textbf{OpRes-CP} (M90 compares retrospective matched-90\% full widths; Shrink reports OpRes-CP's matched-90\% reduction vs.\ Val-Fit). Right: 32-projection native depth set (\textbf{Nat.-32}) vs.\ OpRes-CP's whole-field band support width (\textbf{Box$\to$32}, 100.0\% coverage) and projection-calibrated OpRes-CP (\textbf{Cal-32}, full cov./act./m90).}",
        r"\label{tab:opres-lsci-fair}",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"PDE & Split & Val-Fit Band (cov./act./m90) & LOO Band (cov./act./m90) & OpRes (cov./m90) & Shrink & Nat.-32 & Box$\to$32 & OpRes Cal-32 (cov./act./m90) \\",
        r"\midrule",
    ]
    for pde_lbl, sp_lbl, pde_k, sp_k in rows_spec:
        if pde_lbl == "SEP":
            lines.append(r"\addlinespace[2.5pt]")
            continue
        s = pdes[pde_k]["summary"][sp_k]
        vf = s["valfit_rlcp_band"]
        loo = s["loo_rlcp_band"]
        op = s["opres_whole_field"]
        sh = s["width_shrink_pct_vs_valfit_matched90"]["mean"]
        p32 = s["proj32"]

        vf_str = f"{fmt_val(vf['coverage'],2,True)}/{fmt_val(vf['finite_actual_width'],4)}/{fmt_val(vf['matched90_width'],4)}"
        loo_str = f"{fmt_val(loo['coverage'],2,True)}/{fmt_val(loo['finite_actual_width'],4)}/{fmt_val(loo['matched90_width'],4)}"
        op_str = f"{fmt_val(op['coverage'],2,True)}/{fmt_val(op['matched90_width'],4)}"
        nat_str = f"{fmt_val(p32['lsci_native32_cov'],2,True)}/{fmt_val(p32['lsci_slab32_finite_width'],4)}"
        box_str = f"{fmt_val(p32['opres_full_band_induced_32proj_cov'],1,True)}/{fmt_val(p32['opres_full_band_induced_32proj_width'],4)}"
        c32_str = f"{fmt_val(p32['opres_cal32_cov'],2,True)}/{fmt_val(p32['opres_cal32_actual_width'],4)}/{fmt_val(p32['opres_cal32_matched90_width'],4)}"
        lines.append(
            f"{pde_lbl} & {sp_lbl} & {vf_str} & {loo_str} & {op_str} & {sh:.1f}\\% & {nat_str} & {box_str} & {c32_str} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    new_lsci = "\n".join(lines)
    updated = replace_table_by_label(orig, "tab:opres-lsci-fair", new_lsci)
    write_both("phase3_revision_tables.tex", updated)
    print("Updated phase3_revision_tables.tex (tab:opres-lsci-fair)")


# ---------------------------------------------------------------------------
# 3. Update phase3_promoted_tables.tex (tab:opres-all-baselines LSCI row)
# ---------------------------------------------------------------------------
def build_promoted_tables(t2_v3: dict) -> None:
    orig = read_base_table_tex("phase3_promoted_tables.tex")
    pdes = t2_v3["pdes"]
    d_id = pdes["darcy"]["summary"]["id_test_64"]["valfit_rlcp_band"]
    d_oa = pdes["darcy"]["summary"]["ood_a_64"]["valfit_rlcp_band"]
    d_ob = pdes["darcy"]["summary"]["ood_b_64"]["valfit_rlcp_band"]
    b_id = pdes["burgers"]["summary"]["id_64"]["valfit_rlcp_band"]
    b_oa = pdes["burgers"]["summary"]["ood_64"]["valfit_rlcp_band"]
    b_ob = pdes["burgers"]["summary"]["ood_high_64"]["valfit_rlcp_band"]
    a_id = pdes["allen_cahn"]["summary"]["id_64_20"]["valfit_rlcp_band"]
    a_oa = pdes["allen_cahn"]["summary"]["ood_64_20"]["valfit_rlcp_band"]

    new_row = (
        f"LSCI (Val-Fit band)$^\\dagger$ & "
        f"{fmt_val(d_id['coverage'],2,True)} / {fmt_val(d_id['matched90_width'],4)} & "
        f"{fmt_val(d_oa['coverage'],2,True)} & {fmt_val(d_ob['coverage'],2,True)} & "
        f"{fmt_val(b_id['coverage'],2,True)} / {fmt_val(b_id['matched90_width'],4)} & "
        f"{fmt_val(b_oa['coverage'],2,True)} & {fmt_val(b_ob['coverage'],2,True)} & "
        f"{fmt_val(a_id['coverage'],2,True)} / {fmt_val(a_id['matched90_width'],4)} & "
        f"{fmt_val(a_oa['coverage'],2,True)} \\\\"
    )
    updated, n = re.subn(r"LSCI \(cal\.\\ band\)\$\^\\dagger\$.*?\\\\", lambda _: new_row, orig, count=1)
    assert n == 1, "Failed to replace LSCI (cal. band) row"
    updated = updated.replace(
        r"LSCI (cal.\ band) evaluates the same whole-field rectangular event using all 500 locally weighted calibration residuals",
        r"LSCI (Val-Fit band) fits local coordinate scale on disjoint Validation residuals (eliminating in-sample denominator leakage) and calibrates with the exact weighted max-conformal quantile on 500 Cal residuals",
    )
    write_both("phase3_promoted_tables.tex", updated)
    print("Updated phase3_promoted_tables.tex (tab:opres-all-baselines)")


# ---------------------------------------------------------------------------
# 4. Update phase3_main_tables.tex (tab:opres-horizon partitioned)
# ---------------------------------------------------------------------------
def build_main_tables() -> None:
    orig = read_base_table_tex("phase3_main_tables.tex")
    new_horizon = r"""\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{3.5pt}
\caption{Allen--Cahn $T=2$ coverage (\%) / actual full width across five seeds ($\alpha=0.10$), partitioned into (left) empirical $T=1 \to T=2$ heuristic quantile reuse without $T=2$ labels and (right) independent $T=2$ calibration satisfying finite-sample marginal validity under Corollary~\ref{cor:windows}.}
\label{tab:opres-horizon}
\begin{tabular}{llcc|cc}
\toprule
 & & \multicolumn{2}{c}{Heuristic $T=1 \to T=2$ reuse (no $T=2$ cal.)} & \multicolumn{2}{c}{Independent $T=2$ calibration} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
Backbone & Target & T1 90\% & T1 95\% & T2 90\% (Jac.\ on) & T2 90\% (Jac.\ off) \\
\midrule
FNO & ID $64^2$ & 82.52 / 0.0194 & 94.56 / 0.0208 & 90.60 / 0.0200 & 90.60 / 0.0555 \\
FNO & Paired $128^2$ & 76.56 / 0.0193 & 92.48 / 0.0206 & 86.56 / 0.0198 & 90.48 / 0.0550 \\
FNO & Reaction OOD & 55.32 / 0.0466 & 84.24 / 0.0498 & 73.96 / 0.0480 & 61.00 / 0.1369 \\
\addlinespace[3pt]
U-Net & ID $64^2$ & 83.72 / 0.0207 & 97.76 / 0.0240 & 91.12 / 0.0206 & 89.76 / 0.0498 \\
U-Net & Paired $128^2$ & 67.64 / 0.2508 & 91.52 / 0.2862 & 65.80 / 0.2455 & 99.96 / 0.6065 \\
U-Net & Reaction OOD & 83.96 / 0.0585 & 96.44 / 0.0673 & 85.12 / 0.0581 & 58.36 / 0.1587 \\
\bottomrule
\end{tabular}
\end{table}"""
    # Wait: CMAME prefers NO vertical rules (|), so replace cc|cc with cccc!
    new_horizon = new_horizon.replace(r"\begin{tabular}{llcc|cc}", r"\begin{tabular}{llcccc}")
    updated = replace_table_by_label(orig, "tab:opres-horizon", new_horizon)
    write_both("phase3_main_tables.tex", updated)
    print("Updated phase3_main_tables.tex (tab:opres-horizon)")


# ---------------------------------------------------------------------------
# 5. Build NEW paper_cmame/tables/phase3_rev3_tables.tex
# ---------------------------------------------------------------------------
def build_rev3_tables(t1_v3: dict, t3_v3: dict) -> None:
    blocks = ["% Auto-generated Revision v3 tables from task1_opnorm_gamma.json and task3_kernel_horizon.json\n"]

    # Table A: Darcy OOD-B AMG counterexample
    b = t1_v3["part_b_darcy_oodb_amg"]["summary"]
    fid = b["ood_b_fidelity"]
    sr = b["ood_b_signed_recovery"]
    idb = b["id_bands"]
    obb = b["ood_b_bands"]
    rows_b = [
        ("Liouville DST-II", "liouville"),
        ("AMG-V2 (linear)", "amg_v2"),
        ("AMG-PCG2", "amg_pcg2"),
        ("5-FNO Ensemble std", "ensemble_std"),
    ]
    tbl_b = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.3pt}",
        r"\caption{Darcy $64^2$ high-contrast structural OOD-B diagnostic explaining Liouville DST-II's weaker ranking vs.\ the 5-member ensemble ($\rho=0.489$ vs.\ $0.514$). Retaining the piecewise-constant permeability potential in \texttt{AMG-V2} / \texttt{AMG-PCG2} restores operator-aligned localization ($\rho=0.821/0.725$, $\cos=0.992/0.898$, $\overline\rho_i=2.03/1.92$, ID width $0.0083$--$0.0100$), confirming that the OOD-B limitation stems from Liouville's constant-coefficient potential approximation rather than the Physics-as-Scale principle itself. Note that locked-threshold \texttt{AMG} OOD-B coverage drops to $62.6\%/64.0\%$ because non-stationary coefficient jumps inflate fine-scale model error disproportionately relative to ID ($0.0289/0.0302$ vs.\ Liouville's high-variance $0.1873$).}",
        r"\label{tab:opres-oodb-amg}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Scale proxy & $\cos(Z_h, e_h)$ & Spearman $\rho$ & PeakCap@5\% & $\overline\rho_i$ & $\rho_{\rm cross}$ & ID cov./m90 w & OOD-B cov./m90 w \\",
        r"\midrule",
    ]
    for lbl, k in rows_b:
        f = fid[k]
        cos_s = fmt_ms(sr[k]["cos_mean"], 3) if k in sr else "--"
        id_s = f"{fmt_val(idb[k]['coverage'],2,True)}\\%/{fmt_val(idb[k]['matched90_width'],4)}" if k in idb else "--"
        ob_s = f"{fmt_val(obb[k]['coverage'],2,True)}\\%/{fmt_val(obb[k]['matched90_width'],4)}" if k in obb else "--"
        tbl_b.append(
            f"{lbl} & {cos_s} & {fmt_ms(f['spearman'],3)} & "
            f"{fmt_val(f['peak_capture_5'],1,True)}\\% & {fmt_ms(f['rho_eff_mean'],2)} & "
            f"{fmt_ms(f['rho_cross'],3)} & {id_s} & {ob_s} \\\\"
        )
    tbl_b += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    blocks.append("\n".join(tbl_b))

    # Table B: Temporal damping gamma ablation (Burgers + Allen-Cahn)
    cb = t1_v3["part_c_gamma_ablation"]["burgers"]["summary"]
    cac = t1_v3["part_c_gamma_ablation"]["allen_cahn"]["summary"]
    gammas = ["0", "0.05", "0.1", "0.2", "0.4"]
    t1_drop_cov = fmt_val(cac["0.4"]["T2_ID"]["t1_95reuse_coverage"], 2, True)
    tbl_c = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{1.6pt}",
        f"\\caption{{Five-seed sensitivity to the all-mode temporal regularization parameter $\\gamma$ on Burgers ($64$, locked default $\\gamma=0.1$) and Allen--Cahn ($64^2$, locked default $\\gamma=0.2$), matching Eqs.~\\eqref{{eq:burgers-lift}} and~\\eqref{{eq:ac-strang}}. For $\\gamma=0$, the spatial mean $k=0$ Fourier mode evaluates $\\varphi_1(0)=1$. Performance is flat across $\\gamma \\in [0, 0.2]$ on Burgers and independent Allen--Cahn calibration, whereas overly heavy damping ($\\gamma=0.4$) blurs Allen--Cahn $T=2$ independent width ($0.0254$) and destabilizes naive $T=1 \\to T=2$ reuse (${t1_drop_cov}\\%$, per-seed $26.2$--$47.0\\%$).}}",
        r"\label{tab:opres-gamma-ablation}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"$\gamma$ & Burgers ID $\cos$ & Burgers ID cov./m90 & Visc.-OOD cov./m90 & AC $T=1$ ID cov./m90 & AC React.\ cov./m90 & AC $T=2$ ID cov./m90 & AC $T=1$ 95\%$\to T=2$ cov. \\",
        r"\midrule",
    ]
    for g in gammas:
        gb = cb[g]
        gac = cac[g]
        g_disp = "0.0" if g == "0" else g
        star = "$^*$" if g in ("0.1", "0.2") else ""
        b_note = " (default)" if g == "0.1" else (" (AC def.)" if g == "0.2" else "")
        tbl_c.append(
            f"{g_disp}{b_note} & {fmt_val(gb['ID']['cos'],3)} & "
            f"{fmt_val(gb['ID']['coverage'],2,True)}/{fmt_val(gb['ID']['matched90_width'],4)} & "
            f"{fmt_val(gb['Visc-OOD']['coverage'],2,True)}/{fmt_val(gb['Visc-OOD']['matched90_width'],4)} & "
            f"{fmt_val(gac['T1_ID']['coverage'],2,True)}/{fmt_val(gac['T1_ID']['matched90_width'],4)} & "
            f"{fmt_val(gac['T1_ReactOOD']['coverage'],2,True)}/{fmt_val(gac['T1_ReactOOD']['matched90_width'],4)} & "
            f"{fmt_val(gac['T2_ID']['coverage'],2,True)}/{fmt_val(gac['T2_ID']['matched90_width'],4)} & "
            f"{fmt_val(gac['T2_ID']['t1_95reuse_coverage'],2,True)}\\% \\\\"
        )
    tbl_c += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    blocks.append("\n".join(tbl_c))

    # Table C: Kernel resolution scan + Millard-style extrapolation
    kd = t3_v3["part_a_kernel_scan"]["darcy_32_to_128"]["summary"]
    kb = t3_v3["part_a_kernel_scan"]["burgers_64_to_256"]["summary"]
    mill = t3_v3["part_c_millard_extrapolation"]["summary"]
    lpix_keys = ["0.64", "1", "1.28", "1.5", "2", "2.56", "3", "4"]
    tbl_k = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{1.8pt}",
        r"\caption{Continuous scan of calibration-grid pixel kernel width $\ell_{\rm pix,train}$ across five seeds ($\alpha=0.10$): zero-shot cross-resolution transfer under \textbf{fixed-physical-radius} vs.\ \textbf{fixed-cell-footprint} on Darcy $32^2 \to 128^2$ and Burgers $64 \to 256$ (reporting calibration-grid coverage and target-grid cov./actual width). Bottom: comparison on held-out Darcy $128^2$ against global scalar radius extrapolation in the spirit of Millard et al.~\cite{millard2025function} fit from paired $(32^2, 64^2)$ Uniform-CP quantiles ($h$ and $h^2$ scaling).}",
        r"\label{tab:opres-kernel-scan}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"$\ell_{\rm pix,train}$ & Darcy $32^2$ cov. & Darcy $128^2$ Fix-Phys (cov./w) & Darcy $128^2$ Fix-Cell (cov./w) & Burgers $64$ cov. & Burgers $256$ Fix-Phys (cov./w) & Burgers $256$ Fix-Cell (cov./w) \\",
        r"\midrule",
    ]
    for lk in lpix_keys:
        sd = kd[lk]
        sb = kb[lk]
        l_disp = f"{float(lk):.2f}"
        tbl_k.append(
            f"{l_disp} & {fmt_val(sd['g32']['coverage'],2,True)}\\% & "
            f"{fmt_val(sd['g128_fixed_phys']['coverage'],2,True)}/{fmt_val(sd['g128_fixed_phys']['actual_width'],4)} & "
            f"{fmt_val(sd['g128_fixed_cell']['coverage'],2,True)}/{fmt_val(sd['g128_fixed_cell']['actual_width'],4)} & "
            f"{fmt_val(sb['g64']['coverage'],2,True)}\\% & "
            f"{fmt_val(sb['g256_fixed_phys']['coverage'],2,True)}/{fmt_val(sb['g256_fixed_phys']['actual_width'],4)} & "
            f"{fmt_val(sb['g256_fixed_cell']['coverage'],2,True)}/{fmt_val(sb['g256_fixed_cell']['actual_width'],4)} \\\\"
        )
    tbl_k += [
        r"\midrule",
        r"\multicolumn{7}{l}{\textbf{Held-out Darcy $128^2$ scalar radius extrapolation vs.\ OpRes-CP (five seeds, mean $\pm$ SD):}} \\",
        f"Uniform-CP zero-shot ($32^2\\to 128^2$) & \\multicolumn{{2}}{{r}}{{cov.: {fmt_ms(mill['uniform_zeroshot_cal32']['coverage'],2,True)}\\%}} & \\multicolumn{{2}}{{r}}{{actual width: {fmt_ms(mill['uniform_zeroshot_cal32']['actual_width'],4)}}} & \\multicolumn{{2}}{{r}}{{--}} \\\\",
        f"Millard-style Uniform ($\\hat q(h)$ lin.) & \\multicolumn{{2}}{{r}}{{cov.: {fmt_ms(mill['uniform_millard_lin_h']['coverage'],2,True)}\\%}} & \\multicolumn{{2}}{{r}}{{actual width: {fmt_ms(mill['uniform_millard_lin_h']['actual_width'],4)}}} & \\multicolumn{{2}}{{r}}{{$h^2$: {fmt_val(mill['uniform_millard_h2']['coverage'],2,True)}\\% / {fmt_val(mill['uniform_millard_h2']['actual_width'],4)}}} \\\\",
        f"OpRes-CP zero-shot (Fixed-Cell) & \\multicolumn{{2}}{{r}}{{cov.: {fmt_ms(mill['opres_zeroshot_fixed_cell_cal32']['coverage'],2,True)}\\%}} & \\multicolumn{{2}}{{r}}{{actual width: {fmt_ms(mill['opres_zeroshot_fixed_cell_cal32']['actual_width'],4)}}} & \\multicolumn{{2}}{{r}}{{$+$Millard lin.: {fmt_val(mill['opres_fixed_cell_plus_millard_lin_h']['coverage'],2,True)}\\% / {fmt_val(mill['opres_fixed_cell_plus_millard_lin_h']['actual_width'],4)}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    blocks.append("\n".join(tbl_k))

    # Table D: Stiller-style temporal state-reweighted WCP on Allen-Cahn T=1 -> T=2
    tw = t3_v3["part_b_temporal_wcp"]["summary"]
    tbl_tw = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\caption{Stiller-style~\cite{stiller2026timedependent} temporal state-reweighted WCP on Allen--Cahn $T=1 \to T=2$ (using strictly $T=1$ labeled calibration scores reweighted toward unlabeled $T=2$ states $u(T_2)$ with $C=1$, $\mathrm{cap}=50$, five seeds). Full-grid split-fit state density ratios collapse effective sample size to $\mathrm{ESS}=34.5$ across the 250-sample half, whereas causal operator lifting (\texttt{OpRes-CP}, Table~\ref{tab:opres-horizon}) scales interval magnitude naturally across horizons with $\mathrm{ESS}$ untouched when $T=2$ calibration is available.}",
        r"\label{tab:opres-temporal-wcp}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Method / state features & Protocol & Cov.\ (\%) & Inf.\ (\%) & Fin.\ cov.\ (\%) & Fin.\ width & ESS \\",
        r"\midrule",
        f"Unweighted $T=1$ ($90\\%$) & Zero-shot & {fmt_val(tw['unweighted_t1_90']['coverage'],2,True)} & 0.00 & {fmt_val(tw['unweighted_t1_90']['coverage'],2,True)} & {fmt_val(tw['unweighted_t1_90']['actual_width'],4)} & 500.0 \\\\",
        f"Unweighted $T=1$ ($95\\%$ heuristic) & Zero-shot & {fmt_val(tw['unweighted_t1_95']['coverage'],2,True)} & 0.00 & {fmt_val(tw['unweighted_t1_95']['coverage'],2,True)} & {fmt_val(tw['unweighted_t1_95']['actual_width'],4)} & 500.0 \\\\",
        f"Compact physical summary (3D) & Transductive & {fmt_val(tw['compact_summary_wcp']['transductive']['coverage'],2,True)} & {fmt_val(tw['compact_summary_wcp']['transductive']['infinite_rate'],2,True)} & {fmt_val(tw['compact_summary_wcp']['transductive']['finite_coverage'],2,True)} & {fmt_val(tw['compact_summary_wcp']['transductive']['finite_width'],4)} & {fmt_val(tw['compact_summary_wcp']['transductive']['ess'],1)} \\\\",
        f"Compact physical summary (3D) & Split-fit & {fmt_val(tw['compact_summary_wcp']['split_fit']['coverage'],2,True)} & {fmt_val(tw['compact_summary_wcp']['split_fit']['infinite_rate'],2,True)} & {fmt_val(tw['compact_summary_wcp']['split_fit']['finite_coverage'],2,True)} & {fmt_val(tw['compact_summary_wcp']['split_fit']['finite_width'],4)} & {fmt_val(tw['compact_summary_wcp']['split_fit']['ess'],1)} \\\\",
        f"Full-grid state $u(T)$ ($4098$-D) & Transductive & {fmt_val(tw['full_grid_state_wcp']['transductive']['coverage'],2,True)} & {fmt_val(tw['full_grid_state_wcp']['transductive']['infinite_rate'],2,True)} & {fmt_val(tw['full_grid_state_wcp']['transductive']['finite_coverage'],2,True)} & {fmt_val(tw['full_grid_state_wcp']['transductive']['finite_width'],4)} & {fmt_val(tw['full_grid_state_wcp']['transductive']['ess'],1)} \\\\",
        f"Full-grid state $u(T)$ ($4098$-D) & Split-fit & {fmt_val(tw['full_grid_state_wcp']['split_fit']['coverage'],2,True)} & {fmt_val(tw['full_grid_state_wcp']['split_fit']['infinite_rate'],2,True)} & {fmt_val(tw['full_grid_state_wcp']['split_fit']['finite_coverage'],2,True)} & {fmt_val(tw['full_grid_state_wcp']['split_fit']['finite_width'],4)} & {fmt_val(tw['full_grid_state_wcp']['split_fit']['ess'],1)} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    blocks.append("\n".join(tbl_tw))

    write_both("phase3_rev3_tables.tex", "\n\n".join(blocks) + "\n")
    print("Wrote phase3_rev3_tables.tex")


def wrap_all_cmame_tables() -> None:
    """Wrap all tabular blocks in paper_cmame/tables/*.tex with \\fitcmametable{...} to avoid Overfull hbox."""
    for p in sorted(CMAME_TBL.glob("*.tex")):
        txt = p.read_text(encoding="utf-8")
        if r"\fitcmametable{" in txt:
            continue
        # Wrap only top-level \begin{tabular}...\end{tabular} inside tables
        new_txt = re.sub(
            r"(\\begin\{tabular\}.*?\\end\{tabular\})",
            r"\\fitcmametable{%\n\1%\n}",
            txt,
            flags=re.S,
        )
        p.write_text(new_txt, encoding="utf-8")
    print("Wrapped CMAME tables with \\fitcmametable")


# ---------------------------------------------------------------------------
# 6. Plot fig9_rev3_diagnostics.{pdf,png}
# ---------------------------------------------------------------------------
def plot_fig9(t1_v3: dict, t3_v3: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.35), constrained_layout=True)

    # Panel (a): Kernel resolution scan on Darcy 32->128 and Burgers 64->256
    ax = axes[0]
    lpix_keys = ["0.64", "1", "1.28", "1.5", "2", "2.56", "3", "4"]
    x = np.array([float(k) for k in lpix_keys])
    kd = t3_v3["part_a_kernel_scan"]["darcy_32_to_128"]["summary"]
    kb = t3_v3["part_a_kernel_scan"]["burgers_64_to_256"]["summary"]

    d_phys = [100.0 * kd[k]["g128_fixed_phys"]["coverage"]["mean"] for k in lpix_keys]
    d_cell = [100.0 * kd[k]["g128_fixed_cell"]["coverage"]["mean"] for k in lpix_keys]
    b_phys = [100.0 * kb[k]["g256_fixed_phys"]["coverage"]["mean"] for k in lpix_keys]
    b_cell = [100.0 * kb[k]["g256_fixed_cell"]["coverage"]["mean"] for k in lpix_keys]

    ax.axhline(90.0, color=GRAY, ls="--", lw=1.0, label="90% target")
    ax.axvline(2.0, color=MC_COLOR, ls=":", lw=1.2, label=r"Rule $\ell_{\rm pix,train}=2$")
    ax.plot(x, d_cell, "o-", color=OP_COLOR, lw=1.7, ms=4.5, label=r"Darcy $128^2$ Fix-Cell")
    ax.plot(x, d_phys, "s--", color=OP_COLOR, alpha=0.65, lw=1.4, ms=4.0, label=r"Darcy $128^2$ Fix-Phys")
    ax.plot(x, b_phys, "^-", color=EN_COLOR, lw=1.7, ms=4.5, label=r"Burgers $256$ Fix-Phys")
    ax.plot(x, b_cell, "v--", color=EN_COLOR, alpha=0.65, lw=1.4, ms=4.0, label=r"Burgers $256$ Fix-Cell")
    ax.set_xlabel(r"Calibration pixel kernel width $\ell_{\rm pix,train}$")
    ax.set_ylabel("Zero-shot target coverage (%)")
    ax.set_title(r"(a) $\ell_{\rm pix,train}$ scaling: Fix-Cell vs. Fix-Phys")
    ax.set_ylim(20, 102)
    ax.legend(loc="lower left", frameon=True, framealpha=0.92, edgecolor="none", ncol=1, fontsize=6.4)

    # Panel (b): Temporal damping gamma ablation (AC T=2, AC T=1, Burgers ID)
    ax = axes[1]
    gs = ["0", "0.05", "0.1", "0.2", "0.4"]
    gx = np.arange(len(gs))
    cac = t1_v3["part_c_gamma_ablation"]["allen_cahn"]["summary"]
    cb = t1_v3["part_c_gamma_ablation"]["burgers"]["summary"]

    ac_t2_cov = [100.0 * cac[g]["T2_ID"]["coverage"]["mean"] for g in gs]
    ac_t1reuse = [100.0 * cac[g]["T2_ID"]["t1_95reuse_coverage"]["mean"] for g in gs]
    ac_t2_w = [cac[g]["T2_ID"]["matched90_width"]["mean"] for g in gs]
    b_id_cov = [100.0 * cb[g]["ID"]["coverage"]["mean"] for g in gs]

    ax.axhline(90.0, color=GRAY, ls="--", lw=1.0)
    ax.plot(gx, ac_t2_cov, "o-", color=OP_COLOR, lw=1.8, ms=5, label=r"AC $T=2$ Indep. Cal.")
    ax.plot(gx, b_id_cov, "s-", color=SLATE, lw=1.5, ms=4.5, label="Burgers $64$ ID Cal.")
    ax.plot(gx, ac_t1reuse, "^--", color=EN_COLOR, lw=1.6, ms=5, label=r"AC $T=1$ 95%$\to T=2$")
    ax.set_xticks(gx)
    ax.set_xticklabels(["0.0", "0.05", "0.1*", "0.2*", "0.4"])
    ax.set_xlabel(r"All-mode damping $\gamma$ (*locked default)")
    ax.set_ylabel("Coverage (%)")
    ax.set_ylim(25, 105)
    ax.set_title(r"(b) Temporal damping $\gamma$ sensitivity")

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(gx, ac_t2_w, "D:", color=MC_COLOR, lw=1.4, ms=4.2, label=r"AC $T=2$ m90 width")
    ax2.set_ylabel(r"AC $T=2$ matched-90% width", color=MC_COLOR)
    ax2.tick_params(axis="y", colors=MC_COLOR)
    ax2.set_ylim(0.015, 0.032)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower left", frameon=False, fontsize=6.6)

    # Panel (c): Darcy OOD-B counterexample (Liouville vs 5-Ensemble vs AMG-V2 vs AMG-PCG2)
    ax = axes[2]
    b = t1_v3["part_b_darcy_oodb_amg"]["summary"]["ood_b_fidelity"]
    proxies = ["liouville", "ensemble_std", "amg_pcg2", "amg_v2"]
    plabels = ["Liouville\nDST-II", "5-Member\nEns. std", "AMG-\nPCG2", "AMG-\nV2"]
    px = np.arange(len(proxies))
    w = 0.25
    rho_vals = [b[k]["spearman"]["mean"] for k in proxies]
    pcap_vals = [b[k]["peak_capture_5"]["mean"] for k in proxies]
    rhocross_vals = [b[k]["rho_cross"]["mean"] for k in proxies]

    ax.bar(px - w, rho_vals, width=w, color=SLATE, label=r"Spearman $\rho$")
    ax.bar(px, pcap_vals, width=w, color=OP_COLOR, label="PeakCap@5%")
    ax.bar(px + w, rhocross_vals, width=w, color=EN_COLOR, label=r"$\rho_{\rm cross}$")
    for i, (rv, pv) in enumerate(zip(rho_vals, pcap_vals)):
        ax.text(i - w, rv + 0.02, f"{rv:.2f}", ha="center", va="bottom", fontsize=6.5)
        ax.text(i, pv + 0.02, f"{pv:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(px)
    ax.set_xticklabels(plabels, fontsize=7.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Spatial localization score")
    ax.set_title(r"(c) Darcy $64^2$ OOD-B potential vs. AMG")
    ax.legend(loc="upper left", frameon=False, fontsize=6.8)

    for ext in ("pdf", "png"):
        fig.savefig(OUT_FIG_R3 / f"fig9_rev3_diagnostics.{ext}", bbox_inches="tight")
        fig.savefig(CMAME_FIG / f"fig9_rev3_diagnostics.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig9_rev3_diagnostics.{pdf,png}")


# ---------------------------------------------------------------------------
# 7. Plot CMAME Graphical Abstract (>= 1328 px width, horizontal layout)
# ---------------------------------------------------------------------------
def plot_graphical_abstract() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 4.3), dpi=350)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    # Background title banner
    ax.text(
        50, 38.2,
        "OpRes-CP: Provable Whole-Field Uncertainty via Operator-Preconditioned PDE Residuals",
        ha="center", va="center", fontsize=12.0, fontweight="bold", color=DARK,
    )
    ax.text(
        50, 35.6,
        "Causal Green-function lifting converts noisy PDE residuals into calibrated, grid-invariant error scales from a single frozen surrogate",
        ha="center", va="center", fontsize=8.4, color=GRAY,
    )

    def box(x, y, w, h, fc, ec, title, body, title_color=DARK, lw=1.2):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                           facecolor=fc, edgecolor=ec, linewidth=lw)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h - 2.3, title, ha="center", va="top",
                fontsize=9.0, fontweight="bold", color=title_color)
        ax.text(x + w / 2, y + h / 2 - 1.3, body, ha="center", va="center",
                fontsize=7.6, color=DARK, linespacing=1.32)

    # Left column: Prior UQ failure modes
    box(
        2.0, 18.8, 25.5, 13.8, EN_LIGHT, EN_COLOR,
        "1A. Ensemble Dispersion",
        "Requires 5x training/inference\nDispersive mismatch vs. $e_h$:\nDarcy $\\rho = 0.50$ (3.0x wider bands)",
        title_color=EN_COLOR,
    )
    box(
        2.0, 2.5, 25.5, 13.8, "#F3F4F6", MC_COLOR,
        "1B. Raw-Residual PI-CP",
        "High-freq $\\|A_h\\hat{u} - f\\|$ spikes\n$R_h$ correlates with $O(h^{-2})$ noise;\nmisses dominant low modes",
        title_color=MC_COLOR,
    )

    # Center column: OpRes-CP Causal Mechanism
    box(
        32.5, 5.2, 35.0, 26.2, OP_LIGHT, OP_COLOR,
        "2. Causal Operator-Residual Lifting",
        "Single Frozen Surrogate  $\\hat{u}_h = G_\\theta(a)$\n"
        "-----------------------------------------\n"
        "1. Compute local PDE residual:\n"
        "   $R_h = L_h(\\hat{u}_h) - f_h$\n\n"
        "2. Apply causal preconditioner $B_h \\approx L_h^{-1}$\n"
        "   (Liouville DST-II / Exp. ETDRK / AMG):\n"
        "   $Z_h = B_h R_h \\approx \\hat{u}_h - u_h$\n\n"
        "3. Relative-floored smooth scale:\n"
        "   $\\sigma_h = K_h * \\sqrt{Z_h^2 + (0.1\\,\\mathrm{RMS}(Z_h))^2}$\n"
        "   (1st-order pos. homogeneous -> Theorem 1)",
        title_color=OP_COLOR, lw=1.7,
    )

    # Right column: Provable Whole-Field Outcomes
    box(
        72.5, 18.8, 25.5, 13.8, "#EBF3FA", SLATE,
        "3A. Sharp ID & OOD Bands",
        "2.5x–10.3x narrower than Uniform\n"
        "57%–90% narrower than Val-Fit LSCI\n"
        "Maintains ~90% simultaneous cov.\n"
        "under contrast/viscosity shift (Thm 2)",
        title_color=SLATE,
    )
    box(
        72.5, 2.5, 25.5, 13.8, "#ECFDF5", OP_COLOR,
        "3B. Zero-Shot Transfer",
        "Cross-grid ($32^2 \\to 64^2/128^2$):\n"
        "<1.0 pp coverage drift ($\\ell_{\\mathrm{pix}} \\geq 2$)\n"
        "Temporal $T=1 \\to T=2$ scaling\n"
        "Exact WCP abstention on severe OOD",
        title_color=OP_COLOR,
    )

    # Arrows
    for y_arr in (25.5, 9.5):
        ax.add_patch(FancyArrowPatch((28.3, y_arr), (32.0, y_arr),
                                     arrowstyle="-|>", mutation_scale=13, lw=1.4, color=GRAY))
        ax.add_patch(FancyArrowPatch((68.2, y_arr), (72.0, y_arr),
                                     arrowstyle="-|>", mutation_scale=13, lw=1.5, color=OP_COLOR))

    for ext in ("pdf", "png"):
        fig.savefig(OUT_FIG_R3 / f"graphical_abstract.{ext}", bbox_inches="tight")
        fig.savefig(CMAME_FIG / f"graphical_abstract.{ext}", bbox_inches="tight")
        if _HAS_CMAME:
            fig.savefig(SUB_PKG / f"graphical_abstract.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("Saved graphical_abstract.{pdf,png}")


def collect_manuscript_figures(tex_path: Path) -> list[str]:
    """Recursively collect \\includegraphics stems in exact manuscript appearance order."""
    seen: list[str] = []
    text = tex_path.read_text(encoding="utf-8")
    # Remove LaTeX comments (unescaped %)
    text = re.sub(r"(?<!\\)%.*", "", text)
    tokens = re.finditer(r"\\(input|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}", text)
    for m in tokens:
        cmd, arg = m.group(1), m.group(2).strip()
        if cmd == "input":
            sub = tex_path.parent / (arg if arg.endswith(".tex") else f"{arg}.tex")
            if sub.is_file():
                for item in collect_manuscript_figures(sub):
                    if item not in seen:
                        seen.append(item)
        else:
            stem = Path(arg).name
            if stem not in seen:
                seen.append(stem)
    return seen


def export_ordered_figures() -> None:
    """Export ordered Figure_1..N.{pdf,png} into paper_cmame/submission_package/."""
    main_tex = ROOT / "paper_cmame" / "main.tex"
    if not main_tex.is_file():
        print("[skip] paper_cmame/main.tex not present; skipping ordered manuscript figure export.")
        return
    figs = collect_manuscript_figures(main_tex)
    for idx, fname in enumerate(figs, 1):
        src_pdf = CMAME_FIG / fname
        if not src_pdf.is_file():
            src_pdf = CMAME_FIG / f"{Path(fname).stem}.pdf"
        assert src_pdf.is_file(), f"Missing manuscript figure PDF: {src_pdf}"
        dst_pdf = SUB_PKG / f"Figure_{idx}.pdf"
        shutil.copy2(src_pdf, dst_pdf)

        src_png = CMAME_FIG / f"{src_pdf.stem}.png"
        if not src_png.is_file():
            # Also check results/phase3_figures/ if PNG already exists at 300 dpi
            alt_png = ROOT / "results" / "phase3_figures" / f"{src_pdf.stem}.png"
            if alt_png.is_file():
                shutil.copy2(alt_png, src_png)
            else:
                subprocess.run(
                    ["pdftoppm", "-png", "-r", "300", "-singlefile", str(src_pdf), str(CMAME_FIG / src_pdf.stem)],
                    check=True,
                )
        dst_png = SUB_PKG / f"Figure_{idx}.png"
        shutil.copy2(src_png, dst_png)
        print(f"  Figure_{idx} <- {src_pdf.name}")


def main() -> None:
    t1_v2 = load_json(R2_DIR / "task1_fidelity.json")
    t1_v3 = load_json(R3_DIR / "task1_opnorm_gamma.json")
    t2_v3 = load_json(R3_DIR / "task2_lsci_exact.json")
    t3_v3 = load_json(R3_DIR / "task3_kernel_horizon.json")

    build_rev2_main_tables(t1_v2, t1_v3)
    build_rev_tables(t2_v3)
    build_promoted_tables(t2_v3)
    build_main_tables()
    build_rev3_tables(t1_v3, t3_v3)
    wrap_all_cmame_tables()
    plot_fig9(t1_v3, t3_v3)
    plot_graphical_abstract()
    export_ordered_figures()
    print("ALL REV3 TABLES AND FIGURES GENERATED.")


if __name__ == "__main__":
    main()
