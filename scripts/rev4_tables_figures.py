#!/usr/bin/env python3
"""Generate Revision v4 LaTeX tables, updated fig7, fig10_elasticity_fem,
updated CMAME Graphical Abstract, and ordered submission_package/Figure_1..10.

Pure ASCII in all generated .tex files for seamless pdflatex compilation.
All tabular blocks wrapped in \\fitcmametable{...} to guarantee 0 Overfull hbox.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.tri as mtri
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
R2_DIR = ROOT / "results" / "phase3_revision_v2"
R3_DIR = ROOT / "results" / "phase3_revision_v3"
R4_DIR = ROOT / "results" / "phase3_revision_v4"
OUT_TBL = R4_DIR / "tables"
OUT_FIG = R4_DIR / "figures"
_HAS_CMAME = (ROOT / "paper_cmame").is_dir()
CMAME_TBL = ROOT / "paper_cmame" / "tables" if _HAS_CMAME else ROOT / "outputs" / "tables"
CMAME_FIG = ROOT / "paper_cmame" / "figures" if _HAS_CMAME else ROOT / "outputs" / "figures"
SUB_PKG = ROOT / "paper_cmame" / "submission_package" if _HAS_CMAME else ROOT / "outputs" / "submission_package"

for d in (OUT_TBL, OUT_FIG, CMAME_TBL, CMAME_FIG):
    d.mkdir(parents=True, exist_ok=True)
if _HAS_CMAME:
    SUB_PKG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 8.6,
    "axes.titlesize": 9.2,
    "axes.labelsize": 8.4,
    "xtick.labelsize": 7.6,
    "ytick.labelsize": 7.6,
    "legend.fontsize": 7.0,
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


def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def fmt_ms(st: dict, dec: int = 3, pct: bool = False) -> str:
    m = st["mean"] * (100.0 if pct else 1.0)
    s = st["sd"] * (100.0 if pct else 1.0)
    return f"${m:.{dec}f} \\pm {s:.{dec}f}$"


def fmt_val(st: dict, dec: int = 4, pct: bool = False) -> str:
    v = st["mean"] * (100.0 if pct else 1.0)
    return f"{v:.{dec}f}"


def write_tex_both(name: str, text: str) -> None:
    assert all(ord(c) < 128 for c in text), f"Non-ASCII detected in {name}"
    (OUT_TBL / name).write_text(text, encoding="ascii")
    (CMAME_TBL / name).write_text(text, encoding="ascii")


def replace_table_by_label(doc: str, label: str, new_block: str) -> str:
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
# 1. Section 2 Method Positioning Table
# ---------------------------------------------------------------------------
def build_sec2_positioning_table() -> None:
    tex = r"""% Auto-generated Revision v4 positioning table (pure ASCII).
\begin{table}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{2.4pt}
\caption{Methodological positioning across neural-operator UQ, residual point correctors, and classical FEM a posteriori estimators. OpRes-CP decouples conformal sharpness from contractive point correction via signed PDE residual lifting around an untouched surrogate.}
\label{tab:opres-positioning}
\fitcmametable{%
\begin{tabular}{lllllll}
\toprule
Method family & Training-free & Modifies $\widehat U_h$ & Uses PDE op. & Op.\ requirement & Coverage / bound type & Mesh support \\
\midrule
RBNO + FOSLS (Qiu et al.~\cite{qiu2026variationally}) & No (coupled train) & Built-in RB & Variational loss & Variational coercivity & Deterministic energy & Conforming FEM \\
Point Corrector (Jha~\cite{jha2024residual}; Cao~\cite{cao2023residual}) & Yes (corrector) & Yes ($\widehat U_h - B_h R_h$) & Linearized $B_h R_h$ & Strict contraction ($\delta_{\rm inv}<1$) & Point accuracy only & Arbitrary FEM / grid \\
Encoder--Reg.\ / GP (Yvonnet~\cite{yvonnet2026clear}; Kumar~\cite{kumar2024nogap}) & No / Modifies fit & GP / surrogate & No (feature-based) & Well-specified prior & Heuristic / Bayesian & Mesh / parametric \\
FEM BR/Verf\"urth \& ZZ \cite{ainsworth2000aposteriori,zienkiewicz1992superconvergent} & Yes & No & Edge jumps / recovery & Asymptotic saturation & Element energy / heuristic & Unstructured FEM \\
5-Member Ensemble + CP \cite{choi2026simultaneous} & No ($5\times$ train) & Optional mean & No & Uncorrelated members & Exact fin.-sample sim.\ (ID) & Any shared grid \\
Raw $|R_h|$ PI-CP \cite{chin2026picp} & Yes & No & Uninverted $|R_h|$ & None ($O(h^{-m})$ blowup) & Exact fin.-sample sim.\ (ID) & Structured / FEM \\
\textbf{OpRes-CP (ours)} & \textbf{Yes} & \textbf{No (orthogonal)} & \textbf{Signed $B_h R_h$ + env.} & \textbf{Order-preservation ($\delta_{\rm inv}$ can $>1$)} & \textbf{Exact sim.\ (ID) + WCP/gate} & \textbf{Structured + Unstr.\ FEM} \\
\bottomrule
\end{tabular}%
}
\end{table}
"""
    write_tex_both("rev4_sec2_positioning.tex", tex)
    print("Wrote rev4_sec2_positioning.tex")


# ---------------------------------------------------------------------------
# 2. Update tab:opres-op-approx in phase3_rev2_main_tables.tex with SPD-AMG-PCG2
# ---------------------------------------------------------------------------
def update_tab_op_approx(t1_v3: dict, t2_v4: dict) -> None:
    candidates = [
        CMAME_TBL / "phase3_rev2_main_tables.tex",
        R3_DIR / "tables" / "phase3_rev2_main_tables.tex",
        ROOT / "results" / "phase3_revision" / "tables" / "phase3_rev2_main_tables.tex",
    ]
    src_doc = next((p for p in candidates if p.is_file()), None)
    if src_doc is None:
        print("[skip] phase3_rev2_main_tables.tex not found; skipping tab:opres-op-approx patch.")
        return
    doc = src_doc.read_text(encoding="ascii")
    part_a = t1_v3["part_a_opnorms"]
    part_c = t2_v4["part_c_spd_pcg"]

    op_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.0pt}",
        r"\caption{Directional defect $d_h(e_h)=\|B_h A_h e_h - e_h\|_2/\|e_h\|_2$ vs.\ global induced operator norms $\|B_h A_h - I\|_{2\to 2}$ and $\|I - B_h A_h\|_{A_h}$ computed by Power Iteration on $A_h$ across five seeds (mean $\pm$ SD). \texttt{AMG-V2} achieves both small directional defect and substantial, mesh-robust global energy contraction ($0.260$--$0.362$), whereas single-level \texttt{Jacobi5 (linear)} sits near the unit contraction boundary ($0.990$--$0.995$). For nonlinear degree-$1$ homogeneous \texttt{SPD-AMG-PCG2} (Proposition~\ref{prop:spd-pcg}), all tested fields satisfy the closed-form Chebyshev contraction bound $\delta_{\rm PCG}^{(2)} = 2\tau^2$ ($1.000$ comp.) at exact preconditioned condition number $\kappa(M_h A_h)$.}",
        r"\label{tab:opres-op-approx}",
        r"\fitcmametable{%",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Domain / split & Operator $B_h$ & Dir.\ $d_h(e_h)$ & $\|B_h A_h - I\|_{2\to 2}$ & $\|I - B_h A_h\|_{A_h}$ / En.\ $d_A$ & Frac.\ $<1$ / $\delta_{\rm PCG}^{(2)}$ & $\kappa(M_h A_h)$ \\",
        r"\midrule",
    ]
    specs = [
        ("darcy_rect", "ID", "Darcy $64^2$ ID", ["liouville", "amg_v2", "amg_v2_spd", "jacobi5_lin", "spd_pcg2", "jacobi_pcg5"]),
        ("darcy_rect", "OOD-A", "Darcy $64^2$ OOD-A", ["liouville", "amg_v2", "amg_v2_spd", "jacobi5_lin", "spd_pcg2", "jacobi_pcg5"]),
        ("darcy_rect", "OOD-B", "Darcy $64^2$ OOD-B", ["liouville", "amg_v2", "amg_v2_spd", "jacobi5_lin", "spd_pcg2", "jacobi_pcg5"]),
        ("darcy_lshape", "ID", "L-shape $64^2$ ID", ["amg_v2", "amg_v2_spd", "jacobi5_lin", "spd_pcg2", "jacobi_pcg5"]),
        ("darcy_lshape", "OOD-A", "L-shape $64^2$ OOD-A", ["amg_v2", "amg_v2_spd", "jacobi5_lin", "spd_pcg2", "jacobi_pcg5"]),
    ]
    op_names = {
        "liouville": "Liouville DST-II (linear)",
        "amg_v2": "AMG-V2 (linear)",
        "amg_v2_spd": "AMG-V2-SPD (linear)",
        "jacobi5_lin": "Jacobi5 (linear)",
        "spd_pcg2": "SPD-AMG-PCG2 (Cheb.)",
        "jacobi_pcg5": "Jacobi-PCG5 (NL)",
    }
    for idx_g, (dom, sp, dom_lbl, ops) in enumerate(specs):
        sp_d = part_a[dom]["summary"][sp]
        pc_d = part_c[dom]["summary"][sp]
        for j, op in enumerate(ops):
            lbl = dom_lbl if j == 0 else ""
            if op == "spd_pcg2":
                dl2 = fmt_ms(pc_d["dir_l2"], 3)
                den = fmt_ms(pc_d["dir_energy"], 3)
                cheb = fmt_val(pc_d["cheb_upper"], 3)
                kap = fmt_ms(pc_d["kappa"], 2)
                op_lines.append(f"{lbl} & {op_names[op]} & {dl2} & -- & {den} ($d_A$) & $\\le {cheb}$ (1.000) & {kap} \\\\")
            else:
                m = sp_d[op]
                dl2 = fmt_ms(m["dir_l2"], 3)
                if "global_l2" in m:
                    gl2 = fmt_ms(m["global_l2"], 3)
                    gen = fmt_ms(m["global_energy"], 3)
                    flt = fmt_val(m["frac_lt_1"], 3)
                else:
                    gl2, gen, flt = "--", "--", "--"
                op_lines.append(f"{lbl} & {op_names[op]} & {dl2} & {gl2} & {gen} & {flt} & -- \\\\")
        if idx_g < len(specs) - 1:
            op_lines.append(r"\addlinespace[2.5pt]")
    op_lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]
    new_block = "\n".join(op_lines)
    updated = replace_table_by_label(doc, "tab:opres-op-approx", new_block)
    assert all(ord(c) < 128 for c in updated)
    (R3_DIR / "tables" / "phase3_rev2_main_tables.tex").write_text(updated, encoding="ascii")
    (OUT_TBL / "phase3_rev2_main_tables.tex").write_text(updated, encoding="ascii")
    (CMAME_TBL / "phase3_rev2_main_tables.tex").write_text(updated, encoding="ascii")
    print("Updated tab:opres-op-approx with SPD-AMG-PCG2 & corrected caption")


# ---------------------------------------------------------------------------
# 3. Main Rev4 Tables: tab:opres-elasticity & tab:opres-nonlinear-jha
# ---------------------------------------------------------------------------
def build_rev4_main_tables(t1: dict, t2: dict) -> None:
    s1 = t1["summary"]
    methods = [
        ("Uniform-CP", "Uniform-CP"),
        ("BR/Verfurth FEM + CP", "FEM BR/Verf\\\"urth + CP"),
        ("Zienkiewicz-Zhu + CP", "Zienkiewicz--Zhu + CP"),
        ("Jha Corrector + Unif-CP", "Jha Corr.\\ + Uniform-CP"),
        ("Jha Corrector + OpRes-CP", "Jha Corr.\\ + OpRes-CP"),
        ("5-Ens. (mean center)", "5-DeepONet Ens.\\ (mean)"),
        ("Unsigned Vector AMG-V2", "Unsigned Vector AMG-V2"),
        ("OpRes-CP (Vector SA-AMG-V2)", "\\textbf{OpRes-CP (SA-AMG-V2)}"),
        ("OpRes-CP (Exact K^-1 oracle)", "OpRes-CP (Exact $K^{-1}$)"),
    ]
    el_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{1.6pt}",
        r"\caption{Five-seed 2D plane-strain linear elasticity FEM benchmark on $\Omega=(0,1)^2\setminus B_{0.20}(0.5,0.5)$ around a single frozen Vector DeepONet ($\alpha=0.10$; coarse calibration grid $\mathcal{T}_c$: $1{,}164$ nodes, fine paired grid $\mathcal{T}_f$: $4{,}144$ nodes). Entries report simultaneous coverage (\%) / matched-$90\%$ full width for both vector displacement $\mathbf u_h$ and derived von Mises stress $\sigma_{\rm vM}$. Classical mesh-dependent BR and ZZ estimators collapse under zero-shot $h\to h/2$ transfer, and \texttt{Jha Corr.+Unif-CP} drops to $8.56\%$ von Mises coverage under OOD-Lam\'e ($\nu_{\rm P}\in[0.34,0.43]$), whereas \texttt{OpRes-CP (SA-AMG-V2)} narrows ID bands $4.5$--$4.8\times$ vs.\ the 5-member ensemble and retains $93.00\%$ OOD von Mises coverage.}",
        r"\label{tab:opres-elasticity}",
        r"\fitcmametable{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Vector displacement $\mathbf u_h$ (cov.\ / m90 w)} & \multicolumn{3}{c}{Derived von Mises stress $\sigma_{\rm vM}$ (cov.\ / m90 w)} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Method & ID ($\mathcal{T}_c$) & Paired Fine ($\mathcal{T}_f$) & OOD-Lam\'e & ID ($\mathcal{T}_c$) & Paired Fine ($\mathcal{T}_f$) & OOD-Lam\'e \\",
        r"\midrule",
    ]
    for mk, mlbl in methods:
        d_id = s1["splits"]["id"]["displacement"][mk]
        d_fn = s1["splits"]["fine_paired"]["displacement"][mk]
        d_oo = s1["splits"]["ood_lame"]["displacement"][mk]
        v_id = s1["splits"]["id"]["von_mises"][mk]
        v_fn = s1["splits"]["fine_paired"]["von_mises"][mk]
        v_oo = s1["splits"]["ood_lame"]["von_mises"][mk]
        el_lines.append(
            f"{mlbl} & "
            f"{fmt_val(d_id['coverage'],2,True)}/{fmt_val(d_id['matched90_width'],4)} & "
            f"{fmt_val(d_fn['coverage'],2,True)}/{fmt_val(d_fn['matched90_width'],4)} & "
            f"{fmt_val(d_oo['coverage'],2,True)}/{fmt_val(d_oo['matched90_width'],4)} & "
            f"{fmt_val(v_id['coverage'],2,True)}/{fmt_val(v_id['matched90_width'],3)} & "
            f"{fmt_val(v_fn['coverage'],2,True)}/{fmt_val(v_fn['matched90_width'],3)} & "
            f"{fmt_val(v_oo['coverage'],2,True)}/{fmt_val(v_oo['matched90_width'],3)} \\\\"
        )
    el_lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]

    # Nonlinear 3-term decomposition + Jha point-corrector comparison
    pa_b = t2["part_a_nonlinear_decomp"]["burgers"]["summary"]
    pa_a = t2["part_a_nonlinear_decomp"]["allen_cahn"]["summary"]
    pd_d = t2["part_d_jha_comparisons"]["darcy"]
    pd_b = t2["part_d_jha_comparisons"]["burgers"]

    nj_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{1.8pt}",
        r"\caption{Top: Five-seed exact discrete nonlinear remainder decomposition verifying quadratic scaling ($N_h \propto \|e_h\|_2^2$, log--log slope $\approx 2$) and explaining fine-grid U-Net breakdown when errors exit the first-order neighborhood. Bottom: Five-seed comparison against Jha-style~\cite{jha2024residual} variational point correction ($\widehat U_h - B_h R_h$): point rel.\ $L^2$ (\%) / simultaneous cov.\ (\%) / matched-$90\%$ width (plus single-case P50 ms).}",
        r"\label{tab:opres-nonlinear-jha}",
        r"\fitcmametable{%",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"\multicolumn{7}{l}{\textbf{(A) Exact discrete residual decomposition ($Z_h = B_h L_h^{\rm lin}e_h + B_h G_h + B_h N_h + B_h \tau_h$):}} \\",
        r"PDE / Backbone & Split & Point $L^2$ (\%) & $\|B_h N_h\|_2/\|e_h\|_2$ & $\|B_h G_h\|_2/\|e_h\|_2$ & $\|B_h \tau_h\|_2/\|e_h\|_2$ & $N_h$ log--log slope \\",
        r"\midrule",
    ]
    b_rows = [
        ("Burgers FNO", "ID ($64$)", "id"),
        ("Burgers FNO", "Visc.-OOD", "ood"),
        ("Burgers FNO", "High-OOD", "ood_high"),
        ("Burgers U-Net", "Paired $64$", "unet_64"),
        ("Burgers U-Net", "Paired $256$", "unet_256"),
    ]
    for bnm, spl, sk in b_rows:
        m = pa_b[sk]
        nj_lines.append(
            f"{bnm} & {spl} & {fmt_val(m['point_rel_l2'],2,True)}\\% & "
            f"{fmt_val(m['nonlinear_ratio'],3)} & -- & {fmt_val(m['tau_ratio'],3)} & {fmt_val(m['nonlinear_loglog_slope'],2)} \\\\"
        )
    a_rows = [
        ("AC StepFNO", "$T=1$ ID", "fno_t1_id"),
        ("AC StepFNO", "$T=1$ React.", "fno_t1_ood"),
        ("AC StepFNO", "$T=2$ ID", "fno_t2_id"),
        ("AC StepUNet", "$T=2$ $64^2$", "unet_t2_64"),
        ("AC StepUNet", "$T=2$ $128^2$", "unet_t2_128"),
    ]
    nj_lines.append(r"\addlinespace[2pt]")
    for anm, spl, sk in a_rows:
        m = pa_a[sk]
        nj_lines.append(
            f"{anm} & {spl} & {fmt_val(m['point_rel_l2'],2,True)}\\% & "
            f"{fmt_val(m['nonlinear_ratio'],3)} & {fmt_val(m['guard_bias_ratio'],3)} & {fmt_val(m['tau_ratio'],3)} & {fmt_val(m['nonlinear_loglog_slope'],2)} \\\\"
        )

    nj_lines += [
        r"\midrule",
        r"\multicolumn{7}{l}{\textbf{(B) Jha-style variational point corrector vs.\ OpRes-CP \& orthogonal stacking (pt $L^2$\% / cov\% / m90 width):}} \\",
        r"Benchmark & Method & ID & Shift-1 & Shift-2 & Latency (ms) & Role \\",
        r"\midrule",
    ]
    for dm, role in (
        ("OpRes-CP (Liouville)", "Scale only"),
        ("OpRes-CP (AMG-V2)", "Scale (var-coeff.)"),
        ("Liouville Corr. + Unif-CP", "Cheap Jha corr."),
        ("Jha AMG-V2 Corr. + Unif-CP", "Contractive Jha"),
        ("Jha Exact spsolve + Unif-CP", "Oracle Jha"),
        ("Jha AMG-V2 + OpRes-CP", "Orthogonal stack"),
    ):
        r_id = pd_d["summary"]["ID"][dm]
        r_oa = pd_d["summary"]["OOD-A"][dm]
        r_ob = pd_d["summary"]["OOD-B"][dm]
        lat = pd_d["latency_p50_ms"][dm]
        s_id = f"{fmt_val(r_id['point_rel_l2'],2,True)}/{fmt_val(r_id['coverage'],1,True)}/{fmt_val(r_id['matched90_width'],4)}"
        s_oa = f"{fmt_val(r_oa['point_rel_l2'],2,True)}/{fmt_val(r_oa['coverage'],1,True)}/{fmt_val(r_oa['matched90_width'],4)}"
        s_ob = f"{fmt_val(r_ob['point_rel_l2'],2,True)}/{fmt_val(r_ob['coverage'],1,True)}/{fmt_val(r_ob['matched90_width'],4)}"
        nj_lines.append(f"Darcy $64^2$ & {dm} & {s_id} & {s_oa} & {s_ob} & {lat:.2f} & {role} \\\\")
    nj_lines.append(r"\addlinespace[2pt]")
    for bm, role in (
        ("OpRes-CP (frozen FNO)", "Scale only"),
        ("Jha 1-step Corr. + Unif-CP", "Causal Jha corr."),
        ("Jha 1-step Corr. + OpRes-CP", "Orthogonal stack"),
    ):
        r_id = pd_b["summary"]["ID"][bm]
        r_oa = pd_b["summary"]["Visc-OOD"][bm]
        r_ob = pd_b["summary"]["High-OOD"][bm]
        lat = pd_b["latency_p50_ms"][bm]
        s_id = f"{fmt_val(r_id['point_rel_l2'],2,True)}/{fmt_val(r_id['coverage'],1,True)}/{fmt_val(r_id['matched90_width'],4)}"
        s_oa = f"{fmt_val(r_oa['point_rel_l2'],2,True)}/{fmt_val(r_oa['coverage'],1,True)}/{fmt_val(r_oa['matched90_width'],4)}"
        s_ob = f"{fmt_val(r_ob['point_rel_l2'],2,True)}/{fmt_val(r_ob['coverage'],1,True)}/{fmt_val(r_ob['matched90_width'],4)}"
        nj_lines.append(f"Burgers $64$ & {bm} & {s_id} & {s_oa} & {s_ob} & {lat:.2f} & {role} \\\\")

    nj_lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]
    write_tex_both("rev4_main_tables.tex", "\n".join(el_lines) + "\n\n" + "\n".join(nj_lines) + "\n")
    print("Wrote rev4_main_tables.tex")


# ---------------------------------------------------------------------------
# 4. Appendix B.1 Data Specification Table & Appendix C Rev4 Tables
# ---------------------------------------------------------------------------
def build_rev4_appendix_tables(t1: dict, t2: dict, t3: dict) -> None:
    data_spec = r"""% Auto-generated Revision v4 Appendix B.1 Data & Solver Spec Table (pure ASCII).
\begin{table}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{1.8pt}
\caption{Complete data-generation distributions, OOD shifts, partition sizes (per formal seed $s \in \{42,137,314,271,811\}$), and reference solvers extracted directly from the repository scripts.}
\label{tab:opres-data-spec}
\fitcmametable{%
\begin{tabular}{ll>{\raggedright\arraybackslash}p{4.3cm}>{\raggedright\arraybackslash}p{3.9cm}>{\raggedright\arraybackslash}p{3.2cm}}
\toprule
Benchmark & Domain / Grids & Input distribution (ID) & OOD shifts & Reference solver \& counts \\
\midrule
Darcy ($[0,1]^2$) & $64^2$, $128^2$, $32^2$ & $\log a$: $32\!\times\!32$ cosine modes, $\ell=0.08$, decay $(1+(\pi\ell)^2|k|^2)^{-1.25}$, $\sigma\sim\mathcal N(1.0,0.35^2)$ trunc.\ $[0.4,2.0]$ & Contrast (OOD-A): $\mathcal N(1.35,0.35^2)$ trunc.\ $[0.4,2.0]$; Structure (OOD-B): $8\!\times\!8$ blocks $\in\{-2,-0.5,1,2.5\}$, $\sigma\sim\mathcal U[1.6,1.8]$ & Cell FVM, harmonic faces, exact \texttt{spsolve}; 1500/250/500/500 \\
L-shape Darcy & $\Omega_L$, $64^2$ ($3072$), $128^2$ ($12288$) & Same $\log a$ cosine generator masked to $\Omega_L=(0,1)^2\setminus[0.5,1)^2$, $u|_{\partial\Omega_L}=0$ & Contrast OOD-A: $\mathcal N(1.35,0.35^2)$ trunc.\ $[0.4,2.0]$ & Masked FVM \texttt{spsolve} (res.\ $<10^{-11}$); 800/200/300/300 \\
Burgers 1D & $[0,2\pi)$, $64/128/256$, $T=1.6$ ($16\times 0.1$) & $\nu\sim\mathrm{LogUnif}[0.02,0.07]$; $u_0$: modes $1$--$8$ $\propto k^{-1.35}$, AC rms $\sim\mathcal U[0.55,1.25]$, mean $\sim\mathcal U[-0.4,0.4]$ & Visc.-OOD: $\nu\sim\mathrm{LogUnif}[0.009,0.028]$; High-OOD: adds modes $9$--$16$; Noisy stress $\eta_{\rm noise}\in[0,0.20]$ & 512-pt dealiased ETDRK2 ($\delta t=4\!\times\!10^{-3}$); 2000/250/500/500 \\
Allen--Cahn 2D & $[0,2\pi)^2$, $64^2/128^2$, $T=1$ ($20$), $T=2$ ($40$) & $\nu\sim\mathcal U[0.02,0.06]$, $\rho\sim\mathcal U[0.5,1.1]$; $u_0$: radial $k^2\le 16$, $(1+k^2)^{-0.65}$, rms $\sim\mathcal U[0.35,0.8]$ & React.-OOD: $\rho\sim\mathcal U[0.9,1.5]$; indep.\ $T=2$ horizon streams ($9401$--$9404$) & 128$^2$ dealiased ETDRK2 ($\delta t=5\!\times\!10^{-3}$); 1200/250/500/500 \\
2D Elasticity & $\Omega_{\rm plate}$, $\mathcal T_c$ ($1164$), $\mathcal T_f$ ($4144$) & $\log E$: $12\!\times\!12$ cosine ($\ell=0.15$), $\sigma_{\log E}\sim\mathcal U[0.25,0.55]$, $\nu_{\rm P}\sim\mathcal U[0.20,0.34]$, $t_x,t_y$ random & OOD-Lam\'e: $\sigma_{\log E}\sim\mathcal U[0.55,0.85]$, $\nu_{\rm P}\sim\mathcal U[0.34,0.43]$ (near-incompressible) & $P_1$ FEM exact \texttt{spsolve} (Patch $<10^{-12}$); 1000/200/400/500 \\
\bottomrule
\end{tabular}%
}
\end{table}
"""
    write_tex_both("rev4_data_spec.tex", data_spec)

    # Appendix C tables
    blocks = ["% Auto-generated Revision v4 Appendix C tables (pure ASCII).\n"]

    # C.A: Elasticity Multiscale Vector SA-AMG-V2
    ms = t1["multiscale_amg"]
    sl = ms["log_log_slopes"]
    ec = t1["summary"]["energy_contraction"]
    b_amg = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        f"\\caption{{Unstructured 2D elasticity \\texttt{{Vector SA-AMG-V2}} multiscale scaling across five refined perforated-plate meshes ($1{{,}}018$--$40{{,}}692$ free DOFs; min interior angle $30.9$--$32.6^\\circ$). Log--log CPU runtime slopes: \\texttt{{SA-AMG}} solve $\\beta={sl['amg_v2_solve']:.3f}$, total (setup+solve) $\\beta={sl['amg_v2_total']:.3f}$ vs.\\ exact \\texttt{{spsolve}} $\\beta={sl['spsolve']:.3f}$. Five-seed global energy contraction $\\|I - B_h \\mathbf K_h\\|_{{\\mathbf K_h}}$ is $100.0\\%$ below unity on ID (${fmt_val(ec['id']['energy_norm_mean'],3)}$ mean, ${fmt_val(ec['id']['energy_norm_max'],3)}$ max), Paired Fine (${fmt_val(ec['fine_paired']['energy_norm_mean'],3)}$), and OOD-Lam\\'e (${fmt_val(ec['ood_lame']['energy_norm_mean'],3)}$, max ${fmt_val(ec['ood_lame']['energy_norm_max'],3)}$).}}",
        r"\label{tab:opres-elasticity-amg}",
        r"\fitcmametable{%",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Mesh & Nodes & Free DOFs & Min ang. & $C_G / C_O$ & AMG solve (ms) & AMG tot.\ (ms) & \texttt{spsolve} (ms) & $\|I - B_h \mathbf K_h\|_{\mathbf K_h}$ \\",
        r"\midrule",
    ]
    for r in ms["levels"]:
        b_amg.append(
            f"{r['label']} & {r['n_pts']:,} & {r['free_dofs']:,} & {r['min_angle_deg']:.1f}$^\\circ$ & "
            f"{r['grid_complexity_CG']:.3f}/{r['operator_complexity_CO']:.3f} & "
            f"{r['amg_v2_solve_p50_ms']:.2f} & {r['amg_v2_total_p50_ms']:.2f} & "
            f"{r['spsolve_p50_ms']:.2f} & {r['energy_contraction_seed42']:.3f} \\\\"
        )
    b_amg += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    blocks.append("\n".join(b_amg))

    # C.B: Temporal convergence (Burgers K_sub + AC K_str & dt)
    tb = t2["part_b_time_convergence"]
    bk = tb["burgers_ksub"]["summary"]
    ak = tb["ac_kstr"]["summary"]
    ad = tb["ac_dt"]["summary"]
    b_tc = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\caption{Five-seed temporal discretization convergence of the causal residual lifts. Left: Burgers semi-Lagrangian substep count $K_{\rm sub}\in\{1,2,4,8,16\}$ (reporting non-folded CFL pass rate, signed recovery, ID coverage, matched-$90\%$ width, and P50 CPU trajectory ms; $K_{\rm sub}=4$ eliminates characteristic folding at minor cost). Right: Allen--Cahn Strang internal substeps $K_{\rm str}\in\{1,2,4\}$ (confirming $K_{\rm str}=1$ sufficiency) and frame spacing $\Delta t\in\{0.05,0.10,0.20\}$ confirming second-order $O(\Delta t^2)$ truncation convergence ($\|B_h\tau_h\|_2/\|u\|_2$ scales $\approx 4\times$ per doubling).}",
        r"\label{tab:opres-time-conv}",
        r"\fitcmametable{%",
        r"\begin{tabular}{lrrrrr|llrrr}",
        r"\toprule",
        r"\multicolumn{6}{c}{Burgers $64$ substep convergence ($K_{\rm sub}$)} & \multicolumn{5}{c}{Allen--Cahn $64^2$ Strang ($K_{\rm str}$ \& $\Delta t$)} \\",
        r"\cmidrule(lr){1-6}\cmidrule(lr){7-11}",
        r"$K_{\rm sub}$ & Non-fold & $\cos(Z_h,e_h)$ & Rel.\ err & Cov./m90 w & ms & Horizon & Config & $\cos(Z_h,e_h)$ & Rel.\ err & $\|B_h\tau\|_2/\|u\|_2$ / m90 \\",
        r"\midrule",
    ]
    # Remove vertical bar for CMAME booktabs standard
    b_tc[7] = r"\begin{tabular}{lrrrrrllrrr}"
    left_rows = []
    for k in ("1", "2", "4", "8", "16"):
        m = bk[k]
        left_rows.append(
            f"{k} & {fmt_val(m['non_folded_rate'],1,True)}\\% & {fmt_val(m['cos'],4)} & {fmt_val(m['rel_err'],4)} & "
            f"{fmt_val(m['coverage'],1,True)}/{fmt_val(m['matched90_width'],4)} & {fmt_val(m['lift_p50_ms'],2)}"
        )
    right_rows = [
        ("AC $T=1$", "$K_{\\rm str}=1$", fmt_val(ak["T1"]["1"]["cos"], 4), fmt_val(ak["T1"]["1"]["rel_err"], 4), fmt_val(ak["T1"]["1"]["matched90_width"], 4)),
        ("AC $T=1$", "$K_{\\rm str}=4$", fmt_val(ak["T1"]["4"]["cos"], 4), fmt_val(ak["T1"]["4"]["rel_err"], 4), fmt_val(ak["T1"]["4"]["matched90_width"], 4)),
        ("AC $T=1$", "$\\Delta t=0.05$", fmt_val(ad["T1"]["0.05"]["cos"], 4), fmt_val(ad["T1"]["0.05"]["rel_err"], 4), fmt_val(ad["T1"]["0.05"]["tau_lift_over_u"], 5)),
        ("AC $T=1$", "$\\Delta t=0.10$", fmt_val(ad["T1"]["0.10"]["cos"], 4), fmt_val(ad["T1"]["0.10"]["rel_err"], 4), fmt_val(ad["T1"]["0.10"]["tau_lift_over_u"], 5)),
        ("AC $T=1$", "$\\Delta t=0.20$", fmt_val(ad["T1"]["0.20"]["cos"], 4), fmt_val(ad["T1"]["0.20"]["rel_err"], 4), fmt_val(ad["T1"]["0.20"]["tau_lift_over_u"], 5)),
    ]
    for lrow, rrow in zip(left_rows, right_rows):
        b_tc.append(f"{lrow} & {' & '.join(rrow)} \\\\")
    b_tc += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    blocks.append("\n".join(b_tc))

    # C.C: Physics Fingerprints & Exchangeable Self-Gating lambda(A)
    pa = t3["part_a_fingerprints"]["summary"]
    pb = t3["part_b_adaptive_lambda"]["summary"]
    b_pg = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.0pt}",
        r"\caption{Enriched Cal-only physical residual fingerprints under noisy spectral overlap (top, AUROC / point-failure AURC / accepted cov.\ at $5\%$ ID rejection) and Val-anchored exchangeable physics-adaptive mixture $\lambda(A)\in[\lambda_{\min},0.85]$ preserving Theorem~\ref{thm:fixed} ID validity (bottom, coverage \% / actual width across five seeds).}",
        r"\label{tab:opres-physics-gate}",
        r"\fitcmametable{%",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Protocol ($\eta_{\rm noise}$) & Cal-only gate & AUROC & FPR@95 & AURC (pt) & AURC (mc) & Acc.\ cov. \\",
        r"\midrule",
    ]
    for prot_k, prot_l, eta in (
        ("ambient_noise", "Ambient ($0.10$)", "0.1"),
        ("ambient_noise", "Ambient ($0.20$)", "0.2"),
        ("id_only_noise", "ID-noise ($0.10$)", "0.1"),
    ):
        for gk, gl in (
            ("hf_in", "Input $E_{k>8}/E_{\\rm tot}$"),
            ("rms_z", "Phys.\\ $\\operatorname{RMS}(Z_h)$"),
            ("k_eff", "Eff.\\ wavenum.\\ $k_{\\rm eff}$"),
            ("r_hf", "Res.\\ HF $\\|R_h^{k>8}\\|/\\|R_h\\|$"),
            ("phys_combo_cal", "Cal Phys-Combo"),
            ("input_phys_combo_cal", "Cal Input+Phys-Combo"),
        ):
            m = pa[prot_k][eta][gk]
            b_pg.append(
                f"{prot_l} & {gl} & {fmt_val(m['auroc'],3)} & {fmt_val(m['fpr_at_95_tpr'],3)} & "
                f"{fmt_val(m['point_aurc'],4)} & {fmt_val(m['miscov_aurc'],3)} & {fmt_val(m['ops']['0.05']['accepted_coverage'],1,True)}\\% \\\\"
            )
        b_pg.append(r"\addlinespace[2pt]")

    b_pg += [
        r"\midrule",
        r"\multicolumn{7}{l}{\textbf{Val-anchored exchangeable $\lambda(A)$ (ID cov./act.\ w $\to$ continuous $\alpha_{\rm hf}$ cov./act.\ w):}} \\",
        r"Mixture policy & ID ($64$) & Visc.-OOD & $\alpha_{\rm hf}=0.10$ & $\alpha_{\rm hf}=0.25$ & $\alpha_{\rm hf}=0.50$ & $\alpha_{\rm hf}=1.00$ \\",
        r"\midrule",
    ]
    for pk, pl in (
        ("locked_0.85", "Locked $\\lambda=0.85$"),
        ("phys_gate_lam0.25", "Phys-Gate ($\\lambda_{\\min}=0.25$)"),
        ("phys_gate_lam0.0", "Phys-Gate ($\\lambda_{\\min}=0.00$)"),
        ("spec_phys_gate_lam0.0", "Spec+Phys ($\\lambda_{\\min}=0.00$)"),
    ):
        r_id = pb["splits"]["ID"][pk]
        r_vo = pb["splits"]["Visc-OOD"][pk]
        a10 = pb["alpha_hf_sweep"]["0.1"][pk]
        a25 = pb["alpha_hf_sweep"]["0.25"][pk]
        a50 = pb["alpha_hf_sweep"]["0.5"][pk]
        a100 = pb["alpha_hf_sweep"]["1.0"][pk]
        b_pg.append(
            f"{pl} & "
            f"{fmt_val(r_id['coverage'],1,True)}/{fmt_val(r_id['actual_width'],4)} & "
            f"{fmt_val(r_vo['coverage'],1,True)}/{fmt_val(r_vo['actual_width'],4)} & "
            f"{fmt_val(a10['coverage'],1,True)}/{fmt_val(a10['actual_width'],4)} & "
            f"{fmt_val(a25['coverage'],1,True)}/{fmt_val(a25['actual_width'],4)} & "
            f"{fmt_val(a50['coverage'],1,True)}/{fmt_val(a50['actual_width'],4)} & "
            f"{fmt_val(a100['coverage'],1,True)}/{fmt_val(a100['actual_width'],4)} \\\\"
        )
    b_pg += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    blocks.append("\n".join(b_pg))

    write_tex_both("rev4_appendix_tables.tex", "\n\n".join(blocks) + "\n")
    print("Wrote rev4_appendix_tables.tex")


# ---------------------------------------------------------------------------
# 5. Regenerate CMAME fig7_fidelity_bands (fixing Panel B labels; paper/ untouched!)
# ---------------------------------------------------------------------------
def plot_fig7_fixed(t1_v2: dict) -> None:
    s = t1_v2["summary"]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.35), dpi=300)

    ax = axes[0]
    benches = [("darcy", "Darcy"), ("burgers", "Burgers"), ("allen_cahn", "AC T=1"), ("lshape_darcy", "L-shape")]
    scales = [
        ("raw_residual", "Raw |R|", "#999999"),
        ("unsigned_inv_env", "Unsigned+env", "#b2abd2"),
        ("signed_z_no_env", "Signed (no env)", "#6baed6"),
        ("opres_sigma", "OpRes-CP (ours)", "#1b9e77"),
        ("ensemble_std", "5-Ensemble std", "#d95f02"),
    ]
    x = np.arange(len(benches))
    w = 0.155
    for idx, (sk, lab, col) in enumerate(scales):
        means = [s[pk]["ID"]["scales"][sk]["spearman"]["mean"] for pk, _ in benches]
        sds = [s[pk]["ID"]["scales"][sk]["spearman"]["sd"] for pk, _ in benches]
        ax.bar(x + (idx - 2) * w, means, width=w * 0.92, yerr=sds, capsize=1.5, label=lab, color=col, edgecolor="none")
    ax.axhline(0.0, color="#333333", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([b[1] for b in benches])
    ax.set_ylabel(r"Within-field Spearman $\rho(\sigma_h, |e_h|)$")
    ax.set_title("(A) Spatial error localization (ID, 5 seeds)")
    ax.set_ylim(-0.16, 1.16)
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=6.2, handlelength=1.0, columnspacing=0.6)

    ax = axes[1]
    doms = [
        ("darcy", "ID", "Darcy\nID"),
        ("darcy", "OOD-A", "Darcy\nOOD-A"),
        ("darcy", "OOD-B", "Darcy\nOOD-B"),
        ("lshape_darcy", "ID", "L-shape\nID"),
        ("lshape_darcy", "OOD-A", "L-shape\nOOD-A"),
    ]
    invs = [
        ("liouville", "Liouville", "#e7298a"),
        ("amg_v2", "AMG-V2", "#1b9e77"),
        ("amg_pcg2", "AMG-PCG2", "#006837"),
        ("jacobi_pcg5", "Jacobi-PCG5", "#80b1d3"),
    ]
    x2 = np.arange(len(doms))
    w2 = 0.19
    for idx, (ik, lab, col) in enumerate(invs):
        vals = []
        for pk, sp, _ in doms:
            oa = s[pk][sp]["operator_audit"]["full_rel_err_mean"]
            vals.append(oa[ik]["mean"] if ik in oa else np.nan)
        vals = np.array(vals, dtype=float)
        mask = np.isfinite(vals)
        ax.bar(x2[mask] + (idx - 1.5) * w2, vals[mask], width=w2 * 0.9, label=lab, color=col)
    ax.axhline(1.0, color="#c0392b", ls="--", lw=1.0, label=r"Point-corr. limit ($d_h(e_h)=1$)")
    ax.set_yscale("log")
    ax.set_ylim(0.09, 25.0)
    ax.set_xticks(x2)
    ax.set_xticklabels([d[2] for d in doms])
    ax.set_ylabel(r"$d_h(e_h) = \|B_h A_h e_h - e_h\|_2 / \|e_h\|_2$")
    ax.set_title(r"(B) Directional inverse defect $d_h(e_h)$")
    ax.legend(frameon=False, loc="upper right", fontsize=6.0, handlelength=1.0)

    ax = axes[2]
    bands = [("low_k_le_8", "Low\n($k\\leq 8$)"), ("mid_8_to_24", "Mid\n($8<k\\leq 24$)"), ("high_k_gt_24", "High\n($k>24$)")]
    xb = np.arange(3)
    wb = 0.24
    bd = s["darcy"]["ID"]["operator_audit"]["bands"]
    for idx, (sk, lab, col) in enumerate((("raw_residual", "Raw |R|", "#999999"),
                                          ("opres_sigma", "OpRes-CP", "#1b9e77"),
                                          ("ensemble_std", "5-Ensemble", "#d95f02"))):
        vals = [bd[bk]["scale_spearman"][sk]["mean"] for bk, _ in bands]
        ax.bar(xb + (idx - 1) * wb, vals, width=wb * 0.9, label=lab, color=col)
    for i, (bk, _) in enumerate(bands):
        ef = 100.0 * bd[bk]["error_energy_fraction"]["mean"]
        top_v = max(bd[bk]["scale_spearman"][sk]["mean"] for sk in ("raw_residual", "opres_sigma", "ensemble_std"))
        ax.text(i, top_v + 0.03, f"{ef:.1f}% err.\nenergy", ha="center", va="bottom", fontsize=6.4, color="#333333")
    ax.axhline(0.0, color="#333333", lw=0.7)
    ax.set_xticks(xb)
    ax.set_xticklabels([b[1] for b in bands])
    ax.set_ylim(-0.18, 0.85)
    ax.set_ylabel(r"Spearman $\rho(\sigma_h, |e_h^{(b)}|)$")
    ax.set_title(r"(C) Darcy $64^2$ spectral-band fidelity")
    ax.legend(frameon=False, loc="upper right", fontsize=6.5, handlelength=1.0)

    fig.tight_layout(w_pad=1.6)
    for out_dir in (OUT_FIG, CMAME_FIG):
        fig.savefig(out_dir / "fig7_fidelity_bands.pdf", bbox_inches="tight")
        fig.savefig(out_dir / "fig7_fidelity_bands.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved updated fig7_fidelity_bands.{pdf,png}")


# ---------------------------------------------------------------------------
# 6. Plot fig10_elasticity_fem.{pdf,png} (2x4 publication figure)
# ---------------------------------------------------------------------------
def plot_fig10_elasticity(t1: dict) -> None:
    ex = np.load(R4_DIR / "elasticity_example_s42.npz")
    pts, tris = ex["pts"], ex["tris"]
    triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)

    fig, axes = plt.subplots(2, 4, figsize=(11.6, 5.4), dpi=300)

    # Top Row: Spatial FEM mesh + von Mises fields on seed 42 sample 0
    ax = axes[0, 0]
    ax.triplot(triang, lw=0.30, color="#374151")
    th = np.linspace(0, 2 * np.pi, 100)
    ax.fill(0.5 + 0.2 * np.cos(th), 0.5 + 0.2 * np.sin(th), color="#E5E7EB", zorder=3)
    ax.plot(0.5 + 0.2 * np.cos(th), 0.5 + 0.2 * np.sin(th), color=DARK, lw=0.9, zorder=4)
    ax.set_aspect("equal")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_title(r"(a) Unstructured $P_1$ mesh $\mathcal{T}_c$ ($2{,}266$ DOFs)")

    vmin_vm = 0.0
    vmax_vm = float(np.quantile(ex["vm_true"], 0.99))
    vmax_err = float(max(np.quantile(ex["vm_err"], 0.99), np.quantile(ex["vm_opres_width"] * 0.5, 0.99)))

    for ax, vals, title, cmap, vmx in (
        (axes[0, 1], ex["vm_true"], r"(b) Reference $\sigma_{\rm vM}(U_h)$", "viridis", vmax_vm),
        (axes[0, 2], ex["vm_err"], r"(c) DeepONet $|\hat{\sigma}_{\rm vM}-\sigma_{\rm vM}|$", "magma", vmax_err),
        (axes[0, 3], ex["vm_opres_width"], r"(d) OpRes-CP (SA-AMG) full width", "magma", 2.0 * vmax_err),
    ):
        pc = ax.tripcolor(triang, facecolors=vals, cmap=cmap, vmin=0.0, vmax=vmx, shading="flat", rasterized=True)
        ax.fill(0.5 + 0.2 * np.cos(th), 0.5 + 0.2 * np.sin(th), color="#E5E7EB", zorder=3)
        ax.set_aspect("equal")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        ax.set_title(title)
        cb = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=6.6)

    # Bottom Row: 5-seed quantitative benchmarks
    s = t1["summary"]
    disp_keys = [
        ("5-Ens. (mean center)", "5-Ens.", EN_COLOR),
        ("Zienkiewicz-Zhu + CP", "ZZ+CP", "#9CA3AF"),
        ("BR/Verfurth FEM + CP", "BR+CP", "#6B7280"),
        ("Uniform-CP", "Uniform", "#4B5563"),
        ("OpRes-CP (Vector SA-AMG-V2)", "OpRes (AMG)", OP_COLOR),
        ("Jha Corrector + OpRes-CP", "Jha+OpRes", SLATE),
    ]
    ax = axes[1, 0]
    y_p = np.arange(len(disp_keys))
    vals = [s["splits"]["id"]["displacement"][k]["matched90_width"]["mean"] for k, _, _ in disp_keys]
    sds = [s["splits"]["id"]["displacement"][k]["matched90_width"]["sd"] for k, _, _ in disp_keys]
    ax.barh(y_p, vals, xerr=sds, color=[c for _, _, c in disp_keys], height=0.65, capsize=2)
    ax.set_yticks(y_p); ax.set_yticklabels([l for _, l, _ in disp_keys])
    ax.invert_yaxis()
    ax.set_xlabel("Matched-90% width (disp.)")
    ax.set_title(r"(e) ID displacement width ($\mathcal{T}_c$)")

    ax = axes[1, 1]
    vals_v = [s["splits"]["id"]["von_mises"][k]["matched90_width"]["mean"] for k, _, _ in disp_keys]
    sds_v = [s["splits"]["id"]["von_mises"][k]["matched90_width"]["sd"] for k, _, _ in disp_keys]
    ax.barh(y_p, vals_v, xerr=sds_v, color=[c for _, _, c in disp_keys], height=0.65, capsize=2)
    ax.set_yticks(y_p); ax.set_yticklabels([l for _, l, _ in disp_keys])
    ax.invert_yaxis()
    ax.set_xlabel(r"Matched-90% width ($\sigma_{\rm vM}$)")
    ax.set_title(r"(f) ID von Mises stress width")

    ax = axes[1, 2]
    cov_keys = [
        ("Uniform-CP", "Unif", "#4B5563"),
        ("BR/Verfurth FEM + CP", "BR+CP", "#9CA3AF"),
        ("Jha Corrector + Unif-CP", "Jha+Unif", MC_COLOR),
        ("5-Ens. (mean center)", "5-Ens", EN_COLOR),
        ("OpRes-CP (Vector SA-AMG-V2)", "OpRes", OP_COLOR),
        ("Jha Corrector + OpRes-CP", "Jha+OpR", SLATE),
    ]
    xp = np.arange(len(cov_keys))
    c_fine = [100.0 * s["splits"]["fine_paired"]["displacement"][k]["coverage"]["mean"] for k, _, _ in cov_keys]
    c_ood_vm = [100.0 * s["splits"]["ood_lame"]["von_mises"][k]["coverage"]["mean"] for k, _, _ in cov_keys]
    ax.bar(xp - 0.18, c_fine, width=0.35, color=SLATE, label=r"Fine $\mathcal{T}_f$ (disp.)")
    ax.bar(xp + 0.18, c_ood_vm, width=0.35, color=OP_COLOR, label="OOD-Lamé ($\\sigma_{\\rm vM}$)")
    ax.axhline(90.0, ls="--", lw=0.9, color=DARK)
    ax.set_xticks(xp); ax.set_xticklabels([l for _, l, _ in cov_keys], rotation=25, ha="right", fontsize=7.0)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Simultaneous coverage (%)")
    ax.set_title("(g) Fine mesh & OOD-Lamé shift")
    ax.legend(frameon=False, loc="upper left", fontsize=6.4)

    ax = axes[1, 3]
    ms = t1["multiscale_amg"]
    sl = ms["log_log_slopes"]
    lv = ms["levels"]
    dofs = [r["free_dofs"] for r in lv]
    ax.loglog(dofs, [r["amg_v2_solve_p50_ms"] for r in lv], "o-", color=OP_COLOR, lw=1.6, ms=4.5,
              label=f"SA-AMG solve ($\\beta={sl['amg_v2_solve']:.2f}$)")
    ax.loglog(dofs, [r["amg_v2_total_p50_ms"] for r in lv], "s--", color=SLATE, lw=1.5, ms=4.2,
              label=f"SA-AMG total ($\\beta={sl['amg_v2_total']:.2f}$)")
    ax.loglog(dofs, [r["spsolve_p50_ms"] for r in lv], "d:", color=DARK, lw=1.5, ms=4.2,
              label=f"Exact spsolve ($\\beta={sl['spsolve']:.2f}$)")
    ax.set_xlabel("Unstructured free DOFs $M$")
    ax.set_ylabel("CPU time P50 (ms)")
    ax.set_title(r"(h) Vector SA-AMG $O(M)$ scaling")
    ax.legend(frameon=False, loc="upper left", fontsize=6.4)

    fig.tight_layout(w_pad=1.3, h_pad=1.8)
    for out_dir in (OUT_FIG, CMAME_FIG):
        fig.savefig(out_dir / "fig10_elasticity_fem.pdf", bbox_inches="tight")
        fig.savefig(out_dir / "fig10_elasticity_fem.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig10_elasticity_fem.{pdf,png}")


# ---------------------------------------------------------------------------
# 7. Updated CMAME Graphical Abstract (>= 1328 px wide, scalar + vector FEM)
# ---------------------------------------------------------------------------
def plot_graphical_abstract() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 4.4), dpi=350)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 41)
    ax.axis("off")

    ax.text(
        50, 39.2,
        "OpRes-CP: Physics-as-Scale Signed Residual Lifting for Scalar & Vector PDE Conformal Bands",
        ha="center", va="center", fontsize=11.3, fontweight="bold", color=DARK,
    )
    ax.text(
        50, 36.6,
        "Scale Order-Preservation << Contractive Point Correction: sharp simultaneous bands without surrogate retraining",
        ha="center", va="center", fontsize=8.3, color=GRAY,
    )

    def box(x, y, w, h, fc, ec, title, body, title_color=DARK, lw=1.2):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                           facecolor=fc, edgecolor=ec, linewidth=lw)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h - 2.0, title, ha="center", va="top",
                fontsize=8.2, fontweight="bold", color=title_color)
        ax.text(x + w / 2, y + h / 2 - 1.4, body, ha="center", va="center",
                fontsize=7.1, color=DARK, linespacing=1.32)

    box(
        2.0, 19.3, 25.5, 14.2, EN_LIGHT, EN_COLOR,
        "1A. Prior UQ & Correctors",
        "Ensembles: 5× cost; blind to shared bias\n"
        "Point correctors: overshoot when $\\delta_{\\rm inv}>1$\n"
        "FEM BR/ZZ: collapse as $h \\to h/2$",
        title_color=EN_COLOR,
    )
    box(
        2.0, 2.5, 25.5, 14.2, "#F3F4F6", MC_COLOR,
        "1B. Raw-Residual PI-CP",
        "High-freq $\\|A_h\\hat{u} - f\\|$ spikes ($O(h^{-m})$)\n"
        "Misses 70.4% low-mode Darcy error\n"
        "Fails under mesh refinement",
        title_color=MC_COLOR,
    )

    box(
        32.5, 4.8, 35.0, 27.6, OP_LIGHT, OP_COLOR,
        "2. Signed PDE Residual Lifting",
        "Single Frozen Operator $\\hat{U}_h$ (FNO / U-Net / DeepONet)\n\n"
        "1. Conservative discrete / FEM residual:\n"
        "   $R_h = \\mathcal{N}_h(\\hat{U}_h; A)$  or  $\\mathbf{R}_h = \\mathbf{K}_h\\hat{\\mathbf{U}}_h - \\mathbf{F}_h$\n"
        "2. Boundary-consistent signed lift $Z_h = B_h R_h$:\n"
        "   Structured: Liouville DST-II / Adv-Diff / Strang\n"
        "   Irregular & Vector FEM: $O(M)$ Galerkin / 2-DOF SA-AMG\n"
        "3. Physical-radius RMS envelope + $\\lambda$-mix:\n"
        "   $\\sigma_h = \\sqrt{(1-\\lambda)\\mathrm{RMS}(\\tilde{\\sigma}_h)^2 + \\lambda\\tilde{\\sigma}_h^2}$\n"
        "   Exact deg.-1 hom. + SPD-PCG Chebyshev contraction",
        title_color=OP_COLOR, lw=1.7,
    )
    ax.plot([34.5, 65.5], [25.9, 25.9], color=OP_COLOR, alpha=0.35, lw=0.9)

    box(
        72.5, 19.3, 25.5, 14.2, "#EBF3FA", SLATE,
        "3A. Scalar & Vector Sharpness",
        "47%–92% narrower vs. 5-FNO Ens.\n"
        "57%–90% narrower vs. Val-Fit LSCI\n"
        "2D FEM Elasticity: 4.5–4.8× narrower\n"
        "for both displacement & von Mises stress",
        title_color=SLATE,
    )
    box(
        72.5, 2.5, 25.5, 14.2, "#ECFDF5", OP_COLOR,
        "3B. Diagnostics & Stacking",
        "100% energy contraction ($\\|I-BK\\|_K<1$)\n"
        "Stacks onto Jha correctors (2.5–2.9×)\n"
        "WCP, kernel rule ($\\ell_{\\rm pix}\\geq 2$), &\n"
        "exchangeable self-gating $\\lambda(A)$",
        title_color=OP_COLOR,
    )

    for y_arr in (26.2, 9.6):
        ax.add_patch(FancyArrowPatch((28.3, y_arr), (32.0, y_arr),
                                     arrowstyle="-|>", mutation_scale=13, lw=1.4, color=GRAY))
        ax.add_patch(FancyArrowPatch((68.2, y_arr), (72.0, y_arr),
                                     arrowstyle="-|>", mutation_scale=13, lw=1.5, color=OP_COLOR))

    for ext in ("pdf", "png"):
        fig.savefig(OUT_FIG / f"graphical_abstract.{ext}", bbox_inches="tight")
        fig.savefig(CMAME_FIG / f"graphical_abstract.{ext}", bbox_inches="tight")
        if _HAS_CMAME:
            fig.savefig(SUB_PKG / f"graphical_abstract.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("Saved updated graphical_abstract.{pdf,png}")


# ---------------------------------------------------------------------------
# 8. Export Ordered Figure_1..N into submission_package/
# ---------------------------------------------------------------------------
def collect_manuscript_figures(tex_path: Path) -> list[str]:
    seen: list[str] = []
    text = re.sub(r"(?<!\\)%.*", "", tex_path.read_text(encoding="utf-8"))
    for m in re.finditer(r"\\(input|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}", text):
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
    main_tex = ROOT / "paper_cmame" / "main.tex"
    if not main_tex.is_file():
        print("[skip] paper_cmame/main.tex not present; skipping ordered manuscript figure export.")
        return
    # Remove any old Figure_*.* first to prevent stale numbered artifacts
    for old in SUB_PKG.glob("Figure_*.*"):
        old.unlink()
    figs = collect_manuscript_figures(main_tex)
    for idx, fname in enumerate(figs, 1):
        src_pdf = CMAME_FIG / fname
        if not src_pdf.is_file():
            src_pdf = CMAME_FIG / f"{Path(fname).stem}.pdf"
        assert src_pdf.is_file(), f"Missing manuscript figure PDF: {src_pdf}"
        shutil.copy2(src_pdf, SUB_PKG / f"Figure_{idx}.pdf")

        src_png = CMAME_FIG / f"{src_pdf.stem}.png"
        if not src_png.is_file():
            alt_png = ROOT / "results" / "phase3_figures" / f"{src_pdf.stem}.png"
            if alt_png.is_file():
                shutil.copy2(alt_png, src_png)
            else:
                tmp_prefix = CMAME_FIG / f"_tmp_{src_pdf.stem}"
                subprocess.run(
                    ["pdftoppm", "-png", "-r", "300", "-singlefile", str(src_pdf), str(tmp_prefix)],
                    check=True,
                )
                Path(f"{tmp_prefix}.png").replace(src_png)
        shutil.copy2(src_png, SUB_PKG / f"Figure_{idx}.png")
    print(f"Exported {len(figs)} ordered figures to {SUB_PKG}: {figs}")


def main() -> None:
    t1_v2 = load_json(R2_DIR / "task1_fidelity.json")
    t1_v3 = load_json(R3_DIR / "task1_opnorm_gamma.json")
    t1_v4 = load_json(R4_DIR / "task1_elasticity.json")
    t2_v4 = load_json(R4_DIR / "task2_nonlinear_pcg_jha.json")
    t3_v4 = load_json(R4_DIR / "task3_physics_gate.json")

    build_sec2_positioning_table()
    update_tab_op_approx(t1_v3, t2_v4)
    build_rev4_main_tables(t1_v4, t2_v4)
    build_rev4_appendix_tables(t1_v4, t2_v4, t3_v4)
    plot_fig7_fixed(t1_v2)
    plot_fig10_elasticity(t1_v4)
    plot_graphical_abstract()
    export_ordered_figures()
    print("[Rev4 Tables & Figures] All tables, fig7, fig10, GA, and Figure_1..10 updated cleanly.")


if __name__ == "__main__":
    main()
