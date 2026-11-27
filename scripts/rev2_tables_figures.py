"""Generate publication-grade booktabs LaTeX tables and vector PDF figures for Revision v2.

Reads JSON outputs from results/phase3_revision_v2/ and writes:
  - paper/tables/phase3_rev2_main_tables.tex
  - paper/tables/phase3_rev2_appendix_tables.tex
  - Updates paper/tables/phase3_promoted_tables.tex (adds LSCI cal. band row + U-Net note)
  - Updates paper/tables/phase3_revision_tables.tex (adds fair LSCI same-event table)
  - paper/figures/fig7_fidelity_bands.pdf
  - paper/figures/fig8_stress_and_amg.pdf
(with matching copies in results/phase3_revision_v2/tables/ and figures/).
"""

import json
import math
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path("results/phase3_revision_v2")
TBL_OUT = OUT / "tables"
FIG_OUT = OUT / "figures"
_HAS_PAPER = Path("paper").is_dir()
PAPER_TBL = Path("paper/tables") if _HAS_PAPER else Path("outputs/tables")
PAPER_FIG = Path("paper/figures") if _HAS_PAPER else Path("outputs/figures")
REV_TBL = Path("results/phase3_revision/tables")
REPLAY_TBL = Path("results/manuscript_replay/tables") if _HAS_PAPER else Path("outputs/tables")


def _read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def _msd(node, digits=3, mul=1.0):
    if node is None or node.get("mean") is None:
        return "--"
    m = mul * float(node["mean"])
    s = mul * float(node["sd"]) if node.get("sd") is not None else 0.0
    return rf"${m:.{digits}f} \pm {s:.{digits}f}$"


def _m(node, digits=3, mul=1.0):
    if node is None or node.get("mean") is None:
        return "--"
    return f"{mul * float(node['mean']):.{digits}f}"


def make_table(header, rows, alignment, caption, label, size=r"\scriptsize", tabcolsep="3.0pt", notes=None):
    ncols = len(header)
    for r in rows:
        if r is not None and len(r) != ncols:
            raise ValueError(f"Column count mismatch in {label}: {r}")
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        size,
        rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    for r in rows:
        if r is None:
            lines.append(r"\addlinespace[3pt]")
        else:
            lines.append(" & ".join(r) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if notes:
        lines.append(rf"\par\smallskip\raggedright\footnotesize {notes}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def _write_table_file(filename, blocks, also_sync_rev=False):
    content = "% Auto-generated booktabs LaTeX tables (Revision v2). Requires booktabs.\n\n" + "\n\n".join(blocks) + "\n"
    if content.count(r"\begin{") != content.count(r"\end{") or content.count("{") != content.count("}"):
        raise ValueError(f"Unclosed LaTeX syntax in {filename}")
    if r"\hline" in content:
        raise ValueError(f"Forbidden \\hline in {filename}")
    targets = [TBL_OUT, PAPER_TBL, REPLAY_TBL]
    if also_sync_rev:
        targets.append(REV_TBL)
    for d in targets:
        d.mkdir(parents=True, exist_ok=True)
        (d / filename).write_text(content, encoding="ascii")


def build_task1_tables(t1):
    s = t1["summary"]
    scale_order = (
        ("raw_residual", r"Raw $|R_h|$ (PI-CP)"),
        ("unsigned_inv_env", r"Unsigned Inv+env"),
        ("signed_z_no_env", r"Signed $|Z_h|$ (no env)"),
        ("opres_sigma", r"\textbf{OpRes-CP $\sigma_h$ (ours)}"),
        ("ensemble_std", r"5-Member Ens.\ std"),
    )

    def _fidelity_rows(split_specs):
        rows = []
        for idx, (pde_k, sp_k, lab) in enumerate(split_specs):
            if idx > 0:
                rows.append(None)
            node = s[pde_k][sp_k]["scales"]
            for sk, slab in scale_order:
                m = node[sk]
                is_op = (sk == "opres_sigma")
                row = [
                    lab if sk == scale_order[0][0] else "",
                    slab,
                    _msd(m["spearman"], 3),
                    _msd(m["pearson"], 3),
                    _msd(m["peak_recall_10"], 3),
                    _msd(m["peak_capture_5"], 1, 100.0),
                    _msd(m["sigma_tilde_xstar_mean"], 2),
                    _msd(m["rho_cross"], 3),
                ]
                rows.append(tuple(row))
        return rows

    id_specs = (
        ("darcy", "ID", r"Darcy $64^2$ ID"),
        ("burgers", "ID", r"Burgers $64$ ID"),
        ("allen_cahn", "ID", r"AC $64^2$ $T=1$ ID"),
        ("lshape_darcy", "ID", r"L-shape $64^2$ ID"),
    )
    ood_specs = (
        ("darcy", "OOD-A", r"Darcy OOD-A"),
        ("darcy", "OOD-B", r"Darcy OOD-B"),
        ("burgers", "Visc-OOD", r"Burgers Visc."),
        ("burgers", "High-OOD", r"Burgers High"),
        ("allen_cahn", "React-OOD", r"AC React."),
        ("lshape_darcy", "OOD-A", r"L-shape OOD-A"),
    )

    hdr = ("Benchmark", "Frozen scale", r"$\rho(\sigma,|e|)$", r"$r(\sigma,|e|)$",
           r"PeakRec@10\%", r"PeakCap@5\%", r"$\tilde\sigma(x^*)$", r"$\rho_{\rm cross}$")

    t_fid_id = make_table(
        hdr, _fidelity_rows(id_specs), "llrrrrrr",
        r"Five-seed scale--error spatial fidelity on ID benchmarks (mean $\pm$ SD across formal seeds $42$, $137$, $314$, $271$, and $811$, evaluated against the shared single-FNO error $|e_h|=|\widehat U_h - U_h|$). $\rho$ and $r$ are within-field Spearman and Pearson correlations; PeakRec@10\% is top-decile overlap recall; PeakCap@5\% (\%) is the fraction of cases whose global error peak $x^*$ lies in the top-5\% predicted-scale region; $\tilde\sigma(x^*)=\sigma(x^*)/\overline\sigma$ is normalized peak concentration (Theorem~\ref{thm:fixed}); $\rho_{\rm cross}$ is cross-sample Spearman correlation between $\overline\sigma_i$ and $\|e_i\|_\infty$. Full OOD splits appear in Appendix Table~\ref{tab:opres-fidelity-ood}.",
        "tab:opres-fidelity", size=r"\scriptsize", tabcolsep="1.8pt",
    )

    t_fid_ood = make_table(
        hdr, _fidelity_rows(ood_specs), "llrrrrrr",
        r"Five-seed scale--error spatial fidelity across all six OOD shifts (mean $\pm$ SD, same protocol and definitions as Table~\ref{tab:opres-fidelity}).",
        "tab:opres-fidelity-ood", size=r"\scriptsize", tabcolsep="1.8pt",
    )

    # Signed error recovery + Operator approximation residual table (Main or Section 5 / Appendix)
    rec_specs = (
        ("darcy", "ID", r"Darcy $64^2$ ID"),
        ("darcy", "OOD-A", r"Darcy $64^2$ OOD-A"),
        ("darcy", "OOD-B", r"Darcy $64^2$ OOD-B"),
        None,
        ("burgers", "ID", r"Burgers $64$ ID"),
        ("burgers", "Visc-OOD", r"Burgers $64$ Visc.-OOD"),
        ("burgers", "High-OOD", r"Burgers $64$ High-OOD"),
        None,
        ("allen_cahn", "ID", r"Allen--Cahn $64^2$ $T=1$ ID"),
        ("allen_cahn", "React-OOD", r"Allen--Cahn $64^2$ React.-OOD"),
        None,
        ("lshape_darcy", "ID", r"L-shape $64^2$ ID (AMG-PCG2)"),
        ("lshape_darcy", "OOD-A", r"L-shape $64^2$ OOD-A (AMG-PCG2)"),
    )
    rec_rows = []
    for item in rec_specs:
        if item is None:
            rec_rows.append(None)
            continue
        pk, sk, lab = item
        sr = s[pk][sk]["signed_recovery"]
        op_sc = s[pk][sk]["scales"]["opres_sigma"]
        rec_rows.append((
            lab,
            _msd(sr["cos_mean"], 3),
            _msd(sr["rel_l2_err_mean"], 3),
            _msd(sr["ls_residual_mean"], 3),
            _msd(op_sc["rho_eff_mean"], 2),
            _msd(op_sc["sigma_tilde_xstar_mean"], 2),
        ))
    t_rec = make_table(
        ("Benchmark / split", r"$\cos(Z_h, e_h)$", r"$\|Z_h-e_h\|_2/\|e_h\|_2$",
         r"$\min_\alpha\|\alpha Z_h-e_h\|_2/\|e_h\|_2$", r"Thm.~\ref{thm:fixed} $\overline\rho_i$",
         r"$\tilde\sigma(x^*)$"),
        rec_rows, "lrrrrr",
        r"Signed solution-space error recovery of $Z_h=B_h R_h$ against $e_h=\widehat U_h - U_h$ across all ten benchmark splits (five-seed mean $\pm$ SD), together with Theorem~\ref{thm:fixed}'s effective peak-concentration ratio $\rho_i$ and $\tilde\sigma(x^*)$. Even when the unscaled approximation defect $\|Z_h-e_h\|_2/\|e_h\|_2$ exceeds $1$ (Darcy OOD-A/B), positive homogeneity absorbs global mis-scaling while preserving spatial peak concentration $\tilde\sigma(x^*) > 1.7$.",
        "tab:opres-signed-recovery", size=r"\scriptsize", tabcolsep="3.2pt",
    )

    # Operator approximation error & frequency-band table on Darcy 64^2 and L-shape 64^2
    op_rows = []
    for sp, lab in (("ID", r"Darcy $64^2$ ID"), ("OOD-A", r"Darcy $64^2$ OOD-A"), ("OOD-B", r"Darcy $64^2$ OOD-B")):
        oa = s["darcy"][sp]["operator_audit"]["full_rel_err_mean"]
        op_rows.append((lab, _msd(oa["liouville"], 3), _msd(oa["amg_v2"], 3),
                        _msd(oa["amg_pcg2"], 3), _msd(oa["jacobi_pcg5"], 3)))
    op_rows.append(None)
    for sp, lab in (("ID", r"L-shape $64^2$ ID"), ("OOD-A", r"L-shape $64^2$ OOD-A")):
        oa = s["lshape_darcy"][sp]["operator_audit"]["full_rel_err_mean"]
        op_rows.append((lab, "--", _msd(oa["amg_v2"], 3),
                        _msd(oa["amg_pcg2"], 3), _msd(oa["jacobi_pcg5"], 3)))

    t_op = make_table(
        ("Domain / split", "Liouville DST-II", "AMG-V2", "AMG-PCG2", "Jacobi-PCG5"),
        op_rows, "lrrrr",
        r"Exact discrete operator approximation residual $\|B_h A_h e_h - e_h\|_2 / \|e_h\|_2$ computed with the exact finite-volume matrix $A_h$ across five seeds (mean $\pm$ SD), directly testing Proposition~\ref{prop:resolution}'s operator defect $\delta_{\rm inv}$. On Darcy OOD-B, Liouville's omitted potential drives defect $>1$ (explaining point-correction collapse), whereas multilevel AMG retains contractive accuracy across both rectangular and re-entrant domains.",
        "tab:opres-op-approx", size=r"\footnotesize", tabcolsep="4pt",
    )

    # Frequency-band decomposition table (Darcy ID + L-shape ID)
    band_rows = []
    band_specs = (
        ("low_k_le_8", r"Low ($k \le 8$)"),
        ("mid_8_to_24", r"Mid ($8 < k \le 24$)"),
        ("high_k_gt_24", r"High ($k > 24$)"),
    )
    for dom_idx, (pk, dlab) in enumerate((("darcy", r"Darcy $64^2$ ID"), ("lshape_darcy", r"L-shape $64^2$ ID"))):
        if dom_idx > 0:
            band_rows.append(None)
        bd = s[pk]["ID"]["operator_audit"]["bands"]
        for bk, blab in band_specs:
            b = bd[bk]
            band_rows.append((
                dlab if bk == "low_k_le_8" else "",
                blab,
                _msd(b["error_energy_fraction"], 1, 100.0),
                _msd(b["z_band_cos"], 3),
                _msd(b["scale_spearman"]["raw_residual"], 3),
                _msd(b["scale_spearman"]["opres_sigma"], 3),
                _msd(b["scale_spearman"]["ensemble_std"], 3),
            ))
    t_bands = make_table(
        ("Domain", "Spectral band", r"Error energy (\%)", r"$\cos(Z_h^{(b)}, e_h^{(b)})$",
         r"$\rho(|R_h|,|e^{(b)}|)$", r"$\rho(\sigma_h,|e^{(b)}|)$", r"$\rho(\sigma_{\rm ens},|e^{(b)}|)$"),
        band_rows, "llrrrrr",
        r"Spectral-band decomposition of forecast error on Darcy $64^2$ (DST-II radial modes) and L-shape $64^2$ (Dirichlet Laplacian eigenmodes) across five seeds. Raw $|R_h|$ correlates primarily with the negligible high-frequency tail while missing the dominant low-frequency error band; OpRes-CP recovers the dominant low- and mid-frequency error structure.",
        "tab:opres-freq-bands", size=r"\scriptsize", tabcolsep="2.8pt",
    )
    return t_fid_id, t_rec, t_op, t_bands, t_fid_ood


def build_task2_fair_lsci_table(t2):
    specs = (
        ("Darcy", "darcy", (("id_test_64", r"ID $64^2$"), ("ood_a_64", r"Contr.\ OOD"), ("ood_b_64", r"Struct.\ OOD"))),
        ("Burgers", "burgers", (("id_64", r"ID $64$"), ("ood_64", r"Visc.\ OOD"), ("ood_high_64", r"High OOD"))),
        (r"AC $T=1$", "allen_cahn", (("id_64_20", r"ID $64^2$"), ("ood_64_20", r"React.\ OOD"))),
    )
    rows = []
    for pidx, (plab, pk, spairs) in enumerate(specs):
        if pidx > 0:
            rows.append(None)
        for grp, slab in spairs:
            node = t2["pdes"][pk]["summary"][grp]
            lb = node["lsci_cal_band"]
            op = node["opres_whole_field"]
            pr = node["proj32"]
            lsci_fw = _m(lb["finite_actual_width"], 4)
            lsci_mw = _m(lb["matched90_width"], 4)
            op_mw = _m(op["matched90_width"], 4)
            proj_lsci = _m(pr["lsci_slab32_finite_width"], 4)
            proj_op_full = _m(pr["opres_full_band_induced_32proj_width"], 4)
            proj_op_32 = _m(pr["opres_cal32_matched90_width"], 4)
            rows.append((
                plab if grp == spairs[0][0] else "",
                slab,
                f"{_m(lb['coverage'], 2, 100)}/{lsci_fw}",
                f"{lsci_mw}/{op_mw}",
                f"{_m(pr['lsci_native32_cov'], 2, 100)}/{proj_lsci}",
                f"{_m(pr['opres_full_band_induced_32proj_cov'], 1, 100)}/{proj_op_full}",
                proj_op_32,
            ))
    t_fair = make_table(
        ("PDE", "Split", "LSCI Band", r"M90 w (LSCI/OpRes)",
         "LSCI Nat.-32", r"OpRes Box$\to$32", "OpRes Cal-32"),
        rows, "llrrrrr",
        r"Strictly same-event fair comparison against RLCP-weighted LSCI across five frozen FNO seeds ($\alpha=0.10$, using identical RLCP kernel weights $w$ and 32 random projections $\{\phi_m\}_{m=1}^{32}$). Entries with a slash report simultaneous coverage (\%) / width. Left: whole-field simultaneous rectangular band (LSCI Band uses all 500 locally weighted calibration residuals + OpRes-CP relative floor $\rho_{\rm rel}=0.10$ and exact RLCP weighted max-conformal quantile; M90 w compares empirically matched-$90\%$ full widths). Right: exact 32-projection interval width of LSCI's native depth set vs.\ the support-function width $2q\sum_j|\phi_{m,j}|\sigma_h(x_j)$ induced by OpRes-CP's whole-field band (Box$\to$32, covering $100.0\%$ of 32-slice events) and OpRes-CP calibrated directly on the same 32 projections (Cal-32, matched-$90\%$ width).",
        "tab:opres-lsci-fair", size=r"\scriptsize", tabcolsep="2.2pt",
    )
    return t_fair


def update_promoted_and_revision_tables(t2):
    """Update phase3_promoted_tables.tex (add LSCI cal. band row & Burgers U-Net note)
    and phase3_revision_tables.tex (append tab:opres-lsci-fair while preserving tab:opres-lsci untouched)."""
    # 1) phase3_promoted_tables.tex
    p_path = PAPER_TBL / "phase3_promoted_tables.tex"
    if not p_path.is_file():
        p_path = REV_TBL / "phase3_promoted_tables.tex"
    if p_path.is_file():
        txt = p_path.read_text(encoding="utf-8")

        def _lb_cell(pk, grp, with_mw=False):
            node = t2["pdes"][pk]["summary"][grp]["lsci_cal_band"]
            c = 100.0 * node["coverage"]["mean"]
            if with_mw:
                return f"{c:.2f} / {node['matched90_width']['mean']:.4f}"
            return f"{c:.2f}"

        new_row = (
            r"LSCI (cal.\ band)$^\dagger$ & "
            + f"{_lb_cell('darcy', 'id_test_64', True)} & {_lb_cell('darcy', 'ood_a_64')} & {_lb_cell('darcy', 'ood_b_64')} & "
            + f"{_lb_cell('burgers', 'id_64', True)} & {_lb_cell('burgers', 'ood_64')} & {_lb_cell('burgers', 'ood_high_64')} & "
            + f"{_lb_cell('allen_cahn', 'id_64_20', True)} & {_lb_cell('allen_cahn', 'ood_64_20')}"
            + r" \\"
        )
        if r"LSCI (cal.\ band)" not in txt and r"LSCI (Val-Fit band)" not in txt:
            anchor = r"LSCI (64-draw env.)$^\dagger$"
            lines = txt.splitlines()
            out_lines = []
            for ln in lines:
                out_lines.append(ln)
                if anchor in ln:
                    out_lines.append(new_row)
            txt = "\n".join(out_lines) + "\n"

        # Update tab:opres-all-baselines footnote to reference tab:opres-lsci-fair
        old_fn = r"$^\dagger$RLCP-weighted LSCI (Appendix Table~\ref{tab:opres-lsci}); native set and 64-draw rectangular envelope measure distinct coverage events."
        new_fn = r"$^\dagger$RLCP-weighted LSCI (Appendix Tables~\ref{tab:opres-lsci}--\ref{tab:opres-lsci-fair}); LSCI (cal.\ band) evaluates the same whole-field rectangular event using all 500 locally weighted calibration residuals, whereas native set (32 projections) and 64-draw envelope measure distinct events."
        if old_fn in txt:
            txt = txt.replace(old_fn, new_fn)

        # Add Burgers U-Net note to tab:opres-cross-arch
        old_cross_end = r"Allen--Cahn $T=1$ & U-Net & 88.88 & 85.83\% & 88.28 & -- & $64^2\!\to\!128^2$ & $+11.48$ & $-88.60$ \\" + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}"
        new_cross_end = (
            r"Allen--Cahn $T=1$ & U-Net & 88.88 & 85.83\% & 88.28 & -- & $64^2\!\to\!128^2$ & $+11.48$ & $-88.60$ \\" + "\n"
            + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"
            + r"\par\smallskip\raggedright\footnotesize Note: On Burgers $64\!\to\!256$, the non-resolution-invariant U-Net's point relative $L^2$ error jumps $16\times$ ($3.64\%\to 58.15\%$); Uniform-CP collapses to $1.64\%$ ($-88.84$ pp), whereas OpRes-CP automatically inflates band width by $15.67\times$ to expose the backbone collapse."
        )
        if "non-resolution-invariant U-Net" not in txt:
            assert old_cross_end in txt, "Anchor not found in phase3_promoted_tables.tex"
            txt = txt.replace(old_cross_end, new_cross_end)

        for d in (PAPER_TBL, REV_TBL, REPLAY_TBL, TBL_OUT):
            d.mkdir(parents=True, exist_ok=True)
            (d / "phase3_promoted_tables.tex").write_text(txt, encoding="ascii")

    # 2) phase3_revision_tables.tex: append tab:opres-lsci-fair right after tab:opres-lsci without modifying tab:opres-lsci
    r_path = PAPER_TBL / "phase3_revision_tables.tex"
    if not r_path.is_file():
        r_path = REV_TBL / "phase3_revision_tables.tex"
    if r_path.is_file():
        rtxt = r_path.read_text(encoding="utf-8")
        t_fair = build_task2_fair_lsci_table(t2)
        if r"\label{tab:opres-lsci-fair}" in rtxt:
            # Replace existing block safely by finding \label{tab:opres-lsci-fair}
            lpos = rtxt.index(r"\label{tab:opres-lsci-fair}")
            idx0 = rtxt.rindex(r"\begin{table*}", 0, lpos)
            idx1 = rtxt.index(r"\end{table*}", lpos) + len(r"\end{table*}")
            rtxt = rtxt[:idx0] + t_fair + rtxt[idx1:]
        else:
            marker = r"\label{tab:opres-lsci}"
            m_pos = rtxt.index(marker)
            end_pos = rtxt.index(r"\end{table*}", m_pos) + len(r"\end{table*}")
            rtxt = rtxt[:end_pos] + "\n\n" + t_fair + rtxt[end_pos:]
        for d in (PAPER_TBL, REV_TBL, REPLAY_TBL, TBL_OUT):
            d.mkdir(parents=True, exist_ok=True)
            (d / "phase3_revision_tables.tex").write_text(rtxt, encoding="ascii")


def build_task3_tables(t3):
    # Part A: Continuous shift alpha_hf
    pa = t3["part_a_continuous_shift"]["summary"]
    rows_a = []
    for ahf in t3["alphas_hf"]:
        ak = str(ahf)
        m = pa[ak]["methods"]
        rows_a.append((
            f"{ahf:.2f}",
            f"{100*pa[ak]['mean_hf_energy_ratio']['mean']:.3f}\\%",
            f"{_m(m['opres_locked']['coverage'], 2, 100)} / {_m(m['opres_locked']['matched90_width'], 4)}",
            f"{_m(m['opres_robust']['coverage'], 2, 100)} / {_m(m['opres_robust']['matched90_width'], 4)}",
            f"{_m(m['physics_scalar']['coverage'], 2, 100)} / {_m(m['physics_scalar']['matched90_width'], 4)}",
            f"{_m(m['ensemble_mean']['coverage'], 2, 100)} / {_m(m['ensemble_mean']['matched90_width'], 4)}",
        ))
    t_a = make_table(
        (r"$\alpha_{\rm hf}$", r"$E_{k>8}/E_{\rm tot}$", r"OpRes ($\lambda\!=\!0.85, \ell\!=\!0.08$)",
         r"OpRes ($\lambda\!=\!0.25, \ell\!=\!0.32$)", r"Physics-Scalar ($\lambda\!=\!0$)", r"5-Ensemble (mean)"),
        rows_a, "lrrrrr",
        r"Burgers continuous high-frequency shift injection $\alpha_{\rm hf}\in[0,1]$ on the frozen FNO and five-member ensemble (five seeds, 500 paired trajectories per level). Entries report simultaneous coverage (\%) at the locked ID calibration threshold / empirically matched $90\%$ full width.",
        "tab:opres-burgers-cont", size=r"\scriptsize", tabcolsep="3.0pt",
    )

    # Part B: Noisy spectral overlap (report Ambient background noise + ID-only noise)
    pb_amb = t3["part_b_noisy_spectral_overlap"]["summary"]["ambient_noise"]
    pb_id = t3["part_b_noisy_spectral_overlap"]["summary"]["id_only_noise"]
    rows_b = []
    for proto_idx, (plab, pb, gate_list) in enumerate((
        ("Ambient (ID+OOD)", pb_amb, (
            ("pure_spectral_scalar", r"Pure spec.\ ($E_{k>8}$)"),
            ("joint_input_physics_cal_only", r"In.+phys.\ (Cal)"),
            ("physics_rms_z_cal_only", r"Phys.\ $\operatorname{RMS}(Z_h)$"),
        )),
        ("ID-only noise", pb_id, (
            ("pure_spectral_scalar", r"Pure spec.\ ($E_{k>8}$)"),
            ("pure_spectral_logistic", r"Pure in.\ log."),
            ("joint_input_physics_logistic", r"In.+phys.\ joint"),
        )),
    )):
        if proto_idx > 0:
            rows_b.append(None)
        for eta in t3["etas_noise"]:
            ek = str(eta)
            node = pb[ek]
            for g_idx, (gkey, glab) in enumerate(gate_list):
                g = node[gkey]
                op5 = g["ops"]["0.05"]
                rows_b.append((
                    plab if (eta == 0.0 and g_idx == 0) else "",
                    f"{eta:.2f}" if g_idx == 0 else "",
                    glab,
                    _msd(g["auroc"], 3),
                    _msd(g["fpr_at_95_tpr"], 3),
                    f"{_m(g['tpr_at_1_fpr'], 1, 100)}/{_m(g['tpr_at_5_fpr'], 1, 100)}",
                    f"{_m(g['point_aurc'], 4)}/{_m(g['miscov_aurc'], 3)}",
                    _m(op5["accepted_coverage"], 2, 100),
                ))
    t_b = make_table(
        ("Protocol", r"$\eta_{\rm noise}$", "Gate", "AUROC", r"FPR@95", r"TPR@1/5\%", "AURC (pt/mc)", r"Acc.\ cov."),
        rows_b, "lllrrrrr",
        r"Burgers noisy spectral-overlap stress test injecting relative RMS noise $\eta_{\rm noise}$ into modes $9$--$16$ (five seeds, equal 500+500 ID/High-OOD mixtures, zero retraining). AURC reports both point-failure ($L^2>0.15$) and band-miscoverage Risk--Coverage area (pt/mc). Under ambient noise at $\eta_{\rm noise}=0.20$, combining input spectrum with physics lifted scale lowers point-failure AURC by $35\%$ ($0.0114\to 0.0074$) and $\operatorname{RMS}(Z_h)$ alone cuts it $2.5\times$ ($0.0046$, miscoverage AURC $0.0786$ vs.\ $0.1061$); under ID-only noise at full spectral overlap ($\eta_{\rm noise}=0.10$), joint input+physics detection lifts AUROC from $0.509$ (scalar) / $0.572$ (input logistic) to $0.656$. Acc.\ cov.\ (\%) is evaluated at nominal $5\%$ Cal ID rejection.",
        "tab:opres-burgers-noisy", size=r"\scriptsize", tabcolsep="1.8pt",
    )
    return t_a, t_b


def build_task4_table(t4):
    s = t4["summary"]
    specs = (
        ("darcy_ood_a", r"Darcy contrast (full overlap)", ("Summary (p=5)", "Full-Grid (d=4096)", "Oracle 1D sigma")),
        ("burgers_visc_overlap", r"Burgers visc.\ (overlap, $\delta_{\rm suf}=0$)", ("Summary (p=3)", "Full-Grid (d=65)")),
        ("allen_cahn_react_overlap", r"AC react.\ (overlap, $\delta_{\rm suf}=0$)", ("Summary (p=4)", "Full-Grid (d=4098)")),
    )
    rows = []
    for bidx, (bk, blab, fnames) in enumerate(specs):
        if bidx > 0:
            rows.append(None)
        for fi, fn in enumerate(fnames):
            for proto, plab in (("transductive", "Transductive"), ("split_fit", "Split-fit (Thm.~2)")):
                node = s[bk][fn][proto]
                rows.append((
                    blab if (fi == 0 and proto == "transductive") else "",
                    fn if proto == "transductive" else "",
                    plab,
                    _msd(node["delta_w_heldout_P"], 4),
                    _msd(node["ess"], 1),
                    _m(node["infinite_rate"], 2, 100),
                    _m(node["coverage"], 2, 100),
                ))
    return make_table(
        ("Shift regime", "Features", "Protocol", r"$\widehat\delta_w$ (held-out $P$)", "ESS", r"Inf.\ (\%)", r"Cov.\ (\%)"),
        rows, "lllrrrr",
        r"Quantitative verification of Theorem~\ref{thm:shift}'s density-ratio error $\widehat\delta_w = \frac12\widehat{\mathbb E}_P|\widetilde w - w^*|$ against exact closed-form generator ratios $w^*$ (five-seed mean $\pm$ SD). Darcy contrast has $100\%$ support overlap ($0\%$ out-of-support mass); Burgers viscosity and Allen--Cahn reaction evaluate the overlap-conditioned target where $\delta_{\rm suf}=0$ identically (out-of-support target mass $70.36\%$ and $66.67\%$, respectively). Compact physical summaries reduce $\widehat\delta_w$ by $12$--$15\times$ on Darcy and preserve well-conditioned ESS.",
        "tab:opres-delta-w", size=r"\scriptsize", tabcolsep="2.8pt",
    )


def build_task5_tables(t5):
    # Part A: AC sensitivity
    sa = t5["part_a_ac_sensitivity"]["summary"]
    order = (
        ("locked", r"\textbf{Locked ($2.0, 0, 1.5$)}"),
        ("beta_0.25", r"$\beta_+ = 0.25$"),
        ("beta_0.50", r"$\beta_+ = 0.50$"),
        ("gmax_0.5", r"$g_{\max} = 0.5$"),
        ("gmax_3.0", r"$g_{\max} = 3.0$"),
        ("cclip_1.0", r"$c_{\rm clip} = 1.0$"),
        ("cclip_4.0", r"$c_{\rm clip} = 4.0$"),
        ("jac_off", r"Jacobian off ($J\equiv 0$)"),
    )
    rows_a = []
    for k, lab in order:
        n = sa[k]
        rows_a.append((
            lab,
            f"{_m(n['t1_id']['coverage'], 2, 100)}/{_m(n['t1_id']['matched90_width'], 4)}",
            f"{_m(n['t1_ood']['coverage'], 2, 100)}/{_m(n['t1_ood']['matched90_width'], 4)}",
            f"{_m(n['t2_id_t2cal']['coverage'], 2, 100)}/{_m(n['t2_id_t2cal']['matched90_width'], 4)}",
            f"{_m(n['t2_ood_t2cal']['coverage'], 2, 100)}/{_m(n['t2_ood_t2cal']['matched90_width'], 4)}",
            f"{_m(n['t2_id_t1_95reuse']['coverage'], 2, 100)}",
        ))
    t_ac = make_table(
        ("Guard config.", r"$T\!=\!1$ ID", r"$T\!=\!1$ OOD",
         r"$T\!=\!2$ ID", r"$T\!=\!2$ OOD", r"$T\!=\!2$ (T1 95\%)"),
        rows_a, "lrrrrr",
        r"Five-seed zero-retraining sensitivity of the Allen--Cahn guarded Strang reaction Jacobian parameters $(c_{\rm clip}, \beta_+, g_{\max})$ (Eq.~\eqref{eq:ac-jac}, $\gamma=0.2$) on frozen FNO rollouts. Entries with a slash report simultaneous coverage (\%) / matched $90\%$ full width; $T=2$ uses independent long-window calibration and test streams.",
        "tab:opres-ac-sens", size=r"\scriptsize", tabcolsep="2.5pt",
    )

    # Part B: AMG scaling & contrast convergence
    sb = t5["part_b_amg_scaling"]
    sc = sb["scaling_at_sigma_1.0"]
    cv = sb["convergence_grid"]
    rows_b = []
    for n in sb["grids"]:
        nk = str(n)
        c = sc[nk]["complexity"]
        g10 = cv[nk]["1.0"]
        g20 = cv[nk]["2.0"]
        rows_b.append((
            f"${n}^2$",
            f"{c['active_cells_finest']:,}",
            f"{c['grid_complexity_CG']:.3f}/{c['operator_complexity_CO']:.3f}",
            f"{sc[nk]['amg_v2_total_p50_ms']:.2f}/{sc[nk]['amg_pcg2_total_p50_ms']:.2f}",
            f"{sc[nk]['spsolve_p50_ms']:.2f} ({sc[nk]['speedup_v2_total_vs_spsolve']:.1f}$\\times$)",
            f"{g10['res_red_v2']['mean']:.3f}/{g10['res_red_pcg2']['mean']:.3f}",
            f"{g10['op_err_v2']['mean']:.3f}/{g10['op_err_pcg2']['mean']:.3f}",
            f"{g20['op_err_v2']['mean']:.3f}/{g20['op_err_pcg2']['mean']:.3f}",
        ))
    sl = sb["log_log_slopes"]["n_64_to_512"]
    notes = (
        rf"Analytical $2\times 2$ cell-agglomeration upper bound: $C_G, C_O < 4/3 = 1.3333$. "
        rf"Log--log CPU runtime slopes ($t \propto M^\beta$, $64^2$--$512^2$): "
        rf"\texttt{{AMG-V2}} solve $\beta={sl['amg_v2_solve_p50_ms']:.2f}$, "
        rf"\texttt{{AMG-V2}} total $\beta={sl['amg_v2_total_p50_ms']:.2f}$, "
        rf"\texttt{{AMG-PCG2}} total $\beta={sl['amg_pcg2_total_p50_ms']:.2f}$, "
        rf"exact \texttt{{spsolve}} $\beta={sl['spsolve_p50_ms']:.2f}$."
    )
    t_amg = make_table(
        ("Grid", r"$M$", r"$C_G/C_O$", "V2/PCG2 (ms)", r"\texttt{spsolve} (ms)",
         r"$\|r_2\|/\|r_0\|$", r"Err ($\sigma\!=\!1$)", r"Err ($\sigma\!=\!2$)"),
        rows_b, "lrrrrrrr",
        r"Pure-operator L-shaped Galerkin AMG multiscale scaling and contrast robustness across $n\in\{32,64,128,256,512\}$ ($768$--$196{,}608$ active cells, five seeds). V2/PCG2 timings include hierarchy setup plus solve; residual reduction $\|r_2\|_2/\|r_0\|_2$ (at $\sigma_{\log a}=1$) and operator error $\|B_h A_h e_h - e_h\|_2/\|e_h\|_2$ are reported as V2/PCG2 means.",
        "tab:opres-amg-scaling", size=r"\scriptsize", tabcolsep="2.0pt", notes=notes,
    )
    return t_ac, t_amg


def plot_fig7_fidelity_bands(t1):
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 8.0, "axes.titlesize": 8.8, "axes.labelsize": 8.0,
        "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "legend.fontsize": 6.8,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    s = t1["summary"]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.35), dpi=300)

    # Panel A: Within-field Spearman across 4 ID benchmarks
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

    # Panel B: Exact operator approximation error ||B_h A_h e_h - e_h|| / ||e_h||
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
    ax.axhline(1.0, color="#c0392b", ls="--", lw=1.0, label=r"Point-corr. limit ($\delta_{\mathrm{inv}}=1$)")
    ax.set_yscale("log")
    ax.set_ylim(0.09, 25.0)
    ax.set_xticks(x2)
    ax.set_xticklabels([d[2] for d in doms])
    ax.set_ylabel(r"$\|B_h A_h e_h - e_h\|_2 / \|e_h\|_2$")
    ax.set_title("(B) Exact operator defect (Prop. 1)")
    ax.legend(frameon=False, loc="upper right", fontsize=6.1, handlelength=1.0)

    # Panel C: Darcy 64^2 frequency-band correlation vs error energy
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
    # Annotate error energy % cleanly above each bar group
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
    for out_dir in (FIG_OUT, PAPER_FIG):
        fig.savefig(out_dir / "fig7_fidelity_bands.pdf", bbox_inches="tight")
        fig.savefig(out_dir / "fig7_fidelity_bands.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig8_stress_and_amg(t3, t5):
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 8.0, "axes.titlesize": 8.8, "axes.labelsize": 8.0,
        "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "legend.fontsize": 6.8,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.35), dpi=300)

    # Panel A: Burgers continuous shift coverage curve
    ax = axes[0]
    ahfs = t3["alphas_hf"]
    pa = t3["part_a_continuous_shift"]["summary"]
    styles = [
        ("opres_locked", r"OpRes ($\lambda=0.85$)", "#1b9e77", "-o"),
        ("opres_robust", r"OpRes ($\lambda=0.25, \ell=0.32$)", "#386cb0", "-s"),
        ("physics_scalar", r"Physics-Scalar ($\lambda=0$)", "#7570b3", "--^"),
        ("ensemble_mean", "5-Ensemble (mean)", "#d95f02", ":d"),
    ]
    for mk, lab, col, fmt in styles:
        ys = [100.0 * pa[str(a)]["methods"][mk]["coverage"]["mean"] for a in ahfs]
        es = [100.0 * pa[str(a)]["methods"][mk]["coverage"]["sd"] for a in ahfs]
        ax.errorbar(ahfs, ys, yerr=es, fmt=fmt, color=col, lw=1.5, ms=4.2, capsize=2.0, label=lab)
    ax.axhline(90.0, ls="--", lw=0.8, color="#444444")
    ax.set_xlabel(r"High-frequency injection $\alpha_{\mathrm{hf}}$")
    ax.set_ylabel("Simultaneous coverage (%)")
    ax.set_title("(A) Burgers continuous spectral shift")
    ax.set_ylim(20, 102)
    ax.legend(frameon=False, loc="lower left", fontsize=6.5)

    # Panel B: Noisy spectral overlap AUROC (ID-only overlap & Ambient noise)
    ax = axes[1]
    etas = t3["etas_noise"]
    pb_amb = t3["part_b_noisy_spectral_overlap"]["summary"]["ambient_noise"]
    pb_id = t3["part_b_noisy_spectral_overlap"]["summary"]["id_only_noise"]
    g_styles = [
        (pb_amb, "pure_spectral_scalar", r"Ambient: Pure spectral", "#999999", "--o"),
        (pb_amb, "joint_input_physics_cal_only", r"Ambient: Input + physics", "#1b9e77", "-^"),
        (pb_id, "pure_spectral_logistic", r"ID-noise: Pure input log.", "#d95f02", ":s"),
        (pb_id, "joint_input_physics_logistic", r"ID-noise: Input + physics", "#386cb0", "-.d"),
    ]
    for pb_src, gk, lab, col, fmt in g_styles:
        ys = [pb_src[str(e)][gk]["auroc"]["mean"] for e in etas]
        es = [pb_src[str(e)][gk]["auroc"]["sd"] for e in etas]
        ax.errorbar(etas, ys, yerr=es, fmt=fmt, color=col, lw=1.5, ms=4.2, capsize=2.0, label=lab)
    ax.set_xlabel("Modes 9\u201316 background noise $\\eta_{\\mathrm{noise}}$")
    ax.set_ylabel("OOD detection AUROC")
    ax.set_title("(B) Noisy spectral overlap detection")
    ax.set_ylim(0.45, 1.03)
    ax.legend(frameon=False, loc="lower left", fontsize=6.2)

    # Panel C: L-shape AMG multiscale runtime scaling
    ax = axes[2]
    sb = t5["part_b_amg_scaling"]
    sc = sb["scaling_at_sigma_1.0"]
    sl = sb["log_log_slopes"]["n_64_to_512"]
    ms = [sc[str(n)]["complexity"]["active_cells_finest"] for n in sb["grids"]]
    for key, lab, col, fmt, slk in (
        ("amg_v2_solve_p50_ms", f"AMG-V2 solve ($\\beta={sl['amg_v2_solve_p50_ms']:.2f}$)", "#66c2a5", "--o", "amg_v2_solve_p50_ms"),
        ("amg_v2_total_p50_ms", f"AMG-V2 total ($\\beta={sl['amg_v2_total_p50_ms']:.2f}$)", "#1b9e77", "-s", "amg_v2_total_p50_ms"),
        ("amg_pcg2_total_p50_ms", f"AMG-PCG2 total ($\\beta={sl['amg_pcg2_total_p50_ms']:.2f}$)", "#006837", "-^", "amg_pcg2_total_p50_ms"),
        ("spsolve_p50_ms", f"Exact spsolve ($\\beta={sl['spsolve_p50_ms']:.2f}$)", "#444444", ":d", "spsolve_p50_ms"),
    ):
        ts = [sc[str(n)][key] for n in sb["grids"]]
        ax.loglog(ms, ts, fmt, color=col, lw=1.5, ms=4.5, label=lab)
    ax.set_xlabel("Active cells $M = \\frac{3}{4}n^2$ ($32^2$\u2013$512^2$)")
    ax.set_ylabel("CPU latency P50 (ms)")
    ax.set_title(r"(C) L-shape AMG scaling ($C_G,C_O < 4/3$)")
    ax.legend(frameon=False, loc="upper left", fontsize=6.4)

    fig.tight_layout(w_pad=1.6)
    for out_dir in (FIG_OUT, PAPER_FIG):
        fig.savefig(out_dir / "fig8_stress_and_amg.pdf", bbox_inches="tight")
        fig.savefig(out_dir / "fig8_stress_and_amg.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    t1 = _read("task1_fidelity.json")
    t2 = _read("task2_lsci_fair.json")
    t3 = _read("task3_burgers_stress.json")
    t4 = _read("task4_delta_w.json")
    t5 = _read("task5_ac_amg.json")

    t_fid_id, t_rec, t_op, t_bands, t_fid_ood = build_task1_tables(t1)
    update_promoted_and_revision_tables(t2)
    t3_a, t3_b = build_task3_tables(t3)
    t4_dw = build_task4_table(t4)
    t5_ac, t5_amg = build_task5_tables(t5)

    # Promoted main-body tables: fidelity (ID) + signed recovery + operator approx
    _write_table_file("phase3_rev2_main_tables.tex", [t_fid_id, t_rec, t_op, t_bands], also_sync_rev=True)
    # Appendix tables: OOD fidelity + Burgers stress + Theorem 2 delta_w + AC sensitivity + AMG scaling
    _write_table_file("phase3_rev2_appendix_tables.tex", [t_fid_ood, t3_a, t3_b, t4_dw, t5_ac, t5_amg], also_sync_rev=True)

    plot_fig7_fidelity_bands(t1)
    plot_fig8_stress_and_amg(t3, t5)
    print("[Rev2 Tables & Figures] Generated all tables and figs 7-8 cleanly.", flush=True)


if __name__ == "__main__":
    main()
