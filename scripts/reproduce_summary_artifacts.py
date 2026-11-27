#!/usr/bin/env python3
"""Tier 2: Zero-retraining reproduction of publication tables and diagnostic figures.

Reads the committed five-seed summary JSON files under results/ and regenerates
LaTeX booktabs tables and vector PDF/PNG figures into outputs/tables and
outputs/figures (without requiring any downloaded .pt checkpoint weights).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = [
    "rev2_tables_figures.py",
    "rev3_tables_figures.py",
    "rev4_tables_figures.py",
]


def main() -> None:
    t0 = perf_counter()
    for name in SCRIPTS:
        script_path = ROOT / "scripts" / name
        print(f"[Tier 2] Running scripts/{name} ...", flush=True)
        res = subprocess.run(
            [sys.executable, "-X", "utf8", str(script_path)],
            cwd=str(ROOT),
            check=False,
        )
        if res.returncode != 0:
            raise SystemExit(f"[Tier 2 FAIL] scripts/{name} exited with code {res.returncode}")

    out_tbl = ROOT / "outputs" / "tables"
    out_fig = ROOT / "outputs" / "figures"
    tables = sorted(p.name for p in out_tbl.glob("*.tex")) if out_tbl.is_dir() else []
    figs = sorted(p.name for p in out_fig.glob("*.*")) if out_fig.is_dir() else []

    expected_tables = {
        "phase3_rev2_main_tables.tex",
        "phase3_rev2_appendix_tables.tex",
        "phase3_rev3_tables.tex",
        "rev4_main_tables.tex",
        "rev4_appendix_tables.tex",
        "rev4_sec2_positioning.tex",
    }
    expected_figs = {
        "fig7_fidelity_bands.pdf",
        "fig8_stress_and_amg.pdf",
        "fig9_rev3_diagnostics.pdf",
        "fig10_elasticity_fem.pdf",
        "graphical_abstract.pdf",
    }
    missing_t = sorted(expected_tables - set(tables))
    missing_f = sorted(expected_figs - set(figs))
    if missing_t or missing_f:
        raise SystemExit(f"[Tier 2 FAIL] Missing outputs: tables={missing_t}, figures={missing_f}")

    dt = perf_counter() - t0
    print(f"\n[Tier 2 PASS] Rebuilt {len(tables)} LaTeX tables -> outputs/tables/ and {len(figs)} figure assets -> outputs/figures/ in {dt:.2f}s.")


if __name__ == "__main__":
    main()
