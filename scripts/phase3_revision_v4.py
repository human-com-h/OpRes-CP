"""Phase 3 Revision v4 (CMAME Final): Unified orchestrator for Tasks 1-3 + Tables/Figures.

Usage:
  python -X utf8 scripts/phase3_revision_v4.py
  python -X utf8 scripts/phase3_revision_v4.py --tasks 2 3 tf

Strictly read-only against results/phase2_kaggle_private/.
All new artifacts written under results/phase3_revision_v4/.
Formal seeds locked to [42, 137, 314, 271, 811].
"""

import argparse
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from rev4_task1_elasticity_fem import run_task1_elasticity
from rev4_task2_nonlinear_pcg_jha import run_task2
from rev4_task3_physics_gate import run_task3_physics_gate


def main():
    parser = argparse.ArgumentParser(description="OpRes-CP Revision v4 (CMAME).")
    parser.add_argument("--tasks", nargs="+", default=["1", "2", "3", "tf"],
                        choices=["1", "2", "3", "tf"], help="Subset of steps to execute.")
    parser.add_argument("--workers", type=int, default=6, help="CPU process pool workers.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"[Revision v4] Running steps {args.tasks} on device={device}", flush=True)
    t0 = perf_counter()

    if "1" in args.tasks:
        run_task1_elasticity(device, workers=args.workers)
    if "2" in args.tasks:
        run_task2(device, workers=args.workers)
    if "3" in args.tasks:
        run_task3_physics_gate(device)
    if "tf" in args.tasks:
        from rev4_tables_figures import main as build_tf
        build_tf()

    print(f"[Revision v4] Selected steps finished in {perf_counter() - t0:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
