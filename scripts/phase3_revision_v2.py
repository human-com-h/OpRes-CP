"""Phase 3 Revision v2: Unified zero-retraining orchestrator for Tasks 1-5.

Usage:
  python -X utf8 scripts/phase3_revision_v2.py              # run all tasks 1-5
  python -X utf8 scripts/phase3_revision_v2.py --tasks 1 2  # run selected tasks

Strictly read-only against results/phase2_kaggle_private/.
All new artifacts written under results/phase3_revision_v2/.
Formal seeds locked to [42, 137, 314, 271, 811].
"""

import argparse
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from rev2_task1_fidelity import run_task1
from rev2_task2_lsci import run_task2
from rev2_task3_burgers import run_task3
from rev2_task4_delta_w import run_task4
from rev2_task5_ac_amg import run_task5


def main():
    parser = argparse.ArgumentParser(description="OpRes-CP Phase 3 Revision v2 (zero-retraining audits).")
    parser.add_argument("--tasks", nargs="+", default=["1", "2", "3", "4", "5"],
                        choices=["1", "2", "3", "4", "5"], help="Subset of tasks to execute.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"[Revision v2] Running tasks {args.tasks} on device={device}", flush=True)
    t0 = perf_counter()

    if "1" in args.tasks:
        run_task1(device)
    if "2" in args.tasks:
        run_task2(device)
    if "3" in args.tasks:
        run_task3(device)
    if "4" in args.tasks:
        run_task4()
    if "5" in args.tasks:
        run_task5(device)

    print(f"[Revision v2] Selected tasks finished in {perf_counter() - t0:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
