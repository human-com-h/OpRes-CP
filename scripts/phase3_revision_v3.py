"""Phase 3 Revision v3: Unified zero-retraining orchestrator for Tasks 1-3.

Usage:
  python -X utf8 scripts/phase3_revision_v3.py
  python -X utf8 scripts/phase3_revision_v3.py --tasks 2 3

Strictly read-only against results/phase2_kaggle_private/.
All new artifacts written under results/phase3_revision_v3/.
Formal seeds locked to [42, 137, 314, 271, 811].
"""

import argparse
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from rev3_task1_opnorm_gamma import run_task1
from rev3_task2_lsci_exact import run_task2
from rev3_task3_kernel_horizon import run_task3


def main():
    parser = argparse.ArgumentParser(description="OpRes-CP Phase 3 Revision v3 (zero-retraining P0 audits).")
    parser.add_argument("--tasks", nargs="+", default=["1", "2", "3"],
                        choices=["1", "2", "3"], help="Subset of tasks to execute.")
    parser.add_argument("--workers", type=int, default=6, help="CPU process pool workers.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"[Revision v3] Running tasks {args.tasks} on device={device}", flush=True)
    t0 = perf_counter()

    if "1" in args.tasks:
        run_task1(device, workers=args.workers)
    if "2" in args.tasks:
        run_task2(device)
    if "3" in args.tasks:
        run_task3(device, workers=args.workers)

    print(f"[Revision v3] Selected tasks finished in {perf_counter() - t0:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
