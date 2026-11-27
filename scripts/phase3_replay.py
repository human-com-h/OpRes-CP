"""Re-evaluate frozen Phase 2 models on the locked sample streams."""

import argparse
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

from phase2d_data import Protocol
from phase2d_frozen import mounted_snapshot, sha
from phase3_common import PLAN, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pde", choices=("darcy", "burgers", "allen_cahn"), required=True)
    parser.add_argument("--seed", choices=PLAN["seeds"], type=int, required=True)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--limit", type=int, help="Local interface check using a prefix of each official split.")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name() if device.type == "cuda" else "CPU"
    if args.limit is None and gpu != "Tesla T4":
        parser.error("Full Phase 3 model replay uses the formal private Tesla T4 workers.")
    if args.limit is not None and not 4 <= args.limit < 500:
        parser.error("Use a sample prefix of between 4 and 499 for a short check.")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    frozen = args.frozen or mounted_snapshot(args.seed, args.pde)
    root = Path("results") / ("phase3" if args.limit is None else "phase3_smoke") / args.pde / f"seed_{args.seed}"
    root.mkdir(parents=True, exist_ok=True)
    protocol = Protocol(args.pde, args.seed, 64, root / "cache", frozen, device)
    if args.limit and args.pde == "burgers" and args.seed == PLAN["seeds"][0]:
        previous = Path("results/phase2d_cache/seed_42/burgers")
        for group in ("cal", "id", "ood_high"):
            path = previous / f"{group}.npz"
            if path.is_file():
                protocol.files[group, False] = path
    result, groups = import_module(f"phase3_{args.pde}").run(protocol, root, args.limit)
    for name, data in groups.items():
        ids = data["_id"]
        if len(ids) != (args.limit or 500) or len(np.unique(ids)) != len(ids):
            raise RuntimeError(f"Bad Phase 3 sample IDs for {name}")
    save(root, "metrics", {"pde": args.pde, "seed": args.seed, "gpu": gpu, "limit": args.limit,
         "plan_sha256": sha(Path("configs/phase3.json")), "data_hash_matches": protocol.data_matches,
         "weights": {name: row["sha256"] for name, row in protocol.manifest["weights"].items()}, **result}, groups)


if __name__ == "__main__":
    main()
