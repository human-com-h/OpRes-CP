"""Reference the already-hosted private Phase 2 outputs in a Kaggle worker."""

import hashlib
import json
import os
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reference_records(pde, source, contents):
    if pde == "darcy":
        return [(f"darcy{grid}", source / f"grid{grid}/FNO/member0/fno.pt",
                 contents["tracks"][str(grid)]["FNO"]["weight_sha256"][0],
                 contents["tracks"][str(grid)]["FNO"]["training_seconds"][0]) for grid in (32, 64)]
    if pde == "burgers":
        history = read(source / "FNO/train_log.json")["history"]
        return [(pde, source / "FNO/fno.pt", None, sum(h["seconds"] for h in history))]
    entries = []
    for model in ("FNO", "U-Net"):
        training = contents["results"][model]["training"]
        entries.append((pde if model == "FNO" else "allen_cahn_unet", source / model / "best.pt",
                        training["weight_sha256"], sum(h["elapsed_s"] for h in training["history"])))
    return entries


def mounted_snapshot(seed, pde, mount=Path("/kaggle/input"), output=Path("results/phase2d_mounts")):
    expected = json.loads(os.environ["OPRES_CP_EXPECTED_WEIGHTS"])
    candidates = [path for path in mount.rglob("metrics.json")
                  if path.parent.name == f"seed_{seed}" and path.parent.parent.name == f"phase2_{pde}"]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one mounted Phase 2 source for {pde}, found {len(candidates)}.")
    reference = candidates[0]
    contents = read(reference)
    if contents["seed"] != seed:
        raise ValueError("Mounted frozen reference has the wrong seed.")
    manifest = {"seed":seed,"weights":{},"training_seconds":{},"references":{pde:str(reference.resolve())}}
    for key, path, recorded, seconds in reference_records(pde,reference.parent,contents):
        checksum = sha(path)
        if expected.get(key) != checksum or (recorded and recorded != checksum):
            raise ValueError(f"Mounted {key} checkpoint does not match the accepted Phase 2 weights.")
        manifest["weights"][key] = {"file":str(path.resolve()),"sha256":checksum}
        manifest["training_seconds"][key] = seconds
    if manifest["weights"].keys()!=expected.keys():
        raise ValueError("Mounted reference does not supply every locked checkpoint.")
    root = output / pde / f"seed_{seed}"
    root.mkdir(parents=True,exist_ok=True)
    (root/"frozen_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return root
