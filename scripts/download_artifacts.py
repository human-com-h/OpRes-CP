#!/usr/bin/env python3
"""Tier 3 helper: Download and SHA-256 verify frozen weights & replay assets.

Supports GitHub Releases (default: human-com-h/OpRes-CP v1.0.0) or Zenodo record
file URLs, verifies SHA-256 checksums against SHA256SUMS.txt, and safely extracts
archives into the repository root for full Appendix B.3 frozen replay.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path


DEFAULT_REPO = "human-com-h/OpRes-CP"
DEFAULT_RELEASE = "v1.0.0"
ARCHIVES = {
    "phase2d": "opres_cp_phase2d_frozen_weights.tar.gz",
    "lshape_elasticity": "opres_cp_lshape_elasticity_weights.tar.gz",
}


def load_checksums(path: Path) -> dict[str, str]:
    sums: dict[str, str] = {}
    if not path.is_file():
        return sums
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            digest, rel = parts[0].lower(), parts[1].strip().lstrip("./")
            sums[Path(rel).name] = digest
            sums[rel] = digest
    return sums


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"[download] {url} -> {dst}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "OpRes-CP-ArtifactDownloader/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
        total = resp.headers.get("Content-Length")
        total_mb = int(total) / (1024 * 1024) if total and total.isdigit() else None
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total_mb:
                pct = 100.0 * done / (total_mb * 1024 * 1024)
                print(f"\r  {done / (1024 * 1024):.1f} / {total_mb:.1f} MB ({pct:.1f}%)", end="", flush=True)
        print()
    tmp.replace(dst)


def safe_extract_tar(archive: Path, dest: Path) -> int:
    dest_resolved = dest.resolve()
    count = 0
    with tarfile.open(archive, "r:gz") as tf:
        members = tf.getmembers()
        for m in members:
            member_path = (dest_resolved / m.name).resolve()
            if not str(member_path).startswith(str(dest_resolved)):
                raise RuntimeError(f"Blocked unsafe tar member path: {m.name}")
            if m.issym() or m.islnk():
                raise RuntimeError(f"Blocked symlink/hardlink in tar archive: {m.name}")
        tf.extractall(dest_resolved)
        count = sum(1 for m in members if m.isfile())
    return count


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Download & verify OpRes-CP frozen weights.")
    parser.add_argument("--source", choices=("github", "zenodo"), default="github")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub owner/repo")
    parser.add_argument("--release", default=DEFAULT_RELEASE, help="GitHub release tag")
    parser.add_argument("--zenodo-base-url", default="", help="Zenodo record files base URL (e.g. https://zenodo.org/records/<ID>/files)")
    parser.add_argument("--archives", nargs="+", choices=("all", "phase2d", "lshape_elasticity"), default=["all"])
    parser.add_argument("--sha256-file", type=Path, default=repo_root / "SHA256SUMS.txt")
    parser.add_argument("--dest", type=Path, default=repo_root)
    parser.add_argument("--cache-dir", type=Path, default=repo_root / ".artifact_cache")
    parser.add_argument("--verify-only", action="store_true", help="Verify already downloaded archives in --cache-dir without downloading")
    parser.add_argument("--keep-archives", action="store_true", help="Retain .tar.gz files in --cache-dir after extraction")
    args = parser.parse_args()

    selected = list(ARCHIVES.keys()) if "all" in args.archives else list(dict.fromkeys(args.archives))
    checksums = load_checksums(args.sha256_file)

    for key in selected:
        fname = ARCHIVES[key]
        local_tar = args.cache_dir / fname
        if not args.verify_only and not local_tar.is_file():
            if args.source == "github":
                url = f"https://github.com/{args.repo}/releases/download/{args.release}/{fname}"
            else:
                if not args.zenodo_base_url:
                    raise SystemExit("--zenodo-base-url is required when --source zenodo")
                base = args.zenodo_base_url.rstrip("/")
                url = f"{base}/{fname}?download=1"
            download_file(url, local_tar)

        if not local_tar.is_file():
            raise SystemExit(f"Archive not found: {local_tar}")

        actual = sha256_file(local_tar)
        expected = checksums.get(fname)
        if expected:
            if actual != expected:
                raise SystemExit(f"SHA-256 mismatch for {fname}: got {actual}, expected {expected}")
            print(f"[verify] {fname} SHA-256 OK ({actual[:16]}...)")
        else:
            print(f"[warn] No expected SHA-256 found for {fname}; computed {actual}")

        if not args.verify_only:
            n_files = safe_extract_tar(local_tar, args.dest)
            print(f"[extract] Extracted {n_files} files from {fname} into {args.dest}")
            if not args.keep_archives:
                local_tar.unlink()

    if not args.keep_archives and args.cache_dir.is_dir() and not any(args.cache_dir.iterdir()):
        shutil.rmtree(args.cache_dir, ignore_errors=True)
    print("[done] Tier 3 frozen artifact preparation complete.")


if __name__ == "__main__":
    main()
