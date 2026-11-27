"""P2-B: 5-seed GPU-resident end-to-end Allen-Cahn benchmark (T=1 and T=2, 64^2 and 128^2).

Uses the exact five sealed frozen Allen-Cahn FNO and U-Net checkpoints under
results/phase2_kaggle_private/phase2d_frozen/ plus sealed 5-member ensemble and MC-Dropout
checkpoints under results/phase2_kaggle_private/downloads/2dc/ (read-only; zero retraining).
Implements full GPU-resident Strang reaction lift + separable periodic Gaussian envelope +
quadratic mixture, verifies numerical parity vs CPU SciPy reference across all 5 seeds, and
benchmarks end-to-end single-case and batched latency vs Ensemble and MC-Dropout.

Outputs written under results/phase3_p2/ac_gpu_timing/.
"""

import json
import math
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
from torch.nn import functional as F

torch.set_num_threads(2)

from allen_cahn_phase2_data import initial, solve, spectrum
from allen_cahn_phase2_probe import reconstruct, scale
from phase2d_data import Protocol
from phase3_allen_cahn import strang_gpu
from phase3_summarize import DOWNLOADS, SEEDS, STAGE, stat


OUT = Path("results/phase3_p2/ac_gpu_timing")
REV_CACHE = Path("results/phase3_revision/cache/allen_cahn")


def _gaussian_kernel_1d(sigma, device, dtype, truncate=4.0):
    """Exact 1D Gaussian kernel matching scipy.ndimage.gaussian_filter(truncate=4.0)."""
    radius = int(truncate * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    return k, radius


def scale_gpu(z, config):
    """GPU-resident periodic 2D Gaussian envelope + relative RMS floor + quadratic mixture.

    Matches allen_cahn_phase2_probe.scale(z, config) numerically to float32 precision.
    z shape: (B, T, H, W).
    """
    b, t, h, w = z.shape
    sq = z * z
    avg = sq.mean(dim=(1, 2, 3), keepdim=True)
    sigma = config["spatial_radius"] * w / (2.0 * math.pi)
    k1d, rad = _gaussian_kernel_1d(sigma, z.device, z.dtype, truncate=4.0)
    x = sq.reshape(b * t, 1, h, w)
    # Separable periodic (circular) 2D Gaussian filter
    x = F.pad(x, (rad, rad, 0, 0), mode="circular")
    x = F.conv2d(x, k1d.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, rad, rad), mode="circular")
    x = F.conv2d(x, k1d.view(1, 1, -1, 1))
    smooth = x.reshape(b, t, h, w) + 0.01 * avg
    lam = config["lambda"]
    return torch.sqrt((1.0 - lam) * 1.01 * avg + lam * smooth)


@torch.inference_mode()
def rollout_gpu_tensor(model, u0, nu, rho, frames):
    """Rollout keeping trajectory resident on GPU, shape (B, frames+1, H, W)."""
    y = u0
    traj = [y]
    for _ in range(frames):
        y = model(y, nu, rho)
        traj.append(y)
    return torch.stack(traj, dim=1)


def load_2dc_baselines(protocol, seed, device):
    """Load the 5 sealed ensemble checkpoints and MC-Dropout checkpoint from downloads/2dc."""
    base = DOWNLOADS / "2dc" / f"seed_{seed}" / "results" / "phase2d" / "allen_cahn" / f"seed_{seed}" / "grid64"
    ens_models = []
    # Member 0 is the frozen single FNO; members 1..4 are ensemble_1..4 (or ensemble_0..4)
    ens_dirs = sorted(base.glob("ensemble_*"))
    if len(ens_dirs) == 5:
        for d in ens_dirs:
            pts = sorted(d.glob("*.pt"))
            m = protocol.make_model().to(device)
            m.load_state_dict(torch.load(pts[0], map_location=device, weights_only=True)["model"])
            ens_models.append(m.eval())
    else:
        ens_models.append(protocol.open_frozen())
        for d in ens_dirs:
            pts = sorted(d.glob("*.pt"))
            m = protocol.make_model().to(device)
            m.load_state_dict(torch.load(pts[0], map_location=device, weights_only=True)["model"])
            ens_models.append(m.eval())
    assert len(ens_models) == 5, f"Expected 5 ensemble models for seed {seed}, got {len(ens_models)}"

    mc_dir = base / "mc_dropout"
    mc_pts = sorted(mc_dir.glob("*.pt"))
    mc_model = protocol.make_model(dropout=0.1).to(device)
    mc_model.load_state_dict(torch.load(mc_pts[0], map_location=device, weights_only=True)["model"])
    mc_model.eval()
    mc_model.dropout.train()
    return ens_models, mc_model


def time_fn(fn, device, warmup=4, reps=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    times = []
    for _ in range(reps):
        torch.cuda.synchronize(device)
        t0 = perf_counter()
        fn()
        torch.cuda.synchronize(device)
        times.append(1000.0 * (perf_counter() - t0))
    return float(np.median(times)), float(np.quantile(times, 0.9))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA GPU required for P2-B GPU-resident timing."
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"[P2-B] Running 5-seed GPU-resident Allen-Cahn benchmark on {torch.cuda.get_device_name(device)}", flush=True)

    per_seed = {}
    for seed in SEEDS:
        frozen = STAGE / "phase2d_frozen" / f"seed_{seed}"
        p = Protocol("allen_cahn", seed, 64, OUT / f"tmp_seed_{seed}", frozen, device)
        fno = p.open_frozen(unet=False)
        unet = p.open_frozen(unet=True)
        ens_models, mc_model = load_2dc_baselines(p, seed, device)
        cfg = p.config

        # Load cached 64^2 ID inputs (first 8 samples) and generate matching 128^2 initial conditions
        with np.load(REV_CACHE / f"seed_{seed}" / "data" / "id.npz") as raw:
            u0_64 = raw["u64"][:8, 0].astype(np.float32)
            nu = raw["nu"][:8].astype(np.float32)
            rho = raw["rho"][:8].astype(np.float32)
        # Upsample u0_64 smoothly to 128^2 via spectral zero-padding (exact bandlimited Fourier interp)
        hat = np.fft.rfft2(u0_64)
        hat128 = np.zeros((len(u0_64), 128, 65), dtype=np.complex128)
        hat128[:, :32, :32] = hat[:, :32, :32] * 4.0
        hat128[:, -32:, :32] = hat[:, -32:, :32] * 4.0
        u0_128 = np.fft.irfft2(hat128, s=(128, 128)).astype(np.float32)

        seed_rec = {"parity": {}, "latency_single_ms": {}, "latency_batch8_ms": {}}

        for grid, u0_arr in ((64, u0_64), (128, u0_128)):
            for frames, h_label in ((20, "T1"), (40, "T2")):
                key = f"{grid}_{h_label}"
                u0_1 = torch.as_tensor(u0_arr[:1], device=device)
                nu_1 = torch.as_tensor(nu[:1], device=device)
                rho_1 = torch.as_tensor(rho[:1], device=device)
                u0_8 = torch.as_tensor(u0_arr[:8], device=device)
                nu_8 = torch.as_tensor(nu[:8], device=device)
                rho_8 = torch.as_tensor(rho[:8], device=device)

                # 1. Numerical parity check on batch of 8 (FNO)
                full_gpu = rollout_gpu_tensor(fno, u0_8, nu_8, rho_8, frames)
                z_gpu = strang_gpu(full_gpu, nu_8, rho_8, cfg)
                s_gpu = scale_gpu(z_gpu, cfg).cpu().numpy()
                z_gpu_np = z_gpu.cpu().numpy()

                full_cpu = full_gpu.cpu().numpy()
                z_cpu, _ = reconstruct(full_cpu, nu[:8], rho[:8], cfg, "reaction_strang")
                s_cpu = scale(z_cpu, cfg)
                lift_err = float(np.linalg.norm(z_gpu_np - z_cpu) / np.linalg.norm(z_cpu))
                scale_err = float(np.linalg.norm(s_gpu - s_cpu) / np.linalg.norm(s_cpu))
                seed_rec["parity"][key] = {"lift_rel_l2": lift_err, "scale_rel_l2": scale_err}
                assert scale_err < 1e-4, f"Parity failure on seed {seed} {key}: {scale_err}"

                # 2. Single-case end-to-end latency (copy-in -> rollout -> scale -> copy-out)
                def opres_cpu_path():
                    tr = rollout_gpu_tensor(fno, u0_1, nu_1, rho_1, frames).cpu().numpy()
                    z, _ = reconstruct(tr, nu[:1], rho[:1], cfg, "reaction_strang")
                    return scale(z, cfg)

                def opres_gpu_path():
                    tr = rollout_gpu_tensor(fno, u0_1, nu_1, rho_1, frames)
                    z = strang_gpu(tr, nu_1, rho_1, cfg)
                    return scale_gpu(z, cfg).cpu()

                def opres_gpu_post_only():
                    z = strang_gpu(full_gpu[:1], nu_1, rho_1, cfg)
                    return scale_gpu(z, cfg)

                def ens5_gpu_path():
                    preds = torch.stack([rollout_gpu_tensor(m, u0_1, nu_1, rho_1, frames)[:, 1:] for m in ens_models], dim=0)
                    return preds.mean(dim=0).cpu(), preds.std(dim=0, unbiased=True).cpu()

                def mc30_gpu_path():
                    outs = []
                    for _ in range(30):
                        mc_model.masks.clear()
                        mc_model.fixed_draw = True
                        outs.append(rollout_gpu_tensor(mc_model, u0_1, nu_1, rho_1, frames)[:, 1:])
                    mc_model.fixed_draw = False
                    st = torch.stack(outs, dim=0)
                    return st.mean(dim=0).cpu(), st.std(dim=0, unbiased=True).cpu()

                def unet_gpu_path():
                    tr = rollout_gpu_tensor(unet, u0_1, nu_1, rho_1, frames)
                    z = strang_gpu(tr, nu_1, rho_1, cfg)
                    return scale_gpu(z, cfg).cpu()

                def unet_cpu_path():
                    tr = rollout_gpu_tensor(unet, u0_1, nu_1, rho_1, frames).cpu().numpy()
                    z, _ = reconstruct(tr, nu[:1], rho[:1], cfg, "reaction_strang")
                    return scale(z, cfg)

                # Batch-8 throughput (per-trajectory ms)
                def opres_gpu_b8():
                    tr = rollout_gpu_tensor(fno, u0_8, nu_8, rho_8, frames)
                    z = strang_gpu(tr, nu_8, rho_8, cfg)
                    return scale_gpu(z, cfg).cpu()

                def ens5_gpu_b8():
                    preds = torch.stack([rollout_gpu_tensor(m, u0_8, nu_8, rho_8, frames)[:, 1:] for m in ens_models], dim=0)
                    return preds.mean(dim=0).cpu(), preds.std(dim=0, unbiased=True).cpu()

                gpu_reps = 12
                cpu_reps = 7 if (grid == 128 and frames == 40) else 10
                mc_reps = 5

                row = {}
                for tag, fn, r_cnt in (
                    ("fno_opres_cpu_post", opres_cpu_path, cpu_reps),
                    ("fno_opres_gpu_full", opres_gpu_path, gpu_reps),
                    ("fno_gpu_post_only", opres_gpu_post_only, gpu_reps),
                    ("fno_ens5_gpu", ens5_gpu_path, gpu_reps),
                    ("fno_mc30_gpu", mc30_gpu_path, mc_reps),
                    ("unet_opres_cpu_post", unet_cpu_path, cpu_reps),
                    ("unet_opres_gpu_full", unet_gpu_path, gpu_reps),
                ):
                    p50, p90 = time_fn(fn, device, warmup=2, reps=r_cnt)
                    row[tag] = {"p50_ms": p50, "p90_ms": p90}
                seed_rec["latency_single_ms"][key] = row

                b8_op_p50, b8_op_p90 = time_fn(opres_gpu_b8, device, warmup=2, reps=gpu_reps)
                b8_en_p50, b8_en_p90 = time_fn(ens5_gpu_b8, device, warmup=2, reps=gpu_reps)
                seed_rec["latency_batch8_ms"][key] = {
                    "fno_opres_gpu_per_case_p50_ms": b8_op_p50 / 8.0,
                    "fno_ens5_gpu_per_case_p50_ms": b8_en_p50 / 8.0,
                    "speedup_ens_over_opres_b8": b8_en_p50 / b8_op_p50,
                }
                print(f"  [P2-B] seed={seed} {key}: CPU={row['fno_opres_cpu_post']['p50_ms']:.1f}ms, "
                      f"GPU={row['fno_opres_gpu_full']['p50_ms']:.1f}ms, "
                      f"Ens5={row['fno_ens5_gpu']['p50_ms']:.1f}ms, "
                      f"MC30={row['fno_mc30_gpu']['p50_ms']:.1f}ms, "
                      f"b8={b8_op_p50/8.0:.1f} vs {b8_en_p50/8.0:.1f}ms, parity={scale_err:.2e}", flush=True)

        per_seed[str(seed)] = seed_rec
        r128 = seed_rec["latency_single_ms"]["128_T2"]
        print(f"[P2-B] seed={seed} 128^2 T=2 FNO: CPU-post={r128['fno_opres_cpu_post']['p50_ms']:.1f}ms, "
              f"GPU-full={r128['fno_opres_gpu_full']['p50_ms']:.1f}ms, "
              f"Ens5={r128['fno_ens5_gpu']['p50_ms']:.1f}ms, "
              f"MC30={r128['fno_mc30_gpu']['p50_ms']:.1f}ms | "
              f"parity={seed_rec['parity']['128_T2']['scale_rel_l2']:.2e}", flush=True)

    # Aggregate across 5 seeds
    agg = {"device": torch.cuda.get_device_name(device), "seeds": list(SEEDS),
           "parity": {}, "single_case_ms": {}, "batch8_per_case_ms": {}, "per_seed": per_seed}
    for key in ("64_T1", "64_T2", "128_T1", "128_T2"):
        agg["parity"][key] = {
            m: {"max": float(max(per_seed[str(s)]["parity"][key][m] for s in SEEDS)),
                "mean": float(np.mean([per_seed[str(s)]["parity"][key][m] for s in SEEDS]))}
            for m in ("lift_rel_l2", "scale_rel_l2")
        }
        agg["single_case_ms"][key] = {}
        tags = list(per_seed[str(SEEDS[0])]["latency_single_ms"][key].keys())
        for tag in tags:
            vals = [per_seed[str(s)]["latency_single_ms"][key][tag]["p50_ms"] for s in SEEDS]
            _, avg, sd, lo, hi = stat(vals)
            agg["single_case_ms"][key][tag] = {"mean": avg, "sd": sd, "ci95": [lo, hi]}
        # Speedups from 5-seed means
        s = agg["single_case_ms"][key]
        s["speedup_gpu_over_cpu_fno"] = s["fno_opres_cpu_post"]["mean"] / s["fno_opres_gpu_full"]["mean"]
        s["speedup_gpu_over_ens5_fno"] = s["fno_ens5_gpu"]["mean"] / s["fno_opres_gpu_full"]["mean"]
        s["speedup_gpu_over_mc30_fno"] = s["fno_mc30_gpu"]["mean"] / s["fno_opres_gpu_full"]["mean"]
        s["speedup_gpu_over_cpu_unet"] = s["unet_opres_cpu_post"]["mean"] / s["unet_opres_gpu_full"]["mean"]

        b8_op = [per_seed[str(s)]["latency_batch8_ms"][key]["fno_opres_gpu_per_case_p50_ms"] for s in SEEDS]
        b8_en = [per_seed[str(s)]["latency_batch8_ms"][key]["fno_ens5_gpu_per_case_p50_ms"] for s in SEEDS]
        agg["batch8_per_case_ms"][key] = {
            "fno_opres_gpu_mean": float(np.mean(b8_op)),
            "fno_opres_gpu_sd": float(np.std(b8_op, ddof=1)),
            "fno_ens5_gpu_mean": float(np.mean(b8_en)),
            "fno_ens5_gpu_sd": float(np.std(b8_en, ddof=1)),
            "speedup_over_ens5": float(np.mean(b8_en) / np.mean(b8_op)),
        }

    (OUT / "ac_gpu_benchmark.json").write_text(json.dumps(agg, indent=2) + "\n", encoding="utf-8")
    print(f"[P2-B] COMPLETE! Saved to {OUT / 'ac_gpu_benchmark.json'}", flush=True)


if __name__ == "__main__":
    main()
