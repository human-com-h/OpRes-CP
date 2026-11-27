"""Independent long-window calibration and within-integrator reaction ablation."""

import argparse
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from allen_cahn_phase2_probe import reconstruct, scale
from allen_cahn_horizon_audit import samples
from darcy_day15 import arrays, digest, write_json
from darcy_phase0_eval import boot_interval, quantile
from phase2d import batches, cutoff_points, take
from phase2d_data import Protocol
from phase2d_frozen import mounted_snapshot
from phase2d_statistics import ALPHAS, calibrated_scores, conformal_values, summarize_values


STREAMS = {"cal": 9401, "id": 9402, "paired": 9403, "ood": 9404}
MODES = ("T1", "T1_Bonferroni", "T2_same_window")


def generate(protocol, split):
    streams = {s:n + (400 if protocol.smoke else 0) for s,n in STREAMS.items()}
    c = protocol.config | {"streams":streams}
    path = protocol.root / f"fresh_{split}.npz"
    if not path.exists():
        pending = path.with_suffix(".tmp.npz")
        np.savez(pending, **samples(c,split))
        pending.replace(path)
    with np.load(path) as data:
        expected = streams[split]*10000 + np.arange(c["counts"][split])
        if not np.array_equal(expected,data["id"]) or data["u64"].shape[1] != c["frames"]+1:
            raise ValueError(f"Independent {split} has unexpected trajectory IDs or window.")
    return path


def lift_for(full, nu, rho, config, enabled):
    return reconstruct(full,nu,rho,config,"reaction_strang",reaction_jacobian=enabled)[0]


def trajectory_scales(prediction, inputs, config):
    full = np.concatenate((inputs[0][:,None],prediction),axis=1).astype(float)
    return {enabled:lift_for(full,*inputs[1:],config,enabled) for enabled in (False,True)}


def cal_scores(protocol, base, data, frames):
    inputs, truth = protocol.split(data,64,frames)
    values = defaultdict(list)
    for sl in batches(len(truth),6):
        x = take(inputs,sl)
        prediction = protocol.predict(base,x,64,frames)
        for enabled, lifted in trajectory_scales(prediction,x,protocol.config).items():
            values[enabled].append(calibrated_scores(prediction,scale(lifted,protocol.config),truth[sl]))
    return {enabled:np.concatenate(rows) for enabled,rows in values.items()}


def quantiles(short, long):
    return {enabled:{
        mode:{str(a):quantile(scores,a*factor) for a in ALPHAS}
        for mode,scores,factor in zip(MODES,(short[enabled],short[enabled],long[enabled]),(1.,.5,1.))}
        for enabled in (False,True)}


def compare(protocol, model, data, n, q):
    c = protocol.config
    frames = protocol.frames
    x, y = protocol.split(data,n,frames)
    raw = defaultdict(list)
    blockwise = defaultdict(list)
    for sl in batches(len(y),6):
        inputs, truth = take(x,sl),y[sl]
        prediction = protocol.predict(model,inputs,n,frames)
        top = cutoff_points(truth,"allen_cahn")
        for enabled, lifted in trajectory_scales(prediction,inputs,c).items():
            sigma = scale(lifted,c)
            for mode, thresholds in q[enabled].items():
                for alpha, row in conformal_values(prediction,sigma,truth,top,thresholds).items():
                    raw[enabled,mode,alpha].append(row)
            short = protocol.cal_frames
            t = q[enabled]["T1_Bonferroni"]["0.1"]
            hits = []
            for window in (slice(0,short),slice(short,frames)):
                scores = calibrated_scores(prediction[:,window],scale(lifted[:,window],c),truth[:,window])
                hits.append(scores <= t)
            blockwise[enabled].append(np.column_stack((*hits,hits[0]&hits[1])))
    collected = {key:{k:np.concatenate([v[k] for v in parts]) for k in parts[0]} for key,parts in raw.items()}
    summary = {}
    for enabled,mode,alpha in collected:
        row = collected[enabled,mode,alpha]
        result = summarize_values(row,alpha)
        threshold = q[enabled][mode][alpha]
        result["threshold"] = threshold if np.isfinite(threshold) else "inf"
        if enabled:
            off = collected[False,mode,alpha]
            bump = row["coverage"].astype(float)-off["coverage"]
            result["paired_coverage_diff_vs_off"] = float(bump.mean())
            result["paired_coverage_ci95"] = boot_interval(bump,c["bootstrap_resamples"],c["bootstrap_seed"])
        summary.setdefault("jacobian_on" if enabled else "jacobian_off",{}).setdefault(mode,{})[alpha] = result
    for enabled,pieces in blockwise.items():
        v = np.concatenate(pieces).mean(axis=0)
        summary["jacobian_on" if enabled else "jacobian_off"]["separate_block_Bonferroni"] = {
            "first_coverage":float(v[0]),"second_coverage":float(v[1]),"joint_coverage":float(v[2])}
    return summary


def profile(protocol, model, data, n):
    x,_ = protocol.split(data,n,protocol.frames)
    prediction, reconstruct_ms, scale_ms = [], [], []
    for i in range(6 if protocol.smoke else 23):
        inputs = take(x,slice(i,i+1))
        start = perf_counter()
        pred = protocol.predict(model,inputs,n,protocol.frames)
        t1 = perf_counter()
        full = np.concatenate((inputs[0][:,None],pred),axis=1).astype(float)
        lifted = lift_for(full,*inputs[1:],protocol.config,True)
        t2 = perf_counter()
        scale(lifted,protocol.config)
        t3 = perf_counter()
        if i>=3:
            prediction.append((t1-start)*1000)
            reconstruct_ms.append((t2-t1)*1000)
            scale_ms.append((t3-t2)*1000)
    return {"rollout_including_d2h_p50_ms":float(np.median(prediction)),
            "cpu_lift_p50_ms":float(np.median(reconstruct_ms)),
            "cpu_envelope_p50_ms":float(np.median(scale_ms)),
            "unnecessary_cuda_sync_in_cpu_post":False}


def run(seed, frozen, device, smoke):
    root = Path("results") / ("phase2d_smoke" if smoke else "phase2d") / "allen_cahn_horizon" / f"seed_{seed}"
    p = Protocol("allen_cahn",seed,64,root,frozen,device,smoke)
    files = {name:generate(p,name) for name in STREAMS}
    original = p.data_path("cal")
    id_sets = []
    for path in [original,*files.values()]:
        with np.load(path) as contents:
            id_sets.append(contents["id"])
    combined = np.concatenate(id_sets)
    if len(np.unique(combined)) != len(combined):
        raise ValueError("Original and independent data stream IDs overlap.")
    cal1, cal2 = arrays(original), arrays(files["cal"])
    output, timing = {}, {}
    for name in ("FNO","U-Net"):
        model = p.open_frozen(unet=name=="U-Net")
        q = quantiles(cal_scores(p,model,cal1,p.cal_frames),cal_scores(p,model,cal2,p.frames))
        if not smoke:
            original_q = p.reference["results"][name]["thresholds"]["0.1"]["OpRes-CP"]
            replay = q[True]["T1"]["0.1"]
            if not np.isclose(original_q,replay,rtol=.01):
                raise RuntimeError(f"{name} short calibration did not reproduce within 1%.")
            for level in ("0.1","0.05"):
                accepted = p.reference["results"][name]["thresholds"][level]["OpRes-CP"]
                q[True]["T1"][level] = accepted
                if level == "0.05":
                    q[True]["T1_Bonferroni"]["0.1"] = accepted
        output[name], timing[name] = {}, {}
        for split in ("id","paired","ood"):
            data = arrays(files[split])
            for n in (64,128) if split=="paired" else (64,):
                output[name][f"{split}_{n}"] = compare(p,model,data,n,q)
                if split=="paired":
                    timing[name][str(n)] = profile(p,model,data,n)
                print(f"Phase 2D horizon seed {seed} {name} {split}_{n}",flush=True)
    write_json(root/"metrics.json",{"seed":seed,"smoke":smoke,"fresh_disjoint":True,
        "counts":list(map(len,id_sets)), "sha256":{name:digest(path) for name,path in files.items()},
        "results":output,"latency":timing})


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--seed",type=int,choices=(42,137,314,271,811),required=True)
    cli.add_argument("--frozen",type=Path)
    cli.add_argument("--smoke",action="store_true")
    args=cli.parse_args()
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.smoke and device.type!="cuda":
        raise RuntimeError("Formal Allen–Cahn horizon jobs require GPU acceleration.")
    run(args.seed,args.frozen or mounted_snapshot(args.seed,"allen_cahn"),device,args.smoke)
