"""Compare physical, ensemble, dropout, and localized scales on frozen Phase 2 tasks."""

import argparse
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from darcy_day15 import arrays, digest, write_json
from darcy_phase2 import ENSEMBLE, OP
from phase2d_data import Protocol
from phase2d_frozen import mounted_snapshot
from phase2d_statistics import (ALPHAS, LocalSlicedCP, calibrated_scores, conformal_values,
                                covariate_weights, flat, positive_scale, relative_l2, summarize_values, targets_for)


PDE = ("darcy", "burgers", "allen_cahn")
DROPOUT = "30-draw MC-Dropout + CP"
LSCI = "LSCI v5 sampled band"
WCP = ("Full-grid WCP", "Physical-summary WCP")


def moments(protocol, model_list, inputs, n, frames, draws=None):
    total, mean, m2 = 0, None, None
    for model in model_list if draws is None else [model_list[0]] * draws:
        if draws is not None:
            model.masks.clear()
        value = protocol.predict(model, inputs, n, frames)
        total += 1
        if mean is None:
            mean, m2 = value.copy(), np.zeros_like(value)
        else:
            delta = value - mean
            mean += delta / total
            m2 += delta * (value - mean)
    return mean, np.sqrt(np.maximum(m2 / (total - 1), 0))


def batches(n, size=8):
    return (slice(i, min(i + size, n)) for i in range(0, n, size))


def take(inputs, sl):
    return tuple(a[sl] for a in inputs)


def variance_floor(protocol, inputs, models, draws=None):
    squares, elements = 0., 0
    for sl in batches(len(inputs[0])):
        std = moments(protocol, models, take(inputs,sl), protocol.grid, protocol.cal_frames, draws)[1]
        squares += np.sum(std*std, dtype=float)
        elements += std.size
    return .1*np.sqrt(squares/elements)


def cutoff_points(truth, pde):
    if pde == "burgers":
        gradient = np.abs(np.roll(truth, 1, axis=-1)-np.roll(truth,-1,axis=-1))
    elif pde == "darcy":
        gradient = np.hypot(*np.gradient(truth,axis=(-2,-1)))
    else:
        dx = np.roll(truth,1,axis=-1)-np.roll(truth,-1,axis=-1)
        dy = np.roll(truth,1,axis=-2)-np.roll(truth,-1,axis=-2)
        gradient = np.hypot(dx,dy)
    v = flat(gradient)
    k = int(np.ceil(.1*v.shape[1]))
    return np.argpartition(v,-k,axis=1)[:,-k:]


def adapt_reference(residuals, target_shape, pde):
    ref = residuals
    if pde == "allen_cahn" and target_shape[0] != ref.shape[1]:
        if target_shape[0] != 2 * ref.shape[1]:
            raise ValueError("Unexpected target time horizon for LSCI.")
        ref = np.concatenate((ref,ref),axis=1)
    for axis in (-1,) if pde == "burgers" else (-2,-1):
        factor, rem = divmod(target_shape[axis],ref.shape[axis])
        if rem or factor < 1:
            raise ValueError("Unexpected calibration grid for LSCI.")
        if factor > 1:
            ref = np.repeat(ref,factor,axis=axis)
    return ref


def gaussian_floor(sigma, floor):
    return positive_scale(np.sqrt(sigma*sigma + floor*floor),floor)


class Baselines:
    def __init__(self, protocol, prepared, draws):
        self.protocol, self.draws = protocol, draws
        self.primary = protocol.open_frozen()
        self.ensemble, self.dropout = [], None
        stem = protocol.key
        self.training_seconds = {stem: protocol.manifest["training_seconds"][stem]}
        if "ensemble" in prepared:
            self.ensemble = [self.primary]
            for i in range(1,5):
                label = f"ensemble_{i}"
                model, seconds = protocol.train_member(label,protocol.seed+10000*i)
                self.ensemble.append(model)
                self.training_seconds[label] = seconds
        if "dropout" in prepared:
            self.dropout, seconds = protocol.train_member("mc_dropout",protocol.seed+50000,dropout=.1)
            self.dropout.dropout.train()
            self.dropout.fixed_draw = True
            self.training_seconds["mc_dropout"] = seconds
        val = arrays(protocol.data_path("val"))
        xval,_ = protocol.split(val,protocol.grid,protocol.cal_frames)
        self.floor = {}
        if self.ensemble:
            self.floor["ensemble"] = variance_floor(protocol,xval,self.ensemble)
        if self.dropout:
            self.floor["dropout"] = variance_floor(protocol,xval,[self.dropout],draws)
        cal = arrays(protocol.data_path("cal"))
        self.cal_ids = cal["id"].copy()
        xcal, ycal = protocol.split(cal,protocol.grid,protocol.cal_frames)
        self.full_cal, self.summary_cal = protocol.full_grid(xcal), protocol.summaries(xcal)
        self.spectral_cal = protocol.summaries(xcal,spectral=True) if protocol.pde=="burgers" else None
        self.scores = defaultdict(list)
        residuals = []
        torch.manual_seed(protocol.seed+81000)
        for sl in batches(len(ycal)):
            x = take(xcal,sl)
            maps = self.maps(x,protocol.grid,protocol.cal_frames)
            for method,(center,sigma) in maps.items():
                self.scores[method].append(calibrated_scores(center,sigma,ycal[sl]))
            if protocol.grid == 64:
                residuals.append((ycal[sl]-maps[OP][0]).astype(np.float32))
        self.scores = {name:np.concatenate(rows) for name,rows in self.scores.items()}
        self.residual_cal = np.concatenate(residuals) if residuals else None

    def maps(self, inputs, n, frames):
        p = self.protocol
        primary = p.predict(self.primary,inputs,n,frames)
        maps = {OP:(primary,positive_scale(p.op_scale(primary,inputs),1e-12))}
        if self.ensemble:
            center, std = moments(p,self.ensemble,inputs,n,frames)
            band = gaussian_floor(std,self.floor["ensemble"])
            maps[ENSEMBLE[0]],maps[ENSEMBLE[1]] = (primary,band),(center,band)
        if self.dropout:
            center,std = moments(p,[self.dropout],inputs,n,frames,self.draws)
            maps[DROPOUT] = (center,gaussian_floor(std,self.floor["dropout"]))
        return maps

    def prepare_target(self, inputs, shape, is_ood, n, frames):
        p = self.protocol
        cov = p.summaries(inputs)
        thresholds = {name:targets_for(score) for name,score in self.scores.items()}
        fits = {}
        if is_ood and n == p.grid and self.residual_cal is not None:
            measurements = {WCP[0]:(self.full_cal,p.full_grid(inputs)),
                            WCP[1]:(self.summary_cal,cov)}
            if self.spectral_cal is not None:
                measurements["Physical-summary + spectrum WCP"] = self.spectral_cal,p.summaries(inputs,True)
            for label,(cal_x,test_x) in measurements.items():
                started = perf_counter()
                weights = covariate_weights(cal_x,test_x)
                qs, details = targets_for(self.scores[OP],weights)
                for row in details.values():
                    row["weight_fit_wall_ms"] = (perf_counter()-started)*1000
                    row["feature_dimension"] = cal_x.shape[1]
                fits[label] = qs,details
        local = None
        if self.residual_cal is not None:
            reference = adapt_reference(self.residual_cal,shape,p.pde)
            fit = LocalSlicedCP(reference,self.summary_cal,p.seed+9029)
            local = fit,fit.prepare(cov)
        return thresholds,fits,local

    def evaluate(self, inputs, truth, n, frames, prepared, trial_seed):
        protocols, adjustments, local = prepared
        raw = defaultdict(list)
        torch.manual_seed(trial_seed)
        for sl in batches(len(truth)):
            x,y = take(inputs,sl),truth[sl]
            top = cutoff_points(y,self.protocol.pde)
            maps = self.maps(x,n,frames)
            for name,q in protocols.items():
                for alpha,row in conformal_values(*maps[name],y,top,q[0]).items():
                    raw[name,alpha].append(row)
            for name,(thresholds,_) in adjustments.items():
                qs = {a:q[sl] for a,q in thresholds.items()}
                for alpha,row in conformal_values(*maps[OP],y,top,qs).items():
                    raw[name,alpha].append(row)
            if local is not None:
                fit,weights = local
                center = maps[OP][0]
                for alpha,row in fit.evaluate(weights[sl],y-center,top,relative_l2(center,y)).items():
                    raw[LSCI,alpha].append(row)
        result, fields = defaultdict(dict), {}
        for (name,alpha),chunks in raw.items():
            rows = {k:np.concatenate([v[k] for v in chunks]) for k in chunks[0]}
            summary = summarize_values(rows,alpha,matched=name in protocols)
            if name == LSCI:
                summary |= {"native_slice_set_coverage":float(rows["native_hit"].mean()),
                            "sampled_envelope":64,"slices":32,
                            "post_calibration_p50_ms":float(np.median(rows["region_ms"]))}
            else:
                summary |= (protocols | adjustments)[name][1][alpha]
            result[name][alpha] = summary
            for column, values in rows.items():
                fields[f"{name}__{alpha}__{column}"] = values
        return dict(result), fields

    def timing(self, inputs, n, frames, n_runs):
        timings = defaultdict(list)
        for i in range(n_runs+3):
            x = take(inputs,slice(i,i+1))
            start = perf_counter()
            y = self.protocol.predict(self.primary,x,n,frames)
            forward = (perf_counter()-start)*1000
            start = perf_counter()
            self.protocol.op_scale(y,x)
            elapsed = (perf_counter()-start)*1000
            times = {"FNO forward":forward,"OpRes post":elapsed,"OpRes end-to-end":forward+elapsed}
            for tag,models,draws in (("ensemble",self.ensemble,None),("mc_dropout",[self.dropout] if self.dropout else [],self.draws)):
                if models:
                    start = perf_counter()
                    _,std = moments(self.protocol,models,x,n,frames,draws)
                    gaussian_floor(std,self.floor["ensemble" if tag=="ensemble" else "dropout"])
                    times[tag] = (perf_counter()-start)*1000
            if i>=3:
                for key,value in times.items():
                    timings[key].append(value)
        return {key:{"p50_ms":float(np.median(v)),"p90_ms":float(np.quantile(v,.9))}
                for key,v in timings.items()}


def import_darcy_ensemble(protocol, label, data):
    source = protocol.reference["tracks"]["64"]["FNO"]
    for name in ENSEMBLE:
        old = source["splits"][label][name]
        data[name] = {str(a):old[str(a)]["field"] | {
            "point_rel_l2":old["point_rel_l2"],"matched_width":old[str(a)]["matched_field_width"],
            "provenance":"Phase 2A verified export"
        } for a in ALPHAS}


def label_for(protocol, split, n, frames):
    return f"{split}_{n}_{frames}" if protocol.pde=="allen_cahn" else f"{split}_{n}"


def frozen_replay(protocol, split, n, frames, summary):
    source = protocol.reference
    checks = {}
    for alpha in map(str,ALPHAS):
        if protocol.pde == "darcy":
            old = source["tracks"][str(protocol.grid)]["FNO"]["splits"][f"{split}_{n}"][OP][alpha]["field"]
        elif protocol.pde == "burgers":
            old = source["results"]["FNO"][alpha]["summary"][f"{split}_{n}"]["methods"][OP]
        else:
            old = source["results"]["FNO"]["results"][f"{split}_{n}"][alpha][str(frames)]["methods"][OP]
        new = summary[OP][alpha]
        checks[alpha] = {"coverage_delta":new["coverage"]-old["coverage"],
                         "width_ratio":new["width"]/old["width"]}
    return checks


def run(pde,seed,grid,frozen,device,smoke=False):
    out = Path("results") / ("phase2d_smoke" if smoke else "phase2d") / pde / f"seed_{seed}" / f"grid{grid}"
    p = Protocol(pde,seed,grid,out,frozen,device,smoke)
    methods = ["ensemble", "dropout"] if pde!="darcy" else (["ensemble"] if grid==32 else ["dropout"])
    if pde=="darcy" and smoke and grid==64:
        methods.append("ensemble")
    draws = 4 if smoke else 30
    baselines = Baselines(p,methods,draws)
    output, timing, checks, replay = {}, {}, {}, {}
    field_dir = out/"fields"
    field_dir.mkdir(exist_ok=True)
    for i,(split,n,frames) in enumerate(p.targets()):
        label = label_for(p,split,n,frames)
        data = arrays(p.data_path(split,n))
        ids = data["id"]
        if len(np.intersect1d(ids,baselines.cal_ids)) or len(np.unique(ids))!=len(ids):
            raise ValueError(f"{label} field IDs overlap with Cal or repeat.")
        inputs, truth = p.split(data,n,frames)
        ood = split.startswith("ood")
        prepared = baselines.prepare_target(inputs,truth.shape[1:],ood,n,frames)
        output[label], fields = baselines.evaluate(inputs,truth,n,frames,prepared,seed+81001)
        np.savez_compressed(field_dir/f"{label}.npz",id=ids,**fields)
        if not smoke:
            replay[label] = frozen_replay(p,split,n,frames,output[label])
        if pde=="darcy" and grid==64 and not smoke:
            import_darcy_ensemble(p,label,output[label])
        timing[label] = baselines.timing(inputs,n,frames,2 if smoke else 20)
        for methods_report in output[label].values():
            for row in methods_report.values():
                if row["n"]!=len(ids) or not 0<=row["infinite_rate"]<=1:
                    raise ValueError(f"Incorrect reported denominator for {label}.")
        checks[label] = {"n":len(ids),"first_id":int(ids[0]),"last_id":int(ids[-1]),
                         "disjoint_cal":True}
        if smoke and ood:
            q_before = {k:{a:v.copy() for a,v in value[0].items()} for k,value in prepared[1].items()}
            # Fit again using the same covariates after perturbing labels only.
            future = truth.copy()+1
            if np.array_equal(future,truth):
                raise AssertionError("The leakage canary did not change labels.")
            prepared_after = baselines.prepare_target(inputs,future.shape[1:],ood,n,frames)
            assert all(np.array_equal(q,prepared_after[1][name][0][alpha])
                       for name,qs in q_before.items() for alpha,q in qs.items())
            assert np.array_equal(prepared[2][1],prepared_after[2][1])
            checks[label]["target_label_perturbation_keeps_wcp_thresholds"] = True
            checks[label]["target_label_perturbation_keeps_lsci_weights"] = True
        print(f"Phase 2D {pde} seed {seed} evaluated {label}",flush=True)
    source_notes = {}
    if pde=="darcy" and grid==64 and not smoke:
        src = p.reference["tracks"]["64"]["FNO"]
        source_notes = {"ensemble_total_training_seconds":sum(src["training_seconds"]),
                        "ensemble_timing_export":src["latency_ms"]}
    audit = {"cfg":p.config,"seed":seed,"pde":pde,"base_grid":grid,"smoke":smoke,
             "device":torch.cuda.get_device_name() if device.type=="cuda" else "cpu",
             "dropout_draws":draws if baselines.dropout else None,
             "variance_floor":baselines.floor,"training_seconds":baselines.training_seconds,
             "ensemble_total_training_seconds":sum(v for k,v in baselines.training_seconds.items()
                                                   if k.startswith("ensemble_") or k==p.key)
                                               if baselines.ensemble else None,
             "phase2a_ensemble":source_notes,"frozen_matches":p.data_matches,"cal_test_audit":checks,
             "frozen_opres_replay":replay,"results":output,"latency":timing}
    write_json(out/"metrics.json",audit)
    if pde=="burgers" and not smoke:
        from phase2d_burgers_diagnosis import run as diagnose
        item=p.manifest["weights"]["burgers"]
        result=diagnose(p.config,out/"data",p.frozen/item["file"],p.reference,p.device)
        write_json(out/"high_frequency_diagnosis.json",result)


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--pde",choices=PDE,required=True)
    cli.add_argument("--seed",type=int,choices=(42,137,314,271,811),required=True)
    cli.add_argument("--grid",type=int,choices=(32,64),default=64)
    cli.add_argument("--frozen",type=Path)
    cli.add_argument("--smoke",action="store_true")
    args = cli.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.smoke and device.type != "cuda":
        raise RuntimeError("Formal Phase 2D jobs require GPU acceleration.")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    frozen = args.frozen or mounted_snapshot(args.seed,args.pde)
    run(args.pde,args.seed,args.grid,frozen,device,args.smoke)


if __name__=="__main__":
    main()
