import argparse
import json

import numpy as np

from allen_cahn_phase2_data import CONFIG, ROOT, initial, solve
from allen_cahn_phase2_probe import OP, choose_cal, chunks, eval_split, model_load, observations, scale, torch_device
from darcy_day15 import arrays, digest, write_json
from darcy_phase0_eval import quantile


JACOBIAN_VAL_STREAM = 4115
JACOBIAN_VAL_COUNT = 100
JACOBIAN_SCREEN = {"max_threshold_ratio_relative":.90,
                   "max_matched_width_relative":1.10,"max_abs_field":10.}
JACOBIAN_TEST_STREAMS = {"id":4116,"paired":4117,"ood":4118}


def samples(config, split):
    nu, rho, fields = initial(config, split)
    grids = config["grids"] if split == "paired" else [64]
    trajectories = {n:[] for n in grids}
    for i in range(0, len(nu), 12):
        sl = slice(i, i+12)
        block = solve(fields[sl],nu[sl],rho[sl],config,grids,config["frames"])
        for n, values in block.items():
            trajectories[n].append(values)
    return {**{f"u{n}":np.concatenate(values) for n,values in trajectories.items()},"nu":nu,"rho":rho,
            "id":config["streams"][split]*10000+np.arange(len(nu))}


def split_horizon(config, baseline, model, device):
    lock = baseline["lock"]
    diagnostic = json.loads((ROOT/"horizon_matched_diagnostic.json").read_text(encoding="utf-8"))
    result = {}
    for split in ("id", "diagnostic_id"):
        path = ROOT/f"{split}.npz"
        expected_hash = baseline["data_sha256"]["id"] if split == "id" else diagnostic["test_sha256"]
        if digest(path) != expected_hash:
            raise RuntimeError(f"Frozen {split} trajectories changed.")
        data = arrays(path)
        short = int(round(1/config["dt"]))
        statistics = {}
        for _, error, _, lifted, _, scales in observations(data,64,config,model,device,lock):
            if error.shape[1] != 2*short:
                raise RuntimeError("The audit requires two equally long windows.")
            error_early, error_late = error[:,:short], error[:,short:]
            peak_early = error_early.max(axis=(1,2,3))
            peak_late = error_late.max(axis=(1,2,3))
            for name, full_scale in scales.items():
                early, late = full_scale[:,:short], full_scale[:,short:]
                prefix_scale = scale(lifted[:,:short],config) if name == OP else early
                values = {
                    "error_growth": peak_late/peak_early,
                    "scale_growth": late.mean(axis=(1,2,3))/early.mean(axis=(1,2,3)),
                    "prefix_score": (error_early/prefix_scale).max(axis=(1,2,3)),
                    "early_score": (error_early/early).max(axis=(1,2,3)),
                    "late_score": (error_late/late).max(axis=(1,2,3)),
                }
                for key, value in values.items():
                    statistics.setdefault(name,{}).setdefault(key,[]).append(value)
        summary = {}
        for name, columns in statistics.items():
            values = {key:np.concatenate(value) for key,value in columns.items()}
            q1 = baseline["thresholds"][name]
            prefix_hit = values["prefix_score"] <= q1
            early_hit = values["early_score"] <= q1
            late_hit = values["late_score"] <= q1
            summary[name] = {
                "prefix_covered": int(prefix_hit.sum()),
                "early_covered_with_full_scale": int(early_hit.sum()),
                "late_covered_with_full_scale": int(late_hit.sum()),
                "both_covered": int((early_hit & late_hit).sum()),
                "late_failure_given_early_coverage": float((~late_hit[early_hit]).mean()) if early_hit.any() else None,
                "median_late_over_early_peak_error": float(np.median(values["error_growth"])),
                "median_late_over_early_mean_scale": float(np.median(values["scale_growth"])),
                "median_late_over_early_score": float(np.median(values["late_score"]/values["early_score"])),
            }
        result[split] = {"n":len(data["id"]), "sha256":expected_hash, "methods":summary}
        print(split,summary[OP],flush=True)
    write_json(ROOT/"horizon_failure_decomposition.json",result)


def matched_horizon(c,base,lock,model,d):
    cal= samples(c|{"streams":c["streams"]|{"cal":4113},"counts":c["counts"]|{"cal":150}},"cal")
    by_method={}
    for _,err,_,_,_,ss in observations(cal,64,c,model,d,lock):
        for name,s in ss.items():
            by_method.setdefault(name,[]).extend((err/s).max(axis=(1,2,3)))
    q2={name:quantile(np.array(v),.1) for name,v in by_method.items()}
    cfg=c|{"streams":c["streams"]|{"id":4114},"counts":c["counts"]|{"id":150}}
    test=samples(cfg,"id")
    path=ROOT/"diagnostic_id.npz"
    np.savez(path,**test)
    results,_=eval_split("diagnostic_id",64,c,model,d,lock,q2)
    out={"cal_stream":4113,"test_stream":4114,"n_cal":len(cal["id"]),
         "n_test":len(test["id"]),"test_sha256":digest(path),"candidate":lock["candidate"],
         "q2":q2,"q2_over_original_q1":{k:v/base["thresholds"][k] for k,v in q2.items()},
         "test_t2":results["2.0"]}
    write_json(ROOT/"horizon_matched_diagnostic.json",out)
    print("Fresh Allen–Cahn matched-horizon diagnostic",results["2.0"]["methods"][OP]["coverage"])


def val_scores(data, config, model, device, inverse):
    scores, means = [], []
    largest = 0.
    for selection, prediction, lifted, _ in chunks(data,64,config,model,device,inverse):
        error = abs(prediction[:,1:]-data["u64"][selection,1:])
        sigma = scale(lifted,config)
        scores.extend((error/sigma).max(axis=(1,2,3)))
        means.extend(sigma.mean(axis=(1,2,3)))
        largest = max(largest,float(abs(prediction).max()),float(abs(lifted).max()))
    q90 = float(np.quantile(scores,.90,method="inverted_cdf"))
    return {"score_q90":q90,"matched90_width":float(2*q90*np.mean(means)),"max_abs_field":largest}


def jacobian_val(config, lock, model, device):
    short_path = ROOT/"val.npz"
    if digest(short_path)!=lock["val_sha256"]:
        raise RuntimeError("Frozen short-window validation trajectories changed.")
    short = arrays(short_path)
    long_config = config|{"streams":config["streams"]|{"val":JACOBIAN_VAL_STREAM},
                          "counts":config["counts"]|{"val":JACOBIAN_VAL_COUNT}}
    long = samples(long_config,"val")
    # rho bounds the exact positive reaction derivative for all predicted states.
    derivative_cap = max(config["rho"][1],config["ood_rho"][1])
    candidate = config|{"positive_damping":0.,"max_growth":derivative_cap}
    results = {}
    for name, settings in (("protected",config),("full_positive_jacobian",candidate)):
        t1 = val_scores(short,settings,model,device,lock["candidate"])
        t2 = val_scores(long,settings,model,device,lock["candidate"])
        results[name] = {"t1":t1,"t2":t2,"threshold_ratio":t2["score_q90"]/t1["score_q90"]}
        print(name,results[name],flush=True)
    ref, modified = results["protected"], results["full_positive_jacobian"]
    ratios = {"threshold_ratio_relative":modified["threshold_ratio"]/ref["threshold_ratio"],
              "matched_width_relative":modified["t2"]["matched90_width"]/ref["t2"]["matched90_width"]}
    passed = (ratios["threshold_ratio_relative"]<=JACOBIAN_SCREEN["max_threshold_ratio_relative"]
              and ratios["matched_width_relative"]<=JACOBIAN_SCREEN["max_matched_width_relative"]
              and modified["t2"]["max_abs_field"]<JACOBIAN_SCREEN["max_abs_field"])
    write_json(ROOT/"horizon_jacobian_val.json",
               {"short_val_sha256":lock["val_sha256"],"short_val_n":len(short["id"]),
                "long_val_stream":JACOBIAN_VAL_STREAM,"long_val_n":len(long["id"]),
                "config":config,"candidate_parameters":{"positive_damping":0.,"max_growth":derivative_cap},
                "screen":JACOBIAN_SCREEN,"results":results,"ratios":ratios,"passed":bool(passed)})


def jacobian_test(config, baseline, model, device):
    validation_path = ROOT/"horizon_jacobian_val.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if not validation["passed"] or validation["config"] != config:
        raise RuntimeError("The candidate has no matching validation pass.")
    lock = baseline["lock"]
    candidate = config|validation["candidate_parameters"]
    if digest(ROOT/"cal.npz") != baseline["data_sha256"]["cal"]:
        raise RuntimeError("Frozen short-window calibration trajectories changed.")
    protocol = {
        "config":candidate,"inverse":lock["candidate"],"blind_streams":JACOBIAN_TEST_STREAMS,
        "checkpoint":lock["checkpoint"],"validation_sha256":digest(validation_path),
        "cal_sha256":baseline["data_sha256"]["cal"],"matched_width_upper_ci":1.,
        "max_abs_field":10.,"min_coverage_relative_to": "best_other_method",
    }
    protocol_path = ROOT/"horizon_jacobian_test_lock.json"
    if protocol_path.is_file():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise RuntimeError("The blind test protocol has already been locked.")
    else:
        write_json(protocol_path,protocol)
    threshold = choose_cal(candidate,model,device,lock)
    for method, q in threshold.items():
        if method!=OP and not np.isclose(q,baseline["thresholds"][method],rtol=1e-5):
            raise RuntimeError(f"Frozen baseline calibration changed for {method}.")
    test_config = candidate|{"streams":candidate["streams"]|JACOBIAN_TEST_STREAMS}
    results, hashes = {}, {}
    for split, stream in JACOBIAN_TEST_STREAMS.items():
        name = f"jacobian_{split}"
        path = ROOT/f"{name}.npz"
        if not path.is_file():
            temporary = path.with_suffix(".tmp.npz")
            np.savez(temporary,**samples(test_config,split))
            temporary.replace(path)
        expected = stream*10000+np.arange(config["counts"][split])
        with np.load(path) as cached:
            if not np.array_equal(cached["id"],expected):
                raise RuntimeError(f"Unexpected blind sample IDs for {split}.")
        hashes[split] = digest(path)
        for n in config["grids"] if split == "paired" else (64,):
            windows,_ = eval_split(name,n,candidate,model,device,lock,threshold)
            results[f"{split}_{n}"] = windows
            print(split,n,windows["2.0"]["methods"][OP]["coverage"],flush=True)
    id_long = results["id_64"]["2.0"]
    gates = {
        "finite":all(max(row["fno_max"],row["lift_max"])<protocol["max_abs_field"]
                     for windows in results.values() for row in windows.values()),
        "matched":all(row["matched90_width_ratio_ci95"][1]<protocol["matched_width_upper_ci"]
                      for row in id_long["comp"].values()),
        "rollout":id_long["methods"][OP]["coverage"]>=max(row["coverage"] for method,row
                     in id_long["methods"].items() if method!=OP),
    }
    write_json(ROOT/"horizon_jacobian_test.json",
               {"protocol_sha256":digest(protocol_path),"data_sha256":hashes,"thresholds":threshold,
                "results":results,"gates":gates,"decision":"GO" if all(gates.values()) else "HOLD"})


def main():
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--decompose",action="store_true",
                      help="Audit the two halves of cached T=2 traces under the original threshold.")
    mode.add_argument("--jacobian-val",action="store_true",
                      help="Screen the full positive reaction Jacobian on validation trajectories.")
    mode.add_argument("--jacobian-test",action="store_true",
                      help="Blindly test the locked validation candidate on fresh trajectories.")
    args=parser.parse_args()
    c=json.loads(CONFIG.read_text(encoding="utf-8"))
    base=json.loads((ROOT/"metrics.json").read_text(encoding="utf-8"))
    lock=base["lock"]
    device=torch_device()
    model,_=model_load(c,device)
    if digest(ROOT/"fno.pt")!=lock["checkpoint"] or digest(ROOT/"fixed.npy")!=lock["fixed_sha256"]:
        raise RuntimeError("Frozen model or CP-B scale changed.")
    if args.decompose:
        split_horizon(c,base,model,device)
    elif args.jacobian_val:
        jacobian_val(c,lock,model,device)
    elif args.jacobian_test:
        jacobian_test(c,base,model,device)
    else:
        matched_horizon(c,base,lock,model,device)


if __name__=="__main__":
    main()
