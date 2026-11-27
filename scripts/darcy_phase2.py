import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, roc_curve

from darcy_data import apply_operator, generate
from darcy_day15 import (arrays, band_summary, comparison, digest, estimated_weights,
                         evaluate_bands, features, oracle_weights, value_at_coverage,
                         weighted_quantile, write_json)
from darcy_day15_ops import lift, mixture, physical_scale, rms, timed, timing_summary
from darcy_fno import atomic_checkpoint, relative_l2
from darcy_phase0_eval import bilinear, boot_interval, center_coordinates, quantile, signed_inverse
from darcy_phase2_models import model_for


CONFIG = Path("configs/phase2_darcy.json")
ALPHA = "alpha"
AXES = (-2,-1)
OP = "OpRes-CP"
BASE = ("Uniform-CP","CP-B","PI-CP","Inverse-after-Abs","Physics-Scalar CP",OP,
        "Defect Correction + CP","Mabtoul-adapted CW")
ENSEMBLE = ("5-FNO Ensemble + CP | single center","5-FNO Ensemble + CP | mean center")


def data_config(formal,grid,fine=None):
    base = json.loads(Path("configs/phase0_darcy.json").read_text(encoding="utf-8"))
    return base|{"coarse":grid,"fine":fine or formal["fine_grid"],"counts":formal["counts"]}


def generate_split(plan,seed,config,split,root,workers):
    stream=plan["streams"][split]
    seed_sequence=np.random.SeedSequence([seed,stream])
    generate(config|{"counts":{split:plan["counts"][split]}},seed_sequence,root,workers,True,
             id_offset=stream*10000)


def train_model(kind, init, config, track, root, device, *, factory=model_for):
    path=root/"fno.pt"
    model=factory(kind,config).to(device)
    if (root/"training.json").is_file() and path.is_file():
        return model,path
    root.mkdir(parents=True,exist_ok=True)
    settings=config["model"]
    torch.manual_seed(init)
    model=factory(kind,config).to(device)
    train,val=(arrays(track/f"{name}.npz") for name in ("train","val"))
    a,u,va,vu=(torch.from_numpy(v).to(device) for v in (train["a"],train["u"],val["a"],val["u"]))
    model.input_std.copy_(a.log().std())
    model.output_scale.copy_(u.square().mean().sqrt())
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings["learning_rate"],weight_decay=1e-4)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,settings["epochs"],eta_min=5e-5)
    rng=torch.Generator().manual_seed(init)
    latest=root/"latest.pt"
    start,best,history=0,np.inf,[]
    if latest.exists():
        state=torch.load(latest,map_location="cpu",weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["opt"])
        schedule.load_state_dict(state["scheduler"])
        rng.set_state(state["rng"])
        start,best,history=state["epoch"],state["best"],state["history"]
    for epoch in range(start,settings["epochs"]):
        tick=perf_counter()
        model.train()
        for ids in torch.randperm(len(a),generator=rng).split(settings["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss=relative_l2(model(a[ids]),u[ids])
            loss.backward()
            optimizer.step()
        schedule.step()
        model.eval()
        with torch.inference_mode():
            loss=float(sum(relative_l2(model(x),y)*len(x) for x,y in
                     zip(va.split(settings["batch_size"]),vu.split(settings["batch_size"])))/len(va))
        if loss<best:
            best=loss
            atomic_checkpoint(path,{"model":model.state_dict(),"epoch":epoch+1})
        history.append({"epoch":epoch+1,"val_relative_l2":loss,"seconds":perf_counter()-tick})
        atomic_checkpoint(latest,{"model":model.state_dict(),"opt":optimizer.state_dict(),
                           "scheduler":schedule.state_dict(),"rng":rng.get_state(),"epoch":epoch+1,
                           "best":best,"history":history})
        if (epoch+1)%5==0 or epoch+1==settings["epochs"]:
            print(f"{kind} {init} grid {config['coarse']} epoch {epoch+1}: {loss:.5f}",flush=True)
    write_json(root/"training.json",{"initialization":init,"device":str(device),"history":history,
                                     "parameters":sum(p.numel() for p in model.parameters())})
    latest.unlink(missing_ok=True)
    return model,path


def restore(kind,root,config,device):
    model=model_for(kind,config).to(device)
    model.load_state_dict(torch.load(root/"fno.pt",map_location=device,weights_only=True)["model"])
    return model.eval()


@torch.inference_mode()
def prediction(model,kind,sensors,grid,device,batch=8):
    parts=[]
    for i in range(0,len(sensors),batch):
        a=torch.from_numpy(sensors[i:i+batch]).to(device)
        x=model.forward_at(a,grid) if kind=="DeepONet" else model(a)
        parts.append(x.cpu().numpy())
    return np.concatenate(parts)


def physical(a):
    log=np.log(a)
    q=np.quantile(log,[.1,.9],axis=AXES)
    dx=np.diff(log,axis=-2)
    dy=np.diff(log,axis=-1)
    slope=np.concatenate((abs(dx).reshape(len(a),-1),abs(dy).reshape(len(a),-1)),axis=1)
    return np.column_stack((log.std(axis=AXES),q[1]-q[0],np.quantile(slope,.95,axis=1),
                            (slope>0.2).mean(axis=1),slope.mean(axis=1)))


def mabtoul_x(a,p):
    n=a.shape[-1]
    x,y=np.meshgrid(center_coordinates(n),center_coordinates(n),indexing="ij")
    log=np.log(a)
    grad=np.hypot(*np.gradient(p,1/n,axis=AXES))
    grad_a=np.hypot(*np.gradient(log,1/n,axis=AXES))
    boundary=np.minimum.reduce((x,y,1-x,1-y))
    return np.stack((np.broadcast_to(x,a.shape),np.broadcast_to(y,a.shape),
                    np.broadcast_to(boundary,a.shape),p,grad,log,grad_a),axis=-1)


def keep_fit(val, pred, members, plan):
    a=val["a"].astype(float)
    y=val["u"].astype(float)
    e=abs(pred-y)
    fixed=e.mean(axis=0)
    fixed=np.maximum(fixed, .05*fixed.mean())
    residual=apply_operator(a,pred)-1
    x=mabtoul_x(a,pred).reshape(len(a),-1,7)
    prng=np.random.default_rng(plan["bootstrap_seed"])
    which=np.stack([prng.choice(x.shape[1],plan["regression_pixels_per_field"],replace=False) for _ in a])
    k=np.arange(len(a))[:,None]
    target=np.log(np.maximum(e.reshape(len(a),-1)[k,which],1e-10))
    reg=HistGradientBoostingRegressor(loss="quantile",quantile=.5,max_iter=150,
                                    max_leaf_nodes=31,max_depth=6,random_state=42)
    reg.fit(x[k,which].reshape(-1,x.shape[-1]),target.ravel())
    ensemble_floor=None
    if members is not None:
        ensemble_floor=.10*float(np.sqrt(np.mean(members.std(axis=0,ddof=1,dtype=np.float64)**2)))
    return fixed,float(1/np.median(abs(residual))),reg,ensemble_floor


def scale_methods(a,p,member_pred,fit,config,plan):
    fixed,pi,reg,floor=fit
    if a.shape[-1]!=fixed.shape[-1]:
        fixed=bilinear(fixed[None],a.shape[-1])[0]
    diff=apply_operator(a,p)-1
    z=lift(plan["inverse"],a,diff)
    envelope=physical_scale(z,config)
    unsigned=physical_scale(signed_inverse(a,abs(diff)),config)
    x=mabtoul_x(a,p)
    adaptive=np.exp(reg.predict(x.reshape(-1,x.shape[-1])).reshape(a.shape))
    rows={
        "Uniform-CP":(p,np.ones_like(p)),
        "CP-B":(p,np.broadcast_to(fixed,a.shape)),
        "PI-CP":(p,1+pi*abs(diff)),
        "Inverse-after-Abs":(p,unsigned),
        "Physics-Scalar CP":(p,mixture(envelope,0)),
        OP:(p,mixture(envelope,plan["lambda"])),
        "Defect Correction + CP":(p-z,np.ones_like(p)),
        "Mabtoul-adapted CW":(p,adaptive),
    }
    if member_pred is not None:
        ens_std=np.sqrt(member_pred.std(axis=0,ddof=1,dtype=np.float64)**2+floor**2)
        mean_pred=member_pred.mean(axis=0,dtype=np.float64)
        rows.update({ENSEMBLE[0]:(p,ens_std),ENSEMBLE[1]:(mean_pred,ens_std)})
    return rows,{"input":physical(a),"joint":np.column_stack((physical(a),
                            features(a,diff,p,{"inverse":plan["inverse"]})[:,2:]))}


def evaluate_split(target, sensors, ground, model, kind, extras, fit, config, plan, device, cal=None):
    all_scores,features_by_type={},{}
    a,u,sigma=ground
    split,n=target
    top_shape=u.shape[-1]**2
    k=math.ceil(.1*top_shape)
    thresholds = {} if cal is None else {
        name:{alpha:(quantile(row["score"],alpha),quantile(row["pixel_values"].ravel(),alpha))
              for alpha in plan[ALPHA]} for name,row in cal["methods"].items()}
    for i in range(0,len(a),16):
        block=slice(i,i+16)
        p=prediction(model,kind,sensors[block],n,device).astype(float)
        ex=None if not extras else np.stack([p,*[prediction(m,kind,sensors[block],n,device) for m in extras]])
        rows,f=scale_methods(a[block].astype(float),p,ex,fit,config,plan)
        for key,value in f.items():
            features_by_type.setdefault(key,[]).append(value)
        yy=u[block].astype(float)
        gy=np.hypot(*np.gradient(yy,1/n,axis=AXES)).reshape(len(yy),-1)
        top=np.argpartition(gy,-k,axis=1)[:,-k:]
        for name,(center,s) in rows.items():
            if not np.isfinite(s).all() or np.any(s<=0):
                raise RuntimeError(f"Undefined scale for {name} in {target}.")
            error=abs(center-yy)
            flat=s.reshape(len(yy),-1)
            ratios=error.reshape(len(yy),-1)/flat
            record={"score":ratios.max(axis=1),"scale_mean":flat.mean(axis=1),
                    "point_rel_l2":(rms(error)/rms(yy)).ravel(),
                    "peak_error":error.max(axis=AXES)}
            for alpha in plan[ALPHA]:
                suffix=str(alpha)
                if cal is None:
                    record["pixel_values"]=ratios.astype(np.float32)
                else:
                    q,px=thresholds[name][alpha]
                    band=evaluate_bands(ratios,flat,q,top)
                    pixel=evaluate_bands(ratios,flat,px,top)
                    for label,v in (("field",band),("pixel",pixel)):
                        record.update({f"{label}__{suffix}__{key}":value for key,value in v.items()})
                    record["matched_pix"]=ratios.astype(np.float32)
            dest=all_scores.setdefault(name,{})
            for key,v in record.items():
                dest.setdefault(key,[]).append(v)
    combined={m:{key:np.concatenate(v) for key,v in r.items()} for m,r in all_scores.items()}
    summaries={key:np.concatenate(v) for key,v in features_by_type.items()}
    return {"methods":combined,"features":summaries,"sigma":sigma}


def make_summary(evaluation,cal,plan):
    out={}
    for name,row in evaluation["methods"].items():
        output={"point_rel_l2":float(row["point_rel_l2"].mean()),"peak_error":float(row["peak_error"].mean()),
                "score_peak_spearman":float(spearmanr(row["score"],row["peak_error"]).statistic)}
        for alpha in plan[ALPHA]:
            key=str(alpha)
            summary={}
            for label in ("field","pixel"):
                prefix=f"{label}__{key}__"
                band={k.removeprefix(prefix):v for k,v in row.items() if k.startswith(prefix)}
                summary[label]=band_summary(band)
            summary["matched_field_width"]=float(2*value_at_coverage(row["score"],1-alpha)*row["scale_mean"].mean())
            summary["matched_pixel_width"]=float(2*value_at_coverage(row["matched_pix"],1-alpha)*row["scale_mean"].mean())
            summary["q"]=quantile(cal["methods"][name]["score"],alpha)
            output[key]=summary
        out[name]=output
    return out


def weights(cal, test, config, feature):
    if feature=="oracle":
        return oracle_weights(cal["sigma"],config),oracle_weights(test["sigma"],config)
    return estimated_weights(cal["features"][feature],test["features"][feature],
                             {"base_weight_clip":config["weight_clip"],"weight_logistic_c":1.})


def gate_point(p, bad, ood, base, plan):
    fpr,tpr,_=roc_curve(ood,p,drop_intermediate=False)
    curve=[]
    for threshold in np.unique(p):
        use=p<=threshold
        curve.append({"acceptance":float(use.mean()),"point_risk":float(bad[use].mean()),"p_test":float(threshold)})
    area=sum((current["acceptance"]-previous)*current["point_risk"]
             for previous,current in zip([0.]+[c["acceptance"] for c in curve[:-1]],curve))
    anchors={}
    for t in plan["gate_p_test_thresholds"]:
        use=p<=t
        anchors[str(t)]={**band_summary(base,~use),"id_pass":float(use[~ood].mean()),
                           "ood_recall":float((~use[ood]).mean()),"point_risk":float(bad[use].mean()) if use.any() else None}
    return {"ood_auroc":float(roc_auc_score(ood,p)),"fpr_at_95_tpr":float(fpr[tpr>=.95].min()),
            "point_aurc":area,"ungated_point_risk":float(bad.mean()),"anchors":anchors,"curve":curve}


def shift_gate(id_eval,ood_eval,cal,config,plan):
    features={key:np.concatenate((id_eval["features"][key],ood_eval["features"][key]))
              for key in ("input","joint")}
    ids=np.arange(len(features["input"]))>=len(id_eval["sigma"])
    raw={key:np.concatenate((id_eval["methods"][OP][f"field__0.1__{key}"],
              ood_eval["methods"][OP][f"field__0.1__{key}"]))
         for key in ("coverage","pixel","width","top_width","top_coverage","infinite")}
    error=np.concatenate((id_eval["methods"][OP]["point_rel_l2"],ood_eval["methods"][OP]["point_rel_l2"]))
    summaries={}
    for mode in ("input","joint"):
        wc,wt=weights(cal,{"features":features},config,mode)
        p=wt/(wc.sum()+wt)
        summaries[mode]=gate_point(p,error>plan["point_failure_rel_l2"],ids,raw,plan)
    return summaries


def shift_weighted(eval_result,cal,config,plan,ood_name):
    out={}
    modes=("input","joint","oracle") if ood_name=="ood_a" else ("input","joint")
    for feat in modes:
        wc,wt=weights(cal,eval_result,config,feat)
        out[feat]={}
        for name in ("Uniform-CP","CP-B",OP,*[s for s in ENSEMBLE if s in eval_result["methods"]]):
            row=eval_result["methods"][name]
            scores=cal["methods"][name]["score"]
            result={}
            for alpha in plan[ALPHA]:
                q,p,ess=weighted_quantile(scores,wc,wt,alpha)
                hits=row["score"]<=q
                inf=~np.isfinite(q)
                keep=~inf & (ess>=plan["min_ess"])
                result[str(alpha)]={"coverage":float(hits.mean()),"infinite_rate":float(inf.mean()),
                     "width":"inf" if inf.any() else float(np.mean(2*q*row["scale_mean"])),
                     "ess":ess,"max_p_test":float(p.max()),"gate_accept":float(keep.mean()),
                     "accepted_coverage":float(hits[keep].mean()) if keep.any() else None,
                     "accepted_width":float(np.mean(2*q[keep]*row["scale_mean"][keep])) if keep.any() else None}
            out[feat][name]=result
    return out


def load_grid(path,kind,grid,training):
    data=arrays(path)
    a,u=(data["a"],data["u"]) if grid==training else (data["a_f"],data["u_f"])
    sensors=data["a"] if kind=="DeepONet" else a
    return sensors,(a,u,data["sigma"]),data["id"]


def timing(model,kind,sensors,extras,device,grid,ground,config,plan):
    one=[]
    ensemble=[]
    post=[]
    primary_total=[]
    ens_total=[]
    for i,field in enumerate(sensors[:54]):
        x=field[None]
        a=ground[i:i+1].astype(float)
        pred,t=timed(lambda:prediction(model,kind,x,grid,device,batch=1))
        def op():
            r=apply_operator(a,pred.astype(float))-1
            return mixture(physical_scale(lift(plan["inverse"],a,r),config),plan["lambda"])
        _,post_ms=timed(op)
        def ens():
            members=np.stack([pred,*[prediction(m,kind,x,grid,device,batch=1) for m in extras]])
            return members.mean(axis=0),members.std(axis=0,ddof=1)
        _,elapsed=timed(ens) if extras else (None,None)
        if i>=6:
            one.append(t)
            post.append(post_ms)
            primary_total.append(t+post_ms)
            if elapsed is not None:
                ensemble.append(elapsed)
                ens_total.append(elapsed+t)
    return {"single_forward":timing_summary(one),"opres_post":timing_summary(post),
            "single_plus_opres":timing_summary(primary_total),
            "ensemble_additional":timing_summary(ensemble) if ensemble else None,
            "ensemble_total":timing_summary(ens_total) if ens_total else None}


def run_track(plan,seed,root,workers,device,grid=64):
    track=root/f"grid{grid}"
    config=data_config(plan,grid)
    if plan.get("smoke"):
        config["model"]=config["model"]|{"width":8,"layers":2,"modes":6,"epochs":1}
    transfer=grid==plan["transfer_grid"]
    kinds=("FNO","DeepONet") if transfer else ("FNO","U-Net","DeepONet")
    splits=("paired_test",) if transfer else ("id_test","paired_test","ood_a","ood_b")
    model_paths={}
    for split in ("train","val","cal",*splits):
        generate_split(plan,seed,config,split,track/"data",workers)
    if transfer:
        generate_split(plan,seed,config|{"fine":64},"paired_test",track/"intermediate_data",workers)
    for kind in kinds:
        offsets=plan["ensemble_init_offsets"] if kind=="FNO" and not transfer else [0]
        members=[]
        for i,offset in enumerate(offsets):
            loc=track/kind/f"member{i}"
            train_model(kind,seed+offset,config,track/"data",loc,device)
            members.append(loc)
        model_paths[kind]=members
    output={}
    for kind,paths in model_paths.items():
        model,*extras=(restore(kind,p,config,device) for p in paths)
        val=arrays(track/"data/val.npz")
        pred=prediction(model,kind,val["a"],grid,device).astype(float)
        ensemble=np.stack([pred,*[prediction(m,kind,val["a"],grid,device) for m in extras]]) if extras else None
        fit=keep_fit(val,pred,ensemble,plan)
        cal_s,cal_g,_=load_grid(track/"data/cal.npz",kind,grid,grid)
        cal=evaluate_split(("cal",grid),cal_s,cal_g,model,kind,extras,fit,config,plan,device)
        summaries,timings,cache,paired_reports={}, {}, {}, {}
        for split in splits:
            if split=="paired_test":
                grids=(grid,64,128) if transfer else (grid,128)
                if kind=="U-Net":
                    grids=(grid,)
            else:
                grids=(grid,)
            for n in grids:
                path=track/("intermediate_data" if transfer and n==64 and split=="paired_test" else "data")/f"{split}.npz"
                inputs,target,ids=load_grid(path,kind,n,grid)
                key=f"{split}_{n}"
                if split=="paired_test":
                    if n==grid:
                        paired_ids=ids
                    elif not np.array_equal(paired_ids,ids):
                        raise RuntimeError("Paired physical field namespaces differ.")
                detail=evaluate_split((split,n),inputs,target,model,kind,extras,fit,config,plan,device,cal)
                summaries[key]=make_summary(detail,cal,plan)
                for row in detail["methods"].values():
                    row.pop("matched_pix")
                cache[key]=detail
                if split in ("id_test","paired_test") and n not in timings:
                    timings[n]=timing(model,kind,inputs,extras,device,n,target[0],config,plan)
                print(f"evaluated Phase 2A seed={seed} train={grid} {kind} {key}",flush=True)
        for key,data in cache.items():
            if key.startswith("paired_test_") and key!=f"paired_test_{grid}":
                base=cache[f"paired_test_{grid}"]
                paired_reports[key]={}
                for name in data["methods"]:
                    row={}
                    for alpha in plan[ALPHA]:
                        col=f"field__{alpha}__"
                        a=data["methods"][name]
                        b=base["methods"][name]
                        delta=a[col+"coverage"].astype(float)-b[col+"coverage"]
                        row[str(alpha)]={"coverage_delta":float(delta.mean()),
                              "coverage_delta_ci":boot_interval(delta,plan["bootstrap_resamples"],plan["bootstrap_seed"]),
                              "width_ratio":float(a[col+"width"].mean()/b[col+"width"].mean())}
                    paired_reports[key][name]=row
        wcp={}
        gates={}
        if not transfer:
            for shift in ("ood_a","ood_b"):
                t=cache[f"{shift}_{grid}"]
                wcp[shift]=shift_weighted(t,cal,config,plan,shift)
                gates[shift]=shift_gate(cache[f"id_test_{grid}"],t,cal,config,plan)
        paired_comparisons={}
        for key,data in cache.items():
            pairs=data["methods"]
            paired_comparisons[key]={}
            for alpha in plan[ALPHA]:
                col=f"field__{alpha}__"
                def extract(raw):
                    return {"coverage":raw[col+"coverage"],"scale_mean":raw["scale_mean"],
                            "score":raw["score"],"width":raw[col+"width"],"infinite":raw[col+"infinite"]}
                paired_comparisons[key][str(alpha)]={name:comparison(extract(pairs[OP]),extract(pairs[name]),
                    plan,config|{"alpha":alpha,"bootstrap_resamples":plan["bootstrap_resamples"]})
                    for name in pairs if name!=OP}
        metrics={"splits":summaries,"paired":paired_reports,"wcp":wcp,"gates":gates,
                          "latency_ms":timings,"paired_method_comparisons":paired_comparisons,
                          "training_seconds":[sum(h["seconds"] for h in json.loads((p/"training.json").read_text())["history"])
                                              for p in paths],
                          "weight_sha256":[digest(p/"fno.pt") for p in paths]}
        write_json(track/kind/"metrics.json",metrics)
        np.savez_compressed(track/kind/"field_metrics.npz",**{
           f"{key}__{name}__{col}":val for key,obj in cache.items() for name,stats in obj["methods"].items()
            for col,val in stats.items()})
        output[kind]=metrics
    output["data_sha256"]={str(p.relative_to(track)):digest(p) for p in track.glob("*data/*.npz")}
    return output


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",required=True,type=int)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--workers",type=int,default=4)
    args=p.parse_args()
    plan=json.loads(CONFIG.read_text(encoding="utf-8"))
    if args.seed not in plan["seeds"]:
        p.error("Unregistered seed.")
    if args.smoke:
        plan=plan|{"smoke":True,"counts":dict(zip(plan["counts"],[16,12,24,20,20,20,20])),
                   "regression_pixels_per_field":32,"bootstrap_resamples":50}
    root=Path("results")/("phase2_darcy_smoke" if args.smoke else "phase2_darcy")/f"seed_{args.seed}"
    root.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.smoke and device.type!="cuda":
        raise RuntimeError("Formal 2A requires CUDA.")
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    output={}
    for grid in (plan["primary_grid"],plan["transfer_grid"]):
        output[str(grid)]=run_track(plan,args.seed,root,args.workers,device,grid)
    write_json(root/"metrics.json",{"seed":args.seed,"config":plan,"tracks":output,
               "torch":str(torch.__version__),"gpu":torch.cuda.get_device_name() if device.type=="cuda" else None,
               "sources":{str(p):digest(p) for p in [CONFIG,*Path("scripts").glob("darcy*.py")]}})


if __name__=="__main__":
    main()
