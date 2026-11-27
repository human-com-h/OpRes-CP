import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import burgers_phase1_eval as metrics
from burgers_phase1_data import solve
from burgers_phase1_model import TrajectoryFNO, infer, torch_device
from burgers_phase1_ops import gradient
from burgers_phase2_gpu import adv_diff_gpu
from darcy_day15 import arrays, digest, estimated_weights, write_json
from darcy_day15_ops import timed, timing_summary
from darcy_fno import atomic_checkpoint
from darcy_phase2 import gate_point


CONFIG = Path("configs/phase2_burgers.json")
KINDS=("FNO","U-Net","DeepONet")
LOCKED={"inverse":"adv_diff","lambda":0.85}


class UNet(nn.Module):
    def __init__(self,frames):
        super().__init__()
        def block(a,b):
            return nn.Sequential(nn.Conv1d(a,b,3,padding=1,padding_mode="circular"),nn.GELU(),
                                 nn.Conv1d(b,b,3,padding=1,padding_mode="circular"),nn.GELU())
        self.a=block(2,32)
        self.b=block(32,64)
        self.c=block(64,96)
        self.d=block(160,64)
        self.e=block(96,32)
        self.out=nn.Conv1d(32,frames,1)

    def forward(self,u,nu):
        x=self.a(torch.stack((u,((nu.log()+3.28)/.63)[:,None].expand_as(u)),dim=1))
        y=self.b(F.avg_pool1d(x,2))
        z=self.c(F.avg_pool1d(y,2))
        w=self.d(torch.cat((F.interpolate(z,scale_factor=2,mode="linear",align_corners=False),y),dim=1))
        traj=self.out(self.e(torch.cat((F.interpolate(w,scale_factor=2,mode="linear",align_corners=False),x),dim=1)))
        traj=traj-traj.mean(dim=-1,keepdim=True)+u.mean(dim=-1)[:,None,None]
        return torch.cat((u[:,None],traj),dim=1)


class DeepONet(nn.Module):
    def __init__(self,frames):
        super().__init__()
        self.branch=nn.Sequential(nn.Linear(65,192),nn.GELU(),nn.Linear(192,192),
                                 nn.GELU(),nn.Linear(192,128))
        self.trunk=nn.Sequential(nn.Linear(9,128),nn.GELU(),nn.Linear(128,128),
                                 nn.GELU(),nn.Linear(128,128))
        self.frames=frames

    def forward(self,u,nu):
        n=u.shape[-1]
        if n%64:
            raise ValueError("DeepONet requires aligned 64-sensor inputs.")
        branch=self.branch(torch.cat((u[:,::n//64],((nu.log()+3.28)/.63)[:,None]),dim=1))
        k=torch.arange(1,5,device=u.device,dtype=u.dtype)
        x=torch.arange(n,device=u.device,dtype=u.dtype)/n*(2*math.pi)
        level=torch.arange(1,self.frames+1,device=u.device,dtype=u.dtype)/self.frames
        xy=torch.cat(((x[:,None]*k).cos(),(x[:,None]*k).sin()),dim=1)
        query=torch.cat((xy[None].expand(len(level),-1,-1),level[:,None,None].expand(-1,n,1)),dim=-1)
        traj=(branch@self.trunk(query).reshape(-1,128).T).reshape(len(u),len(level),n)
        traj=traj-traj.mean(dim=-1,keepdim=True)+u.mean(dim=-1)[:,None,None]
        return torch.cat((u[:,None],traj),dim=1)


def make_model(name,c):
    if name=="FNO":
        return TrajectoryFNO(c)
    return (UNet if name=="U-Net" else DeepONet)(c["frames"])


def generate(c,name,root):
    path=root/f"{name}.npz"
    if path.exists():
        return
    rng=np.random.default_rng(np.random.SeedSequence([c["seed"],c["streams"][name]]))
    n=c["counts"][name]
    vr=c["viscosity_ood" if name=="ood" else "viscosity_id"]
    nu=np.exp(rng.uniform(*np.log(vr),size=n))
    mode=np.arange(1,17 if name=="ood_high" else 9)
    coeff=(rng.normal(size=(n,len(mode)))+1j*rng.normal(size=(n,len(mode))))/mode**1.35
    coeff*=rng.uniform(.55,1.25,size=(n,1))/np.sqrt(2*np.sum(abs(coeff)**2,axis=1,keepdims=True))
    coeff=np.column_stack((rng.uniform(-.4,.4,n),coeff))
    grids=c["grids"] if name=="paired" else [64]
    parts={g:[] for g in grids}
    for i in range(0,n,32):
        sl=slice(i,i+32)
        u,_=solve(coeff[sl],nu[sl],c,grids)
        for g,a in u.items():
            parts[g].append(a.astype(np.float32))
    u={f"u{g}":np.concatenate(x) for g,x in parts.items()}
    if not all(np.isfinite(v).all() for v in u.values()):
        raise RuntimeError(f"Invalid Burgers {name} reference.")
    tmp=path.with_suffix(".tmp.npz")
    np.savez(tmp,nu=nu,id=c["streams"][name]*10000+np.arange(n),**u)
    tmp.replace(path)
    print(f"Burgers formal generated {name}: {n}",flush=True)


def train(c,name,root,device,fit,validation,*,initialization=None,factory=make_model):
    folder=root/name
    folder.mkdir(parents=True,exist_ok=True)
    ck=folder/"fno.pt"
    init=c["seed"] if initialization is None else initialization
    torch.manual_seed(init)
    model=factory(name,c).to(device)
    if (folder/"train_log.json").exists():
        model.load_state_dict(torch.load(ck,weights_only=True,map_location=device)["model"])
        return model.eval()
    data,val=[tuple(torch.as_tensor(d[k],dtype=torch.float32,device=device) for k in ("u64","nu"))
              for d in (fit,validation)]
    y,nu=data
    setup=c["model"]
    opt=torch.optim.AdamW(model.parameters(),lr=setup["lr"],weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=setup["lr"],epochs=setup["epochs"],
          steps_per_epoch=math.ceil(len(y)/setup["batch"]),pct_start=.18)
    rng=torch.Generator().manual_seed(c["seed"])
    history,best=[],np.inf
    latest=folder/"latest.pt"
    if latest.exists():
        state=torch.load(latest,map_location="cpu",weights_only=True)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        rng.set_state(state["rng"])
        history,best=state["history"],state["best"]
    for epoch in range(len(history)+1,setup["epochs"]+1):
        start=perf_counter()
        if initialization is not None:
            torch.manual_seed(init+epoch*100000)
        model.train()
        for i in torch.randperm(len(y),generator=rng).split(setup["batch"]):
            opt.zero_grad(set_to_none=True)
            loss=F.mse_loss(model(y[i,0],nu[i])[:,1:],y[i,1:])
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.inference_mode():
            v,w=val
            p=torch.cat([model(a[:,0],b) for a,b in zip(v.split(32),w.split(32))])
            error=float(((p-v).flatten(1).norm(dim=1)/v.flatten(1).norm(dim=1)).mean())
        if error<best:
            best=error
            atomic_checkpoint(ck,{"model":model.state_dict(),"epoch":epoch})
        history.append({"epoch":epoch,"val_rel_l2":error,"seconds":perf_counter()-start})
        atomic_checkpoint(latest,{"model":model.state_dict(),"optimizer":opt.state_dict(),
            "scheduler":sched.state_dict(),"rng":rng.get_state(),"history":history,"best":best})
        if epoch%10==0 or epoch==setup["epochs"]:
            print(f"Burgers {name} seed {c['seed']} epoch {epoch}: {error:.5f}",flush=True)
    write_json(folder/"train_log.json",{"history":history,"parameters":sum(p.numel() for p in model.parameters())})
    latest.unlink(missing_ok=True)
    model.load_state_dict(torch.load(ck,weights_only=True,map_location=device)["model"])
    return model.eval()


def summary_features(data,n,c,lock,model,device):
    u=data[f"u{n}"]
    nu=data["nu"]
    pred=infer(model,device,u[:,0],nu).astype(float)
    r=metrics.trajectory_residual(pred,nu,c)
    z=metrics.lift(lock["candidate"]["inverse"],pred,r,nu,c)
    init=pred[:,0]
    spectrum=abs(np.fft.rfft(init,axis=-1))**2
    high=spectrum[:,9:].sum(axis=-1)/spectrum.sum(axis=-1)
    base=np.column_stack((np.log(nu),np.abs(gradient(init)).max(axis=-1),
                          np.sqrt(np.mean(init*init,axis=-1)),high))
    joint=np.column_stack((base,metrics.input_features(pred,nu,r,z)[:,2:]))
    error=np.linalg.norm((pred[:,1:]-u[:,1:]).reshape(len(u),-1),axis=1)/np.linalg.norm(u[:,1:].reshape(len(u),-1),axis=1)
    return {"input":base,"joint":joint,"error":error}


def mixed_gates(id_raw,ood_raw,id_feat,ood_feat,cal_feat,c):
    n=len(id_feat["error"])
    ood=np.arange(n+len(ood_feat["error"]))>=n
    bad=np.r_[id_feat["error"],ood_feat["error"]]>c["point_failure_rel_l2"]
    base={k:np.concatenate((id_raw[metrics.OP][k],ood_raw[metrics.OP][k])) for k in id_raw[metrics.OP]}
    out={}
    for key in ("input","joint"):
        x=np.concatenate((id_feat[key],ood_feat[key]))
        wc,wt=estimated_weights(cal_feat[key],x,
            {"base_weight_clip":c["weight_clip"],"weight_logistic_c":1.0})
        out[key]=gate_point(wt/(wt+wc.sum()),bad,ood,base,c)
    return out


def train_val(c,root,device):
    for name in ("train","val"):
        generate(c,name,root)
    fit,val=(arrays(root/f"{name}.npz") for name in ("train","val"))
    output={}
    for kind in KINDS:
        model=train(c,kind,root,device,fit,val)
        p=infer(model,device,val["u64"][:,0],val["nu"]).astype(float)
        error=abs(p[:,1:]-val["u64"][:,1:])
        fixed=error.mean(axis=0)
        fixed=np.maximum(fixed,.05*fixed.mean())
        residual=metrics.trajectory_residual(p,val["nu"],c)
        output[kind]=(model,fixed,{"candidate":LOCKED,"config":c,"pi_lambda":float(1/np.median(abs(residual)))})
    return output


def timing_gpu(cfg,models,data,device):
    out={}
    model=models["FNO"][0]
    for n in cfg["grids"]:
        u0=data[f"u{n}"][:48,0]
        nu=data["nu"][:48]
        values=[]
        for i in range(len(u0)):
            visc=nu[i:i+1]
            def path(gpu):
                pred=infer(model,device,u0[i:i+1],visc,batch=1).astype(float)
                r=metrics.trajectory_residual(pred,visc,cfg)
                if gpu:
                    args=[torch.as_tensor(v,dtype=torch.float32,device=device) for v in (pred,r,visc)]
                    z=adv_diff_gpu(*args,cfg).cpu().numpy()
                else:
                    z=metrics.lift("adv_diff",pred,r,visc,cfg)
                return metrics.mixed(metrics.envelope(z,cfg),.85)
            a,ct=timed(lambda:path(False))
            b,gt=timed(lambda:path(True))
            if i>=8:
                values.append((ct,gt,float(np.max(abs(a-b))/a.max())))
        cpu,gpu,parity=zip(*values)
        out[str(n)]={"cpu_path":timing_summary(cpu),"gpu_path":timing_summary(gpu),
                     "max_relative_scale_difference":max(parity)}
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    plan=json.loads(CONFIG.read_text(encoding="utf-8"))
    if args.seed not in plan["seeds"]:
        p.error("Seed not registered.")
    cfg=json.loads(Path(plan["base"]).read_text(encoding="utf-8"))|plan|{"seed":args.seed}
    cfg["model"]["epochs"]=cfg["epochs"]
    if args.smoke:
        cfg["counts"]=dict.fromkeys(cfg["counts"],24)
        cfg["model"]["epochs"]=2
        cfg["model"]["batch"]=8
        cfg["bootstrap_resamples"]=50
    root=Path("results")/("phase2_burgers_smoke" if args.smoke else "phase2_burgers")/f"seed_{args.seed}"
    root.mkdir(parents=True,exist_ok=True)
    device=torch_device()
    if not args.smoke and device.type!="cuda":
        raise RuntimeError("Formal Burgers requires CUDA.")
    models=train_val(cfg,root,device)
    for name in ("cal","id","paired","ood","ood_high"):
        generate(cfg,name,root)
    data={name:arrays(root/f"{name}.npz") for name in ("cal","id","paired","ood","ood_high")}
    results={}
    for kind,(model,fixed,locked) in models.items():
        features={name:summary_features(data[name],64,cfg,locked,model,device)
                  for name in ("cal","id","ood","ood_high")}
        output={}
        for alpha in cfg["alpha_levels"]:
            c=cfg|{"alpha":alpha}
            lock=locked|{"config":c}
            cal,c_feature=metrics.calibration(data["cal"],c,lock,fixed,model,device)
            summary,raw={},{}
            for name,n in (("id",64),("paired",64),("paired",128),("paired",256),("ood",64),("ood_high",64)):
                label=f"{name}_{n}"
                target_data=data[name]
                if name=="ood_high":
                    # Same weighted estimator entry, with its own deterministic namespace.
                    c_eval=c|{"streams":c["streams"]|{"ood":c["streams"][name]}}
                    split="ood"
                else:
                    c_eval,split=c,name
                summary[label],raw[label]=metrics.evaluate(target_data,split,n,c_eval,lock,fixed,cal,c_feature,model,device)
                for method in metrics.METHODS:
                    values=raw[label][method]
                    q=metrics.value_at_coverage(values["score"],1-alpha)
                    summary[label]["methods"][method]["matched_nominal_width"]=float(2*q*values["scale_mean"].mean())
                print(f"Burgers {kind} alpha={alpha} finished {label}",flush=True)
            gates={name:mixed_gates(raw["id_64"],raw[f"{name}_64"],features["id"],
                                    features[name],features["cal"],c) for name in ("ood","ood_high")}
            output[str(alpha)]={"summary":summary,"paired":metrics.paired(raw,summary,c),"gates":gates}
            np.savez_compressed(root/kind/f"fields_{alpha}.npz",**{
                f"{label}__{name}__{col}":v for label,obj in raw.items()
                for name,row in obj.items() for col,v in row.items()})
        results[kind]=output
    result={"seed":args.seed,"settings":cfg,"candidate":LOCKED,"results":results,
            "data_sha256":{name:digest(root/f"{name}.npz") for name in cfg["counts"]},
            "gpu":torch.cuda.get_device_name() if device.type=="cuda" else None,
            "latency":timing_gpu(cfg,models,data["paired"],device) if device.type=="cuda" else None}
    write_json(root/"metrics.json",result)


if __name__=="__main__":
    main()
