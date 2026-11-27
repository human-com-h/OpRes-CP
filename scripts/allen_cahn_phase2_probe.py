import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.fft import irfft2, rfft2
from scipy.ndimage import gaussian_filter
from torch import nn
from torch.nn import functional as F

from allen_cahn_phase2_data import CONFIG, ROOT, initial, solve, spectrum
from burgers_phase1_data import phi
from burgers_phase1_model import torch_device
from darcy_day15 import arrays, band_summary, comparison, digest, evaluate_bands, value_at_coverage, write_json
from darcy_day15_ops import timed, timing_summary
from darcy_fno import SpectralBlock, atomic_checkpoint
from darcy_phase0_eval import bilinear, boot_interval, quantile


INV = ("diffusion", "reaction_strang")
UNIFORM, FIXED, PI, OP = ("Uniform-CP", "CP-B", "Full-Field PI-CP", "OpRes-CP")


class StepFNO(nn.Module):
    def __init__(self, c):
        super().__init__()
        w = c["width"]
        self.in_proj = nn.Conv2d(3, w, 1)
        self.layers = nn.ModuleList(SpectralBlock(w, c["modes"]) for _ in range(c["layers"]))
        self.out_proj = nn.Sequential(nn.Conv2d(w,w,1),nn.GELU(),nn.Conv2d(w,1,1))
        self.dt = c["dt"]

    def forward(self, u, nu, rho):
        diffusivity = ((nu-.04)/.02)[:,None,None].expand_as(u)
        reaction = ((rho-.8)/.3)[:,None,None].expand_as(u)
        y = self.in_proj(torch.stack((u,diffusivity,reaction),dim=1))
        for layer in self.layers:
            y = F.gelu(layer(y))
        return u+self.dt*self.out_proj(y)[:,0]


@torch.inference_mode()
def rollout(model, data, n, frames, device, batch):
    source = data[f"u{n}"][:,0]
    result = []
    for start in range(0,len(source),batch):
        sl = slice(start,start+batch)
        y,nu,rho = [torch.as_tensor(a[sl],dtype=torch.float32,device=device)
                    for a in (source,data["nu"],data["rho"])]
        trajectory = [y]
        for _ in range(frames):
            y = model(y,nu,rho)
            trajectory.append(y)
        result.append(torch.stack(trajectory,dim=1).cpu().numpy())
    return np.concatenate(result)


def train(c, device):
    if (ROOT/"fno.pt").exists():
        return
    train_data, val = (arrays(ROOT/f"{name}.npz") for name in ("train","val"))
    u,nu,rho = [torch.as_tensor(train_data[k],dtype=torch.float32,device=device)
                for k in ("u64","nu","rho")]
    torch.manual_seed(c["seed"])
    model = StepFNO(c).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.002)
    lr = torch.optim.lr_scheduler.CosineAnnealingLR(opt,c["epochs"],eta_min=0.00015)
    generator = np.random.default_rng(c["seed"])
    best, history = np.inf, []
    for epoch in range(1,c["epochs"]+1):
        start = perf_counter()
        model.train()
        for _ in range(c["train_steps_per_epoch"]):
            i,t = (generator.integers(high,size=c["batch"]) for high in (len(u),u.shape[1]-1))
            a,b = (torch.as_tensor(v,device=device) for v in (i,t))
            predicted = model(u[a,b],nu[a],rho[a])
            loss = F.mse_loss(predicted,u[a,b+1])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        lr.step()
        model.eval()
        val_pred = rollout(model,val,64,20,device,c["batch"])
        loss = float(np.linalg.norm(val_pred[:,1:]-val["u64"][:,1:])/np.linalg.norm(val["u64"][:,1:]))
        history.append({"epoch":epoch,"val_relative_l2":loss,"elapsed_s":perf_counter()-start})
        if loss < best:
            best = loss
            atomic_checkpoint(ROOT/"fno.pt",{"model":model.state_dict(),"epoch":epoch})
        print(f"Allen–Cahn epoch {epoch}: Val rollout L2={loss:.5f}",flush=True)
    write_json(ROOT/"training.json",{"history":history,"gpu":torch.cuda.get_device_name() if device.type=="cuda" else None})


def model_load(c,device):
    ck = torch.load(ROOT/"fno.pt",map_location=device,weights_only=True)
    model=StepFNO(c).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(),ck["epoch"]


def reconstruct(pred,nu,rho,c,inv,*,reaction_jacobian=True):
    n = pred.shape[-1]
    dt = c["dt"]
    k2, mask = spectrum(n)
    diff = -nu[:,None,None]*k2
    decay = dt*(diff-c["gamma"])
    ex = np.exp(decay)
    force = dt*phi(decay)[0]
    half = np.exp(decay/2)
    kdiff = diff
    state = np.zeros_like(pred[:,0],dtype=np.float64)
    values,residual = [],[]
    for j in range(1,pred.shape[1]):
        left,right = (pred[:,t].astype(float) for t in (j-1,j))
        mid = (left+right)*.5
        nonlin = irfft2(mask*rfft2(mid-mid**3))
        r = (right-left)/dt - irfft2(kdiff*rfft2(mid)) - rho[:,None,None]*nonlin
        residual.append(r)
        if inv == INV[1]:
            jac = 1.
            if reaction_jacobian:
                jac = np.clip(rho[:,None,None]*(1-3*mid**2),-c["reaction_clip"],c["reaction_clip"])
                jac = np.minimum(jac-c["positive_damping"]*np.maximum(jac,0),c["max_growth"])
                jac = np.exp(dt*jac)
            state = irfft2(half*rfft2(state+dt*r/2))
            state = irfft2(half*rfft2(jac*state))+dt*r/2
        else:
            state = irfft2(ex*rfft2(state)+force*rfft2(r))
        values.append(state)
    return np.stack(values,axis=1),np.stack(residual,axis=1)


def scale(z,c):
    square = z*z
    avg = np.mean(square,axis=(1,2,3),keepdims=True)
    radius = c["spatial_radius"]*z.shape[-1]/(2*np.pi)
    smooth = gaussian_filter(square,(0,0,radius,radius),mode="wrap")+0.01*avg
    return np.sqrt((1-c["lambda"])*1.01*avg+c["lambda"]*smooth)


def chunks(data,n,c,model,device,inv):
    frames = data[f"u{n}"].shape[1]-1
    pred = rollout(model,data,n,frames,device,c["batch"] if n==64 else 8)
    if not np.isfinite(pred).all():
        raise RuntimeError("Nonfinite FNO rollout.")
    for i in range(0,len(pred),8):
        sl = slice(i,i+8)
        z,r = reconstruct(pred[sl],data["nu"][sl],data["rho"][sl],c,inv)
        if not np.isfinite(z).all():
            raise RuntimeError("Nonfinite lifted rollout.")
        yield sl,pred[sl],z,r


def lock(c,model,device,epoch):
    path = ROOT/"lock.json"
    if path.exists():
        raise RuntimeError("Val choice already locked.")
    data = arrays(ROOT/"val.npz")
    val_rows = []
    for inv in INV:
        scores,means,rs,defects = [],[],[],[]
        fixed = np.zeros(data["u64"].shape[2:])
        for sl,p,z,r in chunks(data,64,c,model,device,inv):
            err = abs(p[:,1:]-data["u64"][sl,1:])
            s = scale(z,c)
            if sl.start == 0:
                changed = p.copy()
                changed[:,11:] += 0.9
                a,_ = reconstruct(changed,data["nu"][sl],data["rho"][sl],c,inv)
                prefix = float(np.max(abs(a[:,:10]-z[:,:10])))
            fixed += err.sum(axis=(0,1))
            scores.extend((err/s).max(axis=(1,2,3)))
            means.extend(s.mean(axis=(1,2,3)))
            rs.extend(np.median(abs(r),axis=(1,2,3)))
            defects.extend(np.max(abs(z),axis=(1,2,3)))
        q = value_at_coverage(scores,.9)
        val_rows.append({"inverse":inv,"matched90_width":float(2*q*np.mean(means)),
                         "causal_prefix_delta":prefix,"lift_max":float(max(defects))})
    fixed /= len(data["u64"])*(data["u64"].shape[1]-1)
    fixed = np.maximum(fixed,.05*fixed.mean())
    np.save(ROOT/"fixed.npy",fixed)
    choice = min(val_rows,key=lambda row: row["matched90_width"])
    write_json(path,{"candidate":choice["inverse"],"val":val_rows,
        "pi_lambda":float(1/np.median(rs)),"config":c,"epoch":epoch,
        "checkpoint":digest(ROOT/"fno.pt"),"val_sha256":digest(ROOT/"val.npz"),
        "fixed_sha256":digest(ROOT/"fixed.npy"),
        "sources":{str(x.as_posix()):digest(x) for x in [CONFIG,*Path("scripts").glob("allen_cahn_phase2*.py")]}})
    print("Allen–Cahn locked",choice,flush=True)


def scales(z,r,fixed,c,pi):
    return {UNIFORM:np.ones_like(r),FIXED:np.broadcast_to(fixed,r.shape),
            PI:1+pi*abs(r),OP:scale(z,c)}


def observations(data,n,c,model,device,locked):
    fixed = np.load(ROOT/"fixed.npy")
    if n != 64:
        fixed = bilinear(fixed[None],n)[0]
    for sl,p,z,r in chunks(data,n,c,model,device,locked["candidate"]):
        truth = data[f"u{n}"][sl,1:]
        yield sl,abs(p[:,1:]-truth),p,z,truth,scales(z,r,fixed,c,locked["pi_lambda"])


def choose_cal(c,model,device,locked):
    data = arrays(ROOT/"cal.npz")
    fields = {}
    for _,err,_,_,_,ss in observations(data,64,c,model,device,locked):
        for name,s in ss.items():
            fields.setdefault(name,[]).extend((err/s).max(axis=(1,2,3)))
    return {name:quantile(np.array(x),.1) for name,x in fields.items()}


def eval_split(split,n,c,model,device,locked,qs):
    data = arrays(ROOT/f"{split}.npz")
    nfields = len(data["id"])
    captured = {20:{},40:{}}
    diag = {20:[],40:[]}
    for sl,err,p,z,truth,ss in observations(data,n,c,model,device,locked):
        for frames in captured:
            error = err[:,:frames]
            truth_window = truth[:,:frames]
            l2 = np.linalg.norm(error.reshape(len(p),-1),axis=1)/np.linalg.norm(truth_window.reshape(len(p),-1),axis=1)
            diag[frames].append((float(l2.sum()),float(abs(p[:,1:frames+1]).max()),float(abs(z[:,:frames]).max())))
            dx = np.roll(truth_window,1,axis=-1)-np.roll(truth_window,-1,axis=-1)
            dy = np.roll(truth_window,1,axis=-2)-np.roll(truth_window,-1,axis=-2)
            grad = (dx*dx+dy*dy).reshape(len(p),-1)
            top_n = int(np.ceil(grad.shape[1]*.1))
            top = np.argpartition(grad,-top_n,axis=1)[:,-top_n:]
            for name,s in ss.items():
                # Recompute the scalar RMS using only frames in this evaluation horizon.
                local = scale(z[:,:frames],c) if name==OP else s[:,:frames]
                flat = local.reshape(len(p),-1)
                row = evaluate_bands(error.reshape(len(p),-1)/flat,flat,qs[name],top)
                for k,v in row.items():
                    captured[frames].setdefault(name,{}).setdefault(k,[]).append(v)
    groups,raw = {},{}
    for frames,methods in captured.items():
        raw[frames] = {m:{k:np.concatenate(v) for k,v in x.items()} for m,x in methods.items()}
        summaries = {}
        for name,row in raw[frames].items():
            q90 = value_at_coverage(row["score"],.9)
            summaries[name] = {**band_summary(row),"matched90_width":float(2*q90*row["scale_mean"].mean())}
        checks = {m:comparison(raw[frames][OP],raw[frames][m],c,c|{"alpha":.1}) for m in (UNIFORM,FIXED,PI)}
        summary = np.array(diag[frames])
        groups[str(frames*c["dt"])] = {"methods":summaries,"comp":checks,"n":nfields,
           "mean_rel_l2":float(summary[:,0].sum()/nfields),"fno_max":float(summary[:,1].max()),
           "lift_max":float(summary[:,2].max())}
    return groups,raw


def audit_reference(c):
    output = {}
    for split in ("val","ood"):
        nu,rho,coeff = initial(c,split)
        args = (coeff[:5],nu[:5],rho[:5])
        config = c|{"reference":256,"substeps":2*c["substeps"]}
        ref = solve(*args,c,[64,128],40)
        fine = solve(*args,config,[64,128],40)
        output[split] = {str(n):float(np.linalg.norm(ref[n]-fine[n])/np.linalg.norm(fine[n])) for n in (64,128)}
    return output


def evaluate(c,model,device,epoch):
    lock_path = ROOT/"lock.json"
    locked = json.loads(lock_path.read_text(encoding="utf-8"))
    if locked["config"] != c or locked["epoch"] != epoch or locked["checkpoint"] != digest(ROOT/"fno.pt"):
        raise RuntimeError("Config or weight differs from Val lock.")
    if (locked["fixed_sha256"] != digest(ROOT/"fixed.npy")
            or any(digest(Path(p)) != h for p,h in locked["sources"].items())):
        raise RuntimeError("Val scale or source differs from Val lock.")
    qs = choose_cal(c,model,device,locked)
    all_rows,all_raw = {},{}
    for split,n in (("id",64),("paired",64),("paired",128),("ood",64)):
        label = f"{split}_{n}"
        all_rows[label],all_raw[label] = eval_split(split,n,c,model,device,locked,qs)
        print("Allen–Cahn blind evaluation",label,flush=True)
    drift = {}
    for split in all_raw:
        drift[split] = {}
        for name,short in all_raw[split][20].items():
            long = all_raw[split][40][name]
            d = long["coverage"].astype(float)-short["coverage"]
            drift[split][name] = {"pp":float(100*d.mean()),"ci95":[100*v for v in boot_interval(d,c["bootstrap_resamples"],c["bootstrap_seed"])],
                                  "width_ratio":float(long["width"].mean()/short["width"].mean())}
    latencies = {}
    paired = arrays(ROOT/"paired.npz")
    for n in (64,128):
        values = []
        for i in range(24):
            sample = {key:arr[i:i+1] for key,arr in paired.items()}
            p,model_ms = timed(lambda: rollout(model,sample,n,40,device,1))
            _,op_ms = timed(lambda: scale(reconstruct(p,sample["nu"],sample["rho"],c,locked["candidate"])[0],c))
            if i>=4:
                values.append((model_ms,op_ms,model_ms+op_ms))
        latencies[str(n)] = {k:timing_summary(v) for k,v in zip(("fno","post","total"),zip(*values))}
    out={"lock":locked,"lock_sha256":digest(lock_path),"thresholds":qs,"results":all_rows,
         "horizon_drift":drift,"latencies":latencies,"reference_audit":audit_reference(c),
         "data_sha256":{k:digest(ROOT/f"{k}.npz") for k in c["streams"]}}
    write_json(ROOT/"metrics.json",out)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=("train","lock","eval"))
    arg=p.parse_args()
    c=json.loads(CONFIG.read_text(encoding="utf-8"))
    device=torch_device()
    if arg.stage=="train":
        train(c,device)
    else:
        model,epoch=model_load(c,device)
        (lock if arg.stage=="lock" else evaluate)(c,model,device,epoch)
