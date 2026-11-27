import json
from pathlib import Path

import numpy as np
import torch

from burgers_phase1_data import CONFIG, ROOT
from burgers_phase1_model import infer, model_from_checkpoint, torch_device
from burgers_phase1_ops import envelope, lift, mixed, trajectory_residual
from darcy_day15 import arrays, write_json
from darcy_day15_ops import timed, timing_summary


def propagator(n, nu, dt, gamma):
    k = torch.arange(n//2+1, device=nu.device, dtype=nu.dtype)
    z = -dt*(nu[:,None]*k.square()+gamma)
    ex = z.exp()
    phi1 = torch.where(z == 0, torch.ones_like(z), torch.expm1(z)/z)
    return ex,dt*phi1


def adv_diff_gpu(u, residual, nu, config):
    fields,t,n = residual.shape
    dt = config["frame_dt"]/config["advection_substeps"]
    ex, coef = propagator(n,nu,dt,config["gamma"])
    forcing = torch.fft.rfft(residual)*coef[:,None]
    mid = (u[:,1:]+u[:,:-1])/2
    vel = (mid+mid.roll(1, dims=-1))/2
    depart = torch.arange(n,device=u.device)-dt*vel*n/(2*np.pi)
    cell = depart.floor().long()
    indexes = cell.remainder(n)
    periods = torch.div(cell,n,rounding_mode="floor")
    frac = depart-cell
    dx = depart.roll(-1,dims=-1)-depart
    dx[...,-1] += n
    if bool((dx<=0).any()):
        raise ValueError("Folded conservative transport departure map.")
    state = torch.zeros((fields,n),device=u.device,dtype=u.dtype)
    values = []
    for frame in range(t):
        ix = indexes[:,frame]
        for _ in range(config["advection_substeps"]):
            cumulative = state.cumsum(dim=-1)
            mass = cumulative[:,-1:]
            base=(cumulative-state).gather(-1,ix)
            primitive=base+periods[:,frame]*mass+frac[:,frame]*state.gather(-1,ix)
            z = primitive.roll(-1,dims=-1)-primitive
            z[:,-1:] += mass
            state = torch.fft.irfft(ex*torch.fft.rfft(z)+forcing[:,frame],n=n)
        values.append(state)
    return torch.stack(values,dim=1)


def main():
    config=json.loads(CONFIG.read_text(encoding="utf-8"))
    dev=torch_device()
    if dev.type!="cuda":
        raise RuntimeError("CUDA is required for this comparison.")
    data=arrays(ROOT/"paired.npz")
    model,_=model_from_checkpoint(config,dev)
    out={}
    for n in config["grids"]:
        u=data[f"u{n}"][:64]
        nu=data["nu"][:64]
        pred=infer(model,dev,u[:,0],nu).astype(np.float64)
        r=trajectory_residual(pred,nu,config)
        cpu=lift("adv_diff",pred,r,nu,config)
        tests={}
        for name,tp in (("float64",torch.float64),("float32",torch.float32)):
            a,b,z=[torch.as_tensor(x,device=dev,dtype=tp) for x in (pred,r,nu)]
            gpu=adv_diff_gpu(a,b,z,config).cpu().numpy()
            tests[name]={"max_abs":float(abs(cpu-gpu).max()),"max_relative_to_global_peak":float(abs(cpu-gpu).max()/abs(cpu).max()),
                         "relative_l2":float(np.linalg.norm(cpu-gpu)/np.linalg.norm(cpu))}
        times={"cpu_single_inverse":[],"gpu_single_inverse":[],"cpu_inverse_path_end_to_end":[],
               "gpu_inverse_path_end_to_end":[],"cpu_batch32_per_field":[],"gpu_batch32_per_field":[]}
        for i in range(72):
            idx=i%64
            u0=u[idx:idx+1,0]
            vv=nu[idx:idx+1]
            def total(on_gpu):
                pp=infer(model,dev,u0,vv,batch=1).astype(np.float64)
                rr=trajectory_residual(pp,vv,config)
                if on_gpu:
                    args = [torch.as_tensor(x,device=dev,dtype=torch.float32) for x in (pp,rr,vv)]
                    zz,inv_ms=timed(lambda:adv_diff_gpu(*args,config).cpu().numpy())
                else:
                    zz,inv_ms=timed(lambda:lift("adv_diff",pp,rr,vv,config))
                mixed(envelope(zz,config),.85)
                return inv_ms
            cpu_inv,cpu_ms=timed(lambda:total(False))
            gpu_inv,gpu_ms=timed(lambda:total(True))
            if i>=8:
                for key,v in zip(list(times)[:4],(cpu_inv,gpu_inv,cpu_ms,gpu_ms)):
                    times[key].append(v)
        for i in range(18):
            sl=slice(0,32) if i%2==0 else slice(32,64)
            def gpu_batch():
                args=[torch.as_tensor(x[sl],device=dev,dtype=torch.float32) for x in (pred,r,nu)]
                return adv_diff_gpu(*args,config).cpu()
            _,ct=timed(lambda:lift("adv_diff",pred[sl],r[sl],nu[sl],config))
            _,gt=timed(gpu_batch)
            if i>=2:
                times["cpu_batch32_per_field"].append(ct/32)
                times["gpu_batch32_per_field"].append(gt/32)
        out[str(n)]={"agreement":tests,"timings_ms":{key:timing_summary(t) for key,t in times.items()}}
    write_json(Path("results/phase2_burgers_gpu_audit.json"),{"gpu":torch.cuda.get_device_name(),
                   "torch":str(torch.__version__),"grids":out})


if __name__=="__main__":
    main()
