"""Phase 2D adapters for the frozen Phase 2A–C data generators and FNOs."""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

import allen_cahn_phase2_formal as ac
import burgers_phase2 as burg
import darcy_phase2 as darcy
from allen_cahn_phase2_probe import StepFNO, reconstruct, rollout, scale
from darcy_data import apply_operator
from darcy_day15 import arrays, digest
from darcy_day15_ops import lift, mixture, physical_scale
from phase2d_burgers_diagnosis import features
from phase2d_statistics import flat


class DropoutFNO(nn.Module):
    """Channel dropout on hidden blocks. MC draws keep one mask across a rollout."""

    def __init__(self, model, p):
        super().__init__()
        self.model = model
        self.dropout = nn.Dropout(p)
        self.fixed_draw = False
        self.masks = {}
        blocks = model.blocks if hasattr(model, "blocks") else model.layers
        for block in blocks:
            block.register_forward_hook(self.mask_hidden)

    def mask_hidden(self, module, inputs, value):
        if not self.dropout.training:
            return value
        if module not in self.masks:
            shape = value.shape[:2] + (1,) * (value.ndim - 2)
            self.masks[module] = self.dropout(value.new_ones(shape))
        return value * self.masks[module]

    def forward(self, *values):
        if not self.fixed_draw:
            self.masks.clear()
        return self.model(*values)

    @property
    def input_std(self):
        return self.model.input_std

    @property
    def output_scale(self):
        return self.model.output_scale


class Protocol:
    def __init__(self, pde, seed, grid, root, frozen, device, smoke=False):
        self.pde, self.seed, self.grid, self.root, self.frozen, self.device = pde, seed, grid, root, frozen, device
        self.smoke = smoke
        self.manifest = self.read(frozen / "frozen_manifest.json")
        reference = self.manifest.get("references",{}).get(pde,f"{pde}_reference.json")
        self.reference = self.read(frozen / reference)
        if self.manifest["seed"] != seed or self.reference["seed"] != seed:
            raise ValueError("Frozen snapshot seed differs from the requested seed.")
        self.plan = self.reference["settings" if pde == "burgers" else "config"]
        self.config = darcy.data_config(self.plan, grid) if pde == "darcy" else self.plan.copy()
        self.config["seed"] = seed
        self.key = f"darcy{grid}" if pde == "darcy" else pde
        self.frames = self.config.get("frames")
        self.cal_frames = self.frames // 2 if pde == "allen_cahn" else self.frames
        self.files, self.data_matches = {}, {}
        if smoke:
            self.plan = self.plan | {"counts": dict.fromkeys(self.plan["counts"], 24),
                                     "streams": {k: v+400 for k,v in self.plan["streams"].items()}}
            self.config |= {"counts": self.plan["counts"], "streams": self.plan["streams"]}
            if pde == "darcy":
                self.config["model"] = self.config["model"] | {"epochs": 1}
            elif pde == "burgers":
                self.config["model"] = self.config["model"] | {"epochs": 2}
            else:
                self.config |= {"epochs":1, "train_steps_per_epoch":2}
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    def make_model(self, dropout=0., unet=False):
        if self.pde == "darcy":
            model = darcy.model_for("FNO", self.config)
        elif self.pde == "burgers":
            model = burg.make_model("FNO", self.config)
        else:
            model = (ac.StepUNet if unet else StepFNO)(self.config)
        return DropoutFNO(model, dropout) if dropout else model

    def open_frozen(self, unet=False):
        key = "allen_cahn_unet" if unet else self.key
        record = self.manifest["weights"][key]
        file = self.frozen / record["file"]
        if record["sha256"] != digest(file):
            raise RuntimeError(f"Frozen {key} checkpoint does not match the local manifest.")
        model = self.make_model(unet=unet).to(self.device)
        model.load_state_dict(torch.load(file, map_location=self.device, weights_only=True)["model"])
        return model.eval()

    def data_path(self, split, n=None):
        intermediate = self.pde == "darcy" and self.grid == 32 and split == "paired_test" and n == 64
        key = split, intermediate
        if key in self.files:
            return self.files[key]
        root = self.root / ("intermediate_data" if intermediate else "data")
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{split}.npz"
        if self.pde == "darcy":
            config = self.config | ({"fine":64} if intermediate else {})
            darcy.generate_split(self.plan, self.seed, config, split, root, 1)
            expected = self.reference["tracks"][str(self.grid)]["data_sha256"].get(f"{root.name}/{split}.npz")
        elif self.pde == "burgers":
            burg.generate(self.config, split, root)
            expected = self.reference["data_sha256"][split]
        else:
            ac.cache_data(self.config, split, root)
            expected = self.reference["data_sha256"][split]
        with np.load(path) as raw:
            ids = self.plan["streams"][split] * 10000
            if not np.array_equal(raw["id"], ids + np.arange(self.plan["counts"][split])):
                raise ValueError(f"{self.pde} wrong field IDs for {split}.")
        self.files[key] = path
        self.data_matches[str(path.relative_to(self.root).as_posix())] = digest(path) == expected
        return path

    def train_member(self, label, init, dropout=0.):
        folder = self.root / label
        folder.mkdir(exist_ok=True)
        train, val = (self.data_path(name) for name in ("train", "val"))
        if self.pde == "darcy":
            model, path = darcy.train_model("FNO",init,self.config, train.parent,folder,self.device,
                                            factory=lambda *_: self.make_model(dropout))
            model.load_state_dict(torch.load(path,map_location=self.device,weights_only=True)["model"])
            history = self.read(folder / "training.json")["history"]
        elif self.pde == "burgers":
            model = burg.train(self.config,"FNO",folder,self.device,arrays(train),arrays(val),
                               initialization=init,factory=lambda *_:self.make_model(dropout))
            history = self.read(folder / "FNO" / "train_log.json")["history"]
        else:
            signature = {"initialization":init,"dropout":dropout,"config":self.config,
                         "train":digest(train),"val":digest(val)}
            model, log = ac.train(self.config,"FNO",arrays(train),arrays(val),self.device,folder,
                                  signature,initialization=init,factory=lambda _:self.make_model(dropout))
            history = log["history"]
        return model.eval(), sum(h.get("seconds",h.get("elapsed_s",0.)) for h in history)

    def split(self, data, n, frames):
        if self.pde == "darcy":
            suffix = "" if n == self.grid else "_f"
            return (data[f"a{suffix}"],), data[f"u{suffix}"]
        target = data[f"u{n}"]
        coeff = ("nu", "rho") if self.pde == "allen_cahn" else ("nu",)
        return (target[:,0], *(data[k] for k in coeff)), target[:,1:frames+1]

    def predict(self, model, inputs, n, frames):
        if self.pde == "darcy":
            return darcy.prediction(model,"FNO",inputs[0],n,self.device)
        if self.pde == "burgers":
            return burg.infer(model,self.device,*inputs)[:,1:]
        data = {f"u{n}":inputs[0][:,None], "nu":inputs[1],"rho":inputs[2]}
        return rollout(model,data,n,frames,self.device,len(inputs[0]))[:,1:]

    def op_scale(self, prediction, inputs):
        if self.pde == "darcy":
            a = inputs[0].astype(float)
            error = apply_operator(a,prediction.astype(float))-1
            z = lift(self.plan["inverse"],a,error)
            return mixture(physical_scale(z,self.config),self.plan["lambda"]).astype(np.float32)
        full = np.concatenate((inputs[0][:,None],prediction), axis=1).astype(float)
        if self.pde == "burgers":
            ev = burg.metrics
            r = ev.trajectory_residual(full,inputs[1],self.config)
            z = ev.lift("adv_diff",full,r,inputs[1],self.config)
            return ev.mixed(ev.envelope(z,self.config),.85).astype(np.float32)
        z,_ = reconstruct(full,*inputs[1:],self.config,self.config["inverse"])
        return scale(z,self.config).astype(np.float32)

    def summaries(self, inputs, spectral=False):
        a,*coeff = inputs
        if self.pde == "darcy":
            return darcy.physical(a.astype(float))
        if self.pde == "burgers":
            return features(a,coeff[0])[int(spectral)]
        dx = np.roll(a,1,axis=-1)-np.roll(a,-1,axis=-1)
        dy = np.roll(a,1,axis=-2)-np.roll(a,-1,axis=-2)
        return np.column_stack((*coeff, np.sqrt(np.mean(flat(a)**2,axis=1)),
                                np.sqrt(np.mean(flat(dx*dx+dy*dy),axis=1))))

    def full_grid(self, inputs):
        a,*coeff = inputs
        field = np.log(a) if self.pde == "darcy" else a
        return np.column_stack((flat(field), *coeff))

    def targets(self):
        if self.pde == "darcy":
            if self.grid == 32:
                return [("paired_test",n,None) for n in (32,64,128)]
            return [("id_test",64,None),("ood_a",64,None),("ood_b",64,None),
                    ("paired_test",64,None),("paired_test",128,None)]
        if self.pde == "burgers":
            return [("id",64,self.frames),("ood",64,self.frames),("ood_high",64,self.frames),
                    *(("paired",n,self.frames) for n in self.config["grids"])]
        return [(s,n,h) for s,n in ac.TARGETS for h in (self.cal_frames,self.frames)]
