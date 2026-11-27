import torch
from torch import nn

from darcy_fno import FNO


def coordinates(a):
    xy = (torch.arange(a.shape[-1],device=a.device,dtype=a.dtype)+.5)/a.shape[-1]
    x,y = torch.meshgrid(xy,xy,indexing="ij")
    return torch.stack((x,y)),16*x*(1-x)*y*(1-y)


class DarcyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        def block(a,b):
            return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GELU(),nn.Conv2d(b,b,3,padding=1),nn.GELU())
        self.first=block(3,24)
        self.second=block(24,48)
        self.middle=block(48,96)
        self.decode2=block(144,48)
        self.decode1=block(72,24)
        self.out=nn.Conv2d(24,1,1)
        self.pool=nn.AvgPool2d(2)
        self.up=nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False)
        self.register_buffer("input_std",torch.ones(()))
        self.register_buffer("output_scale",torch.ones(()))

    def forward(self,a):
        xy,mask=coordinates(a)
        x=self.first(torch.cat(((a.log()/self.input_std)[:,None],xy.expand(len(a),-1,-1,-1)),dim=1))
        y=self.second(self.pool(x))
        z=self.middle(self.pool(y))
        w=self.decode2(torch.cat((self.up(z),y),dim=1))
        return self.out(self.decode1(torch.cat((self.up(w),x),dim=1)))[:,0]*mask*self.output_scale


class DarcyDeepONet(nn.Module):
    def __init__(self,sensors):
        super().__init__()
        self.sensors=sensors
        self.branch=nn.Sequential(nn.Linear(sensors*sensors,192),nn.GELU(),nn.Linear(192,128),
                                  nn.GELU(),nn.Linear(128,96))
        self.trunk=nn.Sequential(nn.Linear(2,128),nn.GELU(),nn.Linear(128,128),
                                 nn.GELU(),nn.Linear(128,96))
        self.register_buffer("input_std",torch.ones(()))
        self.register_buffer("output_scale",torch.ones(()))

    def forward_at(self,a,query_size):
        if a.shape[-1]!=self.sensors:
            raise ValueError("DeepONet branch must receive the paired coarse sensor field.")
        target = a.new_empty((1,query_size,query_size))
        xy,mask=coordinates(target)
        branch=self.branch((a.log()/self.input_std).flatten(1))
        trunk=self.trunk(xy.permute(1,2,0)).reshape(-1,96)
        return (branch@trunk.T).reshape(len(a),query_size,query_size)*mask*self.output_scale

    def forward(self,a):
        return self.forward_at(a,self.sensors)


def model_for(name,config):
    if name=="FNO":
        return FNO(config["model"])
    if name=="U-Net":
        return DarcyUNet()
    if name=="DeepONet":
        return DarcyDeepONet(config["coarse"])
    raise ValueError(name)
