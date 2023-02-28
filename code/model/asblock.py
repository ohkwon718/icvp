import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model_utils import init_weights


class ASBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, ndim=2, 
                norm='batch', nonlinearity='leaky_relu', a = 0.1):
        super(ASBlock, self).__init__()
        

        if isinstance(dilation, (int, tuple)):
            dilation = [dilation]
        l = len(dilation)
        if isinstance(out_channels, int):
            out_channels = [out_channels] * l
        if isinstance(kernel_size, (int, tuple)):
            kernel_size = [kernel_size] * l
        
        padding = []
        for kt, dt in zip(kernel_size, dilation):
            if isinstance(kt, int):
                kt = (kt, ) * ndim
            if isinstance(dt, int):
                dt = (dt,) * ndim
            assert len(kt) == ndim and len(dt) == ndim
            padding.append((int((k + (k-1)*(d-1)-1)/2) for k, d in zip(kt, dt)))

        if nonlinearity == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif nonlinearity == 'leaky_relu':
            self.activation = nn.LeakyReLU(negative_slope=a, inplace=True)
        
        bias = norm == 'none'

        convs = []
        for c, ks, ds, ps in zip(out_channels, kernel_size, dilation, padding):
            if ndim == 2:
                convs.append(nn.Conv2d(in_channels, c, ks, 1, ps, dilation=ds, bias=bias))
            elif ndim == 3:
                convs.append(nn.Conv3d(in_channels, c, ks, 1, ps, dilation=ds, bias=bias))                
            
        self.convs = nn.ModuleList(convs)        
        
        init_weights(self, nonlinearity, a)

    def forward(self, x):
        return torch.cat([conv(x) for conv in self.convs], dim=1)
        
