import torch
import torch.nn as nn


def prove_module(model, object, func, i = 0):
    if isinstance(model, object):
        func(model)
        return

    submodules = []
    for sub in dir(model):
        sub = getattr(model, sub)
        if isinstance(sub, nn.ModuleList) or isinstance(sub, nn.Sequential):
            for s in sub:
                if isinstance(s, nn.Module):
                    submodules.append(s)
        elif isinstance(sub, nn.Module):
            submodules.append(sub)    
    
    for sub in submodules:      
        prove_module(sub, object, func, i+1)
    

def init_weights(module, nonlinearity='leaky_relu', a=0.0):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=nonlinearity, a=a)
        elif isinstance(m, nn.ConvTranspose2d):
            w = torch.tensor(   [[0.0625, 0.1250, 0.0625],
                                [0.1250, 0.2500, 0.1250],
                                [0.0625, 0.1250, 0.0625]]   ) / m.in_channels
            m.weight.data += w[None,None]
        elif isinstance(m, nn.ConvTranspose3d):
            w = torch.tensor(   [[[0.0156, 0.0312, 0.0156],
                                [0.0312, 0.0625, 0.0312],
                                [0.0156, 0.0312, 0.0156]],

                                [[0.0312, 0.0625, 0.0312],
                                [0.0625, 0.1250, 0.0625],
                                [0.0312, 0.0625, 0.0312]],

                                [[0.0156, 0.0312, 0.0156],
                                [0.0312, 0.0625, 0.0312],
                                [0.0156, 0.0312, 0.0156]]]  ) / m.in_channels                    
            m.weight.data += w[None,None]
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)