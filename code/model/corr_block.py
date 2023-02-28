import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class corr_block(nn.Module):    
    def __init__(self):
        super(corr_block, self).__init__()
      
    def forward(self, lf, rf, d_max = 64):
        b,c,h,w = lf.shape
        disp_block = torch.zeros(b, 1, d_max, h, w, device=lf.device)
        for d in range(d_max):            
            disp_block[:, :, d, :, d:] = (lf[..., d:] * rf[..., :w-d]).sum(1, keepdim=True)            
        return disp_block / np.sqrt(c)


class gwc_block(nn.Module):    
    def __init__(self, n_head = 16):
        super(gwc_block, self).__init__()
        self.n_head = n_head
      
    def forward(self, lf, rf, d_max = 64):
        b,c,h,w = lf.shape
        n_channels = c // self.n_head
        disp_block = torch.zeros(b, self.n_head, d_max, h, w, device=lf.device)
        for d in range(d_max):            
            disp_block[:, :, d, :, d:] = (lf[..., d:].view(b, self.n_head, n_channels, h, -1) * rf[..., :w-d].view(b, self.n_head, n_channels, h, -1)).sum(2)            
        return disp_block / np.sqrt(n_channels)

