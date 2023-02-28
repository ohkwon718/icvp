import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_smooth_l1(out, gt, auxiliary, mask, d_max = 192):    
    loss = F.smooth_l1_loss(out[mask], gt[mask], reduction='mean') if torch.any(mask) else torch.tensor(0.)    
    return loss

