import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time



# loss(out, gt, auxiliary, mask_, d_max = d_max)

def get_epe(diff, mask):
    err = [diff[i, mask[i]].mean().item() if mask[i].sum() > 0 else 0.0 for i in range(len(diff))]
    return err


def get_bad(diff, th, mask):
    bad = mask.clone()
    bad[diff <= th] = False
    bad = [100.0 * bad[i].sum().item() / mask[i].sum().item() if mask[i].sum() > 0 else 0.0 for i in range(len(diff))]
    return bad


def get_D1(diff, gt, mask):
    bad = mask.clone()
    bad[diff <= 3] = False
    bad[diff/gt.abs() <= 0.05] = False
    bad = [100.0 * bad[i].sum().item() / mask[i].sum().item() if mask[i].sum() > 0 else 0.0 for i in range(len(diff))]
    return bad
