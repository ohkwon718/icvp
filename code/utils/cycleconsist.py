import torch
import torch.nn.functional as F


def warp_disparity(x, disp):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W, device=x.device).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device=x.device).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    vgrid = torch.cat((xx, yy), 1).float()

    # vgrid = Variable(grid)
    vgrid[:,:1,:,:] = vgrid[:,:1,:,:] - disp

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    
    vgrid = vgrid.permute(0, 2, 3, 1)    
    
    # output = F.grid_sample(x, vgrid, align_corners=True) # maybe correct but slow
    output = F.grid_sample(x, vgrid, align_corners=False)    
    
    return output

    
def cycle_consistency_mask(disp_left, disp_right, ths = 0.5):    
    disp_right_warped = warp_disparity(disp_left[:,None], disp_right[:,None])
    img_diff = torch.abs(disp_left - disp_right_warped)    
    mask = img_diff[:,0] < ths

    return mask
