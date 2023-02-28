import torch
from matplotlib import cm
from torchvision.utils import make_grid, save_image


def apply_colomap(value, range=(0, 192), colormap='turbo', is_torch=True):
    out = cm.get_cmap(colormap)((value.cpu().detach() - range[0])/(range[1]-range[0]))[...,:3]
    if is_torch:
        out = torch.from_numpy(out).permute(2,0,1)
    return out

def save_grid(imgs:list, fp: str, **kwargs):
    imgs_grid = []
    for img in imgs:
        img = img.cpu().detach()
        if img.shape[0] == 1:
            img = img.repeat(3,1,1)
        imgs_grid.append(img)
    grid = make_grid(imgs_grid, **kwargs)
    save_image(grid, fp)
    
