import numpy as np

def get_mask(lz, rz, b, f, a):    
    _, h, w = lz.shape
    
    x = np.arange(w)
    y = np.arange(h)    
    xx, yy = np.meshgrid(x, y)    
    c = f * b * w / a
    ld = c / lz[0]   
    xr = xx - ld
    mask = xr >= 0    
    xr_mask_floor = xr[mask].astype(int)
    w1 = (xr[mask] - xr_mask_floor)
    w2 = 1 - w1    
    rz_matched = np.zeros_like(lz)
    rz_matched[0,yy[mask], xx[mask]] = w2 * rz[0,yy[mask],xr_mask_floor] + w1 * rz[0,yy[mask], np.clip(xr_mask_floor+1, None, w)]    
    diff = np.absolute(lz[0] - rz_matched[0])
    mask &= diff < 0.001
    return mask