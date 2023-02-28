import os
import glob
import numpy as np
import cv2
from PIL import Image
from dataloader.stereodata import DenseStereo


left_pattern = "**/**/image_left/*.png"
right_pattern = "**/**/image_right/*.png"
gt_left_pattern = "**/**/disparity/*.png"


class ORStereoDataset(DenseStereo):
    def __init__(self, path="../dataset/IROS_ORStereo_4K", idx_select=None, datasize=None, **kwargs):
        self.flist_img_left = sorted(glob.glob(os.path.join(path, left_pattern)))
        self.flist_img_right = sorted(glob.glob(os.path.join(path, right_pattern)))
        self.flist_disp_left = sorted(glob.glob(os.path.join(path, gt_left_pattern)))        
        self.flist_disp_right = [None] * len(self.flist_disp_left)
        
        self.reader_img = lambda x: np.asarray(Image.open(x))
        self.reader_disp = lambda x: read_lu1_disparity(x)
        
        if "prob_hflip" in kwargs:
            kwargs["prob_hflip"] = 0.0
        
        super(ORStereoDataset, self).__init__(idx_select = idx_select, datasize = datasize, **kwargs)

    def get_mask(self, disp_left):
        mask_left = np.zeros(disp_left.shape, dtype=np.bool)
        mask_left[np.isfinite(disp_left)] = True
        return mask_left





# https://github.com/castacks/iros_2021_orstereo
# coding=utf-8

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-03-13

def read_lu1_disparity(fn):
    '''Read the disparity file with a suffix _lu1 in the file name.
    Arguments:
    fn (string): The file name with full path.

    Returns:
    A single-channel disparity with dtype == np.float32.
    '''
    assert( os.path.isfile(fn) ), \
        f'{fn} does not exist. '
    
    # Read the raw data from a 4-channel PNG file.
    d = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    assert(d.dtype == np.uint8)
    assert(d.shape[2] == 4), \
        f'{fn} has a shape of {d.shape}, not suitable for disparity/depth conversion. '

    # Conver the data to float32.
    d = d.view('<f4')

    # Squeeze the redundant dimension.
    d = np.squeeze(d, axis=-1)

    return d

def read_occlusion(fn):
    '''Read an occlusion file.
    Arguments:
    fn (string): The input filename with full path.

    Returns:
    A single-channel boolean image. True pixels are occlusions.
    '''
    assert( os.path.isfile(fn) ), \
        f'{fn} does not exist. '

    # Read the raw image.
    m = cv2.imread(fn, cv2.IMREAD_UNCHANGED)

    # The pixels with a value other than 255
    # are considered as occlusion (including 
    # out-of-view).
    return m != 255