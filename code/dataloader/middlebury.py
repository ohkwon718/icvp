import os
import glob
import numpy as np
from PIL import Image
from dataloader.stereodata import SparseStereo
from utils.pfm import readPFM


left_pattern = "**/im0.png"
right_pattern = "**/im1.png"
gt_left_pattern = "**/disp0GT.pfm"
gt_right_pattern = "**/disp1GT.pfm"

class MiddDataset(SparseStereo):
    def __init__(self, path="../dataset/MiddEval3", datatype='trainingH', idx_select=None, datasize=None, **kwargs):
        self.flist_img_left = sorted(glob.glob(os.path.join(path, datatype, left_pattern)))
        self.flist_img_right = sorted(glob.glob(os.path.join(path, datatype, right_pattern)))
        self.flist_disp_left = sorted(glob.glob(os.path.join(path, datatype, gt_left_pattern)))        
        self.flist_disp_right = sorted(glob.glob(os.path.join(path, datatype, gt_right_pattern)))        

        self.reader_img = lambda x: np.asarray(Image.open(x))
        self.reader_disp = lambda x: readPFM(x)
        super(MiddDataset, self).__init__(idx_select = idx_select, datasize = datasize, **kwargs)

    def get_mask(self, disp_left):
        mask_left = np.zeros(disp_left.shape, dtype=np.bool)
        mask_left[np.isfinite(disp_left)] = True
        return mask_left
