import os
import glob
import numpy as np
import cv2
from PIL import Image
from dataloader.stereodata import DenseStereo
from utils.pfm import readPFM


left_pattern = "**/*_left.jpg"
right_pattern = "**/*_right.jpg"
gt_left_pattern = "**/*_left.disp.png"
gt_right_pattern = "**/*_right.disp.png"

class CREStereoDataset(DenseStereo):
    def __init__(self, path="../dataset/crestereo", idx_select=None, datasize=None, **kwargs):
        
        
        self.imgs = glob.glob(os.path.join(path, "**/*_left.jpg"), recursive=True)
               
        self.flist_img_left = sorted(glob.glob(os.path.join(path, left_pattern)))
        self.flist_img_right = sorted(glob.glob(os.path.join(path, right_pattern)))
        self.flist_disp_left = sorted(glob.glob(os.path.join(path, gt_left_pattern)))        
        self.flist_disp_right = sorted(glob.glob(os.path.join(path, gt_right_pattern)))        

        self.reader_img = lambda x: np.asarray(Image.open(x))
        super(CREStereoDataset, self).__init__(idx_select = idx_select, datasize = datasize, **kwargs)


    def reader_disp(self, path):
        disp = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return disp.astype(np.float32) / 32
    


