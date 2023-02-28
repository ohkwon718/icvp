import os
import glob
import itertools
import numpy as np
import torch
from PIL import Image
from dataloader.stereodata import SparseStereo
from utils.pfm import readPFM

left_pattern_2015 = "image_2/**_10.png"
right_pattern_2015 = "image_3/**_10.png"
left_pattern_2012 = "colored_0/**_10.png"
right_pattern_2012 = "colored_1/**_10.png"



idx_kitti2012val = [3, 96, 92, 18, 74, 137, 51, 29, 175, 135, 7, 172, 102, 39, 140, 64, 189, 
            19, 139, 76, 122, 54, 77, 27, 142, 184, 191, 5, 44, 181, 84, 12, 170, 115, 
            182, 4, 10, 58, 48, 60, 108, 133, 149, 82, 59, 86, 83, 45]
            
idx_kitti2012training = [57, 117, 130, 174, 169, 73, 112, 167, 55, 85, 109, 158, 185, 153, 43, 30, 163, 159, 
                    132, 193, 148, 17, 127, 46, 145, 134, 165, 143, 125, 101, 136, 146, 38, 180, 124, 104, 
                    35, 37, 93, 94, 161, 47, 110, 160, 79, 13, 126, 67, 50, 187, 33, 90, 128, 8, 62, 121, 
                    80, 42, 11, 78, 41, 173, 119, 14, 190, 154, 177, 1, 65, 28, 186, 34, 164, 141, 56, 25, 
                    147, 91, 70, 40, 129, 20, 162, 99, 75, 61, 22, 183, 87, 131, 106, 66, 21, 144, 9, 81, 
                    95, 49, 68, 138, 116, 89, 168, 71, 176, 72, 150, 114, 103, 188, 26, 155, 120, 111, 36, 
                    100, 16, 113, 107, 97, 6, 2, 178, 63, 171, 31, 88, 69, 192, 15, 156, 23, 52, 105, 152, 24, 
                    157, 118, 179, 166, 151, 123, 98, 32, 53]

class KittiDataset(SparseStereo):
    def __init__(self, path_2015="../dataset/kitti2015", path_2012="../dataset/kitti2012",  
                datasets=["2015/training", "2012/training"], idx_select=None, datasize=None, **kwargs):        
        self.flist_img_left = []
        self.flist_img_right = []
        self.flist_disp_left = []
        self.flist_disp_right = []
        cnt_data = 0
        cnts_before = []
        sizes = []        
        for dataset in datasets:
            cnts_before.append(cnt_data)
            ver, group = dataset.split("/")
            if ver == "2015":
                flist_img_left = sorted(glob.glob(os.path.join(path_2015, group, left_pattern_2015)))
                flist_img_right = sorted(glob.glob(os.path.join(path_2015, group, right_pattern_2015)))
                flist_disp_left = sorted(glob.glob(os.path.join(path_2015, group, "disp_occ_0/**.png")))
                flist_disp_right = [None]*len(flist_disp_left)                    
            elif ver == "2012":
                flist_img_left = sorted(glob.glob(os.path.join(path_2012, group, left_pattern_2012)))
                flist_img_right = sorted(glob.glob(os.path.join(path_2012, group, right_pattern_2012)))
                flist_disp_left = sorted(glob.glob(os.path.join(path_2012, group, "disp_occ/**.png")))
                flist_disp_right = [None]*len(flist_disp_left)                    
            self.flist_img_left.append(flist_img_left)
            self.flist_img_right.append(flist_img_right)
            self.flist_disp_left.append(flist_disp_left)
            self.flist_disp_right.append(flist_disp_right)
            sizes.append(len(flist_disp_left))
            cnt_data += sizes[-1]
        self.flist_img_left = list(itertools.chain.from_iterable(self.flist_img_left))
        self.flist_img_right = list(itertools.chain.from_iterable(self.flist_img_right))
        self.flist_disp_left = list(itertools.chain.from_iterable(self.flist_disp_left))
        self.flist_disp_right = list(itertools.chain.from_iterable(self.flist_disp_right))
        if idx_select:
            idx_select_all = []
            for idx, cnt, size in zip(idx_select, cnts_before, sizes):
                if idx == None:
                    idx_select_all.append([i+cnt for i in range(size)])
                else:
                    idx_select_all.append([i+cnt for i in idx])
            self.idx_select = list(itertools.chain.from_iterable(idx_select_all))        

        self.reader_img = lambda x: np.asarray(Image.open(x))
        self.reader_disp = lambda x: np.asarray(Image.open(x))/256.

        super(KittiDataset, self).__init__(idx_select = self.idx_select, datasize = datasize, **kwargs)

    def get_mask(self, disp_left):
        mask_left = np.zeros(disp_left.shape, dtype=np.bool)
        mask_left[np.isfinite(disp_left)] = True
        mask_left[disp_left == 0.0] = False
        return mask_left


class KittiDxDyDataset(KittiDataset):
    def __init__(self, path_grad2015="../dataset/kt15_full_disparity_plane", path_grad2012="../dataset/kt12_full_disparity_plane",  
                datasets=["2015/training", "2012/training"], idx_select=None, datasize=None, **kwargs): 
        super(KittiDxDyDataset, self).__init__(
            datasets = datasets,
            idx_select = idx_select, 
            datasize = datasize, 
            **kwargs
        )
        gt_dx_left_pattern = "dense/dx/**_10.pfm"
        gt_dy_left_pattern = "dense/dy/**_10.pfm"

        self.flist_disp_dx_left = []
        self.flist_disp_dy_left = []                
        cnt_data = 0
        cnts_before = []
        sizes = []        
        for dataset in datasets:
            cnts_before.append(cnt_data)
            ver, group = dataset.split("/")
            if ver == "2015":
                flist_disp_dx_left = sorted(glob.glob(os.path.join(path_grad2015, "disp_occ_0", gt_dx_left_pattern)))
                flist_disp_dy_left = sorted(glob.glob(os.path.join(path_grad2015, "disp_occ_0", gt_dy_left_pattern)))                
            elif ver == "2012":
                flist_disp_dx_left = sorted(glob.glob(os.path.join(path_grad2012, "disp_occ", gt_dx_left_pattern)))
                flist_disp_dy_left = sorted(glob.glob(os.path.join(path_grad2012, "disp_occ", gt_dy_left_pattern)))
            self.flist_disp_dx_left.append(flist_disp_dx_left)
            self.flist_disp_dy_left.append(flist_disp_dy_left)            
            sizes.append(len(flist_disp_dx_left))
            cnt_data += sizes[-1]
        self.flist_disp_dx_left = list(itertools.chain.from_iterable(self.flist_disp_dx_left))
        self.flist_disp_dy_left = list(itertools.chain.from_iterable(self.flist_disp_dy_left))        
        self.flist_disp_right = [None]*len(self.flist_disp_left)
        
        if self.idx_select:
            self.flist_disp_dx_left = [self.flist_disp_dx_left[i] for i in self.idx_select]
            self.flist_disp_dy_left = [self.flist_disp_dy_left[i] for i in self.idx_select]
        self.prob_hflip = 0.

    def __getitem__(self, index):   
        img_left = self.reader_img(self.flist_img_left[index])[...,:3]
        img_right = self.reader_img(self.flist_img_right[index])[...,:3]
        disp_left = self.reader_disp(self.flist_disp_left[index])  
        disp_dx_left = readPFM(self.flist_disp_dx_left[index])        
        disp_dy_left = readPFM(self.flist_disp_dy_left[index])
        disp_left_stack = np.stack([disp_left, disp_dx_left, disp_dy_left], axis=-1)
        mask_left = self.get_mask(disp_left)
        if self.augmentator:            
            img_left, img_right, disp_left_stack, mask_left = self.augmentator(img_left, img_right, disp_left_stack, mask_left)
                
        disp_left = disp_left_stack[..., 0]
        disp_dx_left = disp_left_stack[..., 1]
        disp_dy_left = disp_left_stack[..., 2]

        img_left = np.ascontiguousarray(img_left)
        img_right = np.ascontiguousarray(img_right)
        disp_left = np.ascontiguousarray(disp_left)
        disp_dx_left = np.ascontiguousarray(disp_dx_left)
        disp_dy_left = np.ascontiguousarray(disp_dy_left)
        mask_left = np.ascontiguousarray(mask_left)
        
        img_left = torch.from_numpy(img_left.transpose((2,0,1))).float()
        img_right = torch.from_numpy(img_right.transpose((2,0,1))).float()
        disp_left = torch.from_numpy(disp_left[None]).float()
        disp_dx_left = torch.from_numpy(disp_dx_left[None]).float()
        disp_dy_left = torch.from_numpy(disp_dy_left[None]).float()
        mask_left = torch.from_numpy(mask_left[None]).bool()
        
        return img_left, img_right, disp_left, disp_dx_left, disp_dy_left, mask_left

