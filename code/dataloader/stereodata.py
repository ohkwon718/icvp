import numpy as np
import torch
from torch.utils.data import Dataset
from utils.aug import SpatialTransform, PureCropper



class Stereo(Dataset):
    def __init__(self, crop_size=None, idx_select=None, datasize=None, prob_vflip=0, prob_hflip=0, augmentators=[], **kwargs):
        if idx_select:
            self.flist_img_left = [self.flist_img_left[i] for i in idx_select]
            self.flist_img_right = [self.flist_img_right[i] for i in idx_select]
            self.flist_disp_left = [self.flist_disp_left[i] for i in idx_select]
            self.flist_disp_right = [self.flist_disp_right[i] for i in idx_select]
        self.datasize = datasize
        self.prob_vflip = prob_vflip
        self.prob_hflip = prob_hflip

        # self.
        if kwargs:      
            self.cropper = SpatialTransform(crop_size = crop_size, isSparse = self.isSparse, **kwargs)
        elif crop_size: 
            self.cropper = PureCropper(crop_size = crop_size)
        else:           
            self.cropper = None
        
        self.augmentators = augmentators

        if self.isSparse == False:
            self.get_mask = lambda disp_left: np.ones(disp_left.shape , dtype=np.bool)


    def __getitem__(self, index):   
        if np.random.rand() >= self.prob_hflip or not self.flist_disp_right[index]:
            img_left = self.reader_img(self.flist_img_left[index])[...,:3]
            img_right = self.reader_img(self.flist_img_right[index])[...,:3]
            disp_left = self.reader_disp(self.flist_disp_left[index])
        else:            
            img_left = self.reader_img(self.flist_img_right[index])[:,::-1,:3]
            img_right = self.reader_img(self.flist_img_left[index])[:,::-1,:3]
            disp_left = self.reader_disp(self.flist_disp_right[index])[:,::-1]            


        if np.random.rand() < self.prob_vflip:
            img_left = img_left[::-1]
            img_right = img_right[::-1]
            disp_left = disp_left[::-1]                    

        mask_left = self.get_mask(disp_left)
        
        if self.augmentators:            
            for augmentator in self.augmentators:
                img_left, img_right, disp_left, mask_left = augmentator(img_left, img_right, disp_left, mask_left)
        if self.cropper:
            img_left, img_right, disp_left, mask_left = self.cropper(img_left, img_right, disp_left, mask_left)


        img_left = np.ascontiguousarray(img_left)
        img_right = np.ascontiguousarray(img_right)
        disp_left = np.ascontiguousarray(disp_left)
        mask_left = np.ascontiguousarray(mask_left)
        
        img_left = torch.from_numpy(img_left.transpose((2,0,1))).float()
        img_right = torch.from_numpy(img_right.transpose((2,0,1))).float()
        disp_left = torch.from_numpy(disp_left[None]).float()
        mask_left = torch.from_numpy(mask_left[None]).bool()
        
        return img_left, img_right, disp_left, mask_left

    def __len__(self):
        if self.datasize:
            return self.datasize
        else:
            return len(self.flist_img_left)


class DenseStereo(Stereo):    
    def __init__(self, **kwargs):
        self.isSparse = False
        super(DenseStereo, self).__init__(**kwargs)    

    def get_mask(self, disp_left):
        return np.ones(disp_left.shape , dtype=np.bool)

    
class SparseStereo(Stereo):    
    def __init__(self, **kwargs):
        self.isSparse = True
        super(SparseStereo, self).__init__(**kwargs)

