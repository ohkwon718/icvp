import os
import glob
import itertools
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from skimage.transform import resize
from utils.pfm import readPFM
from dataloader.stereodata import DenseStereo

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


fmts = {"FlyingThings":"{type:}/{dataset:}/**/**/{side:}/**",
    "Monkaa":"{type:}/{dataset:}/**/{side:}/**",
    "Driving":"{type:}/{dataset:}/**/**/**/{side:}/**"}

class SceneFlowDataset(DenseStereo):
    def __init__(self, path="../dataset/SceneFlow", dataset="train", parts=["FlyingThings", "Monkaa", "Driving"], idx_select=None, datasize=None, **kwargs):
        dataset = dataset.upper()        

        patterns_img_left = [os.path.join(path,fmts[part].format(type='frames_finalpass', dataset=dataset, side='left')) for part in parts]
        patterns_img_right = [os.path.join(path,fmts[part].format(type='frames_finalpass', dataset=dataset, side='right')) for part in parts]
        patterns_disp_left = [os.path.join(path,fmts[part].format(type='disparity', dataset=dataset, side='left')) for part in parts]
        patterns_disp_right = [os.path.join(path,fmts[part].format(type='disparity', dataset=dataset, side='right')) for part in parts]

        self.flist_img_left = sorted(list(itertools.chain.from_iterable([glob.glob(pattern) for pattern in patterns_img_left])))
        self.flist_img_right = sorted(list(itertools.chain.from_iterable([glob.glob(pattern) for pattern in patterns_img_right])))
        self.flist_disp_left = sorted(list(itertools.chain.from_iterable([glob.glob(pattern) for pattern in patterns_disp_left])))
        self.flist_disp_right = sorted(list(itertools.chain.from_iterable([glob.glob(pattern) for pattern in patterns_disp_right])))
        
        self.reader_img = lambda x: np.asarray(Image.open(x))
        self.reader_disp = lambda x: readPFM(x)
        
        super(SceneFlowDataset, self).__init__(
            idx_select = idx_select, 
            datasize = datasize, 
            **kwargs
        )


class ExtendedSceneFlowDataset(SceneFlowDataset):
    def __init__(self, path="../dataset/SceneFlow", dataset="train", idx_select=None, datasize=None, extras=[], **kwargs):
        super(ExtendedSceneFlowDataset, self).__init__(
            dataset = dataset,
            idx_select = idx_select, 
            datasize = datasize, 
            **kwargs
        )
        dataset = dataset.upper()        
        self.flists_extra = [sorted(list(itertools.chain.from_iterable([glob.glob(os.path.join(path, pattern.format(dataset=dataset))) for pattern in extra["fmts"]]))) for extra in extras]
        

    def __getitem__(self, index):   
        img_left = self.reader_img(self.flist_img_left[index])[...,:3]
        img_right = self.reader_img(self.flist_img_right[index])[...,:3]
        disp_left = self.reader_disp(self.flist_disp_left[index])
        mask_left = np.ones(disp_left.shape , dtype=np.bool)

        extras = [ 
            resize(np.load(flist[index])["disparity"][0, ..., 0], (disp_left.shape[-2], disp_left.shape[-1])) 
            for flist in self.flists_extra
        ] 
        disp_left_stack = np.stack([disp_left] + extras, axis=-1)
        if self.augmentator:
            img_left, img_right, disp_left_stack, mask_left = self.augmentator(img_left, img_right, disp_left_stack, mask_left)
        disp_left = disp_left_stack[...,0]
        extras = disp_left_stack[...,1:]

        img_left = np.ascontiguousarray(img_left)
        img_right = np.ascontiguousarray(img_right)
        disp_left = np.ascontiguousarray(disp_left)
        extras = np.ascontiguousarray(extras)
        mask_left = np.ascontiguousarray(mask_left)

        img_left = torch.from_numpy(img_left.transpose((2,0,1))).float()
        img_right = torch.from_numpy(img_right.transpose((2,0,1))).float()
        disp_left = torch.from_numpy(disp_left[None]).float()        
        extras = torch.from_numpy(extras.transpose((2,0,1))).float()
        mask_left = torch.from_numpy(mask_left[None]).bool()
        
        return img_left, img_right, disp_left, mask_left, extras
        


