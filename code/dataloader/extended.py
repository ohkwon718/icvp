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


def ExtendedStereo(base, extras=[], **kwargs):
    class core(base):
        def __init__(self, extras=[], **kwargs):
            super(core, self).__init__(**kwargs)            
            self.extras = extras
            self.flists = [
                sorted(list(itertools.chain.from_iterable([glob.glob(pattern) for pattern in extra["patterns"]])))
                for extra in self.extras
            ]
        def __getitem__(self, index):    
            stereo = super(core, self).__getitem__(index)
            return stereo + tuple(extra["reader"](flist[index]) for extra, flist in zip(self.extras, self.flists))                
    return core(extras, **kwargs)

