import os
import sys
import time
import glob
import json
import argparse
import importlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
import pandas as pd
from PIL import Image


parser = argparse.ArgumentParser(description='Run training')
parser.add_argument('--path-run', type=str, default="", help="path to run")
parser.add_argument('--path-group', type=str, default="./training", help="path to save together")
parser.add_argument('--training-id', type=int, default=-1,  help="saved training number")
parser.add_argument('--training-name', type=str, default="", help="saved training name")
parser.add_argument('--model', type=str, default="", help="Do you want to manually choose the model here?")
parser.add_argument('--rootpath', type=str, default="", help="path to count directory tree")
parser.add_argument('--img-left', type=str, required=True, help="left image file (from rootpath if it exists)")
parser.add_argument('--img-right', type=str, required=True, help="right image file (from rootpath if it exists)")
parser.add_argument('--config', type=str, default="",  help="load configuration")
parser.add_argument('--global-path', default=False, action='store_true', help="import/export on global path")
parser.add_argument('--trained-model', type=str, default="latest.pt",  help="saved file name")
parser.add_argument('--savedir', type=str, default="results",  help="directory to save")
parser.add_argument('--keep-tree', default=False, action='store_true', help="keep the directory structure of input")
parser.add_argument('--batch-size', type=int, default=-1,  help="batch size")
parser.add_argument('--resize', nargs='+', type=int, help="resize input images")
parser.add_argument('--max-disp', type=int, default=192, help="maximum disparity")
parser.add_argument('--shift-disp', type=int, default=0, help="expected minimum disparity")
parser.add_argument('--scale-up', type=int, help="rescale input images")
parser.add_argument('--scale-down', type=int, help="rescale input images")
parser.add_argument('--crop-center', nargs='+', type=int, help="cropping size for original image")
parser.add_argument('--recover-scale', default=False, action='store_true', help="Do you want recover the original scale?")
parser.add_argument('--cycle-test-ths', type=float, default=0.0,  help="Threshold for cycle consistency test. Default 0.0 skips the cycle test")
parser.add_argument('--not-save-img', default=False, action='store_true', help="not save imgs")
parser.add_argument('--not-save-npz', default=False, action='store_true', help="save npz")
parser.add_argument('--benchmark', type=str, default="", help="for submission")
parser.add_argument('--cmap', type=str, default="inferno", help="color map for result images")

args = parser.parse_args()

if args.config != '':
    with open(args.config, 'r') as f:
        config = json.load(f)
    for key, value in config.items():        
        if vars(args)[key] == parser.get_default(key):
            setattr(args, key, value)
        

assert((args.resize!=None)+(args.scale_up!=None)+(args.scale_down!=None) <= 1)
if args.path_run == "":
    if args.training_name != "":    
        dir_training = args.training_name
    elif args.training_id == -1:
        runs = sorted(glob.glob(os.path.join(args.path_group, 'training-*')))
        if runs:
            id_run = int(runs[-1].split('-')[-1])  
        dir_training = 'training-{:03d}'.format(id_run)
    else:
        dir_training = 'training-{:03d}'.format(args.training_id)


    path_run = os.path.join(args.path_group, dir_training)
else:
    path_run = args.path_run

print(path_run)
assert(os.path.exists(path_run))
sys.path.append(path_run)    

import predict
predict.run(args, "./" if args.global_path else path_run)

