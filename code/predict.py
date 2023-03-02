from math import ceil
import os
import time
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from matplotlib import cm
from PIL import Image

import config
from utils.pfm import save_pfm
from model import __models__
import utils.visualize as vis


def load_impair(fname_left, fname_right):
    left_ = np.asarray(Image.open(fname_left))            
    right_ = np.asarray(Image.open(fname_right))
    if len(left_.shape)==2: left_ = np.repeat(left_[...,None], 3, axis=2)        
    if len(right_.shape)==2: right_ = np.repeat(right_[...,None], 3, axis=2)
    left_ = torch.from_numpy(left_[...,:3].transpose((2,0,1))[None].astype(np.float32))
    right_ = torch.from_numpy(right_[...,:3].transpose((2,0,1))[None].astype(np.float32))
    h = min(left_.shape[-2], right_.shape[-2])
    w = min(left_.shape[-1], right_.shape[-1])
    left_ = left_[...,:h,:w]
    right_ = right_[...,:h,:w]
    return left_, right_


def run(args, path_run):    
    path_result = os.path.join(path_run, args.savedir)
    os.makedirs(path_result, exist_ok=True)
    runs = sorted(glob.glob(os.path.join(path_result, 'result-*.npz')))
    id_run = int(os.path.splitext(runs[-1])[0].split('-')[-1]) + 1 if runs else 0
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.model != '':
        model = __models__[args.model].to(device)
    else:
        model = config.model.to(device)


    print('Total Model Params = {:.2f}MB'.format(sum(np.prod(v.size()) for name, v in model.named_parameters() if "aux" not in name)/1e6))

    fullpath_model = os.path.join(path_run, args.trained_model)
    state = torch.load(fullpath_model)
    model.load_state_dict(state['model_state_dict'])
    epoch_start = state['epoch']
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("load : {}, epoch : {}".format(fullpath_model, epoch_start))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    
    model.eval()
    with torch.no_grad():   
        fnames_left = sorted(glob.glob(os.path.join(args.rootpath, args.img_left)))
        fnames_right = sorted(glob.glob(os.path.join(args.rootpath, args.img_right)))
        
        for fname_left, fname_right in zip(fnames_left, fnames_right):        
            left_, right_ = load_impair(fname_left, fname_right)
            b, _, h_in, w_in = left_.shape
            if args.shift_disp:
                empty = torch.zeros_like(right_)
                empty[..., args.shift_disp:] = right_[...,:-args.shift_disp]

            if args.crop_center:                
                empty = torch.full((b,h_in,w_in), -1)
                y_center, x_center = h_in//2, w_in//2                
                y0, x0 = y_center - args.crop_center[0]//2, x_center - args.crop_center[1]//2
                y1, x1 = y0 + args.crop_center[0], x0 + args.crop_center[1]
                x0 -= args.max_disp
                left_ = left_[...,y0:y1,x0:x1]
                right_ = right_[...,y0:y1,x0:x1]
            
            _, _, h_crop, w_crop = left_.shape
            if args.resize:
                size_resize = args.resize
            elif args.scale_up:
                size_resize = (int(h_crop*args.scale_up),int(w_crop*args.scale_up))
            elif args.scale_down:
                size_resize = int(h_crop/args.scale_down),int(w_crop/args.scale_down)
            else:
                size_resize = None            

            left, right = left_.to(device), right_.to(device)      
            start = time.time()
            if size_resize:
                left = F.interpolate(left, size=size_resize, mode='bilinear', align_corners=True) 
                right = F.interpolate(right, size=size_resize, mode='bilinear', align_corners=True)
            out, _ = model(left, right, ceil(args.max_disp*left.shape[-1]/w_crop))                                
            if size_resize and (args.recover_scale or args.crop_center):
                out = w_crop/out.shape[-1] * F.interpolate(out, size=[h_crop,w_crop], mode='bilinear', align_corners=True)            
            torch.cuda.synchronize()
            end = time.time()
            
            elapsed = end-start                
            out_ = out[:,0].cpu()
            if args.shift_disp:
                out_ += args.shift_disp            
            if args.crop_center:            
                empty[...,y0:y1,x1-args.crop_center[1]:x1] = out_[...,args.max_disp:]
                out_ = empty

            if args.keep_tree or args.benchmark == 'kitti':
                relpath = os.path.relpath(fname_left, args.rootpath)
                fullpath_result = os.path.join(path_result, os.path.splitext(relpath)[0])
                os.makedirs(os.path.dirname(fullpath_result), exist_ok=True)
            elif args.benchmark == 'eth3d':                
                fullpath_result = os.path.join(path_result, 'low_res_two_view')
                os.makedirs(fullpath_result, exist_ok = True)		
            else:
                fullpath_result = os.path.join(path_result, 'result-{:03d}'.format(id_run))
                id_run += 1
                
            print("save to ", fullpath_result)            
            print("time {:.3f}, GPU usages: {:}Gb".format(elapsed, torch.cuda.max_memory_reserved()/1e9))
            if args.benchmark == 'kitti':                                   
                Image.fromarray((out_[0].numpy() * 256).astype(np.uint16)).save(fullpath_result + '.png')                                        
            if args.benchmark == 'eth3d':
                assert os.path.basename(os.path.split(fname_left)[0]) == os.path.basename(os.path.split(fname_right)[0])
                fname = os.path.join(fullpath_result, os.path.basename(os.path.split(fname_left)[0]))
                save_pfm(fname+'.pfm', out_[0].numpy())	
                with open(fname+'.txt', 'w') as timing_file:
                    timing_file.write('runtime ' + str(elapsed))                                    
            else:
                if not args.not_save_img:
                    img_out = vis.apply_colomap(out_[0].detach(), range=(args.shift_disp, args.max_disp+args.shift_disp), colormap=args.cmap, is_torch=True)                    
                    save_image(img_out, fullpath_result + '.png')
                if not args.not_save_npz:
                    data = {'disparity' : out_[0].numpy()}
                    np.savez(fullpath_result + '.npz', **data)
                