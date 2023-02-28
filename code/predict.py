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
        imgs_left = sorted(glob.glob(os.path.join(args.rootpath, args.img_left)))
        imgs_right = sorted(glob.glob(os.path.join(args.rootpath, args.img_right)))
        
        for img_left, img_right in zip(imgs_left, imgs_right):        
            left_ = np.asarray(Image.open(img_left))
            if len(left_.shape)==2: left_ = np.repeat(left_[...,None], 3, axis=2)        
            right_ = np.asarray(Image.open(img_right))
            if len(right_.shape)==2: right_ = np.repeat(right_[...,None], 3, axis=2)
            left_ = torch.from_numpy(left_[...,:3].transpose((2,0,1))[None].astype(np.float32))
            right_ = torch.from_numpy(right_[...,:3].transpose((2,0,1))[None].astype(np.float32))
            h = min(left_.shape[-2], right_.shape[-2])
            w = min(left_.shape[-1], right_.shape[-1])
            left_ = left_[...,:h,:w]
            right_ = right_[...,:h,:w]
            if args.resize:
                _, _, h, w = left_.shape        
                left, right = left_.to(device), right_.to(device)      
                start = time.time()                                    
                left = F.interpolate(left, size=args.resize, mode='bilinear', align_corners=True) 
                right = F.interpolate(right, size=args.resize, mode='bilinear', align_corners=True)                      
                out, _ = model(left, right, ceil(args.max_disp*left.shape[-1]/w))                                
                if args.recover_scale:
                    out = w/out.shape[-1] * F.interpolate(out, size=[h,w], mode='bilinear', align_corners=True)        
                torch.cuda.synchronize()
                end = time.time()
                elapsed = end-start

            elif args.scale_up:
                _, _, h, w = left_.shape        
                left, right = left_.to(device), right_.to(device)      
                start = time.time()                 
                left = F.interpolate(left, size=[int(h*args.scale_up),int(w*args.scale_up)], mode='bilinear', align_corners=True) 
                right = F.interpolate(right, size=[int(h*args.scale_up),int(w*args.scale_up)], mode='bilinear', align_corners=True)                
                out, _ = model(left, right, ceil(args.max_disp*args.scale_up))
                if args.recover_scale:         
                    out = w/out.shape[-1] * F.interpolate(out, size=[h,w], mode='bilinear', align_corners=True)        
                torch.cuda.synchronize()    
                end = time.time()
                elapsed = end-start                
            elif args.scale_down:
                _, _, h, w = left_.shape        
                left, right = left_.to(device), right_.to(device)      
                start = time.time()                 
                left = F.interpolate(left, size=[int(h/args.scale_down),int(w/args.scale_down)], mode='bilinear', align_corners=True) 
                right = F.interpolate(right, size=[int(h/args.scale_down),int(w/args.scale_down)], mode='bilinear', align_corners=True)
                out, _ = model(left, right, ceil(args.max_disp/args.scale_up)) 
                if args.recover_scale:
                    out = w/out.shape[-1] * F.interpolate(out, size=[h,w], mode='bilinear', align_corners=True)
                torch.cuda.synchronize()    
                end = time.time()
                elapsed = end-start                   
            else:
                left, right = left_.to(device), right_.to(device)      
                start = time.time()                 
                out, _ = model(left, right, args.max_disp) 
                torch.cuda.synchronize()    
                end = time.time()
                elapsed = end-start   
                

            if args.keep_tree or args.benchmark == 'kitti':
                relpath = os.path.relpath(img_left, args.rootpath)
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
                Image.fromarray((out[0, 0].cpu().numpy() * 256).astype(np.uint16)).save(fullpath_result + '.png')                                        
            if args.benchmark == 'eth3d':
                assert os.path.basename(os.path.split(img_left)[0]) == os.path.basename(os.path.split(img_right)[0])
                fname = os.path.join(fullpath_result, os.path.basename(os.path.split(img_left)[0]))
                save_pfm(fname+'.pfm', out[0, 0].cpu().numpy())	
                with open(fname+'.txt', 'w') as timing_file:
                    timing_file.write('runtime ' + str(elapsed))                
                

            else:
                if not args.not_save_img:
                    img_out = torch.from_numpy(cm.turbo(out[0, 0].cpu().detach()/args.max_disp)[...,:3]).permute(2,0,1)
                    save_image(img_out, fullpath_result + '.png')
                if not args.not_save_npz:
                    data = {'disparity' : out[0, 0].cpu().numpy()}
                    np.savez(fullpath_result + '.npz', **data)
                