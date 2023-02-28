import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import pandas as pd

import config
import utils.visualize as vis


def run(args, path_run):    
    path_eval = os.path.join(path_run, args.savedir)
    print("save to ",path_eval)
    os.makedirs(path_eval, exist_ok=True)

    fullfmt_img = os.path.join(path_eval, "{:06}.png")
    fullpath_log = os.path.join(path_eval, "log")
    fullpath_result = os.path.join(path_eval, "result.txt")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = config.model.to(device)
    print('Total Model Params = {:.2f}MB'.format(sum(np.prod(v.size()) for name, v in model.named_parameters() if "aux" not in name)/1e6))

    evalset = config.evalset
    batch_size = config.eval_batch_size if args.batch_size == -1 else args.batch_size
    loader = DataLoader(evalset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory = False)

    fullpath_model = os.path.join(path_run, args.filename)
    state = torch.load(fullpath_model)
    model.load_state_dict(state['model_state_dict'])
    epoch_start = state['epoch']
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("load : {}, epoch : {}, batch size :{}".format(fullpath_model, epoch_start, batch_size))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


    df_log = pd.DataFrame(columns=['id', 'err']+['bad {:}'.format(th) for th in config.eval_bad_ths])
    df_log = df_log.set_index('id')

    model.eval()
    total_err = 0
    total_bads = {th:0 for th in config.eval_bad_ths}
    loop_batch = tqdm(loader)
    num_samples = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loop_batch):
            left_, right_, disparity_, mask_ = batch
            start = time.time()        
            left, right, disparity = left_.to(device), right_.to(device), disparity_.to(device)
            out = model(left, right, d_max = config.d_max_eval_model)
            torch.cuda.synchronize()    
            end = time.time()
            elapsed = end-start
            
            if isinstance(out, tuple):
                out = out[0]        
            err, bads, mask = eval(out, disparity, max_disparity=config.d_max_eval_eval, ths=config.eval_bad_ths)
            total_err += sum(err)                
            for th in config.eval_bad_ths: total_bads[th] += sum(bads[th])
            num_samples += len(disparity)
            loop_batch.set_description("Error {:.6f}, time {:.3f}, GPU usages: {:}Gb".format(sum(err)/len(disparity), elapsed, torch.cuda.max_memory_reserved()/1e9))
            for i in range(len(out)):
                df_log.loc[batch_idx * batch_size + i] = { 'err': err[i] }
                df_log.loc[batch_idx * batch_size + i].update({'bad {:}'.format(th):bad[i] for th, bad in bads.items()})          
                if args.score_only:
                    continue
                if args.result_only:                
                    img_out = vis.apply_colomap(out[i, 0], range=(0, args.d_max), colormap=args.cmap, is_torch=True)    
                    save_image(img_out, fullfmt_img.format(batch_idx * batch_size + i))                
                else:
                    img_left = left_[i] / 255
                    img_right = right_[i] / 255        
                    img_gt = vis.apply_colomap(disparity_[i, 0], range=(0, args.d_max), colormap=args.cmap, is_torch=True)
                    img_out = vis.apply_colomap(out[i, 0], range=(0, args.d_max), colormap=args.cmap, is_torch=True)
                    vis.save_grid([img_left, img_right, img_gt, img_out], fp = fullfmt_img.format(batch_idx * batch_size + i), nrow = 1)
    total_err /= num_samples
    for th in ths: total_bads[th] /= num_samples
    print("Error : {:.3f}".format(total_err))
    for th in ths: print("Bad {:} : {:.3f}".format(th, total_bads[th]))
    df_log.to_csv(fullpath_log + '.txt')
    with open(fullpath_result, "w") as f:
        f.write("Error : {:.3f}\n".format(total_err))
        for th in ths: f.write("Bad {:} : {:.3f}\n".format(th, total_bads[th]))



def get_bad(diff, th, mask):
    bad = mask.clone()
    bad[diff <= th] = False
    bad = [100.0 * bad[i].sum().item() / mask[i].sum().item() if mask[i].sum() > 0 else 0.0 for i in range(len(diff))]
    return bad

def eval(res, gt, max_disparity = 192, ths = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0]):
    mask = torch.logical_and(gt > 0, gt < max_disparity)
    diff = (res - gt).abs()
    err = [diff[i,mask[i]].mean().item() if mask[i].sum() > 0 else 0.0 for i in range(len(diff))]    
    bads = {th:get_bad(diff, th, mask)  for th in ths}
    return err, bads, mask