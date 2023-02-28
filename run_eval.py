import os
import sys
import glob
import argparse


parser = argparse.ArgumentParser(description='Run training')
parser.add_argument('--path-run', type=str, default="", help="path to run")
parser.add_argument('--path-group', type=str, default="./training", help="path to save together")
parser.add_argument('--training-id', type=int, default=-1, help="saved training number")
parser.add_argument('--training-name', type=str, default="", help="saved training name")
parser.add_argument('--filename', type=str, default="latest.pt", help="saved file name")
parser.add_argument('--savedir', type=str, default="eval", help="directory to save")
parser.add_argument('--d-max', type=int, default=192, help="maximum disparity")
parser.add_argument('--batch-size', type=int, default=-1, help="batch size")
parser.add_argument('--score-only', default=False, action='store_true', help="without saving any images")
parser.add_argument('--result-only', default=False, action='store_true', help="save imgs only with result")
parser.add_argument('--cmap', type=str, default="turbo", help="color map for result images")
args = parser.parse_args()


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

import eval
eval.run(args, path_run)

