import os
import sys
import glob
import shutil
import argparse

parser = argparse.ArgumentParser(description='Run training')
parser.add_argument('--path-target', type=str, default="code", help="target code path to run")
parser.add_argument('--path-run', type=str, default="", help="path to copy")
parser.add_argument('--path-resume', type=str, default="", help="path to resume")
parser.add_argument('--DELETE-OLD', default=False, action='store_true', help="remove if exists")
parser.add_argument('--path-group', type=str, default="./training", help="path to save together")
parser.add_argument('--resume-id', type=int, default=-1, help="saved training number")
parser.add_argument('--resume-name', type=str, default="", help="saved training name")
parser.add_argument('--trained-model', type=str, default="", help="load trained model")
parser.add_argument('--debug', default=False, action='store_true', help="turn on debug mode")
parser.add_argument('--lr', type=float, default=0, help="set the learning rate manually")
args = parser.parse_args()


if args.path_resume != "":
    path_run = args.path_resume
    resume = True    
elif args.path_run == "":    
    if args.resume_name != "":
        dir_training = args.resume_name
        resume = True
    elif args.resume_id == -1:
        runs = sorted(glob.glob(os.path.join(args.path_group, 'training-*')))
        id_run = int(runs[-1].split('-')[-1]) + 1 if runs else 0
        dir_training = 'training-{:03d}'.format(id_run)
        resume= False
    else:
        dir_training = 'training-{:03d}'.format(args.resume_id)
        resume = True

    path_run = os.path.join(args.path_group, dir_training)
else:
    assert os.path.abspath(args.path_run) != os.path.abspath("code")
    path_run = args.path_run    
    resume = False
    if os.path.exists(path_run):
        assert args.DELETE_OLD                
        shutil.rmtree(path_run)

print(path_run)
if not resume:
    shutil.copytree(args.path_target, path_run)
    
sys.path.append(path_run)    

import main

main.run_train(args, path_run, resume)

