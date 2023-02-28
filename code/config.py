import numpy as np
from torch.nn.modules import activation
from dataloader.sceneflow import *
from dataloader.kitti2015 import *
from dataloader.middlebury import MiddDataset
from dataloader.eth3d import Eth3dDataset
from dataloader.crestereo import CREStereoDataset
from dataloader.orstereo import ORStereoDataset

from model.unet import *
from model.icvp import *
from model.loss import *
from utils.aug import *
import torch.optim as optim
import torch_optimizer as optim2
import cv2

use_amp = True
log_period = 1000
imglog_period = 1
clip_grad_max_norm = 0.1
test_bad_ths = [0.5, 1.0, 2.0, 4.0]
test_D1 = True
eval_bad_ths = [0.5, 1.0, 2.0, 4.0]
eval_D1 = True



reset_optimizer = False
epoch_scheduler = 0
scheduler = None


aug_params = {
    "yjitter":2,
    "prob_hflip":0.5,
    "prob_vflip":0.5,
}
augs = [ColorTransform(hue=0.0/3.14, asymmetric_prob=1.0), EraserTransform(num_rect = [5, 6], size=(50,50,100,100), type='replacement')]


# trainset = SceneFlowDataset(dataset="train", crop_size=(540, 576), parts=["FlyingThings"])
trainset = SceneFlowDataset(dataset="train", crop_size=(540, 576), parts=["FlyingThings"],datasize=10)

training_batch_size = 2



# testsets = {
#     "ft":SceneFlowDataset(dataset="test", parts=["FlyingThings"])
# }
testsets = {
    "ft":SceneFlowDataset(dataset="test", parts=["FlyingThings"], datasize=10)
}
test_batch_size = 1

d_max_training_model = 192
d_max_training_eval = 192
d_max_test_models = {
    "ft":192,
}
d_max_test_evals = {
    "ft":192,
}
d_max_eval_model = 192
d_max_eval_eval = 192


evalset = SceneFlowDataset(dataset="test")
eval_batch_size = 1


# ================================================

train_lossfunc = loss_smooth_l1
test_lossfunc = loss_smooth_l1

model = ICVP(channels_3d = [32, 64, 128, 256], channel_3d_last = 32)


epoch = 60
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25,30,35,38,41,44,47], gamma=0.5)
