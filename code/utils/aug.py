# https://github.com/princeton-vl/RAFT-Stereo/blob/main/core/utils/augmentor.py

import numpy as np
import random
import warnings
import os
import time
from glob import glob
from skimage import color, io
from PIL import Image

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
from torchvision.transforms import ColorJitter, functional, Compose
import torch.nn.functional as F



class Augmentor:
    def __init__(self, crop_size, isSparse=False, min_scale=1.0, max_scale=1.0, prob_vflip=0.0, yjitter=0, 
                args_color={
                    "brightness":0.3,
                    "contrast":0.3,
                    "saturation":[0.7,1.3],
                    "hue":0.3/3.14
                },
                gamma=[1,1,1,1], **kwargs):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.prob_scale = 0.8
        self.resizeDisp = resizeSparseDisp if isSparse else resizeDenseDisp

        # flip augmentation params
        self.prob_vflip = prob_vflip
        self.yjitter = yjitter
        
        # photometric augmentation params
        self.photo_aug = Compose([ColorJitter(**args_color), AdjustGamma(*gamma)])
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5       

    def __call__(self, img1, img2, disp, mask):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, disp, mask = self.spatial_transform(img1, img2, disp, mask)
        
        return img1, img2, disp, mask


    def color_transform(self, img1, img2):
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2


    def eraser_transform(self, img1, img2):
        h, w = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, w)
                y0 = np.random.randint(0, h)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color
        return img1, img2


    def spatial_transform(self, img_left, img_right, disp_left, mask_left):        
        if np.random.rand() < self.prob_scale:        
            scale = np.random.uniform(self.min_scale, self.max_scale )
            crop_size = [int(self.crop_size[0]/scale), int(self.crop_size[1]/scale)]        
            img_left, img_right, disp_left, mask_left = self.crop_img(img_left, img_right, disp_left, mask_left, crop_size, self.yjitter)
            
            img_left = cv2.resize(img_left, self.crop_size[::-1], interpolation=cv2.INTER_LINEAR)
            img_right = cv2.resize(img_right, self.crop_size[::-1], interpolation=cv2.INTER_LINEAR)
            disp_left, mask_left = self.resizeDisp(disp_left, mask_left, self.crop_size)                        
        else:
            img_left, img_right, disp_left, mask_left = self.crop_img(img_left,img_right, disp_left, mask_left, self.crop_size, self.yjitter)        
        
        if np.random.rand() < self.prob_vflip:
            img_left = img_left[::-1]
            img_right = img_right[::-1]
            disp_left = disp_left[::-1]        
            mask_left = mask_left[::-1]

        return img_left, img_right, disp_left, mask_left


    def crop_img(self, img_left, img_right, disp_left, mask_left, crop_size, jitter_y):        
        h_img, w_img = img_left.shape[:2]
        h_crop, w_crop = crop_size                
        if h_img >= h_crop and w_img >= w_crop:
            x_left = np.random.randint(w_img - w_crop + 1)
            y_img_left = np.random.randint(h_img - h_crop + 1)
            jitter_y = min(jitter_y, y_img_left, h_img-y_img_left-h_crop)
            y_img_right = y_img_left + np.random.randint(-jitter_y, jitter_y+1)

            crop_img_left = img_left[y_img_left:y_img_left+h_crop, x_left:x_left+w_crop]
            crop_img_right = img_right[y_img_right:y_img_right+h_crop, x_left:x_left+w_crop]
            crop_disp_left = disp_left[y_img_left:y_img_left+h_crop, x_left:x_left+w_crop]
            crop_mask_left = mask_left[y_img_left:y_img_left+h_crop, x_left:x_left+w_crop]
        else:
            crop_img_left = np.zeros((h_crop, w_crop, 3), dtype=np.float)
            crop_img_right = np.zeros((h_crop, w_crop, 3), dtype=np.float)       
            crop_disp_left = np.zeros([h_crop, w_crop] + list(disp_left.shape)[2:], dtype=np.float)             
            crop_mask_left = np.zeros((h_crop, w_crop), dtype=np.bool)
            x_img = np.random.randint(w_img - w_crop + 1) if w_img >= w_crop else 0
            x_crop = np.random.randint(w_crop - w_img + 1) if w_img < w_crop else 0 
            w = min(w_img, w_crop)            
            if h_img >= h_crop: 
                y_img = np.random.randint(h_img - h_crop + 1)
                jitter_y = min(jitter_y, y_img, h_img-y_img-h_crop)
                jitter_y = np.random.randint(-jitter_y, jitter_y+1)
                crop_img_left[:, x_crop:x_crop+w] = img_left[y_img:y_img+h_crop, x_img:x_img+w]            
                crop_img_right[:, x_crop:x_crop+w] = img_right[y_img+jitter_y:y_img+jitter_y+h_crop, x_img:x_img+w]
                crop_disp_left[:, x_crop:x_crop+w] = disp_left[y_img:y_img+h_crop, x_img:x_img+w]
                crop_mask_left[:, x_crop:x_crop+w] = mask_left[y_img:y_img+h_crop, x_img:x_img+w]
            else:
                y_crop = np.random.randint(h_crop - h_img + 1)
                jitter_y = min(jitter_y, y_crop, h_crop-y_crop-h_img)
                jitter_y = np.random.randint(-jitter_y, jitter_y+1)
                crop_img_left[y_crop:y_crop+h_img, x_crop:x_crop+w] = img_left[:, x_img:x_img+w]            
                crop_img_right[y_crop+jitter_y:y_crop+jitter_y+h_img, x_crop:x_crop+w] = img_right[:, x_img:x_img+w]
                crop_disp_left[y_crop:y_crop+h_img, x_crop:x_crop+w] = disp_left[:, x_img:x_img+w]
                crop_mask_left[y_crop:y_crop+h_img, x_crop:x_crop+w] = mask_left[:, x_img:x_img+w]

        return crop_img_left, crop_img_right, crop_disp_left, crop_mask_left



class ColorTransform:
    def __init__(self, brightness = 0.3, contrast = 0.3, saturation = [0.7,1.3], hue=0.0/3.14,                    
                gamma=[1,1,1,1], asymmetric_prob = 0.2):
        
        # photometric augmentation params
        self.photo_aug = Compose([ColorJitter(brightness, contrast, saturation, hue), AdjustGamma(*gamma)])
        self.asymmetric_prob = asymmetric_prob

    def __call__(self, img1, img2, disp, mask):
        if np.random.rand() < self.asymmetric_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2, disp, mask


class EraserTransform:
    def __init__(self, eraser_aug_prob = 0.5, num_rect = (1,3), size = (50, 50, 100, 100), type = 'mean', **kwargs):
        self.eraser_aug_prob = eraser_aug_prob
        self.num_rect = num_rect
        self.size = size
        self.type = type

    def __call__(self, img1, img2, disp, mask):
        h, w = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:            
            for _ in range(np.random.randint(self.num_rect[0], self.num_rect[1])):
                dx = np.random.randint(self.size[0], self.size[2])
                dy = np.random.randint(self.size[1], self.size[3])
                    
                if self.type == 'mean':
                    rect = np.mean(img2.reshape(-1, 3), axis=0)
                elif self.type == 'replacement':
                    x0 = np.random.randint(0, w-dx)
                    y0 = np.random.randint(0, h-dy)                    
                    rect = img2[y0:y0+dy, x0:x0+dx, :]

                x0 = np.random.randint(0, w - dx)
                y0 = np.random.randint(0, h - dy)                
                img2[y0:y0+dy, x0:x0+dx, :] = rect
        return img1, img2, disp, mask


class SpatialTransform:
    def __init__(self, crop_size, isSparse=False, min_scale=0.3, max_scale=1.5, prob_vflip=0.0, yjitter=0, 
                **kwargs):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.prob_scale = 0.8
        self.resizeDisp = resizeSparseDisp if isSparse else resizeDenseDisp

        # flip augmentation params
        self.prob_vflip = prob_vflip
        self.yjitter = yjitter
        

    def __call__(self, img_left, img_right, disp_left, mask_left):        
        if np.random.rand() < self.prob_scale:        
            scale = np.random.uniform(self.min_scale, self.max_scale )
            crop_size = [int(self.crop_size[0]/scale), int(self.crop_size[1]/scale)]        
            img_left, img_right, disp_left, mask_left = self.crop_img(img_left, img_right, disp_left, mask_left, crop_size, self.yjitter)
            
            img_left = cv2.resize(img_left, self.crop_size[::-1], interpolation=cv2.INTER_LINEAR)
            img_right = cv2.resize(img_right, self.crop_size[::-1], interpolation=cv2.INTER_LINEAR)
            disp_left, mask_left = self.resizeDisp(disp_left, mask_left, self.crop_size)                        
        else:
            img_left, img_right, disp_left, mask_left = self.crop_img(img_left,img_right, disp_left, mask_left, self.crop_size, self.yjitter)        
        
        if np.random.rand() < self.prob_vflip:
            img_left = img_left[::-1]
            img_right = img_right[::-1]
            disp_left = disp_left[::-1]        
            mask_left = mask_left[::-1]

        return img_left, img_right, disp_left, mask_left


    def crop_img(self, img_left, img_right, disp_left, mask_left, crop_size, jitter_y):        
        h_img, w_img = img_left.shape[:2]
        h_crop, w_crop = crop_size                
        if h_img >= h_crop and w_img >= w_crop:
            x_left = np.random.randint(w_img - w_crop + 1)
            y_img_left = np.random.randint(h_img - h_crop + 1)
            jitter_y = min(jitter_y, y_img_left, h_img-y_img_left-h_crop)
            y_img_right = y_img_left + np.random.randint(-jitter_y, jitter_y+1)

            crop_img_left = img_left[y_img_left:y_img_left+h_crop, x_left:x_left+w_crop]
            crop_img_right = img_right[y_img_right:y_img_right+h_crop, x_left:x_left+w_crop]
            crop_disp_left = disp_left[y_img_left:y_img_left+h_crop, x_left:x_left+w_crop]
            crop_mask_left = mask_left[y_img_left:y_img_left+h_crop, x_left:x_left+w_crop]
        else:
            crop_img_left = np.zeros((h_crop, w_crop, 3), dtype=np.float)
            crop_img_right = np.zeros((h_crop, w_crop, 3), dtype=np.float)       
            crop_disp_left = np.zeros([h_crop, w_crop] + list(disp_left.shape)[2:], dtype=np.float)             
            crop_mask_left = np.zeros((h_crop, w_crop), dtype=np.bool)
            x_img = np.random.randint(w_img - w_crop + 1) if w_img >= w_crop else 0
            x_crop = np.random.randint(w_crop - w_img + 1) if w_img < w_crop else 0 
            w = min(w_img, w_crop)            
            if h_img >= h_crop: 
                y_img = np.random.randint(h_img - h_crop + 1)
                jitter_y = min(jitter_y, y_img, h_img-y_img-h_crop)
                jitter_y = np.random.randint(-jitter_y, jitter_y+1)
                crop_img_left[:, x_crop:x_crop+w] = img_left[y_img:y_img+h_crop, x_img:x_img+w]            
                crop_img_right[:, x_crop:x_crop+w] = img_right[y_img+jitter_y:y_img+jitter_y+h_crop, x_img:x_img+w]
                crop_disp_left[:, x_crop:x_crop+w] = disp_left[y_img:y_img+h_crop, x_img:x_img+w]
                crop_mask_left[:, x_crop:x_crop+w] = mask_left[y_img:y_img+h_crop, x_img:x_img+w]
            else:
                y_crop = np.random.randint(h_crop - h_img + 1)
                jitter_y = min(jitter_y, y_crop, h_crop-y_crop-h_img)
                jitter_y = np.random.randint(-jitter_y, jitter_y+1)
                crop_img_left[y_crop:y_crop+h_img, x_crop:x_crop+w] = img_left[:, x_img:x_img+w]            
                crop_img_right[y_crop+jitter_y:y_crop+jitter_y+h_img, x_crop:x_crop+w] = img_right[:, x_img:x_img+w]
                crop_disp_left[y_crop:y_crop+h_img, x_crop:x_crop+w] = disp_left[:, x_img:x_img+w]
                crop_mask_left[y_crop:y_crop+h_img, x_crop:x_crop+w] = mask_left[:, x_img:x_img+w]

        return crop_img_left, crop_img_right, crop_disp_left, crop_mask_left



class PureCropper:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img_left, img_right, disp_left, mask_left):        
        h_img, w_img = img_left.shape[:2]
        h_crop, w_crop = self.crop_size                
        if h_img >= h_crop and w_img >= w_crop:
            x_left = np.random.randint(w_img - w_crop + 1)
            y_img = np.random.randint(h_img - h_crop + 1)

            crop_img_left = img_left[y_img:y_img+h_crop, x_left:x_left+w_crop]
            crop_img_right = img_right[y_img:y_img+h_crop, x_left:x_left+w_crop]
            crop_disp_left = disp_left[y_img:y_img+h_crop, x_left:x_left+w_crop]
            crop_mask_left = mask_left[y_img:y_img+h_crop, x_left:x_left+w_crop]
        else:
            crop_img_left = np.zeros((h_crop, w_crop, 3), dtype=np.float)
            crop_img_right = np.zeros((h_crop, w_crop, 3), dtype=np.float)       
            crop_disp_left = np.zeros([h_crop, w_crop] + list(disp_left.shape)[2:], dtype=np.float)                     
            crop_mask_left = np.zeros((h_crop, w_crop), dtype=np.bool)
            x_img = np.random.randint(w_img - w_crop + 1) if w_img >= w_crop else 0
            x_crop = np.random.randint(w_crop - w_img + 1) if w_img < w_crop else 0 
            w = min(w_img, w_crop)            
            if h_img >= h_crop: 
                y_img = np.random.randint(h_img - h_crop + 1)
                crop_img_left[:, x_crop:x_crop+w] = img_left[y_img:y_img+h_crop, x_img:x_img+w]            
                crop_img_right[:, x_crop:x_crop+w] = img_right[y_img:y_img+h_crop, x_img:x_img+w]
                crop_disp_left[:, x_crop:x_crop+w] = disp_left[y_img:y_img+h_crop, x_img:x_img+w]
                crop_mask_left[:, x_crop:x_crop+w] = mask_left[y_img:y_img+h_crop, x_img:x_img+w]
            else:
                y_crop = np.random.randint(h_crop - h_img + 1)
                crop_img_left[y_crop:y_crop+h_img, x_crop:x_crop+w] = img_left[:, x_img:x_img+w]            
                crop_img_right[y_crop:y_crop+h_img, x_crop:x_crop+w] = img_right[:, x_img:x_img+w]
                crop_disp_left[y_crop:y_crop+h_img, x_crop:x_crop+w] = disp_left[:, x_img:x_img+w]
                crop_mask_left[y_crop:y_crop+h_img, x_crop:x_crop+w] = mask_left[:, x_img:x_img+w]

        return crop_img_left, crop_img_right, crop_disp_left, crop_mask_left




class AdjustGamma(object):
    def __init__(self, gamma_min, gamma_max, gain_min=1.0, gain_max=1.0):
        self.gamma_min, self.gamma_max, self.gain_min, self.gain_max = gamma_min, gamma_max, gain_min, gain_max

    def __call__(self, sample):
        gain = random.uniform(self.gain_min, self.gain_max)
        gamma = random.uniform(self.gamma_min, self.gamma_max)
        return functional.adjust_gamma(sample, gamma, gain)

    def __repr__(self):
        return f"Adjust Gamma {self.gamma_min}, ({self.gamma_max}) and Gain ({self.gain_min}, {self.gain_max})"


def resizeDenseDisp(disp_left, mask_left, size):
    scale = size[1] / disp_left.shape[1]
    disp_left = cv2.resize(disp_left, size[::-1], interpolation=cv2.INTER_LINEAR)
    disp_left = scale * disp_left            
    mask_left = cv2.resize(mask_left.astype(np.int16), size[::-1], interpolation=cv2.INTER_NEAREST).astype(np.bool)
    return disp_left, mask_left


def resizeSparseDisp(disp_left, mask_left, size):
    scale_y = size[0] / disp_left.shape[0]
    scale_x = size[1] / disp_left.shape[1]
    h, w = disp_left.shape[:2]    
    coords = np.meshgrid(np.arange(w), np.arange(h))
    coords = np.stack(coords, axis=-1)
    coords = coords.reshape(-1, 2).astype(np.float32)
    disp_shape = disp_left.shape        

    if len(disp_shape) == 2:
        disp_left = disp_left.reshape(-1).astype(np.float32)
    else:
        disp_left = disp_left.reshape(-1, *disp_shape[2:]).astype(np.float32)
    mask_left = mask_left.reshape(-1)

    coords0 = coords[mask_left]
    disp_left0 = disp_left[mask_left]

    coords1 = coords0 * [scale_x, scale_y]
    disp_left1 = disp_left0 * scale_x

    xx = np.round(coords1[:,0]).astype(np.int32)
    yy = np.round(coords1[:,1]).astype(np.int32)

    in_size = (xx > 0) & (xx < size[1]) & (yy > 0) & (yy < size[0])
    xx = xx[in_size]
    yy = yy[in_size]
    disp_left1 = disp_left1[in_size]
        
    if len(disp_shape) == 2:
        disp_left = np.zeros([size[0], size[1]], dtype=np.float32)
    else:
        disp_left = np.zeros([size[0], size[1]]+list(disp_shape[2:]), dtype=np.float32)
    mask_left = np.zeros([size[0], size[1]], dtype=np.bool)

    disp_left[yy, xx] = disp_left1
    mask_left[yy, xx] = True

    return disp_left, mask_left