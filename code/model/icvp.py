import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from model.unet import unet
from model.cna import CNA, NA
from model.feature import *
from model.asblock import ASBlock
from utils.model_utils import init_weights
from model.corr_block import *



  
class ICVP(nn.Module):
    def __init__(self, 
                channels_3d = [32, 64, 128, 256], channel_3d_last = 32,
                channels_2d = None, channel_2d_last = None,
                feature = FeatureHighdim(),
                nonlinearity='leaky_relu', a=0.1):
        super(ICVP, self).__init__()
        if not channels_2d:
            channels_2d = channels_3d
        if not channel_2d_last:
            channel_2d_last = channel_3d_last

        self.n_levels = len(channels_3d)
        self.channel_feature = 16                        
        self.feature = feature

        self.stem2d = nn.Sequential(
                            CNA(3, 16, kernel_size=3, stride=1, deconv=False, ndim=2, nonlinearity=nonlinearity, a=a),
                            CNA(16, 16, kernel_size=3, stride=3, deconv=False, ndim=2, nonlinearity=nonlinearity, a=a),
                            CNA(16, 16, kernel_size=3, stride=1, deconv=False, ndim=2, nonlinearity=nonlinearity, a=a)
                        )
        self.disparity_block = gwc_block(n_head=16)
        self.in_channels_3d = 16
        
        self.init_2d(channels_2d, channel_2d_last, channels_3d, channel_3d_last, nonlinearity, a)
        self.init_3d(channels_3d, channel_3d_last, nonlinearity, a)
        

        init_weights(self, nonlinearity, a)



    def forward(self, img_left, img_right, d_max=192):     
        b, _, h, w = img_left.shape
        img_left = 2 * (img_left / 255.0) - 1.0
        img_right = 2 * (img_right / 255.0) - 1.0
        imgs = torch.cat([img_left, img_right])
        
        features = self.feature(imgs)
        feature_left = features[:b]
        feature_right = features[b:]                
        
        x_2d_enc_layers = []
        x_2d_enc_gates = []
        x_2d_dec_gates = []
        x_2d = self.stem2d(img_left)      
        x_2d_enc_layers.append(x_2d)
        for i in range(self.n_levels):
            x_2d = self.encoder2ds[i](x_2d)
            x_2d_enc_layers.append(x_2d)
            x_2d_enc_gate = self.encoder2d_gates[i](x_2d)
            x_2d_enc_gates.append(x_2d_enc_gate)
        
        for i in range(self.n_levels):
            x_2d = self.decoder2d_ups[i](x_2d, output_size = x_2d_enc_layers[-(i+2)].shape[-2:])
            x_2d = torch.cat([x_2d, x_2d_enc_layers[-(i+2)]],dim=1)
            x_2d = self.decoder2d_completes[i](x_2d)            
            x_2d_dec_gate = self.decoder2d_gates[i](x_2d)
            x_2d_dec_gates.append(x_2d_dec_gate)        

        x_3d_enc_layers = []        
        x_3d = self.disparity_block(feature_left, feature_right, d_max=d_max//3)
        
        x_3d_enc_layers.append(x_3d)        
        for i in range(self.n_levels):
            x_3d = self.encoder3d_downs[i](x_3d) + x_2d_enc_gates[i][:,:,None]
            x_3d = self.encoder3d_completes[i](x_3d)
            x_3d_enc_layers.append(x_3d)
        
        for i in range(self.n_levels):
            x_3d = self.decoder3d_ups[i](x_3d, output_size = x_3d_enc_layers[-(i+2)].shape[-3:])
            x_3d = torch.cat([x_3d, x_3d_enc_layers[-(i+2)]],dim=1)
            x_3d = self.decoder3d_fusions[i](x_3d) + x_2d_dec_gates[i][:,:,None]
            x_3d = self.decoder3d_completes[i](x_3d)        
            
        disp = x_3d
        disp = F.interpolate(disp, [3*disp.shape[-3], img_left.shape[2], img_left.shape[3]], mode='trilinear', align_corners=True)                
        disp = disp[:,0]
        disp_pred = (torch.arange(disp.shape[1], device=disp.device)[None,:,None,None] * F.softmax(disp, 1)).sum(1, keepdim=True)
        return disp_pred, _

    def init_2d(self, channels_2d, channel_2d_last, channels_3d, channel_3d_last, nonlinearity, a):
        encoder2ds = []
        encoder2d = nn.Sequential(
                        CNA(self.channel_feature, channels_2d[0], kernel_size=3, stride=2, padding=1, deconv=False, ndim=2, nonlinearity=nonlinearity, a=a),
                        CNA(channels_2d[0], channels_2d[0], kernel_size=3, stride=1, padding=1, deconv=False, ndim=2, nonlinearity=nonlinearity, a=a),
                        CNA(channels_2d[0], channels_2d[0], kernel_size=3, stride=1, padding=1, deconv=False, ndim=2, nonlinearity=nonlinearity, a=a)
                    )
        encoder2ds.append(encoder2d)        
        for i in range(1, len(channels_2d)):
            encoder2d = nn.Sequential(
                            CNA(channels_2d[i-1], channels_2d[i], kernel_size=3, stride=2, padding=1, deconv=False, ndim=2, nonlinearity=nonlinearity, a=a),
                            CNA(channels_2d[i], channels_2d[i], kernel_size=3, stride=1, padding=1, deconv=False, ndim=2, nonlinearity=nonlinearity, a=a),
                            CNA(channels_2d[i], channels_2d[i], kernel_size=3, stride=1, padding=1, deconv=False, ndim=2, nonlinearity=nonlinearity, a=a)
                        )
            encoder2ds.append(encoder2d)            
        self.encoder2ds = nn.ModuleList(encoder2ds)
        

        encoder2d_gates = [
            nn.Sequential(
                ASBlock(channel_2d, channel_3d, kernel_size=3, dilation=[(1,1),(2,2),(3,3)], ndim=2, nonlinearity=nonlinearity, a=a),
                NA(3*channel_3d, ndim=2, nonlinearity=nonlinearity, a=a),
                nn.Conv2d(3*channel_3d, channel_3d, kernel_size=1, stride=1, padding=0)
            )
        for channel_2d, channel_3d in zip(channels_2d, channels_3d)
        ]        
        self.encoder2d_gates = nn.ModuleList(encoder2d_gates)
        
        decoder2d_ups = [CNA(channels_2d[i], channels_2d[i-1], kernel_size=3, stride=2, padding=1, deconv=True, ndim=2, nonlinearity=nonlinearity, a=a) for i in range(len(channels_2d)-1, 0, -1)]
        decoder2d_ups.append(CNA(channels_2d[0], channel_2d_last , kernel_size=3, stride=2, padding=1, deconv=True, ndim=2, nonlinearity=nonlinearity, a=a))
        self.decoder2d_ups = nn.ModuleList(decoder2d_ups)

        decoder2d_completes = []
        for i in range(len(channels_2d)-2, -1, -1):
            decoder2d_complete = nn.Sequential(
                                        CNA(2*channels_2d[i], channels_2d[i], kernel_size=1, stride=1, padding=0, deconv=False, ndim=2, nonlinearity=nonlinearity, a=a),
                                        CNA(channels_2d[i], channels_2d[i], kernel_size=3, stride=1, padding=1, deconv=False, ndim=2, nonlinearity=nonlinearity, a=a),
                                        CNA(channels_2d[i], channels_2d[i], kernel_size=3, stride=1, padding=1, deconv=False, ndim=2, nonlinearity=nonlinearity, a=a)
                                    )
            decoder2d_completes.append(decoder2d_complete)        
        decoder2d_complete = nn.Sequential(
                                    CNA(self.channel_feature + channel_2d_last, channel_2d_last, kernel_size=1, stride=1, padding=0, deconv=False, ndim=2, nonlinearity=nonlinearity, a=a),
                                    CNA(channel_2d_last, channel_2d_last, kernel_size=3, stride=1, padding=1, deconv=False, ndim=2, nonlinearity=nonlinearity, a=a),
                                    CNA(channel_2d_last, channel_2d_last, kernel_size=3, stride=1, padding=1, deconv=False, ndim=2, nonlinearity=nonlinearity, a=a)
                                )
        decoder2d_completes.append(decoder2d_complete)            
        self.decoder2d_completes = nn.ModuleList(decoder2d_completes)
        
        decoder2d_gates = [
            nn.Sequential(
                ASBlock(channels_2d[i], channels_3d[i], kernel_size=3, dilation=[(1,1),(2,2),(3,3)], ndim=2, nonlinearity=nonlinearity, a=a),
                NA(3*channels_3d[i], ndim=2, nonlinearity=nonlinearity, a=a),
                nn.Conv2d(3*channels_3d[i], channels_3d[i], kernel_size=1, stride=1, padding=0)
            )
        for i in range(len(channels_2d)-2, -1, -1)
        ]        
        decoder2d_gates.append(nn.Conv2d(channel_2d_last, channel_3d_last, kernel_size=3, stride=1, padding=1))
        self.decoder2d_gates = nn.ModuleList(decoder2d_gates)


    def init_3d(self, channels_3d, channel_3d_last, nonlinearity, a):
        encoder3d_downs = [
            nn.Sequential(
                CNA(self.in_channels_3d, channels_3d[0], kernel_size=3, stride=2, padding=1, deconv=False, ndim=3, nonlinearity=nonlinearity, a=a),
                ASBlock(channels_3d[0], channels_3d[0], kernel_size=3, dilation=[(1,1,1),(1,2,2),(1,3,3)], ndim=3, nonlinearity=nonlinearity, a=a),
                NA(3*channels_3d[0], ndim=3, nonlinearity=nonlinearity, a=a),
                nn.Conv3d(3*channels_3d[0], channels_3d[0], kernel_size=1, stride=1, padding=0)
            )
        ]
        for i in range(len(channels_3d)-1):
            encoder3d_down = nn.Sequential(
                CNA(channels_3d[i], channels_3d[i+1], kernel_size=3, stride=2, padding=1, deconv=False, ndim=3, nonlinearity=nonlinearity, a=a),
                ASBlock(channels_3d[i+1], channels_3d[i+1], kernel_size=3, dilation=[(1,1,1),(1,2,2),(1,3,3)], ndim=3, nonlinearity=nonlinearity, a=a),
                NA(3*channels_3d[i+1], ndim=3, nonlinearity=nonlinearity, a=a),
                nn.Conv3d(3*channels_3d[i+1], channels_3d[i+1], kernel_size=1, stride=1, padding=0)
            )            
            encoder3d_downs.append(encoder3d_down)
        self.encoder3d_downs = nn.ModuleList(encoder3d_downs)
        
        encoder3d_completes = []
        for channel in channels_3d:
            encoder3d_complete = nn.Sequential(
                                NA(channel, ndim=3, nonlinearity=nonlinearity, a=a),
                                CNA(channel, channel, kernel_size=3, stride=1, padding=1, deconv=False, ndim=3, nonlinearity=nonlinearity, a=a),
                                CNA(channel, channel, kernel_size=3, stride=1, padding=1, deconv=False, ndim=3, nonlinearity=nonlinearity, a=a)
                            )
            encoder3d_completes.append(encoder3d_complete)
        self.encoder3d_completes = nn.ModuleList(encoder3d_completes)

        decoder3d_ups = [CNA(channels_3d[i], channels_3d[i-1], kernel_size=3, stride=2, padding=1, deconv=True, ndim=3, nonlinearity=nonlinearity, a=a) for i in range(len(channels_3d)-1, 0, -1)]
        decoder3d_ups.append(CNA(channels_3d[0], channel_3d_last, kernel_size=3, stride=2, padding=1, deconv=True, ndim=3, nonlinearity=nonlinearity, a=a))
        self.decoder3d_ups = nn.ModuleList(decoder3d_ups)
        
        decoder3d_fusions = []
        for i in range(len(channels_3d)-2, -1, -1):
            decoder3d_fusion = nn.Sequential(
                CNA(2*channels_3d[i], channels_3d[i], kernel_size=1, stride=1, padding=0, deconv=False, ndim=3, nonlinearity=nonlinearity, a=a),
                ASBlock(channels_3d[i], channels_3d[i], kernel_size=3, dilation=[(1,1,1),(1,2,2),(1,3,3)], ndim=3, nonlinearity=nonlinearity, a=a),
                NA(3*channels_3d[i], ndim=3, nonlinearity=nonlinearity, a=a),
                nn.Conv3d(3*channels_3d[i], channels_3d[i], kernel_size=1, stride=1, padding=0)
            )
            decoder3d_fusions.append(decoder3d_fusion)
        decoder3d_fusion = nn.Conv3d(self.in_channels_3d + channel_3d_last, channel_3d_last, kernel_size=3, stride=1, padding=1)
        decoder3d_fusions.append(decoder3d_fusion)       
        self.decoder3d_fusions = nn.ModuleList(decoder3d_fusions)

        decoder3d_completes = []
        for i in range(len(channels_3d)-2, -1, -1):
            decoder3d_complete = nn.Sequential(
                                        NA(channels_3d[i], ndim=3, nonlinearity=nonlinearity, a=a),
                                        CNA(channels_3d[i], channels_3d[i], kernel_size=3, stride=1, padding=1, deconv=False, ndim=3, nonlinearity=nonlinearity, a=a)
                                    )
            decoder3d_completes.append(decoder3d_complete)       
        decoder3d_complete = nn.Sequential(
                                    NA(channel_3d_last, ndim=3, nonlinearity=nonlinearity, a=a),
                                    nn.Conv3d(channel_3d_last, 1, kernel_size=3, stride=1, padding=1)
                                )
        decoder3d_completes.append(decoder3d_complete)
        self.decoder3d_completes = nn.ModuleList(decoder3d_completes)

    def off_training_3d(self):        
        for p in self.encoder3d_completes.parameters():
            p.requires_grad = False 
        for p in self.decoder3d_ups.parameters():
            p.requires_grad = False         
        for p in self.decoder3d_completes.parameters():
            p.requires_grad = False



class ICVP_wo_1x1(nn.Module):
    def __init__(self, 
                channels_3d = [32, 32, 64, 64], channel_3d_last = 32,
                channels_2d = None, channel_2d_last = None,
                nonlinearity='leaky_relu', a=0.1):
        super(ICVP_wo_1x1, self).__init__()
        if not channels_2d:
            channels_2d = channels_3d
        if not channel_2d_last:
            channel_2d_last = channel_3d_last

        self.n_levels = len(channels_3d)
        self.channel_feature = 16                
        # self.feature = FeatureUnet()
        # self.feature = FeatureUnetTransformer()                
        self.feature = FeatureHighdim()        
        # self.feature = FeatureHighdimTransformer()

        self.stem2d = nn.Sequential(
                            CNA(3, 16, kernel_size=3, stride=1, deconv=False, ndim=2, nonlinearity=nonlinearity, a=a),
                            CNA(16, 16, kernel_size=3, stride=3, deconv=False, ndim=2, nonlinearity=nonlinearity, a=a),
                            CNA(16, 16, kernel_size=3, stride=1, deconv=False, ndim=2, nonlinearity=nonlinearity, a=a)
                        )
        # self.disparity_block = disparity_block()
        # self.disparity_block = corr_block()
        # self.in_channels_3d = 1
        self.disparity_block = gwc_block(n_head=16)
        self.in_channels_3d = 16
        # self.disparity_block = gwc_block(n_head=32)
        # self.in_channels_3d = 32

        
        
        self.init_2d(channels_2d, channel_2d_last, channels_3d, channel_3d_last, nonlinearity, a)
        self.init_3d(channels_3d, channel_3d_last, nonlinearity, a)
        
        init_weights(self, nonlinearity, a)



    def forward(self, img_left, img_right, d_max=192):     
        b, _, h, w = img_left.shape
        img_left = 2 * (img_left / 255.0) - 1.0
        img_right = 2 * (img_right / 255.0) - 1.0
        imgs = torch.cat([img_left, img_right])
        
        features = self.feature(imgs)
        feature_left = features[:b]
        feature_right = features[b:]                
        
        x_2d_enc_layers = []
        x_2d_enc_gates = []
        x_2d_dec_gates = []
        x_2d = self.stem2d(img_left)      
        x_2d_enc_layers.append(x_2d)
        for i in range(self.n_levels):
            x_2d = self.encoder2ds[i](x_2d)
            x_2d_enc_layers.append(x_2d)
            x_2d_enc_gate = self.encoder2d_gates[i](x_2d)
            x_2d_enc_gates.append(x_2d_enc_gate)
        
        for i in range(self.n_levels):
            x_2d = self.decoder2d_ups[i](x_2d, output_size = x_2d_enc_layers[-(i+2)].shape[-2:])
            x_2d = torch.cat([x_2d, x_2d_enc_layers[-(i+2)]],dim=1)
            x_2d = self.decoder2d_completes[i](x_2d)            
            x_2d_dec_gate = self.decoder2d_gates[i](x_2d)
            x_2d_dec_gates.append(x_2d_dec_gate)        

        x_3d_enc_layers = []        
        x_3d = self.disparity_block(feature_left, feature_right, d_max=d_max//3)
        
        x_3d_enc_layers.append(x_3d)        
        for i in range(self.n_levels):
            x_3d = self.encoder3d_downs[i](x_3d) + x_2d_enc_gates[i][:,:,None]
            x_3d = self.encoder3d_completes[i](x_3d)
            x_3d_enc_layers.append(x_3d)
        
        for i in range(self.n_levels):
            x_3d = self.decoder3d_ups[i](x_3d, output_size = x_3d_enc_layers[-(i+2)].shape[-3:])
            x_3d = torch.cat([x_3d, x_3d_enc_layers[-(i+2)]],dim=1)
            x_3d = self.decoder3d_fusions[i](x_3d) + x_2d_dec_gates[i][:,:,None]
            x_3d = self.decoder3d_completes[i](x_3d)        
        
        disp = x_3d
        disp = F.interpolate(disp, [3*disp.shape[-3], img_left.shape[2], img_left.shape[3]], mode='trilinear', align_corners=True)        
        disp = disp[:,0]
        disp_pred = (torch.arange(disp.shape[1], device=disp.device)[None,:,None,None] * F.softmax(disp, 1)).sum(1, keepdim=True)
        
        return disp_pred, _
        

    def init_2d(self, channels_2d, channel_2d_last, channels_3d, channel_3d_last, nonlinearity, a):
        encoder2ds = []
        encoder2d = nn.Sequential(
                        CNA(self.channel_feature, channels_2d[0], kernel_size=3, stride=2, padding=1, deconv=False, ndim=2, nonlinearity=nonlinearity, a=a),
                        CNA(channels_2d[0], channels_2d[0], kernel_size=3, stride=1, padding=1, deconv=False, ndim=2, nonlinearity=nonlinearity, a=a),
                        CNA(channels_2d[0], channels_2d[0], kernel_size=3, stride=1, padding=1, deconv=False, ndim=2, nonlinearity=nonlinearity, a=a)
                    )
        encoder2ds.append(encoder2d)        
        for i in range(1, len(channels_2d)):
            encoder2d = nn.Sequential(
                            CNA(channels_2d[i-1], channels_2d[i], kernel_size=3, stride=2, padding=1, deconv=False, ndim=2, nonlinearity=nonlinearity, a=a),
                            CNA(channels_2d[i], channels_2d[i], kernel_size=3, stride=1, padding=1, deconv=False, ndim=2, nonlinearity=nonlinearity, a=a),
                            CNA(channels_2d[i], channels_2d[i], kernel_size=3, stride=1, padding=1, deconv=False, ndim=2, nonlinearity=nonlinearity, a=a)
                        )
            encoder2ds.append(encoder2d)            
        self.encoder2ds = nn.ModuleList(encoder2ds)
        

        encoder2d_gates = [ASBlock(channel_2d, channel_3d, kernel_size=3, dilation=[(1,1),(2,2),(3,3)], ndim=2, nonlinearity=nonlinearity, a=a) for channel_2d, channel_3d in zip(channels_2d, channels_3d)]
        # encoder2d_gates = [nn.Conv2d(channel_2d, channel_3d, kernel_size=3, stride=1, padding=1) for channel_2d, channel_3d in zip(channels_2d, channels_3d)]
        self.encoder2d_gates = nn.ModuleList(encoder2d_gates)
        
        decoder2d_ups = [CNA(channels_2d[i], channels_2d[i-1], kernel_size=3, stride=2, padding=1, deconv=True, ndim=2, nonlinearity=nonlinearity, a=a) for i in range(len(channels_2d)-1, 0, -1)]
        decoder2d_ups.append(CNA(channels_2d[0], channel_2d_last , kernel_size=3, stride=2, padding=1, deconv=True, ndim=2, nonlinearity=nonlinearity, a=a))
        self.decoder2d_ups = nn.ModuleList(decoder2d_ups)

        decoder2d_completes = []
        for i in range(len(channels_2d)-2, -1, -1):
            decoder2d_complete = nn.Sequential(
                                        CNA(channels_2d[i]*2, channels_2d[i], kernel_size=3, stride=1, padding=1, deconv=False, ndim=2, nonlinearity=nonlinearity, a=a),
                                        CNA(channels_2d[i], channels_2d[i], kernel_size=3, stride=1, padding=1, deconv=False, ndim=2, nonlinearity=nonlinearity, a=a)
                                    )
            decoder2d_completes.append(decoder2d_complete)        
        decoder2d_complete = nn.Sequential(
                                    CNA(self.channel_feature + channel_2d_last, channel_2d_last, kernel_size=3, stride=1, padding=1, deconv=False, ndim=2, nonlinearity=nonlinearity, a=a),
                                    CNA(channel_2d_last, channel_2d_last, kernel_size=3, stride=1, padding=1, deconv=False, ndim=2, nonlinearity=nonlinearity, a=a)
                                )
        decoder2d_completes.append(decoder2d_complete)            
        self.decoder2d_completes = nn.ModuleList(decoder2d_completes)
        
        decoder2d_gates = [ASBlock(channels_2d[i], channels_3d[i], kernel_size=3, dilation=[(1,1),(2,2),(3,3)], ndim=2, nonlinearity=nonlinearity, a=a) for i in range(len(channels_2d)-2, -1, -1)]        
        decoder2d_gates.append(nn.Conv2d(channel_2d_last, channel_3d_last, kernel_size=3, stride=1, padding=1))
        self.decoder2d_gates = nn.ModuleList(decoder2d_gates)


    def init_3d(self, channels_3d, channel_3d_last, nonlinearity, a):
        encoder3d_downs = [
            nn.Sequential(
                CNA(self.in_channels_3d, channels_3d[0], kernel_size=3, stride=2, padding=1, deconv=False, ndim=3, nonlinearity=nonlinearity, a=a),
                ASBlock(channels_3d[0], channels_3d[0], kernel_size=3, dilation=[(1,1,1),(1,2,2),(1,3,3)], ndim=3, nonlinearity=nonlinearity, a=a)
            )
        ]
        for i in range(len(channels_3d)-1):
            encoder3d_down = nn.Sequential(
                CNA(channels_3d[i], channels_3d[i+1], kernel_size=3, stride=2, padding=1, deconv=False, ndim=3, nonlinearity=nonlinearity, a=a),
                ASBlock(channels_3d[i+1], channels_3d[i+1], kernel_size=3, dilation=[(1,1,1),(1,2,2),(1,3,3)], ndim=3, nonlinearity=nonlinearity, a=a)
            )            
            encoder3d_downs.append(encoder3d_down)
        self.encoder3d_downs = nn.ModuleList(encoder3d_downs)
        
        encoder3d_completes = []
        for channel in channels_3d:
            encoder3d_complete = nn.Sequential(
                                NA(3*channel, ndim=3, nonlinearity=nonlinearity, a=a),
                                CNA(3*channel, channel, kernel_size=3, stride=1, padding=1, deconv=False, ndim=3, nonlinearity=nonlinearity, a=a),
                                CNA(channel, channel, kernel_size=3, stride=1, padding=1, deconv=False, ndim=3, nonlinearity=nonlinearity, a=a)
                            )
            encoder3d_completes.append(encoder3d_complete)
        self.encoder3d_completes = nn.ModuleList(encoder3d_completes)

        decoder3d_ups = [CNA(channels_3d[i], channels_3d[i-1], kernel_size=3, stride=2, padding=1, deconv=True, ndim=3, nonlinearity=nonlinearity, a=a) for i in range(len(channels_3d)-1, 0, -1)]
        decoder3d_ups.append(CNA(channels_3d[0], channel_3d_last, kernel_size=3, stride=2, padding=1, deconv=True, ndim=3, nonlinearity=nonlinearity, a=a))
        self.decoder3d_ups = nn.ModuleList(decoder3d_ups)
        
        decoder3d_fusions = []
        for i in range(len(channels_3d)-2, -1, -1):
            decoder3d_fusion = ASBlock(channels_3d[i]*2, channels_3d[i], kernel_size=3, dilation=[(1,1,1),(1,2,2),(1,3,3)], ndim=3, nonlinearity=nonlinearity, a=a)
            decoder3d_fusions.append(decoder3d_fusion)
        decoder3d_fusion = nn.Conv3d(self.in_channels_3d + channel_3d_last, channel_3d_last, kernel_size=3, stride=1, padding=1)
        decoder3d_fusions.append(decoder3d_fusion)       
        self.decoder3d_fusions = nn.ModuleList(decoder3d_fusions)

        decoder3d_completes = []
        for i in range(len(channels_3d)-2, -1, -1):
            decoder3d_complete = nn.Sequential(
                                        NA(3*channels_3d[i], ndim=3, nonlinearity=nonlinearity, a=a),
                                        CNA(3*channels_3d[i], channels_3d[i], kernel_size=3, stride=1, padding=1, deconv=False, ndim=3, nonlinearity=nonlinearity, a=a)
                                    )
            decoder3d_completes.append(decoder3d_complete)       
        decoder3d_complete = nn.Sequential(
                                    NA(channel_3d_last, ndim=3, nonlinearity=nonlinearity, a=a),
                                    nn.Conv3d(channel_3d_last, 1, kernel_size=3, stride=1, padding=1)
                                )
        decoder3d_completes.append(decoder3d_complete)
        self.decoder3d_completes = nn.ModuleList(decoder3d_completes)