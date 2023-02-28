import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model_utils import init_weights


class CNA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, dilation=1, deconv=False, ndim=2, 
                norm='batch', nonlinearity='leaky_relu', a = 0.1):
        super(CNA, self).__init__()
        
        bias = norm == 'none'
        if padding is None:
            padding = int((kernel_size-1)/2)
        if not deconv and ndim == 2:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias)
        elif deconv and ndim == 2:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias)
        elif not deconv and ndim == 3:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias)
        elif deconv and ndim == 3:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias)

        if norm == 'batch':
            if ndim == 2:                
                self.norm = nn.BatchNorm2d(out_channels)
            elif ndim == 3:
                self.norm = nn.BatchNorm3d(out_channels)
        elif norm == 'instance':
            if ndim == 2:                
                self.norm = nn.InstanceNorm2d(out_channels)
            elif ndim == 3:
                self.norm = nn.InstanceNorm3d(out_channels)
        elif norm == 'none':
            self.norm = nn.Sequential()

        if nonlinearity == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif nonlinearity == 'leaky_relu':
            self.activation = nn.LeakyReLU(negative_slope=a, inplace=True)                
        init_weights(self, nonlinearity, a)

    def forward(self, x, output_size=None):
        if output_size is None:
            return self.activation(self.norm(self.conv(x)))
        else:            
            return self.activation(self.norm(self.conv(x, output_size=output_size)))


class NA(nn.Module):
    def __init__(self, out_channels, ndim=2, norm='batch', nonlinearity='leaky_relu', a = 0.1):
        super(NA, self).__init__()

        if norm == 'batch':
            if ndim == 2:                
                self.norm = nn.BatchNorm2d(out_channels)
            elif ndim == 3:
                self.norm = nn.BatchNorm3d(out_channels)
        elif norm == 'instance':
            if ndim == 2:                
                self.norm = nn.InstanceNorm2d(out_channels)
            elif ndim == 3:
                self.norm = nn.InstanceNorm3d(out_channels)
        else:
            self.norm = nn.Sequential()

        if nonlinearity == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif nonlinearity == 'leaky_relu':
            self.activation = nn.LeakyReLU(negative_slope=a, inplace=True)        
        init_weights(self, nonlinearity, a)

    def forward(self, x):
        return self.activation(self.norm(x))
