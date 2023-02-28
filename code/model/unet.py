import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model_utils import init_weights


class unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=16, C = [16, 32, 64, 128, 256],
                kernal_size=5, stride=1, padding = None, skip_connection="concat", nonlinearity='leaky_relu', a = 0.1, norm = 'batch',
                extract=True, resize=True, ndim=2, inlayer_modules = None):
        """
        len(C) = len(kernal_size) = len(stride) = len(padding) = len(extract) = len(skip_connection) + 1 : no skip on last layer
        = (len(inlayer_modules) - 1)/2
        """
        super(unet, self).__init__()

        l = len(C)
        if isinstance(kernal_size, int):
            kernal_size = [kernal_size] * l
        if isinstance(stride, int):
            stride = [stride] * l
        if isinstance(extract, bool):
            extract = [extract] * l
        if isinstance(skip_connection, str):
            assert skip_connection == 'concat' or skip_connection == 'add' or skip_connection == 'none'
            skip_connection = [skip_connection] * (l-1)
        else:
            for item in skip_connection: assert item == 'concat' or item == 'add' or item == 'none'                
        if padding is None:
            padding = [int((k-1)/2) for k in kernal_size]
        elif isinstance(padding, int):
            padding = [padding] * l
        if nonlinearity == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif nonlinearity == 'leaky_relu':
            self.activation = nn.LeakyReLU(negative_slope=a, inplace=True)

        if not isinstance(inlayer_modules, list):
            inlayer_modules = [inlayer_modules] * (2*l+1)

        self.extract = extract
        self.skip_connection = skip_connection
        self.resize = resize

        if ndim == 2:
            conv = nn.Conv2d
            deconv = nn.ConvTranspose2d            
            self.interpolate_mode = 'bilinear'
            if norm == 'batch':
                norm = nn.BatchNorm2d
            elif norm == 'instance':
                norm = nn.InstanceNorm2d
            else:
                norm = lambda x: nn.Sequential()
            
        elif ndim == 3:
            conv = nn.Conv3d
            deconv = nn.ConvTranspose3d            
            self.interpolate_mode = 'trilinear'
            norm = nn.BatchNorm3d
            if norm == 'batch':
                norm = nn.BatchNorm3d
            elif norm == 'instance':
                norm = nn.InstanceNorm3d
            else:
                norm = lambda x: nn.Sequential()

        self.convs = []
        self.norms_conv = []
        self.deconvs = []
        self.norms_deconv = []        

        self.convs.append(conv(in_channels, C[0], kernel_size=kernal_size[0], stride=stride[0], padding=padding[0], bias=False))
        self.norms_conv.append(norm(C[0]))
        for i in range(0, l-1):
            self.convs.append(conv(C[i], C[i+1], kernel_size=kernal_size[i+1], stride=stride[i+1], padding=padding[i], bias=False))
            self.norms_conv.append(norm(C[i+1]))

        in_channels = C[-1]
        for i in range(1, l):
            self.deconvs.append(deconv(in_channels, C[-i-1], kernel_size=kernal_size[-i], stride=stride[-i], padding=padding[-i], bias=False))
            self.norms_deconv.append(norm(C[-i-1]))
            in_channels = 2*C[-i-1] if self.skip_connection[l-i-1]=='concat' else C[-i-1]
        self.deconvs.append(deconv(in_channels, out_channels, kernel_size=kernal_size[0], stride=stride[0], padding=padding[0], bias=False))
        self.norms_deconv.append(norm(out_channels))

        self.convs = nn.ModuleList(self.convs)
        self.norms_conv = nn.ModuleList(self.norms_conv)
        self.deconvs = nn.ModuleList(self.deconvs)
        self.norms_deconv = nn.ModuleList(self.norms_deconv)
        self.inlayer_modules = nn.ModuleList(inlayer_modules)
        init_weights(self, nonlinearity, a)

    def forward(self, x):
        xs = []
        sizes = []
        res = []
        l = len(self.convs)
                
        for i in range(l):
            if self.inlayer_modules[i] is not None:
                x = self.inlayer_modules[i](x)
            xs.append(x)
            sizes.append(x.shape[2:])
            conv = self.convs[i]
            norm = self.norms_conv[i]
            x = self.activation(norm(conv(x)))
        
        for i in range(l-1):
            if self.inlayer_modules[l+i] is not None:
                x = self.inlayer_modules[l+i](x)
            if self.extract[-i-1] == True:
                if self.resize:
                    res.append(F.interpolate(x, size=sizes[0], mode=self.interpolate_mode, align_corners=True))
                else:
                    res.append(x)
            deconv = self.deconvs[i]
            norm = self.norms_deconv[i]
            if self.skip_connection[-i-1]=='concat':
                x = torch.cat((x, xs[-i-1]), dim=1)
                x = self.activation(norm(deconv(x, output_size=sizes[-i-1])))                
            elif self.skip_connection[-i-1]=='add':
                x = x + xs[-i-1]
                x = self.activation(norm(deconv(x, output_size=sizes[-i-1])))                                
            elif self.skip_connection[-i-1]=='none':
                x = self.activation(norm(deconv(x, output_size=sizes[-i-1])))
        
        if self.inlayer_modules[2*l] is not None:
            x = self.inlayer_modules[l+i](x)
        if self.extract[0] == True:
            if self.resize:
                res.append(F.interpolate(x, size=sizes[0], mode=self.interpolate_mode, align_corners=True))
            else:
                res.append(x)
        deconv = self.deconvs[-1]
        norm = self.norms_deconv[-1]
        x = self.activation(norm(deconv(x, output_size=sizes[0])))
        if self.inlayer_modules[-1] is not None:
            x = self.inlayer_modules[-1](x)

        res.append(x)
        if len(res) == 1:
            return res[0]
        else:
            return res

