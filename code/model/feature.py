import numpy as np
import torch
import torch.nn as nn

from model.cna import CNA
from utils.model_utils import init_weights




class FeatureHighdim(nn.Module):
    def __init__(self, channels = [32, 64, 128], out_channels = 128):
        super(FeatureHighdim, self).__init__()

        in_channels = 32
        nonlinearity='leaky_relu'
        a=0.1
        
        self.down = nn.Sequential(
            CNA(3, 16, kernel_size=3, stride=1, padding=1, deconv=False, ndim=2),
            CNA(16, 32, kernel_size=3, stride=3, padding=1, deconv=False, ndim=2),
            CNA(32, 32, kernel_size=3, stride=1, padding=1, deconv=False, ndim=2)
        )
        
        encoders = []
        encoder2d = nn.Sequential(
                        CNA(in_channels, channels[0], kernel_size=3, stride=2, padding=1, deconv=False, ndim=2, nonlinearity=nonlinearity, a=a),
                        CNA(channels[0], channels[0], kernel_size=3, stride=1, padding=1, deconv=False, ndim=2, nonlinearity=nonlinearity, a=a),
                        CNA(channels[0], channels[0], kernel_size=3, stride=1, padding=1, deconv=False, ndim=2, nonlinearity=nonlinearity, a=a)
                    )
        encoders.append(encoder2d)        
        for i in range(1, len(channels)):
            encoder2d = nn.Sequential(
                            CNA(channels[i-1], channels[i], kernel_size=3, stride=2, padding=1, deconv=False, ndim=2, nonlinearity=nonlinearity, a=a),
                            CNA(channels[i], channels[i], kernel_size=3, stride=1, padding=1, deconv=False, ndim=2, nonlinearity=nonlinearity, a=a),
                            CNA(channels[i], channels[i], kernel_size=3, stride=1, padding=1, deconv=False, ndim=2, nonlinearity=nonlinearity, a=a)
                        )
            encoders.append(encoder2d)            
        self.encoders = nn.ModuleList(encoders)       

        
        decoder_ups = [CNA(channels[-1], out_channels-channels[-2], kernel_size=3, stride=2, padding=1, deconv=True, ndim=2, nonlinearity=nonlinearity, a=a)]
        for i in range(len(channels)-2, 0, -1):
            # print(channels[i-1])
            decoder_ups.append(CNA(out_channels, out_channels-channels[i-1], kernel_size=3, stride=2, padding=1, deconv=True, ndim=2, nonlinearity=nonlinearity, a=a))
        decoder_ups.append(CNA(out_channels, out_channels-in_channels, kernel_size=3, stride=2, padding=1, deconv=True, ndim=2, nonlinearity=nonlinearity, a=a))        
        self.decoder_ups = nn.ModuleList(decoder_ups)

        decoder_completes = []
        for i in range(len(channels)):
            decoder_complete = nn.Sequential(
                                        CNA(out_channels, out_channels, kernel_size=3, stride=1, padding=1, deconv=False, ndim=2, nonlinearity=nonlinearity, a=a),
                                        CNA(out_channels, out_channels, kernel_size=3, stride=1, padding=1, deconv=False, ndim=2, nonlinearity=nonlinearity, a=a)
                                    )
            decoder_completes.append(decoder_complete)               
        
        self.decoder_completes = nn.ModuleList(decoder_completes)
        

        init_weights(self, nonlinearity='leaky_relu', a=0.1)


    def forward(self, x):        
        x = self.down(x)       
        x_enc_layers = []
        for i in range(len(self.encoders)):
            x_enc_layers.append(x)
            x = self.encoders[i](x)
            
        for i in range(len(self.decoder_ups)):
            x = self.decoder_ups[i](x, output_size = x_enc_layers[-(i+1)].shape[-2:])
            x = torch.cat([x, x_enc_layers[-(i+1)]],dim=1)            
            x = self.decoder_completes[i](x)
            
        return x


