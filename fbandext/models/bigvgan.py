# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from fbandext.layers import SnakeAlias
from .nwcnet import NWCNetEncoder


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def init_weights(m, mean=0.0, std=0.01):
    if isinstance(m, nn.Conv1d):
        m.weight.data.normal_(mean, std)

class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilations=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(
                nn.Conv1d(
                    channels, channels,
                    kernel_size, 1, dilation=dilation, 
                    padding=get_padding(kernel_size, dilation),
                )
            ) for dilation in dilations
        ])
        self.convs1.apply(init_weights)
        
        self.convs2 = nn.ModuleList([
            weight_norm(
                nn.Conv1d(
                    channels, channels,
                    kernel_size, 1, dilation=1,
                    padding=get_padding(kernel_size, 1),
                )
            ) for _ in dilations
        ])
        self.convs2.apply(init_weights)
        
        self.acts1 =nn.ModuleList([
            SnakeAlias(channels) for _ in range(len(self.convs1))
        ])
        
        self.acts2 =nn.ModuleList([
            SnakeAlias(channels) for _ in range(len(self.convs2))
        ])
        
        self.infer = self.forward

    def forward(self, x):
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, self.acts1, self.acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x
        return x


class BigVGANDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        upsample_rates=(8, 4, 4, 4),
        context_window=[3, 3],
        upsample_initial_channel=512,
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1,3,5], [1,3,5], [1,3,5]],
    ):
        super().__init__()
        self.num_upsamples = len(upsample_rates)
        self.num_kernels = len(resblock_kernel_sizes)
        self.context_window = context_window
        
        # pre conv
        kernel_size = sum(context_window) + 1
        self.conv_pre = weight_norm(nn.Conv1d(in_channels, upsample_initial_channel, kernel_size))
        
        # transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, u in enumerate(upsample_rates):
            k = u * 2
            self.ups.append(
                weight_norm(nn.ConvTranspose1d(
                    upsample_initial_channel // (2 ** i),
                    upsample_initial_channel // (2 ** (i + 1)),
                    k, u, padding=(k - u) // 2
                ))
            )
        self.ups.apply(init_weights)
        
        # residual blocks using anti-aliased multi-periodicity composition modules
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock1(ch, k, d))
        
        # post conv
        self.conv_post = nn.Sequential(
            SnakeAlias(ch),
            weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3)),
        )
        
        # weight initialization
        self.conv_pre.apply(init_weights)
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
    
    def forward(self, x):
        # x: (B, C, T=frame)
        x = F.pad(x, self.context_window)
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = self.ups[i](x)
            xs = 0
            for j in range(self.num_kernels):
                xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = self.conv_post(x)
        return x # (B, 1, t=audio length)


class BigVGAN(nn.Module):
    def __init__(
        self,
        encoder_params={
            "code_size": 32,
            "downsample_factors": [4, 4, 4, 4],
            "in_channels": 64,
            "conv_channels": [32, 64, 128, 256],
            "conv_kernel_size": [9, 7, 5, 3],
            "conv_dilation": [[1,1,1], [1,1,1], [1,1,1], [1,1,1]],
            "conv_groups": [1, 2, 4, 8],
            "act_func": "ReLU",
            "act_params": {},
            "act_func_transition": "Tanh",
            "act_params_transition": {},
            "padding_mode": "same",
            "conv_class_name": "_LightConv1d",
        },
        decoder_params={
            "in_channels": 32,
            "upsample_rates": [4, 4, 4, 4],
            "context_window": [3, 3],
            "upsample_initial_channel": 512,
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],
        },
    ):
        super().__init__()
        
        # encoder network
        self.encoder = NWCNetEncoder(**encoder_params)
        
        # decoder network
        self.decoder = BigVGANDecoder(**decoder_params)
    
    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)
    
    def forward(self, x):
        # x: (B, c=1, t), c=audio channel, t=audio length
        
        # encoder
        e = self.encoder(x) # e: (B, E, T), T=frame
        
        # decoder
        x_hat = self.decoder(e) # (B, 1, t), t=audio length
        y_hat = x_hat + x
        
        # statistic 
        var, mean = torch.var_mean(e.reshape(-1), dim=0)
        
        return y_hat, (mean, var)
    
    @torch.no_grad()
    def infer(self, x):
        # x: (B, c=1, t), c=audio channel, t=audio length
        
        # encoder
        e = self.encoder(x) # e: (B, E, T), T=frame
        
        # decoder
        x_hat = self.decoder(e) # (B, 1, t), t=audio length
        y_hat = x_hat + x
                
        return y_hat


