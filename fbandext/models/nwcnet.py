# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fbandext.layers import Transpose


class _LightConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        conv_channels,
        conv_kernel_size,
        conv_dilation,
        conv_groups=1,
        act_func="PReLU", act_params={"num_parameters": 1, "init": 0.142},
        act_func_transition="Tanh", act_params_transition={},
        padding_mode="causal",
    ):
        super().__init__()
        assert conv_channels % conv_groups == 0
        assert padding_mode.lower() in ["valid", "same", "causal"]
        assert padding_mode.lower() != "same" or conv_kernel_size % 2 == 1
        self.conv_channels = conv_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_dilation = conv_dilation
        self.padding_mode = padding_mode.lower()
        
        p = (self.conv_kernel_size - 1) * self.conv_dilation
        if p > 0 and self.padding_mode != "valid":
            padding = [p//2, p//2] if self.padding_mode == "same" else [p, 0]
        else:
            padding = None
        
        self.convs = nn.Sequential(
            getattr(torch.nn, act_func_transition)(**act_params_transition),
            nn.Linear(in_channels, conv_channels),
            nn.LayerNorm(conv_channels),
            getattr(torch.nn, act_func)(**act_params),
            Transpose(),
            nn.ConstantPad1d(padding, 0.0) if padding is not None else nn.Identity(),
            nn.Conv1d(conv_channels, conv_channels, conv_kernel_size, dilation=conv_dilation, groups=conv_groups),
            Transpose(),
            nn.LayerNorm(conv_channels),
            getattr(torch.nn, act_func_transition)(**act_params_transition),
            nn.Linear(conv_channels, in_channels),
        )
        self.norm = nn.LayerNorm(in_channels)
    
    def forward(self, x):
        # x: (B, T, C)
        x = x + self.convs(x)
        x = self.norm(x)
        return x

class _Downsample(nn.Module):
    def __init__(self, in_channels, factor, act_func_transition="Tanh", act_params_transition={}):
        super().__init__()
        self.factor = factor
        self.conv = nn.Sequential(
            getattr(torch.nn, act_func_transition)(**act_params_transition),
            Transpose(),
            nn.Conv1d(in_channels, in_channels, factor, stride=factor),
            Transpose(),
        )
        
    def forward(self, x):
        # x: (B, T, C)
        x = self.conv(x) # (B, T//factor, C)
        return x

class _Upsample(nn.Module):
    def __init__(self, in_channels, factor, act_func_transition="Tanh", act_params_transition={}):
        super().__init__()
        self.factor = factor
        self.conv = nn.Sequential(
            getattr(torch.nn, act_func_transition)(**act_params_transition),
            Transpose(),
            nn.ConvTranspose1d(in_channels, in_channels, factor, stride=factor),
            Transpose(),
        )
        
    def forward(self, x):
        # x: (B, T, C)
        x = self.conv(x) # (B, T*factor, C)
        return x

class NWCNetEncoder(nn.Module):
    def __init__(
        self,
        code_size,
        downsample_factors,
        in_channels,
        conv_channels:list,
        conv_kernel_size:list,
        conv_dilation:list,
        conv_groups:list,
        act_func="PReLU", act_params={"num_parameters": 1, "init": 0.142},
        act_func_transition="Tanh", act_params_transition={},
        padding_mode="same",
        conv_class_name="_LightConv1d",
    ):
        super().__init__()
        assert in_channels % 2 == 0
        assert len(downsample_factors) == len(conv_channels) \
            == len(conv_kernel_size) == len(conv_dilation) == len(conv_groups)
        
        self.scale_in = nn.Sequential(
            Transpose(),
            nn.Linear(1, in_channels),
        )
        
        convnet = []
        conv_class = globals()[conv_class_name]
        for i, factor in enumerate(downsample_factors):
            convnet += [
                _Downsample(in_channels, factor, act_func_transition, act_params_transition)
            ]
            for d in conv_dilation[i]:
                convnet += [
                    conv_class(
                        in_channels,
                        conv_channels=conv_channels[i],
                        conv_kernel_size=conv_kernel_size[i],
                        conv_dilation=d,
                        conv_groups=conv_groups[i],
                        act_func=act_func, act_params=act_params,
                        act_func_transition=act_func_transition, act_params_transition=act_params_transition,
                        padding_mode=padding_mode,
                    )
                ]
        self.convnet = nn.Sequential(*convnet)
        
        self.scale_out = nn.Sequential(
            nn.Tanh(),
            nn.Linear(in_channels, code_size),
            nn.LayerNorm(code_size),
            nn.Tanh(),
            Transpose(),
        )
        
        # reset parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None: m.bias.data.fill_(0.0)

        self.apply(_reset_parameters)
        
    def forward(self, x):
        # x: (B, 1, t), audio waveform, float32, range to [-1, 1]
        x = self.scale_in(x)
        x = self.convnet(x)
        x = self.scale_out(x)
        return x # (B, C, T), T=t//r

class NWCNetDecoder(nn.Module):
    def __init__(
        self,
        code_size,
        upsample_factors,
        context_window:list,
        in_channels,
        conv_channels:list,
        conv_kernel_size:list,
        conv_dilation:list,
        conv_groups:list,
        act_func="PReLU", act_params={"num_parameters": 1, "init": 0.142},
        act_func_transition="Tanh", act_params_transition={},
        padding_mode="same",
        conv_class_name="_LightConv1d",
    ):
        super().__init__()
        assert len(upsample_factors) == len(conv_channels) \
            == len(conv_kernel_size) == len(conv_dilation) == len(conv_groups)
        self.context_window = context_window
        
        kernel_size = sum(context_window) + 1
        self.scale_in = nn.Sequential(
            nn.Conv1d(code_size, in_channels, kernel_size),
            Transpose(),
            nn.LayerNorm(in_channels),
        )
        
        convnet = []
        conv_class = globals()[conv_class_name]
        for i, factor in enumerate(upsample_factors):
            convnet += [
                _Upsample(in_channels, factor, act_func_transition, act_params_transition)
            ]
            for d in conv_dilation[i]:
                convnet += [
                    conv_class(
                        in_channels,
                        conv_channels=conv_channels[i],
                        conv_kernel_size=conv_kernel_size[i],
                        conv_dilation=d,
                        conv_groups=conv_groups[i],
                        act_func=act_func, act_params=act_params,
                        act_func_transition=act_func_transition, act_params_transition=act_params_transition,
                        padding_mode=padding_mode,
                    )
                ]
        self.convnet = nn.Sequential(*convnet)
        
        self.scale_out = nn.Sequential(
            nn.PReLU(init=0.142),
            Transpose(),
            nn.Conv1d(in_channels, 16, 17, padding=17//2),
            nn.PReLU(init=0.142),
            nn.Conv1d(16, 1, 1),
            nn.Tanh(),
        )
        
        # reset parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None: m.bias.data.fill_(0.0)

        self.apply(_reset_parameters)
    
    def forward(self, x):
        # x: (B, C, T=frame)
        x = F.pad(x, self.context_window)
        x = self.scale_in(x)
        x = self.convnet(x)
        x = self.scale_out(x)
        return x # (B, 1, t=audio length)


class NWCNet(nn.Module):
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
            "code_size": 32,
            "upsample_factors": [4, 4, 4, 4],
            "context_window": [3, 3],
            "in_channels": 64,
            "conv_channels": [256, 128, 64, 32],
            "conv_kernel_size": [5, 7, 11, 17],
            "conv_dilation": [[1,3,5], [1,3,7], [1,3,9], [1,3,9,11]],
            "conv_groups": [8, 4, 2, 1],
            "act_func": "ReLU",
            "act_params": {},
            "act_func_transition": "Tanh",
            "act_params_transition": {},
            "padding_mode": "same",
            "conv_class_name": "_LightConv1d",
        },
        use_weight_norm=False,
    ):
        super().__init__()
                
        # encoder network
        self.encoder = NWCNetEncoder(**encoder_params)
        
        # decoder network
        self.decoder = NWCNetDecoder(**decoder_params)
                
        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()
        
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


