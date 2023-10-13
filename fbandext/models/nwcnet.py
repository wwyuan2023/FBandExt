# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fbandext.layers import PQMF
from fbandext.layers import ChannelNorm


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
            nn.Conv1d(in_channels, conv_channels, 1),
            ChannelNorm(conv_channels),
            getattr(torch.nn, act_func)(**act_params),
            nn.ConstantPad1d(padding, 0.0) if padding is not None else nn.Identity(),
            nn.Conv1d(conv_channels, conv_channels, conv_kernel_size, dilation=conv_dilation, groups=conv_groups),
            ChannelNorm(conv_channels),
            getattr(torch.nn, act_func_transition)(**act_params_transition),
            nn.Conv1d(conv_channels, in_channels, 1),
        )
        self.norm = ChannelNorm(in_channels)
    
    def forward(self, x):
        # x: (B, C, T)
        x = x + self.convs(x)
        x = self.norm(x)
        return x

class _Downsample(nn.Module):
    def __init__(self, in_channels, factor, act_func_transition="Tanh", act_params_transition={}):
        super().__init__()
        self.factor = factor
        self.conv = nn.Sequential(
            getattr(torch.nn, act_func_transition)(**act_params_transition),
            nn.Conv1d(in_channels, in_channels, factor, stride=factor),
        )
        
    def forward(self, x):
        # x: (B, C, T)
        x = self.conv(x) # (B, C, T//factor)
        return x

class _Upsample(nn.Module):
    def __init__(self, in_channels, factor, act_func_transition="Tanh", act_params_transition={}):
        super().__init__()
        self.factor = factor
        self.conv = nn.Sequential(
            getattr(torch.nn, act_func_transition)(**act_params_transition),
            nn.ConvTranspose1d(in_channels, in_channels, factor, stride=factor),
        )
        
    def forward(self, x):
        # x: (B, C, T)
        x = self.conv(x) # (B, C, T*factor)
        return x

class Encoder(nn.Module):
    def __init__(
        self,
        downsample_factors,
        code_size,
        code_bits,
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
        self.frame_size = np.prod(downsample_factors)
        self.downsample_factors = downsample_factors
        self.code_size = code_size
        self.code_bits = code_bits
        
        self.scale_in = nn.Conv1d(1, in_channels, 1)
        
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
            nn.Conv1d(in_channels, code_size, 1),
            ChannelNorm(code_size) if code_size > 1 else nn.Identity(),
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
    
    def quantize(self, x, bits=7):
        Q = 2.0 ** bits
        q = (torch.clamp(x, min=-1, max=0.999) * Q).int()
        f = q.detach().float() / Q
        x = x + (f - x).detach()
        q = (q + Q).long() # [0, 2**(b+1)-1), eg [0, 255) if code_bits==8
        return x, q
    
    def forward(self, x):
        # x: (B, 1, t), audio waveform, float32, range to [-1, 1]
        x = self.scale_in(x)
        x = self.convnet(x)
        x = self.scale_out(x)
        
        e, q = self.quantize(x, self.code_bits - 1) # (B, code_size, t)
        return e, q

class Decoder(nn.Module):
    def __init__(
        self,
        upsample_factors,
        code_size,
        code_bits,
        context_window:list,
        embed_size, 
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
        self.frame_size = np.prod(upsample_factors)
        self.upsample_factors = upsample_factors
        self.code_size = code_size
        self.code_bits = code_bits
        self.Q = int(2**(code_bits-1))
        self.context_window = context_window
        self.kernel_size = sum(context_window) + 1
        self.embed_size = embed_size
        self.in_channels = in_channels
        
        self.embedding = nn.Embedding(int(2**code_bits), embed_size//code_size)
        self.conv = nn.Sequential(
            nn.Conv1d(code_size, embed_size, self.kernel_size, bias=False),
            ChannelNorm(embed_size),
            nn.ReLU(),
        )
        
        self.scale_in = nn.Sequential(
            nn.Conv1d(embed_size, in_channels, 1),
            ChannelNorm(in_channels),
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
            nn.Conv1d(in_channels, 32, 17, padding=17//2),
            nn.PReLU(init=0.142),
            nn.Conv1d(32, 2, 1),
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
        # x: (B, C, T)
        B, _, T = x.size()
        
        # embedding
        q = torch.clamp(x, min=-1, max=0.999) * self.Q # (B, C, T)
        q = q.long() + self.Q # [-Q, Q) -> [0, 2*Q)
        qe = self.embedding(q).transpose(2, 3).reshape(B, self.embed_size, T) # (B, E, T)
        
        x = F.pad(x, self.context_window)
        xe = self.conv(x) # (B, E, T)
        x = xe + qe
        
        # convs
        x = self.scale_in(x)
        x = self.convnet(x)
        x = self.scale_out(x) # (B, 2, T*self.frame_size)
        
        return x

class NWCNet(nn.Module):
    def __init__(
        self,
        sampling_rate=16000,
        downsample_factors=(4, 4, 4, 4),
        upsample_factors=(4, 4, 4, 4),
        code_size=20,
        code_bits=8,
        context_window=(2,2),
        encoder_params={
            "in_channels": 64,
            "conv_channels": [32, 64, 128, 256],
            "conv_kernel_size": [9, 7, 7, 1],
            "conv_dilation": [[1,], [1,], [1,], [1,]],
            "conv_groups": [1, 2, 4, 8],
            "act_func": "ReLU",
            "act_params": {},
            "act_func_transition": "Tanh",
            "act_params_transition": {},
            "padding_mode": "same",
            "conv_class_name": "_LightConv1d",
        },
        decoder_params={
            "embed_size": 640,
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
        pqmf_params={
            "subbands": 2,
            "taps": 14,
            "beta": 8.0,
        },
        use_weight_norm=False,
    ):
        super().__init__()
        assert all([ x>=0 for x in context_window])
        self.sampling_rate = sampling_rate
        self.code_size = code_size
        self.code_bits = code_bits
        self.context_window = context_window
                
        # encoder network
        self.encoder = Encoder(downsample_factors, code_size, code_bits, **encoder_params)
        
        # decoder network
        self.decoder = Decoder(upsample_factors, code_size, code_bits, context_window, **decoder_params)
        
        # pqmf
        self.pqmf = PQMF(**pqmf_params)
        
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
    
    def forward(self, x, steps=0):
        # x: (B, c=1, T), c=audio channel, T=audio length
        
        # encoder
        e, _ = self.encoder(x) # e: (B, E, T)
        
        # decoder
        x_hat = self.decoder(e) # (B, 2, T)
        
        # statistic 
        var, mean = torch.var_mean(e.reshape(-1), dim=0)
        
        # synthesis
        x = F.pad(x, [0, 0, 0, 1])
        x_hat = x_hat + x
        y_hat = self.pqmf.synthesis(x_hat)
        
        return y_hat, (mean, var)
    
    @torch.no_grad()
    def infer(self, x):
        # x: (B, c=1, T), c=audio channel, T=audio length
        
        # encoder
        e, _ = self.encoder(x) # e: (B, E, T)
        
        # decoder
        x_hat = self.decoder(e) # (B, 1, T)
        
        # synthesis
        x_hat[:, 0:1] += x
        y_hat = self.pqmf.synthesis(x_hat)
        
        return y_hat


