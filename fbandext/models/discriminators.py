# -*- coding: utf-8 -*-

import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Conv1d, Conv2d, LeakyReLU
from torch.nn.utils import weight_norm, spectral_norm


LRELU_SLOPE = 0.2

class WaveDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels,
        kernel_size=5,
        layers=10,
        conv_channels=64,
        use_weight_norm=False,
    ):
        super().__init__()
        
        fnorm = weight_norm if use_weight_norm else spectral_norm
        convs = [
            fnorm(Conv1d(in_channels, conv_channels, 1, padding=0, dilation=1)),
            LeakyReLU(LRELU_SLOPE), 
        ]
        for i in range(layers - 2):
            convs += [
                fnorm(Conv1d(conv_channels, conv_channels, kernel_size, padding=0, dilation=i+2)),
                LeakyReLU(LRELU_SLOPE), 
            ]
        convs += [ 
            fnorm(Conv1d(conv_channels, 1, 1, padding=0, dilation=1))
        ]
        self.convs = nn.Sequential(*convs)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, Conv1d) or isinstance(m, Conv2d):
                nn.init.xavier_uniform_(
                    m.weight,
                    gain=nn.init.calculate_gain("leaky_relu", LRELU_SLOPE)
                )
                if m.bias is not None: m.bias.data.fill_(0.0)

        self.apply(_reset_parameters)
        
    def forward(self, x):
        return self.convs(x).squeeze(1)


class MultiWaveDiscriminator(nn.Module):
    def __init__(
        self,
        num_dwt=5,
        kernel_size=5,
        layers=10,
        conv_channels=64,
        use_weight_norm=False,
    ):
        super().__init__()
        self.num_dwt = num_dwt
        self.discriminators = nn.ModuleList([
            WaveDiscriminator(
                2**i,
                kernel_size,
                layers,
                conv_channels+i*32,
                use_weight_norm=use_weight_norm
            ) for i in range(num_dwt)
        ])

    def forward(self, x):
        outs = []
        for i, d in enumerate(self.discriminators, 1):
            outs.append(d(x))
            if i == self.num_dwt: break
            b, c, t = x.shape
            period = 2**i
            if t % period != 0: # pad first
                n_pad = period - (t % period)
                x = F.pad(x, (0, n_pad), "reflect")
                t = t + n_pad
            x = x.view(b, period, -1)
        return outs


class STFTDiscriminator(nn.Module):
    def __init__(
        self,
        fft_size=1024,
        hop_size=256,
        win_size=1024,
        num_layers=4,
        kernel_size=3,
        stride=1,
        conv_channels=256,
        use_weight_norm=False,
    ):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        self.register_buffer('window', torch.hann_window(win_size), persistent=False)
        
        fnorm = weight_norm if use_weight_norm else spectral_norm
        F = fft_size//4 + 1 # !!! only focus on high frequency 
        s0 = int(F ** (1.0 / float(num_layers)))
        s1 = stride
        k0 = s0 * 2 + 1
        k1 = kernel_size
        cc = conv_channels
        
        convs = [
            fnorm(Conv2d(1, cc, (k0,k1), stride=(s0,s1), padding=0)),
            LeakyReLU(LRELU_SLOPE),
        ]
        F = int((F - k0) / s0 + 1)
        assert F > 0, f"fft_size={fft_size}, F={F}"
        for i in range(num_layers - 2):
            convs += [
                fnorm(Conv2d(cc, cc, (k0,k1), stride=(s0,s1), padding=0)),
                LeakyReLU(LRELU_SLOPE),
            ]
            F = int((F - k0) / s0 + 1)
            assert F > 0, f"fft_size={fft_size}, i={i}, F={F}"
        convs += [
            fnorm(Conv2d(cc, 1, (F,1), stride=(1,1), padding=0)),
        ]
        
        self.convs = nn.Sequential(*convs)
        
        # apply reset parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, Conv1d) or isinstance(m, Conv2d):
                nn.init.xavier_uniform_(
                    m.weight,
                    gain=nn.init.calculate_gain("leaky_relu", LRELU_SLOPE)
                )
                if m.bias is not None: m.bias.data.fill_(0.0)

        self.apply(_reset_parameters)
        
    def forward(self, x):
        # x: (B, t)
        x = torch.stft(x,
            n_fft=self.fft_size, hop_length=self.hop_size, 
            win_length=self.win_size, window=self.window, return_complex=False)  # (B, F, T, 2)
        x = x[:, self.fft_size//4:] # (B, F//2+1, T, 2), only focus on high frequency !!!
        x = torch.norm(x, p=2, dim=-1) # (B, F//2+1, T)
        x = x.unsqueeze(1) # (B, 1, F//2+1, T)
        x = self.convs(x) # (B, 1, F//2+1, T) -> (B, 1, 1, T')
        return x.squeeze() # (B, T')


class MultiSTFTDiscriminator(nn.Module):
    def __init__(
        self,
        fft_sizes=[128, 256, 512, 1024],
        hop_sizes=[32, 64, 128, 256],
        win_sizes=[128, 256, 512, 1024],
        num_layers=[5, 6, 7, 8],
        kernel_sizes=[5, 5, 5, 5],
        conv_channels=[64, 64, 64, 64],
        use_weight_norm=False,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList([
            STFTDiscriminator(
                fft_size=fft_size,
                hop_size=hop_size,
                win_size=win_size,
                num_layers=num_layer,
                kernel_size=kernel_size,
                conv_channels=conv_channel,
                use_weight_norm=use_weight_norm
            ) for fft_size, hop_size, win_size, num_layer, kernel_size, conv_channel in \
                zip(fft_sizes, hop_sizes, win_sizes, num_layers, kernel_sizes, conv_channels)
        ])
    
    def forward(self, x):
        return [d(x) for d in self.discriminators]


class MultiWaveSTFTDiscriminator(nn.Module):
    def __init__(
        self,
        multi_wave_discriminator_params={
            "num_dwt": 5,
            "kernel_size": 5,
            "layers": 8,
            "conv_channels": 64,
            "use_weight_norm": False,
        },
        multi_stft_discriminator_params={
            "fft_sizes": [128, 256, 512, 1024, 2048],
            "hop_sizes": [32, 64, 128, 256, 512],
            "win_sizes": [128, 256, 512, 1024, 2048],
            "num_layers": [3, 4, 5, 6, 7],
            "kernel_sizes": [5, 5, 5, 5, 5],
            "conv_channels": [128, 128, 128, 128, 128],
            "use_weight_norm": False,
        },
        
    ):
        super().__init__()
        self.mwd = MultiWaveDiscriminator(**multi_wave_discriminator_params)
        self.mfd = MultiSTFTDiscriminator(**multi_stft_discriminator_params)
        
    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input signal (B, t).

        Returns:
            Tensor: List of output tensor.

        """
        return self.mwd(x.unsqueeze(1)) + self.mfd(x)

