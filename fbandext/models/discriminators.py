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
        use_weight_norm=True,
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
        use_weight_norm=True,
    ):
        super().__init__()
        self.num_dwt = num_dwt
        self.discriminators = nn.ModuleList([
            WaveDiscriminator(
                2**i,
                kernel_size,
                layers-i,
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
        window="hann_window",
        num_layers=4,
        kernel_size=3,
        stride=1,
        conv_channels=256,
        use_logmag=False,
        use_weight_norm=True,
    ):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        self.use_logmag = use_logmag
        
        fnorm = weight_norm if use_weight_norm else spectral_norm
        F = fft_size//2 + 1
        s0 = int(F ** (1.0 / float(num_layers)))
        s1 = stride
        k0 = s0 * 2 + 1
        k1 = kernel_size
        cc = conv_channels
        
        convs = [
            fnorm(Conv2d(1, cc, (k0,k1), stride=(s0,s1), padding=0)),
            LeakyReLU(LRELU_SLOPE, True),
        ]
        F = int((F - k0) / s0 + 1)
        for i in range(num_layers - 2):
            convs += [
                fnorm(Conv2d(cc, cc, (k0,k1), stride=(s0,s1), padding=0)),
                LeakyReLU(LRELU_SLOPE, True),
            ]
            F = int((F - k0) / s0 + 1)
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
        # x: (B, F, T), F=n_fft//2+1, T=auido_len//hop_size+1
        if self.use_logmag: x = torch.log10(x + 1e-7)
        x = x.unsqueeze(1) # (B, 1, F, T)
        x = self.convs(x) # (B, 1, F, T) -> (B, 1, 1, T')
        return x.squeeze_(1).squeeze_(2) # (B, T')


class MultiSTFTDiscriminator(nn.Module):
    def __init__(
        self,
        fft_sizes=[512, 1024, 2048],
        hop_sizes=[128, 256, 512],
        win_sizes=[512, 1024, 2048],
        window="hann_window",
        num_layers=[4, 4, 4],
        kernel_sizes=[3, 3, 3],
        conv_channels=[256, 256, 256],
        use_logmag=False,
        use_weight_norm=True,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList([
            STFTDiscriminator(
                fft_size=fft_size,
                hop_size=hop_size,
                win_size=win_size,
                window=window,
                num_layers=num_layer,
                kernel_size=kernel_size,
                conv_channels=conv_channel,
                use_weight_norm=use_weight_norm
            ) for fft_size, hop_size, win_size, num_layer, kernel_size, conv_channel in \
                zip(fft_sizes, hop_sizes, win_sizes, num_layers, kernel_sizes, conv_channels)
        ])
    
    def forward(self, xs):
        outs = []
        for x,d in zip(xs, self.discriminators):
            outs.append(d(x))
        return outs


class MultiWaveSTFTDiscriminator(nn.Module):
    def __init__(
        self,
        multi_wave_discriminator_params={
            "num_dwt": 5,
            "kernel_size": 5,
            "layers": 8,
            "conv_channels": 64,
            "use_weight_norm": True,
        },
        multi_stft_discriminator_params={
            "fft_sizes": [128, 256, 512, 1024, 2048],
            "hop_sizes": [32, 64, 128, 256, 512],
            "win_sizes": [128, 256, 512, 1024, 2048],
            "num_layers": [3, 4, 5, 6, 7],
            "kernel_sizes": [5, 5, 5, 5, 5],
            "conv_channels": [64, 64, 64, 64, 64],
            "use_weight_norm": False,
        },
        
    ):
        super().__init__()
        self.mwd = MultiWaveDiscriminator(**multi_wave_discriminator_params)
        self.mfd = MultiSTFTDiscriminator(**multi_stft_discriminator_params)
        
    def forward(self, x, s):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input signal (B, 1, T).
            s (Tensor): List of spectrum of input signal (B, F, T).

        Returns:
            Tensor: List of output tensor (B, T')

        """
        return self.mwd(x) + self.mfd(s)

