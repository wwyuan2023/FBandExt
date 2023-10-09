# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fbandext.layers import ChannelNorm


class _RNNBlock(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        dropout=0,
    ):
        super().__init__()
        self.dropout = dropout
        self.skiper = nn.Linear(input_size, input_size)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, input_size)
        self.norm = nn.LayerNorm(input_size)
        
    def forward(self, x):
        # x: (B, T, C)
        x_ =self.skiper(x)
        x, _ = self.rnn(x)
        x = F.dropout(self.fc(x), self.dropout)
        x = self.norm(x + x_)
        return x # (B, T, C)

class _Downsample(nn.Module):
    def __init__(self, input_size, factor):
        super().__init__()
        assert input_size % factor == 0
        self.input_size = input_size
        self.factor = factor
        self.fc = nn.Linear(input_size, input_size//factor)
        self.norm = nn.LayerNorm(input_size)
        
    def forward(self, x):
        # x: (B, T, C)
        x = self.fc(x).view(x.size(0), -1, self.input_size) # (B, T//factor, C)
        x = self.norm(x)
        return x

class _Upsample(nn.Module):
    def __init__(self, input_size, factor):
        super().__init__()
        self.input_size = input_size
        self.factor = factor
        self.fc = nn.Linear(input_size, input_size*factor)
        self.norm = nn.LayerNorm(input_size)
        
    def forward(self, x):
        # x: (B, T, C)
        x = self.fc(x).view(x.size(0), -1, self.input_size) # (B, T*factor, C)
        x = self.norm(x)
        return x

class _REBlock(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        downsample_factor,
        dropout=0,
    ):
        super().__init__()
        self.blocks = _RNNBlock(input_size, hidden_size, dropout)
        self.downsample = _Downsample(input_size, downsample_factor)
    
    def forward(self, x):
        # x: (B, T, C)
        e = self.blocks(x) # (B, T, C)
        x = self.downsample(e) # (B, C, T//factor)
        return x, e


class _RDBlock(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        upsample_factor,
        dropout=0,
    ):
        super().__init__()
        self.upsample = _Upsample(input_size, upsample_factor)
        self.blocks = _RNNBlock(input_size, hidden_size, dropout)
    
    def forward(self, x, e):
        # x: (B, T, C)
        # e: (B, T*factor, C)
        x = self.upsample(x) # (B, T*factor, C)
        x = self.blocks(x + e) # (B, T*factor, C)
        return x


class ResRNNUNet(nn.Module):
    def __init__(
        self,
        fft_size=1024,
        hop_size=256,
        downsample_factors=[2, 2, 2, 2],
        use_weight_norm=True,
    ):
        super().__init__()
        assert len(downsample_factors) == 4
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.downsample_factors = downsample_factors
        
        self.register_buffer('window1', torch.hann_window(fft_size), persistent=False)
        self.register_buffer('window2', torch.hann_window(fft_size*2), persistent=False)
                        
        # prenet
        input_size = fft_size // 2
        self.prenet = nn.Sequential(
            nn.Linear(input_size-16, input_size),
            nn.LayerNorm(input_size)
        )
        
        # encoder
        self.encoder1 = _REBlock(input_size,  64, downsample_factors[0], 0.1)
        self.encoder2 = _REBlock(input_size, 128, downsample_factors[1], 0.1)
        self.encoder3 = _REBlock(input_size, 192, downsample_factors[2], 0.1)
        self.encoder4 = _REBlock(input_size, 256, downsample_factors[3], 0.1)
        
        # bottleneck
        self.bottle = _RNNBlock(input_size, input_size, 0.25)
        
        # decoder
        self.decoder4 = _RDBlock(input_size, 256, downsample_factors[3], 0.1)
        self.decoder3 = _RDBlock(input_size, 192, downsample_factors[2], 0.1)
        self.decoder2 = _RDBlock(input_size, 128, downsample_factors[1], 0.1)
        self.decoder1 = _RDBlock(input_size,  64, downsample_factors[0], 0.1)
        
        # postnet
        self.postnet = nn.Linear(input_size, (fft_size//2+16)*2)
                
        # reset parameters
        self.reset_parameters()
        
        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()
    
    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                #nn.init.normal_(m.weight, 0.0, 0.2)
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

        self.apply(_reset_parameters)
        
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
    
    def forward(self, y):
        # y: target signal 32kHz sampling rate, shape=(B, t), t=audio length
        
        # stft
        sy = torch.stft(y,
            n_fft=self.fft_size*2, hop_length=self.hop_size*2, 
            win_length=self.fft_size*2, window=self.window2, return_complex=False)[:, :-1] # (B, F-1, T, 2)
        sx = sy[:, :self.fft_size//2-16]
        mx = torch.norm(sx, p=2, dim=-1).transpose(1, 2) # (B, T, C)
                        
        # prenet
        B, T, _ = mx.size()
        x0 = self.prenet(mx)
        
        # encoder
        x1, e1 = self.encoder1(x0)
        x2, e2 = self.encoder2(x1)
        x3, e3 = self.encoder3(x2)
        x4, e4 = self.encoder4(x3)
        
        # bottleneck
        x4 = self.bottle(x4)
        
        # decoder
        x3 = self.decoder4(x4, e4)
        x2 = self.decoder3(x3, e3)
        x1 = self.decoder2(x2, e2)
        x0 = self.decoder1(x1, e1)
        
        # postnet
        x0 = self.postnet(x0).reshape(B, T, -1 ,2).transpose(1, 2) # (B, F-1, T, 2)
        
        # istft
        sy_hat = F.pad(sx, [0, 0, 0, 0, 0, 1+self.fft_size//2+16])
        sy_hat[:, self.fft_size//2-16:-1] = x0
        y_hat = torch.istft(torch.complex(sy_hat[..., 0], sy_hat[..., 1]),
            n_fft=self.fft_size*2, hop_length=self.hop_size*2,
            win_length=self.fft_size*2, window=self.window2, return_complex=False)
        
        # focus only on the high frequencies
        sy, sy_hat = sy[:, self.fft_size//2-16:], sy_hat[:, self.fft_size//2-16:-1]
        
        # loss
        spec_loss = F.mse_loss(sy_hat, sy)
        m, m_hat = torch.norm(sy, p=2, dim=-1), torch.norm(sy_hat, p=2, dim=-1)
        mag_loss = F.mse_loss(m_hat, m) + F.l1_loss(torch.log(m_hat + 1e-5), torch.log(m + 1e-5))
        
        return y_hat, spec_loss, mag_loss
    
    @torch.no_grad()
    def _infer(self, x):
        # x: source signal 16kHz sampling rate, shape=(B, t), t=audio length
        
        # stft
        sx = torch.stft(x,
            n_fft=self.fft_size, hop_length=self.hop_size, 
            win_length=self.fft_size, window=self.window1, return_complex=False)[:, :-1] # (B, F-1, T, 2)
        sx = sx[:, :self.fft_size//2-16]
        mx = torch.norm(sx, p=2, dim=-1).transpose(1, 2) # (B, T, C)
        
        # prenet
        B, T, _ = mx.size()
        x0 = self.prenet(mx)
        
        # encoder
        x1, e1 = self.encoder1(x0)
        x2, e2 = self.encoder2(x1)
        x3, e3 = self.encoder3(x2)
        x4, e4 = self.encoder4(x3)
        
        # bottleneck
        x4 = self.bottle(x4)
        
        # decoder
        x3 = self.decoder4(x4, e4)
        x2 = self.decoder3(x3, e3)
        x1 = self.decoder2(x2, e2)
        x0 = self.decoder1(x1, e1)
        
        # postnet
        x0 = self.postnet(x0).reshape(B, T, -1 ,2).transpose(1, 2) # (B, F-1, T, 2)
        
        # istft
        sy_hat = F.pad(sx, [0, 0, 0, 0, 0, 1+self.fft_size//2+16])
        sy_hat[:, self.fft_size//2-16:-1] = x0
        y_hat = torch.istft(torch.complex(sy_hat[..., 0], sy_hat[..., 1]),
            n_fft=self.fft_size*2, hop_length=self.hop_size*2,
            win_length=self.fft_size*2, window=self.window2, return_complex=False)
        assert x.size(1) * 2 == y_hat.size(1), f"x.size={x.size()}, y_hat.size={y_hat.size()}\n"
        
        return y_hat # (B, t*2)
    
    @torch.no_grad()
    def infer(self, x, segment_length=48896, overlap_length=16000):
        # x: source signal 16kHz sampling rate, shape=(B, t), t=audio length
        assert (segment_length // self.hop_size + 1) % np.prod(self.downsample_factors) == 0
        hop_length = segment_length - overlap_length
        
        # split segments
        orig_length = x.size(1)
        start, end = 0, segment_length
        segments = []
        while end <= x.size(1):
            segment = x[:, start:end]
            start = start + hop_length
            end = end + hop_length
            if segment.size(1) < segment_length:
                pad = segment_length - segment.size(1)
                segment = F.pad(segment, [0, pad])
            segments.append(segment)
        xs = torch.stack(segments, dim=0) # (S, B, L)
        
        # inference
        S, B, L = xs.size()
        xs_hat = torch.zeros(S, B, L*2, device=xs.device)
        for i in range(xs.size(0)):
            xs_hat[i] = self._infer(xs[i]) # (B, L) -> (B, L*2)
        
        # combine segments
        segment_length, hop_length, overlap_length = segment_length*2, hop_length*2, overlap_length*2
        window = torch.hann_window(overlap_length*2).to(xs.device)
        inc_window, dec_window = window[:overlap_length].unsqueeze(0), window[overlap_length:].unsqueeze(0)
        output = torch.zeros(B, S*L*2, device=xs.device)
        output[:, :segment_length] = xs_hat[0]
        start = hop_length
        for i in range(1, xs_hat.size(0)):
            output[:, start:start+overlap_length] *= dec_window
            xs_hat[i, :, :overlap_length] *= inc_window
            output[:, start:start+segment_length] += xs_hat[i]
            start += hop_length
        y = output[:, :orig_length*2]
        
        return y


