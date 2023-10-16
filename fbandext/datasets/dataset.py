# -*- coding: utf-8 -*-

import os
import logging
import numpy as np
import torch, torchaudio
import soundfile as sf
import librosa

from torch.utils.data import Dataset


def _load_scpfn(filename):
    scplist = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                scplist.append(line)
    return scplist

def _filter_audiofile(filename, sampling_rate_needed):
    if filename == "": return False
    meta = torchaudio.info(filename)
    if meta.sample_rate != sampling_rate_needed:
        return False
    return True

def _repeat_list(_list, num_repeat=1):
    relist = []
    for _ in range(num_repeat):
        for p in _list:
            relist.append(p)
    return relist


class AudioSCPDataset(Dataset):
    """PyTorch compatible audio dataset based on scp files."""

    def __init__(
        self,
        wav_scpfn,
        segment_size,
        sampling_rate_source=16000,
        sampling_rate_target=32000,
        return_utt_id=False,
        num_repeat=1,
    ):
        """Initialize dataset.
        """
        self.sr_source = sampling_rate_source
        self.sr_target = sampling_rate_target
        self.r = sampling_rate_target / sampling_rate_source
        self.segment_size = int(segment_size)
        
        wavfn_list = []
        ndrop = 0
        for wavfn in _load_scpfn(wav_scpfn):
            if _filter_audiofile(wavfn, self.sr_target):
                wavfn_list.append(wavfn)
                continue
            logging.warning(f"Drop files, vocal file={wavfn}")
            ndrop += 1
        logging.warning(f"Vocal: {ndrop} files are dropped, and ({len(wavfn_list)}) files are loaded.")
        
        self.wavfn_list = _repeat_list(wavfn_list, num_repeat)
        self.return_utt_id = return_utt_id
    
    def _load_audio_segment(self, filename):
        # load pcm
        y, sr = sf.read(filename, dtype='float32')
        assert sr == self.sr_target
        
        # normalized energy
        y /= abs(y).max()
        
        # adjust length
        if len(y) <= self.segment_size:
            pad = self.segment_size - len(y) + 256
            y = np.pad(y, [pad//2, pad-pad//2], mode='reflect')
        
        # random slice segment
        start = np.random.randint(0, len(y) - self.segment_size)
        y = y[start:start+self.segment_size]
        
        # downsample
        res_types = ['soxr_mq', 'soxr_hq', 'soxr_vhq', 'scipy', 'scipy', 'scipy']
        x = librosa.resample(y, orig_sr=self.sr_target, target_sr=self.sr_source, res_type=res_types[np.random.randint(len(res_types))], axis=0)
        x = np.clip(x, -1., 1.)
        assert len(x) * self.r == len(y), f"len(x)={len(x)}, len(y)={len(y)}\n"
        x = librosa.resample(x, orig_sr=self.sr_source, target_sr=self.sr_target, res_type="scipy", axis=0)
        x = np.clip(x, -1., 1.)
        assert len(x) == len(y), f"len(x)={len(x)}, len(y)={len(y)}\n"
        
        return x, y
    
    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_utt_id = True).
            tensor: source audio signal (1, T).
            tensor: target audio signal (1, T).

        """
        wavfn = self.wavfn_list[idx]
        x, y = self._load_audio_segment(wavfn)
        x = torch.from_numpy(x).unsqueeze_(0)
        y = torch.from_numpy(y).unsqueeze_(0)

        if self.return_utt_id:
            utt_id = os.path.splitext(os.path.basename(wavfn))[0]
            items = utt_id, x, y
        else:
            items = x, y

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.wavfn_list)


