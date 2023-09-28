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
        return_utt_id=False,
        num_repeat=1,
    ):
        """Initialize dataset.
        """
        self.sr_source = 16000
        self.sr_target = 32000
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
        r = self.sr_target // self.sr_source
        assert r == 2
        
        # normalized energy
        y /= abs(y).max()
        
        # adjust length
        if len(y) <= self.segment_size * r:
            pad = self.segment_size * r - len(y) + 256
            y = np.pad(y, [pad//2, pad-pad//2], mode='reflect')
        
        # random slice segment
        start = np.random.randint(0, len(y) - self.segment_size*r)
        y = y[start:start+self.segment_size*r]
        
        return y
    
    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_utt_id = True).
            tensor: Audio signal (T,).

        """
        wavfn = self.wavfn_list[idx]
        y = self._load_audio_segment(wavfn)
        y = torch.from_numpy(y)

        if self.return_utt_id:
            utt_id = os.path.splitext(os.path.basename(wavfn))[0]
            items = utt_id, y
        else:
            items = y

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.wavfn_list)


