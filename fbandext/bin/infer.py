#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Decode with trained vocoder Generator."""

import os
import torch
import torch.nn.functional as F
import yaml
import numpy as np
import librosa
from scipy import signal

import fbandext
from fbandext.utils import load_model
from fbandext import __version__


def butter_highpass_filter(x, sr, order=5, cuttoff=70):
    b, a = signal.butter(order, 2*cuttoff/sr, 'highpass')
    y = signal.filtfilt(b, a, x, axis=0)
    return y


class NeuralFBandExt(object):
    """ Neural Vocals and Accompaniment Demix """
    def __init__(self, checkpoint_path=None, config_path=None, device="cpu"):
        
        if checkpoint_path is None:
            checkpoint_path = os.path.join(fbandext.__path__[0], "checkpoint", "checkpoint.pkl")
    
        # setup config
        if config_path is None:
            dirname = os.path.dirname(checkpoint_path)
            config_path = os.path.join(dirname, "config.yml")
        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.Loader)

        self.hop_size = np.prod(self.config["downsample_factors"])
        self.sampling_rate_source = self.config["sampling_rate_source"]
        self.sampling_rate_target = self.config["sampling_rate_target"]
        
        # setup device
        self.device = torch.device(device)
        
        # setup model
        model = load_model(checkpoint_path, self.config)
        model.remove_weight_norm()
        self.model = model.eval().to(self.device)
        
        # alias inference
        self.inference = self.infer
    
    @torch.no_grad()
    def infer(self, x):
        # x: (C, T), ndarray
        x = librosa.resample(x, orig_sr=self.sampling_rate_source, target_sr=self.sampling_rate_target, res_type="scipy", axis=1)
        x = torch.from_numpy(x.T) # (B=C, T)
        
        # padding
        orig_length = x.size(-1)
        if orig_length % self.hop_size > 0:
            pad_length = self.hop_size - orig_length % self.hop_size
            x = F.pad(x, (0, pad_length))
        else:
            pad_length = 0
        
        # inference
        y = self.model.infer(x.unsqueeze(1)).squeeze(1).cpu().numpy()
                
        # cut off
        if pad_length > 0:
            y = y[..., :orig_length]
        
        return y # (C, T)


def main():
    
    import argparse
    import logging
    import time
    import soundfile as sf
    
    from tqdm import tqdm
    from fbandext.utils import find_files
    
    
    """Run decoding process."""
    parser = argparse.ArgumentParser(
        description=f"Extract vocals from noise with trained Neural FBandExt Generator, version = {__version__} "
                    "(See detail in fbandext/bin/infer.py).")
    parser.add_argument("--wav-scp", "--scp", default=None, type=str,
                        help="wav.scp file. "
                             "you need to specify either wav-scp or dumpdir.")
    parser.add_argument("--dumpdir", default=None, type=str,
                        help="directory including feature files. "
                             "you need to specify either wav-scp or dumpdir.")
    parser.add_argument("--outdir", type=str, required=True,
                        help="directory to save generated speech.")
    parser.add_argument("--checkpoint", "--ckpt", default=None, type=str, 
                        help="checkpoint file to be loaded.")
    parser.add_argument("--config", "--conf", default=None, type=str,
                        help="yaml format configuration file. if not explicitly provided, "
                             "it will be searched in the checkpoint directory. (default=None)")
    parser.add_argument("--sampling-rate", "--sr", default=32000, type=int,
                        help="target sampling rate for stored wav file.")
    parser.add_argument("--highpass", default=None, type=float,
                        help="highpass filter after inference.")
    parser.add_argument("--device", default="cpu", type=str,
                        help="use cpu or cuda. (default=cpu)")
    parser.add_argument("--verbose", type=int, default=1,
                        help="logging level. higher is more logging. (default=1)")
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # setup model
    model = NeuralFBandExt(args.checkpoint, args.config, args.device)
    
    # check arguments
    if (args.wav_scp is not None and args.dumpdir is not None) or \
            (args.wav_scp is None and args.dumpdir is None):
        raise ValueError("Please specify either --dumpdir or --wav-scp.")

    # get wav files
    wav_files = dict()
    if args.dumpdir is not None:
        for filename in find_files(args.dumpdir, "*.wav"):
            utt_id = os.path.splitext(os.path.basename(filename))[0]
            wav_files[utt_id] = filename
        logging.info("From {} find {} wav files.".format(args.dumpdir, len(wav_files)))
    else:
        with open(args.wav_scp) as fid:
            for line in fid:
                line = line.strip()
                if line == "" or line[0] == "#" or line[-len(".wav"):] != ".wav": continue
                utt_id = os.path.splitext(os.path.basename(line))[0]
                wav_files[utt_id] = line
        logging.info("From {} find {} wav files.".format(args.wav_scp, len(wav_files)))
    logging.info(f"The number of wav to frequency band extend = {len(wav_files)}.")

    # start generation
    total_rtf = 0.0
    with torch.no_grad(), tqdm(wav_files.items(), desc="[fbandext]") as pbar:
        for idx, (utt_id, wavfn) in enumerate(pbar, 1):
            start = time.time()
            
            # load wav
            x, sr = sf.read(wavfn, dtype=np.float32) # x: (T, C) or (T,)
            
            # inference
            if model.sampling_rate_source < args.sampling_rate <= model.sampling_rate_target:
                if sr != model.sampling_rate_source:
                    x = librosa.resample(x, orig_sr=sr, target_sr=model.sampling_rate_source, res_type="scipy", axis=0)
                    sr = model.sampling_rate_source
                x /= abs(x).max()
                x = x.T if x.ndim > 1 else np.expand_dims(x, axis=1) # (C, T)
                x = model.infer(x) # (B=C, T)
                x = x.T if x.shape[0] > 1 else x.flatten() # (T, C) or (T,)
                sr = model.sampling_rate_target
            
            if args.sampling_rate != sr:
                x = librosa.resample(x, orig_sr=sr, target_sr=args.sampling_rate, res_type="scipy", axis=0)
            
            if args.highpass is not None:
                x = butter_highpass_filter(x, args.sampling_rate, cuttoff=args.highpass)
            
            # save wav
            sf.write(os.path.join(args.outdir, f"{utt_id}.wav"), x, args.sampling_rate, "PCM_16")
            
            rtf = (time.time() - start) / (len(x) / args.sampling_rate)
            pbar.set_postfix({"RTF": rtf})
            total_rtf += rtf

    # report average RTF
    logging.info(f"Finished generation of {idx} utterances (RTF = {total_rtf / idx:.03f}).")


if __name__ == "__main__":
    main()
