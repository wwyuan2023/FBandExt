#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train Parallel WaveGAN."""

import os, sys
import argparse
import logging

import numpy as np
import torch
import yaml

import fbandext
import fbandext.models
from fbandext.utils import load_model


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description="Export FBandExt (See detail in fbandext/bin/export.py).")
    parser.add_argument("--outdir", type=str, required=True,
                        help="directory to save checkpoint.")
    parser.add_argument("--checkpoint", "--ckpt", nargs="+", type=str, required=True,
                        help="checkpoint files to be averaged.")
    parser.add_argument("--config", "--conf", default=None, type=str,
                        help="yaml format configuration file.")
    parser.add_argument("--verbose", type=int, default=1,
                        help="logging level. higher is more logging. (default=1)")
    args = parser.parse_args()

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG, stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        logging.warning("Skip DEBUG/INFO messages")
    
    # load config
    config_path = args.config
    if config_path is None:
        dirname = args.checkpoint if os.path.isdir(args.checkpoint) else \
            os.path.dirname(args.checkpoint)
        config_path = os.path.join(dirname, "config.yml")
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    
    # load model
    logging.info(f"Load [{args.checkpoint[0]}]")
    model = load_model(args.checkpoint[0], config)
    if len(args.checkpoint) > 1:
        avg = torch.load(args.checkpoint[0], map_location="cpu")['model']["generator"]
        for ckpt in args.checkpoint[1:]:
            logging.info(f"Load [{ckpt}] for averaging.")
            states = torch.load(ckpt, map_location="cpu")['model']["generator"]
            for k in avg.keys():
                avg[k] += states[k]
        for k in avg.keys():
            avg[k] = torch.true_divide(avg[k], len(args.checkpoint))
        model.load_state_dict(avg)
    
    # save config to outdir
    config["version"] = fbandext.__version__   # add version info
    with open(os.path.join(args.outdir, "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
    for key, value in config.items():
        logging.info(f"{key} = {value}")
    
    # print parameters
    logging.info(model)
    total_params = sum([np.prod(p.size()) for p in model.parameters()])
    logging.info(f"Total parameters: {total_params}")
    
    # save model to outdir
    checkpoint_path = os.path.join(args.outdir, "checkpoint.pth")
    state_dict = {
        "model": {"generator": model.state_dict()}
    }
    torch.save(state_dict, checkpoint_path)
    logging.info(f"Successfully export model parameters from [{args.checkpoint}] to [{checkpoint_path}].")


if __name__ == "__main__":
    main()
