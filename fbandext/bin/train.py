#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train Model."""

import os
import sys
import argparse
import logging

import matplotlib
import numpy as np
import soundfile as sf
import yaml
import torch
import torch.nn as nn

from collections import defaultdict
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import fbandext
import fbandext.models
import fbandext.optimizers
import fbandext.lr_scheduler

from fbandext.datasets import AudioSCPDataset
from fbandext.layers import MultiResolutionSTFTLoss
from fbandext.utils import eval_sdr

# set to avoid matplotlib error in CLI environment
matplotlib.use("Agg")


class Trainer(object):
    """Customized trainer module for Parallel WaveGAN training."""

    def __init__(
        self,
        steps,
        epochs,
        data_loader,
        sampler,
        model,
        criterion,
        optimizer,
        scheduler,
        config,
        device=torch.device("cpu"),
    ):
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            data_loader (dict): Dict of data loaders. It must contrain "train" and "dev" loaders.
            model (dict): Dict of models. It must contrain "generator" models.
            criterion (dict): Dict of criterions. It must contrain "ce" criterions.
            optimizer (dict): Dict of optimizers. It must contrain "generator" optimizer.
            scheduler (dict): Dict of schedulers. It must contrain "generator" scheduler.
            config (dict): Config dict loaded from yaml format configuration file.
            device (torch.deive): Pytorch device instance.

        """
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.sampler = sampler
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.writer = SummaryWriter(config["outdir"])
        self.finish_train = False
        self.total_train_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)
        
        stft_params = self.config["stft_params"]
        self.fft_sizes = stft_params["fft_sizes"]
        self.hop_sizes = stft_params["hop_sizes"]
        self.win_sizes = stft_params["win_sizes"]
        self.windows = [
            getattr(torch, stft_params.get("window", "hann_window"))(win_size).to(self.device) for win_size in self.win_sizes
        ]

    def run(self):
        """Run training."""
        while True:
            # train one epoch
            self._train_epoch()

            # check whether training is finished
            if self.finish_train:
                break

        logging.info("Finished training.")

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be saved.

        """
        state_dict = {
            "optimizer": {
                "generator": self.optimizer["generator"].state_dict(),
                "discriminator": self.optimizer["discriminator"].state_dict(),
            },
            "scheduler": {
                "generator": self.scheduler["generator"].state_dict(),
                "discriminator": self.scheduler["discriminator"].state_dict(),
            },
            "steps": self.steps,
            "epochs": self.epochs,
        }
        if self.config["distributed"]:
            state_dict["model"] = {
                "generator": self.model["generator"].module.state_dict(),
                "discriminator": self.model["discriminator"].module.state_dict(),
            }
        else:
            state_dict["model"] = {
                "generator": self.model["generator"].state_dict(),
                "discriminator": self.model["discriminator"].state_dict(),
            }

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if self.config["distributed"]:
            self.model["generator"].module.load_state_dict(state_dict["model"]["generator"])
            self.model["discriminator"].module.load_state_dict(state_dict["model"]["discriminator"])
        else:
            self.model["generator"].load_state_dict(state_dict["model"]["generator"])
            self.model["discriminator"].load_state_dict(state_dict["model"]["discriminator"])
        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer["generator"].load_state_dict(state_dict["optimizer"]["generator"])
            self.optimizer["discriminator"].load_state_dict(state_dict["optimizer"]["discriminator"])
            self.scheduler["generator"].load_state_dict(state_dict["scheduler"]["generator"])
            self.scheduler["discriminator"].load_state_dict(state_dict["scheduler"]["discriminator"])
    
    def _stfts(self, x):
        # x: (B, 1, t)
        x = x.squeeze(1)
        
        # stft
        specs, mags = [], []
        for fft_size, hop_size, win_size, window in zip(self.fft_sizes, self.hop_sizes, self.win_sizes, self.windows):
            spec = torch.stft(x, 
                n_fft=fft_size, hop_length=hop_size, win_length=win_size, window=window,
                center=True, pad_mode='reflect', return_complex=False) # (B, F, T, 2); F=n_fft//2+1, T=t//hop_size+1
            real, imag = spec[..., 0], spec[..., 1]
            # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
            mag = torch.sqrt(torch.clamp(real**2 + imag**2, min=1e-7)) # (B, F, T)
            specs.append(spec) # (B, F, T, 2)
            mags.append(mag) # (B, F, T)
        
        return specs, mags
    
    def _train_step(self, batch):
        """Train model one step."""
        # parse batch
        x = batch[0].to(self.device) # 16kHz PCM
        y = batch[1].to(self.device) # 32kHz PCM
        
        #######################
        #      Generator      #
        #######################    
        y_, outputs = self.model["generator"](x, self.steps)
        gen_loss = 0.0
        
        # stfts
        _, y_mags = self._stfts(y)
        _, y_mags_ = self._stfts(y_)
        
        # multi-resolution stft loss
        if self.config.get("lambda_stft", 0) > 0:
            sc_loss, mag_loss = self.criterion["stft"](y_mags_, y_mags)
            self.total_train_loss["train/spectral_convergence_loss"] += sc_loss.item()
            self.total_train_loss["train/log_stft_magnitude_loss"] += mag_loss.item()
            gen_loss += self.config["lambda_stft"] * (sc_loss + mag_loss)
                
        # make sure the coding are nearly Normal
        mean, var = outputs[0], outputs[1]
        self.total_train_loss["train/coding_mean"] += mean.item()
        self.total_train_loss["train/coding_var"] += var.item()
        if self.config.get("lambda_punish", 0) > 0:
            #if self.steps <= self.config["discriminator_train_start_steps"]:
            gen_loss += self.config["lambda_punish"] * 10 * torch.abs(mean)
            gen_loss += self.config["lambda_punish"] * var
        
        # mse loss of samples
        if self.config.get("lambda_mse", 0) > 0:
            mse_loss = self.criterion["mse"](y_, y)
            self.total_train_loss["train/mse_loss"] += mse_loss.item()
            gen_loss += self.config["lambda_mse"] * mse_loss
        
        # adversarial loss
        if self.steps > self.config["discriminator_train_start_steps"]:
            p_ = self.model["discriminator"](y_, y_mags_)
            adv_loss = 0.0
            for i in range(len(p_)):
                adv_loss += self.criterion["mse"](p_[i], p_[i].new_ones(p_[i].size()))
            adv_loss /= float(i + 1)
            self.total_train_loss["train/adversarial_loss"] += adv_loss.item()
            lambda_adv = min((self.steps - self.config["discriminator_train_start_steps"]) / 100000.0, 1.0) * self.config["lambda_adv"]
            gen_loss += lambda_adv * adv_loss
            
        # total loss
        self.total_train_loss["train/generator_loss"] += gen_loss.item()
        
        # record sdr
        sdr = eval_sdr(y_.detach().unsqueeze(1), y.unsqueeze(1))
        self.total_train_loss["train/sdr"] += sdr.item()

        # update generator
        self.optimizer["generator"].zero_grad()
        gen_loss.backward()
        if self.config["generator_grad_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model["generator"].parameters(),
                self.config["generator_grad_norm"])
        self.optimizer["generator"].step()
        self.scheduler["generator"].step()
        
        #######################
        #    Discriminator    #
        #######################
        if self.steps > self.config["discriminator_train_start_steps"]:
            p = self.model["discriminator"](y, y_mags)
            p_ = self.model["discriminator"](y_.detach(), [m.detach() for m in y_mags_])
            
            # multi-discriminator loss
            real_loss = 0.0
            fake_loss = 0.0
            for i in range(len(p)):
                real_loss += self.criterion["mse"](p[i], p[i].new_ones(p[i].size()))
                fake_loss += self.criterion["mse"](p_[i], p_[i].new_zeros(p_[i].size()))
            real_loss /= float(i + 1)
            fake_loss /= float(i + 1)
            dis_loss = real_loss + fake_loss

            self.total_train_loss["train/real_loss"] += real_loss.item()
            self.total_train_loss["train/fake_loss"] += fake_loss.item()
            self.total_train_loss["train/discriminator_loss"] += dis_loss.item()
            
            # update discriminator
            self.optimizer["discriminator"].zero_grad()
            dis_loss.backward()
            if self.config["discriminator_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model["discriminator"].parameters(),
                    self.config["discriminator_grad_norm"])
            self.optimizer["discriminator"].step()
            self.scheduler["discriminator"].step()
        
        # update counts
        self.steps += 1
        self._check_train_finish()

    def _train_epoch(self):
        """Train model one epoch."""
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)

            # check interval
            if self.config["rank"] == 0:
                self._check_log_interval()
                self._check_eval_interval()
                self._check_save_interval()

            # check whether training is finished
            if self.finish_train:
                return

        # update
        self.epochs += 1
        self.train_steps_per_epoch = train_steps_per_epoch
        logging.info(f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
                     f"({self.train_steps_per_epoch} steps per epoch).")

        # record learning rate
        if self.config["rank"] == 0:
            lr_per_epoch = defaultdict(float)
            for key in self.scheduler:
                lr_per_epoch[f"learning_rate/{key}"] = self.scheduler[key].get_last_lr()[0]
            self._write_to_tensorboard(lr_per_epoch)

        # needed for shuffle in distributed training
        if self.config["distributed"]:
            self.sampler["train"].set_epoch(self.epochs)

    @torch.no_grad()
    def _eval_step(self, batch):
        """Evaluate model one step."""
        # parse batch
        x = batch[0].to(self.device) # 16kHz PCM
        y = batch[1].to(self.device) # 32kHz PCM
        
        #######################
        #      Generator      #
        #######################
        y_, outputs = self.model["generator"](x)
        gen_loss = 0.0
        
        # stfts
        _, y_mags = self._stfts(y)
        _, y_mags_ = self._stfts(y_)
        
        # multi-resolution stft loss
        if self.config.get("lambda_stft", 0) > 0:
            sc_loss, mag_loss = self.criterion["stft"](y_mags_, y_mags)
            self.total_eval_loss["eval/spectral_convergence_loss"] += sc_loss.item()
            self.total_eval_loss["eval/log_stft_magnitude_loss"] += mag_loss.item()
            gen_loss = self.config["lambda_stft"] * (sc_loss + mag_loss)
                
        # make sure the coding are nearly Normal
        mean, var = outputs[0], outputs[1]
        self.total_eval_loss["eval/coding_mean"] += mean.item()
        self.total_eval_loss["eval/coding_var"] += var.item()
        if self.config.get("lambda_punish", 0) > 0:
            #if self.steps <= self.config["discriminator_train_start_steps"]:
            gen_loss += self.config["lambda_punish"] * 10 * torch.abs(mean)
            gen_loss += self.config["lambda_punish"] * var
        
        # mse loss of samples
        if self.config.get("lambda_mse", 0) > 0:
            mse_loss = self.criterion["mse"](y_, y)
            self.total_eval_loss["eval/mse_loss"] += mse_loss.item()
            gen_loss += self.config["lambda_mse"] * mse_loss
        
        # adversarial loss
        if self.steps > self.config["discriminator_train_start_steps"]:
            # multi-discriminator loss
            p_ = self.model["discriminator"](y_, y_mags_)
            adv_loss = 0.0
            for i in range(len(p_)):
                adv_loss += self.criterion["mse"](p_[i], p_[i].new_ones(p_[i].size()))
            adv_loss /= float(i + 1)
            self.total_eval_loss["eval/adversarial_loss"] += adv_loss.item()
            lambda_adv = min((self.steps - self.config["discriminator_train_start_steps"]) / 100000.0, 1.0) * self.config["lambda_adv"]
            gen_loss += lambda_adv * adv_loss
            
        # total loss
        self.total_eval_loss["eval/generator_loss"] += gen_loss.item()
        
        # record sdr
        sdr = eval_sdr(y_.detach().unsqueeze(1), y.unsqueeze(1))
        self.total_eval_loss["eval/sdr"] += sdr.item()
        
        #######################
        #    Discriminator    #
        #######################
        if self.steps > self.config["discriminator_train_start_steps"]:
            p = self.model["discriminator"](y, y_mags)
            p_ = self.model["discriminator"](y_, y_mags_)
            
            # multi-discriminator loss
            real_loss = 0.0
            fake_loss = 0.0
            for i in range(len(p)):
                real_loss += self.criterion["mse"](p[i], p[i].new_ones(p[i].size()))
                fake_loss += self.criterion["mse"](p_[i], p_[i].new_zeros(p_[i].size()))
            real_loss /= float(i + 1)
            fake_loss /= float(i + 1)
            dis_loss = real_loss + fake_loss

            self.total_eval_loss["eval/real_loss"] += real_loss.item()
            self.total_eval_loss["eval/fake_loss"] += fake_loss.item()
            self.total_eval_loss["eval/discriminator_loss"] += dis_loss.item()
        
    def _eval_epoch(self):
        """Evaluate model one epoch."""
        logging.info(f"(Steps: {self.steps}) Start evaluation.")
        # change mode
        for key in self.model.keys():
            self.model[key].eval()

        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(self.data_loader["dev"], 1):
            # eval one step
            self._eval_step(batch)

            # save intermediate result
            if eval_steps_per_epoch == 1:
                self._genearete_and_save_intermediate_result(batch)

        logging.info(f"(Steps: {self.steps}) Finished evaluation "
                     f"({eval_steps_per_epoch} steps per epoch).")

        # average loss
        for key in self.total_eval_loss.keys():
            self.total_eval_loss[key] /= eval_steps_per_epoch
            logging.info(f"(Steps: {self.steps}) {key} = {self.total_eval_loss[key]:.4f}.")
                
        # record
        self._write_to_tensorboard(self.total_eval_loss)

        # reset
        self.total_eval_loss = defaultdict(float)

        # restore mode
        for key in self.model.keys():
            self.model[key].train()

    @torch.no_grad()
    def _genearete_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        # delayed import to avoid error related backend error
        import matplotlib.pyplot as plt
        
        def _demphasis(y, alpha=self.config.get("alpha", 0)):
            y_ = np.zeros_like(y)
            for i in range(1, len(y)):
                y_[i] = y[i-1] + alpha * y_[i-1]
            return y_

        # generate
        x = batch[0].to(self.device)
        y = batch[1]
        y_, _ = self.model["generator"](x)
        
        # check directory
        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for idx, (r, g) in enumerate(zip(y, y_), 1):
            # convert to ndarray
            r = r.view(-1).cpu().numpy().flatten() # groundtruth
            g = g.view(-1).cpu().numpy().flatten() # generated
            
            # de-emphasis
            r = _demphasis(r, self.config.get("alpha", 0))
            g = _demphasis(g, self.config.get("alpha", 0))
            
            # plot figure and save it
            figname = os.path.join(dirname, f"{idx}.png")
            plt.subplot(2, 1, 1)
            plt.plot(r)
            plt.title("groundtruth speech")
            plt.subplot(2, 1, 2)
            plt.plot(g)
            plt.title(f"generated speech @ {self.steps} steps")
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()

            # save as wavfile
            r = np.clip(r, -1, 1)
            g = np.clip(g, -1, 1)
            sf.write(figname.replace(".png", "_ref.wav"), r, self.config["sampling_rate"]*2, "PCM_16")
            sf.write(figname.replace(".png", "_gen.wav"), g, self.config["sampling_rate"]*2, "PCM_16")

            if idx >= self.config["num_save_intermediate_results"]:
                break

    def _write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value, self.steps)

    def _check_save_interval(self):
        if self.steps % self.config["save_interval_steps"] == 0:
            self.save_checkpoint(
                os.path.join(self.config["outdir"], f"checkpoint-{self.steps}steps.pkl"))
            logging.info(f"Successfully saved checkpoint @ {self.steps} steps.")

    def _check_eval_interval(self):
        if self.steps % self.config["eval_interval_steps"] == 0:
            self._eval_epoch()

    def _check_log_interval(self):
        if self.steps % self.config["log_interval_steps"] == 0:
            for key in self.total_train_loss.keys():
                self.total_train_loss[key] /= self.config["log_interval_steps"]
                logging.info(f"(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}.")
            self._write_to_tensorboard(self.total_train_loss)

            # reset
            self.total_train_loss = defaultdict(float)

    def _check_train_finish(self):
        if self.steps >= self.config["train_max_steps"]:
            self.finish_train = True


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description="Train FBandExt (See detail in fbandext/bin/train.py).")
    parser.add_argument("--train-scp", type=str, required=True,
                        help="train.scp file for training.")
    parser.add_argument("--dev-scp", type=str, required=True,
                        help="valid.scp file for validation. ")
    parser.add_argument("--outdir", type=str, required=True,
                        help="directory to save checkpoints.")
    parser.add_argument("--config", type=str, required=True,
                        help="yaml format configuration file.")
    parser.add_argument("--pretrain", default="", type=str, nargs="?",
                        help="checkpoint file path to load pretrained params. (default=\"\")")
    parser.add_argument("--resume", default="", type=str, nargs="?",
                        help="checkpoint file path to resume training. (default=\"\")")
    parser.add_argument("--verbose", type=int, default=1,
                        help="logging level. higher is more logging. (default=1)")
    parser.add_argument("--rank", "--local_rank", default=0, type=int,
                        help="rank for distributed training. no need to explictly specify.")
    args = parser.parse_args()

    args.distributed = False
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        # effective when using fixed size inputs
        # see https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(args.rank)
        # setup for distributed training
        # see example: https://github.com/NVIDIA/apex/tree/master/examples/simple/distributed
        if "WORLD_SIZE" in os.environ:
            args.world_size = int(os.environ["WORLD_SIZE"])
            args.distributed = args.world_size > 1
        if args.distributed:
            torch.distributed.init_process_group(backend="nccl", init_method="env://")

    # suppress logging for distributed training
    if args.rank != 0:
        sys.stdout = open(os.devnull, "w")

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

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # load and save config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))
    config["version"] = fbandext.__version__   # add version info
    if args.rank == 0:
        with open(os.path.join(args.outdir, "config.yml"), "w") as f:
            yaml.dump(config, f, Dumper=yaml.Dumper)
        for key, value in config.items():
            logging.info(f"{key} = {value}")

    # get dataset
    train_dataset = AudioSCPDataset(args.train_scp, config["batch_max_steps"], config["sampling_rate"])
    logging.info(f"The number of training files = {len(train_dataset)}.")
    dev_dataset = AudioSCPDataset(args.dev_scp, config["batch_max_steps"], config["sampling_rate"])
    logging.info(f"The number of development files = {len(dev_dataset)}.")
    dataset = {
        "train": train_dataset,
        "dev": dev_dataset,
    }

    # get data loader
    sampler = {"train": None, "dev": None}
    if args.distributed:
        # setup sampler for distributed training
        from torch.utils.data.distributed import DistributedSampler
        sampler["train"] = DistributedSampler(
            dataset=dataset["train"],
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
        )
        sampler["dev"] = DistributedSampler(
            dataset=dataset["dev"],
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False,
        )
    data_loader = {
        "train": DataLoader(
            dataset=dataset["train"],
            shuffle=False if args.distributed else True,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            sampler=sampler["train"],
            pin_memory=config["pin_memory"],
        ),
        "dev": DataLoader(
            dataset=dataset["dev"],
            shuffle=False if args.distributed else True,
            batch_size=config["batch_size"],
            num_workers=1,
            sampler=sampler["dev"],
            pin_memory=config["pin_memory"],
        ),
    }

    # define models
    generator_class = getattr(fbandext.models, config["generator_type"])
    discriminator_class = getattr(fbandext.models, config["discriminator_type"])
    model = {
        "generator": generator_class(**config["generator_params"]).to(device),
        "discriminator": discriminator_class(**config["discriminator_params"]).to(device),
    }
    logging.info(model["generator"])
    logging.info(model["discriminator"])
    
    # print parameters
    total_params, trainable_params, nontrainable_params = 0, 0, 0
    for param in model["generator"].parameters():
        num_params = np.prod(param.size())
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        else:
            nontrainable_params += num_params
    logging.info(f"Total parameters of Generator: {total_params}")
    logging.info(f"Trainable parameters of Generator: {trainable_params}")
    logging.info(f"Non-trainable parameters of Generator: {nontrainable_params}\n")
    
    total_params, trainable_params, nontrainable_params = 0, 0, 0
    for param in model["discriminator"].parameters():
        num_params = np.prod(param.size())
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        else:
            nontrainable_params += num_params
    logging.info(f"Total parameters of Discriminator: {total_params}")
    logging.info(f"Trainable parameters of Discriminator: {trainable_params}")
    logging.info(f"Non-trainable parameters of Discriminator: {nontrainable_params}\n")
    
    # define criterion and optimizers
    criterion = {
        "stft": MultiResolutionSTFTLoss(
            **config["stft_loss_params"]).to(device),
        "mae": torch.nn.L1Loss().to(device),
        "mse": torch.nn.MSELoss().to(device),
    }
    
    generator_optimizer_class = getattr(fbandext.optimizers, config["generator_optimizer_type"])
    discriminator_optimizer_class = getattr(fbandext.optimizers, config["discriminator_optimizer_type"])
    optimizer = {
        "generator": generator_optimizer_class(
            model["generator"].parameters(),
            **config["generator_optimizer_params"],
        ),
        "discriminator": discriminator_optimizer_class(
            model["discriminator"].parameters(),
            **config["discriminator_optimizer_params"],
        ),
    }
    generator_scheduler_class = getattr(fbandext.lr_scheduler, config["generator_scheduler_type"])
    discriminator_scheduler_class = getattr(fbandext.lr_scheduler, config["discriminator_scheduler_type"])
    scheduler = {
        "generator": generator_scheduler_class(
            optimizer=optimizer["generator"],
            **config["generator_scheduler_params"],
        ),
        "discriminator": discriminator_scheduler_class(
            optimizer=optimizer["discriminator"],
            **config["discriminator_scheduler_params"],
        ),
    }
    
    if args.distributed:
        # wrap model for distributed training
        from torch.nn.parallel import DistributedDataParallel
        model["generator"] = DistributedDataParallel(model["generator"])
        model["discriminator"] = DistributedDataParallel(model["discriminator"])

    # define trainer
    trainer = Trainer(
        steps=0,
        epochs=0,
        data_loader=data_loader,
        sampler=sampler,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
    )

    # load pretrained parameters from checkpoint
    if len(args.pretrain) != 0:
        trainer.load_checkpoint(args.pretrain, load_only_params=True)
        logging.info(f"Successfully load parameters from {args.pretrain}.")

    # resume from checkpoint
    if len(args.resume) != 0:
        trainer.load_checkpoint(args.resume)
        logging.info(f"Successfully resumed from {args.resume}.")
    
    # run training loop
    try:
        trainer.run()
    except KeyboardInterrupt:
        logging.info(f"KeyboardInterrupt @ {trainer.steps}steps.")


if __name__ == "__main__":
    main()
