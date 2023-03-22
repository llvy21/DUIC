import argparse
import logging
import math
import os
import random
import shutil
import sys
from builtins import ValueError
from logging import handlers
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from compressai.datasets import ImageFolder
from compressai.entropy_models import EntropyModel
from compressai.zoo import image_models
from compressai.zoo import image_models as pretrained_models
from matplotlib import pyplot as plt
from PIL import Image
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.waseda import Cheng2020Attention
from models.AdapterWraper import (Cheng2020AttentionAdapter,
                                  Cheng2020AttentionAdapterNew,
                                  Cheng2020AttentionRefine, WeightSplitLoRA,
                                  WeightSplitLoRAGateV2)
from utils import AverageMeter, RDAverageMeter, compute_padding, configure_optimizers

# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.


class WeightFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    # def __init__(self, root, transform=None):
    #     splitdir = Path(root)
    #     self.samples = [f for f in splitdir.iterdir() if f.is_dir()]
    #     self.images = []
    #     self.weight1 = []
    #     self.weight2 = []
    #     for i in tqdm(range(len(self.samples))):
    #         image = Image.open(os.path.join(root, str(i), 'input.png')).convert("RGB")
    #         with open(os.path.join(root, str(i), f'weight_{i}_0.bin'), "rb") as f:
    #             weight1 = f.read()
    #         with open(os.path.join(root, str(i), f'weight_{i}_1.bin'), "rb") as f:
    #             weight2 = f.read()
    #         self.images.append(image)
    #         self.weight1.append(weight1)
    #         self.weight2.append(weight2)

    #     self.transform = transform

    def __init__(self, root, transform=None, net=None):
        splitdir = Path(root)
        self.samples = [f for f in splitdir.iterdir() if f.is_dir()]
        self.images = []
        self.weight0_list = []
        self.weight1_list = []
        for p in tqdm(self.samples):
            self.images.append(os.path.join(p, 'input.png'))

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """

        img = Image.open(self.images[index]).convert("RGB")
        # weight0 = self.weight0_list[index]
        # weight1 = self.weight1_list[index]

        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]
        return out


class RateDistortionLossExtra(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["extra_bpp_loss"] = output["extra_bit"] / num_pixels
        out["extra_bit"] = output["extra_bit"]
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"] + out["extra_bpp_loss"] 
        # out["loss"] = self.lmbda * 255**2 * out["mse_loss"]

        return out



def train_one_image(
    model, criterion, d, optimizer, lr_scheduler, epoch, clip_max_norm, log
):
    # model.train()
    model.eval()
    device = next(model.parameters()).device
    for i in range(epoch):
        d = d.to(device)
        optimizer.zero_grad()
        N, _, H, W = d.size()
        num_pixels = N * H * W
        
        h, w = d.size(2), d.size(3)
        pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
        x_padded = F.pad(d, pad, mode="constant", value=0)

        out_net = model(x_padded)
        out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)
        out_criterion = criterion(out_net, d) 
        extra_bit = model.compress()
        extra_bpp_loss = extra_bit/num_pixels
        loss = out_criterion["loss"] + extra_bpp_loss
        loss.backward()
        
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        # lr_scheduler.step()
        
        if i % 100 == 0:
            bpp_loss = out_criterion["bpp_loss"]
            loss = out_criterion["loss"] + extra_bpp_loss
            # extra_bpp_loss = model.compress()
            psnr = - 10 * math.log10(out_criterion["mse_loss"].cpu())

            bpp_sum = bpp_loss + extra_bpp_loss
            log.logger.info(
                f"Val epoch {i}: "
                f"Loss: {loss:.3f} |"
                f"Basic Bpp: {bpp_loss:.4f} |"
                f"Extra Bit: {extra_bit.cpu()} |"
                f"Bpp_extra: {extra_bpp_loss.cpu():.4f} |"
                f"Bpp_sum: {bpp_sum:.4f} |"
            )
            log.logger.info("[%d/%d] * Avg. PSNR:%.2f bpp:%.4f loss:%.5f" %
                            (i, epoch, psnr, bpp_sum, loss))


    out_net = model(x_padded)
    out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)
    out_criterion = criterion(out_net, d)
    psnr = - 10 * math.log10(out_criterion["mse_loss"].cpu())
    log.logger.info("[%d/%d] * Avg. PSNR:%.2f" % (epoch, epoch, psnr))
    return out_net["x_hat"], psnr


def train_one_image_adapter(
    model, criterion, d, optimizer, lr_scheduler, epoch, clip_max_norm, log
):
    model.eval()
    device = next(model.parameters()).device
    psnr_list = []
    for i in range(epoch):
        d = d.to(device)
        optimizer.zero_grad()

        h, w = d.size(2), d.size(3)
        pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
        x_padded = F.pad(d, pad, mode="constant", value=0)
        
        out_net = model(x_padded)
        out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)
        out_criterion = criterion(out_net, d)
        
        out_criterion["loss"].backward()
        # if clip_max_norm > 0:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        # lr_scheduler.step()
        
        if i % 99 == 0:
            bpp_loss = out_criterion["bpp_loss"]
            loss = out_criterion["loss"]
            extra_bpp_loss = out_criterion["extra_bpp_loss"]
            extra_bit = out_criterion["extra_bit"]
            psnr = - 10 * math.log10(out_criterion["mse_loss"].cpu())
            psnr_list.append(psnr)
            bpp_sum = extra_bpp_loss + bpp_loss
            log.logger.info(
                f"Val epoch {i}: "
                f"Loss: {loss:.3f} |"
                f"Basic Bpp: {bpp_loss:.4f} |"
                f"Extra Bit: {extra_bit} |"
                f"Bpp_extra: {extra_bpp_loss:.4f} |"
                f"Bpp_sum: {bpp_sum:.4f} |"
            )
            log.logger.info("Adapter [%d/%d] * Avg. PSNR:%.2f bpp:%.4f loss:%.5f" % (i, epoch, psnr, bpp_sum, loss))
    
    out_net = model(x_padded)
    out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)
    out_criterion = criterion(out_net, d)
    psnr = - 10 * math.log10(out_criterion["mse_loss"].cpu())       
    log.logger.info("[%d/%d] * Avg. PSNR:%.2f" % (epoch, epoch, psnr)) 
    log.logger.info(psnr_list)
    return out_net["x_hat"], psnr
     
def train_one_image_adapter_wo_refine(
    model:Cheng2020AttentionAdapter, criterion, d, optimizer, lr_scheduler, epoch, clip_max_norm, log
):
    model.eval()
    device = next(model.parameters()).device
    psnr_list = []
    d = d.to(device)
    h, w = d.size(2), d.size(3)
    pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
    x_padded = F.pad(d, pad, mode="constant", value=0)
    with torch.no_grad():
        y = model.g_a(x_padded)
        z = model.h_a(y)
    for i in range(epoch):
        optimizer.zero_grad()
        out_net = model.forward1(y, z)
        out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)
        out_criterion = criterion(out_net, d)
        
        out_criterion["loss"].backward()
        optimizer.step()
        if i % 100 == 99:
            bpp_loss = out_criterion["bpp_loss"]
            loss = out_criterion["loss"]
            extra_bpp_loss = out_criterion["extra_bpp_loss"]
            extra_bit = out_criterion["extra_bit"]
            psnr = - 10 * math.log10(out_criterion["mse_loss"].cpu())
            psnr_list.append(psnr)
            bpp_sum = extra_bpp_loss + bpp_loss
            log.logger.info(
                f"Val epoch {i}: "
                f"Loss: {loss:.3f} |"
                f"Basic Bpp: {bpp_loss:.4f} |"
                f"Extra Bit: {extra_bit} |"
                f"Bpp_extra: {extra_bpp_loss:.4f} |"
                f"Bpp_sum: {bpp_sum:.4f} |"
            )
            log.logger.info("Adapter [%d/%d] * Avg. PSNR:%.2f bpp:%.4f loss:%.5f" % (i, epoch, psnr, bpp_sum, loss))

    return out_net["x_hat"], psnr, bpp_sum.detach(), loss.detach()

     
def train_one_image_adapter_refine(
    model:Cheng2020AttentionAdapter, criterion, d, y, z, optimizer, lr_scheduler, epoch, clip_max_norm, log
):
    model.eval()
    device = next(model.parameters()).device
    psnr_list = []
    d = d.to(device)
    h, w = d.size(2), d.size(3)
    pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
    
    for i in range(epoch):
        optimizer.zero_grad()
        out_net = model.forward1(y.clone(), z.clone())
        out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)
        out_criterion = criterion(out_net, d)
        
        out_criterion["loss"].backward()
        optimizer.step()
        
        if i % 100 == 99:
            bpp_loss = out_criterion["bpp_loss"]
            loss = out_criterion["loss"]
            extra_bpp_loss = out_criterion["extra_bpp_loss"]
            extra_bit = out_criterion["extra_bit"]
            psnr = - 10 * math.log10(out_criterion["mse_loss"].cpu())
            psnr_list.append(psnr)
            bpp_sum = extra_bpp_loss + bpp_loss
            log.logger.info(
                f"Val epoch {i}: "
                f"Loss: {loss:.3f} |"
                f"Basic Bpp: {bpp_loss:.4f} |"
                f"Extra Bit: {extra_bit} |"
                f"Bpp_extra: {extra_bpp_loss:.4f} |"
                f"Bpp_sum: {bpp_sum:.4f} |"
            )
            log.logger.info("Adapter + Refine [%d/%d] * Avg. PSNR:%.2f bpp:%.4f loss:%.5f" % (i, epoch, psnr, bpp_sum, loss))

    return out_net["x_hat"], psnr, out_criterion["bpp_loss"].detach(), loss.detach()


def train_one_image_refine(
    model:Cheng2020Attention, criterion, d, epoch, log
):
    model.eval()
    device = next(model.parameters()).device
    psnr_list = []
    d = d.to(device)
    h, w = d.size(2), d.size(3)
    pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
    x_padded = F.pad(d, pad, mode="constant", value=0)
    with torch.no_grad():
        y = model.g_a(x_padded)
        z = model.h_a(y)
    y.requires_grad_(True)
    z.requires_grad_(True)
    optimizer = torch.optim.Adam([y, z], 1e-3)
    
    tau_decay_it = 0
    tau_decay_factor = 0.001
    for i in range(epoch):
        decaying_iter: int = epoch - tau_decay_it
        tau: float = min(0.5, 0.5 * np.exp(-tau_decay_factor * decaying_iter))
        model.entropy_bottleneck.quantize = lambda x, mode, medians=None: quantize_sga(
            x, tau, medians
        )
        model.gaussian_conditional.quantize = (
            lambda x, mode, medians=None: quantize_sga(x, tau, medians)
        )
        
        optimizer.zero_grad()
        
        out_net = model.forward1(y.clone(),z.clone())
        out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)
        out_criterion = criterion(out_net, d)

        out_criterion["loss"].backward()
        optimizer.step()
        
        if i % 100 == 99:
            bpp_loss = out_criterion["bpp_loss"]
            loss = out_criterion["loss"]
            psnr = - 10 * math.log10(out_criterion["mse_loss"].cpu())
            psnr_list.append(psnr)
            # log.logger.info(
            #     f"Val epoch {i}: "
            #     f"Loss: {loss:.3f} |"
            #     f"Basic Bpp: {bpp_loss:.4f} |"
            # )
            log.logger.info("Refine [%d/%d] * Avg. PSNR:%.2f bpp:%.4f loss:%.5f" % (i, epoch, psnr, bpp_loss, loss))
    
    return loss.detach(), psnr, out_criterion["bpp_loss"].detach(), y, z


def train_one_image_LoRA_wo_refine_RL_V2(
    model: WeightSplitLoRAGateV2, criterion, d, optimizer, lr_scheduler, epoch, clip_max_norm, log
):
    model.eval()
    device = next(model.parameters()).device
    embedding_dim = model.N
    r = 2
    
    lora_list = nn.ParameterList()
    for i in range(16):
        lora_A = torch.nn.parameter.Parameter(torch.randn((r, embedding_dim), device='cuda'), requires_grad=True)
        nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
        lora_B = torch.nn.parameter.Parameter(torch.zeros((embedding_dim, r), device='cuda'), requires_grad=True)
        lora_list.append(lora_A)
        lora_list.append(lora_B)
    
    optimizer = torch.optim.Adam(lora_list, 1e-3)
    optimizer1 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 1e-4)
    psnr_list = []
    d = d.to(device)

    b, c, h, w = d.shape
    NUM_PIXEL = b*h*w
    pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
    x_padded = F.pad(d, pad, mode="constant", value=0)
    with torch.no_grad():
        y = model.g_a(x_padded)
        z = model.h_a(y)
        
    loss_best = float("inf")
    psnr_return = 0.
    bpp_return = 0.
    for i in range(epoch):
        optimizer.zero_grad()
        optimizer1.zero_grad()
        
        lora_w_list = []
        for k in range(8):
            lora_A = lora_list[k*2]
            lora_B = lora_list[k*2+1]
            w = (lora_B.clone() @ lora_A.clone()).unsqueeze(2).unsqueeze(3)
            lora_w_list.append(w)
            
        out_net = model.forward1(y, z, lora_w_list, i<100)
        out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)
        out_criterion = criterion(out_net, d)
        gate = out_net["gate"]
        
        extra_bit = 0.
        gate_int = []
        for k in range(8):
            gate[k] = gate[k].squeeze()
            gate_int.append(gate[k].item())
            # print(gate[k])
            extra_bit += gate[k] * model.compress(lora_list[2*k], lora_list[2*k+1])
            
        extra_bpp_loss = extra_bit / NUM_PIXEL
        loss = out_criterion["loss"] + extra_bpp_loss
        loss.backward()
        
        if loss < loss_best:
            loss_best = loss 
            psnr_return = - 10 * math.log10(out_criterion["mse_loss"].cpu())
            bpp_return = out_criterion["bpp_loss"] + extra_bpp_loss
        
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        optimizer1.step()

        if i % 100 == 99:
            bpp_loss = out_criterion["bpp_loss"]
            loss = extra_bpp_loss + out_criterion["loss"] 
            psnr = - 10 * math.log10(out_criterion["mse_loss"].cpu())
            psnr_list.append(psnr)
            bpp_sum = bpp_loss + extra_bpp_loss
            
            log.logger.info(
                f"Val epoch {i}: "
                f"Loss: {loss:.3f} |"
                f"Basic Bpp: {bpp_loss:.4f} |"
                f"Extra Bit: {extra_bit.cpu()} |"
                f"Bpp_extra: {extra_bpp_loss.cpu():.4f} |"
                f"Bpp_sum: {bpp_sum:.4f} |"
                f"Gate:{gate_int}"
            )
            # print(gate)
            # log.logger.info("LoRA [%d/%d] * Avg. PSNR_actual:%.2f " % (i, epoch, psnr))
            log.logger.info("LoRA_ad [%d/%d] * Avg. PSNR:%.2f bpp:%.4f loss:%.5f" %
                            (i, epoch, psnr, bpp_sum, loss))
        
    return out_net["x_hat"], psnr_return, bpp_return.detach(), loss_best.detach()


def train_one_image_LoRA_refine_RL_V2(
    model: WeightSplitLoRAGateV2, criterion, d, y, z, optimizer, lr_scheduler, epoch, clip_max_norm, log
):
    model.eval()
    device = next(model.parameters()).device
    embedding_dim = model.N
    r = 2
    
    lora_list = nn.ParameterList()
    for i in range(16):
        lora_A = torch.nn.parameter.Parameter(torch.randn((r, embedding_dim), device='cuda'), requires_grad=True)
        nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
        lora_B = torch.nn.parameter.Parameter(torch.zeros((embedding_dim, r), device='cuda'), requires_grad=True)
        lora_list.append(lora_A)
        lora_list.append(lora_B)
    
    optimizer = torch.optim.Adam(lora_list, 1e-3)
    optimizer1 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 1e-4)
    psnr_list = []
    d = d.to(device)

    b, c, h, w = d.shape
    NUM_PIXEL = b*h*w
    pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
    x_padded = F.pad(d, pad, mode="constant", value=0)
        
    loss_best = float("inf")
    psnr_return = 0.
    bpp_return = 0.
    for i in range(epoch):
        optimizer.zero_grad()
        optimizer1.zero_grad()
        
        lora_w_list = []
        for k in range(8):
            lora_A = lora_list[k*2]
            lora_B = lora_list[k*2+1]
            w = (lora_B.clone() @ lora_A.clone()).unsqueeze(2).unsqueeze(3)
            lora_w_list.append(w)
            
        out_net = model.forward1(y.clone(), z.clone(), lora_w_list, i<100)
        out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)
        out_criterion = criterion(out_net, d)
        gate = out_net["gate"]
        
        extra_bit = 0.
        gate_int = []
        for k in range(8):
            gate[k] = gate[k].squeeze()
            gate_int.append(gate[k].item())
            # print(gate[k])
            extra_bit += gate[k] * model.compress(lora_list[2*k], lora_list[2*k+1])
            
        extra_bpp_loss = extra_bit / NUM_PIXEL
        loss = out_criterion["loss"] + extra_bpp_loss
        loss.backward()
        
        if loss < loss_best:
            loss_best = loss 
            psnr_return = - 10 * math.log10(out_criterion["mse_loss"].cpu())
            bpp_return = out_criterion["bpp_loss"] + extra_bpp_loss
        
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        optimizer1.step()

        if i % 100 == 99:
            bpp_loss = out_criterion["bpp_loss"]
            loss = extra_bpp_loss + out_criterion["loss"] 
            psnr = - 10 * math.log10(out_criterion["mse_loss"].cpu())
            psnr_list.append(psnr)
            bpp_sum = bpp_loss + extra_bpp_loss
            
            log.logger.info(
                f"Val epoch {i}: "
                f"Loss: {loss:.3f} |"
                f"Basic Bpp: {bpp_loss:.4f} |"
                f"Extra Bit: {extra_bit.cpu()} |"
                f"Bpp_extra: {extra_bpp_loss.cpu():.4f} |"
                f"Bpp_sum: {bpp_sum:.4f} |"
                f"Gate:{gate_int}"
            )
            # print(gate)
            # log.logger.info("LoRA [%d/%d] * Avg. PSNR_actual:%.2f " % (i, epoch, psnr))
            log.logger.info("LoRA_ad [%d/%d] * Avg. PSNR:%.2f bpp:%.4f loss:%.5f" %
                            (i, epoch, psnr, bpp_sum, loss))
        
    return out_net["x_hat"], psnr_return, bpp_return.detach(), loss_best.detach()


def train_one_image_LoRA_wo_refine(
    model: WeightSplitLoRA, criterion, d, optimizer, lr_scheduler, epoch, clip_max_norm, log
):
    model.eval()
    device = next(model.parameters()).device
    embedding_dim = model.N
    r = 2
    lora_A = torch.nn.parameter.Parameter(torch.randn((r, embedding_dim), device='cuda'), requires_grad=True)
    nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
    lora_B = torch.nn.parameter.Parameter(torch.zeros((embedding_dim, r), device='cuda'), requires_grad=True)
    optimizer = torch.optim.Adam([lora_A, lora_B], 1e-3)
    psnr_list = []
    psnr_actual_list = []
    d = d.to(device)

    b, c, h, w = d.shape
    pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
    x_padded = F.pad(d, pad, mode="constant", value=0)
    with torch.no_grad():
        y = model.g_a(x_padded)
        z = model.h_a(y)
        
    for i in range(epoch):
        optimizer.zero_grad()
        BA = (lora_B.clone() @ lora_A.clone()).unsqueeze(2).unsqueeze(3)
        
        # out_net = model(x_padded, w=BA)
        out_net = model.forward1(y, z, BA)
        out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)
        out_criterion = criterion(out_net, d)
        
        extra_bpp_loss = model.compress(lora_A, lora_B) / (b*h*w)
        loss = out_criterion["loss"] + extra_bpp_loss
        loss.backward()
        
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        
        if i % 100 == 99:
            A_quant, B_quant = model.quant(lora_A, lora_B)
            BA_quant = (B_quant @ A_quant).unsqueeze(2).unsqueeze(3)
            # out_net = model(x_padded, w=BA)
            out_net1 = model.forward1(y.clone(), z.clone(), BA_quant)
            out_net1["x_hat"] = F.pad(out_net1["x_hat"], unpad)
            out_criterion1 = criterion(out_net1, d)
            psnr_actual = - 10 * math.log10(out_criterion1["mse_loss"].cpu())
            psnr_actual_list.append(psnr_actual)
            
            bpp_loss = out_criterion["bpp_loss"]
            loss = extra_bpp_loss + out_criterion["loss"] 
            extra_bit = model.compress(lora_A, lora_B)
            extra_bpp_loss = extra_bit / (b*h*w)
            psnr = - 10 * math.log10(out_criterion["mse_loss"].cpu())
            psnr_list.append(psnr)
            bpp_sum = bpp_loss + extra_bpp_loss
            log.logger.info(
                f"Val epoch {i}: "
                f"Loss: {loss:.3f} |"
                f"Basic Bpp: {bpp_loss:.4f} |"
                f"Extra Bit: {extra_bit.cpu()} |"
                f"Bpp_extra: {extra_bpp_loss.cpu():.4f} |"
                f"Bpp_sum: {bpp_sum:.4f} |"
            )
            # log.logger.info("LoRA [%d/%d] * Avg. PSNR_actual:%.2f " % (i, epoch, psnr))
            log.logger.info("LoRA [%d/%d] * Avg. PSNR:%.2f bpp:%.4f loss:%.5f" %
                            (i, epoch, psnr, bpp_sum, loss))

    return out_net["x_hat"], psnr, bpp_sum.detach(), loss.detach(), lora_A, lora_B
   

def train_one_image_LoRA_refine(
    model: WeightSplitLoRA, criterion, d, y, z, optimizer, lr_scheduler, epoch, clip_max_norm, log
):
    model.eval()
    device = next(model.parameters()).device
    embedding_dim = model.N
    r = 2
    lora_A = torch.nn.parameter.Parameter(torch.randn((r, embedding_dim), device='cuda'), requires_grad=True)
    nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
    lora_B = torch.nn.parameter.Parameter(torch.zeros((embedding_dim, r), device='cuda'), requires_grad=True)
    optimizer = torch.optim.Adam([lora_A, lora_B], 1e-3)
    psnr_list = []
    psnr_actual_list = []
    d = d.to(device)

    b, c, h, w = d.shape
    pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
    x_padded = F.pad(d, pad, mode="constant", value=0)

    for i in range(epoch):
        optimizer.zero_grad()
        BA = (lora_B.clone() @ lora_A.clone()).unsqueeze(2).unsqueeze(3)
        
        # out_net = model(x_padded, w=BA)
        out_net = model.forward1(y.clone(), z.clone(), BA)
        out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)
        out_criterion = criterion(out_net, d)
        
        extra_bpp_loss = model.compress(lora_A, lora_B) / (b*h*w)
        loss = out_criterion["loss"] + extra_bpp_loss
        loss.backward()
        
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        
        if i % 100 == 99:
            A_quant, B_quant = model.quant(lora_A, lora_B)
            BA_quant = (B_quant @ A_quant).unsqueeze(2).unsqueeze(3)
            # out_net = model(x_padded, w=BA)
            out_net1 = model.forward1(y.clone(), z.clone(), BA_quant)
            out_net1["x_hat"] = F.pad(out_net1["x_hat"], unpad)
            out_criterion1 = criterion(out_net1, d)
            psnr_actual = - 10 * math.log10(out_criterion1["mse_loss"].cpu())
            psnr_actual_list.append(psnr_actual)
            
            bpp_loss = out_criterion["bpp_loss"]
            loss = extra_bpp_loss + out_criterion["loss"] 
            extra_bit = model.compress(lora_A, lora_B)
            extra_bpp_loss = extra_bit / (b*h*w)
            psnr = - 10 * math.log10(out_criterion["mse_loss"].cpu())
            psnr_list.append(psnr)
            bpp_sum = bpp_loss + extra_bpp_loss
            log.logger.info(
                f"Val epoch {i}: "
                f"Loss: {loss:.3f} |"
                f"Basic Bpp: {bpp_loss:.4f} |"
                f"Extra Bit: {extra_bit.cpu()} |"
                f"Bpp_extra: {extra_bpp_loss.cpu():.4f} |"
                f"Bpp_sum: {bpp_sum:.4f} |"
            )
            # log.logger.info("LoRA + Refine [%d/%d] * Avg. PSNR_actual:%.2f " % (i, epoch, psnr))
            log.logger.info("LoRA + Refine [%d/%d] * Avg. PSNR:%.2f bpp:%.4f loss:%.5f" %
                            (i, epoch, psnr, bpp_sum, loss))

    return out_net["x_hat"], psnr, loss, lora_A, lora_B
        
     
def train_one_image_finetune(
    model, criterion, d, optimizer, lr_scheduler, epoch, clip_max_norm, log
):
    model.eval()
    device = next(model.parameters()).device
    psnr_list = []
    for i in range(epoch):
        d = d.to(device)
        optimizer.zero_grad()

        h, w = d.size(2), d.size(3)
        pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
        x_padded = F.pad(d, pad, mode="constant", value=0)
        
        out_net = model(x_padded)
        out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)
        out_criterion = criterion(out_net, d)
        
        out_criterion["loss"].backward()
        # if clip_max_norm > 0:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        # lr_scheduler.step()
        
        if i % 50 == 0:
            bpp_loss = out_criterion["bpp_loss"]
            loss = out_criterion["loss"]
            # extra_bpp_loss = out_criterion["extra_bpp_loss"]
            # extra_bit = out_criterion["extra_bit"]
            psnr = - 10 * math.log10(out_criterion["mse_loss"].cpu())
            psnr_list.append(psnr)
            bpp_sum =  bpp_loss
            log.logger.info(
                f"Val epoch {i}: "
                f"Loss: {loss:.3f} |"
                f"Basic Bpp: {bpp_loss:.4f} |"
                # f"Extra Bit: {extra_bit} |"
                # f"Bpp_extra: {extra_bpp_loss:.4f} |"
                f"Bpp_sum: {bpp_sum:.4f} |"
            )
            log.logger.info("Finetune [%d/%d] * Avg. PSNR:%.2f bpp:%.4f loss:%.5f" % (i, epoch, psnr, bpp_sum, loss))
    
    out_net = model(x_padded)
    out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)
    out_criterion = criterion(out_net, d)
    psnr = - 10 * math.log10(out_criterion["mse_loss"].cpu())       
    log.logger.info("[%d/%d] * Avg. PSNR:%.2f" % (epoch, epoch, psnr)) 
    log.logger.info(psnr_list)
    return out_net["x_hat"], psnr
   
     
def train_one_image_multi_LoRA(
    model, criterion, d, optimizer, lr_scheduler, epoch, clip_max_norm, log
):
    model.eval()
    device = next(model.parameters()).device
    d = d.to(device)
    b, c, h, w = d.shape
    pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
    x_padded = F.pad(d, pad, mode="constant", value=0)
    
    r = 2

    with torch.no_grad():
        out_net = model(x_padded)
        out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)
        out_criterion = criterion(out_net, d)
        psnr_b = - 10 * math.log10(out_criterion["mse_loss"].cpu())
        log.logger.info("[w/o lora_C] * Avg. PSNR:%.2f bpp:%.4f loss:%.5f" % (psnr_b, out_criterion["bpp_loss"].cpu(), out_criterion["loss"].cpu()))
    
    optimizer = torch.optim.Adam(model.lora_C, 1e-3)   
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 500)   
    for i in range(epoch):
        optimizer.zero_grad()
        out_net = model(x_padded, C=True)
        out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)
        out_criterion = criterion(out_net, d)
        
        # extra_bpp_loss = model.compress(lora_A, lora_B) / (b*h*w)
        loss = out_criterion["loss"]
        loss.backward()
        
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        lr_scheduler.step()
        
        if i % 100 == 0:
            bpp_loss = out_criterion["bpp_loss"]
            loss = out_criterion["loss"] 
            # extra_bit = model.compress(lora_A, lora_B)
            # extra_bpp_loss = extra_bit / (b*h*w)
            psnr = - 10 * math.log10(out_criterion["mse_loss"].cpu())

            bpp_sum = bpp_loss
            log.logger.info(
                f"Val epoch {i}: "
                f"Loss: {loss:.3f} |"
                f"Basic Bpp: {bpp_loss:.4f} |"
                # f"Extra Bit: {extra_bit.cpu()} |"
                # f"Bpp_extra: {extra_bpp_loss.cpu():.4f} |"
                # f"Bpp_sum: {bpp_sum:.4f} |"
            )
            log.logger.info("[%d/%d] * Avg. PSNR:%.2f bpp:%.4f loss:%.5f" %
                            (i, epoch, psnr, bpp_loss, loss))


    out_net = model(x_padded, C=True)
    out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)
    out_criterion = criterion(out_net, d)
    psnr = - 10 * math.log10(out_criterion["mse_loss"].cpu())
    log.logger.info("[%d/%d] * Avg. PSNR:%.2f" % (epoch, epoch, psnr))
    return out_net["x_hat"], psnr, psnr_b, model.lora_C


def val_baseline(
    model, criterion, d, log
):
    model.eval()
    device = next(model.parameters()).device
    d = d.to('cuda')
    model.eval()
    # d = F.interpolate(d, scale_factor=0.5)
    h, w = d.size(2), d.size(3)
    pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
    x_padded = F.pad(d, pad, mode="constant", value=0)
    out_net = model(x_padded)
    out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)
    out_criterion = criterion(out_net, d)
    psnr_b = - 10 * math.log10(out_criterion["mse_loss"].cpu())
    log.logger.info("[Baseline] * Avg. PSNR:%.2f bpp:%.4f loss:%.5f" % (psnr_b, out_criterion["bpp_loss"].cpu(), out_criterion["loss"].cpu()))
    return out_net["x_hat"], psnr_b, out_criterion["bpp_loss"].detach().cpu(), out_criterion["loss"].detach().cpu()


def train_one_image_LoRA4(
    model, criterion, d, optimizer, lr_scheduler, epoch, clip_max_norm, log
):
    model.eval()
    device = next(model.parameters()).device
    d = d.to(device)
    b, c, h, w = d.shape
    pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
    x_padded = F.pad(d, pad, mode="constant", value=0)
    
    r = 2
    lora_A = torch.load('/mnt/group-ai-medical-cq/private/joylv/data/lora/cheng2020-attn-adapter1-natual-r4/q6-lambda0.045-64/weight_0.pt')
    lora_B = torch.load('/mnt/group-ai-medical-cq/private/joylv/data/lora/cheng2020-attn-adapter1-natual-r4/q6-lambda0.045-64/weight_1.pt')
    # lora_A = torch.load('/mnt/group-ai-medical-cq/private/joylv/data/lora/cheng2020-attn-adapter1-pathology/q6-lambda0.045-64/weight_0.pt')
    # lora_B = torch.load('/mnt/group-ai-medical-cq/private/joylv/data/lora/cheng2020-attn-adapter1-pathology/q6-lambda0.045-64/weight_1.pt')
    lora_C = torch.nn.parameter.Parameter(torch.eye(r, device='cuda'), requires_grad=True)

    with torch.no_grad():
        BA = (lora_B @ lora_A).unsqueeze(2).unsqueeze(3)
        out_net = model(x_padded, w=BA)
        out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)
        out_criterion = criterion(out_net, d)
        psnr_b = - 10 * math.log10(out_criterion["mse_loss"].cpu())
        log.logger.info("[w/o lora_C] * Avg. PSNR:%.2f bpp:%.4f loss:%.5f" % (psnr_b, out_criterion["bpp_loss"].cpu(), out_criterion["loss"].cpu()))
    
    optimizer = torch.optim.Adam([lora_C], 1e-3)   
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 500)   
    for i in range(epoch):
        optimizer.zero_grad()
        BA = (lora_B @ lora_C.clone() @ lora_A).unsqueeze(2).unsqueeze(3)
        
        out_net = model(x_padded, w=BA)
        out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)
        out_criterion = criterion(out_net, d)
        
        # extra_bpp_loss = model.compress(lora_A, lora_B) / (b*h*w)
        loss = out_criterion["loss"]
        loss.backward()
        
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        lr_scheduler.step()
        
        if i % 100 == 0:
            bpp_loss = out_criterion["bpp_loss"]
            loss = out_criterion["loss"] 
            # extra_bit = model.compress(lora_A, lora_B)
            # extra_bpp_loss = extra_bit / (b*h*w)
            psnr = - 10 * math.log10(out_criterion["mse_loss"].cpu())

            bpp_sum = bpp_loss
            log.logger.info(
                f"Val epoch {i}: "
                f"Loss: {loss:.3f} |"
                f"Basic Bpp: {bpp_loss:.4f} |"
                # f"Extra Bit: {extra_bit.cpu()} |"
                # f"Bpp_extra: {extra_bpp_loss.cpu():.4f} |"
                # f"Bpp_sum: {bpp_sum:.4f} |"
            )
            log.logger.info("[%d/%d] * Avg. PSNR:%.2f bpp:%.4f loss:%.5f" %
                            (i, epoch, psnr, bpp_loss, loss))


    out_net = model(x_padded, w=BA)
    out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)
    out_criterion = criterion(out_net, d)
    psnr = - 10 * math.log10(out_criterion["mse_loss"].cpu())
    log.logger.info("[%d/%d] * Avg. PSNR:%.2f" % (epoch, epoch, psnr))
    return out_net["x_hat"], psnr, psnr_b, lora_C


def train_one_image_LoRA(
    model, criterion, d, optimizer, lr_scheduler, epoch, clip_max_norm, log
):
    model.eval()
    device = next(model.parameters()).device
    embedding_dim = model.N
    r = 2
    lora_A = torch.nn.parameter.Parameter(torch.randn((r, embedding_dim), device='cuda'), requires_grad=True)
    nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
    lora_B = torch.nn.parameter.Parameter(torch.zeros((embedding_dim, r), device='cuda'), requires_grad=True)
    optimizer = torch.optim.Adam([lora_A, lora_B], 1e-3)
    psnr_list = []
    for i in range(epoch):
        d = d.to(device)
        optimizer.zero_grad()

        b, c, h, w = d.shape
        pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
        x_padded = F.pad(d, pad, mode="constant", value=0)
        
        BA = (lora_B.clone() @ lora_A.clone()).unsqueeze(2).unsqueeze(3)
        
        # model.g_s[8].conv2.weight = torch.nn.parameter.Parameter(W_0 + BA)
        
        out_net = model(x_padded, w=BA)
        out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)
        out_criterion = criterion(out_net, d)
        
        extra_bpp_loss = model.compress(lora_A, lora_B) / (b*h*w)
        loss = out_criterion["loss"]
        loss.backward()
        
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        
        
        if i % 1 == 0:
            bpp_loss = out_criterion["bpp_loss"]
            loss = extra_bpp_loss + out_criterion["loss"] 
            extra_bit = model.compress(lora_A, lora_B)
            extra_bpp_loss = extra_bit / (b*h*w)
            psnr = - 10 * math.log10(out_criterion["mse_loss"].cpu())
            psnr_list.append(psnr)
            bpp_sum = bpp_loss
            log.logger.info(
                f"Val epoch {i}: "
                f"Loss: {loss:.3f} |"
                f"Basic Bpp: {bpp_loss:.4f} |"
                f"Extra Bit: {extra_bit.cpu()} |"
                f"Bpp_extra: {extra_bpp_loss.cpu():.4f} |"
                f"Bpp_sum: {bpp_sum:.4f} |"
            )
            log.logger.info("LoRA [%d/%d] * Avg. PSNR:%.2f bpp:%.4f loss:%.5f" %
                            (i, epoch, psnr, bpp_sum, loss))


    out_net = model(x_padded, w=BA)
    out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)
    out_criterion = criterion(out_net, d)
    psnr = - 10 * math.log10(out_criterion["mse_loss"].cpu())
    log.logger.info("[%d/%d] * Avg. PSNR:%.2f" % (epoch, epoch, psnr))
    log.logger.info(psnr_list)
    return out_net["x_hat"], psnr, lora_A, lora_B
     

def train_one_image_LoRASR(
    model, criterion, d, optimizer, lr_scheduler, epoch, clip_max_norm, log
):
    model.eval()
    device = next(model.parameters()).device
    embedding_dim = model.N
    r = 192
    kernel = 1
    lora_A = torch.nn.parameter.Parameter(torch.randn((r, embedding_dim * kernel), device='cuda'), requires_grad=True)
    nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
    lora_B = torch.nn.parameter.Parameter(torch.zeros((kernel * embedding_dim * 4, r), device='cuda'), requires_grad=True)
    # optimizer = torch.optim.Adam([lora_A, lora_B], 1e-2)
    optimizer = torch.optim.Adam(model.g_s[8].subpel_conv[0].parameters(), 1e-2)
    psnr_list = []
    d = d.to(device)
    b, c, h, w = d.shape
    pad, unpad = compute_padding(h, w, min_div=2**7)
    d_padded = F.pad(d, pad, mode="constant", value=0)
    
    dd = F.interpolate(d_padded, scale_factor=0.5)
    
    for i in range(epoch):
        optimizer.zero_grad()
        # BA = (lora_B.clone() @ lora_A.clone()).unsqueeze(2).unsqueeze(3)
        BA = (lora_B.clone() @ lora_A.clone()).reshape(embedding_dim * 4, embedding_dim, kernel, kernel)
        out_net = model(dd, w=BA)
        out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)
        out_criterion = criterion(out_net, d)
        extra_bpp_loss = model.compress(lora_A, lora_B) / (b*h*w)
        loss = out_criterion["loss"]
        loss.backward()
        
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        
        
        if i % 100 == 0:
            bpp_loss = out_criterion["bpp_loss"]
            loss = extra_bpp_loss + out_criterion["loss"] 
            extra_bit = model.compress(lora_A, lora_B)
            extra_bpp_loss = extra_bit / (b*h*w)
            psnr = - 10 * math.log10(out_criterion["mse_loss"].cpu())
            psnr_list.append(psnr)
            bpp_sum = bpp_loss
            log.logger.info(
                f"Val epoch {i}: "
                f"Loss: {loss:.3f} |"
                f"Basic Bpp: {bpp_loss:.4f} |"
                f"Extra Bit: {extra_bit.cpu()} |"
                f"Bpp_extra: {extra_bpp_loss.cpu():.4f} |"
                f"Bpp_sum: {bpp_sum:.4f} |"
            )
            log.logger.info("LoRA [%d/%d] * Avg. PSNR:%.2f bpp:%.4f loss:%.5f" %
                            (i, epoch, psnr, bpp_sum, loss))


    out_net = model(dd, w=BA)
    out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)
    out_criterion = criterion(out_net, d)
    psnr = - 10 * math.log10(out_criterion["mse_loss"].cpu())
    log.logger.info("[%d/%d] * Avg. PSNR:%.2f" % (epoch, epoch, psnr))
    log.logger.info(psnr_list)
    return out_net["x_hat"], psnr, lora_A, lora_B
       
  
def quantize_sga(y: torch.Tensor, tau: float, medians=None, eps: float = 1e-5):
    # use Gumbel Softmax implemented in tfp.distributions.RelaxedOneHotCategorical

    # (N, C, H, W)
    if medians is not None:
        y -= medians
    y_floor = torch.floor(y)
    y_ceil = torch.ceil(y)
    # (N, C, H, W, 2)
    y_bds = torch.stack([y_floor, y_ceil], dim=-1)
    # (N, C, H, W, 2)
    ry_logits = torch.stack(
        [
            -torch.atanh(torch.clamp(y - y_floor, -1 + eps, 1 - eps)) / tau,
            -torch.atanh(torch.clamp(y_ceil - y, -1 + eps, 1 - eps)) / tau,
        ],
        axis=-1,
    )
    # last dim are logits for DOWN or UP
    ry_dist = torch.distributions.RelaxedOneHotCategorical(tau, logits=ry_logits)
    ry_sample = ry_dist.rsample()
    outputs = torch.sum(y_bds * ry_sample, dim=-1)
    if medians is not None:
        outputs += medians
    return outputs
  
   
def save_checkpoint(state, is_best, model_prefix, filename="checkpoint.pth.tar"):
    torch.save(state, model_prefix + filename)
    if is_best:
        torch.save(state, model_prefix + "checkpoint_best_loss.pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="bmshj2018-factorized",
        choices=image_models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-td", "--test_dataset", type=str, required=True, help="Testing dataset"
    )
    parser.add_argument(
        "-vd", "--val_dataset", type=str, required=True, help="Validation dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=3,
        help="compress quality",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--finetune", action="store_true", default=True, help="Finetune with post processing"
    )
    parser.add_argument(
        "--adapter", action="store_true", default=True, help="Finetune with post processing"
    )    
    parser.add_argument(
        "--refinement", action="store_true", default=True, help="Finetune with post processing"
    )    
    parser.add_argument(
        "--lora", action="store_true", default=True, help="Finetune with post processing"
    )    
    parser.add_argument(
        "--refine_adapter", action="store_true", default=True, help="Finetune with post processing"
    )
    parser.add_argument(
        "--refine_lora", action="store_true", default=True, help="Finetune with post processing"
    )
    
    
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument(
        "--model_prefix",
        default="./",
        type=str, required=True,
        help="Path to save checkpoints and logs",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self, filename, level='info', when='W0', backCount=3, fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        th = logging.handlers.TimedRotatingFileHandler(
            filename=filename, when=when, encoding='utf-8')
        th.setFormatter(format_str)
        self.logger.addHandler(sh)
        self.logger.addHandler(th)


def report_params(model):
    n_params_total: int = 0
    n_params_update: int = 0
    for key, p in model.named_parameters():
        n_param = np.prod(p.shape)
        n_params_total += n_param
        if p.requires_grad:
            n_params_update += n_param
            print(key, n_param, p.shape)

    print(f"#updating params/#total params: {n_params_update}/{n_params_total}")


def main(argv):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms = torchvision.transforms.Compose([
        # torchvision.transforms.RandomCrop(256),
        torchvision.transforms.ToTensor(),
        #  torchvision.transforms.RandomVerticalFlip(0.5),
        #  torchvision.transforms.RandomHorizontalFlip(0.5)
    ])
    # train_transforms = torchvision.transforms.Compose(
    #     [torchvision.transforms.ToTensor()]
    # )
    # train_dataset = ImageFolder(f'/mnt/group-ai-medical-cq/private/joylv/data/BRACS/test', transform=train_transforms)
    # train_dataset = WeightFolder('/mnt/group-ai-medical-cq/private/joylv/data/adapter-overfit-label-1/cheng2020-attn-overfit-pathology-BRACS-256/q6-lambda0.045-64', transform=train_transforms)
    train_dataset = ImageFolder(args.val_dataset, transform=train_transforms, split="")

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )
    
    lmd_list = [0.0016, 0.0032, 0.0075, 0.015, 0.03, 0.045]
    args.lmbda = lmd_list[args.quality-1]
    if args.quality <= 3:
        N = 128
    else:
        N = 192
    
    net_baseline = Cheng2020Attention(N)
    # net_baseline = pretrained_models['cheng2020-attn'](quality=args.quality, metric='mse', pretrained=True)
    state_dict = pretrained_models['cheng2020-attn'](quality=args.quality, metric='mse', pretrained=True).state_dict()
    net_baseline.load_state_dict(state_dict, strict=False)

    device = 'cuda'
    net_lora = WeightSplitLoRA(N)
    state_dict = pretrained_models['cheng2020-attn'](quality=args.quality, metric='mse', pretrained=True).state_dict()
    net_lora.load_state_dict(state_dict, strict=False)
    
    net_lora_ad = WeightSplitLoRAGateV2(N)
    net_lora_ad.load_state_dict(state_dict, strict=False)
    
    net_adapter : Cheng2020AttentionAdapter = Cheng2020AttentionAdapter(N)
    net_adapter.load_state_dict(state_dict, strict=False)
        
    # net_adapter2 : Cheng2020AttentionAdapterNew = Cheng2020AttentionAdapterNew(N)
    # net_adapter2.load_state_dict(state_dict, strict=False)
    # net_adapter2 = net_adapter2.to(device)
    
    net_adapter = net_adapter.to(device)
    net_lora = net_lora.to(device)
    net_lora_ad = net_lora_ad.to(device)
    net_baseline = net_baseline.to(device)
    
    for name, p in net_adapter.named_parameters():
        p.requires_grad = False
        if "adapter" in name:
            p.requires_grad = True
            print(name)
            
    # for name, p in net_adapter2.named_parameters():
    #     p.requires_grad = False
    #     # if "g_s.8.conv1" in name or "g_s.8.conv2" in name:
    #     if "g_s." in name:
    #         p.requires_grad = True
            
    for name, p in net_lora_ad.named_parameters():
        p.requires_grad = False
        if "gate" in name:
            p.requires_grad = True
            print(name)
            
    # optimizer = configure_optimizers(net_hyper, args)
    optimizer_baseline = configure_optimizers(net_baseline, args)
    optimizer_adapter = configure_optimizers(net_adapter, args)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer_baseline, 1800)
    
    log = Logger(filename=os.path.join(args.model_prefix, 'train.log'), 
                level='info', 
                fmt="%(asctime)s - %(message)s")
        

    log.logger.info(args)
    
    RD_baseline = RDAverageMeter()
    RD_refine = RDAverageMeter()
    RD_refine_adapter = RDAverageMeter()
    RD_refine_lora = RDAverageMeter()
    
    epoch = 2000
    for k in range(1):
        for i, d in enumerate(train_dataloader):
            net_adapter.init_adapter()
            criterion = RateDistortionLoss(lmbda=args.lmbda)
            criterion1 = RateDistortionLossExtra(lmbda=args.lmbda)
            
            x_hat, psnr_b, bpp_b, loss_b = val_baseline(net_baseline, criterion, d, log)

            loss_r, psnr_r, bpp_r, y, z = train_one_image_refine(net_baseline, criterion, d, epoch, log)
            # x_hat, psnr_a, bpp_a, loss_a = train_one_image_adapter_wo_refine(net_adapter, criterion1, d, optimizer_adapter, lr_scheduler, epoch, args.clip_max_norm, log)
            x_hat, psnr_ar, bpp_ar, loss_ar = train_one_image_adapter_refine(net_adapter, criterion1, d, y, z, optimizer_adapter, lr_scheduler, epoch, args.clip_max_norm, log)
            # x_hat, psnr_l, bpp_l, loss_l, A, B = train_one_image_LoRA_wo_refine(net_lora, criterion, d, None, None, epoch, args.clip_max_norm, log)
            # x_hat, psnr_lr, loss_lr, A, B = train_one_image_LoRA_refine(net_lora, criterion, d, y, z, None, None, 2000, args.clip_max_norm, log)
            # x_hat, psnr_la, bpp_la, loss_la = train_one_image_LoRA_wo_refine_RL_V2(net_lora_ad, criterion, d, None, None, epoch, args.clip_max_norm, log)
            x_hat, psnr_lar, bpp_lar, loss_lar = train_one_image_LoRA_refine_RL_V2(net_lora_ad, criterion, d, y, z, None, None, epoch, args.clip_max_norm, log)

            RD_baseline.update(psnr_b, psnr_b, loss_b)
            RD_refine.update(psnr_r, psnr_r, loss_r)
            RD_refine_adapter.update(psnr_ar, psnr_ar, loss_ar)
            RD_refine_lora.update(psnr_lar, psnr_lar, loss_lar)
            
            log.logger.info(f'[{i}] Baseline \t\t PSNR avg: {RD_baseline.psnr.avg:.2f} \t bpp avg: {RD_baseline.bpp.avg:.4f} \t loss avg: {RD_baseline.loss.avg:.5f}')
            log.logger.info(f'[{i}] Refine \t\t PSNR avg: {RD_refine.psnr.avg:.2f} \t bpp avg: {RD_refine.bpp.avg:.4f} \t loss avg: {RD_refine.loss.avg:.5f}')
            log.logger.info(f'[{i}] Refine+Adapter \t PSNR avg: {RD_refine_adapter.psnr.avg:.2f} \t bpp avg: {RD_refine_adapter.bpp.avg:.4f} \t loss avg: {RD_refine_adapter.loss.avg:.5f}')
            log.logger.info(f'[{i}] Refine+LoRA \t PSNR avg: {RD_refine_lora.psnr.avg:.2f} \t bpp avg: {RD_refine_lora.bpp.avg:.4f} \t loss avg: {RD_refine_lora.loss.avg:.5f}')
        
if __name__ == "__main__":
    main(sys.argv[1:])
