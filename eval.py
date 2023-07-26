import argparse
import logging
import math
import os
import random
import sys
from logging import handlers

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from compressai.zoo import image_models
from compressai.zoo import image_models as pretrained_models
from PIL import Image

from models.waseda import Cheng2020Attention
from models.AdapterWraper import Cheng2020DynamicAdapt
from utils import AverageMeter, RDAverageMeter, compute_padding


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

def latent_refine(model: Cheng2020Attention, criterion, d, epoch, log):
    model.eval()
    device = next(model.parameters()).device
    psnr_list = []
    d = d.to(device)
    h, w = d.size(2), d.size(3)
    pad, unpad = compute_padding(h, w, min_div=2**6)
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
        # following [Yang+, NeurIPS 20]
        decaying_iter: int = i - tau_decay_it
        tau: float = min(0.5, 0.5 * np.exp(-tau_decay_factor * decaying_iter))
        model.entropy_bottleneck.quantize = lambda x, mode, medians=None: quantize_sga(
            x, tau, medians
        )
        model.gaussian_conditional.quantize = (
            lambda x, mode, medians=None: quantize_sga(x, tau, medians)
        )

        optimizer.zero_grad()

        out_net = model.forward1(y.clone(), z.clone())
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
            log.logger.info(
                "Latent Refine [%d/%d] * Avg. PSNR:%.2f bpp:%.4f loss:%.5f" % (i, epoch, psnr, bpp_loss, loss))

    return loss.detach().cpu(), psnr, out_criterion["bpp_loss"].detach().cpu(), y.detach(), z.detach()

def dynamic_adapt(model: Cheng2020DynamicAdapt, criterion, d, y, z, epoch, log, lr):
    model.eval()

    device = next(model.parameters()).device
    embedding_dim = model.N
    r = 2

    lora_list = nn.ParameterList()
    N = 11
    # weight matrix decompose
    for i in range(N):
        lora_A = torch.nn.parameter.Parameter(torch.randn(
            (r, embedding_dim), device='cuda'), requires_grad=True)
        nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
        lora_B = torch.nn.parameter.Parameter(torch.zeros(
            (embedding_dim, r), device='cuda'), requires_grad=True)
        lora_list.append(lora_A)
        lora_list.append(lora_B)

    # optimize model update decompose low-rank matrix
    optimizer = torch.optim.Adam(lora_list, lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, epoch//5*4)
    # optimize gate network
    optimizer1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 1e-5)
    d = d.to(device)

    b, c, h, w = d.shape
    NUM_PIXEL = b*h*w
    # pad to allow 6 strides of 2
    pad, unpad = compute_padding(h, w, min_div=2**6)

    step1 = epoch//20
    step2 = epoch//10
    final_gate = None
    for i in range(epoch):
        optimizer.zero_grad()
        optimizer1.zero_grad()

        lora_w_list = []
        lora_w_list_quant = []
        for k in range(N):
            lora_A = lora_list[k*2]
            lora_B = lora_list[k*2+1]
            w = (lora_B.clone() @ lora_A.clone()).unsqueeze(2).unsqueeze(3)
            lora_w_list.append(w)
            A_quant, B_quant = model.quant(lora_A, lora_B)
            BA_quant = (B_quant @ A_quant).unsqueeze(2).unsqueeze(3)
            lora_w_list_quant.append(BA_quant)

        if i < step2:
            out_net = model.forward1(y.clone(), z.clone(), lora_w_list_quant, i < step1)
        else:
            out_net = model.forward1(y.clone(), z.clone(), lora_w_list_quant, i < step1, final_gate)

        out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)
        gate = out_net["gate"]

        extra_bit = 0.
        gate_int = []
        for k in range(N):
            gate[k] = gate[k].squeeze()
            if i > step2:
                gate[k] = gate[k].detach()
            gate_int.append(int(gate[k].item()))
            extra_bit += gate[k] * \
                model.compress(lora_list[2*k], lora_list[2*k+1])
        gate_num = sum(gate_int)

        extra_bpp_loss = extra_bit / NUM_PIXEL

        out_criterion = criterion(out_net, d)
        bpp_loss = out_criterion["bpp_loss"]
        bpp_sum = bpp_loss + extra_bpp_loss
        loss = out_criterion["loss"] + extra_bpp_loss
        psnr = - 10 * math.log10(out_criterion["mse_loss"].cpu())
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        if i < step2:
            optimizer1.step()
            if gate_num >= 1:
                final_gate = [g.detach() for g in gate]
        if i == step2:
            for name, p in model.named_parameters():
                p.requires_grad = False

        if i % 100 == 99:
            log.logger.info(
                f"Val step {i}: "
                f"Loss: {loss:.3f} "
                f"PSNR: {psnr:.2f} "
                f"bpp_sum: {bpp_sum:.4f} ({bpp_loss:.4f} + {extra_bpp_loss.cpu():.4f}) |"
                f"Gate:{gate_int} |"
            )
            log.logger.info("Dynamic Adapt [%d/%d] * Avg. PSNR:%.2f bpp:%.4f loss:%.5f" %
                            (i, epoch, psnr, bpp_sum, loss))


    return psnr, bpp_sum.detach().cpu(), loss.detach().cpu(), gate_num, out_net["x_hat"], final_gate, lora_list

def val_baseline(
    model, criterion, d, log
):
    model.eval()
    d = d.to('cuda')
    with torch.no_grad():
        # d = F.interpolate(d, scale_factor=0.5)
        h, w = d.size(2), d.size(3)
        # pad to allow 6 strides of 2
        pad, unpad = compute_padding(h, w, min_div=2**6)
        x_padded = F.pad(d, pad, mode="constant", value=0)
        out_net = model(x_padded)
        out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)
        out_criterion = criterion(out_net, d)
        psnr_b = - 10 * math.log10(out_criterion["mse_loss"].cpu())
    log.logger.info("Baseline * Avg. PSNR:%.2f bpp:%.4f loss:%.5f" %
                    (psnr_b, out_criterion["bpp_loss"].cpu(), out_criterion["loss"].cpu()))
    return psnr_b, out_criterion["bpp_loss"].detach().cpu(), out_criterion["loss"].detach().cpu(), out_net["x_hat"] 

# following [Yang+, NeurIPS 20]
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
    ry_dist = torch.distributions.RelaxedOneHotCategorical(
        tau, logits=ry_logits)
    ry_sample = ry_dist.rsample()
    outputs = torch.sum(y_bds * ry_sample, dim=-1)
    if medians is not None:
        outputs += medians
    return outputs

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
        "-i", "--image", type=str, required=True, help="Input Image."
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=2000,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
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
        "--quality",
        type=int,
        default=3,
        help="compress quality",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")

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

def main(argv):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    # load image
    img = Image.open(args.image).convert("RGB")
    d = train_transforms(img).unsqueeze(0)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    lmd_list = [0.0016, 0.0032, 0.0075, 0.015, 0.03, 0.045]
    args.lmbda = lmd_list[args.quality-1]
    if args.quality <= 3:
        N = 128
    else:
        N = 192

    # baseline model
    net_baseline = Cheng2020Attention(N)
    state_dict = pretrained_models['cheng2020-attn'](quality=args.quality, metric='mse', pretrained=True).state_dict()
    net_baseline.load_state_dict(state_dict, strict=False)
    net_baseline = net_baseline.to(device)

    # adaptation model
    net_adapt = Cheng2020DynamicAdapt(N)
    net_adapt.load_state_dict(state_dict, strict=False)
    net_adapt = net_adapt.to(device)

    for name, p in net_adapt.named_parameters():
        p.requires_grad = False
        if "gate" in name:
            p.requires_grad = True

    log = Logger(filename=os.path.join(args.model_prefix, 'train.log'),
                 level='info',
                 fmt="%(asctime)s - %(message)s")

    log.logger.info(args)

    RD_baseline = RDAverageMeter()
    RD_adapt = RDAverageMeter()
    gate = AverageMeter()
    epoch = args.epochs

    criterion = RateDistortionLoss(lmbda=args.lmbda)
    for name, p in net_adapt.named_parameters():
        p.requires_grad = False
        if "gate" in name:
            p.requires_grad = True
    psnr_b, bpp_b, loss_b, x_hat_baseline = val_baseline(net_baseline, criterion, d, log)
    
    # content adaptation
    loss_r, psnr_r, bpp_r, y, z = latent_refine(net_baseline, criterion, d, epoch, log)
    
    # decoder adaptation
    psnr_lar6, bpp_lar6, loss_lar6, gate_num6, x_hat_adapt, gate, lora_list = dynamic_adapt(net_adapt, criterion, d, y, z, epoch, log, args.learning_rate)

    torchvision.utils.save_image(x_hat_baseline, 'result_baseline.png')
    torchvision.utils.save_image(x_hat_adapt, 'result_adapt.png')
    
    RD_baseline.update(psnr_b, bpp_b, loss_b)
    RD_adapt.update(psnr_lar6, bpp_lar6, loss_lar6)
    gate.update(gate_num6)

    log.logger.info(f'Baseline \t\t PSNR avg: {RD_baseline.psnr.avg:.2f} \t bpp avg: {RD_baseline.bpp.avg:.4f} \t loss avg: {RD_baseline.loss.avg:.5f}')
    log.logger.info(f'Dynamic Adapt \t PSNR avg: {RD_adapt.psnr.avg:.2f} \t bpp avg: {RD_adapt.bpp.avg:.4f} \t loss avg: {RD_adapt.loss.avg:.5f} \t gate avg: {gate.avg:.2f}')

    extra_bit = 0
    # save model updates
    with open(f'model_update.bin', "wb") as f:
        gate_i = [str(int(t.item())) for t in gate]
        gate_str = ''.join(gate_i)
        gate_b = int(gate_str, 2).to_bytes(2, 'big')
        f.write(gate_b)
        f.write(b'\n'b'\n')
        extra_bit += len(gate_b)
        
    for k in range(len(lora_list)//2):
        if gate[k] == 1:
            lora_A = lora_list[k*2]
            lora_B = lora_list[k*2+1]
            strings = net_adapt.compress2bit(lora_A, lora_B)
            with open(f'model_update.bin', "ab") as f:
                f.write(strings[0][0])
                f.write(b'\n'b'\n')
                f.write(strings[1][0])
                f.write(b'\n'b'\n')
                extra_bit += (len(strings[0][0]) + len(strings[1][0])) * 8
                
    # load model update
    file = 'model_update.bin'
    gate = decompress_gate(file)
    lora_list = decompress_diff(net_adapt, file)
    lora_w_list = []
    k = 0 
    # load model update weights according to gate information
    for g in gate:
        if g == 0:
            lora_A = lora_list[0]
            lora_B = lora_list[1]
            w = (lora_B @ lora_A).unsqueeze(2).unsqueeze(3)
            lora_w_list.append(torch.zeros_like(w))
        else:            
            lora_A = lora_list[k*2]
            lora_B = lora_list[k*2+1] 
            w = (lora_B @ lora_A).unsqueeze(2).unsqueeze(3)  
            lora_w_list.append(w)
            k+=1
            
    # inference image with updated model
    with torch.no_grad():
        d = d.to(device)
        h, w = d.size(2), d.size(3)
        pad, unpad = compute_padding(h, w, min_div=2**6)
        x_padded = F.pad(d, pad, mode="constant", value=0)
        out_net = net_adapt.forward1(y.clone(), z.clone(), lora_w_list, False, gate)
        b, c, h, w = d.shape
        NUM_PIXEL = b*h*w
        extra_bpp_loss = extra_bit / NUM_PIXEL
        out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)
        out_criterion = criterion(out_net, d)
        bpp_loss = out_criterion["bpp_loss"]
        bpp_sum = bpp_loss + extra_bpp_loss
        loss = out_criterion["loss"] + extra_bpp_loss
        psnr = - 10 * math.log10(out_criterion["mse_loss"].cpu())
        log.logger.info("[Load model update] "
            f"Loss: {loss:.3f} |"
            f"PSNR: {psnr:.2f} |"
            f"Basic Bpp: {bpp_loss:.4f} |"
            f"Extra Bit: {extra_bit*8} |"
            f"Bpp_extra: {extra_bpp_loss:.4f} |"
            f"Bpp_sum: {bpp_sum:.4f} |"
            f"Gate:{gate} |"
        )
            
def decompress_diff(net, file='model_update.bin'):
    with open(file, "rb") as f:
        gate_b = f.read()
        weights = gate_b.split(b'\n'b'\n')[1:-1]
        
        lora_list = []
        idx = 0
        for w in weights:
            w = decompress_from_bytes(net, [w], idx)
            lora_list.append(w)
            idx += 1
    return lora_list
        
def decompress_gate(file='model_update.bin'):
    with open(file, "rb") as f:
        gate_b = f.read(2)
        gate_str = str(bin(int.from_bytes(gate_b,'big')))[2:]
        gate = []
        for i in range(len(gate_str)):
            gate.append(int(gate_str[i]))        
        while len(gate) < 11:
            gate.insert(0,0)
    return gate

def decompress_from_bytes(net, weight, idx):
    N = net.N
    if idx % 2 == 0:
        param = torch.ones((2, N))
    else:
        param = torch.ones((N, 2))
    diff = net.w_ent.decompress(weight, (param.numel(),))
    diff = diff.reshape(param.shape)
    return diff

if __name__ == "__main__":
    main(sys.argv[1:])
