import copy
import math
import string

import torch
import torch.nn as nn
from compressai.layers import (AttentionBlock, ResidualBlock,
                               ResidualBlockUpsample, ResidualBlockWithStride,
                               conv3x3, subpel_conv3x3)
from compressai.zoo import image_models
from einops import rearrange
# from layers.gdn import GDNLoRA
from layers import (ResidualBlockAdapter, ResidualBlockHyper,
                    ResidualBlockLoRA, ResidualBlockLoRAUpsample,
                    ResidualBlockSVD)
from torch.nn.parameter import Parameter

from .cdf_utils import (LogisticCDF, SpikeAndSlabCDF, WeightEntropyModule)
from .waseda import Cheng2020Attention


class WeightSplitLoRAGateV4(Cheng2020Attention):
    def __init__(self, N=192):
        super().__init__(N=N)
        self.dim=2
        self.N = N
        
        self.scale = 0.05
        self.distrib = LogisticCDF(scale=self.scale)
        self.w_ent = WeightEntropyModule(self.distrib, 0.01, data_type='uint8').to('cuda')

        self.g_s = nn.Sequential(
            AttentionBlock(N),
            ResidualBlockLoRA(N, N),
            ResidualBlockLoRAUpsample(N, N, 2),
            ResidualBlockLoRA(N, N),
            ResidualBlockLoRAUpsample(N, N, 2),
            AttentionBlock(N),
            ResidualBlockLoRA(N, N),
            ResidualBlockLoRAUpsample(N, N, 2),
            ResidualBlockLoRA(N, N),
            subpel_conv3x3(N, 3, 2),
        )
        
    def compress2bit(self, A, B):
        strings = []
        for param in [A, B]:
            w_shape = param.reshape(1, 1, -1).shape
            diff = param.reshape(w_shape)
            string = self.w_ent.compress(diff)
            strings.append(string)
        return strings
        
    def compress(self, A, B):
        weights = nn.ParameterList([A, B])
        weights_q = []
        likelihoods = []
        for w in weights:
            w_shape = w.reshape(1, 1, -1).shape
            diff = w.reshape(w_shape)
            diff_q, likelihood = self.w_ent(diff)
            likelihoods.append(likelihood)
            weights_q.append(diff_q.reshape(w.shape))
            
        extra_bit = sum(
            (torch.log(likelihood).sum() / -math.log(2))
            for likelihood in likelihoods
        )
        return extra_bit
    
    def quant(self, A, B):
        weights = nn.ParameterList([A, B])
        weights_q = []
        likelihoods = []
        for w in weights:
            w_shape = w.reshape(1, 1, -1).shape
            diff = w.reshape(w_shape)
            diff_q, likelihood = self.w_ent(diff)
            likelihoods.append(likelihood)
            weights_q.append(diff_q.reshape(w.shape))
            
        return weights_q[0], weights_q[1]
     
    def forward1(self, y, z, lora, warm_up=False, warm_up_list=None):
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
    
        params = self.h_s(z_hat)
        y_hat = self.gaussian_conditional.quantize(y, "noise" if self.training else "dequantize")
            
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(torch.cat((params, ctx_params), dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
                 
        temp = y_hat
            
        lora_upsample_layer = [2, 4, 7]
        lora_layer = [1, 3, 6, 8]
        gate = []
        for i in range(len(self.g_s)):
            if i in lora_layer:
                k = lora_layer.index(i)
                if warm_up_list is not None:
                    warm_up0 = warm_up_list[3 * k]
                    warm_up1 = warm_up_list[3 * k + 1]
                else:
                    warm_up0 = warm_up
                    warm_up1 = warm_up
                temp = self.g_s[i](temp, lora[3 * k], lora[3 * k + 1], warm_up0, warm_up1)
                gate.append(self.g_s[i].g1)
                gate.append(self.g_s[i].g2)
            elif i in lora_upsample_layer:
                if warm_up_list is not None:
                    warm_up2 = warm_up_list[3 * k + 2]
                else:
                    warm_up2 = warm_up
                k = lora_upsample_layer.index(i)
                temp = self.g_s[i](temp, lora[3 * k + 2], warm_up2)
                gate.append(self.g_s[i].g)
            else:
                temp = self.g_s[i](temp)
            
        x_hat = temp.clamp_(0, 1)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "gate": gate
        }
        
    def forward(self, x, w1, w2):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
    
        params = self.h_s(z_hat)
        y_hat = self.gaussian_conditional.quantize(y, "noise" if self.training else "dequantize")
            
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(torch.cat((params, ctx_params), dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
                 
        temp = y_hat
        for i in range(len(self.g_s)-2):
            temp = self.g_s[i](temp)
            
        identity = temp
        # out = self.g_s[len(self.g_s)-2].conv1(temp)
        w1_gt = self.g_s[len(self.g_s)-2].conv1.weight
        b1_gt = self.g_s[len(self.g_s)-2].conv1.bias
        if len(w1.shape) == 5:
            w1_gt = w1_gt.unsqueeze(0)
            out = batch_conv(w1_gt+w1, out)
        else:
            out = torch.nn.functional.conv2d(out, w1_gt+w1, b1_gt, padding=1)
        out = self.g_s[len(self.g_s)-2].leaky_relu(out)
        
        
        # out = self.g_s[8].conv2(out)
        w2_gt = self.g_s[len(self.g_s)-2].conv2.weight
        b2_gt = self.g_s[len(self.g_s)-2].conv2.bias
        if len(w1.shape) == 5:
            w2_gt = w2_gt.unsqueeze(0)
            out = batch_conv(w2_gt+w2, out)
        else:
            out = torch.nn.functional.conv2d(out, w2_gt+w2, b2_gt, padding=1)
        out = self.g_s[len(self.g_s)-2].leaky_relu(out)
        temp = out + identity
        
        x_hat = self.g_s[len(self.g_s)-1](temp).clamp_(0, 1)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
    
