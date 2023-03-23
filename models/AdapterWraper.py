import copy
import math

import torch
import torch.nn as nn
from compressai.layers import (AttentionBlock, ResidualBlock,
                               ResidualBlockUpsample, ResidualBlockWithStride,
                               conv3x3, subpel_conv3x3)
from compressai.zoo import image_models
from einops import rearrange
from layers import (ResidualBlockAdapter, ResidualBlockHyper,
                    ResidualBlockLoRA, ResidualBlockLoRAUpsample)
from torch.nn.parameter import Parameter

from .cdf_utils import (LogisticCDF, SpikeAndSlabCDF, WeightEntropyModule,
                        cal_modified_bpp_cost, compress_weight,
                        decompress_weight)
from .waseda import Cheng2020Attention


@torch.no_grad()
def init_adapter_layer(adapter_layer: nn.Module):
    if isinstance(adapter_layer, nn.Conv2d):
        # adapter_layer.weight.fill_(0.0)
        adapter_layer.weight.normal_(0.0, 0.02)

        if adapter_layer.bias is not None:
            adapter_layer.bias.fill_(0.0)
            
class Cheng2020AttentionAdapter(Cheng2020Attention):
    def __init__(
        self,
        N=192,
        **kwargs
    ):
        super().__init__(N, **kwargs)
            
        self.dim = 2
        
        self.g_s = nn.Sequential(
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlockAdapter(N, N, self.dim),
            subpel_conv3x3(N, 3, 2),
        )
        
        distrib = LogisticCDF(scale=0.05)
        self.w_ent = WeightEntropyModule(distrib, 0.06, data_type='uint8').to('cuda')
        
    def init_adapter(self):
        self.g_s[8].adapter.apply(init_adapter_layer)
        
    def get_adapter_param(self):
        adapter_dict = dict()
        for name, p in self.named_parameters():
            if "adapter" in name:
                adapter_dict[name] = p
        return adapter_dict
        
    def compress_adapter_to_bytes(self):
        adapter_dict = self.get_adapter_param()
        strings = []
        for key, param in adapter_dict.items():
            w_shape = param.reshape(1, 1, -1).shape
            diff = param.reshape(w_shape)
            string = self.w_ent.compress(diff)
            strings.append(string)
        return strings
    
    def compress_adapter(self):
        adapter_dict = self.get_adapter_param()
        for name, p in self.named_parameters():
            if "adapter" in name:
                adapter_dict[name] = p

        likelihoods = []
        for key, param in adapter_dict.items():
            w_shape = param.reshape(1, 1, -1).shape
            # p_init = torch.zeros_like(param)
            # diff = (param - p_init).reshape(w_shape)
            diff = param.reshape(w_shape)
            diff_q, likelihood = self.w_ent(diff)
            likelihoods.append(likelihood)
            
        return likelihoods
    
    def decompress_adapter_from_bytes(self, weight1, weight2):
        param1 = self.g_s[8].adapter[0].weight
        diff1 = self.w_ent.decompress(weight1, (param1.numel(),))
        diff1 = diff1.reshape(param1.shape)
        param2 = self.g_s[8].adapter[1].weight
        diff2 = self.w_ent.decompress(weight2, (param2.numel(),))
        diff2 = diff2.reshape(param2.shape)
        return diff1, diff2
    
    def forward(self, x):
        out = super().forward(x)
        likelihoods = self.compress_adapter()
        out["extra_bit"] = sum(
            (torch.log(likelihood).sum() / -math.log(2))
            for likelihood in likelihoods
        )
        return out
        
    def load_state_dict(self, state_dict, strict: bool = True):
        super(Cheng2020Attention, self).load_state_dict(state_dict, strict=strict)
   
    def forward1(self, y, z):
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
    
        params = self.h_s(z_hat)
        y_hat = self.gaussian_conditional.quantize(y, "noise" if self.training else "dequantize")
            
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(torch.cat((params, ctx_params), dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
            
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        likelihoods = self.compress_adapter()
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "extra_bit": sum(
                (torch.log(likelihood).sum() / -math.log(2))
                for likelihood in likelihoods
            )
        }
        
        
class Cheng2020AttentionRefine(Cheng2020Attention):
    def __init__(
        self,
        N=192,
        **kwargs
    ):
        super().__init__(N, **kwargs)
            
        self.dim = 2
        
        # self.quant = nn.parameter.Parameter(torch.ones(), requires_grad=True)
        
    def forward(self, y, z):
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
    
        params = self.h_s(z_hat)
        y_hat = self.gaussian_conditional.quantize(y, "noise" if self.training else "dequantize")
            
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(torch.cat((params, ctx_params), dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
            
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
        
    def load_state_dict(self, state_dict, strict: bool = True):
        super(Cheng2020Attention, self).load_state_dict(state_dict, strict=strict)
  
   
class Cheng2020AttentionSplit(Cheng2020Attention):
    def __init__(
        self,
        N=192,
        **kwargs
    ):
        super().__init__(N, **kwargs)
            
        self.dim = 2
        
                
    def forward0(self, x):
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
        for i in range(len(self.g_s)-1):
            temp = self.g_s[i](temp)
        
        return {
            "temp": temp,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
    
    def forward1(self, temp, out_net):
        x_hat = self.g_s[len(self.g_s)-1](temp).clamp_(0, 1)
        out_net["x_hat"] = x_hat
        return out_net
    

class Cheng2020AttentionAdapterNew(Cheng2020Attention):
    def __init__(self, N=192, **kwargs):
        super().__init__(N, **kwargs)
        width = 5e-3
        distrib = SpikeAndSlabCDF(width=width)
        self.w_ent = WeightEntropyModule(distrib, width=width, data_type='uint8').to('cuda')
        
        # distrib = LogisticCDF(scale=0.05)
        # self.w_ent = WeightEntropyModule(distrib, 0.06, data_type='uint8').to('cuda')
        self.origin_conv_weight = {}
    
    def init_origin_params(self):
        self.origin_conv1_weight = copy.deepcopy(self.g_s[8].conv1.weight)
        self.origin_conv2_weight = copy.deepcopy(self.g_s[8].conv2.weight)
        for name, p in self.named_parameters():
            if "g_s" in name:
                self.origin_conv_weight[name] = copy.deepcopy(p).to('cuda')
        
    def compress_diff(self):
        diff = self.g_s[8].conv1.weight - self.origin_conv1_weight
        return self.w_ent.compress_spike_and_slab(diff)
    
    
    def compress_diff2(self):
        diff1 = self.g_s[8].conv1.weight - self.origin_conv1_weight
        diff1_byte = self.w_ent.compress_spike_and_slab(diff1)
        diff2 = self.g_s[8].conv2.weight - self.origin_conv2_weight
        diff2_byte = self.w_ent.compress_spike_and_slab(diff2)
        return diff1_byte, diff2_byte
    
    def decompress_diff(self, weight):
        param = self.origin_conv1_weight
        diff = self.w_ent.decompress(weight, (param.numel(),))
        diff = diff.reshape(param.shape)
        return diff
    
    def decompress_diff2(self, weight1, weight2):
        param1 = self.origin_conv1_weight
        diff1 = self.w_ent.decompress(weight1, (param1.numel(),))
        diff1 = diff1.reshape(param1.shape)
        param2 = self.origin_conv2_weight
        diff2 = self.w_ent.decompress(weight2, (param2.numel(),))
        diff2 = diff2.reshape(param2.shape)
        return diff1, diff2
        
    @staticmethod
    def cal_modified_bpp_cost(weight):
        return len(weight[0]) * 8

    
    def update_weights(self, diff):
        self.g_s[8].conv1.weight = Parameter(diff + self.origin_conv1_weight)  
        
    def update_weights2(self, diff1, diff2):
        self.g_s[8].conv1.weight = Parameter(diff1 + self.origin_conv1_weight)  
        self.g_s[8].conv2.weight = Parameter(diff2 + self.origin_conv2_weight)  
        
    def compress_all_diff(self):
        adapter_dict = dict()
        for name, p in self.named_parameters():
            if "g_s" in name:
                adapter_dict[name] = p
        likelihoods = []
        for key, param in adapter_dict.items():
            w_shape = param.reshape(1, 1, -1).shape
            p_init = self.origin_conv_weight[key]
            diff = (param - p_init).reshape(w_shape)
            diff_q, likelihood = self.w_ent(diff)
            likelihoods.append(likelihood)
            
        return likelihoods
    
    def evaluate(self, net: Cheng2020Attention, x):
        for key, param in net.named_parameters():
            if "g_s" in key:
                trained_param = None
                for key1, param1 in self.named_parameters():
                    if key == key1:
                        # print(key)
                        trained_param = param1
                        break
                w_shape = trained_param.reshape(1, 1, -1).shape
                p_init = self.origin_conv_weight[key]
                diff = (trained_param - p_init).reshape(w_shape)
                diff_q, likelihood = self.w_ent(diff)
                diff_q = diff_q.reshape(trained_param.shape)
                param.data = Parameter(diff_q + self.origin_conv_weight[key])  
                # print(torch.sum(diff), torch.sum(diff_q))
        
        out = net.forward(x)
        out["extra_bit"] = 0
        return out
        
    def forward(self, x):
        # diff1 = self.g_s[8].conv1.weight - self.origin_conv1_weight
        # diff_q1, likelihood1 = self.w_ent(diff1)
        
        # diff2 = self.g_s[8].conv2.weight - self.origin_conv2_weight
        # diff_q2, likelihood2 = self.w_ent(diff2)
        # self.update_weights(diff_q)
        
        out = super().forward(x)
        # out["extra_bit"] = likelihood1.log().sum() / -math.log(2) + likelihood2.log().sum() / -math.log(2) 
        likelihoods = self.compress_all_diff()
        
        out["extra_bit"] = 0
        # out["extra_bit"] = sum(
        #     (torch.log(likelihood).sum() / (-math.log(2)))
        #     for likelihood in likelihoods)
        
        return out

        
    def load_state_dict(self, state_dict, strict: bool = True):
        super(Cheng2020Attention, self).load_state_dict(state_dict, strict=strict)
        self.init_origin_params()

class WeightGenerator(Cheng2020AttentionAdapter):
    def __init__(self, N=192):
        super().__init__(N)
        
        # self.weight_generator = nn.Sequential(
        #     ResidualBlockWithStride(3, N, stride=4),
        #     ResidualBlock(N, N),
        #     ResidualBlockWithStride(N, N*2, stride=8),
        #     AttentionBlock(N*2),
        #     ResidualBlock(N*2, N*2),
        #     ResidualBlockWithStride(N*2, N*4, stride=8),
        #     ResidualBlock(N*4, N*4),
        #     conv3x3(N*4, N*8, stride=8)
        # )
        
        self.generator_feature_extractor = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2)
        )
        
        self.generator_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.generator_fc = nn.Sequential(
            nn.Linear(N, N),
            nn.LeakyReLU(0.2),
            nn.Linear(N, N*self.dim*2)
        )
        
        # self.weight_generator1 = nn.Sequential(
        #         nn.Linear(N*8, N),
        #         nn.LeakyReLU(0.2),
        #         nn.Linear(N, N),
        #         nn.LeakyReLU(0.2),
        #         nn.Linear(N, N*self.dim*2)
        #     )

        self.N = N
        
    def update_weight(self, weight):
        weight0 = weight[:, :self.dim * self.N].reshape(self.g_s[8].adapter[0].weight.shape)
        weight1 = weight[:, self.dim * self.N:].reshape(self.g_s[8].adapter[1].weight.shape)
        self.g_s[8].adapter[0].weight = Parameter(weight0)  
        self.g_s[8].adapter[1].weight = Parameter(weight1)  
        return weight0, weight1
        
    def forward(self, x):
        # weight = self.weight_generator(x)
        # b,_,_,_ = weight.shape
        # weight = weight.view(b,-1)
        # weight = self.weight_generator1(weight)
        # weight0, weight1 = self.update_weight(weight)
        
        f = self.generator_feature_extractor(x)
        b, c, _, _ = f.shape
        f = self.generator_avg_pool(f).view(b, c)
        weight = self.generator_fc(f)
        # weight0, weight1 = self.update_weight(weight)
        weight0 = 0
        weight1 = 1
        out = super().forward(x)
        out["weight0"] = weight0
        out["weight1"] = weight1
        return out   
        

class WeightSplit(nn.Module):
    def __init__(self, N=192):
        super().__init__()
        self.dim=2
        
        self.generator_feature_extractor = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2)
        )
        
        self.generator_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.generator_fc = nn.Sequential(
            nn.Linear(N, N),
            nn.LeakyReLU(0.2),
            nn.Linear(N, N*self.dim*2)
        )
    
        self.N = N
        
        distrib = LogisticCDF(scale=0.05)
        self.w_ent = WeightEntropyModule(distrib, 0.06, data_type='uint8').to('cuda')
        
    def compress(self, weight0, weight1):
        # weight0 = weight[:, :self.dim * self.N].reshape(weight_adapter_0.shape) + weight_adapter_0
        # weight1 = weight[:, self.dim * self.N:].reshape(weight_adapter_1.shape) + weight_adapter_1
        weights = [weight0, weight1]
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
        return {
            "extra_bit": extra_bit,
            "weight0_q": weights_q[0],
            "weight1_q": weights_q[1],
        }
     
    def forward(self, x):
        
        f = self.generator_feature_extractor(x)
        b, c, _, _ = f.shape
        f = self.generator_avg_pool(f).view(b, c)
        weight = self.generator_fc(f)
        
        return weight   
        
class WeightSplitHyper(Cheng2020Attention):
    def __init__(self, N=192, z_dim=32, NUM=3):
        super().__init__()
        self.dim=2
        DIM = 192 // NUM
        self.hyper = HyperNetwork(z_dim=z_dim, out_size=DIM, in_size=DIM)
        self.emb1 = Embedding([NUM,NUM], z_dim)
        self.emb2 = Embedding([NUM,NUM], z_dim)
        self.resblock = ResidualBlockHyper(N, N)
        self.N = N
        
        distrib = LogisticCDF(scale=0.05)
        self.w_ent = WeightEntropyModule(distrib, 0.06, data_type='uint8').to('cuda')
        
    def compress(self):
        weights = self.emb2.z_list
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
     
    def forward(self, x):
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
            
        # w1 = self.emb1(self.hyper)
        w2 = self.emb2(self.hyper)
        w1_gt = self.g_s[len(self.g_s)-2].conv1.weight
        b1_gt = self.g_s[len(self.g_s)-2].conv1.bias
        temp = self.resblock(temp, w1_gt, w2, b1_gt)        
        
        x_hat = self.g_s[len(self.g_s)-1](temp).clamp_(0, 1)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        } 

class MultiLoRA(Cheng2020Attention):
    def __init__(self, N=192):
        super().__init__()
        self.N = N
        self.dim = 2
        self.r = 2
        self.lora_dim = [1, 3, 6, 8]
        self.lora_A = nn.ParameterList()
        self.lora_B = nn.ParameterList()
        self.lora_C = nn.ParameterList()
        for k in range(len(self.lora_dim) * 2):
            lora_A = nn.parameter.Parameter(torch.randn((self.r, self.N)), requires_grad=True)
            nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
            lora_B = torch.nn.parameter.Parameter(torch.zeros((self.N, self.r)), requires_grad=True)
            self.lora_A.append(lora_A)
            self.lora_B.append(lora_B)
            if k % 2 == 0:
                lora_C = nn.parameter.Parameter(torch.eye(self.r), requires_grad=True)
                self.lora_C.append(lora_C)
            
        self.g_s = nn.Sequential(
            AttentionBlock(N),
            ResidualBlockLoRA(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlockLoRA(N, N),
            ResidualBlockUpsample(N, N, 2),
            AttentionBlock(N),
            ResidualBlockLoRA(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlockLoRA(N, N),
            subpel_conv3x3(N, 3, 2),
        )
        
        distrib = LogisticCDF(scale=0.05)
        self.w_ent = WeightEntropyModule(distrib, 0.06, data_type='uint8').to('cuda')
        
    def compress(self):
        weights = self.lora_C
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
     
    def forward(self, x, C=False):
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
        k = 0
        for i in range(len(self.g_s)-1):
            if i in self.lora_dim:
                if C:
                    BA1 = (self.lora_B[2*k] @ self.lora_C[k] @ self.lora_A[2*k]).unsqueeze(2).unsqueeze(3)
                    BA2 = (self.lora_B[2*k+1] @ self.lora_C[k] @ self.lora_A[2*k+1]).unsqueeze(2).unsqueeze(3)
                else:
                    BA1 = (self.lora_B[2*k] @ self.lora_A[2*k]).unsqueeze(2).unsqueeze(3)
                    BA2 = (self.lora_B[2*k+1] @ self.lora_A[2*k+1]).unsqueeze(2).unsqueeze(3)
                w1_gt = self.g_s[i].conv1.weight
                w2_gt = self.g_s[i].conv2.weight
                b1 = self.g_s[i].conv1.bias
                b2 = self.g_s[i].conv2.bias
                k += 1
                temp = self.g_s[i](temp, w1_gt+BA1, w2_gt+BA2, b1, b2)
            else:
                temp = self.g_s[i](temp)
        x_hat = self.g_s[len(self.g_s)-1](temp).clamp_(0, 1)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
        
class WeightSplitLoRA(Cheng2020Attention):
    def __init__(self, N=192):
        super().__init__(N=N)
        self.dim=2
        self.N = N
        
        distrib = LogisticCDF(scale=0.05)
        self.w_ent = WeightEntropyModule(distrib, 0.06, data_type='uint8').to('cuda')
        
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
            
        extra_bit = sum(
            (torch.log(likelihood).sum() / -math.log(2))
            for likelihood in likelihoods
        )
        return weights_q[0], weights_q[1]
     
    def forward1(self, y, z, w):
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
        out = self.g_s[len(self.g_s)-2].conv1(temp)
        out = self.g_s[len(self.g_s)-2].leaky_relu(out)
        
        # out = self.g_s[8].conv2(out)
        w2_gt = self.g_s[len(self.g_s)-2].conv2.weight
        b2_gt = self.g_s[len(self.g_s)-2].conv2.bias
        if len(w.shape) == 5:
            w2_gt = w2_gt.unsqueeze(0)
            out = batch_conv(w2_gt+w, out)
        else:
            out = torch.nn.functional.conv2d(out, w2_gt+w, b2_gt, padding=1)
        out = self.g_s[len(self.g_s)-2].leaky_relu(out)
        temp = out + identity
        
        x_hat = self.g_s[len(self.g_s)-1](temp).clamp_(0, 1)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
        
class WeightSplitSVD(Cheng2020Attention):
    def __init__(self, N=192):
        super().__init__(N=N)
        self.dim=2
        self.N = N
    
        distrib = LogisticCDF(scale=0.05)
        self.w_ent = WeightEntropyModule(distrib, 0.06, data_type='uint8').to('cuda')
        
    def compress(self, A):
        weights = nn.ParameterList([A])
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
    
    def quant(self, A):
        weights = nn.ParameterList([A])
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
        return weights_q[0]
     
    def forward1(self, y, z, sigma):
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
        out = self.g_s[len(self.g_s)-2].conv1(temp)
        out = self.g_s[len(self.g_s)-2].leaky_relu(out)
        
        # out = self.g_s[8].conv2(out)
        w2_gt = self.g_s[len(self.g_s)-2].conv2.weight
        b2_gt = self.g_s[len(self.g_s)-2].conv2.bias
        c1, c2, h, w = w2_gt.shape

        A = rearrange(w2_gt, 'c1 c2 h w-> c1 (c2 h w)')
        # A = rearrange(w2_gt, 'c1 c2 h w-> (c1 h) (c2 w)')
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        w_svd = U@torch.diag(S+sigma)@Vh
        w_svd = rearrange(w_svd, 'c1 (c2 h w)-> c1 c2 h w', c2=c2, h=h, w=w)
        # w_svd = rearrange(w_svd, '(c1 h) (c2 w)-> c1 c2 h w', c1=c1, c2=c2, h=h, w=w)
        
        out = torch.nn.functional.conv2d(out, w_svd, b2_gt, padding=1)
        out = self.g_s[len(self.g_s)-2].leaky_relu(out)
        temp = out + identity
        
        x_hat = self.g_s[len(self.g_s)-1](temp).clamp_(0, 1)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
        
    def forward(self, x, w):
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
        out = self.g_s[len(self.g_s)-2].conv1(temp)
        out = self.g_s[len(self.g_s)-2].leaky_relu(out)
        
        # out = self.g_s[8].conv2(out)
        w2_gt = self.g_s[len(self.g_s)-2].conv2.weight
        b2_gt = self.g_s[len(self.g_s)-2].conv2.bias

        A = rearrange(w2_gt, 'c1 c2 h w-> c1 (c2 h w)')
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        w_svd = U@torch.diag(S+w)@Vh
        
        out = torch.nn.functional.conv2d(out, w_svd, b2_gt, padding=1)
        out = self.g_s[len(self.g_s)-2].leaky_relu(out)
        temp = out + identity
        
        x_hat = self.g_s[len(self.g_s)-1](temp).clamp_(0, 1)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

# Feedforward-Gate (FFGate-I)
class FeedforwardGateI(nn.Module):
    """ Use Max Pooling First and then apply to multiple 2 conv layers.
    The first conv has stride = 1 and second has stride = 2"""
    def __init__(self, pool_size=5, channel=10):
        super(FeedforwardGateI, self).__init__()
        self.pool_size = pool_size
        self.channel = channel

        self.maxpool = nn.MaxPool2d(2)
        self.conv1 = conv3x3(channel, channel)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu1 = nn.ReLU(inplace=True)

        # adding another conv layer
        self.conv2 = conv3x3(channel, channel, stride=2)
        self.bn2 = nn.BatchNorm2d(channel)
        self.relu2 = nn.ReLU(inplace=True)

        pool_size = math.floor(pool_size/2)  # for max pooling
        pool_size = math.floor(pool_size/2 + 0.5)  # for conv stride = 2

        self.avg_layer = nn.AvgPool2d(pool_size)
        self.linear_layer = nn.Conv2d(in_channels=channel, out_channels=2,
                                      kernel_size=1, stride=1)
        self.prob_layer = nn.Softmax()
        self.logprob = nn.LogSoftmax()

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.avg_layer(x)
        x = self.linear_layer(x).squeeze()
        softmax = self.prob_layer(x)
        logprob = self.logprob(x)
        print(x.shape, logprob.shape)
        # discretize output in forward pass.
        # use softmax gradients in backward pass
        x = (softmax[:, 1] > 0.5).float().detach() - \
            softmax[:, 1].detach() + softmax[:, 1]

        x = x.view(x.size(0), 1, 1, 1)
        return x, logprob
# FFGate-II
class FeedforwardGateII(nn.Module):
    """ use single conv (stride=2) layer only"""
    def __init__(self, pool_size=5, channel=10):
        super(FeedforwardGateII, self).__init__()
        self.pool_size = pool_size
        self.channel = channel

        self.conv1 = conv3x3(channel, channel, stride=2)
        # self.bn1 = nn.BatchNorm2d(channel)
        self.relu1 = nn.ReLU(inplace=True)

        pool_size = math.floor(pool_size/2 + 0.5) # for conv stride = 2

        # self.avg_layer = nn.AvgPool2d(pool_size)
        self.avg_layer = nn.AdaptiveAvgPool2d(1)
        self.linear_layer = nn.Conv2d(in_channels=channel, out_channels=2,
                                      kernel_size=1, stride=1)
        self.prob_layer = nn.Softmax()
        self.logprob = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu1(x)

        x = self.avg_layer(x)
        x = self.linear_layer(x)[:, :]
        softmax = self.prob_layer(x)
        logprob = self.logprob(x)
        # discretize
        x = (softmax[:, 1] > 0.5).float().detach() - \
            softmax[:, 1].detach() + softmax[:, 1]
        # x = softmax[:, 1].contiguous()
        x = x.view(x.size(0), 1, 1, 1)
        return x, logprob


class WeightSplitLoRAGate(Cheng2020Attention):
    def __init__(self, N=192):
        super().__init__(N=N)
        self.dim=2
        self.N = N
        
        distrib = LogisticCDF(scale=0.05)
        self.w_ent = WeightEntropyModule(distrib, 0.06, data_type='uint8').to('cuda')

        self.gate1 = FeedforwardGateII(channel=N)
        self.gate2 = FeedforwardGateII(channel=N)
        self.gate3 = FeedforwardGateII(channel=N)
        # self.g_s = nn.Sequential(
        #     AttentionBlock(N),
        #     ResidualBlock(N, N),
        #     ResidualBlockUpsample(N, N, 2),
        #     ResidualBlock(N, N),
        #     ResidualBlockUpsample(N, N, 2),
        #     AttentionBlock(N),
        #     ResidualBlock(N, N),
        #     ResidualBlockUpsample(N, N, 2),
        #     ResidualBlock(N, N),
        #     subpel_conv3x3(N, 3, 2),
        # )
        
        self.cnt = 0
        
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
            
        extra_bit = sum(
            (torch.log(likelihood).sum() / -math.log(2))
            for likelihood in likelihoods
        )
        return weights_q[0], weights_q[1]
     
    def forward1(self, y, z, w1, w2, w3):
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
    
        params = self.h_s(z_hat)
        y_hat = self.gaussian_conditional.quantize(y, "noise" if self.training else "dequantize")
            
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(torch.cat((params, ctx_params), dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
                 
        temp = y_hat
        for i in range(len(self.g_s)-3):
            temp = self.g_s[i](temp)
            
        identity = temp
        C = 10

        out = self.g_s[len(self.g_s)-3].subpel_conv(temp)
        out = self.g_s[len(self.g_s)-3].leaky_relu(out)
        # out = self.g_s[len(self.g_s)-3].conv(out)
        w0_gt = self.g_s[len(self.g_s)-3].conv.weight
        b0_gt = self.g_s[len(self.g_s)-3].conv.bias
        g3, prob = self.gate3(out)
        self.cnt += 1
        if self.cnt<=C:
            g3 = torch.ones((1, 1, 1, 1), device='cuda', requires_grad=True)
        w3 = g3*w3
        
        out = torch.nn.functional.conv2d(out, w0_gt+w3, b0_gt, padding=1)
        
        out = self.g_s[len(self.g_s)-3].igdn(out)
        identity = self.g_s[len(self.g_s)-3].upsample(temp)
        out += identity
        
        temp = out
        identity = temp
        out = temp
        
        g1, prob = self.gate1(out)
        if self.cnt<=C:
            g1 = torch.ones((1, 1, 1, 1), device='cuda', requires_grad=True)
        w1 = g1*w1
        w1_gt = self.g_s[len(self.g_s)-2].conv1.weight
        b1_gt = self.g_s[len(self.g_s)-2].conv1.bias
        if len(w1.shape) == 5:
            w1_gt = w1_gt.unsqueeze(0)
            out = batch_conv(w1_gt+w1, out)
        else:
            out = torch.nn.functional.conv2d(out, w1_gt+w1, b1_gt, padding=1)
            
        out = self.g_s[len(self.g_s)-2].leaky_relu(out)
        
        g2, prob = self.gate2(out)
        if self.cnt<=C:
            g2 = torch.ones((1, 1, 1, 1), device='cuda', requires_grad=True)
        w2 = g2*w2
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
        
        # x_hat = self.g_s[len(self.g_s)-1](temp).clamp_(0, 1)
        
        w3_gt = self.g_s[len(self.g_s)-1][0].weight
        b3_gt = self.g_s[len(self.g_s)-1][0].bias
        # print(w3_gt.shape, w3.shape, b3_gt.shape)
        temp = torch.nn.functional.conv2d(temp, w3_gt, b3_gt, padding=1)
        temp = self.g_s[len(self.g_s)-1][1](temp)
        x_hat = temp.clamp_(0, 1)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "gate1": g1,
            "gate2": g2,
            "gate3": g3,
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


class WeightSplitLoRAGateV2(Cheng2020Attention):
    def __init__(self, N=192):
        super().__init__(N=N)
        self.dim=2
        self.N = N
        
        distrib = LogisticCDF(scale=0.05)
        self.w_ent = WeightEntropyModule(distrib, 0.06, data_type='uint8').to('cuda')

        self.g_s = nn.Sequential(
            AttentionBlock(N),
            ResidualBlockLoRA(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlockLoRA(N, N),
            ResidualBlockUpsample(N, N, 2),
            AttentionBlock(N),
            ResidualBlockLoRA(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlockLoRA(N, N),
            subpel_conv3x3(N, 3, 2),
        )
        
        
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
            
        extra_bit = sum(
            (torch.log(likelihood).sum() / -math.log(2))
            for likelihood in likelihoods
        )
        return weights_q[0], weights_q[1]
     
    def forward1(self, y, z, lora, warm_up=False):
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
    
        params = self.h_s(z_hat)
        y_hat = self.gaussian_conditional.quantize(y, "noise" if self.training else "dequantize")
            
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(torch.cat((params, ctx_params), dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
                 
        temp = y_hat
            
        lora_layer = [1, 3, 6, 8]
        gate = []
        for i in range(len(self.g_s)):
            if i in lora_layer:
                k = lora_layer.index(i)
                temp = self.g_s[i](temp, lora[2 * k], lora[2 * k + 1], warm_up)
                gate.append(self.g_s[i].g1)
                gate.append(self.g_s[i].g2)
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


class Cheng2020AttentionSR(Cheng2020Attention):
    def __init__(self, N=192):
        super().__init__()
        self.dim=2
        self.N = N
    
        self.g_s = nn.Sequential(
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlockUpsample(N, N, 2),
            subpel_conv3x3(N, 3, 2),
        )

class WeightSplitLoRASR(Cheng2020AttentionSR):
    def __init__(self, N=192):
        super().__init__(N)
        self.dim=2
        self.N = N
        
        distrib = LogisticCDF(scale=0.05)
        self.w_ent = WeightEntropyModule(distrib, 0.06, data_type='uint8').to('cuda')
        # width = 5e-3
        # distrib = SpikeAndSlabCDF(width=width)
        # self.w_ent = WeightEntropyModule(distrib, width=width, data_type='uint8').to('cuda')
        
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
     
    def forward1(self, y, z, w):
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
        w2_gt = self.g_s[len(self.g_s)-2].subpel_conv[0].weight
        b2_gt = self.g_s[len(self.g_s)-2].subpel_conv[0].bias
        out = torch.nn.functional.conv2d(temp, w2_gt+w, b2_gt, padding=1)
        out = self.g_s[len(self.g_s)-2].subpel_conv[1](out)
        out = self.g_s[len(self.g_s)-2].leaky_relu(out)
        out = self.g_s[len(self.g_s)-2].conv(out)
        out = self.g_s[len(self.g_s)-2].igdn(out)
        identity = self.g_s[len(self.g_s)-2].upsample(identity)
        # out = self.g_s[len(self.g_s)-2].conv1(temp)
        # out = self.g_s[len(self.g_s)-2].leaky_relu(out)

        # out = self.g_s[8].conv2(out)
        # w2_gt = self.g_s[len(self.g_s)-2].conv2.weight
        # b2_gt = self.g_s[len(self.g_s)-2].conv2.bias
        
        # if len(w.shape) == 5:
        #     w2_gt = w2_gt.unsqueeze(0)
        #     out = batch_conv(w2_gt+w, out)
        # else:
        # out = torch.nn.functional.conv2d(out, w2_gt+w, b2_gt, padding=1)
        # out = torch.nn.functional.conv2d(out, w)
        # out = torch.nn.functional.pixel_shuffle(out, 2)
        
        # out = self.g_s[len(self.g_s)-2].leaky_relu(out)
        # identity = torch.nn.functional.interpolate(identity, scale_factor=2, mode='bilinear', align_corners=True)
        # pad = nn.ReplicationPad2d(padding=(1, 1, 1, 1))
        # identity = pad(identity)
        temp = out + identity
        
        x_hat = self.g_s[len(self.g_s)-1](temp).clamp_(0, 1)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
        
    def forward(self, x, w):
        y = self.g_a(x)
        z = self.h_a(y)
        return self.forward1(y, z, w)
 
        
def batch_conv(weights, inputs):
    b, ch, _, _ = inputs.shape
    _, ch_out, ch_in, k, _ = weights.shape
  
    weights = weights.reshape(b*ch_out, ch, k, k)
    inputs = torch.cat(torch.split(inputs, 1, dim=0), dim=1)
    out = torch.nn.functional.conv2d(inputs, weights, stride=1, padding=1,groups=b)
    out = torch.cat(torch.split(out, ch_out, dim=1), dim=0)
    
    return out

class WeightGeneratorContrast(Cheng2020AttentionAdapter):
    def __init__(self, N=192):
        super().__init__(N)
        
        self.generator_feature_extractor = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2)
        )
        
        self.generator_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.generator_fc = nn.Sequential(
            nn.Linear(N, N),
            nn.LeakyReLU(0.2),
            nn.Linear(N, N*self.dim*2)
        )

        self.N = N
        self.criterion = nn.CrossEntropyLoss().cuda()
        
    def update_weight(self, weight):
        weight0 = weight[:, :self.dim * self.N].reshape(self.g_s[8].adapter[0].weight.shape)
        weight1 = weight[:, self.dim * self.N:].reshape(self.g_s[8].adapter[1].weight.shape)
        self.g_s[8].adapter[0].weight = Parameter(weight0)  
        self.g_s[8].adapter[1].weight = Parameter(weight1)  
        return weight0, weight1
        
    def forward(self, x):
        # weight = self.weight_generator(x)
        # b,_,_,_ = weight.shape
        # weight = weight.view(b,-1)
        # weight = self.weight_generator1(weight)
        # weight0, weight1 = self.update_weight(weight)
        
        f = self.generator_feature_extractor(x)
        b, c, _, _ = f.shape
        f = self.generator_avg_pool(f).view(b, c)
        weight = self.generator_fc(f)
        q = weight[:1, :] # [1, N * dim *2]
        
        if self.training:
            # q_n = nn.functional.normalize(q.reshape(self.N*self.dim*2, -1), dim=1) # q: N*1
            q_n = nn.functional.normalize(q, dim=1)
            k = weight[1:, :] # keys: NxC
            k = nn.functional.normalize(k, dim=1)
            # positive logits: Nx1 1x1
            l_pos = torch.einsum("nc,nc->n", [q_n, q_n]).unsqueeze(-1)
            # negative logits: NxK 1x(b-1)
            # l_neg = torch.einsum("nc,ck->nk", [q_n, k])
            l_neg = torch.einsum("nc,kc->nk", [q_n, k])
            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)
            
            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            
            
        weight0, weight1 = self.update_weight(q)
        # weight0 = 0
        # weight1 = 0
        
        out = super().forward(x)
        out["weight0"] = weight0
        out["weight1"] = weight1
        if self.training:
            out["CE"] = self.criterion(logits, labels)
        return out   
        
class Cheng2020AttentionAdapterBottleNeck(Cheng2020Attention):
    def __init__(
        self,
        N=192,
        **kwargs
    ):
        super().__init__(N, **kwargs)
        
        
        self.feature_extractor = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N//3, stride=2),
            AttentionBlock(N//3),
        )
        
    #     self.conv_generator = nn.Sequential(
    #         ResidualBlockWithStride(3, N, stride=2),
    #         ResidualBlock(N, N),
    #         ResidualBlockWithStride(N, N, stride=2),
    #         AttentionBlock(N),
    #         ResidualBlock(N, N),
    #         ResidualBlockWithStride(N, N, stride=2),
    #         ResidualBlock(N, N),
    #         conv3x3(N, N, stride=2),
    #         AttentionBlock(N),
    #     )
        
    def init_origin_params(self):
        self.origin_conv1_weight = copy.deepcopy(self.g_s[8].conv1.weight)
        self.origin_conv2_weight = copy.deepcopy(self.g_s[8].conv2.weight)
        
    def compress_diff_weights(self):
        diff1 = self.g_s[8].conv1.weight - self.origin_conv1_weight
        diff2 = self.g_s[8].conv2.weight - self.origin_conv2_weight
        weight1 = compress_weight(diff1)
        weight2 = compress_weight(diff2)
        return weight1, weight2
    
    def update_weights(self, weight):
        diff = decompress_weight(weight, self.origin_conv1_weight)
        self.g_s[8].conv1.weight = Parameter(diff + self.origin_conv1_weight)  
        
    
    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
    
        params = self.h_s(z_hat)
        y_hat = self.gaussian_conditional.quantize(y, "noise" if self.training else "dequantize")
            
        ctx_params = self.context_prediction(y_hat)
        
        gaussian_params = self.entropy_parameters(torch.cat((params, ctx_params), dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
            
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        
        weight1, weight2 = self.compress_diff_weights()
        extra_bit = cal_modified_bpp_cost(weight1)
        # extra_bit = cal_modified_bpp_cost(weight1) + cal_modified_bpp_cost(weight2)
        
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "extra_bit": extra_bit
        }
        
    def load_state_dict(self, state_dict, strict: bool = True):
        super(Cheng2020Attention, self).load_state_dict(state_dict, strict=strict)
        self.init_origin_params()

