from typing import Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from compressai.entropy_models import EntropyModel
from torch import distributions as D


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


class WeightEntropyModule(EntropyModel):
    """entropy module for network parameters
    width * [- (self.n_bins // 2 - 1), ..., -1, 0, 1, 2, ..., self.n_bins //  2 - 1]
    e.g.) n_bins = 56, pmf_lengths = 55

    cdf: 1 / (1 + alpha) * slab + alpha / (1 + alpha) * spike
      spike: N (0, width / 6)
      slab: N(0, sigma)

    quantization interval: width
    """

    def __init__(
        self, cdf: Callable, width: float = 5e-3, data_type: str = "float32", **kwargs
    ):
        super().__init__(**kwargs)
        self.cdf = cdf
        self.width: float = width
        self._tail_mass = 1e-9
        # used for compression
        self.data_type = data_type

        self.register_buffer("_n_bins", torch.IntTensor())
        self.update(force=True)

    def update(self, force: bool = False) -> bool:
        if self._n_bins.numel() > 0 and not force:
            return False

        delta = self.width / 2
        # accept self.width * 10000 * interval difference at maximum
        intervals: torch.Tensor = torch.arange(1, 10001)
        upper = self._likelihood_cumulative(
            intervals * self.width + delta, stop_gradient=True
        )
        lower = self._likelihood_cumulative(
            -intervals * self.width - delta, stop_gradient=True
        )
        # (upper - lower) - (1 - self._tail_mass)
        diff: torch.Tensor = self._tail_mass - lower - (1 - upper)
        if not (diff >= 0).any():
            self._n_bins = intervals[-1]
        else:
            n_bins = intervals[diff.argmax()]
            # even value
            # self._n_bins = ((n_bins - 1) // 2 + 1) * 2
            self._n_bins = (torch.div(n_bins - 1, 2, rounding_mode="trunc") + 1) * 2
        self._n_bins = self._n_bins.reshape((1,))

        # bound = (self._n_bins - 1) // 2
        bound = torch.div(self._n_bins - 1, 2, rounding_mode="trunc")
        bound = torch.clamp(bound.int(), min=0)

        self._offset = -bound

        pmf_start = -bound
        pmf_length = 2 * bound + 1

        max_length = pmf_length.max().item()
        device = pmf_start.device
        samples = torch.arange(max_length, device=device)

        samples = samples[None, :] + pmf_start[:, None, None]

        half = self.width / 2

        lower = self._likelihood_cumulative(
            samples * self.width - half, stop_gradient=True
        )
        upper = self._likelihood_cumulative(
            samples * self.width + half, stop_gradient=True
        )
        pmf = upper - lower

        pmf = pmf[:, 0, :]
        tail_mass = lower[:, 0, :1] + (1 - upper[:, 0, -1:])

        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._cdf_length = pmf_length + 2
        return True

    def quantize(self, w: torch.Tensor, mode: str, means=None) -> torch.Tensor:
        if mode == "noise":
            assert self.training
            # add uniform noise: [-self.width / 2, self.width / 2]
            noise = (torch.rand_like(w) - 0.5) * self.width
            return w + noise

        symbols: torch.Tensor = torch.round(w / self.width)
        if mode == "symbols":
            # bound: torch.Tensor = (self._n_bins - 1) // 2
            bound: torch.Tensor = torch.div(self._n_bins - 1, 2, rounding_mode="trunc")
            symbols = torch.min(torch.max(symbols, -bound), bound)
            return symbols.int()
        elif mode == "dequantize":
            w_bound: torch.Tensor = (self._n_bins - 1) * self.width / 2
            # clamp with (-w_bound, w_bound)
            w_hat: torch.Tensor = torch.min(
                torch.max(symbols * self.width, -w_bound), w_bound
            )
            return (w_hat - w).detach() + w
        else:
            raise NotImplementedError

    def dequantize(
        self, inputs: torch.Tensor, means=None, dtype: torch.dtype = torch.float
    ) -> torch.Tensor:
        outputs = (inputs * self.width).type(dtype)
        return outputs

    # modified from _logits_cumulative
    def _likelihood_cumulative(
        self, inputs: torch.Tensor, stop_gradient: bool
    ) -> torch.Tensor:
        if stop_gradient:
            with torch.no_grad():
                return self.cdf(inputs)
        else:
            return self.cdf(inputs)

    def _likelihood(self, inputs: torch.Tensor) -> torch.Tensor:
        delta = self.width / 2
        v0 = inputs - delta
        v1 = inputs + delta
        lower = self._likelihood_cumulative(v0, stop_gradient=False)
        upper = self._likelihood_cumulative(v1, stop_gradient=False)
        likelihood = upper - lower
        return likelihood

    def forward(self, x: torch.Tensor, training=None) -> tuple:
        if self.width == 0:
            outputs = x
            likelihood = torch.ones_like(x) * (2 ** -32)
            return outputs, likelihood

        if training is None:
            training = self.training

        if not torch.jit.is_scripting():
            # x from B x C x ... to C x B x ...
            perm = np.arange(len(x.shape))
            perm[0], perm[1] = perm[1], perm[0]
            # Compute inverse permutation
            inv_perm = np.arange(len(x.shape))[np.argsort(perm)]
        else:
            # TorchScript in 2D for static inference
            # Convert to (channels, ... , batch) format
            perm = (1, 2, 3, 0)
            inv_perm = (3, 0, 1, 2)

        x = x.permute(*perm).contiguous()
        shape = x.size()
        values = x.reshape(x.size(0), 1, -1)

        # Add noise or quantize
        outputs = self.quantize(values, "dequantize")
        outputs_ent = self.quantize(values, "noise") if self.training else outputs

        likelihood = self._likelihood(outputs_ent)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)

        # Convert back to input tensor shape
        outputs = outputs.reshape(shape)
        outputs = outputs.permute(*inv_perm).contiguous()

        likelihood = likelihood.reshape(shape)
        likelihood = likelihood.permute(*inv_perm).contiguous()

        return outputs, likelihood

    @staticmethod
    def _build_indexes(size):
        dims = len(size)
        N = size[0]
        C = size[1]

        view_dims = np.ones((dims,), dtype=np.int64)
        view_dims[1] = -1
        indexes = torch.arange(C).view(*view_dims)
        indexes = indexes.int()

        return indexes.repeat(N, 1, *size[2:])

    def compress(self, x):
        if self.width == 0:
            strings = list()
            for i in range(len(x)):
                string = encode_array(x[i].flatten().cpu().numpy(), self.data_type)
                strings.append(string)
            return strings

        indexes = self._build_indexes(x.size())
        return super().compress(x, indexes)
    
    def compress_spike_and_slab(self, x):
        w_shape = x.reshape(1, 1, -1).shape
        x = x.reshape(w_shape)
    
        indexes = self._build_indexes(x.size())
        symbols = self.quantize(x, "symbols")

        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()

        strings = []
        for i in range(symbols.size(0)):
            rv = self.entropy_coder.encode_with_indexes(
                symbols[i].reshape(-1).int().tolist(),
                indexes[i].reshape(-1).int().tolist(),
                self._quantized_cdf.tolist(),
                self._cdf_length.reshape(-1).int().tolist(),
                self._offset.reshape(-1).int().tolist(),
            )
            strings.append(rv)
        return strings

    def decompress(self, strings, size):
        output_size = (len(strings), self._quantized_cdf.size(0), *size)
        if self.width == 0:
            xs = list()
            for string in strings:
                x = decode_array(string, self.data_type)
                x = torch.from_numpy(x.copy()).to(self._quantized_cdf.device)
                xs.append(x)
            xs = torch.stack(xs).float().reshape(output_size)
            return xs

        indexes = self._build_indexes(output_size).to(self._quantized_cdf.device)
        return super().decompress(strings, indexes, torch.float32)


def encode_array(x: np.ndarray, data_type: str) -> bytes:
    if data_type == "float32":
        return x.astype(np.float32).tobytes()
    if data_type == "float16":
        return x.astype(np.float16).tobytes()
    # Zou+, ISM 21
    elif data_type == "uint8":
        bias = x.min()
        x_ = x - bias
        scale: float = (255 / x_.max()).astype(np.float32)
        arr_qua = np.round(x_ * scale).astype(np.uint8)
        return arr_qua.tobytes() + bias.tobytes() + scale.tobytes()
    else:
        raise NotImplementedError


def decode_array(string: bytes, data_type: str) -> np.ndarray:
    if data_type == "float32":
        return np.frombuffer(string, dtype=np.float32)
    if data_type == "float16":
        return np.frombuffer(string, dtype=np.float16).astype(np.float32)
    # Zou+, ISM 21
    elif data_type == "uint8":
        arr = np.frombuffer(string[:-8], dtype=np.uint8)
        bias = np.frombuffer(string[-8:-4], dtype=np.float32)
        scale = np.frombuffer(string[-4:], dtype=np.float32)
        return arr / scale + bias
    else:
        raise NotImplementedError

class SpikeAndSlabCDF:
    def __init__(
        self, width: float = 5e-3, sigma: float = 5e-2, alpha: float = 1000, mean=torch.tensor(0.0)
    ) -> None:
        self.alpha = alpha
        self.slab = D.Normal(mean, torch.tensor(sigma))
        if width != 0:
            self.spike = D.Normal(mean, torch.tensor(width / 6))
        else:
            self.spike = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        cdf_slab = self.slab.cdf(x)
        if self.spike is None:
            return cdf_slab
        else:
            cdf_spike = self.spike.cdf(x)
            return (cdf_slab + self.alpha * cdf_spike) / (1 + self.alpha)

class LogisticCDF:

    def __init__(self, scale: float, loc: float = 0.0) -> None:
        self.loc = loc
        self.scale = scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 + 0.5 * torch.tanh((x - self.loc) / self.scale / 2)

# def cal_adapter_bpp_cost(net):
#     device = 'cuda'
#     adapter_dict = dict()
#     for name, p in net.named_parameters():
#         if "adapter" in name:
#             adapter_dict[name] = p

#     extra_bit_sum = 0
#     for key, param in adapter_dict.items():
#         distrib = LogisticCDF(scale=0.05)
#         w_ent = WeightEntropyModule(distrib, 0.06, data_type='uint8').to(device)
#         w_shape = param.reshape(1, 1, -1).shape
#         p_init = torch.zeros_like(param)
#         diff = (param - p_init).reshape(w_shape)
#         weight = w_ent.compress(diff)
#         extra_bit = len(weight[0]) * 8
#         extra_bit_sum += extra_bit
        
#     return extra_bit_sum

# def cal_extra_bpp_cost(net, loc=0.0):
#     device = 'cuda'
#     adapter_dict = dict()
#     for name, p in net.named_parameters():
#         if "adapter" in name:
#             adapter_dict[name] = p

#     extra_bit_sum = 0
#     for key, param in adapter_dict.items():
#         # distrib = LogisticCDF(scale=0.05)
#         distrib = SpikeAndSlabCDF(mean=loc)
#         w_ent = WeightEntropyModule(distrib, 0.06, data_type='uint8').to(device)
#         w_shape = param.reshape(1, 1, -1).shape
#         p_init = torch.zeros_like(param)
#         diff = (param - p_init).reshape(w_shape)
#         weight = w_ent.compress(diff)
#         extra_bit = len(weight[0]) * 8
#         # print(param)
#         # print(key, param.shape, w_shape, extra_bit,)
#         extra_bit_sum += extra_bit
        
#     return extra_bit_sum

# def compress_weight(param, width=5e-3):
#     device = 'cuda'
#     distrib = SpikeAndSlabCDF(width=width)
#     w_ent = WeightEntropyModule(distrib, width=width, data_type='uint8').to(device)
#     w_shape = param.reshape(1, 1, -1).shape
#     diff = (param).reshape(w_shape)
#     weight = w_ent.compress(diff)
#     return weight

# def decompress_weight(weight, param, width=5e-3):
#     device = 'cuda'
#     distrib = SpikeAndSlabCDF(width=width)
#     w_ent = WeightEntropyModule(distrib, width=width, data_type='uint8').to(device)
#     diff = w_ent.decompress(weight, (param.numel(),))
#     diff = diff.reshape(param.shape)
#     return diff

# def cal_modified_bpp_cost(weight):
#     return len(weight[0]) * 8