# Derived from the GPL-3.0 csnbench LPPNM implementation.
# LPPN authors: Ziye Yu and Weitao Wang.
# Upstream source: https://github.com/cangyeone/csnbench/blob/66614c5c1ffc13fcabd0f4c0b7a599e17bebb839/models/LPPNM.py
# Modified by Qineng Song in 2026 (see THIRD_PARTY_NOTICES.md).
# Changes: Student wrapper, explicit regression dimensions and logits access.
# See LICENSE and THIRD_PARTY_NOTICES.md for attribution and terms.

"""Original LPPNM network used as the LPPN-BASE starting architecture.

The layer graph, heads and forward behavior are preserved from the upstream
LPPNM implementation. Future variants must live in separate modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import fuse_modules


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        n_in,
        n_out,
        kernel_size=5,
        stride=1,
        groups=1,
        norm_layer=nn.BatchNorm2d,
    ):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                n_in,
                n_out,
                [1, kernel_size],
                stride,
                [0, padding],
                groups=groups,
                bias=False,
            ),
            norm_layer(n_out),
            nn.ReLU(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, n_in, n_out, stride, expand_ratio, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(n_in * expand_ratio))
        self.use_res = self.stride == 1 and n_in == n_out
        layers = []
        if expand_ratio != 1:
            layers.append(
                ConvBNReLU(n_in, hidden_dim, kernel_size=1, norm_layer=norm_layer)
            )
        layers.extend(
            [
                ConvBNReLU(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                ),
                nn.Conv2d(hidden_dim, n_out, 1, 1, 0, bias=False),
                norm_layer(n_out),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res:
            return x + self.conv(x)
        return self.conv(x)


class QInvertedResidual(InvertedResidual):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_res:
            return x + self.conv(x)
        return self.conv(x)

    def fuse_model(self):
        for index in range(len(self.conv)):
            if type(self.conv[index]) is nn.Conv2d:
                fuse_modules(self.conv, [str(index), str(index + 1)], inplace=True)


class ConvTBNReLU(nn.Sequential):
    def __init__(
        self,
        n_in,
        n_out,
        kernel_size=5,
        stride=1,
        padding=1,
        output_padding=1,
        bias=True,
        dilation=1,
        groups=1,
        norm_layer=nn.BatchNorm2d,
    ):
        del kernel_size, padding, output_padding, bias, dilation, groups, norm_layer
        super().__init__(
            nn.UpsamplingNearest2d(scale_factor=tuple(stride)),
            QInvertedResidual(n_in, n_out, 1, 2),
        )


class LPPNBase(nn.Module):
    """Verified LPPNM baseline graph (N=8, S=8).

    ``model_n`` and ``n_stride`` are explicit so configuration cannot claim a
    structural change that the graph did not implement. Future N/S choices
    must use a separately named model implementation.
    """

    def __init__(self, model_n=8, n_stride=8):
        super().__init__()
        if model_n != 8 or n_stride != 8:
            raise ValueError(
                "LPPNBase is verified only for model_n=8 and n_stride=8; "
                "implement a separate model variant for other N/S values."
            )
        self.model_n = model_n
        self.n_stride = n_stride
        self.layers = nn.Sequential(
            ConvBNReLU(3, 8),
            QInvertedResidual(8, 16, 2, 2),
            QInvertedResidual(16, 16, 1, 2),
            QInvertedResidual(16, 24, 2, 2),
            QInvertedResidual(24, 24, 1, 2),
            QInvertedResidual(24, 32, 2, 2),
            QInvertedResidual(32, 32, 1, 2),
        )
        self.class_encoder = nn.Sequential(
            QInvertedResidual(32, 32, 2, 2),
            QInvertedResidual(32, 32, 2, 2),
            QInvertedResidual(32, 32, 2, 2),
            ConvTBNReLU(32, 32, [1, 5], stride=[1, 2], padding=[0, 2], output_padding=[0, 1], bias=False, dilation=1),
            ConvTBNReLU(32, 32, [1, 5], stride=[1, 2], padding=[0, 2], output_padding=[0, 1], bias=False, dilation=1),
            ConvTBNReLU(32, 32, [1, 5], stride=[1, 2], padding=[0, 2], output_padding=[0, 1], bias=False, dilation=1),
        )
        self.cl = nn.Conv2d(32 * 2, 7, 1)
        self.tm = nn.Conv2d(32 * 2, 1, 1)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.qfunc = nn.quantized.FloatFunctional()

    def fuse_model(self):
        for module in self.modules():
            if type(module) is ConvBNReLU:
                fuse_modules(module, ["0", "1", "2"], inplace=True)
            if type(module) is QInvertedResidual:
                module.fuse_model()

    def forward(self, x):
        x = x.unsqueeze(2)
        x1 = self.layers(x)
        x2 = self.class_encoder(x1)
        x = torch.cat([x1, x2], dim=1)
        out_class = self.cl(x).squeeze(dim=2)
        out_time = self.tm(x)
        out_time = out_time.sigmoid().squeeze(dim=2).squeeze(dim=1) * self.n_stride
        if self.training:
            return out_class, out_time
        out_class = F.softmax(out_class, dim=1)
        return out_class, out_time


def forward_logits(model: nn.Module, wave: torch.Tensor):
    """Return logits while retaining eval-mode BatchNorm behavior."""

    x = wave.unsqueeze(2)
    x1 = model.layers(x)
    x2 = model.class_encoder(x1)
    x = torch.cat([x1, x2], dim=1)
    out_class = model.cl(x).squeeze(dim=2)
    out_time = model.tm(x).sigmoid().squeeze(dim=2).squeeze(dim=1) * model.n_stride
    return out_class, out_time
