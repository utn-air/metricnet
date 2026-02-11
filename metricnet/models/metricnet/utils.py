import math

import torch
from torch import nn


class TokenBatchNorm1d(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


class Conv2dResidualBlock(nn.Module):
    def __init__(self, d_in: int = 1280, d: int = 384, k: int = 5) -> None:
        super().__init__()
        self.in1 = nn.Conv2d(d_in, d, 1, bias=False)
        self.in1_bn = nn.BatchNorm2d(d)

        self.dw = nn.Conv2d(d, d, k, padding=k // 2, groups=d, bias=False)
        self.dw_bn = nn.BatchNorm2d(d)

        self.act = nn.GELU(approximate="tanh")
        self.se = nn.Sequential(  # squeeze and excite block
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(d, max(4, d // 4), 1, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Conv2d(max(4, d // 4), d, 1, bias=True),
            nn.Sigmoid(),
        )
        self.out1 = nn.Conv2d(d, d, 1, bias=False)
        self.out1_bn = nn.BatchNorm2d(d)

        self.post_bn_tokens = TokenBatchNorm1d(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.in1_bn(self.in1(x)))
        h = self.act(self.dw_bn(self.dw(y)))
        y = self.act(y + self.out1_bn(self.out1(h * self.se(h))))
        return self.post_bn_tokens(y.flatten(2).transpose(1, 2))


class LinearResidualBlock(nn.Module):
    def __init__(self, d: int = 384, expansion: int = 4, drop: float = 0.0) -> None:
        super().__init__()
        h = int(d * expansion)
        self.in1 = nn.Sequential(
            nn.Linear(d, h, bias=False),
            TokenBatchNorm1d(h),
            nn.GELU(approximate="tanh"),
        )
        self.in2 = nn.Sequential(
            nn.Linear(h, d, bias=False),
            TokenBatchNorm1d(d),
            nn.GELU(approximate="tanh"),
        )
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.drop(self.in2(self.in1(x))))


class PositionalEncoding(nn.Module):
    # from NoMaD at github.com/robodhruv/visualnav-transformer
    def __init__(self, d_model, max_seq_len=6):
        super().__init__()

        pos_enc = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        pos_enc = pos_enc.unsqueeze(0)
        self.register_buffer("pos_enc", pos_enc)

    def forward(self, x):
        x = x + self.pos_enc[:, : x.size(1), :]
        return x
