import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        groups=1,
        act=True,
        bias=False
    ):
        super().__init__()
        if isinstance(kernel_size, tuple):
            if padding is None:
                padding = tuple(k // 2 for k in kernel_size)
        else:
            if padding is None:
                padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class LightweightContextMixer(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.dw_h = ConvBNAct(
            channels, channels,
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2),
            groups=channels,
            act=True
        )
        self.dw_v = ConvBNAct(
            channels, channels,
            kernel_size=(kernel_size, 1),
            padding=(kernel_size // 2, 0),
            groups=channels,
            act=True
        )
        self.pw = ConvBNAct(
            channels, channels,
            kernel_size=1,
            padding=0,
            act=False
        )

    def forward(self, x):
        return self.pw(self.dw_v(self.dw_h(x)))


class DSMM(nn.Module):
    """
    Discrepancy-Guided Selective Modulation Module
    """
    def __init__(
        self,
        master_channels,
        slave_channels,
        embed_dim=None,
        kernel_size=5,
        scale_factor=0.25,
        bias_factor=0.05,
        mixer_residual_ratio=0.3
    ):
        super().__init__()

        if embed_dim is None:
            embed_dim = max(16, slave_channels // 2)

        self.scale_factor = scale_factor
        self.bias_factor = bias_factor
        self.mixer_residual_ratio = mixer_residual_ratio

        self.master_proj = ConvBNAct(master_channels, embed_dim, kernel_size=1, padding=0, act=True)
        self.slave_proj = ConvBNAct(slave_channels, embed_dim, kernel_size=1, padding=0, act=True)

        # only keep base + diff
        self.context_reduce = ConvBNAct(embed_dim * 2, embed_dim, kernel_size=1, padding=0, act=True)

        self.context_mixer = LightweightContextMixer(embed_dim, kernel_size=kernel_size)

        # local gate only
        self.gate_head = nn.Sequential(
            ConvBNAct(embed_dim, embed_dim, kernel_size=3, act=True),
            nn.Conv2d(embed_dim, 1, kernel_size=1, bias=True)
        )

        # shared lightweight modulation heads
        self.scale_head = nn.Conv2d(embed_dim, slave_channels, kernel_size=1, bias=True)
        self.bias_head = nn.Conv2d(embed_dim, slave_channels, kernel_size=1, bias=True)

    def forward(self, master, slave):
        if master.shape[-2:] != slave.shape[-2:]:
            master = F.interpolate(
                master,
                size=slave.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        master_embed = self.master_proj(master)
        slave_embed = self.slave_proj(slave)

        base_context = master_embed + slave_embed
        diff_context = torch.abs(master_embed - slave_embed)

        context = torch.cat([base_context, diff_context], dim=1)
        context = self.context_reduce(context)

        mixed = self.context_mixer(context)
        context = context + self.mixer_residual_ratio * mixed

        gate = torch.sigmoid(self.gate_head(diff_context))   # [B,1,H,W]

        scale = self.scale_factor * torch.tanh(self.scale_head(context))
        bias = self.bias_factor * torch.tanh(self.bias_head(context))

        refined_slave = slave * (1.0 + scale * gate) + bias * gate

        return refined_slave

    # slave + scale * gate * slave + bias * gate
    # slava + gate(scale * slave + bias)
