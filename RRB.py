import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    """
    Convolution-BatchNorm-Activation block.

    Abbreviation:
        CBA = ConvBNAct
    """
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
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

# in-->out_UNetPvt           out_UNetPvt-->out_UNetPvt
# conv3x3 bn act --> conv3x3 bn --> (x+out_UNetPvt) act
class ResidualRefinementBlock(nn.Module):
    """
    RRB
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = ConvBNAct(
            in_channels,
            out_channels,
            kernel_size=3,
            act=True
        )
        self.conv2 = ConvBNAct(
            out_channels,
            out_channels,
            kernel_size=3,
            act=False
        )

        if in_channels != out_channels:
            self.shortcut = ConvBNAct(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                act=False
            )
        else:
            self.shortcut = nn.Identity()

        self.act = nn.GELU()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.act(out + identity)
        return out


if __name__ == "__main__":
    x = torch.randn(2, 64, 128, 128)
    model = FeatureRefinementBlock(64, 64)
    y = model(x)
    print("Module :", "Feature Refinement Block (FRB)")
    print("Input  :", x.shape)
    print("Output :", y.shape)
