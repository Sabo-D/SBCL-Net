import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.RRB import ConvBNAct


class FactorizedLargeKernelDWConv(nn.Module):
    """
    Factorized depthwise large-kernel convolution.

    k x k  -->  1 x k  +  k x 1

    This module approximates a large-kernel depthwise response
    with lower computational cost.

    Abbreviation:
        FLKDWConv = FactorizedLargeKernelDWConv
    """
    def __init__(self, channels, kernel_size):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd."

        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(1, kernel_size),
                padding=(0, pad),
                groups=channels,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(kernel_size, 1),
                padding=(pad, 0),
                groups=channels,
                bias=False
            ),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return self.block(x)


class MultiScaleEdgeResponseUnit(nn.Module):
    """
    Multi-scale edge response unit.

    For a given scale, it models:
    - peripheral large-kernel response
    - center local response
    - learnable center suppression

    Abbreviation:
        MERU = MultiScaleEdgeResponseUnit
    """
    def __init__(self, channels, kernel_size):
        super().__init__()

        self.peripheral_branch = nn.Sequential(
            FactorizedLargeKernelDWConv(channels, kernel_size),
            ConvBNAct(channels, channels, kernel_size=1, act=False)
        )

        self.center_branch = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False
            ),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        peripheral_feat = self.peripheral_branch(x)
        center_feat = self.center_branch(x)
        edge_feat = F.gelu(peripheral_feat - 0.5 * center_feat)
        return edge_feat


class BoundaryStructureExtractionModule(nn.Module):
    """
    Boundary Structure Extraction Module
    BSEM
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=32,
        hidden_ratio=0.5,
        scales=(5, 7, 9)
    ):
        super().__init__()

        assert len(scales) >= 2, "scales should contain at least 2 kernel sizes."
        assert all(k % 2 == 1 for k in scales), "All scale kernels must be odd."

        hidden_channels = max(8, int(out_channels * hidden_ratio))
        self.scales = scales
        self.hidden_channels = hidden_channels
        self.num_scales = len(scales)

        # ------------------------------------------------------------------
        # Fixed structural prior buffers
        # ------------------------------------------------------------------
        laplacian_kernel = torch.tensor(
            [[0, -1,  0],
             [-1, 4, -1],
             [0, -1,  0]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        sobel_kernel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        sobel_kernel_y = torch.tensor(
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        rgb_to_gray = torch.tensor(
            [0.299, 0.587, 0.114],
            dtype=torch.float32
        ).view(1, 3, 1, 1)

        self.register_buffer("laplacian_kernel", laplacian_kernel)
        self.register_buffer("sobel_kernel_x", sobel_kernel_x)
        self.register_buffer("sobel_kernel_y", sobel_kernel_y)
        self.register_buffer("rgb_to_gray", rgb_to_gray)

        # ------------------------------------------------------------------
        # Branch 1: explicit prior encoding
        # ------------------------------------------------------------------
        self.prior_encoder = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        # ------------------------------------------------------------------
        # Branch 2: frequency-adaptive dynamic modeling
        # ------------------------------------------------------------------
        self.feature_stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),

            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                bias=False
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )

        # Descriptor channels = 3: gradient / variance / detail
        self.scale_gate = nn.Conv2d(
            3,
            self.num_scales,
            kernel_size=1,
            bias=True
        )

        self.multi_scale_units = nn.ModuleList([
            MultiScaleEdgeResponseUnit(hidden_channels, kernel_size=k)
            for k in scales
        ])

        self.dynamic_encoder = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        # ------------------------------------------------------------------
        # Prior-guided gated fusion
        # ------------------------------------------------------------------
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

        self.output_proj = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    @staticmethod
    def _align_dtype(buffer_tensor, ref_tensor):
        return buffer_tensor.to(dtype=ref_tensor.dtype, device=ref_tensor.device)

    def _fixed_depthwise_conv(self, x, kernel):
        """
        Apply a fixed 3x3 kernel in depthwise mode.

        Args:
            x: [B, C, H, W]
            kernel: [1, 1, 3, 3]

        Returns:
            y: [B, C, H, W]
        """
        channels = x.size(1)
        weight = kernel.repeat(channels, 1, 1, 1)
        weight = self._align_dtype(weight, x)
        return F.conv2d(x, weight, padding=1, groups=channels)

    def _build_frequency_descriptor(self, feat):
        """
        Build the frequency-aware descriptor from:
            - gradient magnitude
            - local variance
            - local detail residual

        Args:
            feat: [B, C, H, W]

        Returns:
            desc: [B, 3, H, W]
        """
        grad_x = self._fixed_depthwise_conv(feat, self.sobel_kernel_x)
        grad_y = self._fixed_depthwise_conv(feat, self.sobel_kernel_y)
        grad_mag = torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-6)

        local_mean = F.avg_pool2d(feat, kernel_size=3, stride=1, padding=1)
        local_mean_sq = F.avg_pool2d(feat * feat, kernel_size=3, stride=1, padding=1)
        local_var = torch.clamp(local_mean_sq - local_mean * local_mean, min=0.0)

        local_detail = torch.abs(feat - local_mean)

        grad_mag = grad_mag.mean(dim=1, keepdim=True)
        local_var = local_var.mean(dim=1, keepdim=True)
        local_detail = local_detail.mean(dim=1, keepdim=True)

        desc = torch.cat([grad_mag, local_var, local_detail], dim=1)
        return desc

    def _extract_explicit_prior(self, x):
        """
        Explicit structural prior branch:
            RGB -> grayscale -> Laplacian response -> prior encoder
        """
        rgb_to_gray = self._align_dtype(self.rgb_to_gray, x)
        laplacian_kernel = self._align_dtype(self.laplacian_kernel, x)

        gray = (x * rgb_to_gray).sum(dim=1, keepdim=True)
        prior_edge = torch.abs(F.conv2d(gray, laplacian_kernel, padding=1))
        prior_feat = self.prior_encoder(prior_edge)
        return prior_feat

    def _extract_dynamic_prior(self, x):
        """
        Frequency-adaptive dynamic branch:
            stem -> descriptor -> scale gating -> multi-scale response -> projection
        """
        stem_feat = self.feature_stem(x)

        descriptor = self._build_frequency_descriptor(stem_feat)
        scale_weights = F.softmax(self.scale_gate(descriptor), dim=1)  # [B, 3, H, W]

        multi_scale_feats = [
            unit(stem_feat) for unit in self.multi_scale_units
        ]   # [B, hidden_channels, H, W]

        dynamic_feat = torch.zeros_like(multi_scale_feats[0])
        for idx, feat_i in enumerate(multi_scale_feats):
            dynamic_feat = dynamic_feat + scale_weights[:, idx:idx + 1] * feat_i

        dynamic_feat = self.dynamic_encoder(dynamic_feat)
        return dynamic_feat

    def forward(self, x):
        """
        Args:
            x: input image tensor [B, 3, H, W]

        Returns:
            boundary_feat: boundary-aware structural feature [B, out_channels, H, W]
        """
        prior_feat = self._extract_explicit_prior(x)
        dynamic_feat = self._extract_dynamic_prior(x)

        fusion_weight = self.fusion_gate(torch.cat([prior_feat, dynamic_feat], dim=1))
        fused_feat = prior_feat + fusion_weight * dynamic_feat

        boundary_feat = self.output_proj(fused_feat)
        return [boundary_feat,prior_feat]


# -----------------------------------------------------------------------------
# Quick test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    x = torch.randn(2, 3, 256, 256)

    model = BoundaryPriorExtractionModule(
        in_channels=3,
        out_channels=64,
        hidden_ratio=0.5,
        scales=(5, 7, 9)
    )

    y = model(x)

    print("Module : Boundary Prior Extraction Module (BPEM)")
    print("Input  :", x.shape)
    print("Output :", y.shape)
