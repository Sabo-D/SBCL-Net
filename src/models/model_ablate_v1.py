import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class BasicConv_1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 主路径（两卷积）
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)

        # 残差路径（处理通道变化）
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.residual(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.relu(out + identity)

class DecoderBlock(nn.Module):
    """
    上采样 lgag concat 卷积 mgfab
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ResBlock(out_channels + skip_channels, out_channels)
        self.lgag = LGAG(skip_channels, skip_channels)
        self.mgfab = MGAFB(out_channels)

    def forward(self, x, skip):
        x = self.up(x)  # 320
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        skip = self.lgag(x, skip)
        x_fuse = torch.cat([x, skip], dim=1)
        x = self.conv(x_fuse)
        x = self.mgfab(x)

        return x

class PVTBackbone(nn.Module):
    def __init__(self, model_name='pvt_v2_b2', pretrained=True):
        super(PVTBackbone, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        self.out_channels = [f['num_chs'] for f in self.backbone.feature_info]  # e.g. [64, 128, 320, 512]

    def forward(self, x):
        return self.backbone(x)

class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        x_cat = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        x_sa = self.conv(x_cat)
        return x * self.sigmoid(x_sa)

class SABWeight(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        x_cat = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        x_sa = self.conv(x_cat)
        weight = self.sigmoid(x_sa)
        return weight

class CBAM(nn.Module):
    def __init__(self, ch_in, reduction=4, spatial_kernel=7):
        super(CBAM, self).__init__()
        self.channel_attention = CABWeight(ch_in, reduction)
        self.spatial_attention = SAB(kernel_size=spatial_kernel)

    def forward(self, x):
        x_ca = x * self.channel_attention(x)     # 通道注意力
        x_sa = self.spatial_attention(x_ca)      # 空间注意力
        out = x + x_sa                           # 残差连接
        return out

class CABWeight(nn.Module):
    def __init__(self, ch_in, reduction=4):
        super(CABWeight, self).__init__()
        self.ch_in = ch_in
        if self.ch_in < reduction:
            reduction = self.ch_in
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch_in, ch_in//reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in//reduction, ch_in, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x1 = self.avg_pool(x)
        x1 = self.fc(x1)
        x2 = self.max_pool(x)
        x2 = self.fc(x2)
        weight = self.sigmoid(x1 + x2)
        return weight

class MGAFB(nn.Module):
    """
    Multi-Scale Grouped Attention Fusion Block
    多尺度分组注意力融合模块
    """
    def __init__(self, ch_in, kernel_sizes=[1, 3, 5, 7], reduction=4):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_in // 4
        self.softmax = nn.Softmax(dim=1)
        """
        MSGConv
        多尺度分组卷积
        """
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch_in, self.ch_out, k, padding=k // 2, groups=self.ch_in // 4, bias=False),
                nn.BatchNorm2d(self.ch_out),
                nn.ReLU(inplace=True)
            )for k in kernel_sizes
        ])

        self.se_blocks = nn.ModuleList([
            CABWeight(self.ch_out, reduction=reduction)
            for _ in range(4)
        ])
        self.sab = SAB()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.shape[0]
        height = x.shape[2]
        width = x.shape[3]
        feats = [conv(x) for conv in self.conv_blocks]  # 每个 shape: [B, C/4, H, W]
        ses = [se(feat) for feat, se in zip(feats, self.se_blocks)]  # 每个 shape: [B, C/4, 1, 1]

        fused = torch.cat(feats, dim=1)      # [B, C, H, W]
        fused = fused.view(batch_size, 4, self.ch_out, height, width)  # [B, 4, C/4, H, W]
        ses_fused = torch.cat(ses, dim=1)    # [B, C, 1, 1]
        ses_fused = ses_fused.view(batch_size, 4, self.ch_out, 1, 1)  # [B, 4, C/4, 1, 1]
        attn = self.softmax(ses_fused)
        final = fused * attn  # [B, 4, C/4, H, W]

        out = final.permute(0, 2, 1, 3, 4)  # [B, C/4, 4, H, W]
        out = out.reshape(batch_size, self.ch_in, height, width)
        out = self.sab(out)

        return x + out

class LGAG(nn.Module):
    """
    LGAG Large-kernel Grouped Attention Gate (LGAG) Block
    大核分组门控注意力模块
    """
    def __init__(self, ch_feature_high, ch_feature_low, kernel_size=3):
        super(LGAG, self).__init__()
        self.ch_mid = ch_feature_low // 2
        self.groups = ch_feature_low // 2
        # 高层语义通道压缩分组卷积
        self.W_g = nn.Sequential(
            nn.Conv2d(ch_feature_high, self.ch_mid, kernel_size=kernel_size, stride=1,
                      padding=kernel_size // 2, groups=self.groups, bias=True),
            nn.BatchNorm2d(self.ch_mid),
        )
        # 低层语义通道压缩分组卷积
        self.W_x = nn.Sequential(
            nn.Conv2d(ch_feature_low, self.ch_mid, kernel_size=kernel_size, stride=1,
                      padding=kernel_size // 2, groups=self.groups, bias=True),
            nn.BatchNorm2d(self.ch_mid),
        )
        # concat 后通道交互卷积
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * self.ch_mid, self.ch_mid, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(self.ch_mid),
            nn.Sigmoid()
        )

        # 注意力权重模块
        self.psi = nn.Sequential(
            nn.Conv2d(self.ch_mid, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        fused = torch.cat([g1, x1], dim=1)
        fused = self.fusion(fused)
        psi = self.psi(fused)
        # 低层语义
        return x * psi

class MKEFE(nn.Module):
    """
    Multi-Kernel Edge Feature Extractor
    多核边缘特征提取器
    """
    def __init__(self, out_channels=32, learnable_weight=True):
        super().__init__()

        #固定的边缘检测核（5 个）
        self.kernels = nn.Parameter(torch.tensor([
            [[1, 1, 1], [1, -8, 1], [1, 1, 1]],           # Laplacian
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],         # Sobel X
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],         # Sobel Y
            [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],         # Prewitt X
            [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]          # Prewitt Y
        ], dtype=torch.float32).unsqueeze(1), requires_grad=False)  # shape: (5,1,3,3)

        self.rgb_weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)

        if learnable_weight:
            self.edge_weights = nn.Parameter(torch.ones(5) / 5)  # 可以训练
        else:
            self.register_buffer('edge_weights', torch.ones(5))  # 固定平均

        self.fuse = ResBlock(5, out_channels)
        self.mgafb = MGAFB(out_channels)

    def forward(self, x):
        gray = (x * self.rgb_weights.to(x.device)).sum(dim=1, keepdim=True)

        edge_maps = []
        for i in range(5):
            """
            Multi-Kernel Edge Response
            多核边缘响应
            """
            edge = F.conv2d(gray, self.kernels[i:i+1].to(x.device), padding=1)
            edge = torch.abs(edge) * self.edge_weights[i]
            edge_maps.append(edge)  # 每个edge: (B,1,H,W)

        edge_stack = torch.cat(edge_maps, dim=1)
        edge_feat = self.fuse(edge_stack)  # (B, out_channels, H, W)
        edge_feat = self.mgafb(edge_feat)

        return edge_feat

class BEB(nn.Module):
    """
    边界加强模块
    Boundary Enhancement Block(BEB)
    """
    def __init__(self):
        super(BEB, self).__init__()
        self.conv_edge = BasicConv_1(32, 16)
        self.conv_mask = BasicConv_1(32, 16)
        self.attn_s = SABWeight()
        self.attn_c = CABWeight(16)
        self.conv_out = BasicConv_1(16, 32)
        self.msdf = MGAFB(16, kernel_sizes=[1, 3, 5, 7], reduction=4)

    def forward(self, f_edge, f_mask):
        feat_edge = self.conv_edge(f_edge)
        feat_mask = self.conv_mask(f_mask)
        attn = self.attn_s(feat_edge) * self.attn_c(feat_edge)
        features = feat_mask * attn + feat_mask

        out = self.msdf(features)
        out = self.conv_out(out)

        return out

class PVT_MFA_ablate_v1(nn.Module):
    """
    V0 + Boundary Branch (MKEFE + CBAM + ResBlock)
    Adds a structure-aware boundary prediction path;
    however, it operates independently without any semantic interaction.
    """

    def __init__(self, num_classes=1, backbone_name='pvt_v2_b2'):
        super(PVT_MFA_ablate_v1, self).__init__()
        self.encoder = PVTBackbone(backbone_name)
        channels = self.encoder.out_channels  # [64, 128, 320, 512]

        self.center = ResBlock(channels[-1], channels[-1])
        self.decoder4 = DecoderBlock(channels[3], channels[2], channels[2])
        self.decoder3 = DecoderBlock(channels[2], channels[1], channels[1])
        self.decoder2 = DecoderBlock(channels[1], channels[0], channels[0])
        self.decoder1 = DecoderBlock(channels[0], 32, 32)

        self.skip0 = ResBlock(3, 32)  # Only used in decoder

        self.out_mask = nn.Conv2d(32, num_classes, kernel_size=1)

        # Independent edge branch (structure path only)
        self.mkefe = MKEFE()
        self.cbam = CBAM(32)
        self.edge_conv = ResBlock(32, 32)
        self.out_edge = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        skips = self.encoder(x)
        x0 = self.skip0(x)  # structure feature (only for decoder)

        e1, e2, e3, e4 = skips

        # Main decoder path
        center = self.center(e4)
        d4 = self.decoder4(center, e3)
        d3 = self.decoder3(d4, e2)
        d2 = self.decoder2(d3, e1)
        d1 = self.decoder1(d2, x0)
        mask = self.out_mask(d1)

        # Edge branch (independent)
        f_edge = self.mkefe(x)  # [B, 32, H, W]
        f_edge = self.cbam(f_edge)  # attention refinement
        f_edge = self.edge_conv(f_edge)
        edge = self.out_edge(f_edge)

        return [mask, edge]

if __name__ == '__main__':
    x = torch.rand(8, 3, 32, 32)
    model = PVT_MFA_ablate_v1()
    y = model(x)
    print(len(y))
    print(y[0].shape)
    print(y[1].shape)