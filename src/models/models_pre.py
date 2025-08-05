import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class PVTBackbone(nn.Module):
    def __init__(self, model_name='pvt_v2_b2', pretrained=True):
        super(PVTBackbone, self).__init__()
        # 从 timm 加载 PVT 模型，features_only=True 表示只返回多层特征
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        self.out_channels = [f['num_chs'] for f in self.backbone.feature_info]  # 每层输出通道数

    def forward(self, x):
        features = self.backbone(x)
        return features

class BasicConv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=1, padding=0):
        super(BasicConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class DSRUB(nn.Module):
    """
    Depthwise-Separable Residual Upsampling(DSRU) Block
    深度可分离残差上采样块
    """
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(DSRUB, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)

        # 主分支：Depthwise + Pointwise
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                                   groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn_main = nn.BatchNorm2d(out_channels)

        # shortcut 分支：1x1 Conv + BN
        self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn_short = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

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
        x_up = self.upsample(x)

        # 主路径
        x1 = self.depthwise(x_up)
        x1 = self.pointwise(x1)
        x1 = self.bn_main(x1)

        # 残差路径
        x2 = self.shortcut_conv(x_up)
        x2 = self.bn_short(x2)

        # 残差融合
        y = x1 + x2
        y = self.relu(y)
        return y

class LGAG(nn.Module):
    """
    LGAG Large-kernel Grouped Attention Gate (LGAG) Block
    大核分组门控注意力模块
    """
    def __init__(self, ch_feature_high, ch_feature_low, ch_mid, kernel_size=5, groups=1):
        super(LGAG, self).__init__()
        if kernel_size == 1:  # 避免1*1卷积使用分组，无意义
            groups = 1
        # 高层语义通道压缩分组卷积
        self.W_g = nn.Sequential(
            nn.Conv2d(ch_feature_high, ch_mid, kernel_size=kernel_size, stride=1,
                      padding=kernel_size // 2, groups=groups, bias=True),
            nn.BatchNorm2d(ch_mid),
        )
        # 低层语义通道压缩分组卷积
        self.W_x = nn.Sequential(
            nn.Conv2d(ch_feature_low, ch_mid, kernel_size=kernel_size, stride=1,
                      padding=kernel_size // 2, groups=groups, bias=True),
            nn.BatchNorm2d(ch_mid),
        )
        # concat 后通道交互卷积
        self.fusion = BasicConv(2 * ch_mid, ch_mid, kernel_size=1)

        # 注意力权重模块
        self.psi = nn.Sequential(
            nn.Conv2d(ch_mid, 1, kernel_size=1, stride=1, padding=0, bias=True),
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

class MKEFE(nn.Module):
    """
    Multi-Kernel Edge Feature Extractor
    多核边缘特征提取器
    """
    def __init__(self, out_channels=16, learnable_weight=True):
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

        self.fuse = nn.Sequential(
            nn.Conv2d(5, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        gray = (x * self.rgb_weights.to(x.device)).sum(dim=1, keepdim=True)

        edge_maps = []
        for i in range(5):
            edge = F.conv2d(gray, self.kernels[i:i+1].to(x.device), padding=1)
            edge = torch.abs(edge) * self.edge_weights[i]
            edge_maps.append(edge)  # 每个edge: (B,1,H,W)

        edge_stack = torch.cat(edge_maps, dim=1)

        edge_feat = self.fuse(edge_stack)  # (B, out_channels, H, W)

        return edge_feat

class BEB(nn.Module):
    """
    边界加强模块
    """
    def __init__(self):
        super(BEB, self).__init__()
        self.conv_edge = BasicConv(32, 16, 1)
        self.conv_mask = BasicConv(32, 16, 1)
        self.attn_s = SABWeight()
        self.attn_c = CABWeight(16)
        self.conv_out = BasicConv(16, 32)
        self.msdf = MGAFB(32, kernel_sizes=[1, 3, 5, 7], reduction=4)

    def forward(self, f_edge, f_mask):
        feat_edge = self.conv_edge(f_edge)
        feat_mask = self.conv_mask(f_mask)
        attn = self.attn_s(feat_edge) * self.attn_c(feat_edge)
        features = feat_mask * attn + feat_mask

        out = self.conv_out(features)
        out = self.msdf(out)

        return out

class PVT_MFA(nn.Module):
    """
    PVT-MFA: A Pyramid Vision Transformer with Multi-scale Fusion and Attention for Parcel Extraction in Remote Sensing Images
    PVT-MFA：一种用于遥感图像地块提取的金字塔视觉 Transformer 与多尺度融合注意力网络
    """
    def __init__(self, ch_out):
        super(PVT_MFA, self).__init__()
        self.backbone = PVTBackbone(pretrained=True)
        self.up1 = DSRUB(512, 320)
        self.up2 = DSRUB(320, 128)
        self.up3 = DSRUB(128, 64)

        self.msdf1 = MGAFB(512)
        self.msdf2 = MGAFB(320)
        self.msdf3 = MGAFB(128)
        self.msdf4 = MGAFB(64)
        self.msdf5 = MGAFB(32)

        self.up4 = DSRUB(64, 32, scale_factor=4)
        self.up5 = DSRUB(64, 16, scale_factor=4)

        self.mask_conv = nn.Conv2d(ch_out, ch_out, kernel_size=1)
        self.dist_conv = nn.Conv2d(ch_out, ch_out, kernel_size=1)

        self.ag1 = LGAG(320, 320, 320 // 2, groups=320//2)
        self.ag2 = LGAG(128, 128, 128 // 2, groups=128//2)
        self.ag3 = LGAG(64, 64, 64 // 2, groups=64//2)

        self.edge_lap = MKEFE()
        self.cabweight = CABWeight(32)
        self.beb = BEB()
        self.conv_out_a = BasicConv(32, ch_out, kernel_size=1)
        self.conv_out_b = BasicConv(32, ch_out, kernel_size=1)
        self.out_dist = nn.Conv2d(ch_out, ch_out, kernel_size=1)
        self.out_mask = nn.Conv2d(ch_out, ch_out, kernel_size=1)
        self.out_edge = nn.Conv2d(ch_out, ch_out, kernel_size=1)


    def forward(self, x):
        # 64, 128, 320, 512 (4, 8, 16,32)
        x_list = self.backbone(x)

        x_list[3] = self.msdf1(x_list[3])  # 特征加强
        x3 = self.up1(x_list[3])  # 上采样 320

        x3_skip = self.ag1(x3, x_list[2])  # 320
        x3_fuse = x3 + x3_skip  # 320
        x3_fuse = self.msdf2(x3_fuse)

        x2 = self.up2(x3_fuse)  # 128
        x2_skip = self.ag2(x2, x_list[1])  # 128
        x2_fuse = x2 + x2_skip  # 128
        x2_fuse = self.msdf3(x2_fuse)

        x1 = self.up3(x2_fuse)  # 64
        x1_skip = self.ag3(x1, x_list[0])  # stage1 经过lgag加强 64 H/4
        x1_fuse = x1 + x1_skip # 64
        x1_fuse = self.msdf4(x1_fuse)

        f_mask = self.up4(x1_fuse)  # H/4 --> H

        # edge分支
        edge_lap = self.edge_lap(x)  # H 16
        x_low = self.up5(x1_skip)  # H/4 --> H  32--> 16
        f_edge = torch.cat([x_low, edge_lap], dim=1)  # H 32
        f_edge_strong = self.msdf5(f_edge)

        out_b = self.conv_out_b(f_edge_strong)
        out_edge = self.out_edge(out_b)

        f_edge_mask = self.beb(f_edge_strong, f_mask)  # H 32
        out_a = self.conv_out_a(f_edge_mask)
        out_dist = self.dist_conv(out_a)
        out_mask = self.mask_conv(out_a)

        return out_mask, out_dist, out_edge



if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    model = PVT_MFA(ch_out=1)
    y = model(x)
    print(len(y))
    print(y[0].shape)
    print(y[1].shape)
    print(y[2].shape)


