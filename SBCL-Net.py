import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class SEB(nn.Module):
    def __init__(self, C_sem, C_bnd, heads=4, dim_head=32, kv_downsample=2, use_gate=True):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.kv_downsample = kv_downsample
        self.q_proj = nn.Conv2d(C_sem, heads*dim_head, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(C_bnd, heads*dim_head, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(C_bnd, heads*dim_head, kernel_size=1, bias=False)
        self.out = nn.Conv2d(heads*dim_head, C_bnd, kernel_size=1, bias=False)
        self.use_gate = use_gate
        if use_gate:
            self.gate = nn.Sequential(
                nn.Conv2d(C_bnd + C_sem, C_bnd, kernel_size=1, bias=False),
                nn.Sigmoid()
            )
        self.norm = nn.BatchNorm2d(C_bnd)

    def forward(self, Fsem, Fbnd):
        B, Cs, Hq, Wq = Fsem.shape
        # align spatial sizes if needed outside this module
        # Project Q
        q = self.q_proj(Fsem)  # [B, heads*dim, H, W]
        q = q.view(B, self.heads, self.dim_head, Hq*Wq)  # [B,heads,dim,HWq]
        q = q.permute(0,1,3,2).contiguous()  # [B,heads,HWq,dim]

        # optionally downsample K/V to reduce cost
        if self.kv_downsample > 1:
            Kv = F.adaptive_avg_pool2d(Fbnd, (Hq//self.kv_downsample, Wq//self.kv_downsample))
        else:
            Kv = Fbnd
        Bk, Ck, Hk, Wk = Kv.shape
        k = self.k_proj(Kv).view(B, self.heads, self.dim_head, Hk*Wk).permute(0,1,3,2)  # [B,heads,HWk,dim]
        v = self.v_proj(Kv).view(B, self.heads, self.dim_head, Hk*Wk).permute(0,1,3,2)  # [B,heads,HWk,dim]

        # attention
        attn = torch.matmul(q, k.transpose(-2,-1)) * self.scale  # [B,heads,HWq,HWk]
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)  # [B,heads,HWq,dim]
        out = out.permute(0,1,3,2).contiguous().view(B, self.heads*self.dim_head, Hq, Wq)
        out = self.out(out)  # [B, C_bnd, H, W]

        # fusion: residual + optional gated fusion
        if self.use_gate:
            g = self.gate(torch.cat([Fbnd, F.adaptive_max_pool2d(Fsem, (Hq, Wq))], dim=1))
            Fbnd_new = Fbnd * (1 - g) + out * g
        else:
            Fbnd_new = self.norm(Fbnd + out)

        return Fbnd_new

class SEB_Light(nn.Module):
    def __init__(self, C_sem, C_bnd, heads=4, dim_head=32, kv_downsample=2, use_gate=True):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.kv_downsample = kv_downsample
        self.use_gate = use_gate

        # ---- 1. lightweight Q/K/V projection ----
        def sep_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, groups=in_c, bias=False),  # depthwise
                nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),  # pointwise
                nn.BatchNorm2d(out_c)
            )

        self.q_proj = sep_conv(C_sem, heads * dim_head)
        self.k_proj = sep_conv(C_bnd, heads * dim_head)
        self.v_proj = sep_conv(C_bnd, heads * dim_head)
        self.out = nn.Conv2d(heads * dim_head, C_bnd, kernel_size=1, bias=False)

        # ---- 2. optional gated fusion ----
        if use_gate:
            self.gate = nn.Sequential(
                nn.Conv2d(C_bnd + C_sem, C_bnd, kernel_size=1, bias=False),
                nn.BatchNorm2d(C_bnd),
                nn.Sigmoid()
            )

        # ---- 3. optional kv reduction ----
        self.kv_reduction = (
            nn.Conv2d(C_bnd, C_bnd, kernel_size=3, stride=kv_downsample, padding=1, groups=C_bnd)
            if kv_downsample > 1 else nn.Identity()
        )

        self.norm = nn.BatchNorm2d(C_bnd)

    def forward(self, Fsem, Fbnd):
        B, Cs, H, W = Fsem.shape
        Hk, Wk = H // self.kv_downsample, W // self.kv_downsample

        # ---- Q/K/V ----
        q = self.q_proj(Fsem).view(B, self.heads, self.dim_head, H * W).permute(0, 1, 3, 2)
        Fbnd_ds = self.kv_reduction(Fbnd)
        k = self.k_proj(Fbnd_ds).view(B, self.heads, self.dim_head, Hk * Wk).permute(0, 1, 3, 2)
        v = self.v_proj(Fbnd_ds).view(B, self.heads, self.dim_head, Hk * Wk).permute(0, 1, 3, 2)

        # ---- Attention ----
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).contiguous().view(B, self.heads * self.dim_head, H, W)
        out = self.out(out)

        # ---- Gated Fusion ----
        if self.use_gate:
            Fsem_pool = F.adaptive_avg_pool2d(Fsem, (H, W))
            g = self.gate(torch.cat([Fbnd, Fsem_pool], dim=1))
            Fbnd_new = Fbnd * (1 - g) + out * g
        else:
            Fbnd_new = Fbnd + out

        Fbnd_new = self.norm(Fbnd_new)
        return Fbnd_new

class LocalSEB(nn.Module):
    def __init__(self, C_sem, C_bnd, heads=4, dim_head=32, window_size=8, use_gate=True):
        """

        :param C_sem: semantic通道数 32
        :param C_bnd: boundary通道数 32
        :param heads: 头数
        :param dim_head: 每个头的维度
        :param window_size: 窗口大小
        :param use_gate: 门控
        """
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.window_size = window_size
        self.scale = dim_head ** -0.5  # 缩放因子
        self.use_gate = use_gate

        # 深度可分离卷积 Q KV
        self.q_proj = nn.Sequential(
            nn.Conv2d(C_sem, C_sem, kernel_size=3, padding=1, groups=C_sem, bias=False),
            nn.Conv2d(C_sem, heads * dim_head, kernel_size=1, bias=False)
        )
        self.kv_proj = nn.Sequential(
            nn.Conv2d(C_bnd, C_bnd, kernel_size=3, padding=1, groups=C_bnd, bias=False),
            nn.Conv2d(C_bnd, heads * dim_head * 2, kernel_size=1, bias=False)
        )
        self.out = nn.Conv2d(heads * dim_head, C_bnd, kernel_size=1, bias=False)

        if use_gate:
            self.gate = nn.Sequential(
                nn.Conv2d(C_bnd + C_sem, C_bnd, kernel_size=1, bias=False),
                nn.Sigmoid()
            )

        self.norm = nn.BatchNorm2d(C_bnd)

    def window_partition(self, x):
        """
        特征图划分为不重叠的局部窗口
        :param x: [B,C,H,W]
        :return: [B*num_windows, C, w, w]
        """
        B, C, H, W = x.shape
        w = self.window_size
        # 特征图拆分为网格状窗口 [B,C,H方向窗口数，窗口大小，W方向窗口数，窗口大小]
        x = x.view(B, C, H // w, w, W // w, w)
        # [B*num_windows, C, w, w]
        return x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, w, w)

    def window_reverse(self, x, H, W):
        """
        将局部窗口还原为完整特征图
        :param x: [B*num_windows, C, w, w]
        :param H:
        :param W:
        :return: [B,C,H,W]
        """
        w = self.window_size
        # 计算原始batch大小
        B = x.shape[0] // (H // w * W // w)
        # 恢复窗口的网格结构 [B,H//w,W//w,C,w,w]
        x = x.view(B, H // w, W // w, -1, w, w)
        return x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, -1, H, W)

    def forward(self, Fsem, Fbnd):
        B, Cs, H, W = Fsem.shape
        # 投影QKV
        q = self.q_proj(Fsem)
        kv = self.kv_proj(Fbnd)
        k, v = kv.chunk(2, dim=1)

        # 分窗口 [B,C,H,W] --> [B*num_windows, C, w, w], C=heads*dim_head
        q_windows = self.window_partition(q)
        k_windows = self.window_partition(k)
        v_windows = self.window_partition(v)

        # reshape [B*num_windows, C, w, w] --> [B*num_windows ,heads, head_dim, w*w]
        # --> [B*num_windows, heads, w^2, dim_head] [...token长度, embedding维度]
        q_windows = q_windows.view(-1, self.heads, self.dim_head, self.window_size**2).permute(0,1,3,2)
        k_windows = k_windows.view(-1, self.heads, self.dim_head, self.window_size**2).permute(0,1,3,2)
        v_windows = v_windows.view(-1, self.heads, self.dim_head, self.window_size**2).permute(0,1,3,2)

        # local attention
        attn = torch.matmul(q_windows, k_windows.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v_windows)

        out = out.permute(0,1,3,2).contiguous().view(-1, self.heads*self.dim_head, self.window_size, self.window_size)
        out = self.window_reverse(out, H, W)
        out = self.out(out)

        # gated fusion
        if self.use_gate:
            g = self.gate(torch.cat([Fbnd, F.adaptive_max_pool2d(Fsem, (H, W))], dim=1))
            Fbnd_new = Fbnd * (1 - g) + out * g
        else:
            Fbnd_new = self.norm(Fbnd + out)

        return Fbnd_new

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

class MKEFE_ablation(nn.Module):
    """
    Ablation version of Multi-Kernel Edge Feature Extractor
    —— replace handcrafted MKER with learnable conv (same output dims)
    """
    def __init__(self, out_channels=32):
        super().__init__()

        # 与原模块一致，将RGB转灰度（保留一致性）
        self.rgb_weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)

        # 替换 MKER: 用可学习卷积代替5个固定算子
        self.learnable_conv = nn.Conv2d(1, 5, kernel_size=3, padding=1, bias=True)

        # 与原模块一致的后续结构
        self.fuse = ResBlock(5, out_channels)
        self.mgafb = MGAFB(out_channels)

    def forward(self, x):
        gray = (x * self.rgb_weights.to(x.device)).sum(dim=1, keepdim=True)

        # 替换 MKER: 学习卷积提取边缘
        edge_stack = self.learnable_conv(gray)

        # 后续与原模块一致
        edge_feat = self.fuse(edge_stack)
        edge_feat = self.mgafb(edge_feat)

        return edge_feat

class MKEFE(nn.Module):
    """
    Multi-Kernel Edge Feature Extractor
    多核边缘特征提取器
    """
    def __init__(self, out_channels=32, learnable_weight=True):
        super().__init__()

        """
        Multi-Kernel Edge Response (MKER)
        """
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

class SBCL(nn.Module):
    """
    SBCL-Net: A Bidirectional Semantic–Boundary Closed-Loop Network for
    Agricultural Parcel Delineation in Remote Sensing Imagery
    """
    def __init__(self, num_classes=1, backbone_name='pvt_v2_b2'):
        super(SBCL, self).__init__()
        self.encoder = PVTBackbone(backbone_name)
        channels = self.encoder.out_channels    # [64, 128, 320, 512]

        self.center = ResBlock(channels[-1], channels[-1])  # 512 512

        self.decoder4 = DecoderBlock(channels[3], channels[2], channels[2])  # 512 320 320
        self.decoder3 = DecoderBlock(channels[2], channels[1], channels[1])  # 320 128 128
        self.decoder2 = DecoderBlock(channels[1], channels[0], channels[0])  # 128 64 64
        self.decoder1 = DecoderBlock(channels[0], 32, 32)  # 64 32 32
        self.skip0 = ResBlock(3, 32)

        self.out_mask = nn.Conv2d(32, num_classes, kernel_size=1)
        self.mkefe = MKEFE(32)
        self.edge_conv = ResBlock(32, 32)
        self.beb = BEB()
        self.seb = LocalSEB(32,32)
        self.out_edge = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        skips = self.encoder(x)  # [B, c_i, H/4, H/8, H/16, H/32]
        x0 = self.skip0(x)       # 原始输入作为浅层特征 skip0

        e1, e2, e3, e4 = skips   # 对应 PVT 输出的 4 个阶段

        center = self.center(e4)  # 底层特征
        d4 = self.decoder4(center, e3)  #
        d3 = self.decoder3(d4, e2)
        d2 = self.decoder2(d3, e1)
        d1 = self.decoder1(d2, x0)  # 32

        """ edge """
        f_edge = self.mkefe(x)  # 32
        edge_strong = self.seb(x0, f_edge)
        x0_edge = self.edge_conv(edge_strong)  # 32
        edge = self.out_edge(x0_edge)

        beb_out = self.beb(x0_edge, d1) # 32
        mask = self.out_mask(beb_out)

        return [mask, edge]
        # return{
        #         "sem_feats": {"enc1":e1, "enc2":e2},
        #         "bdy_feats": {"mkefe_out":f_edge, "seb_out":edge_strong}
        #       }

if __name__ == '__main__':
    from src.utils.utils import complexity
    model = SBCL(1)
    complexity(model)
