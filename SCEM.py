import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.RRB import ConvBNAct



class SemanticLocalAggregationUnit(nn.Module):
    """
    Semantic local aggregation unit.

    It captures fine-to-medium local semantic context using
    lightweight multi-scale depthwise convolutions.

    Abbreviation:
        SLAU = SemanticLocalAggregationUnit
    """
    def __init__(self, channels):
        super().__init__()

        self.branch3 = nn.Sequential(
            ConvBNAct(
                channels,
                channels,
                kernel_size=3,
                groups=channels,
                act=True
            ),
            ConvBNAct(
                channels,
                channels,
                kernel_size=1,
                padding=0,
                act=False
            )
        )

        self.branch5 = nn.Sequential(
            ConvBNAct(
                channels,
                channels,
                kernel_size=5,
                groups=channels,
                act=True
            ),
            ConvBNAct(
                channels,
                channels,
                kernel_size=1,
                padding=0,
                act=False
            )
        )

        self.fusion = ConvBNAct(
            channels * 2,
            channels,
            kernel_size=1,
            padding=0,
            act=True
        )

    def forward(self, x):
        feat_3 = self.branch3(x)
        feat_5 = self.branch5(x)
        out = torch.cat([feat_3, feat_5], dim=1)
        out = self.fusion(out)
        return out


class TopKSparseGlobalContext(nn.Module):
    """
    Top-k sparse global context module.

    Input:
        [B, C, H, W]
    Output:
        [B, C, H, W]

    Design:
        - Query from full-resolution feature
        - Key/Value from optional spatially reduced feature
        - Top-k sparsification to suppress noisy dense attention

    Abbreviation:
        TSGC = TopKSparseGlobalContext
    """
    def __init__(
        self,
        dim,
        num_heads=2,
        topk_ratio=0.25,
        sr_ratio=2,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.topk_ratio = topk_ratio
        self.sr_ratio = sr_ratio

        self.query_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.key_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.value_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

        if sr_ratio > 1:
            if sr_ratio == 2:
                self.spatial_reduction = nn.Sequential(
                    nn.Conv2d(
                        dim, dim,
                        kernel_size=2,
                        stride=2,
                        padding=0,
                        bias=False
                    ),
                    nn.BatchNorm2d(dim)
                )
            elif sr_ratio == 4:
                self.spatial_reduction = nn.Sequential(
                    nn.Conv2d(
                        dim, dim,
                        kernel_size=4,
                        stride=4,
                        padding=1,
                        bias=False
                    ),
                    nn.BatchNorm2d(dim)
                )
            else:
                raise ValueError("Only sr_ratio=2 or 4 is supported.")
        else:
            self.spatial_reduction = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.output_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.output_drop = nn.Dropout(proj_drop)

    @staticmethod
    def _apply_topk_mask(attn_map, k):
        """
        Args:
            attn_map: [B, heads, Nq, Nk]
        """
        if k >= attn_map.size(-1):
            return attn_map

        topk_values, topk_indices = torch.topk(attn_map, k=k, dim=-1)
        masked_attn = torch.full_like(attn_map, float("-inf"))
        masked_attn.scatter_(-1, topk_indices, topk_values)
        return masked_attn

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        num_query_tokens = height * width

        query = self.query_proj(x)

        if self.spatial_reduction is not None:
            kv_input = self.spatial_reduction(x)
        else:
            kv_input = x

        _, _, h_kv, w_kv = kv_input.shape
        num_key_tokens = h_kv * w_kv

        key = self.key_proj(kv_input)
        value = self.value_proj(kv_input)

        query = query.reshape(
            batch_size, self.num_heads, self.head_dim, num_query_tokens
        ).permute(0, 1, 3, 2)  # [B, heads, Nq, d]

        key = key.reshape(
            batch_size, self.num_heads, self.head_dim, num_key_tokens
        ).permute(0, 1, 3, 2)  # [B, heads, Nk, d]

        value = value.reshape(
            batch_size, self.num_heads, self.head_dim, num_key_tokens
        ).permute(0, 1, 3, 2)  # [B, heads, Nk, d]

        attn_map = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        k_keep = max(1, int(self.topk_ratio * num_key_tokens))
        attn_map = self._apply_topk_mask(attn_map, k_keep)
        attn_map = F.softmax(attn_map, dim=-1)
        attn_map = self.attn_drop(attn_map)

        out = torch.matmul(attn_map, value)  # [B, heads, Nq, d]
        out = out.permute(0, 1, 3, 2).reshape(batch_size, channels, height, width)

        out = self.output_proj(out)
        out = self.output_drop(out)
        return out


class SemanticContextEnhancementModule(nn.Module):
    """
    SCEM
    """
    def __init__(
        self,
        dim,
        hidden_ratio=0.5,
        num_heads=2,
        topk_ratio=0.25,
        sr_ratio=2,
        drop=0.0,
        attn_drop=0.0,
        use_global=True
    ):
        super().__init__()
        self.use_global = use_global

        hidden_dim = max(int(dim * hidden_ratio), 16)
        hidden_dim = max(hidden_dim, num_heads)
        hidden_dim = (hidden_dim // num_heads) * num_heads
        hidden_dim = max(hidden_dim, num_heads)

        self.channel_reduction = ConvBNAct(
            dim, hidden_dim, kernel_size=1, padding=0, act=True
        )

        self.local_aggregation = SemanticLocalAggregationUnit(hidden_dim)

        if self.use_global:
            self.global_context = TopKSparseGlobalContext(
                dim=hidden_dim,
                num_heads=num_heads,
                topk_ratio=topk_ratio,
                sr_ratio=sr_ratio,
                attn_drop=attn_drop,
                proj_drop=drop
            )
            fusion_in_channels = hidden_dim * 2
        else:
            self.global_context = None
            fusion_in_channels = hidden_dim

        self.fusion = nn.Sequential(
            ConvBNAct(
                fusion_in_channels,
                hidden_dim,
                kernel_size=1,
                padding=0,
                act=True
            ),
            nn.Dropout(drop)
        )

        self.channel_restoration = ConvBNAct(
            hidden_dim,
            dim,
            kernel_size=1,
            padding=0,
            act=False
        )

    def forward(self, x):
        identity = x

        x = self.channel_reduction(x)
        local_feat = self.local_aggregation(x)

        if self.use_global:
            global_feat = self.global_context(local_feat)
            out = torch.cat([local_feat, global_feat], dim=1)
        else:
            out = local_feat

        out = self.fusion(out)
        out = self.channel_restoration(out)
        out = out + identity
        return out


