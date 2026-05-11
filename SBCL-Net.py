import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


from src.model.BSEM import BoundaryStructureExtractionModule as BSEM
from src.model.RRB import ResidualRefinementBlock as RRB
from src.model.RRB import ConvBNAct
from src.model.SCEM import SemanticContextEnhancementModule as SCEM
from src.model.DSMM import DSMM


class PVTBackbone(nn.Module):
    def __init__(self, model_name='pvt_v2_b2', pretrained=True):
        super(PVTBackbone, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        self.out_channels = [f['num_chs'] for f in self.backbone.feature_info]  # e.g. [64, 128, 320, 512]

    def forward(self, x):
        return self.backbone(x)

class UPB(nn.Module):
    """
    Upsample Projection Block
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factor=2,
        mode="bilinear",
        align_corners=False
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

        self.proj = ConvBNAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            act=True
        )

    def forward(self, x, target_size=None):
        if target_size is not None:
            if self.mode in ["linear", "bilinear", "bicubic", "trilinear"]:
                x = F.interpolate(
                    x,
                    size=target_size,
                    mode=self.mode,
                    align_corners=self.align_corners
                )
            else:
                x = F.interpolate(
                    x,
                    size=target_size,
                    mode=self.mode
                )
        else:
            if self.mode in ["linear", "bilinear", "bicubic", "trilinear"]:
                x = F.interpolate(
                    x,
                    scale_factor=self.scale_factor,
                    mode=self.mode,
                    align_corners=self.align_corners
                )
            else:
                x = F.interpolate(
                    x,
                    scale_factor=self.scale_factor,
                    mode=self.mode
                )

        x = self.proj(x)
        return x

class CRB(nn.Module):
    """
    Concatenation Reduction Block.
    """
    def __init__(
        self,
        in_channels_1,
        in_channels_2,
        out_channels
    ):
        super().__init__()

        self.reduction = ConvBNAct(
            in_channels=in_channels_1 + in_channels_2,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            act=True
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.reduction(x)
        return x

class SBCL(nn.Module):
    """
    SBCL-Net: A Bidirectional Semantic–Boundary Closed-Loop Network for
    Agricultural Parcel Delineation in Remote Sensing Imagery
    """
    def __init__(self, num_classes=1, backbone_name='pvt_v2_b2'):
        super(SBCL, self).__init__()
        self.encoder = PVTBackbone(backbone_name)
        channels = self.encoder.out_channels    # [64, 128, 320, 512]

        # stage 0-4
        self.refine_4 = RRB(channels[3], channels[3])   # 512 -> 512
        self.refine_3 = RRB(channels[2], channels[2])   # 320 -> 320
        self.refine_2 = RRB(channels[1], channels[1])   # 128 -> 128
        self.refine_1 = RRB(channels[0], channels[0])   # 64  -> 64
        self.refine_0 = RRB(3, 32) # RGB -> 32

        self.UPB_4 = UPB(channels[3], channels[2], scale_factor=2)   # 512 -> 320
        self.UPB_3 = UPB(channels[2], channels[1], scale_factor=2)   # 320 -> 128
        self.UPB_2 = UPB(channels[1], channels[0], scale_factor=2)   # 128 -> 64
        self.UPB_1 = UPB(channels[0], 32, scale_factor=4)  # 64 -> 32

        self.CRB_3 = CRB(channels[2], channels[2], channels[2])  # 320+320 -> 320
        self.CRB_2 = CRB(channels[1], channels[1], channels[1])  # 128+128 -> 128
        self.CRB_1 = CRB(channels[0], channels[0], channels[0])  # 64+64   -> 64
        self.CRB_0 = CRB(32, 32, 32) # 32+32   -> 32

        self.enhance_4_3 = SCEM(dim=channels[2], sr_ratio=2)
        self.enhance_3_2 = SCEM(dim=channels[1], sr_ratio=4)
        self.enhance_2_1 = SCEM(dim=channels[0], sr_ratio=4)
        self.enhance_1_0 = SCEM(dim=32, use_global=False)

        self.prior = BSEM(3,32)
        self.S2B = DSMM (32,32)  # RETURN REFINED SLAVE
        self.refine_b = RRB(32,32)

        self.B2S = DSMM(32,32)

        self.edge = nn.Conv2d(32, 1, kernel_size=1, bias=True)
        self.mask = nn.Conv2d(32, 1, kernel_size=1, bias=True)

    def forward(self, x):
        skips = self.encoder(x)
        x1, x2, x3, x4 = skips

        x4 = self.refine_4(x4)
        x3 = self.refine_3(x3)
        x2 = self.refine_2(x2)
        x1 = self.refine_1(x1)
        x0 = self.refine_0(x)

        x4_up = self.UPB_4(x4, target_size=x3.shape[-2:])
        x4_3_fuse = self.CRB_3(x4_up, x3)
        x4_3_enhance = self.enhance_4_3(x4_3_fuse)

        x3_up = self.UPB_3(x4_3_enhance, target_size=x2.shape[-2:])
        x3_2_fuse = self.CRB_2(x3_up, x2)
        x3_2_enhance = self.enhance_3_2(x3_2_fuse)

        x2_up = self.UPB_2(x3_2_enhance, target_size=x1.shape[-2:])
        x2_1_fuse = self.CRB_1(x2_up, x1)
        x2_1_enhance = self.enhance_2_1(x2_1_fuse)

        x1_up = self.UPB_1(x2_1_enhance, target_size=x0.shape[-2:])
        x1_0_fuse = self.CRB_0(x1_up, x0)
        x1_0_enhance = self.enhance_1_0(x1_0_fuse)

        prior, lap = self.prior(x)
        s2b = self.S2B(x0, prior)
        refine_b = self.refine_b(s2b)

        B2S = self.B2S(refine_b,x1_0_enhance)

        edge = self.edge(refine_b)
        mask = self.mask(B2S)

        return [mask, edge]






if __name__ == '__main__':
    from src.utils.utils import complexity
    model = SBCL(1)
    complexity(model)
