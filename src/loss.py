import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLoss:
    def __init__(self, weights=[1.0, 1.0, 0.1]):
        self.criterion1 = BCE_DiceLoss()
        self.criterion2 = FocalLoss()
        self.criterion3 = EdgeConsistencyLoss()
        self.weights = weights

    #                 pred_mask  pred_edge  mask      edge
    def __call__(self, outputs1, outputs2, targets1, targets2):
        loss = (
                  self.weights[0] * self.criterion1(outputs1, targets1)
                + self.weights[1] * self.criterion2(outputs2, targets2)
                + self.weights[2] * self.criterion3(outputs2, targets1)
        )

        return loss

class BCE_DiceLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=0.5):
        super(BCE_DiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        # BCE 部分，输入为 logits（未 sigmoid）
        bce = F.binary_cross_entropy_with_logits(logits, targets)

        # Dice 部分，先 sigmoid 再计算
        probs = torch.sigmoid(logits)
        smooth = 1.0  # 防止除零

        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2. * intersection + smooth) / (probs_flat.sum() + targets_flat.sum() + smooth)
        dice_loss = 1 - dice

        # 加权组合
        total_loss = self.bce_weight * bce + self.dice_weight * dice_loss
        return total_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 正类权重
        self.gamma = gamma  # 难例调节参数
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: logits，形状 [B, 1, H, W]
        targets: 二值标签，形状 [B, 1, H, W] 或 [B, H, W]
        """
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class EdgeConsistencyLoss(nn.Module):
    """
    使用 Sobel 提取 GT 掩码的边缘，作为 edge_pred 的监督目标。
    edge_pred 是网络直接输出的边缘图（未 Sigmoid）。
    """
    def __init__(self, reduction='mean'):
        super(EdgeConsistencyLoss, self).__init__()
        self.reduction = reduction
        self.l1_loss = nn.L1Loss(reduction=reduction)

        # Sobel X 和 Y 卷积核（仅用于 GT mask）
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer('weight_x', sobel_x)
        self.register_buffer('weight_y', sobel_y)

    def get_edge(self, mask):
        # 将 Sobel 核移动到输入张量所在的设备（CPU or GPU）
        weight_x = self.weight_x.to(mask.device)
        weight_y = self.weight_y.to(mask.device)

        grad_x = F.conv2d(mask, weight_x, padding=1)
        grad_y = F.conv2d(mask, weight_y, padding=1)
        edge = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        return edge  # 强度

    def forward(self, edge_pred, mask_gt):
        """
        mask_gt: Ground Truth 掩码 (B, 1, H, W)，0/1 float tensor
        edge_pred: 网络输出的边缘图 (B, 1, H, W)，未 sigmoid
        """
        with torch.no_grad():
            edge_gt = self.get_edge(mask_gt)
        edge_pred_sigmoid = torch.sigmoid(edge_pred)

        return self.l1_loss(edge_pred_sigmoid, edge_gt)


