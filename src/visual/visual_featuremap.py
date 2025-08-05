import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize_feature_response(feature_map, mode='average', cmap='viridis', save_path='feature_response.png'):
    """
    可视化 feature map（支持 [B, C, H, W] 或 [C, H, W]）为热力图（无留白）：
    - mode = 'average'：通道平均响应图
    - mode = 'max'：最大响应通道（均值最大）
    """

    # 1. 转 numpy
    if isinstance(feature_map, torch.Tensor):
        feature_map = feature_map.detach().cpu().numpy()

    # 2. 处理 batch 维度
    if len(feature_map.shape) == 4:  # [B, C, H, W]
        assert feature_map.shape[0] == 1, "Only batch size 1 is supported"
        feature_map = feature_map[0]  # → [C, H, W]
    elif len(feature_map.shape) != 3:
        raise ValueError("Expected input shape [C, H, W] or [1, C, H, W]")

    # 3. 获取通道
    if mode == 'average':
        vis_map = np.mean(feature_map, axis=0)
    elif mode == 'max':
        idx = feature_map.mean(axis=(1, 2)).argmax()
        vis_map = feature_map[idx]
    else:
        raise ValueError("Mode must be 'average' or 'max'.")

    # 4. 归一化
    vis_map -= vis_map.min()
    if vis_map.max() > 0:
        vis_map /= vis_map.max()

    # 5. 保存纯图
    plt.figure(figsize=(5, 5))
    plt.imshow(vis_map, cmap=cmap)
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"[✓] Saved feature response map to {save_path} ({mode}, no margin)")
