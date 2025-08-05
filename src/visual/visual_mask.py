import os
import cv2
import numpy as np
from tqdm import tqdm

def visualize_and_save_confusion_map(gt_path, pred_path, save_path, threshold=0.5):
    # 读取图像并处理为二值
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

    if gt is None or pred is None:
        print(f"[警告] 无法读取图像: {gt_path} 或 {pred_path}")
        return

    gt_bin = (gt > 127).astype(np.uint8)
    pred_bin = (pred / 255.0 >= threshold).astype(np.uint8)

    h, w = gt.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    TP = (pred_bin == 1) & (gt_bin == 1)
    TN = (pred_bin == 0) & (gt_bin == 0)
    FP = (pred_bin == 1) & (gt_bin == 0)
    FN = (pred_bin == 0) & (gt_bin == 1)

    rgb[TP] = [255, 255, 255]  # 白色
    rgb[TN] = [0, 0, 0]        # 黑色
    rgb[FP] = [255, 0, 0]      # 红色
    rgb[FN] = [0, 0, 255]      # 蓝色

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, rgb)

def batch_visualize_confusion(gt_dir, pred_dir, save_dir, threshold=0.5, suffix='.png'):
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(suffix)])

    for fname in tqdm(gt_files, desc="Processing"):
        gt_path = os.path.join(gt_dir, fname)
        pred_path = os.path.join(pred_dir, fname)
        save_path = os.path.join(save_dir, fname)

        if not os.path.exists(pred_path):
            print(f"[跳过] 缺少预测文件: {fname}")
            continue

        visualize_and_save_confusion_map(gt_path, pred_path, save_path, threshold=threshold)

if __name__ == "__main__":
    # 设置你的路径
    gt_dir = r"C:\Users\Administrator\Desktop\data\out\ablate\edges"  # Ground Truth masks 文件夹
    pred_dir = r"C:\Users\Administrator\Desktop\data\out\ablate\out_edges\edges"  # 模型预测 masks 文件夹
    save_dir = r"C:\Users\Administrator\Desktop\data\out\ablate\out_edges_01"  # 保存可视化结果

    batch_visualize_confusion(gt_dir, pred_dir, save_dir, threshold=0.5)