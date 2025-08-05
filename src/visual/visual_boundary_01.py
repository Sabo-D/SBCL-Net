import os
import cv2
import numpy as np

import os
import cv2
import numpy as np
from scipy.ndimage import binary_dilation


def compute_boundary_error_map_with_tolerance(gt_folder, pred_folder, output_folder, tolerance=1):
    """
    Compare GT and predicted edge maps with spatial tolerance and generate RGB error maps.

    Args:
        gt_folder (str): Folder containing GT edge maps (0/255 binary images)
        pred_folder (str): Folder containing predicted edge maps (0/255 binary images)
        output_folder (str): Folder to save color-coded error maps
        tolerance (int): Pixel tolerance (radius) for TP matching
    """
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(gt_folder):
        if not filename.lower().endswith(('.png', '.jpg', '.tif')):
            continue

        gt_path = os.path.join(gt_folder, filename)
        pred_path = os.path.join(pred_folder, filename)

        if not os.path.exists(pred_path):
            print(f"[Warning] Missing prediction: {filename}")
            continue

        # Load as grayscale (0-255)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        # Ensure binary masks: 0 or 1
        gt_bin = (gt > 127).astype(np.uint8)
        pred_bin = (pred > 127).astype(np.uint8)

        # Dilate GT and pred to create tolerance masks
        gt_dilated = binary_dilation(gt_bin, iterations=tolerance)
        pred_dilated = binary_dilation(pred_bin, iterations=tolerance)

        # True positives: pred overlaps GT (within tolerance)
        tp = gt_dilated & pred_bin

        # False positives: pred not in GT (even after tolerance)
        fp = pred_bin & (~gt_dilated)

        # False negatives: GT not in pred (even after tolerance)
        fn = gt_bin & (~pred_dilated)

        # Create RGB overlay image
        overlay = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
        overlay[tp.astype(bool)] = [255, 255, 255]  # TP → white
        overlay[fp.astype(bool)] = [255, 0, 0]  # FP → red
        overlay[fn.astype(bool)] = [0, 0, 255]  # FN → blue

        # Save output
        out_path = os.path.join(output_folder, filename)
        cv2.imwrite(out_path, overlay)

        print(f"[✓] Saved: {filename}")

    print(f"\n✅ Done. All error maps saved to '{output_folder}' (tolerance = {tolerance} pixels).")


# Example usage:
# compute_boundary_error_map_with_tolerance('GT_edge', 'pred_edge', 'error_output', tolerance=2)

# Example usage:
# compute_boundary_error_map('GT_edge_folder', 'pred_edge_folder', 'output_folder')
if __name__ == "__main__":
    # 设置你的路径
    gt_dir = r"C:\Users\Administrator\Desktop\data\out\ablate\edges"  # Ground Truth masks 文件夹
    pred_dir = r"C:\Users\Administrator\Desktop\data\out\ablate\out_edges\edges"  # 模型预测 masks 文件夹
    save_dir = r"C:\Users\Administrator\Desktop\data\out\ablate\out_edges_02"  # 保存可视化结果

    compute_boundary_error_map_with_tolerance(gt_dir, pred_dir, save_dir)