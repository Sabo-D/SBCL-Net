
import cv2
import numpy as np
import os

def dilate_boundary(boundary, kernel_size=1):
    """
    膨胀边界，使边界线变粗（kernel_size建议1~5）
    输入为三通道二值图，输出三通道图
    """
    if len(boundary.shape) == 3 and boundary.shape[2] == 3:
        boundary_gray = cv2.cvtColor(boundary, cv2.COLOR_BGR2GRAY)
    else:
        boundary_gray = boundary.copy()

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(boundary_gray, kernel, iterations=1)
    return cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)

def generate_boundary_quads(image, gt_boundary, pred_boundary,
                            save_dir="./boundary_vis", prefix="sample", line_thickness=3):
    """
    输入 image, gt_boundary, pred_boundary（三通道、0/255），输出四张图（独立保存）
    line_thickness 控制边界线粗细（默认2）
    """

    os.makedirs(save_dir, exist_ok=True)

    def ensure_rgb(img):
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:
            return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
        return img

    image = ensure_rgb(image)

    # 🔁 膨胀边界
    gt_boundary = dilate_boundary(gt_boundary, kernel_size=line_thickness)
    pred_boundary = dilate_boundary(pred_boundary, kernel_size=line_thickness)

    gt_boundary = ensure_rgb(gt_boundary)
    pred_boundary = ensure_rgb(pred_boundary)

    H, W = image.shape[:2]
    background = np.ones_like(image) * 30  # 暗灰底

    # ------ 掩码计算 ------
    red_mask = gt_boundary[:, :, 0] > 127
    green_mask = pred_boundary[:, :, 1] > 127
    overlap_mask = red_mask & green_mask
    only_gt = red_mask & (~green_mask)
    only_pred = green_mask & (~red_mask)

    # ------ 图1：原图 + GT边界（红） ------
    img1 = image.copy()
    img1[red_mask] = [255, 255, 255]
    cv2.imwrite(os.path.join(save_dir, f"{prefix}_image_with_gt.png"), img1)

    # ------ 图2：GT边界图（红） ------
    img2 = background.copy()
    img2[red_mask] = [0, 0, 255]
    cv2.imwrite(os.path.join(save_dir, f"{prefix}_gt_boundary.png"), img2)

    # ------ 图3：Pred边界图（绿） ------
    img3 = background.copy()
    img3[green_mask] = [0, 255, 0]
    cv2.imwrite(os.path.join(save_dir, f"{prefix}_pred_boundary.png"), img3)

    # ------ 图4：边界对比图（红/绿/白） ------
    img4 = background.copy()
    img4[only_gt] = [0, 0, 255]
    img4[only_pred] = [0, 255, 0]
    img4[overlap_mask] = [255, 255, 255]
    cv2.imwrite(os.path.join(save_dir, f"{prefix}_boundary_comparison.png"), img4)

    print(f"[✓] Saved 4 boundary images to: {save_dir} (line_thickness={line_thickness})")




if __name__ == '__main__':
    image_path = r"C:\Users\Administrator\Desktop\data\out\ablate\images\3.tif"
    gt_edge_path = r"C:\Users\Administrator\Desktop\data\out\ablate\edges\3.tif"
    pred_edge_path = r"C:\Users\Administrator\Desktop\data\out\ablate\out_edges\edges\v0_3.png"
    output_path = r"C:\Users\Administrator\Desktop\data\out\ablate\out_edges_01"
    generate_boundary_quads(
        image=cv2.imread(image_path),
        gt_boundary=cv2.imread(gt_edge_path),  # 三通道
        pred_boundary=cv2.imread(pred_edge_path),  # 三通道
        save_dir=output_path,
        prefix="v5_3",
        line_thickness=1
    )

