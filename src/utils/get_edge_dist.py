from utils import *
import os
import cv2

def make_dist_mask(mask_path, dist_path):
    geo_trans, geo_proj, image_data = get_geo_info(mask_path)
    if image_data.shape[0] == 3:
        image_data = image_data[0, :, :]
    # 准欧几里德距离
    result = cv2.distanceTransform(src=image_data, distanceType=cv2.DIST_L2, maskSize=3)
    # 欧几里得距离
    # result = cv2.distanceTransform(src=result, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE)
    # min-max归一化
    min_value = np.min(result)
    max_value = np.max(result)
    scaled_image = ((result - min_value) / (max_value - min_value)) * 255
    result = scaled_image.astype(np.uint8)
    save_with_geo(dist_path, geo_trans, geo_proj, result)
    print("OK")
    return 0

def make_dist_edge(edge_path, dist_path):
    geo_trans, geo_proj, image_data = get_geo_info(edge_path)
    if image_data.shape[0] == 3:
        image_data = image_data[0, :, :]

    if image_data.max() > 1:
        binary_mask = (image_data > 127).astype(np.uint8)
    else:
        binary_mask = image_data.astype(np.uint8)
    inverted_mask = 1 - binary_mask
    # 准欧几里德距离
    result = cv2.distanceTransform(src=inverted_mask, distanceType=cv2.DIST_L2, maskSize=3)
    # 欧几里得距离
    # result = cv2.distanceTransform(src=result, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE)
    # min-max归一化
    min_value = np.min(result)
    max_value = np.max(result)
    scaled_image = ((result - min_value) / (max_value - min_value)) * 255
    result = scaled_image.astype(np.uint8)
    save_with_geo(dist_path, geo_trans, geo_proj, result)
    print("OK")
    return 0

def make_edge(mask_path, edge_path, edge_width=2):
    geo_trans, geo_proj, image_data = get_geo_info(mask_path)
    if image_data.shape[0] == 3:
        image_data = image_data[0, :, :]

    # 边缘检测
    edge = cv2.Canny(image_data, 100, 200)

    # 调整边缘宽度
    if edge_width > 1:
        kernel = np.ones((edge_width, edge_width), np.uint8)
        edge = cv2.dilate(edge, kernel, iterations=1)

    save_with_geo(edge_path, geo_trans, geo_proj, edge)
    print("OK")
    return 0


if __name__ == '__main__':
     mask_dir      = r"C:\Users\Administrator\Desktop\data\out\ablate\out_masks\masks"
    # dist_edge_dir = r'C:\Users\Administrator\Desktop\data\visual\data\ai4\dist_edges'
    # dist_mask_dir = r'C:\Users\Administrator\Desktop\data\visual\data\ai4\dist_masks'
     edge_dir      = r'C:\Users\Administrator\Desktop\data\out\ablate\out_edges\edges'
     for file_name in os.listdir(mask_dir):
         edge_path = os.path.join(edge_dir, file_name)
         mask_path = os.path.join(mask_dir, file_name)
         make_edge(mask_path, edge_path)
    #     dist_mask_path = os.path.join(dist_mask_dir, file_name)
    #     make_dist_mask(mask_path, dist_mask_path)
    #     dist_edge_path = os.path.join(dist_edge_dir, file_name)
    #     make_dist_edge(edge_path, dist_edge_path)
    # mask_path = r"C:\Users\Administrator\Desktop\data\out\ablate\masks"
    # edge_path = r"C:\Users\Administrator\Desktop\data\visual_boundary\JS\pred_edge_01.png"
    # make_edge(mask_path, edge_path)