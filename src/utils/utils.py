import torch
from matplotlib import pyplot as plt
from datetime import datetime
import os
from osgeo import gdal
from skimage.measure import label
import numpy as np
import cv2


gdal.UseExceptions()

def plot_metrics(train_process, type, out_path):
    metrics = ['loss', 'iou', 'f1', 'precision', 'recall']
    titles = ['Loss', 'IoU', 'F1 Score', 'Precision', 'Recall']

    plt.figure(figsize=(10, 20))  # 每行一个图，纵向拉长

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        plt.subplot(5, 1, idx + 1)
        plt.xlabel('Epoch')
        plt.ylabel(title)
        plt.title(title)
        if type == 'train':
            plt.plot(train_process.epoch, train_process[f'train_{metric}'], 'r-', label=f"train_{metric}")
            plt.plot(train_process.epoch, train_process[f'val_{metric}'], 'b--', label=f"val_{metric}")
        elif type == 'test':
            plt.plot(train_process.epoch, train_process[f'test_{metric}'], 'r-', label=f"test_{metric}")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
    out_name = f"metrics_plot_{type}_{current_time}.png"
    out_full_path = os.path.join(out_path, out_name)

    plt.savefig(out_full_path)
    plt.show()

def compute_iou_binary(outputs, targets, threshold=0.5, smooth=1e-6):
    """
    计算二分类任务的 IoU（交并比） ----->  batch mean
    outputs: tensor of shape (B, 1, H, W) or (B, H, W) -- raw logits
    targets: tensor of shape (B, 1, H, W) or (B, H, W) -- binary ground truth mask
    """
    # 降维(B, H, W)
    if outputs.dim() == 4:
        outputs = outputs.squeeze(1)
    if targets.dim() == 4:
        targets = targets.squeeze(1)

    # preds数值化
    preds = (torch.sigmoid(outputs) > threshold).float()
    targets = targets.float()

    # 交集 直接相乘为1 求和
    intersection = (preds * targets).sum(dim=(1, 2))  # 每个样本交集
    # 并集 相加减去交集
    union = preds.sum(dim=(1, 2)) + targets.sum(dim=(1, 2)) - intersection

    iou = (intersection + smooth) / (union + smooth)
    mean_iou = iou.mean().item()

    return mean_iou, iou.tolist()

def get_geo_info(image_path):
    data = gdal.Open(image_path)
    width = data.RasterXSize
    height = data.RasterYSize
    # 放射变换矩阵
    geo_transform = data.GetGeoTransform()
    # 投影坐标系统
    geo_projection = data.GetProjection()
    # 获取影像数据
    image_data = data.ReadAsArray(0, 0, width, height)

    return geo_transform, geo_projection, image_data

def save_with_geo(filename, geo_transform, geo_projection, image_data):
    dtype = image_data.dtype
    if dtype == np.uint8:
        datatype = gdal.GDT_Byte
    elif dtype == np.uint16:
        datatype = gdal.GDT_UInt16
    elif dtype == np.int16:
        datatype = gdal.GDT_Int16
    elif dtype == np.float32:
        datatype = gdal.GDT_Float32
    elif dtype == np.float64:
        datatype = gdal.GDT_Float64
    else:
        raise ValueError(f"不支持的数据类型: {dtype}")

    if len(image_data.shape) == 3:
        im_bands, im_height, im_width = image_data.shape
    else:
        im_bands = 1
        im_height, im_width = image_data.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(geo_projection)

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(image_data)
    else:
        for band_idx in range(im_bands):
            dataset.GetRasterBand(band_idx + 1).WriteArray(image_data[band_idx])

    dataset.FlushCache()
    del dataset

def out_to_ndarray(outputs):
    """
    降维 数值化 转ndarray 0 255
    :param outputs:
    :return:
    """
    outputs = outputs.squeeze()
    preds = torch.sigmoid(outputs).detach().cpu().numpy()
    mask_ndarray = (preds >= 0.5).astype(np.uint8) * 255

    return mask_ndarray

def binary_classification_metrics(pred_logits, label, threshold=0.5):
    """
    计算二分类图像分割的评估指标 ----> batch mean

    Args:
        pred_logits (Tensor): raw logits，(B, 1, H, W) 或 (B, H, W)
        label (Tensor): GT 标签（0或1），形状同 pred_logits
        threshold (float): 二值化阈值，默认0.5

    Returns:
        dict: 包含 TP, TN, FP, FN, Precision, Recall, F1, OA, IoU
    """
    pred_probs = torch.sigmoid(pred_logits)  # sigmoid
    pred_binary = (pred_probs >= threshold).float()  # 数值化
    label = label.float()

    # 降维(B, H, W)
    if label.dim() == 4:
        label = label.squeeze(1)
    if pred_binary.dim() == 4:
        pred_binary = pred_binary.squeeze(1)

    TP = (pred_binary * label).sum()        # 真正
    FP = (pred_binary * (1 - label)).sum()  # 错正 pred - pred * label  所有预测正减真正
    FN = ((1 - pred_binary) * label).sum()  # 错负 label - pred * label 所有正减正真 得到 剩下没有预测到的正 即为FN
    TN = ((1 - pred_binary) * (1 - label)).sum()  # 真负 pred 和 label都取反相乘

    precision = TP / (TP + FP + 1e-10)  # 预测所有正 真实正的比例
    recall = TP / (TP + FN + 1e-10)     # 所有真实正 预测到的比例
    oa = (TP + TN) / (TP + TN + FP + FN + 1e-10)  # 所有中 预测真实的比例
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    iou = TP / (TP + FP + FN + 1e-10)  # 真正/预测中正和真实正

    return {
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'oa': oa.item(),
        'iou': iou.item(),
    }

def preprocess_mask(mask):
    """
    mask转为np.uint8
    :param mask:
    :return:
    """
    mask = mask.cpu().numpy()  # 转 numpy 再用 astype
    mask = mask.astype(np.uint8)
    return mask

def preprocess_pred(pred):
    """
    pred数值化01 转为np.uint8
    :param pred:
    :return:
    """
    pred = torch.sigmoid(pred).detach().cpu().numpy()
    pred = (pred >= 0.5).astype(np.uint8)
    return pred

def calculate_boundary_metrics(pred_mask, gt_mask, buffer_distance=1):
    """
    边界指标
    :param pred_mask: 二值01 ndarray
    :param gt_mask:  二值01 ndarray
    :param buffer_distance: 边界缓冲距离
    :return:
    """
    # 输入验证和转换
    pred_mask = np.asarray(pred_mask, dtype=np.uint8)
    gt_mask = np.asarray(gt_mask, dtype=np.uint8)
    assert pred_mask.shape == gt_mask.shape, "Input masks must have same shape"
    assert buffer_distance >= 0, "Buffer distance must be non-negative"

    # 二值化处理 01
    pred_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)

    """
    采用形态学提取边缘 sobel/canny只对自然图形有效 
    """
    # 优化边界提取 - 使用更高效的形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))   # 创建形态学结构元素（卷积核）
    # 形态学梯度提取Pred前景边缘 定义：膨胀-腐蚀 = 边界 膨胀：1的区域扩大 腐蚀：1的区域缩小
    pred_boundary = cv2.morphologyEx(pred_bin, cv2.MORPH_GRADIENT, kernel)
    # 形态学梯度提取gt前景边缘
    gt_boundary = cv2.morphologyEx(gt_bin, cv2.MORPH_GRADIENT, kernel)

    # 生成缓冲区 使用距离变换实现精确缓冲
    def create_buffer(boundary, distance):
        if distance == 0:
            return boundary
        # 计算背景到边界的距离
        dist_transform = cv2.distanceTransform(
            1 - boundary, cv2.DIST_L2, 3)
        # 对距离图应用阈值 阈值为边界缓冲距离
        return (dist_transform <= distance).astype(np.uint8)

    # 生成buffer
    pred_buffer = create_buffer(pred_boundary, buffer_distance)
    gt_buffer = create_buffer(gt_boundary, buffer_distance)

    # 计算指标
    gt_edge_count = np.count_nonzero(gt_boundary)      # gt边界中非零像素和
    pred_edge_count = np.count_nonzero(pred_boundary)  # pred边界中非零像素和

    # 完整度：真实边界在预测缓冲区内 真实边界被覆盖了多少
    completeness = (np.sum(gt_boundary * pred_buffer) / gt_edge_count
                    if gt_edge_count > 0 else 0.0)

    # 正确度：预测边界在真实缓冲区内 预测边界有多少是正确
    correctness = (np.sum(pred_boundary * gt_buffer) / pred_edge_count
                   if pred_edge_count > 0 else 0.0)

    # Fbdy分数
    fbdy = (2 * completeness * correctness / (completeness + correctness)
            if (completeness + correctness) > 0 else 0.0)

    return {
             "Completeness": round(float(completeness), 4),
             "Correctness": round(float(correctness), 4),
             "Fbdy": round(float(fbdy), 4)
        }

def compute_object_metrics(pred_mask, gt_mask):
    """
    对象级指标评估
    GOC 多预测的对象数（误检程度 对象面积） GUC 漏掉的对象数（漏检程度 对象面积） GTC 总体评估
    :param pred_mask:
    :param gt_mask:
    :return:
    """
    # 对每个独立前景块标记编号1，2，3…… 连通性考虑 8连通
    # 同一连通区域标号一致 返回为数图
    pred_label = label(pred_mask)
    gt_label = label(gt_mask)

    # 特殊情况处理
    no_prediction = pred_label.max() == 0  # 无预测对象
    no_groundtruth = gt_label.max() == 0  # 无真实对象

    if no_prediction and no_groundtruth:
        return {"GOC": 0.0, "GUC": 0.0, "GTC": 0.0}  # 完美情况
    elif no_prediction:
        return {"GOC": 1.0, "GUC": 1.0, "GTC": 1.0}  # 最差漏检
    elif no_groundtruth:
        return {"GOC": 1.0, "GUC": 1.0, "GTC": 1.0}  # 最差误检

    # 正常情况计算（原逻辑）
    n = pred_label.max()  # 预测对象数量
    total_area = 0  # 所有预测对象总面积
    sum_so = 0  # GOC累计加权误差 误检
    sum_su = 0  # GUC累计加权误差 漏检
    sum_st = 0  # GTC累计加权误差

    # 遍历每一个 预测对象
    for i in range(1, n + 1):
        pi = (pred_label == i)  # 提取预测对象 并布尔化
        pi_area = pi.sum()  # 当前预测对象面积
        if pi_area == 0:
            continue
        """ 与真实标签重叠分析 """
        # np.bincount 返回一个数组 索引i表示整数i出现的次数
        # gt_label返回的是一维数组 存储了对应索引为true的元素
        """
        gt_label = np.array([[0, 1, 1],
                             [0, 2, 2],
                             [0, 2, 0]])
        pi = np.array([[False, True, True],
                       [False, True, True],
                       [False, False, False]])
        gt_label[pi] = [1, 1, 2, 2]
        overlap_counts = np.bincount([1, 1, 2, 2])=([0, 2, 2])
        """
        overlap_counts = np.bincount(gt_label[pi])  #
        overlap_counts[0] = 0
        # 当前预测对象与所有真实对象都没有交集
        if overlap_counts.sum() == 0:
            sum_su += 1.0 * pi_area  # 全部漏检
            sum_so += 0.0 * pi_area  # 零误检
            sum_st += 0.7071 * pi_area  # sqrt(0.5)
            total_area += pi_area  # 总面积
            continue

        # 预测对象 pi 所重叠最多的真实对象编号
        gt_idx = np.argmax(overlap_counts)
        # 对应真实对象的 mask（用于和 pi 计算交集）
        ri = (gt_label == gt_idx)
        # 预测对象与其匹配的真实对象的交集像素数
        intersection = np.logical_and(pi, ri).sum()
        # 真实对象的像素数量
        area_ri = ri.sum()

        so = 1 - intersection / (area_ri + 1e-6)  # 误检程度 交集/真实
        su = 1 - intersection / (pi_area + 1e-6)  # 漏检程度 交集/面积
        st = np.sqrt((so ** 2 + su  ** 2) / 2)

        sum_so += so * pi_area
        sum_su += su * pi_area
        sum_st += st * pi_area
        total_area += pi_area

    goc = sum_so / (total_area + 1e-6)
    guc = sum_su / (total_area + 1e-6)
    gtc = sum_st / (total_area + 1e-6)

    return {
        "GOC": float(goc),
        "GUC": float(guc),
        "GTC": float(gtc)
    }

def edge_eval(pred_path, true_path):
    """
    HBG论文采用
    :param pred_path: pred_edge
    :param true_path: true_edge
    :return:
    """
    try:
        # 读取预测的边缘和实际的边缘tif影像
        # with rasterio.open(pred_path) as pred_img:
        #     predicted_edges = pred_img.read(1)
        #
        # with rasterio.open(true_path) as actual_img:
        #     actual_edges = actual_img.read(1)

        pred_img = gdal.Open(pred_path)
        predicted_edges = pred_img.ReadAsArray()

        actual_img = gdal.Open(true_path)
        actual_edges = actual_img.ReadAsArray()

        # 将tif影像转换为二进制掩码
        predicted_edges = (predicted_edges == 255).astype(np.uint8)
        actual_edges = (actual_edges == 255).astype(np.uint8)

        # 定义膨胀结构元素，例如：一个3x3的单位矩阵（全1）
        structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        # 对图像进行膨胀操作
        predicted_edges = cv2.dilate(predicted_edges, structuring_element)
        actual_edges = cv2.dilate(actual_edges, structuring_element)

        # 计算Completeness (Com)
        true_positive = np.sum(predicted_edges & actual_edges)
        completeness = true_positive / np.sum(actual_edges)



        # 计算Correctness (Corr)
        correctness = true_positive / np.sum(predicted_edges)


        # 计算F1-score (Fedge)
        fedge = 2 * (completeness * correctness) / (completeness + correctness)

        return completeness, correctness, fedge
    except Exception as e:
        return 0, 0, 0

def mask_to_edge(mask):
    """
    HBGNet采用
    :param mask:
    :return:
    """
    mask_bin = (mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edge = cv2.morphologyEx(mask_bin, cv2.MORPH_GRADIENT, kernel)
    return edge

def edge_eval_from_masks(pred_mask, gt_mask):
    """
    HBGNet采用方法 调整输入为mask
    :param pred_mask:
    :param gt_mask:
    :return:
    """
    try:
        # 转成二值uint8
        pred_mask = np.asarray(pred_mask, dtype=np.uint8)
        gt_mask = np.asarray(gt_mask, dtype=np.uint8)
        assert pred_mask.shape == gt_mask.shape, "掩码尺寸必须相同"

        # 生成边缘
        predicted_edges = mask_to_edge(pred_mask)
        actual_edges = mask_to_edge(gt_mask)

        # 膨胀结构元素 5x5
        structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # 膨胀操作
        predicted_edges = cv2.dilate(predicted_edges, structuring_element)
        actual_edges = cv2.dilate(actual_edges, structuring_element)

        # 计算Completeness (Com)
        true_positive = np.sum(predicted_edges & actual_edges)
        completeness = true_positive / np.sum(actual_edges) if np.sum(actual_edges) > 0 else 0

        # 计算Correctness (Corr)
        correctness = true_positive / np.sum(predicted_edges) if np.sum(predicted_edges) > 0 else 0

        # 计算F1-score (Fedge)
        fedge = 2 * (completeness * correctness) / (completeness + correctness) if (completeness + correctness) > 0 else 0

        return {
            "Completeness": round(float(completeness), 4),
            "Correctness": round(float(correctness), 4),
            "Fbdy": round(float(fedge), 4)
        }

    except Exception as e:
        print("Error:", e)
        return 0, 0, 0














