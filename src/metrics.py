"""分割评价指标: Dice / IoU / Sensitivity / Specificity / Accuracy / HD95。

HD95（95% Hausdorff Distance）用 scipy 距离变换实现：分别计算预测掩膜表面
到真值表面、以及真值表面到预测掩膜表面的距离，取全部表面距离的 95 百分位数。
"""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def confusion(pred: np.ndarray, gt: np.ndarray):
    """返回 TP, FP, FN, TN（逐像素统计）。"""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())
    return tp, fp, fn, tn


def dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    tp, fp, fn, _ = confusion(pred, gt)
    return (2 * tp + eps) / (2 * tp + fp + fn + eps)


def iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    tp, fp, fn, _ = confusion(pred, gt)
    return (tp + eps) / (tp + fp + fn + eps)


def sensitivity(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    """敏感度 / 召回率: TP / (TP + FN)，衡量病灶是否被漏检。"""
    tp, _, fn, _ = confusion(pred, gt)
    return (tp + eps) / (tp + fn + eps)


def specificity(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    """特异度: TN / (TN + FP)，衡量背景是否被误判。"""
    _, fp, _, tn = confusion(pred, gt)
    return (tn + eps) / (tn + fp + eps)


def accuracy(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    tp, fp, fn, tn = confusion(pred, gt)
    return (tp + tn + eps) / (tp + fp + fn + tn + eps)


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """95% Hausdorff 距离（单位: 像素，越小越好）。

    一方为空时的处理: 两者都空返回 0；仅一方为空返回图像对角线长度作为惩罚。
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        h, w = gt.shape
        return float(np.hypot(h, w))

    pred_surf = pred ^ binary_erosion(pred)
    gt_surf = gt ^ binary_erosion(gt)
    # 每个像素到目标表面的最近距离
    dt_to_gt = distance_transform_edt(~gt_surf)
    dt_to_pred = distance_transform_edt(~pred_surf)
    d_pred2gt = dt_to_gt[pred_surf]
    d_gt2pred = dt_pred = dt_to_pred[gt_surf]
    return float(np.percentile(np.concatenate([d_pred2gt, d_gt2pred]), 95))


def compute_all(pred: np.ndarray, gt: np.ndarray) -> dict:
    """一次计算全部指标，pred/gt 为二值 0/1 数组。"""
    return {
        "dice": dice(pred, gt),
        "iou": iou(pred, gt),
        "se": sensitivity(pred, gt),
        "sp": specificity(pred, gt),
        "acc": accuracy(pred, gt),
        "hd95": hd95(pred, gt),
    }
