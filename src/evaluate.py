"""测试集评估: 逐图指标 + 汇总(均值±标准差) + 预测可视化。

用法:
    python src/evaluate.py --checkpoint outputs/checkpoints/unet_busi_best.pth

输出:
- outputs/metrics/{tag}_per_image.csv   逐图指标
- outputs/metrics/{tag}_summary.json    汇总指标(均值/标准差)
- outputs/figures/{tag}_predictions.png 预测可视化(原图/真值/预测/边缘叠加)
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import binary_erosion

from dataset import IMAGENET_MEAN, IMAGENET_STD, UltrasoundSegDataset
from metrics import compute_all
from models import build_model

METRIC_KEYS = ["dice", "iou", "se", "sp", "acc", "hd95"]


def setup_cjk_font():
    """设置图中文字体（避免标题中文显示为方框）"""
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback", "WenQuanYi Zen Hei"):
        if cand in names:
            plt.rcParams["font.sans-serif"] = [cand]
            break
    plt.rcParams["axes.unicode_minus"] = False


setup_cjk_font()


@torch.no_grad()
def predict_prob(model, image_t, device, use_amp):
    model.eval()
    with torch.amp.autocast("cuda", enabled=use_amp):
        logits = model(image_t.unsqueeze(0).to(device))
    return torch.sigmoid(logits)[0, 0].float().cpu().numpy()


def make_prediction_figure(model, dataset, save_path: Path, device, use_amp, num_vis=6, seed=0):
    n = len(dataset)
    idxs = np.linspace(0, n - 1, min(num_vis, n)).astype(int)
    fig, axes = plt.subplots(len(idxs), 4, figsize=(13, 3.3 * len(idxs)))
    col_titles = ["原图", "真值 (GT)", "预测", "边缘叠加 (绿=GT, 红=预测)"]
    for row, idx in enumerate(idxs):
        sample = dataset[int(idx)]
        prob = predict_prob(model, sample["image"], device, use_amp)
        pred = (prob > 0.5).astype(np.uint8)
        gt = sample["mask"][0].numpy().astype(np.uint8)
        disp = sample["image"].permute(1, 2, 0).numpy() * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
        disp = np.clip(disp, 0, 1)
        overlay = disp.copy()
        gt_edge = gt.astype(bool) ^ binary_erosion(gt.astype(bool))
        pred_edge = pred.astype(bool) ^ binary_erosion(pred.astype(bool))
        overlay[gt_edge] = [0.0, 0.9, 0.0]
        overlay[pred_edge] = [0.9, 0.0, 0.0]
        for col, (im, title) in enumerate(
            zip([disp, gt, pred, overlay], col_titles)
        ):
            ax = axes[row][col]
            ax.imshow(im, cmap="gray" if col in (1, 2) else None)
            ax.set_title(title if row == 0 else "", fontsize=10)
            ax.axis("off")
        axes[row][0].set_ylabel(sample["name"], fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"预测可视化已保存: {save_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--num-vis", type=int, default=6)
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    targs = ckpt["args"]
    tag = targs.get("tag") or f"{targs['model']}_{targs['dataset']}"
    dataset_name = targs["dataset"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    model = build_model(targs["model"], base=targs.get("base", 64))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    data_root = targs.get("data_root", "data")
    img_size = targs.get("img_size", 256)
    split_file = Path(data_root) / "splits" / f"{dataset_name}_{args.split}.txt"
    ds = UltrasoundSegDataset(split_file, data_root, img_size, train=False)
    print(f"== 评估 {tag} | 权重 epoch {ckpt['epoch']} (验证 Dice {ckpt['val_dice']:.4f}) | {args.split} 集 {len(ds)} 张 ==")

    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    rows = []
    for batch in loader:
        images = batch["image"]
        masks = batch["mask"]
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images.to(device))
        probs = torch.sigmoid(logits).float().cpu().numpy()[:, 0]
        for i, name in enumerate(batch["name"]):
            pred = (probs[i] > 0.5).astype(np.uint8)
            gt = masks[i, 0].numpy().astype(np.uint8)
            m = compute_all(pred, gt)
            m = {"name": name, **m}
            rows.append(m)

    # 汇总
    summary = {}
    for k in METRIC_KEYS:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        summary[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}

    metrics_dir = Path(args.out_dir) / "metrics"
    figures_dir = Path(args.out_dir) / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = metrics_dir / f"{tag}_{args.split}_per_image.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name"] + METRIC_KEYS)
        writer.writeheader()
        for r in rows:
            writer.writerow({kk: (round(vv, 6) if isinstance(vv, float) else vv) for kk, vv in r.items()})

    json_path = metrics_dir / f"{tag}_{args.split}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'指标':<8}{'均值':>10}{'标准差':>10}")
    print("-" * 30)
    for k in METRIC_KEYS:
        print(f"{k:<8}{summary[k]['mean']:>10.4f}{summary[k]['std']:>10.4f}")
    print(f"\n逐图指标: {csv_path}")
    print(f"汇总: {json_path}")

    make_prediction_figure(model, ds, figures_dir / f"{tag}_predictions.png", device, use_amp, args.num_vis)


if __name__ == "__main__":
    main()
