"""统一训练脚本（U-Net 与 Attention U-Net 使用完全相同的训练协议，保证公平对比）。

用法:
    python src/train.py --dataset busi  --model unet
    python src/train.py --dataset tn3k  --model attunet --epochs 150

协议:
- 输入 256x256, ImageNet 归一化; 训练集做翻转/旋转/缩放平移/亮度对比度增强
- 损失 = BCEWithLogits + Soft Dice
- Adam(lr=1e-3) + 余弦退火; 混合精度; 早停(验证集 Dice, patience=30)
- 最佳权重 -> outputs/checkpoints/{tag}_best.pth
- 日志 -> outputs/logs/{tag}.csv; 曲线 -> outputs/figures/{tag}_curves.png
"""
import argparse
import csv
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import UltrasoundSegDataset
from models import build_model


def setup_cjk_font():
    """设置图中文字体（避免标题中文显示为方框）"""
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback", "WenQuanYi Zen Hei"):
        if cand in names:
            plt.rcParams["font.sans-serif"] = [cand]
            break
    plt.rcParams["axes.unicode_minus"] = False


setup_cjk_font()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(1, 2, 3)) + eps
    den = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps
    return 1 - (num / den).mean()


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets) + dice_loss(logits, targets)


@torch.no_grad()
def validate(model, loader, device, use_amp):
    """返回 (平均验证损失, 平均逐图 Dice)。"""
    model.eval()
    losses, dices = [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = bce_dice_loss(logits, masks)
        preds = (torch.sigmoid(logits) > 0.5).float()
        inter = (preds * masks).sum(dim=(1, 2, 3))
        union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
        dices += ((2 * inter + 1e-7) / (union + 1e-7)).tolist()
        losses.append(loss.item())
    return float(np.mean(losses)), float(np.mean(dices))


def plot_curves(log_rows, save_path: Path):
    epochs = [r["epoch"] for r in log_rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [r["train_loss"] for r in log_rows], label="train")
    axes[0].plot(epochs, [r["val_loss"] for r in log_rows], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss (BCE+Dice)")
    axes[0].set_title("损失曲线")
    axes[0].legend()
    axes[1].plot(epochs, [r["val_dice"] for r in log_rows], color="tab:green")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("val Dice")
    axes[1].set_title("验证集 Dice 曲线")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["busi", "tn3k"], required=True)
    ap.add_argument("--model", choices=["unet", "attunet"], required=True)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=30, help="早停耐心值")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--max-train-batches", type=int, default=0, help="调试用: 每轮限制批数, 0 不限制")
    args = ap.parse_args()

    tag = args.tag or f"{args.model}_{args.dataset}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    set_seed(args.seed)

    # 目录
    ckpt_dir = Path(args.out_dir) / "checkpoints"
    log_dir = Path(args.out_dir) / "logs"
    fig_dir = Path(args.out_dir) / "figures"
    for d in (ckpt_dir, log_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 数据
    split_dir = Path(args.data_root) / "splits"
    train_ds = UltrasoundSegDataset(split_dir / f"{args.dataset}_train.txt", args.data_root, args.img_size, train=True)
    val_ds = UltrasoundSegDataset(split_dir / f"{args.dataset}_val.txt", args.data_root, args.img_size, train=False)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # 模型 / 优化器
    model = build_model(args.model, base=args.base).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"== 训练 {tag} ==")
    print(f"设备: {device} | 参数量: {n_params/1e6:.2f} M | 训练集: {len(train_ds)} | 验证集: {len(val_ds)}")

    log_rows = []
    best_dice, best_epoch, no_improve = -1.0, -1, 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n_batches = 0.0, 0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"[{tag}] epoch {epoch}/{args.epochs}", leave=False)
        for bi, batch in enumerate(pbar):
            if args.max_train_batches and bi >= args.max_train_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = bce_dice_loss(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()

        train_loss = running / max(n_batches, 1)
        val_loss, val_dice = validate(model, val_loader, device, use_amp)
        epoch_sec = time.time() - t0
        log_rows.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_dice": val_dice,
             "lr": optimizer.param_groups[0]["lr"], "sec": round(epoch_sec, 1)}
        )
        marker = ""
        if val_dice > best_dice:
            best_dice, best_epoch, no_improve = val_dice, epoch, 0
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "epoch": epoch, "val_dice": val_dice},
                ckpt_dir / f"{tag}_best.pth",
            )
            marker = " *best"
        else:
            no_improve += 1
        print(
            f"epoch {epoch:3d}/{args.epochs} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} "
            f"| val_dice {val_dice:.4f} | {epoch_sec:.0f}s{marker}"
        )
        if no_improve >= args.patience:
            print(f"验证集 Dice 连续 {args.patience} 轮无提升，提前停止（最佳 epoch {best_epoch}, dice {best_dice:.4f}）")
            break

    # 保存日志与曲线
    csv_path = log_dir / f"{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_dice", "lr", "sec"])
        writer.writeheader()
        writer.writerows(log_rows)
    plot_curves(log_rows, fig_dir / f"{tag}_curves.png")

    total_min = (time.time() - t_start) / 60
    print(f"训练完成: {tag} | 最佳验证 Dice {best_dice:.4f} (epoch {best_epoch}) | 总耗时 {total_min:.1f} 分钟")
    print(f"权重: {ckpt_dir / f'{tag}_best.pth'}")


if __name__ == "__main__":
    main()
