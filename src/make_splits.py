"""生成 train/val/test 划分文件。

- BUSI: 无官方划分。剔除 normal 类（无病灶区域，无法参与二值分割评估），
  对 benign/malignant 按类别做 8:1:1 分层划分（固定随机种子，可复现）。
- TN3K: 直接使用数据集自带官方划分 MainPatient/{train,val,test}.txt。

输出: data/splits/{dataset}_{split}.txt
每行格式: 图像路径 <TAB> 掩膜路径[{,掩膜路径}...]（路径相对 --data-root）

用法:
    python src/make_splits.py --data-root data --out-dir data/splits
"""
import argparse
import random
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def collect_busi(root: Path, include_normal: bool):
    """扫描 BUSI 目录，返回 {类别: [(图像路径, [掩膜路径...])]}"""
    base = root / "BUSI" / "Dataset_BUSI_with_GT"
    per_class = {}
    for cls_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        if cls_dir.name == "normal" and not include_normal:
            continue
        items = []
        for img in sorted(cls_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXTS or "_mask" in img.stem:
                continue
            masks = sorted(cls_dir.glob(f"{img.stem}_mask*"))
            if not masks:
                print(f"[警告] 无掩膜, 跳过: {img}")
                continue
            items.append((img, masks))
        per_class[cls_dir.name] = items
    return per_class


def stratified_busi(per_class, ratios, seed):
    """按类别分层划 8:1:1，返回 {'train': [...], 'val': [...], 'test': [...]}"""
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for cls, items in per_class.items():
        items = items[:]
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        splits["train"] += items[:n_train]
        splits["val"] += items[n_train:n_train + n_val]
        splits["test"] += items[n_train + n_val:]
        print(f"BUSI {cls:9s}: 共 {n:4d} -> train {n_train} / val {n_val} / test {n - n_train - n_val}")
    for k in splits:
        splits[k].sort()
    return splits


def collect_tn3k(root: Path):
    """读取 TN3K 官方划分文件"""
    base = root / "TN3K" / "ThyroidNodule-TN3K"
    splits = {}
    for split in ("train", "val", "test"):
        stems = [
            s.strip()
            for s in (base / "MainPatient" / f"{split}.txt").read_text().splitlines()
            if s.strip()
        ]
        items = []
        for s in stems:
            imgs = sorted((base / "img").glob(f"{s}.*"))
            masks = sorted((base / "label").glob(f"{s}.*"))
            if not imgs or not masks:
                print(f"[警告] {split} 缺少 {s}, 跳过")
                continue
            items.append((imgs[0], [masks[0]]))
        splits[split] = sorted(items)
        print(f"TN3K {split:5s}: {len(items)} 条")
    return splits


def write_split(path: Path, items, data_root: Path):
    lines = []
    for img, masks in items:
        rel_img = img.relative_to(data_root).as_posix()
        rel_masks = ",".join(m.relative_to(data_root).as_posix() for m in masks)
        lines.append(f"{rel_img}\t{rel_masks}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"写入 {path}: {len(lines)} 条")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out-dir", default="data/splits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-normal", action="store_true", help="BUSI 是否包含 normal 类(默认剔除)")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    busi = stratified_busi(collect_busi(data_root, args.include_normal), (0.8, 0.1, 0.1), args.seed)
    for split, items in busi.items():
        write_split(out_dir / f"busi_{split}.txt", items, data_root)

    tn3k = collect_tn3k(data_root)
    for split, items in tn3k.items():
        write_split(out_dir / f"tn3k_{split}.txt", items, data_root)


if __name__ == "__main__":
    main()
