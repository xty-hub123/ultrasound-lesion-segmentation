"""超声分割数据集（BUSI / TN3K 通用）。

掩膜规则: 转灰度后 >127 视为前景（TN3K 掩膜经过重采样，值分布在 0~255；
BUSI 为 0/255，两者用同一阈值统一二值化）；多张掩膜（如 BUSI 的
'_mask_1'）取逐像素最大值合并。
"""
from pathlib import Path

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(img_size: int, train: bool) -> A.Compose:
    ops = [A.Resize(img_size, img_size)]
    if train:
        ops += [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=25, border_mode=0, p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), border_mode=0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        ]
    ops += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(ops)


class UltrasoundSegDataset(Dataset):
    """读取 make_splits.py 生成的划分文件（每行: 图像路径 <TAB> 掩膜路径[,...]）。"""

    def __init__(self, split_file, data_root="data", img_size=256, train=True):
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.items = []
        for line in Path(split_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            img_rel, masks_rel = line.split("\t")
            self.items.append((img_rel, masks_rel.split(",")))
        self.transform = build_transform(img_size, train)

    def __len__(self):
        return len(self.items)

    def load_mask(self, mask_rels) -> np.ndarray:
        masks = [np.array(Image.open(self.data_root / rel).convert("L")) for rel in mask_rels]
        mask = np.maximum.reduce(masks) if len(masks) > 1 else masks[0]
        return (mask > 127).astype(np.uint8) * 255

    def __getitem__(self, idx):
        img_rel, mask_rels = self.items[idx]
        image = np.array(Image.open(self.data_root / img_rel).convert("RGB"))
        mask = self.load_mask(mask_rels)
        out = self.transform(image=image, mask=mask)
        image_t = out["image"].float()
        # resize 的插值可能产生中间值，这里再统一二值化一次
        mask_t = (out["mask"] > 127).float().unsqueeze(0)
        return {"image": image_t, "mask": mask_t, "name": Path(img_rel).stem}


if __name__ == "__main__":
    # 冒烟测试: python src/dataset.py [划分文件]
    import sys

    split = sys.argv[1] if len(sys.argv) > 1 else "data/splits/busi_train.txt"
    ds = UltrasoundSegDataset(split, img_size=256, train=True)
    print(f"{split}: {len(ds)} 条")
    s = ds[0]
    print(
        "image:", tuple(s["image"].shape), s["image"].dtype,
        f"min={s['image'].min():.2f} max={s['image'].max():.2f}",
    )
    print("mask :", tuple(s["mask"].shape), s["mask"].dtype, "前景像素:", int(s["mask"].sum()))
