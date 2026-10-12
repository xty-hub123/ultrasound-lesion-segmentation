"""汇总 4 组实验的测试集指标，生成 markdown 对比表。

用法: python3 src/summarize.py
读取 outputs/metrics/{model}_{dataset}_test_summary.json
输出到 outputs/metrics/comparison_table.md 并打印。
"""
import json
from pathlib import Path

MODELS = ["unet", "attunet"]
DATASETS = ["busi", "tn3k"]
KEYS = ["dice", "iou", "se", "sp", "acc", "hd95"]
MODEL_NAMES = {"unet": "U-Net", "attunet": "Attention U-Net"}
DATASET_NAMES = {"busi": "BUSI", "tn3k": "TN3K"}


def main():
    lines = ["| 数据集 | 模型 | Dice | IoU | SE | SP | ACC | HD95 |", "|---|---|---|---|---|---|---|---|"]
    missing = []
    for ds in DATASETS:
        for m in MODELS:
            path = Path(f"outputs/metrics/{m}_{ds}_test_summary.json")
            if not path.exists():
                missing.append(str(path))
                continue
            s = json.loads(path.read_text(encoding="utf-8"))
            cells = " | ".join(f"{s[k]['mean']:.4f}±{s[k]['std']:.4f}" for k in KEYS)
            lines.append(f"| {DATASET_NAMES[ds]} | {MODEL_NAMES[m]} | {cells} |")
    table = "\n".join(lines)
    print(table)
    out = Path("outputs/metrics/comparison_table.md")
    out.write_text(table + "\n", encoding="utf-8")
    print(f"\n已保存: {out}")
    if missing:
        print("以下实验结果缺失（对应实验未完成）:")
        for m in missing:
            print(" -", m)


if __name__ == "__main__":
    main()
