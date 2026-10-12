#!/usr/bin/env bash
# 一次跑完 2 模型 × 2 数据集 = 4 组实验（训练 + 测试集评估）
# 用法: bash run_all.sh   （建议 nohup bash run_all.sh > outputs/logs/run_all.log 2>&1 & 后台运行）
set -e
cd "$(dirname "$0")"

for ds in busi tn3k; do
  for m in unet attunet; do
    echo "##################### train ${m} on ${ds} #####################"
    python3 src/train.py --dataset "$ds" --model "$m"
    echo "##################### evaluate ${m} on ${ds} #####################"
    python3 src/evaluate.py --checkpoint "outputs/checkpoints/${m}_${ds}_best.pth"
  done
done
echo "全部实验完成: 2 模型 × 2 数据集"
