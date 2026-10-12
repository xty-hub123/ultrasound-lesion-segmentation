#!/usr/bin/env bash
# 接着跑剩余实验: unet_tn3k + attunet_tn3k（BUSI 两组已完成, 不重复跑）
# 用法: nohup bash run_rest.sh > outputs/logs/run_rest.log 2>&1 &
set -e
cd "$(dirname "$0")"

for m in unet attunet; do
  echo "##################### train ${m} on tn3k #####################"
  python3 src/train.py --dataset tn3k --model "$m"
  echo "##################### evaluate ${m} on tn3k #####################"
  python3 src/evaluate.py --checkpoint "outputs/checkpoints/${m}_tn3k_best.pth"
done
echo "剩余实验完成: unet_tn3k + attunet_tn3k"
