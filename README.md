# 超声病灶分割：U-Net 与 Attention U-Net 对比研究

《深度学习与医学应用》小组作业 —— 路线二：**超声病灶分割**。

在乳腺超声（BUSI）与甲状腺超声（TN3K）两个数据集上，训练并评估
**U-Net** 与 **Attention U-Net** 两个 CNN 模型（2 模型 × 2 数据集 = 4 组实验）。

## 实验设计

| 项目 | 说明 |
|---|---|
| 任务 | 医学超声图像二值分割（病灶区域） |
| 模型 1 | U-Net（Ronneberger et al., MICCAI 2015），约 31.0M 参数 |
| 模型 2 | Attention U-Net（Oktay et al., 2018），约 31.4M 参数 |
| 数据集 1 | BUSI 乳腺超声，780 张（剔除 normal 类后 647 张参与分割实验） |
| 数据集 2 | TN3K 甲状腺结节超声，3493 张 |

**公平对比设计**：两个模型在 `src/models.py` 中共用完全相同的编码器/解码器
结构，唯一区别是 Attention U-Net 在 4 处跳跃连接上加入了 Attention Gate。

## 数据

### BUSI（乳腺超声）
- 来源：原论文 Al-Dhabyani et al., Data in Brief, 2020；本仓库使用 ModelScope
  镜像（`yurkie/busi`，即官方 `Dataset_BUSI_with_GT.zip`）。
- 结构：`data/BUSI/Dataset_BUSI_with_GT/{benign,malignant,normal}/`，
  每张图对应 `*_mask.png`（部分图有多张 mask，读取时取最大值合并）。
- **划分**：无官方划分。剔除 normal（无病灶，无法做二值分割评估），对
  benign/malignant 按类别分层 8:1:1（seed=42）→ 训练 517 / 验证 64 / 测试 66。

### TN3K（甲状腺结节超声）
- 来源：Gong et al., ISBI 2021（南方医科大学珠江医院）；本仓库使用
  HuggingFace `us-segmentator/us-segmentation-dataset` 的 TN3K 包（via hf-mirror）。
- 结构：`data/TN3K/ThyroidNodule-TN3K/{img,label}/`，图像已统一为 256×256；
  掩膜值分布 0~255（重采样导致），统一按 `>127` 二值化。
- **划分**：官方划分 训练 2303 / 验证 576 / 测试 614（`MainPatient/*.txt`）。

## 训练协议（两模型完全一致）

- 输入 256×256，ImageNet 归一化
- 数据增强（仅训练集）：水平翻转 / 旋转 ±25° / 缩放平移 / 亮度对比度
- 损失：BCEWithLogits + Soft Dice；优化器 Adam(lr=1e-3) + 余弦退火至 1e-5
- 150 epochs，早停 patience=30（监控验证集 Dice），batch 16，混合精度，seed=42

## 评价指标

Dice、IoU、敏感度 SE、特异度 SP、像素准确率 ACC、HD95（95% Hausdorff 距离，像素）。

## 目录结构

```
├── data/
│   ├── BUSI/            # 原始数据（解压自 Dataset_BUSI.zip）
│   ├── TN3K/
│   ├── downloads/       # 原始 zip
│   └── splits/          # 生成的划分文件（图像路径<TAB>掩膜路径[,...]）
├── src/
│   ├── make_splits.py   # 生成划分文件
│   ├── dataset.py       # 数据集与增强
│   ├── models.py        # U-Net / Attention U-Net
│   ├── metrics.py       # 指标计算
│   ├── train.py         # 训练
│   ├── evaluate.py      # 评估与可视化
│   └── summarize.py     # 四组结果汇总（生成 comparison_table.md）
├── outputs/
│   ├── checkpoints/     # 最佳权重 {model}_{dataset}_best.pth
│   ├── logs/            # 训练日志 csv
│   ├── figures/         # 训练曲线、预测可视化
│   └── metrics/         # 逐图指标 csv + 汇总 json
└── run_all.sh           # 一键跑完 4 组实验
```

## 运行步骤

```bash
# 1) 生成划分（已执行过, 可跳过）
python3 src/make_splits.py

# 2) 单组实验
python3 src/train.py --dataset busi --model unet
python3 src/evaluate.py --checkpoint outputs/checkpoints/unet_busi_best.pth

# 3) 汇总四组结果 → outputs/metrics/comparison_table.md
python3 src/summarize.py

# 4) 或一键跑完 4 组（后台运行）
nohup bash run_all.sh > outputs/logs/run_all.log 2>&1 &
tail -f outputs/logs/run_all.log   # 查看进度
```

## 环境

- Python 3.10（conda env `deepcre_tpm`），PyTorch 2.5.1+cu121，GPU RTX 3070 (8GB)
- 依赖：`albumentations==2.0.8`、`Pillow`、`scipy`、`numpy<2`、`tqdm`、`matplotlib`
- 注：环境中 numpy 已固定为 1.26.4（兼容原有 tensorflow 2.15），安装依赖时勿升级 numpy。

## 结果（2026-10-09 完成）

四组实验在各自测试集上的最终结果（均值±标准差，详见 `outputs/metrics/comparison_table.md` 与 `docs/课程设计报告.md` 第 5 节）：

| 数据集 | 模型 | Dice | IoU | SE | SP | ACC | HD95 |
|---|---|---|---|---|---|---|---|
| BUSI | U-Net | 0.7490±0.2287 | 0.6428±0.2495 | 0.7652±0.2711 | 0.9802±0.0345 | 0.9566±0.0518 | **39.14**±38.71 |
| BUSI | Attention U-Net | **0.7564**±0.2443 | **0.6563**±0.2527 | **0.7726**±0.2713 | **0.9822**±0.0363 | **0.9575**±0.0553 | 41.63±57.59 |
| TN3K | U-Net | 0.8138±0.2094 | 0.7252±0.2263 | 0.8432±0.2257 | 0.9856±0.0233 | 0.9690±0.0423 | 31.88±55.75 |
| TN3K | Attention U-Net | **0.8217**±0.2071 | **0.7357**±0.2226 | **0.8441**±0.2244 | **0.9861**±0.0219 | **0.9703**±0.0411 | **31.05**±56.79 |

**主要结论**：Attention U-Net 在两个数据集上 Dice 均小幅提升（BUSI +0.0074、TN3K +0.0079，相对约 +1%），逐图看涨跌互现（TN3K 614 张中 235 涨、205 跌）；TN3K 整体优于 BUSI（数据量 4.5 倍、结节形态更规整）；HD95 在 TN3K 上更优、BUSI 上略差。

**参考基线（文献值，划分可能不同，仅作参照）**：
- BUSI：SOTA 级 Dice ≈ 80%（DCCE-UNet 79.85%；UESA-Net 73.44%）
- TN3K：SOTA 级 Dice ≈ 89%~91%（DCCE-UNet 89.29%；UESA-Net 91.39%）
