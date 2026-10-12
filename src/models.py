"""U-Net 与 Attention U-Net 实现。

两个模型共用完全相同的编码器/解码器结构，唯一区别是 Attention U-Net 在
跳跃连接处加入了 Attention Gate，从而保证对比实验的公平性。

参考文献:
- Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image
  Segmentation", MICCAI 2015.
- Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas",
  arXiv:1804.03999, 2018.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> ReLU) x 2"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionGate(nn.Module):
    """Oktay et al. (2018) 注意力门控。

    g: 解码器上采样后的门控信号, 形状 (B, f_g, H, W)
    x: 编码器跳跃连接特征,   形状 (B, f_l, H, W)
    输出: 逐像素加权后的 x（权重 alpha 由 g 与 x 共同决定）
    """

    def __init__(self, f_g: int, f_l: int, f_int: int):
        super().__init__()
        self.w_g = nn.Sequential(nn.Conv2d(f_g, f_int, kernel_size=1), nn.BatchNorm2d(f_int))
        self.w_x = nn.Sequential(nn.Conv2d(f_l, f_int, kernel_size=1), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1), nn.BatchNorm2d(1), nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        alpha = self.psi(self.relu(self.w_g(g) + self.w_x(x)))
        return x * alpha


class UpBlock(nn.Module):
    """上采样 -> [可选] 注意力门控 -> 与编码器特征拼接 -> DoubleConv"""

    def __init__(self, in_ch: int, sk_ch: int, out_ch: int, use_attention: bool):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, sk_ch, kernel_size=2, stride=2)
        self.use_attention = use_attention
        if use_attention:
            self.att = AttentionGate(f_g=sk_ch, f_l=sk_ch, f_int=max(sk_ch // 2, 1))
        self.conv = DoubleConv(sk_ch * 2, out_ch)

    def forward(self, x_dec: torch.Tensor, x_enc: torch.Tensor) -> torch.Tensor:
        x_dec = self.up(x_dec)
        # 输入尺寸不是 2 的幂时，上采样结果可能与编码器特征差一个像素，这里对齐
        if x_dec.shape[2:] != x_enc.shape[2:]:
            x_dec = F.interpolate(
                x_dec, size=x_enc.shape[2:], mode="bilinear", align_corners=False
            )
        if self.use_attention:
            x_enc = self.att(g=x_dec, x=x_enc)
        return self.conv(torch.cat([x_enc, x_dec], dim=1))


class UNet(nn.Module):
    """经典 U-Net。use_attention=True 时即 Attention U-Net（其余结构完全相同）。"""

    def __init__(
        self, in_ch: int = 3, out_ch: int = 1, base: int = 64, use_attention: bool = False
    ):
        super().__init__()
        c = base
        self.inc = DoubleConv(in_ch, c)                                        # 1/1
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c, c * 2))      # 1/2
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c * 2, c * 4))  # 1/4
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c * 4, c * 8))  # 1/8
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c * 8, c * 16)) # 1/16 瓶颈
        self.up1 = UpBlock(c * 16, c * 8, c * 8, use_attention)
        self.up2 = UpBlock(c * 8, c * 4, c * 4, use_attention)
        self.up3 = UpBlock(c * 4, c * 2, c * 2, use_attention)
        self.up4 = UpBlock(c * 2, c, c, use_attention)
        self.out_conv = nn.Conv2d(c, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out_conv(x)  # 输出 logits，损失函数内部再做 sigmoid


def build_model(name: str, in_ch: int = 3, out_ch: int = 1, base: int = 64) -> nn.Module:
    """按名称构建模型: 'unet' 或 'attunet'"""
    if name == "unet":
        return UNet(in_ch, out_ch, base, use_attention=False)
    if name == "attunet":
        return UNet(in_ch, out_ch, base, use_attention=True)
    raise ValueError(f"未知模型名称: {name}")


if __name__ == "__main__":
    # 自检：参数量与输出尺寸
    for n in ("unet", "attunet"):
        m = build_model(n)
        num = sum(p.numel() for p in m.parameters())
        print(f"{n:8s} {num/1e6:.2f} M 参数")
    x = torch.randn(2, 3, 256, 256)
    for n in ("unet", "attunet"):
        with torch.no_grad():
            y = build_model(n)(x)
        print(f"{n:8s} 输入 {tuple(x.shape)} -> 输出 {tuple(y.shape)}")
