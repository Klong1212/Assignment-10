"""Custom UNet built from scratch for single-class lane segmentation.

Input:  3x48x48 RGB image
Output: 1x48x48 binary lane mask (raw logits; apply sigmoid for probabilities)
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> ReLU) x2, preserves spatial size."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """MaxPool2x2 followed by DoubleConv."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """ConvTranspose2x2 upsample, concat with encoder skip, then DoubleConv."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch * 2, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """4-level UNet sized for 48x48 inputs (48 -> 24 -> 12 -> 6 -> 3 at the bottleneck)."""

    def __init__(self, in_channels=3, num_classes=1, base_ch=32):
        super().__init__()
        self.in_conv = DoubleConv(in_channels, base_ch)        # 48x48, base_ch
        self.down1 = Down(base_ch, base_ch * 2)                # 24x24, base_ch*2
        self.down2 = Down(base_ch * 2, base_ch * 4)            # 12x12, base_ch*4
        self.down3 = Down(base_ch * 4, base_ch * 8)            # 6x6,   base_ch*8
        self.down4 = Down(base_ch * 8, base_ch * 16)           # 3x3,   base_ch*16 (bottleneck)

        self.up1 = Up(base_ch * 16, base_ch * 8)               # -> 6x6
        self.up2 = Up(base_ch * 8, base_ch * 4)                # -> 12x12
        self.up3 = Up(base_ch * 4, base_ch * 2)                # -> 24x24
        self.up4 = Up(base_ch * 2, base_ch)                    # -> 48x48

        self.out_conv = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.out_conv(x)


if __name__ == "__main__":
    model = UNet()
    dummy = torch.randn(2, 3, 48, 48)
    out = model(dummy)
    print(f"Input:  {tuple(dummy.shape)}")
    print(f"Output: {tuple(out.shape)}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")
