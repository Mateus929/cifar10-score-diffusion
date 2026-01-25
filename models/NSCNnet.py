import torch.nn as nn
import torch.nn.functional as F

class CondResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=False, dilation=1):
        super().__init__()
        stride = 2 if down else 1
        padding = dilation

        self.conv1 = nn.Conv2d(
            in_ch, out_ch, 3, stride=stride, padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=padding, dilation=dilation)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, stride=stride) if in_ch != out_ch or down else nn.Identity()

    def forward(self, x):
        h = F.silu(self.conv1(x))
        h = self.conv2(F.silu(h))
        return h + self.skip(x)

class CondRefineBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = F.silu(self.conv1(x))
        h = self.conv2(F.silu(h))
        return h + self.skip(x)

class NSCNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.in_conv = nn.Conv2d(3, 128, 3, padding=1)

        # Downsampling / ResBlocks
        self.b1 = CondResBlock(128, 128)
        self.b2 = CondResBlock(128, 128)

        self.b3 = CondResBlock(128, 256, down=True)
        self.b4 = CondResBlock(256, 256)

        self.b5 = CondResBlock(256, 256, down=True, dilation=2)
        self.b6 = CondResBlock(256, 256, dilation=2)

        self.b7 = CondResBlock(256, 256, down=True, dilation=4)
        self.b8 = CondResBlock(256, 256, dilation=4)

        # Refine / upsampling blocks
        self.r1 = CondRefineBlock(256, 256)
        self.r2 = CondRefineBlock(256, 256)

        # Upsample channels from 256 -> 128
        self.upsample1 = nn.Conv2d(256, 128, 1)
        self.r3 = CondRefineBlock(128, 128)

        self.upsample2 = nn.Conv2d(128, 128, 1)
        self.r4 = CondRefineBlock(128, 128)

        self.out_conv = nn.Conv2d(128, 3, 3, padding=1)

    def forward(self, x):
        # Downsampling path
        x = self.in_conv(x)
        x = self.b1(x)
        x = self.b2(x)

        x = self.b3(x)
        x = self.b4(x)

        x = self.b5(x)
        x = self.b6(x)

        x = self.b7(x)
        x = self.b8(x)

        # Refinement
        x = self.r1(x)
        x = self.r2(x)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.upsample1(x)
        x = self.r3(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.upsample2(x)
        x = self.r4(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.out_conv(x)



if __name__ == "__main__":
    import torch
    model = NSCNModel()
    model.eval()
    B, C, H, W = 4, 3, 32, 32
    x = torch.randn(B, C, H, W)
    sigma = torch.rand(B) * 0.99 + 0.01  # (0.01, 1)

    with torch.no_grad():
        y = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)
