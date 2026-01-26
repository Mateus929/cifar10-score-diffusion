import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SigmaEmbed(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, sigma):
        """
        sigma: (B,) tensor
        returns: (B, embed_dim)
        """
        device = sigma.device
        half_dim = self.embed_dim // 2
        emb = torch.log(sigma + 1e-8)[:, None] * torch.exp(
            torch.arange(half_dim, device=device) * -np.log(10000) / (half_dim - 1)
        )
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.fc(emb)

class CondResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=False, dilation=1, embed_dim=128):
        super().__init__()
        stride = 2 if down else 1
        padding = dilation

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=padding, dilation=dilation)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, stride=stride) if in_ch != out_ch or down else nn.Identity()

        # Project sigma embedding to channel dimension
        self.embed_proj = nn.Linear(embed_dim, out_ch)

    def forward(self, x, sigma_emb):
        h = self.conv1(x)
        h = h + self.embed_proj(sigma_emb)[:, :, None, None]
        h = F.silu(h)
        h = self.conv2(h)
        h = F.silu(h)
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
    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.sigma_embed = SigmaEmbed(embed_dim)

        # Initial conv
        self.in_conv = nn.Conv2d(3, 128, 3, padding=1)

        # Downsampling / Residual blocks
        self.b1 = CondResBlock(128, 128, embed_dim=embed_dim)
        self.b2 = CondResBlock(128, 128, embed_dim=embed_dim)

        self.b3 = CondResBlock(128, 256, down=True, embed_dim=embed_dim)
        self.b4 = CondResBlock(256, 256, embed_dim=embed_dim)

        self.b5 = CondResBlock(256, 256, down=True, dilation=2, embed_dim=embed_dim)
        self.b6 = CondResBlock(256, 256, dilation=2, embed_dim=embed_dim)

        self.b7 = CondResBlock(256, 256, down=True, dilation=4, embed_dim=embed_dim)
        self.b8 = CondResBlock(256, 256, dilation=4, embed_dim=embed_dim)

        # Refinement / Upsampling blocks
        self.r1 = CondRefineBlock(256, 256)
        self.r2 = CondRefineBlock(256, 256)
        self.upsample1 = nn.Conv2d(256, 128, 1)
        self.r3 = CondRefineBlock(128, 128)
        self.upsample2 = nn.Conv2d(128, 128, 1)
        self.r4 = CondRefineBlock(128, 128)

        # Output conv
        self.out_conv = nn.Conv2d(128, 3, 3, padding=1)

    def forward(self, x, sigma):
        """
        x: (B, 3, H, W)
        sigma: (B,) noise levels
        """

        if sigma.ndim > 1:
            sigma = sigma.view(-1)


        sigma_emb = self.sigma_embed(sigma)

        # Downsampling path
        x = self.in_conv(x)
        x = self.b1(x, sigma_emb)
        x = self.b2(x, sigma_emb)
        x = self.b3(x, sigma_emb)
        x = self.b4(x, sigma_emb)
        x = self.b5(x, sigma_emb)
        x = self.b6(x, sigma_emb)
        x = self.b7(x, sigma_emb)
        x = self.b8(x, sigma_emb)

        # Refinement / Upsampling path
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
    B, C, H, W = 4, 3, 32, 32
    x = torch.randn(B, C, H, W)
    sigma = torch.rand(B) * 0.99 + 0.01  # random sigma in (0.01,1)

    model = NSCNModel()
    with torch.no_grad():
        y = model(x, sigma)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)
