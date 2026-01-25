import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------
# Utilities
# ---------------------------

def gn(ch):
    """Safe GroupNorm"""
    return nn.GroupNorm(min(32, ch), ch)


# ---------------------------
# Time Embedding
# ---------------------------

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t):
        """
        t: (B,), (B,1), or (B,1,1,1)
        returns: (B, dim)
        """
        if t.dim() == 4:
            t = t[:, 0, 0, 0]
        elif t.dim() == 2:
            t = t[:, 0]

        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )

        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return self.mlp(emb)


# ---------------------------
# Residual Block
# ---------------------------

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()

        self.norm1 = gn(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = gn(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_ch)

        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


# ---------------------------
# Down / Up Sampling
# ---------------------------

class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ---------------------------
# Score UNet
# ---------------------------

class ScoreUNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        base_channels=128,
        channel_mults=(1, 2, 2, 4),
        time_dim=128
    ):
        super().__init__()

        self.time_embed = TimeEmbedding(time_dim)

        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Downsampling path
        self.downs = nn.ModuleList()
        ch = base_channels
        self.skip_channels = []

        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            self.downs.append(
                nn.ModuleList([
                    ResBlock(ch, out_ch, time_dim),
                    ResBlock(out_ch, out_ch, time_dim),
                    Downsample(out_ch) if i != len(channel_mults) - 1 else nn.Identity()
                ])
            )
            self.skip_channels.append(out_ch)
            ch = out_ch

        # Bottleneck
        self.mid1 = ResBlock(ch, ch, time_dim)
        self.mid2 = ResBlock(ch, ch, time_dim)

        # Upsampling path
        self.ups = nn.ModuleList()
        for i, (mult, skip_ch) in enumerate(
                reversed(list(zip(channel_mults, self.skip_channels)))
        ):
            out_ch = base_channels * mult
            self.ups.append(
                nn.ModuleList([
                    ResBlock(ch + skip_ch, out_ch, time_dim),
                    ResBlock(out_ch, out_ch, time_dim),
                    Upsample(out_ch) if i != len(channel_mults) - 1 else nn.Identity()
                ])
            )
            ch = out_ch

        self.final_norm = gn(ch)
        self.final_conv = nn.Conv2d(ch, in_channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)

        h = self.init_conv(x)
        skips = []

        for rb1, rb2, down in self.downs:
            h = rb1(h, t_emb)
            h = rb2(h, t_emb)
            skips.append(h)
            h = down(h)

        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        for (rb1, rb2, up), skip in zip(self.ups, reversed(skips)):
            h = torch.cat([h, skip], dim=1)
            h = rb1(h, t_emb)
            h = rb2(h, t_emb)
            h = up(h)

        h = F.silu(self.final_norm(h))
        return self.final_conv(h)

if __name__ == "__main__":
    model = ScoreUNet()
    x = torch.randn(2, 3, 32, 32)
    t = torch.rand(2, 1, 1, 1)

    y = model(x, t)
    print(y.shape)  # must be (2, 3, 32, 32)
