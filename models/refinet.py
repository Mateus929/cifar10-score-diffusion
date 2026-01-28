import torch
import torch.nn as nn
from models.layers import ResidualBlock, RefineNetBlock, CondInstanceNorm, AdaptiveConvBlock


class RefineNet(nn.Module):
    def __init__(self, in_channels, hidden_channels=(64, 128, 256, 512), n_noise_scale=5):
        super().__init__()

        # -------------------
        # Encoder residual blocks (memory-friendly)
        # -------------------
        self.res1 = ResidualBlock(in_channels, hidden_channels[0], n_layers=1, downsample='stride')
        self.res2 = ResidualBlock(hidden_channels[0], hidden_channels[1], n_layers=1, downsample='dilation', dilation=1)
        self.res3 = ResidualBlock(hidden_channels[1], hidden_channels[2], n_layers=1, downsample='dilation', dilation=1)
        self.res4 = ResidualBlock(hidden_channels[2], hidden_channels[3], n_layers=1, downsample='dilation', dilation=1)

        # -------------------
        # RefineNet blocks
        # -------------------
        self.refine1 = RefineNetBlock(x1_in=hidden_channels[-1], x2_in=hidden_channels[-1],
                                      channels=hidden_channels[-1], n_noise_scale=n_noise_scale)
        self.refine2 = RefineNetBlock(x1_in=hidden_channels[-2], x2_in=hidden_channels[-1],
                                      channels=hidden_channels[-2], n_noise_scale=n_noise_scale)
        self.refine3 = RefineNetBlock(x1_in=hidden_channels[-3], x2_in=hidden_channels[-2],
                                      channels=hidden_channels[-3], n_noise_scale=n_noise_scale)
        self.refine4 = RefineNetBlock(x1_in=hidden_channels[-4], x2_in=hidden_channels[-3],
                                      channels=hidden_channels[-4], n_noise_scale=n_noise_scale)

        # -------------------
        # Upsample to original resolution
        # -------------------
        self.up_norm = CondInstanceNorm(hidden_channels[-4], n_noise_scale)
        self.up_conv = nn.ConvTranspose2d(hidden_channels[-4], hidden_channels[-4],
                                          kernel_size=3, stride=2, padding=1, output_padding=1)
        self.out = AdaptiveConvBlock(hidden_channels[-4], in_channels, n_noise_scale=n_noise_scale)

    def forward(self, x, noise_scale_idx):
        # Encoder
        h1 = self.res1(x, noise_scale_idx)
        h2 = self.res2(h1, noise_scale_idx)
        h3 = self.res3(h2, noise_scale_idx)
        h4 = self.res4(h3, noise_scale_idx)

        # Refinement
        h = self.refine1(h4, x2=None, noise_scale_idx=noise_scale_idx)
        h = self.refine2(h3, h, noise_scale_idx)
        h = self.refine3(h2, h, noise_scale_idx)
        h = self.refine4(h1, h, noise_scale_idx)

        # Upsample
        h = self.up_norm(h, noise_scale_idx)
        h = self.up_conv(h)
        h = self.out(h, noise_scale_idx)

        return h


# -------------------
# Quick shape test
# -------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RefineNet(in_channels=3, hidden_channels=(64, 128, 256, 512), n_noise_scale=5).to(device)

    x = torch.randn(4, 3, 32, 32, device=device)
    t = torch.randint(0, 5, (4,), device=device)  # n_noise_scale=5

    with torch.no_grad():
        y = model(x, t)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)
    assert y.shape == x.shape, "❌ Shape mismatch!"
    print("✅ Shape test passed")
