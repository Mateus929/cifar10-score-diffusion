import torch
import torch.nn as nn

# ---------------------------------
# Conditional Instance Norm
# ---------------------------------
class CondInstanceNorm(nn.Module):
    def __init__(self, channels, n_noise_scale=10, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(n_noise_scale, channels))
        self.beta = nn.Parameter(torch.zeros(n_noise_scale, channels))
        self.eps = eps

    def forward(self, x, noise_idx):
        # x: (B, C, H, W), noise_idx: (B,)
        b, c, h, w = x.shape
        gamma = self.gamma[noise_idx].view(b, c, 1, 1)
        beta = self.beta[noise_idx].view(b, c, 1, 1)

        mu = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return gamma * x + beta

# ---------------------------------
# Simple Residual Block
# ---------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels, n_noise_scale=10):
        super().__init__()
        self.norm1 = CondInstanceNorm(channels, n_noise_scale)
        self.norm2 = CondInstanceNorm(channels, n_noise_scale)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, noise_idx):
        h = self.norm1(x, noise_idx)
        h = self.act(h)
        h = self.conv1(h)

        h = self.norm2(h, noise_idx)
        h = self.act(h)
        h = self.conv2(h)

        return x + h

# ---------------------------------
# Simple Encoder-Decoder Network
# ---------------------------------
class RefineNet(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=(64, 128, 256), n_noise_scale=10):
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv2d(in_channels, hidden_channels[0], kernel_size=3, padding=1, stride=2)
        self.enc2 = nn.Conv2d(hidden_channels[0], hidden_channels[1], kernel_size=3, padding=1, stride=2)
        self.enc3 = nn.Conv2d(hidden_channels[1], hidden_channels[2], kernel_size=3, padding=1, stride=2)

        # Bottleneck
        self.res1 = ResidualBlock(hidden_channels[2], n_noise_scale)
        self.res2 = ResidualBlock(hidden_channels[2], n_noise_scale)

        # Decoder
        self.dec3 = nn.ConvTranspose2d(hidden_channels[2], hidden_channels[1], kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(hidden_channels[1], hidden_channels[0], kernel_size=4, stride=2, padding=1)
        self.dec1 = nn.ConvTranspose2d(hidden_channels[0], in_channels, kernel_size=4, stride=2, padding=1)

        self.norm_enc = nn.ModuleList([CondInstanceNorm(c, n_noise_scale) for c in hidden_channels])
        self.norm_bottleneck = nn.ModuleList([CondInstanceNorm(c, n_noise_scale) for c in hidden_channels])
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, noise_idx):
        # Encoder
        h1 = self.act(self.norm_enc[0](self.enc1(x), noise_idx))
        h2 = self.act(self.norm_enc[1](self.enc2(h1), noise_idx))
        h3 = self.act(self.norm_enc[2](self.enc3(h2), noise_idx))

        # Bottleneck
        h = self.res1(h3, noise_idx)
        h = self.res2(h, noise_idx)

        # Decoder
        h = self.act(self.dec3(h) + h2)  # skip connection
        h = self.act(self.dec2(h) + h1)  # skip connection
        h = self.dec1(h)

        return h


# ---------------------------------
# Quick shape test
# ---------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RefineNet(in_channels=3, hidden_channels=(64, 128, 256), n_noise_scale=5).to(device)

    x = torch.randn(4, 3, 32, 32, device=device)
    t = torch.randint(0, 5, (4,), device=device)  # n_noise_scale=5

    with torch.no_grad():
        y = model(x, t)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)
    assert y.shape == x.shape, "❌ Shape mismatch!"
    print("✅ Shape test passed")
