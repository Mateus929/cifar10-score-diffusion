import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def nonlinearity(x):
    return F.silu(x)  # SiLU is the same as Swish

def get_timestep_embedding(timesteps, embedding_dim):
    """
    Build sinusoidal embeddings (from Fairseq).
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


class ScoreNet(nn.Module):
    def __init__(self, ch=128):
        super(ScoreNet, self).__init__()
        self.ch = ch

        self.temb_dense0 = nn.Linear(ch, ch * 4)
        self.temb_dense1 = nn.Linear(ch * 4, ch * 4)
        self.temb_proj = nn.Linear(ch * 4, ch)

        self.conv1 = nn.Conv2d(3, ch, 3, padding=1)

        # Dilated blocks (Receptive field grows without shrinking image size)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(ch, ch, 3, padding=4, dilation=4)
        self.conv4 = nn.Conv2d(ch, ch, 3, padding=1, dilation=1)

        self.conv_out = nn.Conv2d(ch, 3, 3, padding=1)

        self.gn1 = nn.GroupNorm(8, ch)
        self.gn2 = nn.GroupNorm(8, ch)
        self.gn3 = nn.GroupNorm(8, ch)
        self.gn4 = nn.GroupNorm(8, ch)

    def forward(self, x, t):
        if t.ndim > 1:
            t = t.view(-1)

        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb_dense0(temb)
        temb = self.temb_dense1(nonlinearity(temb))

        temb = self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h1 = F.silu(self.gn1(self.conv1(x)))

        h2 = self.conv2(h1)
        h2 = F.silu(self.gn2(h2 + temb))
        h2 = h2 + h1

        h3 = self.conv3(h2)
        h3 = F.silu(self.gn3(h3))
        h3 = h3 + h2

        h4 = self.conv4(h3)
        h4 = F.silu(self.gn4(h4))
        h4 = h4 + h3

        return self.conv_out(h4)