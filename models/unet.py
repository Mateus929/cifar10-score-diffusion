import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def nonlinearity(x):
    return F.silu(x)  # SiLU is the same as Swish


def normalize(x, num_groups=32):
    return F.group_norm(x, num_groups=num_groups)


class Upsample(nn.Module):
    def __init__(self, channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        if self.with_conv:
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels, dropout, conv_shortcut=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.temb_proj = nn.Linear(temb_channels, out_channels)

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # Initialize conv2 with zeros
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

        if in_channels != out_channels:
            if conv_shortcut:
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        # Add timestep embedding
        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.q = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

        # Initialize proj_out with zeros
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x):
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        # Compute attention
        B, C, H, W = q.shape
        q = q.reshape(B, C, H * W).permute(0, 2, 1)  # B, HW, C
        k = k.reshape(B, C, H * W)  # B, C, HW
        v = v.reshape(B, C, H * W).permute(0, 2, 1)  # B, HW, C

        w = torch.bmm(q, k) * (int(C) ** (-0.5))  # B, HW, HW
        w = F.softmax(w, dim=2)

        h = torch.bmm(w, v)  # B, HW, C
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        h = self.proj_out(h)

        return x + h


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


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, temb_ch, num_res_blocks, dropout, attn_resolutions, curr_res, has_downsample,
                 resamp_with_conv):
        super().__init__()
        self.block = nn.ModuleList()
        self.attn = nn.ModuleList()
        self.has_downsample = has_downsample

        block_in = in_ch
        for i_block in range(num_res_blocks):
            self.block.append(ResnetBlock(block_in, out_ch, temb_ch, dropout))
            block_in = out_ch
            if curr_res in attn_resolutions:
                self.attn.append(AttnBlock(block_in))

        if has_downsample:
            self.downsample = Downsample(block_in, resamp_with_conv)

    def forward(self, x, temb):
        hs = []
        for i_block in range(len(self.block)):
            x = self.block[i_block](x, temb)
            if i_block < len(self.attn):
                x = self.attn[i_block](x)
            hs.append(x)

        if self.has_downsample:
            x = self.downsample(x)
            hs.append(x)

        return x, hs


# class UpBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, skip_ch, temb_ch, num_res_blocks, dropout, attn_resolutions, curr_res,
#                  has_upsample, resamp_with_conv):
#         super().__init__()
#         self.block = nn.ModuleList()
#         self.attn = nn.ModuleList()
#         self.has_upsample = has_upsample
#         self.num_res_blocks = num_res_blocks

#         for i_block in range(num_res_blocks + 1):
#             self.block.append(ResnetBlock(in_ch + skip_ch, out_ch, temb_ch, dropout))
#             in_ch = out_ch
#             if curr_res in attn_resolutions:
#                 self.attn.append(AttnBlock(out_ch))

#         if has_upsample:
#             self.upsample = Upsample(out_ch, resamp_with_conv)

#     def forward(self, x, hs, temb):
#         for i_block in range(self.num_res_blocks + 1):
#             x = torch.cat([x, hs.pop()], dim=1)
#             x = self.block[i_block](x, temb)
#             if i_block < len(self.attn):
#                 x = self.attn[i_block](x)

#         if self.has_upsample:
#             x = self.upsample(x)

#         return x

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch, temb_ch, num_res_blocks, dropout, attn_resolutions, curr_res,
                 has_upsample, resamp_with_conv):
        super().__init__()
        self.block = nn.ModuleList()
        self.attn = nn.ModuleList()
        self.has_upsample = has_upsample
        self.num_res_blocks = num_res_blocks
        for i_block in range(num_res_blocks + 1):
            if i_block == 0:
                block_in = in_ch + skip_ch
            else:
                block_in = out_ch + skip_ch  
            self.block.append(ResnetBlock(block_in, out_ch, temb_ch, dropout))
            block_in = out_ch + skip_ch 
            if curr_res in attn_resolutions:
                self.attn.append(AttnBlock(out_ch))
        if has_upsample:
            self.upsample = Upsample(out_ch, resamp_with_conv)
    def forward(self, x, hs, temb):
        for i_block in range(self.num_res_blocks + 1):
            skip = hs.pop()
            x = torch.cat([x, skip], dim=1)
            x = self.block[i_block](x, temb)
            if i_block < len(self.attn):
                x = self.attn[i_block](x)
        if self.has_upsample:
            x = self.upsample(x)
        return x

class DiffusionUNet(nn.Module):
    def __init__(
            self,
            ch=128,
            out_ch=3,
            ch_mult=(1, 2, 2, 4),
            num_res_blocks=2,
            attn_resolutions=(16,),
            dropout=0.0,
            resamp_with_conv=True,
            in_channels=3,
            resolution=32,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.resolution = resolution

        # Timestep embedding
        self.temb_dense0 = nn.Linear(ch, ch * 4)
        self.temb_dense1 = nn.Linear(ch * 4, ch * 4)

        # Downsampling
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            self.down.append(DownBlock(
                block_in, block_out, ch * 4, num_res_blocks, dropout,
                attn_resolutions, curr_res,
                has_downsample=(i_level != self.num_resolutions - 1),
                resamp_with_conv=resamp_with_conv
            ))

            if i_level != self.num_resolutions - 1:
                curr_res = curr_res // 2

        # Middle
        block_in = ch * ch_mult[-1]
        self.mid_block_1 = ResnetBlock(block_in, block_in, ch * 4, dropout)
        self.mid_attn_1 = AttnBlock(block_in)
        self.mid_block_2 = ResnetBlock(block_in, block_in, ch * 4, dropout)

        # Upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block_out = ch * ch_mult[i_level]
            skip_ch = ch * ch_mult[i_level]

            self.up.append(UpBlock(
                block_out, block_out, skip_ch, ch * 4, num_res_blocks, dropout,
                attn_resolutions, curr_res,
                has_upsample=(i_level != 0),
                resamp_with_conv=resamp_with_conv
            ))

            if i_level != 0:
                curr_res = curr_res * 2

        # End
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=ch * ch_mult[0])
        self.conv_out = nn.Conv2d(ch * ch_mult[0], out_ch, kernel_size=3, stride=1, padding=1)

        # Initialize conv_out with zeros
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x, t):
        if t.ndim > 1:
          t = t.view(-1)
        # Timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb_dense0(temb)
        temb = self.temb_dense1(nonlinearity(temb))

        # Downsampling
        h = self.conv_in(x)
        hs = [h]

        for i_level in range(self.num_resolutions):
            h, hs_level = self.down[i_level](h, temb)
            hs.extend(hs_level)

        # Middle
        h = self.mid_block_1(h, temb)
        h = self.mid_attn_1(h)
        h = self.mid_block_2(h, temb)

        # Upsampling
        for i_level in reversed(range(self.num_resolutions)):
            h = self.up[i_level](h, hs, temb)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h
