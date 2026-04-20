# models/mini_raft_stereo.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================
#  Basic building blocks
# ============================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = ConvBlock(ch, ch)
        self.conv2 = ConvBlock(ch, ch)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


# ============================
#  Feature & context encoders
# ============================

class FeatureEncoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=32):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, base_ch, 3, 2, 1)
        self.conv2 = ConvBlock(base_ch, base_ch * 2, 3, 2, 1)
        self.res1 = ResidualBlock(base_ch * 2)
        self.res2 = ResidualBlock(base_ch * 2)
        self.out_conv = ConvBlock(base_ch * 2, 128, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x)
        x = self.res2(x)
        return self.out_conv(x)


class ContextEncoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=32):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, base_ch, 3, 2, 1)
        self.conv2 = ConvBlock(base_ch, base_ch * 2, 3, 2, 1)
        self.res1 = ResidualBlock(base_ch * 2)
        self.res2 = ResidualBlock(base_ch * 2)

        self.hidden_conv = nn.Conv2d(base_ch * 2, 96, 3, 1, 1)
        self.context_conv = nn.Conv2d(base_ch * 2, 64, 3, 1, 1)

    def forward(self, x):
        x = self.res2(self.res1(self.conv2(self.conv1(x))))
        hidden = torch.tanh(self.hidden_conv(x))
        context = F.relu(self.context_conv(x))
        return hidden, context


# ============================
#  Correlation pyramid
# ============================

def build_corr_volume(f1, f2, max_disp=4):
    B, C, H, W = f1.shape
    vols = []
    for d in range(-max_disp, max_disp + 1):
        if d < 0:
            shifted = F.pad(f2[:, :, :, :W + d], (-d, 0, 0, 0))
        elif d > 0:
            shifted = F.pad(f2[:, :, :, d:], (0, d, 0, 0))
        else:
            shifted = f2
        vols.append((f1 * shifted).mean(1, keepdim=True))
    return torch.cat(vols, dim=1)


class CorrPyramid(nn.Module):
    def __init__(self, max_disp=4, num_levels=3):
        super().__init__()
        self.max_disp = max_disp
        self.num_levels = num_levels

    def forward(self, f1, f2):
        pyr = []
        for _ in range(self.num_levels):
            pyr.append(build_corr_volume(f1, f2, self.max_disp))
            f1 = F.avg_pool2d(f1, 2, 2)
            f2 = F.avg_pool2d(f2, 2, 2)
        return pyr

    def sample(self, pyr, coords):
        B, _, H, W = coords.shape
        up = [F.interpolate(c, (H, W), mode="bilinear", align_corners=False) for c in pyr]
        return torch.cat(up, dim=1)


# ============================
#  ConvGRU update block
# ============================

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super().__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, 1, 1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, 1, 1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, 1, 1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        return (1 - z) * h + z * q


class UpdateBlock(nn.Module):
    def __init__(self, hidden_dim=96, corr_dim=27, context_dim=64):
        super().__init__()
        input_dim = corr_dim + context_dim + 1
        self.gru = ConvGRU(hidden_dim, input_dim)
        self.delta_head = nn.Conv2d(hidden_dim, 1, 3, 1, 1)

    def forward(self, h, context, corr, disp):
        x = torch.cat([context, corr, disp], dim=1)
        h = self.gru(h, x)
        return h, self.delta_head(h)


# ============================
#  Mini RAFT Stereo
# ============================

class MiniRAFTStereo(nn.Module):
    def __init__(self, iters=8, max_disp=4, num_corr_levels=3, feature_base_ch=32):
        super().__init__()
        self.iters = iters

        self.fnet = FeatureEncoder(3, feature_base_ch)
        self.cnet = ContextEncoder(3, feature_base_ch)
        self.corr_block = CorrPyramid(max_disp, num_corr_levels)

        corr_dim = (2 * max_disp + 1) * num_corr_levels
        self.update_block = UpdateBlock(96, corr_dim, 64)

    @staticmethod
    def initialize_disp(fmap):
        B, C, H, W = fmap.shape
        return torch.zeros(B, 1, H, W, device=fmap.device)

    def forward(self, left, right, iters=None):
        if iters is None:
            iters = self.iters

        f1 = self.fnet(left)
        f2 = self.fnet(right)

        h, context = self.cnet(left)
        corr_pyr = self.corr_block(f1, f2)
        disp = self.initialize_disp(f1)

        B, _, H, W = f1.shape
        y, x = torch.meshgrid(torch.arange(H, device=f1.device),
                              torch.arange(W, device=f1.device),
                              indexing="ij")
        coords = torch.stack((x, y), 0)[None].repeat(B, 1, 1, 1)

        preds = []
        for _ in range(iters):
            corr = self.corr_block.sample(corr_pyr, coords)
            h, delta = self.update_block(h, context, corr, disp)
            disp = disp + delta
            preds.append(disp)

        return preds
