import math

import torch.fft
import torch
import torch.nn as nn
import einops
from timm.models.layers import trunc_normal_, DropPath

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SpectralGatingNetwork(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        # self.w = w
        # self.h = h
        self.dim = dim

    def forward(self, x, spatial_size=None):
        # B, C, H, W = x.shape
        # N = H * W
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        # x = x.permute(0, 2, 3, 1).reshape(B, -1, C)

        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        complex_weight = nn.Parameter(torch.randn(H, W//2+1, self.dim, 2, dtype=torch.float32) * 0.02).cuda()
        weight = torch.view_as_complex(complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
        x = x.reshape(B, N, C)
        # x = x.reshape(B, C, H, W)
        # if self.dim != self.out_dim:
        #     x = self.conv_1x1(x)

        return x

class EfficientAdditiveAttnetion(nn.Module):
    """
    Efficient Additive Attention module for SwiftFormer.
    Input: tensor in shape [B, N, D]
    Output: tensor in shape [B, N, D]
    """

    def __init__(self, in_dims=512, token_dim=256, num_heads=2):
        super().__init__()

        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)

        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor = token_dim ** -0.5
        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.final = nn.Linear(token_dim * num_heads, token_dim)

    def forward(self, x):
        query = self.to_query(x)
        key = self.to_key(x)

        query = torch.nn.functional.normalize(query, dim=-1)  # BxNxD
        key = torch.nn.functional.normalize(key, dim=-1)  # BxNxD

        query_weight = query @ self.w_g  # BxNx1 (BxNxD @ Dx1)
        A = query_weight * self.scale_factor  # BxNx1

        A = torch.nn.functional.normalize(A, dim=1)  # BxNx1

        G = torch.sum(A * query, dim=1)  # BxD

        G = einops.repeat(
            G, "b d -> b repeat d", repeat=key.shape[1]
        )  # BxNxD

        out = self.Proj(G * key) + query  # BxNxD

        out = self.final(out)  # BxNxD

        return out

class SpectrumTransformer(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.norm1 = norm_layer(in_planes)
        self.filter = SpectralGatingNetwork(in_planes)
        self.attn = EfficientAdditiveAttnetion(in_dims=in_planes, token_dim=in_planes, num_heads=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(in_planes)
        mlp_hidden_dim = int(in_planes * mlp_ratio)
        self.mlp = Mlp(in_features=in_planes, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.conv1x1 = nn.Conv2d(in_planes, out_planes, 1, stride=stride)

    def forward(self, x):
        B, C, H, W = x.shape
        # N = H * W
        x = x.permute(0, 2, 3, 1).reshape(B, -1, C)
        # H, W = math.sqrt(N)
        # x = x.permute(0, 2, 3, 1)
        # x = x + self.drop_path(self.mlp(self.norm2(self.attn(self.norm1(x)))))
        x = x + self.filter(x)
        x = x + self.attn(self.norm1(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x.reshape(B, C, H, W)
        if self.in_planes != self.out_planes:
            x = self.conv1x1(x)

        return x


# 输入 B, N, C,  输出 B, N, C
if __name__ == '__main__':
    block = SpectralGatingNetwork(8, 24, 2).cuda()
    input = torch.rand(1, 8, 64, 64).cuda()
    output = block(input)
    print(input.size(), output.size())
