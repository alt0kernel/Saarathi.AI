import torch
import torch.nn as nn

class CSMMBlock(nn.Module):
    """
    Channel and Spatial Mixing Module (CSMM) - Eq. (37).
    U_k^(1) = BN(GELU(PatchEmbed_k(F_in)))
    U_k^(2) = U_k^(1) + BN(GELU(DConv_kxk(U_k^(1))))
    F_k     = BN(GELU(PConv_1x1(U_k^(2))))
    """
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        pad = kernel_size // 2
        self.patch_embed = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=pad, groups=channels),
            nn.GELU(),
            nn.BatchNorm2d(channels)
        )
        self.dconv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=pad, groups=channels),
            nn.GELU(),
            nn.BatchNorm2d(channels)
        )
        self.pconv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(channels)
        )

    def forward(self, f_in: torch.Tensor) -> torch.Tensor:
        u1 = self.patch_embed(f_in)
        u2 = u1 + self.dconv(u1)
        f_k = self.pconv(u2)
        return f_k

class SEAMAttention(nn.Module):
    """
    Separated Enhanced Attention Mechanism (SEAM) - Eq. (22).
    F_pooled = (1/K) * sum_{k=1}^K F_k
    """
    def __init__(self, channels: int, scales=(3, 5, 7)):
        super().__init__()
        self.scales = nn.ModuleList([CSMMBlock(channels, k) for k in scales])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_pooled = torch.stack([branch(x) for branch in self.scales], dim=0).mean(dim=0)
        return x * torch.sigmoid(f_pooled)