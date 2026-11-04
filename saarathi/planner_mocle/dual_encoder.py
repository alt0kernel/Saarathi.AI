import torch
import torch.nn as nn
import torch.nn.functional as F

class DualBranchVisualEncoder(nn.Module):
    """
    Implements Section V-G (Eqs. 31-32 & 38):
    Branch 1: Low-Resolution ViT (224x224) -> Y_Low in R^{(N+1) x D}
    Branch 2: High-Resolution 4-stage Swin-T (1120x1120) -> Y_High sub-regions
    """
    def __init__(self, d_model: int = 512, patch_size: int = 16, num_patches: int = 196, m_sub: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_patches = num_patches
        self.m_sq = m_sub * m_sub

        # Low-Res ViT Patch Embedding (224x224 with 16x16 patches -> N=196)
        self.vit_patch_proj = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d_model) * 0.02)
        self.vit_block = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True)

        # High-Res Swin-style hierarchical feature extractor
        self.swin_stages = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=4),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(256, d_model, kernel_size=2, stride=2),
        )

    def forward(self, i_annotated: torch.Tensor):
        B = i_annotated.size(0)
        # Eq. (31): Downsample to 224x224 for ViT
        i_low = F.interpolate(i_annotated, size=(224, 224), mode="bilinear", align_corners=False)
        x_p = self.vit_patch_proj(i_low).flatten(2).transpose(1, 2)  # (B, 196, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        y_low = torch.cat([cls_tokens, x_p], dim=1) + self.pos_embed
        y_low = self.vit_block(y_low)

        # Eq. (32): Upsample/resize to 1120x1120 for 4-stage high-res branch
        i_high = F.interpolate(i_annotated, size=(1120, 1120), mode="bilinear", align_corners=False)
        y_high_map = self.swin_stages(i_high)  # (B, D, 35, 35)
        # Partition into M^2 sub-regions aligned with the N=196 query patches
        y_high_sub = F.adaptive_avg_pool2d(y_high_map, (14 * 4, 14 * 4))
        y_high_sub = y_high_sub.unfold(2, 4, 4).unfold(3, 4, 4) # (B, D, 14, 14, 4, 4)
        y_high_sub = y_high_sub.permute(0, 2, 3, 4, 5, 1).reshape(B, self.num_patches, self.m_sq, self.d_model)

        return y_low[:, 1:, :], y_high_sub