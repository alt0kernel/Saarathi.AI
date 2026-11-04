import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResNet34HAL(nn.Module):
    """
    Implements Eq. (27): ResNet-34 backbone trained with Hierarchical Auxiliary Loss (HAL)
    alongside CosFace margin loss for fine-grained shelf SKU recognition.
    """
    def __init__(self, embedding_dim: int = 512, num_fine_classes: int = 100, num_macro_classes: int = 10):
        super().__init__()
        backbone = models.resnet34(weights=None)
        in_feats = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.embed_proj = nn.Linear(in_feats, embedding_dim)

        # Hierarchical classification heads for training
        self.fine_head = nn.Linear(embedding_dim, num_fine_classes, bias=False)
        self.macro_head = nn.Linear(embedding_dim, num_macro_classes, bias=False)

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        # Eq. (27): z_i = f_theta(B_i) / ||f_theta(B_i)||_2
        raw_feats = self.embed_proj(self.backbone(crops))
        return F.normalize(raw_feats, p=2, dim=-1)

    def compute_hal_loss(self, z: torch.Tensor, fine_labels: torch.Tensor,
                         macro_labels: torch.Tensor, s: float = 30.0, m: float = 0.2,
                         lambda_hal: float = 0.3) -> torch.Tensor:
        # CosFace margin loss on fine-grained SKUs + hierarchical macro-category auxiliary loss
        w_fine = F.normalize(self.fine_head.weight, p=2, dim=-1)
        cosine = F.linear(z, w_fine)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, fine_labels.view(-1, 1), 1.0)
        logits_fine = s * (cosine - one_hot * m)

        logits_macro = self.macro_head(z)
        loss_fine = F.cross_entropy(logits_fine, fine_labels)
        loss_macro = F.cross_entropy(logits_macro, macro_labels)
        return loss_fine + lambda_hal * loss_macro