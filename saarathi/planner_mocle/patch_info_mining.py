import torch
import torch.nn as nn

class PatchInformationMining(nn.Module):
    """
    Implements Eq. (33): Cross-attentive Patch Information Mining between
    low-resolution queries Q (Y_Low) and M^2 high-resolution sub-region keys/values (Y_High).
    """
    def __init__(self, d_model: int = 512):
        super().__init__()
        self.phi_q = nn.Linear(d_model, d_model)
        self.phi_k = nn.Linear(d_model, d_model)
        self.phi_v = nn.Linear(d_model, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, y_low: torch.Tensor, y_high_sub: torch.Tensor) -> torch.Tensor:
        # y_low: (B, N, D), y_high_sub: (B, N, M^2, D)
        Q = self.phi_q(y_low).unsqueeze(2)       # (B, N, 1, D)
        K = self.phi_k(y_high_sub)               # (B, N, M^2, D)
        V = self.phi_v(y_high_sub)               # (B, N, M^2, D)

        scale = Q.size(-1) ** 0.5
        attn = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) / scale, dim=-1)
        context = torch.matmul(attn, V).squeeze(2) # (B, N, D)
        return self.mlp(y_low + context)