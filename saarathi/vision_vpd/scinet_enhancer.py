import torch
import torch.nn as nn

class SCINetEnhancer(nn.Module):
    """
    Self-Calibrating Illumination Network (SCINet) - Eq. (21).
    Computes enhanced frame I_enhanced = I_input + Delta_I.
    """
    def __init__(self, channels: int = 3):
        super().__init__()
        self.SharedStage = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, i_input: torch.Tensor):
        delta_i = self.SharedStage(i_input)
        i_enhanced = torch.clamp(i_input + delta_i, 0.0, 1.0)
        return i_enhanced, delta_i

    @staticmethod
    def unsupervised_loss(y_pred: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        # Eq. (21): L_unsupervised = (1/N) * sum(||y_i - y_hat_i||^2)
        return torch.mean((y_target - y_pred) ** 2)