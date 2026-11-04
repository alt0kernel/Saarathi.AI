import torch
import torch.nn as nn
import torch.nn.functional as F

class MoCLERouter(nn.Module):
    """
    Implements Section V-G (Eq. 34 & 39):
    k=3 Task Clusters (S1: Chat, S2: Navigation, S3: Object Search) + Universal Expert W_u.
    T_output = sum_{e=1}^E G_e * W_e * x_i + W_0 * x_i
    T_final  = T_output + (1 - G_max) * W_u * x_i
    """
    def __init__(self, d_model: int = 512, r: int = 16, num_experts: int = 3, tau: float = 1.0):
        super().__init__()
        self.num_experts = num_experts
        self.tau = tau
        self.w0 = nn.Linear(d_model, d_model, bias=False)

        self.experts_A = nn.ParameterList([nn.Parameter(torch.randn(d_model, r) * 0.02) for _ in range(num_experts)])
        self.experts_B = nn.ParameterList([nn.Parameter(torch.zeros(r, d_model)) for _ in range(num_experts)])

        self.u_A = nn.Parameter(torch.randn(d_model, r) * 0.02)
        self.u_B = nn.Parameter(torch.zeros(r, d_model))

        self.centroids = nn.Parameter(torch.randn(num_experts, d_model), requires_grad=False)
        self.w_gate = nn.Linear(d_model, num_experts)

    def assign_cluster(self, query_embed: torch.Tensor) -> int:
        # k-means nearest centroid assignment: min ||e_i - c_j||^2
        dists = torch.cdist(query_embed.mean(dim=1), self.centroids)
        return int(torch.argmin(dists, dim=-1).item())

    def forward(self, x: torch.Tensor, query_embed: torch.Tensor):
        c_xi = query_embed.mean(dim=1) # (B, D)
        eps = torch.randn_like(self.w_gate(c_xi)) * 0.01 if self.training else 0.0
        gates = F.softmax((self.w_gate(c_xi) + eps) / self.tau, dim=-1) # (B, E)
        g_max, top_cluster = gates.max(dim=-1, keepdim=True)            # (B, 1)

        t_output = self.w0(x)
        for e in range(self.num_experts):
            w_e_x = torch.matmul(torch.matmul(x, self.experts_A[e]), self.experts_B[e])
            t_output = t_output + gates[:, e].view(-1, 1, 1) * w_e_x

        w_u_x = torch.matmul(torch.matmul(x, self.u_A), self.u_B)
        t_final = t_output + (1.0 - g_max.unsqueeze(-1)) * w_u_x
        return t_final, gates, int(top_cluster[0].item())