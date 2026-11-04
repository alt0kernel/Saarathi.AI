import torch
import torch.nn.functional as F

def caqe_refine_embeddings(z_embeddings: torch.Tensor, boxes: torch.Tensor, radius: float = 150.0) -> torch.Tensor:
    """
    Implements Eq. (28): Context-Aware Query Expansion (CAQE).
    z_tilde_i = (z_i + sum_{n in N(i)} w_in * z_n) / ||z_i + sum_{n in N(i)} w_in * z_n||_2
    """
    centers = torch.stack([(boxes[:, 0] + boxes[:, 2]) * 0.5,
                           (boxes[:, 1] + boxes[:, 3]) * 0.5], dim=-1)
    dist_matrix = torch.cdist(centers, centers, p=2)

    neighbor_mask = (dist_matrix > 0.0) & (dist_matrix <= radius)
    sigma = radius * 0.5
    weights = torch.exp(-(dist_matrix ** 2) / (2.0 * (sigma ** 2))) * neighbor_mask.float()

    aggregated = z_embeddings + torch.matmul(weights, z_embeddings)
    return F.normalize(aggregated, p=2, dim=-1)

def retrieve_sku_from_catalogue(z_tilde: torch.Tensor, catalogue_z: torch.Tensor,
                                priorities: torch.Tensor, tau_sim: float = 0.75):
    """
    Implements Algorithm 2 (Eqs. 29-30):
    s(i, j) = z_tilde_i^T * z_j^ref, j* = argmax_j s(i, j)
    """
    sim_matrix = torch.matmul(z_tilde, catalogue_z.T)  # (N, M)
    best_sims, best_j_indices = sim_matrix.max(dim=-1)

    i_star = int(torch.argmax(priorities).item())
    j_star = int(best_j_indices[i_star].item())
    confidence = float(best_sims[i_star].item())

    if confidence >= tau_sim:
        return j_star, confidence, True
    return None, confidence, False