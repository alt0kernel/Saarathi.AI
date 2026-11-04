import torch
import torch.nn.functional as F
from saarathi.retrieval_sku.caqe_expansion import caqe_refine_embeddings, retrieve_sku_from_catalogue

def test_caqe_and_matching():
    z = F.normalize(torch.randn(4, 512), p=2, dim=-1)
    boxes = torch.tensor([[10, 10, 50, 50], [55, 10, 95, 50], [100, 10, 140, 50], [400, 400, 450, 450]], dtype=torch.float32)
    z_tilde = caqe_refine_embeddings(z, boxes, radius=100.0)
    assert torch.allclose(z_tilde.norm(p=2, dim=-1), torch.ones(4), atol=1e-5)

    priorities = torch.tensor([0.95, 0.6, 0.4, 0.2])
    j_star, conf, ok = retrieve_sku_from_catalogue(z_tilde, z_tilde, priorities, tau_sim=0.7)
    assert ok and j_star == 0 and conf > 0.99