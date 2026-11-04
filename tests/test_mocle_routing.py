import torch
from saarathi.planner_mocle.mocle_router import MoCLERouter

def test_mocle_gradient_isolation():
    mocle = MoCLERouter(d_model=128, r=8, num_experts=3)
    x = torch.randn(2, 10, 128)
    q = torch.randn(2, 5, 128)
    out, gates, cluster = mocle(x, q)
    loss = out.sum()
    loss.backward()
    assert out.shape == (2, 10, 128)
    assert mocle.u_A.grad is not None