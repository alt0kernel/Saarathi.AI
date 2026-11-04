import torch
from saarathi.vision_vpd.yolo_dual_label import VPDDetector

def test_vpd_forward_and_shapes():
    vpd = VPDDetector()
    img = torch.rand(1, 3, 320, 320)
    i_enh, i_ann, dets, feats = vpd(img)
    assert i_enh.shape == img.shape
    assert i_ann.shape == img.shape
    assert len(dets) > 0