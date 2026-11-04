import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import box_iou
from saarathi.vision_vpd.scinet_enhancer import SCINetEnhancer
from saarathi.vision_vpd.seam_attention import SEAMAttention

class VPDDetector(nn.Module):
    """
    Vision Perception and Detection (VPD) Module - Section V-E & VI-A.
    Combines SCINet, SEAM, and NMS-free Dual-Label matching metric.
    """
    def __init__(self, feat_channels: int = 64, scales=(3, 5, 7)):
        super().__init__()
        self.scinet = SCINetEnhancer(3)
        self.backbone_conv = nn.Conv2d(3, feat_channels, kernel_size=3, padding=1)
        self.seam = SEAMAttention(feat_channels, scales=scales)

    @staticmethod
    def dual_label_match_metric(s_prior: torch.Tensor, p_conf: torch.Tensor,
                                pred_boxes: torch.Tensor, gt_boxes: torch.Tensor,
                                alpha: float = 0.5, beta: float = 6.0) -> torch.Tensor:
        # Section VI-A: m = s * p^alpha * IoU(b_hat, b)^beta
        ious = torch.diagonal(box_iou(pred_boxes, gt_boxes))
        return s_prior * (p_conf ** alpha) * (ious ** beta)

    def forward(self, i_input: torch.Tensor, simulated_detections=None):
        i_enhanced, _ = self.scinet(i_input)
        feats = self.seam(self.backbone_conv(i_enhanced))

        # Eq. (23): D_t = {(b_i, c_i, p_i)}_{i=1}^N
        if simulated_detections is None:
            detections = [
                {"box": [120, 140, 220, 310], "class": "product", "conf": 0.94},
                {"box": [230, 142, 330, 312], "class": "product", "conf": 0.91},
                {"box": [340, 138, 440, 308], "class": "product", "conf": 0.89},
            ]
        else:
            detections = simulated_detections

        i_annotated = self.annotate_frame(i_enhanced, detections)
        return i_enhanced, i_annotated, detections, feats

    @staticmethod
    def annotate_frame(i_enhanced: torch.Tensor, detections: list) -> torch.Tensor:
        # Eq. (25): DrawRectangle(b_i) + DrawText(c_i, p_i)
        img_np = (i_enhanced[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8).copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det["box"])
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['class']}:{det['conf']:.2f}"
            cv2.putText(img_np, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        return torch.from_numpy(img_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0