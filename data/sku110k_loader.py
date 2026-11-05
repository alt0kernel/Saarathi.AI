import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import torchvision.transforms.functional as TF

class SKU110KDataset(Dataset):
    """
    SKU-110K Dataset Loader using Hugging Face ('finedet/sku110k').
    Downloads and caches dataset splits automatically.
    """
    def __init__(self, split: str = "train", target_size=(640, 640)):
        self.target_size = target_size
        
        # Map split names to Hugging Face split keys
        hf_split_map = {
            "train": "train",
            "val": "validation",
            "validation": "validation",
            "test": "test"
        }
        hf_split = hf_split_map.get(split, "train")
        
        # Load dataset directly from Hugging Face Hub
        print(f"Loading SKU-110K '{hf_split}' split from Hugging Face...")
        self.dataset = load_dataset("finedet/sku110k", split=hf_split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        orig_w, orig_h = image.size

        # Resize image to target tensor dimensions
        image_resized = TF.resize(image, self.target_size)
        img_tensor = TF.to_tensor(image_resized)

        scale_x = self.target_size[1] / orig_w
        scale_y = self.target_size[0] / orig_h

        # 'finedet/sku110k' stores bounding boxes in absolute COCO format [x, y, w, h]
        raw_boxes = item["objects"]["bbox"]
        
        boxes = []
        for box in raw_boxes:
            x, y, w, h = box
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
            # Scale coordinates to the resized tensor dimensions
            boxes.append([x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y])

        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)

        labels = torch.zeros((len(boxes_tensor),), dtype=torch.long) # Single 'object' class

        meta = {
            "boxes": boxes_tensor,
            "labels": labels,
            "image_idx": idx
        }

        return img_tensor, meta