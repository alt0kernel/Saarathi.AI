import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

class SKU110KDataset(Dataset):
    """
    SKU-110K Dataset Loader (Section VII-A).
    8 CSV fields: image_name, x1, y1, x2, y2, class_id, img_width, img_height
    """
    COLS = ["image_name", "x1", "y1", "x2", "y2", "class_id", "width", "height"]

    def __init__(self, root_dir: str, split: str = "train", target_size=(640, 640)):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "images")
        self.target_size = target_size
        csv_path = os.path.join(root_dir, "annotations", f"annotations_{split}.csv")

        if os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path, names=self.COLS)
            self.groups = list(self.df.groupby("image_name"))
        else:
            self.groups = []

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        img_name, group = self.groups[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        image_resized = TF.resize(image, self.target_size)
        img_tensor = TF.to_tensor(image_resized)

        scale_x = self.target_size[1] / orig_w
        scale_y = self.target_size[0] / orig_h

        boxes = torch.tensor(group[["x1", "y1", "x2", "y2"]].values, dtype=torch.float32)
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        labels = torch.zeros((len(boxes),), dtype=torch.long) # Single class 'object'

        return img_tensor, {"boxes": boxes, "labels": labels, "image_name": img_name}