from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from torchvision.io import read_image
import torch


class LandcoverDataset(Dataset):
    def __init__(
        self, satimgs_dir, masks_dir, image_ids, transform=None, target_transform=None
    ):
        self.satimgs_dir = satimgs_dir
        self.masks_dir = masks_dir
        self.image_ids = image_ids
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        sat_img = read_image(str(self.satimgs_dir / f"{image_id}_sat.jpg")).float()

        with torch.no_grad():
            sat_img = sat_img / 255.0

        mask = read_image(str(self.masks_dir / f"{image_id}_mask.png")).long()

        if self.transform is not None:
            sat_img = self.transform(sat_img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return sat_img, mask.squeeze()
