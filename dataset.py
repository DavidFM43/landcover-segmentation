import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import random

data_dir = Path("data")
images_dir = data_dir / "images"
masks_dir = data_dir / "masks"

train_ids = open(data_dir / "splits" / "train.txt").read().splitlines()
val_ids = open(data_dir / "splits" / "val.txt").read().splitlines()
test_ids = open(data_dir / "splits" / "test.txt").read().splitlines()

int2str = {
    0: "urban_land",
    1: "agriculture_land",
    2: "rangeland",
    3: "forest_land",
    4: "water",
    5: "barren_land",
    6: "unknown",
}
int2rgb = {
    0: (0, 255, 255),
    1: (255, 255, 0),
    2: (255, 0, 255),
    3: (0, 255, 0),
    4: (0, 0, 255),
    5: (255, 255, 255),
    6: (0, 0, 0),
}


class LandcoverDataset(Dataset):
    def __init__(
        self,
        image_ids,
        images_dir=images_dir,
        masks_dir=masks_dir,
        transform=None,
        target_transform=None,
        augmentations=False,
    ):
        self.image_ids = image_ids
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.target_transform = target_transform
        self.augmentations = augmentations

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = self.images_dir / f"{image_id}_sat.jpg"
        mask_path = self.masks_dir / f"{image_id}_mask.png"
        sat_img = Image.open(image_path)
        mask = Image.open(mask_path)

        if self.transform is not None:
            sat_img = self.transform(sat_img)
        if self.target_transform is not None:
            mask = self.target_transform(mask).long()
        if self.augmentations:
            if np.random.random() > 0.5:
                sat_img = transforms.functional.hflip(sat_img)
                mask = transforms.functional.hflip(mask)
            if np.random.random() > 0.5:
                degree = random.choice([90, 180, 270])
                sat_img = transforms.functional.rotate(sat_img, degree)
                mask = transforms.functional.rotate(mask, degree)
        return sat_img, mask.squeeze()
