import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

data_dir = Path("data")
images_dir = data_dir / "images"
masks_dir = data_dir / "masks"

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
        images_dir=images_dir,
        masks_dir=masks_dir,
        transform=None,
        target_transform=None,
    ):
        self.image_ids = sorted([f.split("_")[0] for f in os.listdir(images_dir)])
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.target_transform = target_transform

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
            mask = self.target_transform(mask).squeeze().long()
        return sat_img, mask
