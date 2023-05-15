import os
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


data_dir = Path("data")
images_dir = data_dir / "images"
masks_dir = data_dir / "masks"
classes = pd.read_csv(data_dir / "class_dict.csv")

# refactor this guys
class_rgb_colors = [tuple(row[1:].tolist()) for _, row in classes.iterrows()]
class_names = classes["name"].tolist()
label_to_name = {idx: name for idx, name in enumerate(class_names)}


class LandcoverDataset(Dataset):
    def __init__(
        self,
        images_dir=images_dir,
        masks_dir=masks_dir,
        transform=None,
        target_transform=None,
    ):
        self.image_ids = [f.split("_")[0] for f in os.listdir(images_dir)]
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
            mask_resize = self.target_transform(mask)
        transform = transforms.ToTensor()
        return sat_img, mask_resize.squeeze().long(), transform(mask)
