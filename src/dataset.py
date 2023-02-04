import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

import pandas as pd

from pathlib import Path

# When running local
# data_dir = Path("../data")
# masks_dir = Path("../data")

data_dir = Path("/kaggle/input/deepglobe-land-cover-classification-dataset")
masks_dir = Path("/kaggle/input/processed-masks")
satimgs_dir = data_dir / "train"
masks_dir = masks_dir / "train_masks"

annotations_file = pd.read_csv(data_dir / "metadata.csv")
classes = pd.read_csv(data_dir / "class_dict.csv")

image_ids = annotations_file[annotations_file["split"] == "train"]["image_id"].values
# class rgb values
class_colors = [tuple(row[1:].tolist()) for _, row in classes.iterrows()]


class LandcoverDataset(Dataset):
    def __init__(self, image_ids, transform=None, target_transform=None):
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

        # TODO: Probably need to refactor using torchvision transforms
        # scale images
        with torch.no_grad():
            sat_img = sat_img / 255.0

        mask = read_image(str(self.masks_dir / f"{image_id}_mask.png")).long()

        if self.transform is not None:
            sat_img = self.transform(sat_img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return sat_img, mask.squeeze()
