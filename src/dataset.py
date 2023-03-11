import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

import pandas as pd

from sklearn.model_selection import train_test_split

from pathlib import Path

# When running local
# data_dir = Path("../data/raw")
# masks_dir = Path("../data/interm")

data_dir = Path("/kaggle/input/deepglobe-land-cover-classification-dataset")
masks_dir = Path("/kaggle/input/processed-masks")
satimgs_dir = data_dir / "train"
masks_dir = masks_dir / "train_masks"
annotations_file = pd.read_csv(data_dir / "metadata.csv")
classes = pd.read_csv(data_dir / "class_dict.csv")

# split into train and test sets
image_ids = annotations_file[annotations_file["split"] == "train"]["image_id"].values
train_ids, test_ids = train_test_split(
    image_ids, train_size=0.8, shuffle=True, random_state=42
)

# class rgb values
class_colors = [tuple(row[1:].tolist()) for _, row in classes.iterrows()]
class_names = classes["name"].tolist()
class_labels = {idx: name for idx, name in enumerate(class_names)}


class LandcoverDataset(Dataset):
    def __init__(self, train, transform=None, target_transform=None):
        self.image_ids = train_ids if train else test_ids
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        sat_img = read_image(str(satimgs_dir / f"{image_id}_sat.jpg")).float()

        # TODO: Probably need to refactor using torchvision transforms
        # TODO: Normalize with the mean and std of the dataset
        with torch.no_grad():
            sat_img = sat_img / 255.0  # scale images

        mask = read_image(str(masks_dir / f"{image_id}_mask.png")).long()

        if self.transform is not None:
            sat_img = self.transform(sat_img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return sat_img, mask.squeeze()
