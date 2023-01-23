from torch.utils.data import DataLoader, Dataset
import pandas as pd
from pathlib import Path
from torchvision.io import read_image

data_dir = Path("data")
annotations = pd.read_csv(data_dir / "metadata.csv")
image_ids = annotations[annotations["split"] == "train"]["image_id"].values


class LandcoverDataset(Dataset):
    def __init__(
        self, satimgs_dir, masks_dir, image_ids, transform=None, augmentations=None
    ):
        self.satimgs_dir = satimgs_dir
        self.masks_dir = masks_dir
        self.image_ids = image_ids
        self.transform = transform
        self.augmentations = augmentations

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = image_ids[idx]
        # probably need to rescale the input values
        sat_img = read_image(self.satimgs_dir / f"{image_id}_sat.jpg").float()
        mask = read_image(self.masks_dir / f"{image_id}_mask.png").long()

        if self.transform is not None:
            sat_img = self.transform(sat_img)
        if self.augmentations is not None:
            mask = self.augmentations(mask)

        return sat_img, mask.squeeze()
