import os
import shutil
import kaggle
from tqdm import tqdm
from pathlib import Path

import torch
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image


def rgb_to_label(mask, class_colors):
    """
    Transforms a mask image from RGB format to label encoding.
        Parameters:
            mask: Torch tensor of shape (3, H, W)
            class_colors: list of tuples of the RGB values for each class
        Returns:
            Torch tensor of shape (H, W) of label enconded classes
    """
    h, w = mask.shape[1:]  # shape expected to be (C, H, W)
    semantic_map = torch.zeros((h, w), dtype=torch.uint8)
    for idx, color in enumerate(class_colors):
        color = torch.tensor(color).view(3, 1, 1)  # rgb value
        class_map = torch.all(torch.eq(mask, color), 0)
        semantic_map[class_map] = idx
    return semantic_map


ds_name = "balraj98/deepglobe-land-cover-classification-dataset"
images_dir = Path("images")
processed_masks_dir = Path("masks")
raw_masks_dir = Path("raw_masks")
class_colors = [
    (0, 255, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 255),
    (0, 0, 0),
]

# download dataset from kaggle
data_path = os.path.dirname(os.path.abspath(__file__))
api = kaggle.api
api.authenticate()
api.dataset_download_cli(ds_name, path=data_path, unzip=True)

# clean folders
shutil.rmtree("valid")
shutil.rmtree("test")
shutil.rmtree("metadata.csv")
os.rename("train", "images")
os.mkdir(processed_masks_dir)
os.mkdir(raw_masks_dir)


mask_files = [f for f in os.listdir(images_dir) if f.endswith(".png")]
# transform all the masks and save them in out_dir
for mask_file in tqdm(
    mask_files,
    desc="Processing masks",
    unit="img",
):
    img_mask = read_image(str(images_dir / mask_file))
    # process mask
    label_enc_mask = rgb_to_label(img_mask, class_colors)
    # move raw mask
    os.rename(images_dir / mask_file, raw_masks_dir / mask_file)
    # save processed mask
    # TODO: Maybe save the images in a better format
    to_pil_image(label_enc_mask).save(processed_masks_dir / mask_file)
