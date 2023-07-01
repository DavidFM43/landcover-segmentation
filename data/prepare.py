import os
import shutil
import sys
from pathlib import Path

import kaggle
import torch
from dataset import int2rgb
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from utils import rgb2label

sys.path.insert(0, "..")

ds_name = "balraj98/deepglobe-land-cover-classification-dataset"
# download dataset from kaggle
raw_data_path = Path("raw_data")
os.mkdir(raw_data_path)
api = kaggle.api
api.authenticate()
api.dataset_download_cli(ds_name, path=raw_data_path, unzip=True)

images_dir = Path("images")
masks_dir = Path("masks")
os.mkdir(images_dir)
os.mkdir(masks_dir)

ids = [f.split("_")[0] for f in os.listdir(raw_data_path / "train") if f.endswith("sat.jpg")]

for id in tqdm(ids, desc="Preparing dataset"):
    sat_img = f"{id}_sat.jpg"
    mask_file = f"{id}_mask.png"
    # copy sat image to images dir
    shutil.copy(raw_data_path / "train" / sat_img, images_dir / sat_img)
    # convert mask to label encoded mask
    mask = read_image(str(raw_data_path / "train" / mask_file))
    le_mask = rgb2label(mask, int2rgb).type(torch.uint8)
    to_pil_image(le_mask).save(masks_dir / mask_file)  # TODO: Maybe save the images in a better format
