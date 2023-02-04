import torch
import torchvision
import pandas as pd
import wandb
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from dataset import LandcoverDataset
from model import Unet
from utils import mask_to_img
from torchvision.transforms.functional import to_pil_image, resize
from torchvision.utils import make_grid

torch.manual_seed(1)
IN_KAGGLE = True
LOGGING = True

if LOGGING:
    wandb.login(key="5f5a6e6618ddafd57c6c7b40a8313449bfd7a04e")

if IN_KAGGLE:
    data_dir = Path("/kaggle/input/deepglobe-land-cover-classification-dataset")
    masks_dir = Path("/kaggle/input/processed-masks")
else:
    data_dir = Path("data")
    masks_dir = Path("data")

# TODO: refactor this part
# paths and device
train_dir = data_dir / "train"
transformed_masks_path = masks_dir / "train_masks"
annotations = pd.read_csv(data_dir / "metadata.csv")
classes = pd.read_csv(data_dir / "class_dict.csv")
image_ids = annotations[annotations["split"] == "train"]["image_id"].values
device = "cuda" if torch.cuda.is_available() else "cpu"
class_to_rgb = {}
for idx, row in classes.iterrows():
    class_to_rgb[row[0]] = row[1:].to_list()
class_colors = [tuple(x) for x in class_to_rgb.values()]

# split in train and test sets
train_ids, test_ids = train_test_split(
    image_ids, train_size=0.8, shuffle=True, random_state=42
)

# training params
BATCH_SIZE = 4
EPOCHS = 500
MODEL = Unet().to(device)
OPTIMIZER = torch.optim.Adam(MODEL.parameters())
LOSS_FN = nn.CrossEntropyLoss()
RESIZE_RES = 512

# datasets
train_dataset = LandcoverDataset(
    train_dir,
    transformed_masks_path,
    train_ids,
    transform=torchvision.transforms.Resize(RESIZE_RES),
    target_transform=torchvision.transforms.Resize(RESIZE_RES),
)
test_dataset = LandcoverDataset(
    train_dir,
    transformed_masks_path,
    test_ids,
    transform=torchvision.transforms.Resize(RESIZE_RES),
    target_transform=torchvision.transforms.Resize(RESIZE_RES),
)

# create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# log training and data config
if LOGGING:
    wandb.init(
        project="landcover-segmentation",
        config={
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "resize_res": RESIZE_RES,
            "optimizer": type(OPTIMIZER).__name__,
            "loss_fn": type(LOSS_FN).__name__,
        },
    )


# training loop
X, y = next(iter(train_dataloader))
X, y = X.to(device), y.to(device)
MODEL.train()

if LOGGING:
    wandb.watch(MODEL, log_freq=10)

for epoch in range(EPOCHS):
    # for X, y in train_dataloader:
    #    X, y = X.to(device), y.to(device)
    # forward pass
    pred = MODEL(X)
    loss = LOSS_FN(pred, y)
    # backward pass
    OPTIMIZER.zero_grad()
    loss.backward()
    OPTIMIZER.step()

    if epoch % 20 == 0:
        print(f"loss: {loss.detach().item():>7f}  [{epoch:>5d}/{EPOCHS:>5d}]")

    if LOGGING:
        metrics = {"train/train_loss": loss.detach().item()}
        wandb.log(metrics, step=epoch)

# predictions
with torch.no_grad():
    preds = MODEL(X)
dis_preds = torch.argmax(preds, 1)

if LOGGING:
    for i in range(BATCH_SIZE):
        image = (X[i] * 255).type(torch.uint8).to("cpu")
        target = mask_to_img(y[i].to("cpu"), class_colors)
        pred = mask_to_img(dis_preds[i].to("cpu"), class_colors)
        imgs = make_grid([image, target, pred])

        wandb.log(
            {f"image_{i}": wandb.Image(to_pil_image(imgs), caption="sat/gt/pred")}
        )

    wandb.finish()
