import torch
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from dataset import LandcoverDataset 
from model import Unet


torch.manual_seed(1)
IN_KAGGLE = True
LOGGING = False


if LOGGING:
    wandb.login(key="5f5a6e6618ddafd57c6c7b40a8313449bfd7a04e")

if IN_KAGGLE:
    data_dir = Path("/kaggle/input/deepglobe-land-cover-classification-dataset")
    masks_dir = Path("/kaggle/input/processed-masks")
else:
    data_dir = Path("data")
    masks_dir = Path("data")
    
print("READING PATHS")
train_dir = data_dir / "train"
transformed_masks_path = masks_dir / "train_masks"
annotations = pd.read_csv(data_dir / "metadata.csv")
image_ids = annotations[annotations["split"] == "train"]["image_id"].values
device = "cuda" if torch.cuda.is_available() else "cpu"

# split in train and test sets
train_ids, test_ids = train_test_split(
    image_ids, train_size=0.8, shuffle=True, random_state=42
)

print("CREATING DATASETS")
# create datasets
train_dataset = LandcoverDataset(
    train_dir,
    transformed_masks_path,
    train_ids,
    transform=torchvision.transforms.Resize(512),
    augmentations=torchvision.transforms.Resize(324),
)
test_dataset = LandcoverDataset(
    train_dir,
    transformed_masks_path,
    test_ids,
    transform=torchvision.transforms.Resize(512),
    augmentations=torchvision.transforms.Resize(324),
)


print("CREATING PARAMS")
# training params
BATCH_SIZE = 2
LR = 0.001
EPOCHS = 100
MODEL = Unet().to(device)
OPTIMIZER = torch.optim.SGD(MODEL.parameters(), lr=LR)
LOSS_FN = nn.CrossEntropyLoss()

# create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("LOADING BATCH")
X, y = next(iter(train_dataloader))
X, y = X.to(device), y.to(device)
lossi = []

if LOGGING:
    wandb.init(
            project="landcover-segmentation",
            config={
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                })
    
# training loop
print("TRAINING :)")
model.train()
for epoch in range(EPOCHS):
    # forward pass
    pred = MODEL(X)
    loss = LOSS_FN(pred, y)
    # backward pass
    OPTIMIZER.zero_grad()
    loss.backward()
    OPTIMIZER.step()

    if epoch % 50 == 0:
        print(f"loss: {loss.detach().item():>7f}  [{epoch:>5d}/{EPOCHS:>5d}]")
        
    metrics = {
        "train/train_loss": loss.detach().item(), 
        "train/epoch": epoch
    }
    if LOGGING:
        wandb.log(metrics)
    
# predictions
with torch.no_grad():
    preds = MODEL(X)
dis_preds = torch.argmax(preds, 1)
n_pixels = torch.bincount(dis_preds.view(-1), minlength=7).tolist()


# log predictions

# table = wandb.Table(columns=["pred", "target"] +[f"n_pixels_class_{i}" for i in range(7)])
# for i in range(BATCH_SIZE):
#     table.add_data()
# for img, pred, targ, prob in zip(images.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")):
#     table.add_data(wandb.Image(img[0].numpy()*255), pred, targ, *prob.numpy())
# wandb.log({"predictions_table":table}, commit=False)
    
