import torch
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
from torch import nn
from pathlib import Path
from sklearn.model_selection import train_test_split

from dataset import LandcoverDataset, DataLoader
from baseline import UNet


torch.manual_seed(1)

base_dir = Path("data")
train_dir = base_dir / "train"
transformed_masks_path = base_dir / "train_masks"
annotations = pd.read_csv(base_dir / "metadata.csv")
image_ids = annotations[annotations["split"] == "train"]["image_id"].values
device = "cuda" if torch.cuda.is_available() else "cpu"

# split in train and test sets
train_ids, test_ids = train_test_split(
    image_ids, train_size=0.8, shuffle=True, random_state=42
)

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

train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=5, shuffle=False)

LR = 0.001
EPOCHS = 1000

lossi = []
model = UNet().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

X, y = next(iter(train_dataloader))
X, y = X.to(device), y.to(device)

model.train()
for i in range(EPOCHS):
    pred = model(X)
    loss = loss_fn(pred, y)
    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    lossi.append(loss.detach().item())
    if i % 50 == 0:
        print(f"loss: {loss.item():>7f}  [{i:>5d}/{EPOCHS:>5d}]")

    plt.plot(lossi)
