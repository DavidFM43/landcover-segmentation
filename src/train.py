import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import wandb

from dataset import LandcoverDataset, class_names, class_labels
from model import Unet
from utils import label_to_onehot


# reproducibility
torch.manual_seed(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
wandb_log = True
# data
resize_res = 512
batch_size = 5
epochs = 5
device = "cuda" if torch.cuda.is_available() else "cpu"
# init model and optimizer
model = Unet()
model.to(device)
optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()
# init datasets and dataloaders
transform_args = dict(
    transform=torchvision.transforms.Resize(resize_res),
    target_transform=torchvision.transforms.Resize(resize_res),
)
train_dataset = LandcoverDataset(train=True, **transform_args)
valid_dataset = LandcoverDataset(train=False, **transform_args)

# TODO: Try setting pin_memory to true in order to speed up training
loader_args = dict(
    batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=False, shuffle=True
)
train_dataloader = DataLoader(train_dataset, **loader_args)
valid_dataloader = DataLoader(valid_dataset, **loader_args)

# log training and data config
if wandb_log:
    wandb.login(key="5f5a6e6618ddafd57c6c7b40a8313449bfd7a04e")
    wandb.init(
        project="landcover-segmentation",
        save_code=True,
        config=dict(
            epochs=epochs,
            batch_size=batch_size,
            resize_res=resize_res,
            optimizer=type(optimizer).__name__,
            loss_fn=type(loss_fn).__name__,
            model=type(model).__name__,
            num_workers=os.cpu_count(),
        ),
    )
    wandb.watch(model, log_freq=10)  # record model gradients
print(
    "Train/Validation: {}/{} batches of size {}*{}".format(
        len(train_dataloader),
        len(valid_dataloader),
        batch_size,
        (torch.cuda.device_count() if device == "cuda" else 1),
    )
)

for epoch in range(1, epochs + 1):
    # nij is the number of pixels of class i predicted to belong to class j
    n = torch.zeros(7, 7)

    model.train()
    # training loop
    with tqdm(
        total=len(train_dataset), desc=f"Epoch {epoch}/{epochs}", unit="img"
    ) as pbar:
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            # forward pass
            logits = model(X)
            loss = loss_fn(logits, y)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # calculate predictions per class
            with torch.no_grad():
                pred = torch.argmax(logits, 1)
                ohe_pred = label_to_onehot(pred, num_classes=7)
                ohe_y = label_to_onehot(y, num_classes=7)
                n += torch.stack(
                    [
                        (ohe_pred & ohe_y[:, c].unsqueeze(1)).sum(dim=[0, 2, 3])
                        for c in range(7)
                    ]
                ).cpu()
            # record train loss
            if wandb_log:
                wandb.log({"train/loss": loss.item()})
            # update pbar
            pbar.update(X.shape[0])
            pbar.set_postfix(**{"loss (batch)": loss.item()})

        if wandb_log:
            with torch.no_grad():
                tp = n.diag()  # true positives
                fp = n.sum(1) - tp  # false positives
                n_class = n.sum(1)  # total per class
                accuracy = tp / (n_class + 1e-5)
                iou = tp / (n_class - tp + fp + 1e-5)
            # log metrics avearge and per class
            wandb.log(
                {
                    "epoch": epoch,
                    "train/mean_accuracy": accuracy.mean().item(),
                    **{
                        f"train/accuracy_{c_name}": acc
                        for c_name, acc in zip(class_names, accuracy.tolist())
                    },
                    "train/mean_iou": iou.mean().item(),
                    **{
                        f"train/iou_{c_name}": iou
                        for c_name, iou in zip(class_names, iou.tolist())
                    },
                }
            )

        # TODO: Add validation cadence
        # validation loop
        n *= 0
        val_loss = 0.0
        model.eval()
        pred_table = wandb.Table(columns=["ID", "Image"])
        with torch.no_grad():
            with tqdm(
                total=len(valid_dataset),
                desc=f"Validation {epoch}/{epochs}",
                unit="img",
            ) as pbar:
                for batch, (X, y) in enumerate(valid_dataloader):
                    X, y = X.to(device), y.to(device)
                    # forward pass
                    logits = model(X)
                    loss = loss_fn(logits, y)
                    val_loss += loss.item()
                    # calculate predictions per class
                    pred = torch.argmax(logits, 1)
                    ohe_pred = label_to_onehot(pred, num_classes=7)
                    ohe_y = label_to_onehot(y, num_classes=7)
                    n += torch.stack(
                        [
                            (ohe_pred & ohe_y[:, c].unsqueeze(1)).sum(dim=[0, 2, 3])
                            for c in range(7)
                        ]
                    ).cpu()
                    # log image predictions
                    if wandb_log:
                        for idx in range(len(X)):
                            id = (idx + 1) + (batch * batch_size)
                            img = wandb.Image(
                                X[idx].cpu(),
                                masks={
                                    "predictions": {
                                        "mask_data": pred[idx].cpu().numpy(),
                                        "class_labels": class_labels,
                                    },
                                    "ground_truth": {
                                        "mask_data": y[idx].cpu().numpy(),
                                        "class_labels": class_labels,
                                    },
                                },
                            )
                            pred_table.add_data(id, img)
                    pbar.update(X.shape[0])
            val_loss /= len(valid_dataloader)
        if wandb_log:
            # TODO: Don't consider the unknown class for evaluation
            with torch.no_grad():
                tp = n.diag()  # true positives
                fp = n.sum(1) - tp  # false positives
                n_class = n.sum(1)  # total predictions per class
                accuracy = tp / (n_class + 1e-5)
                iou = tp / (n_class - tp + fp + 1e-5)
            # log metrics avearge and per class
            wandb.log(
                {
                    "epoch": epoch,
                    "Predictions table": pred_table,
                    "val/mean_loss": val_loss,
                    "val/mean_accuracy": accuracy.mean().item(),
                    **{
                        f"val/accuracy_{c_name}": acc
                        for c_name, acc in zip(class_names, accuracy.tolist())
                    },
                    "val/mean_iou": iou.mean().item(),
                    **{
                        f"val/iou_{c_name}": iou
                        for c_name, iou in zip(class_names, iou.tolist())
                    },
                }
            )
