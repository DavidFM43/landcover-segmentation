import torch
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import wandb

from dataset import LandcoverDataset, class_names, class_labels
from model import Unet
from utils import (
    class_counts,
    calculate_conf_matrix,
    calculate_metrics,
    dice_loss
)


# reproducibility
torch.manual_seed(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
wandb_log = True
# data
resize_res = 512
batch_size = 5
epochs = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
# init model and optimizer
model = Unet()
model.to(device)
lr = 3e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
save_cp = True
if save_cp:
    os.mkdir("checkpoints/")
transform_args = dict(
    transform=transforms.Resize(resize_res),
    target_transform=transforms.Resize(resize_res),
)
loader_args = dict(
    batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True, shuffle=True
)
# init datasets and dataloaders
train_dataset = LandcoverDataset(train=True, **transform_args)
valid_dataset = LandcoverDataset(train=False, **transform_args)
n_train = len(train_dataset)
n_val = len(valid_dataset)
train_dataloader = DataLoader(train_dataset, **loader_args)
valid_dataloader = DataLoader(valid_dataset, **loader_args)

# count the class labels in the training dataset
counts = class_counts(train_dataset, transform=transform_args["target_transform"])
weights = 1 - counts / counts.sum()  # [0.893, 0.427, 0.916, 0.880, 0.966, 0.914, 0.999]
weights[-1] = 0
weights = weights.to(device)
#loss_fn = nn.CrossEntropyLoss(weight=weights)
print("CE weights:", weights.tolist())

# log training and data config
if wandb_log:
    wandb.login(key="2699e8522063dc2ad0f359c8230e5cc09db3ebd8")
    wandb.init(
        tags=["baseline"],
        notes="3 epochs",
        project="landcover-segmentation",
        config=dict(
            ce_weights=weights,
            epochs=epochs,
            batch_size=batch_size,
            resize_res=resize_res,
            optimizer=type(optimizer).__name__,
            loss_fn="DiceLoss",
            lr=lr,
            model=type(model).__name__,
            num_workers=os.cpu_count(),
        ),
    )
    wandb.watch(model, log_freq=10)  # record model gradients every 10 steps

print(
    f"""Starting training:
    Epochs:            {epochs}
    Batch size:        {batch_size}
    Learning rate:     {lr}
    Training size:     {len(train_dataloader) * batch_size} 
    Validation size:   {len(valid_dataloader) * batch_size}
    Device:            {device}
    Images resolution: {resize_res}
    """
)

# ij is the number of pixels of class i predicted to belong to class j
conf_matrix = torch.zeros((7, 7), device=device)
for epoch in range(1, epochs + 1):
    conf_matrix.zero_()
    # training loop
    model.train()
    with tqdm(total=n_train, desc=f"Train epoch {epoch}/{epochs}", unit="img") as pb:
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            # forward pass
            logits = model(X)

            loss = dice_loss(logits,y, weight=weights )
            #loss = loss_fn(logits, y)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            conf_matrix += calculate_conf_matrix(logits, y)
            # log the train loss
            if wandb_log:
                wandb.log({"train/loss": loss.item()})
            # update progress bar and add batch loss as postfix
            pb.update(X.shape[0])
            pb.set_postfix(**{"loss (batch)": loss.item()})

    if wandb_log:
        metrics_dict = calculate_metrics("train", conf_matrix, class_labels)
        wandb.log({"epoch": epoch, **metrics_dict})

    # save checkpoints
    if epoch % 20 == 0:
        torch.save(model.state_dict(), "checkpoints/" + f"CP_epoch{epoch}.pth")

    # validation loop
    conf_matrix.zero_()
    val_loss = 0.0
    model.eval()
    pred_table = wandb.Table(columns=["ID", "Image"])  # table of prediction masks
    with tqdm(total=n_val, desc=f"Valid epoch {epoch}/{epochs}", unit="img") as pb:
        with torch.no_grad():
            for batch, (X, y) in enumerate(valid_dataloader):
                X, y = X.to(device), y.to(device)
                # forward pass
                logits = model(X)
                loss = dice_loss(logits, y, weight=weights)
                #loss = loss_fn(logits, y)
                val_loss += loss.item()
                # log prediction matrix
                conf_matrix += calculate_conf_matrix(logits, y)
                pred = torch.argmax(logits, 1).detach()
                # log image predictions at the last validation epoch
                if wandb_log and epoch == epochs:
                    for idx in range(len(X)):
                        id = (idx + 1) + (batch * batch_size)
                        overlay_image = wandb.Image(
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
                        pred_table.add_data(id, overlay_image)
                pb.update(X.shape[0])  # update validation progress bar
        val_loss /= len(valid_dataloader)
    if wandb_log:
        metrics_dict = calculate_metrics("val", conf_matrix, class_labels)
        wandb.log(
            {
                "epoch": epoch,
                "Predictions table": pred_table,
                "val/mean_loss": val_loss,
                **metrics_dict,
            }
        )
        wandb.save("checkpoints/*")
