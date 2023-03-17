import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import wandb

from dataset import LandcoverDataset, class_names, class_labels
from model import Unet
from utils import label_to_onehot, class_counts


# reproducibility
torch.manual_seed(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
wandb_log = True
# data
resize_res = 512
batch_size = 5
epochs = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
# init model and optimizer
model = Unet()
model.to(device)
lr = 3e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
save_cp = True
if save_cp:
    os.mkdir("checkpoints/")
# init datasets and dataloaders
transform_args = dict(
    transform=torchvision.transforms.Resize(resize_res),
    target_transform=torchvision.transforms.Resize(resize_res),
)
train_dataset = LandcoverDataset(train=True, **transform_args)
valid_dataset = LandcoverDataset(train=False, **transform_args)
n_train = len(train_dataset)
n_val = len(valid_dataset)
loader_args = dict(
    batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True, shuffle=True
)
train_dataloader = DataLoader(train_dataset, **loader_args)
valid_dataloader = DataLoader(valid_dataset, **loader_args)

# count the classes of the training data to add weights to the loss function
counts = class_counts(train_dataset, transform=transform_args["target_transform"])
weights = 1 - counts / counts.sum()  # [0.893, 0.427, 0.916, 0.880, 0.966, 0.914, 0.999]
weights[-1] = 0
weights = weights.to(device)
loss_fn = nn.CrossEntropyLoss(weight=weights)
print("CE weights:", weights.tolist())

# log training and data config
if wandb_log:
    wandb.login(key="5f5a6e6618ddafd57c6c7b40a8313449bfd7a04e")
    wandb.init(
        tags=["baseline"],
        notes="100 epochs",
        project="landcover-segmentation",
        config=dict(
            ce_weights=weights,
            epochs=epochs,
            batch_size=batch_size,
            resize_res=resize_res,
            optimizer=type(optimizer).__name__,
            loss_fn=type(loss_fn).__name__,
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
# nij is the number of pixels of class i predicted to belong to class j
n = torch.zeros((7, 7), device=device)
for epoch in range(1, epochs + 1):
    # training loop
    model.train()
    with tqdm(total=n_train, desc=f"Train epoch {epoch}/{epochs}", unit="img") as pb:
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            # forward pass
            logits = model(X)
            loss = loss_fn(logits, y)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ## calculate classifications per label
            # transform both the predictions and targets to one hot encoding,
            # perform an element-wise product between a fixed target class and
            # all the class predictions, then sum over all the dims except for the
            # class dim in order to get the clasifications of the fixed target class
            with torch.no_grad():
                pred = torch.argmax(logits, 1)
                ohe_pred = label_to_onehot(pred, num_classes=7)
                ohe_y = label_to_onehot(y, num_classes=7)
                n += torch.stack(
                    [(ohe_pred * ohe_y[:, [c]]).sum(dim=[0, 2, 3]) for c in range(7)]
                )
            # log the train loss
            if wandb_log:
                wandb.log({"train/loss": loss.item()})
            # update progress bar and add batch loss as postfix
            pb.update(X.shape[0])
            pb.set_postfix(**{"loss (batch)": loss.item()})

    if wandb_log:
        with torch.no_grad():
            tp = n.diag()  # true positives
            fp = n.sum(0) - tp  # false positives
            n_class = n.sum(1)  # total class gts
            accuracy = tp / (n_class + 1e-5)
            iou = tp / (n_class + fp + 1e-5)
        # log metrics mean and per class
        wandb.log(
            {
                "epoch": epoch,
                "train/mean_accuracy": accuracy[:-1].mean().item(),
                **{
                    f"train/accuracy_{c_name}": acc
                    for c_name, acc in zip(class_names, accuracy.tolist())
                },
                "train/mean_iou": iou[:-1].mean().item(),
                **{
                    f"train/iou_{c_name}": iou
                    for c_name, iou in zip(class_names, iou.tolist())
                },
            }
        )
    # save checkpoints
    if epoch % 20 == 0:
        torch.save(model.state_dict(), "checkpoints/" + f"CP_epoch{epoch}.pth")

    # validation loop
    n *= 0
    val_loss = 0.0
    model.eval()
    pred_table = wandb.Table(columns=["ID", "Image"])  # table of prediction masks
    with tqdm(total=n_val, desc=f"Valid epoch {epoch}/{epochs}", unit="img") as pb:
        with torch.no_grad():
            for batch, (X, y) in enumerate(valid_dataloader):
                X, y = X.to(device), y.to(device)
                # forward pass
                logits = model(X)
                loss = loss_fn(logits, y)
                val_loss += loss.item()
                ## calculate classifications per label
                # transform both the predictions and targets to one hot encoding,
                # perform an element-wise product between a fixed target class and
                # all the class predictions, then sum over all the dims except for the
                # class dim in order to get the clasifications of the fixed target class
                pred = torch.argmax(logits, 1)
                ohe_pred = label_to_onehot(pred, num_classes=7)
                ohe_y = label_to_onehot(y, num_classes=7)
                n += torch.stack(
                    [(ohe_pred * ohe_y[:, [c]]).sum(dim=[0, 2, 3]) for c in range(7)]
                )
                # log image predictions at the last validation epoch
                if wandb_log and epoch == epochs:
                    for idx in range(len(X)):
                        id = (idx + 1) + (batch * batch_size)
                        pred_table.add_data(
                            id,
                            # save satellite image, real mask and prediction mask
                            wandb.Image(
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
                            ),
                        )
                pb.update(X.shape[0])  # update validation progress bar
        val_loss /= len(valid_dataloader)
    if wandb_log:
        tp = n.diag()  # true positives
        fp = n.sum(0) - tp  # false positives
        n_class = n.sum(1)  # total class gts
        accuracy = tp / (n_class + 1e-5)
        iou = tp / (n_class + fp + 1e-5)
        # log metrics avearge and per class
        wandb.log(
            {
                "epoch": epoch,
                "Predictions table": pred_table,
                "val/mean_loss": val_loss,
                "val/mean_accuracy": accuracy[:-1].mean().item(),
                **{
                    f"val/accuracy_{c_name}": acc
                    for c_name, acc in zip(class_names, accuracy.tolist())
                },
                "val/mean_iou": iou[:-1].mean().item(),
                **{
                    f"val/iou_{c_name}": iou
                    for c_name, iou in zip(class_names, iou.tolist())
                },
            }
        )
        wandb.save("checkpoints/*")
