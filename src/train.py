import torch
import torchvision
from torchvision.transforms.functional import to_pil_image
from torch import nn
from torch.utils.data import DataLoader

import wandb

from dataset import LandcoverDataset, class_names
from model import Unet
from utils import ohe_mask


# reproducibility
torch.manual_seed(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# logging
wandb_log = True
# data
resize_red = 512
batch_size = 4
epochs = 1
device = "cuda" if torch.cuda.is_available() else "cpu"

# init model and optimizer
model = Unet()
model.to(device)
optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

# init datasets and dataloaders
train_dataset = LandcoverDataset(
    train=True,
    transform=torchvision.transforms.Resize(resize_red),
    target_transform=torchvision.transforms.Resize(resize_red),
)
test_dataset = LandcoverDataset(
    train=False,
    transform=torchvision.transforms.Resize(resize_red),
    target_transform=torchvision.transforms.Resize(resize_red),
)
# TODO: Try setting pin_memory to true in order to speed up training
# TODO: Play around with num_workers
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, num_workers=0, pin_memory=False, shuffle=True
)
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, num_workers=0, pin_memory=False, shuffle=True
)

# log training and data config
if wandb_log:
    wandb.login(key="5f5a6e6618ddafd57c6c7b40a8313449bfd7a04e")
    wandb.init(
        project="landcover-segmentation",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "resize_res": resize_red,
            "optimizer": type(optimizer).__name__,
            "loss_fn": type(loss_fn).__name__,
            "model": type(model).__name__,
        },
    )
    # record model gradients
    wandb.watch(model, log_freq=10)


for epoch in range(1, epochs + 1):
    print(
        "Epoch {} of {}, {}/{} batches of size {}*{}".format(
            epoch,
            epochs,
            len(train_dataloader),
            len(test_dataloader),
            batch_size,
            (torch.cuda.device_count() if device == "cuda" else 1),
        )
    )

    # training loop
    class_preds = torch.zeros(7, 7)
    model.train()
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
            ohe_pred = ohe_mask(pred, 7)
            ohe_y = ohe_mask(y, 7)
            class_preds += torch.stack(
                [
                    (ohe_pred & ohe_y[:, c].unsqueeze(1)).sum(dim=[0, 2, 3])
                    for c in range(7)
                ]
            ).to("cpu")

        if batch % 10 == 0:
            print(
                f"loss: {loss.detach().item():>7f}  [{batch:>5d}/{len(train_dataloader):>5d}]"
            )
        if wandb_log:
            wandb.log({"train/loss": loss.detach().item()})

    if wandb_log:
        with torch.no_grad():
            tp = class_preds.diag()
            fp = class_preds.sum(1) - tp
            n_class = class_preds.sum(1)
            accuracy = tp / (n_class + 1e-5)
            iou = tp / (n_class - tp + fp + 1e-5)

        wandb.log({"epoch": epoch})
        # log accuracy
        wandb.log(
            {
                "train/mean_accuracy": accuracy.mean().item(),
                "epoch": epoch,
            }
        )
        wandb.log(
            {
                **{
                    f"train/accuarcy_{c_name}": acc
                    for c_name, acc in zip(class_names, accuracy.tolist())
                },
                **{"epoch": epoch},
            }
        )
        # log iou
        wandb.log(
            {
                "train/mean_iou": iou.mean().item(),
                "epoch": epoch,
            }
        )
        wandb.log(
            {
                **{
                    f"train/iou_{c_name}": iou
                    for c_name, iou in zip(class_names, iou.tolist())
                },
                **{"epoch": epoch},
            }
        )

    # TODO: Add validation cadence
    # validation loop
    class_preds *= 0
    val_loss = 0.0
    model.eval()
    size = len(test_dataloader)
    with torch.no_grad():
        for batch, (X, y) in enumerate(test_dataloader):
            X, y = X.to(device), y.to(device)
            # forward pass
            logits = model(X)
            loss = loss_fn(logits, y)
            val_loss += loss.detach().item()

            with torch.no_grad():
                pred = torch.argmax(logits, 1)
                ohe_pred = ohe_mask(pred, 7)
                ohe_y = ohe_mask(y, 7)
                class_preds += torch.stack(
                    [
                        (ohe_pred & ohe_y[:, c].unsqueeze(1)).sum(dim=[0, 2, 3])
                        for c in range(7)
                    ]
                ).to("cpu")

                # log image predictions
                if wandb_log:
                    for i in range(len(X)):
                        image = to_pil_image((X[i] * 255).type(torch.uint8).to("cpu"))
                        wandb.log(
                            {
                                f"Image_{(i + 1) + (batch*batch_size)}": wandb.Image(
                                    image,
                                    masks={
                                        "predictions": {
                                            "mask_data": pred[i].to("cpu").numpy(),
                                            "class_labels": {
                                                idx: name
                                                for idx, name in enumerate(class_names)
                                            },
                                        },
                                        "ground_truth": {
                                            "mask_data": y[i].to("cpu").numpy(),
                                            "class_labels": {
                                                idx: name
                                                for idx, name in enumerate(class_names)
                                            },
                                        },
                                    },
                                )
                            }
                        )

    val_loss /= size
    # TODO: Don't consider the unknown class for evaluation
    if wandb_log:
        with torch.no_grad():
            tp = class_preds.diag()
            fp = class_preds.sum(1) - tp
            n_class = class_preds.sum(1)
            accuracy = tp / (n_class + 1e-5)
            iou = tp / (n_class - tp + fp + 1e-5)

        # log metrics
        wandb.summary = {
            **wandb.summary,
            "val/mean_loss": val_loss,
            "val/mean_accuracy": accuracy.mean().item(),
            **{
                f"val/accuarcy_{c_name}": acc
                for c_name, acc in zip(class_names, accuracy.tolist())
            },
            "val/mean_iou": iou.mean().item(),
            **{
                f"val/iou_{c_name}": iou
                for c_name, iou in zip(class_names, iou.tolist())
            },
        }


if wandb_log:
    wandb.finish()
