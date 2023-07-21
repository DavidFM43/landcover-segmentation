import os
import pprint

import segmentation_models_pytorch as smp
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import LandcoverDataset, int2str, train_ids, val_ids, test_ids
from get_key import wandb_key
from utils import UnNormalize
from metrics import IouMetric
from fcn import FCN8
from scheduler import LR_Scheduler

config = {
    "downsize_res": 512,
    "batch_size": 6,
    "epochs": 50,
    "lr": 5e-5,
    "model_architecture": "FCN8",
    "model_config": {
        "encoder_name": "VGG16",
        "encoder_weights": "imagenet",
        "in_channels": 3,
        "classes": 7,
    },
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# reproducibility
torch.manual_seed(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# logging
wandb_log = True
wandb_image_size = 800
wandb_resize_input = transforms.Resize(
    (wandb_image_size, wandb_image_size), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True
)
wandb_resize_label = transforms.Resize(
    (wandb_image_size, wandb_image_size), interpolation=transforms.InterpolationMode.NEAREST, antialias=True
)
checkpoint_log_step = 50
log_image_step = 7
max_log_imgs = 7
# data
downsize_res = config["downsize_res"]
batch_size = config["batch_size"]
epochs = config["epochs"]
num_classes = 7
# model
model = FCN8(num_classes, 0)
model.to(device)
# optimizer
lr = float(config["lr"])
optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": lr}], weight_decay=5e-4)
# checkpoints
save_cp = True
if save_cp and not os.path.exists("checkpoints/"):
    os.mkdir("checkpoints/")
# dataset statistics
mean = [0.4085, 0.3798, 0.2822]
std = [0.1410, 0.1051, 0.0927]
# transforms
downsize_input = transforms.Resize(
    (downsize_res, downsize_res), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True
)
downsize_label = transforms.Resize(
    (downsize_res, downsize_res), interpolation=transforms.InterpolationMode.NEAREST, antialias=True
)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
target_transform = transforms.PILToTensor()
undo_normalization = UnNormalize(mean, std)
# datasets
train_ds = LandcoverDataset(train_ids, transform=transform, target_transform=target_transform, augmentations=True)
valid_ds = LandcoverDataset(val_ids, transform=transform, target_transform=target_transform)
# dataloaders
loader_args = dict(batch_size=batch_size, pin_memory=True, num_workers=3)
train_dl = DataLoader(train_ds, shuffle=True, **loader_args)
valid_dl = DataLoader(valid_ds, shuffle=False, **loader_args)
# scheduler
scheduler = LR_Scheduler("poly", lr, epochs, len(train_dl))
# crossentropy loss fn weights
weight = torch.tensor([0.8987, 0.4091, 1.5, 0.8886, 0.9643, 1.2, 0.0], device=device)
# TODO: Implement focal loss from scratch
loss_fn = torch.hub.load("adeelh/pytorch-multi-class-focal-loss", model="FocalLoss", gamma=3, reduction="mean")


# log training and data config
if wandb_log:
    wandb.login(key=wandb_key)
    wandb.init(
        tags=["FCtl"],
        entity="landcover-classification",
        notes="",
        project="ml-experiments",
        config=dict(
            ce_weights=[round(w.item(), 2) for w in weight],
            optimizer=type(optimizer).__name__,
            loss_fn=type(loss_fn).__name__,
            num_workers=os.cpu_count(),
            wandb_size=wandb_image_size,
            **config,
        ),
    )
    wandb.watch(model, log_freq=10)  # record model gradients every 10 steps
    print("Run Config")
    pprint.pprint(dict(wandb.config))

    # TODO: refactor this to a from_pretrained method of the model
    # api = wandb.Api()
    # run = api.run("landcover-classification/ml-experiments/zecg724v")
    # run.file("checkpoints/CP_epoch30.pth").download(replace=True)
    # model.load_state_dict(torch.load("checkpoints/CP_epoch30.pth"))

# metrics
train_iou = IouMetric(num_classes=num_classes, int2str=int2str, ignore_index=6, prefix="train")
val_iou = IouMetric(num_classes=num_classes, int2str=int2str, ignore_index=6, prefix="val")

best_pred = 0.0
for epoch in range(0, epochs):
    train_loss: float = 0.0
    model.train()
    pbar = tqdm(total=len(train_ds), desc=f"Train epoch {epoch}/{epochs}", unit="img")
    # training loop
    for batch, (X, y) in enumerate(train_dl):
        scheduler(optimizer, batch, epoch, best_pred)  # update lr
        X, y = X.to(device), y.to(device)
        X_down = downsize_input(X)
        y_down = downsize_label(y)
        # forward pass
        logits = model(X_down)
        loss = loss_fn(logits, y_down)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        preds = torch.argmax(logits, 1).detach()
        # resize to original resolution
        preds = transforms.functional.resize(preds, y.shape[-2:], interpolation=transforms.InterpolationMode.NEAREST, antialias=True)
        train_iou.process(preds, y)
        # log the train loss
        if wandb_log:
            wandb.log({"train/loss": loss.item()})
        train_loss += loss.item()
        pbar.update(X.shape[0])
    train_loss /= len(train_dl)
    pbar.close()

    if wandb_log:
        metrics_dict = train_iou.compute()
        wandb.log({"epoch": epoch, "train/mean_loss": train_loss, **metrics_dict})
        train_iou.reset()

    # save checkpoints
    if epoch % checkpoint_log_step == 0:
        # TODO: save the optimizer state as well
        # check this tutorial: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
        torch.save(model.state_dict(), "checkpoints/" + f"CP_epoch{epoch}.pth")

    val_loss: float = 0.0
    model.eval()
    num_logged_imgs = 0
    # validation loop
    pbar = tqdm(total=len(valid_ds), desc=f"Valid epoch {epoch}/{epochs}", unit="img")
    for batch, (X, y) in enumerate(valid_dl):
        with torch.no_grad():
            X, y = X.to(device), y.to(device)
            X_down = downsize_input(X)
            y_down = downsize_label(y)
            # forward pass
            logits = model(X_down)
            loss = loss_fn(logits, y_down)
        val_loss += loss.item()
        preds = torch.argmax(logits, 1)
        # resize to original resolution
        preds = transforms.functional.resize(preds, y.shape[-2:], interpolation=transforms.InterpolationMode.NEAREST, antialias=True)
        # log prediction matrix
        val_iou.process(preds, y)

        # log image predictions
        if wandb_log and epoch % log_image_step == 0:
            for idx in range(len(X)):
                # log only a few images
                num_logged_imgs += 1
                if num_logged_imgs >= max_log_imgs:
                    break

                # TODO: refactor this to a function
                img_id = (idx + 1) + (batch * batch_size)
                sat_img = wandb_resize_input(undo_normalization(X[idx]))
                pred_img = wandb_resize_input(preds[idx].unsqueeze(0)).squeeze().cpu().numpy()
                label_img = wandb_resize_label(y[idx].unsqueeze(0)).squeeze().cpu().numpy()
                overlay_image = wandb.Image(
                    sat_img,
                    masks={
                        "predictions": {
                            "mask_data": pred_img,
                            "class_labels": int2str,
                        },
                        "ground_truth": {
                            "mask_data": label_img,
                            "class_labels": int2str,
                        },
                    },
                )
                wandb.log({f"Image No. {img_id}": overlay_image, "epoch": epoch})

        pbar.update(X.shape[0])  # update validation progress bar

    val_loss /= len(valid_dl)
    pbar.close()

    if wandb_log:
        metrics_dict = val_iou.compute()
        wandb.log(
            {
                "epoch": epoch,
                "val/mean_loss": val_loss,
                **metrics_dict,
            }
        )
        wandb.save("checkpoints/*")
        val_iou.reset()

if wandb_log:
    wandb.finish()
