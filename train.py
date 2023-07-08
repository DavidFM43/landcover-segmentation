import os
import pprint

import segmentation_models_pytorch as smp
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import LandcoverDataset, int2str
from get_key import wandb_key
from utils import UnNormalize
from metrics import IouMetric

config = {
    "downsize_res": 512,
    "batch_size": 12,
    "epochs": 15,
    "lr": 1e-4,
    "model_architecture": "Unet",
    "model_config": {
        "encoder_name": "resnet34",
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
wandb_log           = True
wandb_image_size    = 800
wandb_resize        = transforms.Resize(wandb_image_size, antialias=True)
checkpoint_log_step = 10
log_image_step      = 10
max_log_imgs        = 7
# data
downsize_res = config["downsize_res"]
batch_size   = config["batch_size"]
epochs       = config["epochs"]
num_classes  = 7
# model
model_architecture = getattr(smp, config["model_architecture"])
model = model_architecture(**config["model_config"])
model.to(device)
# optimizer
lr = float(config["lr"])
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# checkpoints
save_cp = True
if save_cp and not os.path.exists("checkpoints/"):
    os.mkdir("checkpoints/")
# dataset statistics
mean = [0.4085, 0.3798, 0.2822]
std = [0.1410, 0.1051, 0.0927]
# transforms
downsize_t = transforms.Resize(downsize_res, antialias=True)
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)
target_transform = transforms.Compose(
    [
        transforms.PILToTensor(),
    ]
)
undo_normalization = UnNormalize(mean, std)
# datasets
ds = LandcoverDataset(transform=transform, target_transform=target_transform)
train_ds, valid_ds, test_ds = torch.utils.data.random_split(
    ds, [454, 207, 142], generator=torch.Generator().manual_seed(42)
)
loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
# dataloaders
train_dl = DataLoader(train_ds, shuffle=True, **loader_args)
valid_dl = DataLoader(valid_ds, shuffle=False, **loader_args)
test_dl = DataLoader(test_ds, shuffle=False, **loader_args)
# crossentropy loss fn weights
weight = torch.tensor([0.8987, 0.4091, 1.5, 0.8886, 0.9643, 1.2, 0.0], device=device)
loss_fn = nn.CrossEntropyLoss(weight=weight)
# # TODO: Implement focal loss from scratch
# loss_fn = torch.hub.load(
#     "adeelh/pytorch-multi-class-focal-loss", model="FocalLoss", alpha=weight, gamma=2, reduction="mean"
# )


# log training and data config
if wandb_log:
    wandb.login(key=wandb_key)
    wandb.init(
        tags=["Unet"],
        entity="landcover-classification",
        notes="Decrease learning rate",
        project="ml-experiments",
        config=dict(
            ce_weights=weight.tolist(),
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

for epoch in range(1, epochs + 1):
    train_loss: float = 0.0
    model.train()
    pbar = tqdm(total=len(train_ds), desc=f"Train epoch {epoch}/{epochs}", unit="img")
    # training loop
    for batch, (X, y) in enumerate(train_dl):
        X, y = X.to(device), y.to(device)
        y_down = downsize_t(y)
        X_down = downsize_t(X)
        # forward pass
        logits = model(X_down)
        loss = loss_fn(logits, y_down)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        preds = torch.argmax(logits, 1).detach()
        # resize to original resolution
        preds = transforms.functional.resize(preds, y.shape[-2:], antialias=True)
        train_iou.process(preds, y)
        # log the train loss
        if wandb_log:
            wandb.log({"train/loss": loss.item()})
        train_loss += loss.item()
        pbar.update(X.shape[0])
        pbar.set_postfix(
            **{"batch loss": loss.item(), "mem_alloc": f"{int(torch.cuda.memory_allocated() / 1024**2)} Mb"}

        )
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
            X_down = downsize_t(X)
            y_down = downsize_t(y)
            # forward pass
            logits = model(X_down)
            loss = loss_fn(logits, y_down)
        val_loss += loss.item()
        preds = torch.argmax(logits, 1)
        # resize to original resolution
        preds = transforms.functional.resize(preds, y.shape[-2:], antialias=True)
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
                sat_img = wandb_resize(undo_normalization(X[idx]))
                pred_img = wandb_resize(preds[idx].unsqueeze(0)).squeeze().cpu().numpy()
                label_img = wandb_resize(y[idx].unsqueeze(0)).squeeze().cpu().numpy()
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
        pbar.set_postfix(**{"mem_alloc": f"{int(torch.cuda.memory_allocated() / 1024**2)} Mb"})

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
