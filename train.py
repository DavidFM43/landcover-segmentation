import argparse
import os

import segmentation_models_pytorch as smp
import torch
import wandb
import yaml

from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import pprint

from dataset import LandcoverDataset, class_names, label_to_name
from get_key import wandb_key
from utils import calculate_conf_matrix, calculate_metrics, UnNormalize

## load configuration
# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--config", help="Path to configuration file", default="config.yaml"
# )
# parser.add_argument(
#     "--log",
#     help="Log training to Weights and Biases",
#     action=argparse.BooleanOptionalAction,
#     default=False,
# )
# args = parser.parse_args()
# with open(args.config, "r") as file:
#     config = yaml.safe_load(file)

# Temporal para debugear mas facil en Kaggle
config = {
    "downsize_res": 512,
    "batch_size": 6,
    "epochs": 50,
    "lr": 3e-4,
    "model_architecture": "Unet",
    "model_config": {
        "encoder_name": "resnet34",
        "encoder_weights": "imagenet",
        "in_channels": 3,
        "classes": 7,
    },
}


# reproducibility
torch.manual_seed(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# logging
wandb_log = True
wandb_image_size = 800
wandb_resize = transforms.Resize(wandb_image_size, antialias=True)
# data
downsize_res = config["downsize_res"]
batch_size = config["batch_size"]
epochs = config["epochs"]
checkpoint_log_step = 10
# evaluation
max_log_imgs = 7
log_epochs = 10
# model
device = "cuda" if torch.cuda.is_available() else "cpu"
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
train_ds, valid_ds, test_ds = torch.utils.data.random_split(ds, [454, 207, 142], generator=torch.Generator().manual_seed(42))
loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
# dataloaders
train_dl = DataLoader(train_ds, shuffle=True, **loader_args)
valid_dl = DataLoader(valid_ds, shuffle=False, **loader_args)
test_dl = DataLoader(test_ds, shuffle=False, **loader_args)
# crossentropy loss fn weights
weight = torch.tensor([0.8987, 0.4091, 0.9165, 0.8886, 0.9643, 0.9231, 0.0], device=device)
# loss_fn = nn.CrossEntropyLoss(weight=weights)
loss_fn = torch.hub.load(
    "adeelh/pytorch-multi-class-focal-loss",
    model="FocalLoss",
    alpha=weight,
    gamma=2,
    reduction="mean"
)


# log training and data config
if wandb_log:
    wandb.login(key=wandb_key)
    wandb.init(
        tags=["Unet"],
        entity="landcover-classification",
        notes="50 epochs", 
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
    # download checkpoints from wandb
    # api = wandb.Api()
    # run = api.run("landcover-classification/ml-experiments/zecg724v")
    # run.file("checkpoints/CP_epoch30.pth").download(replace=True)
    # model.load_state_dict(torch.load("checkpoints/CP_epoch30.pth"))

# confusion matrix: columns are the predictions and rows are the real labels
conf_matrix = torch.zeros((7, 7), device=device)
for epoch in range(0, epochs + 1):
    conf_matrix.zero_()
    # training loop
    model.train()
    with tqdm(total=len(train_ds), desc=f"Train epoch {epoch}/{epochs}", unit="img") as pb:
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

            pred = torch.argmax(logits, 1).detach()
            # resize to evaluate with the original image
            pred = transforms.functional.resize(pred, y.shape[-2:], antialias=True)
            conf_matrix += calculate_conf_matrix(pred, y)
            # log the train loss
            if wandb_log:
                wandb.log({"train/loss": loss.item()})
            # update progress bar and add batch loss as postfix
            pb.update(X.shape[0])
            pb.set_postfix(**{"loss (batch)": loss.item()})

    if wandb_log:
        metrics_dict = calculate_metrics("train", conf_matrix, class_names)
        wandb.log({"epoch": epoch, **metrics_dict})

    # save checkpoints
    if epoch % checkpoint_log_step == 0:
        torch.save(model.state_dict(), "checkpoints/" + f"CP_epoch{epoch}.pth")

    # validation loop
    conf_matrix.zero_()
    val_loss = 0.0
    model.eval()
    num_logged_imgs = 0
    with tqdm(total=len(valid_ds), desc=f"Valid epoch {epoch}/{epochs}", unit="img") as pb:
        with torch.no_grad():
            for batch, (X, y) in enumerate(valid_dl):
                X, y = X.to(device), y.to(device)
                X_down = downsize_t(X)
                y_down = downsize_t(y)
                # forward pass
                logits = model(X_down)
                loss = loss_fn(logits, y_down)
                val_loss += loss.item()
                preds = torch.argmax(logits, 1).detach()
                # resize to evaluate with the original image
                preds = transforms.functional.resize(preds, y.shape[-2:], antialias=True)
                # log prediction matrix
                conf_matrix += calculate_conf_matrix(preds, y)

                # log image predictions
                if wandb_log and epochs % log_epochs == 0:
                    for idx in range(len(X)):
                        if num_logged_imgs >= max_log_imgs:
                            break
                        num_logged_imgs += 1
                        img_id = (idx + 1) + (batch * batch_size)
                        sat_img = wandb_resize(undo_normalization(X[idx]))
                        pred_img = wandb_resize(preds[idx].unsqueeze(0)).squeeze().cpu().numpy()
                        label_img = wandb_resize(y[idx].unsqueeze(0)).squeeze().cpu().numpy()
                        overlay_image = wandb.Image(
                            sat_img,
                            masks={
                                "predictions": {
                                    "mask_data": pred_img,
                                    "class_labels": label_to_name,
                                },
                                "ground_truth": {
                                    "mask_data": label_img,
                                    "class_labels": label_to_name,
                                },
                            },
                        )
                        wandb.log({f"Image No. {img_id}": overlay_image, "epoch": epoch})
                pb.update(X.shape[0])  # update validation progress bar
        val_loss /= len(valid_dl)

    if wandb_log:
        metrics_dict = calculate_metrics("val", conf_matrix, class_names)
        wandb.log(
            {
                "epoch": epoch,
                "val/mean_loss": val_loss,
                **metrics_dict,
            }
        )
        wandb.save("checkpoints/*")

if wandb_log:
    wandb.finish()
