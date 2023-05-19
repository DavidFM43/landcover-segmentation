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

# load configuration
parser = argparse.ArgumentParser()
parser.add_argument("--config", help="Path to configuration file")
parser.add_argument(
    "--log", help="Path to configuration file", type=bool, default=False
)
args = parser.parse_args()
with open(args.config, "r") as file:
    config = yaml.safe_load(file)

## Temporal para debugear mas facil en Kaggle
# config = {
#     "downsize_res": 512,
#     "batch_size": 6,
#     "epochs": 20,
#     "lr": 3e-4,
#     "model_architecture": "Unet",
#     "model_config": {
#         "encoder_name": "resnet34",
#         "encoder_weights": "imagenet",
#         "in_channels": 3,
#         "classes": 7,
#     },
# }


# reproducibility
torch.manual_seed(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# logging
wandb_log = args.log
# data
downsize_res = config["downsize_res"]
batch_size = config["batch_size"]
epochs = config["epochs"]
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
save_cp = True
if save_cp and not os.path.exists("checkpoints/"):
    os.mkdir("checkpoints/")
# data transformation
mean = [0.4085, 0.3798, 0.2822]
std = [0.1410, 0.1051, 0.0927]

downsize = transforms.Resize(downsize_res)
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
train_ds, valid_ds, test_ds = torch.utils.data.random_split(ds, [454, 207, 142])
loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
# dataloaders
train_dl = DataLoader(train_ds, shuffle=True, **loader_args)
valid_dl = DataLoader(valid_ds, shuffle=False, **loader_args)
test_dl = DataLoader(test_ds, shuffle=False, **loader_args)
# crossentropy loss fn weights
weights = torch.tensor(
    [0.8987, 0.4091, 0.9165, 0.8886, 0.9643, 0.9231, 0.0], device=device
)
loss_fn = nn.CrossEntropyLoss(weight=weights)
print("CE weights:", weights.tolist())

# log training and data config
if wandb_log:
    wandb.login(key=wandb_key)
    wandb.init(
        tags=["baseline"],
        entity="landcover-classification",
        notes="Same as run 58 but with improved prediction viz",
        project="ml-experiments",
        config=dict(
            ce_weights=weights.tolist(),
            optimizer=type(optimizer).__name__,
            loss_fn=type(loss_fn).__name__,
            num_workers=os.cpu_count(),
            **config,
        ),
    )
    wandb.watch(model, log_freq=10)  # record model gradients every 10 steps

print("Run Config")
pprint.pprint(dict(wandb.config))

# confusion matrix: columns are the predictions and rows are the real labels
conf_matrix = torch.zeros((7, 7), device=device)
for epoch in range(1, epochs + 1):
    conf_matrix.zero_()
    # training loop
    model.train()
    with tqdm(total=len(train_ds), desc=f"Train epoch {epoch}/{epochs}", unit="img") as pb:
        for batch, (X, y) in enumerate(train_dl):
            X, y = X.to(device), y.to(device)
            y_down = downsize(y)
            X_down = downsize(X)
            # forward pass
            logits = model(X_down)
            loss = loss_fn(logits, y_down)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = torch.argmax(logits, 1).detach()
            # resize to evaluate with the original image
            pred = transforms.functional.resize(pred, y.shape[-2:])
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
    if epoch % 20 == 0:
        torch.save(model.state_dict(), "checkpoints/" + f"CP_epoch{epoch}.pth")

    # validation loop
    conf_matrix.zero_()
    val_loss = 0.0
    model.eval()
    num_log_imgs = 0
    with tqdm(total=len(valid_ds), desc=f"Valid epoch {epoch}/{epochs}", unit="img") as pb:
        with torch.no_grad():
            for batch, (X, y) in enumerate(valid_dl):
                X, y = X.to(device), y.to(device)
                X_down = downsize(X)
                y_down = downsize(y)
                # forward pass
                logits = model(X_down)
                loss = loss_fn(logits, y_down)
                val_loss += loss.item()
                preds = torch.argmax(logits, 1).detach()
                # resize to evaluate with the original image
                preds = transforms.functional.resize(preds, y.shape[-2:])
                # log prediction matrix
                conf_matrix += calculate_conf_matrix(preds, y)

                # log image predictions at the last validation epoch
                if wandb_log and epochs % log_epochs == 0:
                    for idx in range(len(X)):
                        if num_log_imgs >= max_log_imgs:
                            break
                        num_log_imgs += 1
                        img_id = (idx + 1) + (batch * batch_size)
                        sat_img = undo_normalization(X[idx])
                        pred_img = preds[idx].cpu().numpy()
                        label_img = y[idx].cpu().numpy()
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
                        wandb.log({f"Image No. {img_id}": overlay_image})
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
wandb.finish()
