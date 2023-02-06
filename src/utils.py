import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import draw_segmentation_masks
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["savefig.bbox"] = "tight"


def show(imgs):
    """Helper function to show images in the torch format"""
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(100, 100))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = T.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def preprocess_mask(mask, class_colors):
    """
    Transforms a mask from RGB format to a tensor of shape (H, W)
    where a pixel represents the class label
    """
    num_classes = len(class_colors)
    h, w = mask.shape[1:]  # shape expected to be (C, H, W)
    semantic_map = torch.zeros((h, w), dtype=torch.uint8)
    # iterate over all the classes
    for idx, color in enumerate(class_colors):
        color = torch.tensor(color).view(3, 1, 1)  # rgb value
        class_map = torch.all(torch.eq(mask, color), 0)
        semantic_map[class_map] = idx
    return semantic_map


def ohe_mask(mask, num_classes):
    """Turns a label tensor of shape(H, W) to (num_classes, H, W)"""
    return torch.permute(
        F.one_hot(mask.type(torch.long), num_classes=num_classes).type(torch.bool),
        (2, 0, 1),
    )


def mask_to_img(mask, class_colors):
    """Turns a segmentation mask tensor with indices to and RGB tensor image"""
    num_classes = len(class_colors)
    return draw_segmentation_masks(
        torch.zeros((3, mask.shape[-1], mask.shape[-1]), dtype=torch.uint8),
        ohe_mask(mask, num_classes),
        alpha=1,
        colors=class_colors,
    )
