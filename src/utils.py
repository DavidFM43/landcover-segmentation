import torch
import torch.nn.functional as F
from torchvision.utils import draw_segmentation_masks
from torchvision.io import read_image


@torch.no_grad()
def rgb_to_label(mask, class_colors):
    """
    Transforms a mask image from RGB format to label encoding.
        Parameters:
            mask: Torch tensor of shape (3, H, W)
            class_colors: list of tuples of the RGB values for each class
        Returns:
            Torch tensor of shape (H, W) of label enconded classes
    """
    h, w = mask.shape[1:]  # shape expected to be (C, H, W)
    semantic_map = torch.zeros((h, w), dtype=torch.uint8)
    for idx, color in enumerate(class_colors):
        color = torch.tensor(color).view(3, 1, 1)  # rgb value
        class_map = torch.all(torch.eq(mask, color), 0)
        semantic_map[class_map] = idx
    return semantic_map


@torch.no_grad()
def label_to_onehot(mask, num_classes):
    """
    Transforms a label encoded tensor to one hot encoding.
        Parameters:
            mask: Torch tensor of shape (H, W)
            num_classes: Total number of classes.
        Returns:
            Torch tensor of shape (num_classes, H, W).
    """
    dims_p = (2, 0, 1) if mask.ndim == 2 else (0, 3, 1, 2)
    return torch.permute(
        F.one_hot(mask.type(torch.long), num_classes=num_classes).type(torch.bool),
        dims_p,
    )


@torch.no_grad()
def label_to_rgb(mask, class_colors):
    """
    Transforms a label encoded tensor to rgb.
        Parameters:
            mask: Torch tensor of shape (H, W)
            class_colors: list of tuples of the RGB values for each class
        Returns:
            Torch tensor of shape (3, H, W).
    """
    num_classes = len(class_colors)
    return draw_segmentation_masks(
        torch.zeros((3, mask.shape[-1], mask.shape[-1]), dtype=torch.uint8),
        label_to_onehot(mask, num_classes),
        alpha=1,
        colors=class_colors,
    )


@torch.no_grad()
def class_counts(dataset, transform=None, num_classes=7):
    """
    Counts the number of pixels of each class.
    """
    counts = torch.zeros((7,), dtype=torch.int64)
    for image_id in dataset.image_ids:
        if transform is not None:
            mask = transform(read_image(str(dataset.masks_dir / f"{image_id}_mask.png")))
        else:  
            mask = read_image(str(dataset.masks_dir / f"{image_id}_mask.png"))
        counts += torch.bincount(mask.view(-1), minlength=num_classes)
    return counts