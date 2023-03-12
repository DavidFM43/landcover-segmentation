import torch
import torch.nn.functional as F
from torchvision.utils import draw_segmentation_masks


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

def count_classes(dataloader, device, num_classes=7):
    """
    Counts the number of pixels of each class.
        Parameters:
            dataloader: Torch dataloader of pairs X, y where y contains the class labels.
            device: Device where to store output tensor. 
            num_classes: Number of classes.
        Returns:
            Torch tensor of shape (num_classes,) that contains the counts of each class.
    """
    counts = torch.zeros((num_classes,))
    for _, y in dataloader:
        counts += torch.bincount(y.view(-1), minlength=num_classes)
    counts = counts.to(device)
    return counts