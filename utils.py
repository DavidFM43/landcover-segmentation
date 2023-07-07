import torch
import torch.nn.functional as F
import torchvision
from torchvision.utils import draw_segmentation_masks
from tqdm import tqdm


class UnNormalize(torchvision.transforms.Normalize):
    """Transformation that reverts normalization given original mean and std."""

    def __init__(self, mean, std, *args, **kwargs):
        new_mean = [-m / s for m, s in zip(mean, std)]
        new_std = [1 / s for s in std]
        super().__init__(new_mean, new_std, *args, **kwargs)


def rgb2label(mask: torch.Tensor, int2rgb: dict) -> torch.Tensor:
    """
    Transforms an image tensor from RGB format to label encoding format.
        Parameters:
            mask: Torch tensor of shape (C, H, W)
            class_colors: dict mapping class labels to RGB values
        Returns:
            Torch tensor of shape (H, W) of label enconded classes
    """
    h, w = mask.shape[1:]  # shape expected to be (C, H, W)
    le_mask = torch.zeros((h, w), dtype=torch.long)
    for idx, color in int2rgb.items():
        color = torch.tensor(color).view(3, 1, 1)  # rgb value
        class_map = torch.all(torch.eq(mask, color), dim=0)
        le_mask[class_map] = idx
    return le_mask


def label2onehot(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Transforms a label encoded tensor to one hot encoding.
        Parameters:
            mask: Torch tensor of shape (H, W)
            num_classes: Total number of classes.
        Returns:
            Torch tensor of shape (num_classes, H, W).
    """
    dims_permute = (2, 0, 1) if mask.ndim == 2 else (0, 3, 1, 2)  # channels first to channels last
    ohe_mask = torch.permute(F.one_hot(mask.long(), num_classes=num_classes), dims_permute)
    return ohe_mask


def label2rgb(mask: torch.Tensor, int2rgb: dict) -> torch.Tensor:
    """
    Transforms a label encoded tensor to rgb.
        Parameters:
            mask: Torch tensor of shape (H, W)
            class_colors: dict mapping class labels to RGB values
        Returns:
            Torch tensor of shape (3, H, W).
    """
    num_classes = len(int2rgb)
    h, w = mask.shape
    return draw_segmentation_masks(
        torch.zeros((3, h, w), dtype=torch.uint8),
        label2onehot(mask, num_classes).bool(),
        alpha=1,
        colors=list(int2rgb.values()),
    )


def count_classes(dataset: torch.utils.data.Dataset, num_classes: int = 7) -> torch.Tensor:
    """
    Counts the number of pixels of each class.
    """
    train_dl = torch.utils.data.DataLoader(dataset, batch_size=30, shuffle=False, num_workers=2)
    counts = torch.zeros((num_classes,), dtype=torch.int64)
    with tqdm(desc="Class counts calculation", total=len(dataset)) as pbar:
        for X, y in train_dl:
            counts += torch.bincount(y.view(-1), minlength=num_classes)
            pbar.update(X.shape[0])
    return counts


def calculate_channel_stats(dataset: torch.utils.data.Dataset):
    """
    Calculates the mean and standard deviation of each color channel in the dataset.
    """
    h, w = dataset[0][0].shape[1:]
    dl = torch.utils.data.DataLoader(dataset, batch_size=30, shuffle=False, num_workers=2)
    mean = 0
    with tqdm(desc="Channel-wise mean calculation", total=len(dataset)) as pbar:
        for X, y in dl:
            mean += X.sum([0, 2, 3])
            pbar.update(X.shape[0])
    mean /= h * w * len(dataset)

    std = 0
    with tqdm(desc="Channel-wise std calculation", total=len(dataset)) as pbar:
        for X, y in dl:
            std += ((X - mean.view(1, -1, 1, 1)) ** 2).sum([0, 2, 3])
            pbar.update(X.shape[0])
    std /= h * w * len(dataset)
    std **= 0.5

    return mean, std


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))  # para que este orden?
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)  # memoria contigua y operaciones lineales


def dice_loss(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.
    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    """
    Our target is in label encode to use crossentropy. 
    For dice loss we need our target in one hot with the 7 dimensions 
    """
    target = label2onehot(target, 7)
    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 1 - sum(2 * (intersect / denominator.clamp(min=epsilon)))


def print_mem():
    """Prints current memory usage info"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Current device:         ", device)
    print("Total memory allocated: ", round(torch.cuda.memory_allocated() / 1024**2, 2), "MB")
    print("Max memory allocated:   ", round(torch.cuda.max_memory_allocated() / 1024**2, 2), "MB")
    print("Memory reserved:        ", round(torch.cuda.memory_reserved() / 1024**2, 2), "MB")
    print("Max memory reserved:    ", round(torch.cuda.max_memory_reserved() / 1024**2, 2), "MB")
