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
            mask = transform(
                read_image(str(dataset.masks_dir / f"{image_id}_mask.png"))
            )
        else:
            mask = read_image(str(dataset.masks_dir / f"{image_id}_mask.png"))
        counts += torch.bincount(mask.view(-1), minlength=num_classes)
    return counts


@torch.no_grad()
def calculate_conf_matrix(logits, y, num_classes=7):
    """
    Calculate classifications per label
    transform both the predictions and targets to one hot encoding,
    perform an element-wise product between a fixed target class and
    all the class predictions, then sum over all the dims except for the
    class dim in order to get the clasifications of the fixed target class
    """
    pred = torch.argmax(logits, 1)
    ohe_pred = label_to_onehot(pred, num_classes=num_classes)
    ohe_y = label_to_onehot(y, num_classes=num_classes)
    return torch.stack(
        [(ohe_pred * ohe_y[:, [c]]).sum(dim=[0, 2, 3]) for c in range(num_classes)]
    )


@torch.no_grad()
def calculate_metrics(stage, conf_matrix, class_names):
    """
    Calculates the IoU(and accuracy) per class and average given the confusion matrix.
    """
    tp = conf_matrix.diag()  # true positives
    fp = conf_matrix.sum(0) - tp  # false positives
    n_class = conf_matrix.sum(1)  # total class gts
    accuracy = tp / (n_class + 1e-5)
    iou = tp / (n_class + fp + 1e-5)
    # log metrics average and per class
    return {
        f"{stage}/mean_accuracy": accuracy[:-1].mean().item(),
        **{
            f"{stage}/accuracy_{c_name}": acc
            for c_name, acc in zip(class_names, accuracy.tolist())
        },
        f"{stage}/mean_iou": iou[:-1].mean().item(),
        **{
            f"{stage}/iou_{c_name}": iou
            for c_name, iou in zip(class_names, iou.tolist())
        },
    }


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim())) # para que este orden?
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1) #memoria contigua y operaciones lineales



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
    target= label_to_onehot(target,7)
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
    denominator = (input ).sum(-1) + (target ).sum(-1)
    return 1 -sum(2 * (intersect / denominator.clamp(min=epsilon)))/7
