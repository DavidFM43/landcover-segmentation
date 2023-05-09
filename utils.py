import torch
import torch.nn.functional as F
from torchvision.utils import draw_segmentation_masks
from tqdm import tqdm


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
def class_counts(dataset, num_classes=7):
    """
    Counts the number of pixels of each class.
    """
    train_dl = torch.utils.data.DataLoader(
        dataset, batch_size=30, shuffle=False, num_workers=2
    )
    counts = torch.zeros((num_classes,), dtype=torch.int64)
    with tqdm(desc="Class counts calculation", total=len(dataset)) as pbar:
        for X, y in train_dl:
            counts += torch.bincount(y.view(-1), minlength=num_classes)
            pbar.update(X.shape[0])
    return counts

def calculate_channel_stats(ds):
    """
    Calculates the mean and standard deviation of each color channel in the dataset.
    """
    h, w = ds[0][0].shape[1:]
    dl = torch.utils.data.DataLoader(ds, batch_size=30, shuffle=False, num_workers=2)
    mean = 0
    with tqdm(desc="Channel-wise mean calculation", total=len(ds)) as pbar:
        for X, y in dl:
            mean += X.sum([0, 2, 3])
            pbar.update(X.shape[0])
    mean /= h * w * len(ds)

    std = 0
    with tqdm(desc="Channel-wise std calculation", total=len(ds)) as pbar:
        for X, y in dl:
            std += ((X - mean.view(1, -1, 1, 1)) ** 2).sum([0, 2, 3])
            pbar.update(X.shape[0])
    std /= h * w * len(ds)
    std **= 0.5

    return mean, std


@torch.no_grad()
def calculate_conf_matrix(logits, y, num_classes=7):
    """
    Calculates the confusison matrix of the labels and predictions.
    The rows are true labels and columns are predictions.

    Especifically transform both the predictions and targets to one hot encoding,
    perform an element-wise product between a fixed label class and
    all the class predictions, then sum over all the dims except for the
    class dim in order to get the predictions given the fixed true class.
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
    target = label_to_onehot(target, 7)
    # input and target shapes must match
    assert (
        input.size() == target.size()
    ), "'input' and 'target' must have the same shape"

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
