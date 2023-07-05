import torch


class IouMetric:
    def __init__(self, num_classes: int, int2str: dict, ignore_index: int = 255, prefix="train"):
        """
        Args:
            num_classes: number of classes
            int2str: dictionary mapping class index to class name
            ignore_index: index to ignore in the metric calculations
            prefix: prefix to use for logging
        """
        self.area_intersect = torch.zeros(num_classes)
        self.area_label = torch.zeros(num_classes)
        self.area_pred = torch.zeros(num_classes)
        self.num_classes = num_classes
        self.int2str = int2str
        self.ignore_index = ignore_index
        self.prefix = prefix

    def process(self, preds, labels):
        ## NOTE: Including this would be conceptually correct but ive decided to ignore it for now
        ##       as other implementations dont include it
        # mask = labels != self.ignore_index
        # preds = preds[mask]
        # labels = labels[mask]

        # compute area of intersection, label and prediction
        intersect = preds[preds == labels]
        area_intersect = torch.histc(intersect.float(), bins=self.num_classes, min=0, max=self.num_classes - 1)
        area_label = torch.histc(labels.float(), bins=self.num_classes, min=0, max=self.num_classes - 1)
        area_pred = torch.histc(preds.float(), bins=self.num_classes, min=0, max=self.num_classes - 1)

        # update results
        self.area_intersect += area_intersect
        self.area_label += area_label
        self.area_pred += area_pred

    def compute(self) -> dict:
        iou = self.area_intersect / (self.area_label + self.area_pred - self.area_intersect)
        if 0 <= self.ignore_index <= self.num_classes - 1:
            iou[self.ignore_index] = torch.nan
        mean_iou = torch.nanmean(iou)

        metrics = {
            f"{self.prefix}/iou/{self.int2str[idx]}": round(iou[idx].item(), 4)
            for idx in range(len(iou))
            if idx != self.ignore_index
        }
        metrics[f"{self.prefix}/mean_iou"] = round(mean_iou.item(), 4)

        return metrics

    def reset(self):
        self.area_intersect.zero_()
        self.area_label.zero_()
        self.area_pred.zero_()
