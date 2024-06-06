import torch


def intersect_and_union(pred, target, num_classes: int, ignore_index: int):
    assert len(pred) == len(target) == 1, "batchsize==1 is required"
    assert pred.shape == target.shape

    # mask out ignore index
    mask = (target != ignore_index)
    pred = pred[mask]
    target = target[mask]

    # intersect and union
    intersect = pred[pred == target]
    area_intersect = torch.histc(intersect.float(), bins=num_classes, min=0, max=num_classes - 1).cpu()
    area_pred = torch.histc(pred.float(), bins=num_classes, min=0, max=num_classes - 1).cpu()
    area_target = torch.histc(target.float(), bins=num_classes, min=0, max=num_classes - 1).cpu()
    area_union = area_pred + area_target - area_intersect
    return dict(
        area_intersect=area_intersect.unsqueeze(0),
        area_union=area_union.unsqueeze(0),
        area_pred=area_pred.unsqueeze(0),
        area_target=area_target.unsqueeze(0),
    )


def total_area_to_metrics(
        total_area_intersect,
        total_area_union,
        total_area_pred,
        total_area_target,
):
    all_acc = total_area_intersect.sum() / total_area_target.sum()
    acc = total_area_intersect / total_area_target
    iou = total_area_intersect / total_area_union
    dice = 2 * total_area_intersect / (total_area_pred + total_area_target)
    return dict(
        all_acc=all_acc,
        acc=acc,
        iou=iou,
        dice=dice,
    )
