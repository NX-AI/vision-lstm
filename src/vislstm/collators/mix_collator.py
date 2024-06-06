import torch
from torch.utils.data import default_collate

from ksuit.data.collators import StochasticCollator


class MixCollator(StochasticCollator):
    """
    apply_mode:
    - "batch": apply either all samples in the batch or don't apply
    - "sample": decide for each sample whether or not to apply mixup/cutmix
    lamb_mode:
    - "batch": use the same lambda/bbox for all samples in the batch
    - "sample": sample a lambda/bbox for each sample
    shuffle_mode:
    - "roll": mix sample 0 with sample 1; sample 1 with sample 2; ...
    - "flip": mix sample[0] with sample[-1]; sample[1] with sample[-2]; ... requires even batch_size
    - "random": mix each sample with a randomly drawn other sample
    """

    def __init__(
            self,
            mixup_alpha: float = None,
            cutmix_alpha: float = None,
            mixup_p: float = None,
            cutmix_p: float = None,
            apply_mode: str = "batch",
            lamb_mode: str = "batch",
            shuffle_mode: str = "flip",
            **kwargs,
    ):
        super().__init__(**kwargs)
        # check probabilities
        assert (mixup_p is not None) or (cutmix_p is not None), REQUIRES_MIXUP_P_OR_CUTMIX_P
        # assign 0 probability if one is not specified
        mixup_p = mixup_p or 0.
        cutmix_p = cutmix_p or 0.

        # check probabilities
        assert isinstance(mixup_p, (int, float)) and 0. <= mixup_p <= 1., f"invalid mixup_p {mixup_p}"
        assert isinstance(cutmix_p, (int, float)) and 0. <= cutmix_p <= 1., f"invalid mixup_p {mixup_p}"
        assert 0. < mixup_p + cutmix_p <= 1., f"0 < mixup_p + cutmix_p <= 1 (got {mixup_p + cutmix_p})"
        if mixup_p + cutmix_p != 1.:
            raise NotImplementedError

        # check alphas
        if mixup_p == 0.:
            assert mixup_alpha is None
        else:
            assert isinstance(mixup_alpha, (int, float)) and 0. < mixup_alpha
        if cutmix_p == 0.:
            assert cutmix_alpha is None
        else:
            assert isinstance(cutmix_alpha, (int, float)) and 0. < cutmix_alpha

        # check modes
        assert apply_mode in ["batch", "sample"], f"invalid apply_mode {apply_mode}"
        assert lamb_mode in ["batch", "sample"], f"invalid lamb_mode {lamb_mode}"
        assert shuffle_mode in ["roll", "flip", "random"], f"invalid shuffle_mode {shuffle_mode}"

        # initialize
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_p = mixup_p
        self.cutmix_p = cutmix_p
        self.apply_mode = apply_mode
        self.lamb_mode = lamb_mode
        self.shuffle_mode = shuffle_mode

    @property
    def total_p(self) -> float:
        return self.mixup_p + self.cutmix_p

    def __call__(self, batch):
        if isinstance(batch, (tuple, list)):
            batch = default_collate(batch)
        assert isinstance(batch, dict)
        # extract properties from batch
        x = batch.get("x", None)
        y = batch.get("class", None)
        is_binary_classification = False
        if y is not None:
            y = y.float()
            # y has to be 2d tensor of in one-hot format (multi-class) or 1d tensor (binary-classification)
            if y.ndim != 2:
                assert y.ndim == 1 and 0. <= y.min() and y.max() <= 1., \
                    "MixCollator expects classes to be in one-hot format"
                y = y.unsqueeze(1)
                is_binary_classification = True
        batch_size = len(x)

        # sample apply
        if self.apply_mode == "batch":
            apply = torch.full(size=(batch_size,), fill_value=self.rng.random() < self.total_p)
        elif self.apply_mode == "sample":
            apply = torch.from_numpy(self.rng.random(batch_size)) < self.total_p
        else:
            raise NotImplementedError

        # sample parameters (use_cutmix, lamb, bbox)
        permutation = None
        if self.lamb_mode == "batch":
            use_cutmix = self.rng.random() * self.total_p < self.cutmix_p
            alpha = self.cutmix_alpha if use_cutmix else self.mixup_alpha
            lamb = torch.tensor([self.rng.beta(alpha, alpha)])
            # apply x
            if x is not None:
                x2, permutation = self.shuffle(item=x, permutation=permutation)
                if use_cutmix:
                    h, w = x.shape[2:]
                    bbox, lamb = self.get_random_bbox(h=h, w=w, lamb=lamb)
                    top, left, bot, right = bbox[0]
                    x[..., top:bot, left:right] = x2[..., top:bot, left:right]
                else:
                    # bbox = torch.full(size=(4,), fill_value=-1)
                    x_lamb = lamb.view(-1, *[1] * (x.ndim - 1))
                    x.mul_(x_lamb).add_(x2.mul_(1. - x_lamb))
            # apply y
            if y is not None:
                y2, permutation = self.shuffle(item=y, permutation=permutation)
                y_lamb = lamb.view(-1, 1)
                y.mul_(y_lamb).add_(y2.mul_(1. - y_lamb))
        elif self.lamb_mode == "sample":
            # sample
            use_cutmix = torch.from_numpy(self.rng.random(batch_size) * self.total_p) < self.cutmix_p
            mixup_lamb, cutmix_lamb, bbox = None, None, None
            if self.mixup_p > 0.:
                mixup_lamb = torch.from_numpy(self.rng.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size))
                mixup_lamb = mixup_lamb.float()
            else:
                mixup_lamb = torch.empty(batch_size)
            if self.cutmix_p > 0.:
                cutmix_lamb = torch.from_numpy(self.rng.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size))
                h, w = x.shape[2:]
                bbox, cutmix_lamb = self.get_random_bbox(h=h, w=w, lamb=cutmix_lamb)
                cutmix_lamb = cutmix_lamb.float()
            else:
                cutmix_lamb = torch.empty(batch_size)
            lamb = torch.where(use_cutmix, cutmix_lamb, mixup_lamb)
            # apply
            if x is not None:
                x2_indices, permutation = self.shuffle(item=torch.arange(batch_size), permutation=permutation)
                x_clone = x.clone()
                bbox_idx = 0
                for i in range(batch_size):
                    j = x2_indices[i]
                    if use_cutmix[j]:
                        top, left, bot, right = bbox[bbox_idx]
                        x[i, ..., top:bot, left:right] = x_clone[j, ..., top:bot, left:right]
                        bbox_idx += 1
                    else:
                        x_lamb = lamb[i].view(*[1] * (x.ndim - 1))
                        x[i] = x[i].mul_(x_lamb).add_(x_clone[j].mul_(1 - x_lamb))
            if y is not None:
                y2, permutation = self.shuffle(item=y, permutation=permutation)
                y_lamb = lamb.view(-1, 1)
                y.mul_(y_lamb).add_(y2.mul_(1. - y_lamb))
        else:
            raise NotImplementedError

        # update properties in batch
        if x is not None:
            batch["x"] = x
        if y is not None:
            if is_binary_classification:
                y = y.squeeze(1)
            batch["y"] = y
        return batch

    def get_random_bbox(self, h, w, lamb):
        n_bboxes = len(lamb)
        bbox_hcenter = torch.from_numpy(self.rng.integers(h, size=(n_bboxes,)))
        bbox_wcenter = torch.from_numpy(self.rng.integers(w, size=(n_bboxes,)))

        area_half = 0.5 * (1.0 - lamb).sqrt()
        bbox_h_half = (area_half * h).floor()
        bbox_w_half = (area_half * w).floor()

        top = torch.clamp(bbox_hcenter - bbox_h_half, min=0).type(torch.long)
        bot = torch.clamp(bbox_hcenter + bbox_h_half, max=h).type(torch.long)
        left = torch.clamp(bbox_wcenter - bbox_w_half, min=0).type(torch.long)
        right = torch.clamp(bbox_wcenter + bbox_w_half, max=w).type(torch.long)
        bbox = torch.stack([top, left, bot, right], dim=1)

        lamb_adjusted = 1.0 - (bot - top) * (right - left) / (h * w)

        return bbox, lamb_adjusted

    def shuffle(self, item, permutation):
        if len(item) == 1:
            return item.clone(), None
        if self.shuffle_mode == "roll":
            return item.roll(shifts=1, dims=0), None
        if self.shuffle_mode == "flip":
            assert len(item) % 2 == 0
            return item.flip(0), None
        if self.shuffle_mode == "random":
            if permutation is None:
                permutation = self.rng.permutation(len(item))
            return item[permutation], permutation
        raise NotImplementedError
