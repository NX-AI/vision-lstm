import math
from itertools import product

import einops
import torch
import torch.nn.functional as F

from ksuit.distributed import (
    is_distributed,
    get_world_size,
    get_rank,
    all_reduce_sum_nograd,
)


def _preprocess(
        train_x,
        test_x,
        train_y,
        test_y,
        k,
        tau,
        batch_normalize,
        eps,
        batch_size,
        inplace,
):
    # check x and y
    assert len(train_x) == len(train_y), f"{len(train_x)} != {len(train_y)}"
    assert len(test_x) == len(test_y), f"{len(test_x)} != {len(test_y)}"
    assert train_x.ndim == 2 and test_x.ndim == 2
    # check k
    assert isinstance(k, (int, tuple, list)) or k == float("inf")
    ks = [k] if isinstance(k, (int, float)) else k
    assert all((isinstance(k, int) and k >= 1) or k == float("inf") for k in ks)
    # check batch_size
    assert batch_size is None or (batch_size is not None and isinstance(batch_size, int) and batch_size >= 1)
    # check tau
    assert isinstance(tau, (float, tuple, list))
    taus = [tau] if isinstance(tau, float) else tau
    assert all(isinstance(tau, float) and tau > 0 for tau in taus)

    # filter k that are larger than number of train samples
    ks = [k for k in ks if k <= len(train_y) or math.isinf(k)]

    # apply batch normalization
    if batch_normalize:
        mean = train_x.mean(dim=0, keepdim=True)
        std = train_x.std(dim=0, keepdim=True) + eps
        # use inplace operations to be more memory efficient
        if inplace:
            train_x.sub_(mean).div_(std)
            test_x.sub_(mean).div_(std)
        else:
            train_x = (train_x - mean) / std
            test_x = (test_x - mean) / std

    # normalize to length 1 for cosine distance
    if inplace:
        # F.normalize is not memory efficient (it expands denom to the shape of train_x)
        train_x.div_(train_x.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12))
    else:
        train_x = F.normalize(train_x, dim=1)
    test_x = F.normalize(test_x, dim=1)

    # split for distributed
    test_size = len(test_y)
    if is_distributed():
        world_size = get_world_size()
        rank = get_rank()
        test_x = test_x.chunk(world_size)[rank]
        test_y = test_y.chunk(world_size)[rank]
    # calculate in chunks to avoid OOM
    num_chunks = math.ceil(len(test_y) / (batch_size or len(test_y)))

    return train_x, test_x, test_y, ks, taus, test_size, num_chunks


@torch.no_grad()
def multiclass_knn(
        train_x,
        test_x,
        train_y,
        test_y,
        k=10,
        tau=0.07,
        batch_normalize=True,
        eps=1e-6,
        batch_size=None,
        inplace=False,
        mode="accuracy",
):
    train_x, test_x, test_y, ks, taus, test_size, num_chunks = _preprocess(
        train_x=train_x,
        test_x=test_x,
        train_y=train_y,
        test_y=test_y,
        k=k,
        tau=tau,
        batch_normalize=batch_normalize,
        eps=eps,
        batch_size=batch_size,
        inplace=inplace,
    )
    assert train_y.ndim == 1 and test_y.ndim == 1

    # initialize onehot vector per class (used for counting votes in classification)
    num_classes = max(train_y.max(), test_y.max()) + 1
    class_onehot = torch.diag(torch.ones(max(2, num_classes).item(), device=train_x.device))

    if mode == "accuracy":
        num_correct = torch.zeros(len(ks) * len(taus), device=train_x.device)
        preds = None
    elif mode == "predict":
        num_correct = None
        preds = torch.full(size=(len(ks) * len(taus), len(test_y)), fill_value=-1, device=train_x.device)
    else:
        raise NotImplementedError
    test_idx_start = 0
    for test_x_chunk, test_y_chunk in zip(test_x.chunk(num_chunks), test_y.chunk(num_chunks)):
        i = 0
        test_idx_end = test_idx_start + len(test_y_chunk)
        # calculate similarity
        similarities = test_x_chunk @ train_x.T
        for k in ks:
            # retrieve k-nearest-neighbors and their labels
            # in some cases it might be faster if this would be outside the knn loop with sorted=True and knn=max(knns)
            # the trade-off is (not sure how large len(k) has to be for the current version to be slower):
            # - large len(k): sorted=True + knn=max(k) + "topk_indices = ..." outside of knns loop
            # - small len(k): current version
            if math.isinf(k):
                topk_similarities = similarities
                flat_nn_labels = train_y
                reshape_k = len(train_x)
            else:
                topk_similarities, topk_indices = similarities.topk(k=k, dim=1)
                flat_topk_indices = einops.rearrange(topk_indices, "num_test knn -> (num_test knn)")
                flat_nn_labels = train_y[flat_topk_indices]
                reshape_k = k

            # calculate accuracy of a knn classifier
            flat_nn_onehot = class_onehot[flat_nn_labels]
            nn_onehot = einops.rearrange(
                flat_nn_onehot,
                "(num_test k) num_classes -> k num_test num_classes",
                k=reshape_k,
            )
            for tau in taus:
                if math.isinf(tau):
                    # uniform weights
                    logits = nn_onehot.sum(dim=0)
                    knn_classes = logits.argmax(dim=1)
                else:
                    # 0.07 is used as default by DINO/solo-learn
                    # https://github.com/facebookresearch/dino/blob/main/eval_knn.py#L196
                    # https://github.com/vturrisi/solo-learn/blob/main/solo/utils/knn.py#L31
                    cur_topk_similarities = (topk_similarities / tau).exp_()
                    cur_topk_similarities = einops.rearrange(cur_topk_similarities, "num_test knn -> knn num_test 1")
                    logits = (nn_onehot * cur_topk_similarities).sum(dim=0)
                    knn_classes = logits.argmax(dim=1)
                if mode == "accuracy":
                    num_correct[i] += (test_y_chunk == knn_classes).sum()
                elif mode == "predict":
                    preds[i, test_idx_start:test_idx_end] = knn_classes
                else:
                    raise NotImplementedError
                i += 1
        test_idx_start += len(test_y_chunk)

    if mode == "accuracy":
        # convert to accuracy
        if is_distributed():
            num_correct = all_reduce_sum_nograd(num_correct)
        accuracy = num_correct / test_size
        # convert to dictionary of float
        result = {(k, tau): accuracy for (k, tau), accuracy in zip(product(ks, taus), accuracy.tolist())}
    elif mode == "predict":
        result = {(k, tau): preds[i] for i, (k, tau) in enumerate(product(ks, taus))}
    else:
        raise NotImplementedError

    return result
