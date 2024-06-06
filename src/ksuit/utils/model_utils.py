import torch


# noinspection PyProtectedMember


def get_nograd_paramnames(model):
    return [name for name, param in model.named_parameters() if param.grad is None and param.requires_grad]


@torch.no_grad()
def copy_params(source_model, target_model):
    for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
        target_param.copy_(source_param)
    for target_buffer, source_buffer in zip(target_model.buffers(), source_model.buffers()):
        target_buffer.copy_(source_buffer)


@torch.no_grad()
def update_ema(source_model, target_model, target_factor, copy_buffers=False):
    # basic inplace implementation
    # for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
    #     target_param.mul_(target_factor).add_(source_param, alpha=1. - target_factor)

    # fused inplace implementation
    target_param_list = list(target_model.parameters())
    source_param_list = list(source_model.parameters())
    # noinspection PyProtectedMember
    torch._foreach_mul_(target_param_list, target_factor)
    # noinspection PyProtectedMember
    torch._foreach_add_(target_param_list, source_param_list, alpha=1 - target_factor)

    if copy_buffers:
        for target_buffer, source_buffer in zip(target_model.buffers(), source_model.buffers()):
            target_buffer.copy_(source_buffer)


def get_named_models(model):
    submodels = model.submodels
    if len(submodels) == 1:
        # single model
        return submodels
    else:
        # composite model
        result = {}
        for name, submodel in submodels.items():
            if submodel is None:
                continue
            named_submodels = get_named_models(submodel)
            for key, value in named_submodels.items():
                result[f"{name}.{key}"] = value
        return result


def get_param_count(model):
    return sum(p.numel() for p in model.parameters())


def get_trainable_param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_frozen_param_count(model):
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze(model):
    for p in model.parameters():
        p.requires_grad = True
