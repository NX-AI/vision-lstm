import logging

import torch
from torch.cuda.amp import GradScaler

from ksuit.factory import MasterFactory
from ksuit.utils.amp_utils import NoopGradScaler
from ksuit.utils.bidict import Bidict
from ksuit.utils.formatting_utils import float_to_scientific_notation


class OptimWrapper:
    """
    wrapper for torch optimizers that also handles
    - learning rate scaling (with batchsize)
    - creating parameter groups (e.g. excluding bias/norm from weight decay, layerwise lr scaling)
    - stateless learning rate scheduling
    - gradient clipping
    """

    def __init__(
            self,
            model,
            torch_optim_ctor,
            schedule=None,
            weight_decay_schedule=None,
            clip_grad_value=None,
            clip_grad_norm=None,
            param_group_modifiers=None,
            exclude_bias_from_wd=True,
            exclude_norm_from_wd=True,
            update_counter=None,
            lr_scale_factor=None,
            lr_scaler=None,
    ):
        self.logger = logging.getLogger(type(self).__name__)
        self.model = model
        self.update_counter = update_counter
        self.clip_grad_value = clip_grad_value
        self.clip_grad_norm = clip_grad_norm
        assert self.clip_grad_value is None or self.clip_grad_value > 0
        assert self.clip_grad_norm is None or self.clip_grad_norm > 0

        # scale lr
        assert "lr" in torch_optim_ctor.keywords, f"no learning rate specified for optimizer"
        lr_scaler = MasterFactory.get("lr_scaler").create(lr_scaler)
        base_lr = torch_optim_ctor.keywords["lr"]
        self.logger.info(f"base lr: {float_to_scientific_notation(base_lr, max_precision=2)}")
        if lr_scaler is not None:
            lr_scale_factor = lr_scale_factor or update_counter.effective_batch_size
            scaled_lr = lr_scaler.scale_lr(base_lr=base_lr, lr_scale_factor=lr_scale_factor)
            self.logger.info(f"scaled lr: {float_to_scientific_notation(scaled_lr, max_precision=2)}")
            self.logger.info(f"lr_scaler={lr_scaler}")
            self.logger.info(f"lr_scale_factor={lr_scale_factor}")
            torch_optim_ctor.keywords["lr"] = scaled_lr
        else:
            self.logger.warning(f"no lr_scaler defined -> typically learning rate is scaled with the batch_size")

        # create a param group for each parameter
        param_group_modifiers = MasterFactory.get("param_group_modifier").create_list(param_group_modifiers)
        if hasattr(model, "get_param_group_modifiers"):
            param_group_modifiers = model.get_param_group_modifiers() + param_group_modifiers
        param_groups = []
        self.logger.info(
            f"exclude_bias_from_wd={exclude_bias_from_wd} exclude_norm_from_wd={exclude_norm_from_wd} "
            f"param_group_modifiers=[{' '.join(str(pgm) for pgm in param_group_modifiers)}]"
        )
        for name, param in model.named_parameters():
            properties = {}
            # excluding norm and bias params is very common for all models -> support with simple flag
            # bias has ndim == 1, so it needs to be checked before
            # the bias of norm layers is considered a bias, not a norm parameter
            if name.split(".")[-1] == "bias" and exclude_bias_from_wd:
                properties["weight_decay"] = 0.
            # timm does it like this...not sure if other parameters can also have ndim <= 1
            # https://github.com/rwightman/pytorch-image-models/blob/master/timm/optim/optim_factory.py
            elif param.ndim <= 1 and exclude_norm_from_wd and name.split(".")[-1] != "bias":
                properties["weight_decay"] = 0.

            for param_group_modifier in param_group_modifiers:
                for key, value in param_group_modifier.get_properties(model, name, param).items():
                    if key in properties and key == "lr_scale":
                        properties[key] *= value
                    else:
                        assert key not in properties
                        properties[key] = value
            assert "param" not in properties
            assert "name" not in properties
            properties["name"] = name
            properties["params"] = [param]
            param_groups.append(properties)

        # check that param group modifiers were successfully applied (e.g. check that param name was found in model)
        for param_group_modifier in param_group_modifiers:
            assert param_group_modifier.was_applied_successfully(), f"{param_group_modifier} failed"

        # merge same groups with same parameters (useful for logging)
        merged_groups = []
        merged_groups_properties = []
        merged_groups_paramnames = []
        for param_group in param_groups:
            param_name = param_group.pop("name")
            properties = {k: v for k, v in param_group.items() if k != "params"}
            matching_group_idx = None
            for i, merged_group_properties in enumerate(merged_groups_properties):
                if properties == merged_group_properties:
                    matching_group_idx = i
                    break
            if matching_group_idx is None:
                merged_groups.append(param_group)
                merged_groups_properties.append(properties)
                merged_groups_paramnames.append([param_name])
            else:
                merged_groups[matching_group_idx]["params"] += param_group["params"]
                merged_groups_paramnames[matching_group_idx].append(param_name)

        # add name to param_groups
        for param_group in merged_groups:
            names = []
            for key, value in param_group.items():
                if key == "params":
                    continue
                if isinstance(value, float):
                    value_str = float_to_scientific_notation(value, max_precision=1, remove_plus=True)
                else:
                    raise NotImplementedError
                names.append(f"{key}={value_str}")
            if len(names) == 0:
                param_group["name"] = "default"
            else:
                param_group["name"] = "&".join(names)

        # log param groups
        self.logger.info(f"using {len(merged_groups)} param groups:")
        for param_group in merged_groups:
            self.logger.info(
                " ".join(
                    [
                        f"{key}={value}" for key, value in param_group.items()
                        if key not in ["params", "name"]
                    ] + [f"len(params)={len(param_group['params'])}"]
                )
            )

        # torch optimizer organizes parameters by enumerating them (not by name)
        # so for loading an arbitrary optim state_dict an association from param_name to param_idx has to be stored
        self.param_idx_to_name = Bidict()
        idx = 0
        for group_paramnames in merged_groups_paramnames:
            for param_name in group_paramnames:
                self.param_idx_to_name.set_forward(idx, param_name)
                idx += 1

        # initialize torch optim
        self.torch_optim = torch_optim_ctor(merged_groups)

        # for grad clipping all parameters of the optimizer are required
        self.all_parameters = None
        if self.clip_grad_value is not None or self.clip_grad_norm is not None:
            self.all_parameters = list(model.parameters())

        # scale lr (e.g. layerwise_lr_decay_modifier)
        for param_group in self.torch_optim.param_groups:
            if "lr_scale" in param_group:
                assert "original_lr" not in param_group
                param_group["original_lr"] = param_group["lr"]
                # lr is float so inplace operation is fine
                # this scaling is only relevant for logging and epoch based schedules
                # for update based schedule the value is anyway scaled again at the start of the update
                param_group["lr"] *= param_group["lr_scale"]
                self.logger.info(
                    f"scaled lr of param_group '{param_group['name']}' "
                    f"from {float_to_scientific_notation(param_group['original_lr'], max_precision=2)} "
                    f"to {float_to_scientific_notation(param_group['lr'], max_precision=2)}"
                )

        # create schedules
        self.schedule = MasterFactory.get("schedule").create(
            schedule,
            update_counter=self.update_counter,
            max_value=self.torch_optim.defaults["lr"],
        )
        self.weight_decay_schedule = MasterFactory.get("schedule").create(
            weight_decay_schedule,
            update_counter=self.update_counter,
            max_value=self.torch_optim.defaults["weight_decay"]
        )
        # store initial_lr/initial_wd in param_groups
        # NOTE: torch optimizer broadcasts all values to all param groups (so every param_group has a weight_decay)
        if self.weight_decay_schedule is not None:
            for param_group in self.torch_optim.param_groups:
                assert "exclude_from_wd" not in param_group
                param_group["exclude_from_wd"] = param_group["weight_decay"] == 0.

    def _has_param_with_grad(self):
        for param_group in self.torch_optim.param_groups:
            for p in param_group["params"]:
                if p.grad is not None:
                    return True
        return False

    def step(self, grad_scaler=None):
        # grad_scaler doesnt support update without gradient (e.g. GAN setting)
        # Error: AssertionError: No inf checks were recorded for this optimizer
        if isinstance(grad_scaler, GradScaler):
            if not self._has_param_with_grad():
                return

        # grad scaler is not strictly required
        # (e.g. if OptimWrapper is only used for excluding bias/norm parameters from weight decay)
        if grad_scaler is None:
            grad_scaler = NoopGradScaler()

        # NOTE: closure is not supported with GradScaler
        if self.clip_grad_value is not None or self.clip_grad_norm is not None:
            grad_scaler.unscale_(self.torch_optim)
        # clip gradients
        if self.clip_grad_value is not None:
            torch.nn.utils.clip_grad_value_(self.all_parameters, self.clip_grad_value)
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.all_parameters, self.clip_grad_norm)
        # torch optim step with grad scaler
        grad_scaler.step(self.torch_optim)
        grad_scaler.update()

    def schedule_step(self):
        if self.schedule is not None:
            lr_scale = self.schedule.get_value()
            for param_group in self.torch_optim.param_groups:
                if "lr_scale" in param_group:
                    # lr_scale -> current lr from schedule
                    # param_group["lr_scale"] -> scale form layer-wise lr decay
                    param_group["lr"] = param_group["lr_scale"] * lr_scale
                else:
                    param_group["lr"] = lr_scale
        if self.weight_decay_schedule is not None:
            wd_scale = self.weight_decay_schedule.get_value()
            for param_group in self.torch_optim.param_groups:
                if not param_group["exclude_from_wd"]:
                    param_group["weight_decay"] = wd_scale

    def zero_grad(self, set_to_none=True):
        # set_to_none is True by default (unlike torch.optim.optimizer)
        # because it has better performance (https://www.youtube.com/watch?v=9mS1fIYj1So)
        self.torch_optim.zero_grad(set_to_none)

    def state_dict(self):
        sd = self.torch_optim.state_dict()
        sd["param_idx_to_name"] = self.param_idx_to_name.to_forward()
        return sd

    def load_state_dict(self, state_dict_to_load):
        # torch optim state_dict stores param_groups and the states of each parameter
        # if a torch optim state_dict is loaded it would overwrite all param_groups from the checkpoint
        # this results in unexpected behavior when loading an optimizer (e.g. for resuming a run from a checkpoint)
        # - add new parameters (e.g. unfreeze something)
        # - change weight_decay or other param_group properties: the load_state_dict would overwrite the actual
        #   weight_decay with the weight_decay from the checkpoint
        if "param_idx_to_name" in state_dict_to_load:
            # torch optim stores:
            # - a list of param_idxs in each param_group
            # - a dict from param_idxs to state for the state of the param
            # -> match the param_idxs and overwrite the state
            loaded_param_idx_to_name = Bidict(forward=state_dict_to_load["param_idx_to_name"])
            loaded_states = state_dict_to_load["state"]
            cur_state_dict = self.torch_optim.state_dict()
            cur_states = cur_state_dict["state"]
            cur_param_groups = cur_state_dict["param_groups"]
            for cur_param_group in cur_param_groups:
                for cur_param_idx in cur_param_group["params"]:
                    param_name = self.param_idx_to_name.get_forward(cur_param_idx)
                    loaded_param_idx = loaded_param_idx_to_name.get_backward(param_name)
                    if loaded_param_idx not in loaded_states:
                        # if no optim step was done no state exists -> dont load the state
                        cur_states.pop(loaded_param_idx, None)
                    else:
                        # overwrite state with loaded state
                        cur_states[cur_param_idx] = loaded_states[loaded_param_idx]
            state_dict_to_load = dict(state=cur_states, param_groups=cur_param_groups)
        self.torch_optim.load_state_dict(state_dict_to_load)
