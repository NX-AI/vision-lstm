from collections import defaultdict

import torch

from ksuit.callbacks.base import PeriodicCallback
from ksuit.distributed import is_rank0
from ksuit.utils.checkpoint import Checkpoint
from ksuit.utils.select_with_path import select_with_path


class EmaCallback(PeriodicCallback):
    def __init__(
            self,
            target_factors,
            model_paths=None,
            save_weights=True,
            save_last_weights=True,
            save_latest_weights=False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_paths = model_paths or [None]
        self.target_factors = target_factors
        self.save_weights = save_weights
        self.save_last_weights = save_last_weights
        self.save_latest_weights = save_latest_weights
        self.parameters = defaultdict(dict)
        self.buffers = defaultdict(dict)
        self._was_resumed = False

    def resume_from_checkpoint(self, stage_name, stage_id, model):
        for model_path in self.model_paths:
            cur_model = select_with_path(obj=model, path=model_path)
            if model_path is None:
                model_name_with_path = model.name
            else:
                model_name_with_path = f"{model.name}.{model_path}"
            for target_factor in self.target_factors:
                sd = torch.load(
                    self.path_provider.get_stage_output_path(stage_name=stage_name, stage_id=stage_id)
                    / "checkpoints"
                    / f"{model_name_with_path} cp=latest ema={target_factor} model.th"
                )["state_dict"]
                for name, _ in cur_model.named_parameters():
                    self.parameters[(model_path, target_factor)][name] = sd[name]
                for name, _ in cur_model.named_buffers():
                    self.buffers[model_path][name] = sd[name]
        self._was_resumed = True

    def _before_training(self, model, **kwargs):
        if not is_rank0():
            return
        if self._was_resumed:
            return
        for model_path in self.model_paths:
            cur_model = select_with_path(obj=model, path=model_path)
            for target_factor in self.target_factors:
                for name, param in cur_model.named_parameters():
                    self.parameters[(model_path, target_factor)][name] = param.clone()
            for name, buffer in cur_model.named_buffers():
                self.buffers[model_path][name] = buffer.clone()

    def _track_after_update_step(self, model, **kwargs):
        if not is_rank0():
            return
        for model_path in self.model_paths:
            cur_model = select_with_path(obj=model, path=model_path)
            for target_factor in self.target_factors:
                key = (model_path, target_factor)
                # fused inplace implementation
                target_param_list = list(self.parameters[key].values())
                source_param_list = list(cur_model.parameters())
                # noinspection PyProtectedMember
                torch._foreach_mul_(target_param_list, target_factor)
                # noinspection PyProtectedMember
                torch._foreach_add_(target_param_list, source_param_list, alpha=1 - target_factor)

                # basic implementation
                # for name, param in cur_model.named_parameters():
                #     self.parameters[key][name].mul_(target_factor).add_(param, alpha=1. - target_factor)
            for name, buffer in cur_model.named_buffers():
                self.buffers[model_path][name].copy_(buffer)

    def _save(self, ckpt, model):
        if not is_rank0():
            return
        ckpt_str = str(ckpt)
        for model_path in self.model_paths:
            cur_model = select_with_path(obj=model, path=model_path)
            for target_factor in self.target_factors:
                state_dict = {**self.parameters[(model_path, target_factor)], **self.buffers[model_path]}
                ckpt_dict = dict(
                    state_dict=state_dict,
                    ctor_kwargs=cur_model.ctor_kwargs,
                    abs_ckpt=dict(self.update_counter.cur_checkpoint),
                    stage_id=self.path_provider.stage_id,
                    ema=target_factor,
                )
                if model_path is None:
                    cur_model_path = model.name
                else:
                    cur_model_path = f"{model.name}.{model_path}"
                if self.save_weights:
                    fname = f"{cur_model_path} cp={ckpt_str} ema={target_factor} model.th"
                    torch.save(ckpt_dict, self.path_provider.checkpoint_path / fname)
                if self.save_latest_weights:
                    fname = f"{cur_model_path} cp=latest ema={target_factor} model.th"
                    torch.save(ckpt_dict, self.path_provider.checkpoint_path / fname)

    def _periodic_callback(self, model, **kwargs):
        self._save(ckpt=self.update_counter.cur_checkpoint, model=model)

    def _after_training(self, model, **kwargs):
        if self.save_last_weights:
            self._save(ckpt="last", model=model)
