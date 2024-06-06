from ksuit.callbacks.base.periodic_callback import PeriodicCallback
from ksuit.utils.formatting_utils import short_number_str
from ksuit.utils.model_utils import get_trainable_param_count, get_frozen_param_count


class CheckpointCallback(PeriodicCallback):
    def __init__(
            self,
            save_weights=True,
            save_optim=False,
            save_latest_weights=False,
            save_latest_optim=False,
            model_name=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        assert save_weights or save_latest_weights or save_optim or save_latest_optim
        self.save_weights = save_weights
        self.save_optim = save_optim
        self.save_latest_weights = save_latest_weights
        self.save_latest_optim = save_latest_optim
        self.model_names = []
        if model_name is not None:
            self.model_names.append(model_name)

    def _before_training(self, model, **kwargs):
        frozen_count = get_frozen_param_count(model)
        trainable_count = get_trainable_param_count(model)

        weight_bytes = (frozen_count + trainable_count) * 4
        self.logger.info(f"estimated checkpoint size: {short_number_str(weight_bytes * 3)}B")
        self.logger.info(f"estimated weight checkpoint size: {short_number_str(weight_bytes)}B")
        # hardcoded for adam/adamw (SGD would have lower size)
        self.logger.info(f"estimated optim checkpoint size: {short_number_str(weight_bytes * 2)}B")

        # (not 100% accurate...multiple intervals are not considered)
        num_checkpoints = 1
        if self.every_n_epochs is not None:
            num_checkpoints += self.update_counter.end_checkpoint.epoch // self.every_n_epochs
        if self.every_n_updates is not None:
            num_checkpoints += int(self.update_counter.end_checkpoint.update / self.every_n_updates)
        if self.every_n_samples is not None:
            num_checkpoints += int(self.update_counter.end_checkpoint.sample / self.every_n_samples)
        multiplier = 0
        if self.save_weights:
            multiplier += 1
        if self.save_optim:
            multiplier += 2
        self.logger.info(
            f"estimated size for {num_checkpoints} checkpoints: "
            f"{short_number_str(num_checkpoints * weight_bytes * multiplier)}B"
        )

    # noinspection PyMethodOverriding
    def _periodic_callback(self, model, trainer, **kwargs):
        self.checkpoint_writer.save(
            model=model,
            trainer=trainer,
            checkpoint=self.update_counter.cur_checkpoint,
            save_weights=self.save_weights,
            save_optim=self.save_optim,
            save_latest_weights=self.save_latest_weights,
            save_latest_optim=self.save_latest_optim,
            model_names_to_save=self.model_names,
            save_frozen_weights=True,
        )

    def _after_training(self, model, trainer, **kwargs):
        self.checkpoint_writer.save(
            model=model,
            trainer=trainer,
            checkpoint="last",
            save_weights=self.save_weights,
            save_optim=self.save_optim,
            save_latest_weights=self.save_latest_weights,
            save_latest_optim=self.save_latest_optim,
            model_names_to_save=self.model_names,
            save_frozen_weights=True,
        )
