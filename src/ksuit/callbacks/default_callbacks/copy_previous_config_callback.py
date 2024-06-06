from ksuit.callbacks.base.callback_base import CallbackBase
from ksuit.initializers import PreviousRunInitializer
from ksuit.models import CompositeModel
from ksuit.utils.model_utils import get_named_models


class CopyPreviousConfigCallback(CallbackBase):
    @staticmethod
    def _should_include_key(key):
        # exclude irrelevant stuff (e.g. device or dataloader params are irrelevant)
        if key in ["stage_name"]:
            return False
        # dependent on the hardware
        if key in ["device", "trainer/accumulation_steps", "trainer/batch_size"]:
            return False
        if key.startswith("dataloader/") or key.startswith("dist/") or key.startswith("code/"):
            return False
        return True

    def _before_training(self, model, **_):
        # collect configs
        configs = {}
        for model_name, model in get_named_models(model).items():
            if isinstance(model, CompositeModel):
                continue
            for initializer in model.initializers:
                if not isinstance(initializer, PreviousRunInitializer):
                    continue
                config = self.config_provider.get_config_of_previous_stage(
                    stage_name=initializer.stage_name,
                    stage_id=initializer.stage_id,
                )
                if config is None:
                    self.logger.info(
                        f"no config found for initializer of {model_name} (stage_name='{initializer.stage_name}' "
                        f"stage_id={initializer.stage_id}) -> don't copy anything"
                    )
                    continue
                if initializer.stage_name in configs:
                    self.logger.info(
                        f"duplicate stage_name when copying configs from {PreviousRunInitializer.__name__} "
                        "-> using first config"
                    )
                    if config != configs[initializer.stage_name]:
                        self.logger.warning(f"configs are not the same -> only first configs is copied")
                    continue
                configs[initializer.stage_name] = config

        # add to config
        for previous_stage_name, config in configs.items():
            # check validity of previous_stage_name
            if previous_stage_name in self.config_provider:
                self.logger.warning(f"'{previous_stage_name}' already exists in config_provider -> skip copying")
                continue
            # filter unnecessary keys
            config = {k: v for k, v in config.items() if self._should_include_key(k)}
            self.config_provider[previous_stage_name] = config
