from .base import CheckpointInitializer


class PreviousRunInitializer(CheckpointInitializer):
    """
    initializes a model from a checkpoint of a previous run (specified by the stage_id)
    load_optim=False as this is usually used for frozen/pretrained models
    """

    def __init__(
            self,
            load_optim=False,
            keys_to_remove=None,
            patterns_to_remove=None,
            patterns_to_rename=None,
            **kwargs,
    ):
        super().__init__(load_optim=load_optim, **kwargs)
        self.keys_to_remove = keys_to_remove or []
        self.patterns_to_remove = patterns_to_remove or []
        self.patterns_to_rename = patterns_to_rename or []

    def init_weights(self, model):
        sd, model_name, ckpt_uri = self._get_model_state_dict(model)
        if len(self.keys_to_remove) > 0:
            self.logger.info(f"removing keys {self.keys_to_remove} from {ckpt_uri}")
            for key in self.keys_to_remove:
                sd.pop(key)
        if len(self.patterns_to_remove) > 0:
            for pattern in self.patterns_to_remove:
                self.logger.info(f"removing pattern {pattern} from {ckpt_uri}")
                for key in list(sd.keys()):
                    if pattern in key:
                        self.logger.info(f"removing key {key}")
                        sd.pop(key)
        if len(self.patterns_to_rename) > 0:
            for pattern in self.patterns_to_rename:
                src_pattern = pattern["src"]
                dst_pattern = pattern["dst"]
                self.logger.info(f"renaming pattern {src_pattern} to {dst_pattern} in {ckpt_uri}")
                for key in list(sd.keys()):
                    if src_pattern in key:
                        new_value = sd.pop(key)
                        dst_key = key.replace(src_pattern, dst_pattern)
                        if dst_key in sd:
                            self.logger.info(f"overwriting key {dst_key} with {key}")
                        else:
                            self.logger.info(f"renaming key {key} to {dst_key}")
                        sd[dst_key] = new_value

        model.load_state_dict(sd)
        self.logger.info(f"loaded weights of {model_name} from {ckpt_uri}")
