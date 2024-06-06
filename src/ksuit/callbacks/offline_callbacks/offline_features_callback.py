from functools import partial

import torch

from ksuit.callbacks.base.periodic_callback import PeriodicCallback
from ksuit.data.wrappers import ModeWrapper
from ksuit.factory import MasterFactory
from ksuit.models.extractors.base.forward_hook import StopForwardException


class OfflineFeaturesCallback(PeriodicCallback):
    def __init__(self, dataset_key, extractors, forward_kwargs=None, items_to_store=None, **kwargs):
        super().__init__(**kwargs)
        self.dataset_key = dataset_key
        self.extractors = extractors
        self.forward_kwargs = MasterFactory.create_dict(forward_kwargs)
        self.items_to_store = items_to_store or []
        self.__config_id = None
        self.dataset_mode = None
        self.out = self.path_provider.stage_output_path / "features"

    def _register_sampler_configs(self, trainer):
        self.dataset_mode = trainer.dataset_mode
        for item in self.items_to_store:
            self.dataset_mode = ModeWrapper.add_item(mode=trainer.dataset_mode, item=item)
        self.__config_id = self._register_sampler_config_from_key(key=self.dataset_key, mode=self.dataset_mode)

    def _before_training(self, model, **kwargs):
        self.out.mkdir(exist_ok=True)
        self.extractors = MasterFactory.get("extractor").create_list(self.extractors, static_ctx=model.static_ctx)
        for extractor in self.extractors:
            extractor.register_hooks(model)
            extractor.disable_hooks()

    def _forward(self, batch, trainer_model, trainer):
        try:
            trainer.update(
                ddp_model=trainer_model,
                batch=batch,
                training=False,
                forward_kwargs=self.forward_kwargs,
            )
        except StopForwardException:
            pass
        features = {}
        for extractor in self.extractors:
            features[str(extractor)] = extractor.extract().cpu()
        items = {item: batch[item].clone() for item in self.items_to_store}
        return features, items

    # noinspection PyMethodOverriding
    def _periodic_callback(self, model, trainer_model, trainer, batch_size, data_iter, **_):
        # setup
        for extractor in self.extractors:
            extractor.enable_hooks()

        # foward
        features, items = self.iterate_over_dataset(
            forward_fn=partial(self._forward, trainer_model=trainer_model, trainer=trainer),
            config_id=self.__config_id,
            batch_size=batch_size,
            data_iter=data_iter,
        )

        # store
        for key, value in features.items():
            features_uri = self.out / f"{self.update_counter.cur_checkpoint}_features_{key}.th"
            self.logger.info(f"saving features to {features_uri}")
            torch.save(value, features_uri)
        for item in self.items_to_store:
            item_uri = self.out / f"{self.update_counter.cur_checkpoint}_{item}.th"
            self.logger.info(f"saving {item} to {item_uri}")
            torch.save(items[item], self.out / item_uri)

        # cleanup
        for extractor in self.extractors:
            extractor.disable_hooks()
