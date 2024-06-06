from functools import partial

from ksuit.metrics import multiclass_knn

from ksuit.callbacks import PeriodicCallback
from ksuit.data.wrappers import ModeWrapper
from ksuit.factory import MasterFactory
from ksuit.models.extractors import StopForwardException
from ksuit.utils.formatting_utils import dict_to_string


class OfflineMulticlassKnnCallback(PeriodicCallback):
    def __init__(
            self,
            train_dataset_key,
            test_dataset_key,
            extractors,
            knns=None,
            taus=None,
            forward_kwargs=None,
            inplace=False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.train_dataset_key = train_dataset_key
        self.test_dataset_key = test_dataset_key
        self.extractors = extractors
        self.forward_kwargs = MasterFactory.create_dict(forward_kwargs)
        self.knns = knns or [10]
        self.taus = taus or [0.07]
        self.inplace = inplace
        self.__train_config_id = None
        self.__test_config_id = None
        self.__dataset_mode = None
        self._num_classes = None

    def _register_sampler_configs(self, trainer):
        dataset_mode = trainer.dataset_mode
        dataset_mode = ModeWrapper.add_item(mode=dataset_mode, item="class")
        self.__dataset_mode = dataset_mode
        self.__train_config_id = self._register_sampler_config_from_key(key=self.train_dataset_key, mode=dataset_mode)
        self.__test_config_id = self._register_sampler_config_from_key(key=self.test_dataset_key, mode=dataset_mode)

    def _before_training(self, model, **kwargs):
        # num_classes
        class_shape = self.data_container.get_dataset(self.test_dataset_key).getshape_class()
        assert len(class_shape) == 1
        self._num_classes = class_shape[0]

        # extractors
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
        return features, batch["class"].clone()

    # noinspection PyMethodOverriding
    def _periodic_callback(self, model, trainer_model, trainer, batch_size, data_iter, **_):
        # setup
        for extractor in self.extractors:
            extractor.enable_hooks()

        # train_dataset foward
        train_features, train_y = self.iterate_over_dataset(
            forward_fn=partial(self._forward, trainer_model=trainer_model, trainer=trainer),
            config_id=self.__train_config_id,
            batch_size=batch_size,
            data_iter=data_iter,
        )
        # test_dataset forward
        test_features, test_y = self.iterate_over_dataset(
            forward_fn=partial(self._forward, trainer_model=trainer_model, trainer=trainer),
            config_id=self.__test_config_id,
            batch_size=batch_size,
            data_iter=data_iter,
        )

        # calculate/log metrics
        train_y = train_y.to(model.device)
        test_y = test_y.to(model.device)
        # check that len(train_features) == len(train_y) -> raises error when 2 views are propagated
        assert all(len(v) == len(train_y) for v in train_features.values())
        for feature_key in train_features.keys():
            train_x = train_features[feature_key].to(model.device)
            test_x = test_features[feature_key].to(model.device)

            knn_kwargs = dict(
                train_x=train_x,
                test_x=test_x,
                train_y=train_y,
                test_y=test_y,
                k=self.knns,
                tau=self.taus,
                batch_size=min(1024, batch_size),
                inplace=self.inplace,
            )
            forward_kwargs_str = f"/{dict_to_string(self.forward_kwargs)}" if len(self.forward_kwargs) > 0 else ""
            if train_y.ndim == 1:
                # multiclass
                accuracies = multiclass_knn(**knn_kwargs)
                for metric_key in accuracies.keys():
                    k, tau = metric_key
                    key = (
                        f"knn_accuracy/k{k}-tau{str(tau).replace('.', '')}/{feature_key}/"
                        f"{self.train_dataset_key}-{self.test_dataset_key}"
                        f"{forward_kwargs_str}"
                    )
                    self.writer.add_scalar(key, accuracies[metric_key], logger=self.logger, format_str=".6f")
            else:
                raise NotImplementedError

        # cleanup
        for extractor in self.extractors:
            extractor.disable_hooks()
