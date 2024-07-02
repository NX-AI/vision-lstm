from ksuit.distributed import get_world_size
from ksuit.factory import MasterFactory
from ksuit.models import CompositeModel


class UpernetModel(CompositeModel):
    def __init__(
            self,
            encoder,
            extractor,
            postprocessor,
            decoder,
            auxiliary=None,
            auxiliary_feature_idx=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.auxiliary_feature_idx = auxiliary_feature_idx
        # encoder
        self.encoder = MasterFactory.get("model").create(
            encoder,
            input_shape=self.input_shape,
            update_counter=self.update_counter,
            path_provider=self.path_provider,
            dynamic_ctx=self.dynamic_ctx,
            static_ctx=self.static_ctx,
        )
        # extractor
        self.extractor = MasterFactory.get("extractor").create(
            extractor,
            static_ctx=self.encoder.static_ctx,
        )
        self.extractor.register_hooks(self.encoder)
        self.extractor.disable_hooks()
        # postprocessor
        self.postprocessor = MasterFactory.get("model").create(
            postprocessor,
            input_dim=self.encoder.dim,
            input_shape=self.encoder.output_shape,
            output_shape=self.output_shape,
            update_counter=self.update_counter,
            path_provider=self.path_provider,
            dynamic_ctx=self.dynamic_ctx,
            static_ctx=self.static_ctx,
        )
        # decoder
        self.decoder = MasterFactory.get("model").create(
            decoder,
            input_dim=self.encoder.dim,
            input_shape=self.encoder.output_shape,
            output_shape=self.output_shape,
            update_counter=self.update_counter,
            path_provider=self.path_provider,
            dynamic_ctx=self.dynamic_ctx,
            static_ctx=self.static_ctx,
        )
        # auxiliary
        self.auxiliary = MasterFactory.get("model").create(
            auxiliary,
            input_dim=self.encoder.dim,
            input_shape=self.encoder.output_shape,
            output_shape=self.output_shape,
            update_counter=self.update_counter,
            path_provider=self.path_provider,
            dynamic_ctx=self.dynamic_ctx,
            static_ctx=self.static_ctx,
        )

    @property
    def submodels(self):
        return dict(
            encoder=self.encoder,
            postprocessor=self.postprocessor,
            decoder=self.decoder,
            **({} if self.auxiliary is None else dict(auxiliary=self.auxiliary)),
        )

    def forward(self, x):
        assert get_world_size() == 1, "UperNet had much worse results when training on 8 GPUs, no idea why"
        # encoder
        with self.extractor:
            _ = self.encoder(x)

        # postprocessor
        features = self.extractor.extract()
        features = self.postprocessor(features)

        # decoder
        decoder_pred = self.decoder(features)
        preds = dict(decoder=decoder_pred)

        # auxiliary
        if self.auxiliary is not None:
            auxiliary_pred = self.auxiliary(features[self.auxiliary_feature_idx])
            preds["auxiliary"] = auxiliary_pred

        return preds

    def segment(self, x):
        return self.forward(x)["decoder"]
