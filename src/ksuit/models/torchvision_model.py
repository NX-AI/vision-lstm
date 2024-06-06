import torchvision.models
from torch import nn

from .base import SingleModel


class TorchvisionModel(SingleModel):
    def __init__(self, model_name, model_kwargs=None, input_shape=None, output_shape=None, **kwargs):
        super().__init__(input_shape=input_shape, output_shape=output_shape, **kwargs)
        model_kwargs = model_kwargs or {}
        if input_shape is not None:
            if input_shape != (3, 224, 224) and type(self) == TorchvisionModel:
                self.logger.warning(
                    f"input_shape to torchvision model is not (3, 224, 224) but {input_shape} "
                    f"-> might lead to complications"
                )
        if output_shape is not None:
            # classification
            assert len(output_shape) == 1, "TorchvisionModel only supports classification or feature extraction"
            assert "num_classes" not in model_kwargs
            num_classes = output_shape[0]
            model_kwargs["num_classes"] = num_classes
            self.logger.info(f"using TorchvisionModel as classifier with {num_classes} classes")
        ctor = getattr(torchvision.models, model_name)
        self.model = ctor(**(model_kwargs or {}))
        # remove head for feature extraction
        if output_shape is None:
            self.logger.info(f"using TorchvisionModel as feature extractor -> remove classification head")
            if (
                    "resnet" in model_name
                    or model_name.startswith("shufflenet_v2")
                    or model_name in ["inception_v3", "googlenet"]
            ):
                self.model.fc = nn.Identity()
            elif (
                    model_name.startswith("vgg")
                    or model_name == "alexnet"
                    or model_name == "mobilenet_v2"
                    or model_name.startswith("densenet")
                    or model_name.startswith("mobilenet_v3")
                    or model_name.startswith("mnasnet")
            ):
                self.model.classifier = nn.Identity()
            else:
                raise NotImplementedError

    def forward(self, x):
        return self.model(x)

    def classify(self, x):
        assert self.output_shape is not None, "torchvision_model was not initialized as classifier"
        return dict(main=self.model(x))
