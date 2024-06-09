from torch import nn

from ksuit.initializers.base import InitializerBase
from ksuit.utils.select_with_path import select_with_path


class ResetClassifierHeadInitializer(InitializerBase):
    """
    change a classification head to a different number of classes and randomly initialize it
    useful for e.g. fine-tuning a pre-trained model (e.g. loaded via torchhub) on a different task
    e.g. fine-tune a ImageNet-1K pre-trained model on cifar10
    """

    def __init__(self, classifier_path, init="mae", **kwargs):
        super().__init__(**kwargs)
        self.classifier_path = classifier_path
        self.init = init

    def init_weights(self, model):
        assert len(model.output_shape) == 1, f"classification task expects output_shape in tuple (num_classes,)"
        self.logger.info(f"resetting classifier '{self.classifier_path}' to {model.output_shape[0]} classes")
        split = self.classifier_path.split(".")
        prefix = ".".join(split[:-1])
        postfix = split[-1]
        parent = select_with_path(obj=model, path=prefix)
        old_classifier = getattr(parent, postfix)
        input_dim = old_classifier.weight.shape[1]
        setattr(parent, postfix, nn.Linear(input_dim, model.output_shape[0]))
        new_classifier = getattr(parent, postfix)
        if self.init == "mae":
            # following MAE https://github.com/facebookresearch/mae/blob/main/main_finetune.py#L257
            nn.init.trunc_normal_(new_classifier.weight, std=2e-5)
            nn.init.zeros_(new_classifier.bias)
        else:
            raise NotImplementedError
