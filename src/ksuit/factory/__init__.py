from .base import SingleFactory, MasterFactory
from .dataset_factory import DatasetFactory
from .model_factory import ModelFactory
from .optim_factory import OptimFactory
from .schedule_factory import ScheduleFactory
from .transform_factory import TransformFactory


def _setup_master_factory():
    # callback
    MasterFactory.set(
        "callback",
        SingleFactory(
            module_names=[
                "ksuit.callbacks.checkpoint_callbacks.{kind}",
                "ksuit.callbacks.default_callbacks.{kind}",
                "ksuit.callbacks.monitor_callbacks.{kind}",
                "ksuit.callbacks.offline_callbacks.{kind}",
                "ksuit.callbacks.online_callbacks.{kind}",
                "ksuit.callbacks.visualization_callbacks.{kind}",
            ]
        ),
    )
    # data
    MasterFactory.set(
        "transform",
        TransformFactory(
            module_names=[
                "ksuit.data.transforms",
                "torchvision.transforms",
            ],
        ),
    )
    MasterFactory.set("collator", SingleFactory(module_names=["ksuit.data.collators.{kind}"]))
    dataset_wrapper_factory = SingleFactory(module_names=["ksuit.data.wrappers.dataset_wrappers.{kind}"])
    MasterFactory.set("dataset_wrapper", dataset_wrapper_factory)
    sample_wrapper_factory = SingleFactory(module_names=["ksuit.data.wrappers.sample_wrappers.{kind}"])
    MasterFactory.set("sample_wrapper", sample_wrapper_factory)
    MasterFactory.set(
        "dataset",
        DatasetFactory(
            dataset_wrapper_factory=dataset_wrapper_factory,
            sample_wrapper_factory=sample_wrapper_factory,
            module_names=["ksuit.datasets.{kind}"],
        ),
    )
    # model
    MasterFactory.set("initializer", SingleFactory(module_names=["ksuit.initializers.{kind}"]))
    MasterFactory.set("loss", SingleFactory(module_names=["ksuit.losses.{kind}", "ksuit.losses.basic.{kind}"]))
    MasterFactory.set("freezer", SingleFactory(module_names=["ksuit.freezers.{kind}"]))
    MasterFactory.set("finalizer", SingleFactory(module_names=["ksuit.models.extractors.finalizers.{kind}"]))
    MasterFactory.set("extractor", SingleFactory(module_names=["ksuit.models.extractors.{kind}"]))
    MasterFactory.set("pooling", SingleFactory(module_names=["ksuit.models.poolings.{kind}"]))
    from ksuit.models import LinearModel, IdentityModel, TorchhubModel, TorchvisionModel
    MasterFactory.set(
        "model",
        ModelFactory(
            kind_to_ctor=dict(
                linear_model=LinearModel,
                identity_model=IdentityModel,
                torchhub_model=TorchhubModel,
                torchvision_model=TorchvisionModel,
            ),
        ),
    )
    # optim
    MasterFactory.set("param_group_modifier", SingleFactory(module_names=["ksuit.optim.param_group_modifiers.{kind}"]))
    MasterFactory.set("lr_scaler", SingleFactory(module_names=["ksuit.optim.lr_scalers.{kind}"]))
    MasterFactory.set("optim", OptimFactory(module_names=["torch.optim", "timm.optim.{kind}"]))
    # trainer
    MasterFactory.set("early_stopper", SingleFactory(module_names=["ksuit.trainers.early_stoppers.{kind}"]))
    MasterFactory.set("trainer", SingleFactory(module_names=["ksuit.trainers.{kind}"]))
    # schedule
    MasterFactory.set("schedule", ScheduleFactory())
    # pattern_matcher
    MasterFactory.set("pattern_matcher", SingleFactory(module_names=["ksuit.pattern_matchers.{kind}"]))


_setup_master_factory()
