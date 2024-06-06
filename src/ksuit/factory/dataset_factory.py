import logging

from .base import SingleFactory


class DatasetFactory(SingleFactory):
    def __init__(
            self,
            dataset_wrapper_factory: SingleFactory = None,
            sample_wrapper_factory: SingleFactory = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset_wrapper_factory = dataset_wrapper_factory
        self.sample_wrapper_factory = sample_wrapper_factory

    def instantiate(self, kind=None, dataset_wrappers=None, sample_wrappers=None, optional_kwargs=None, **kwargs):
        # instantiate dataset
        dataset = super().instantiate(kind, optional_kwargs=optional_kwargs, **kwargs)

        # wrap with dataset_wrappers
        if dataset_wrappers is not None:
            assert isinstance(dataset_wrappers, list)
            for dataset_wrapper_kwargs in dataset_wrappers:
                dataset_wrapper_kind = dataset_wrapper_kwargs.pop("kind")
                logging.info(f"instantiating dataset_wrapper {dataset_wrapper_kind}")
                dataset = self.dataset_wrapper_factory.instantiate(
                    kind=dataset_wrapper_kind,
                    dataset=dataset,
                    **dataset_wrapper_kwargs,
                )

        # wrap with sample_wrappers
        if sample_wrappers is not None:
            assert isinstance(sample_wrappers, list)
            for sample_wrapper_kwargs in sample_wrappers:
                sample_wrapper_kind = sample_wrapper_kwargs.pop("kind")
                logging.info(f"instantiating sample_wrapper {sample_wrapper_kind}")
                dataset = self.sample_wrapper_factory.instantiate(
                    kind=sample_wrapper_kind,
                    dataset=dataset,
                    **sample_wrapper_kwargs,
                )

        return dataset
