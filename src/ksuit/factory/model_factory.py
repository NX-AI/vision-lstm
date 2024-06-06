import logging
from copy import deepcopy

import yaml
from torch import nn

from ksuit.factory import MasterFactory
from .base import SingleFactory


class ModelFactory(SingleFactory):
    def instantiate(self, kind=None, path_provider=None, data_container=None, optional_kwargs=None, **kwargs):
        # exclude update_counter from copying (otherwise model and trainer have different update_counter objects)
        update_counter = kwargs.pop("update_counter", None)
        static_ctx = kwargs.pop("static_ctx", None)
        dynamic_ctx = kwargs.pop("dynamic_ctx", None)
        kwargs = deepcopy(kwargs)
        if path_provider is None and optional_kwargs is not None:
            path_provider = optional_kwargs.pop("path_provider")
        if data_container is None and optional_kwargs is not None:
            data_container = optional_kwargs.pop("data_container")

        # allow setting multiple kwargs in yaml; but allow also overwriting it
        # kind: vit.vit
        # kwargs: ${select:${vars.encoder_model_key}:${yaml:models/vit}}
        # patch_size: [128, 1] # this will overwrite the patch_size in kwargs
        kwargs_from_yaml = kwargs.pop("kwargs", {})
        kwargs = {**kwargs_from_yaml, **kwargs}

        # try to load kwargs from checkpoint
        if "initializers" in kwargs:
            # only first one can have use_checkpoint_kwargs
            initializer_kwargs = kwargs["initializers"][0]
            assert all("use_checkpoint_kwargs" not in obj for obj in kwargs["initializers"][1:])
            use_checkpoint_kwargs = initializer_kwargs.pop("use_checkpoint_kwargs", False)
            initializer = MasterFactory.get("initializer").create(initializer_kwargs, path_provider=path_provider)
            if use_checkpoint_kwargs:
                ckpt_kwargs = initializer.get_model_kwargs()
                if kind is None and "kind" in ckpt_kwargs:
                    kind = ckpt_kwargs.pop("kind")
                else:
                    ckpt_kwargs.pop("kind", None)
                # check if keys overlap; this can be intended
                # - vit trained with drop_path_rate but then for evaluation this should be set to 0
                # if keys overlap the explicitly specified value dominates (i.e. from yaml or from code)
                kwargs_intersection = set(kwargs.keys()).intersection(set(ckpt_kwargs.keys()))
                if len(kwargs_intersection) > 0:
                    logging.info(f"checkpoint_kwargs overlap with kwargs (intersection={kwargs_intersection})")
                    for intersecting_kwarg in kwargs_intersection:
                        if intersecting_kwarg.endswith("_kwargs"):
                            # if the parameter endswith "_kwargs" the two dictionaries should be merged instead of
                            # being replaced e.g. NnclrHead in stage2 has queue_kwargs=dict(guidance="oracle") and in
                            # stage3 dict(topk=20) the result should be dict(topk=20, guidance="oracle")
                            ckpt_dict = ckpt_kwargs.pop(intersecting_kwarg)
                            yaml_dict = kwargs.pop(intersecting_kwarg)
                            merged = {**ckpt_dict, **yaml_dict}
                            kwargs[intersecting_kwarg] = merged
                            logging.info(
                                f"found overlapping dictionaries as kwargs -> "
                                f"merging {ckpt_dict} with {yaml_dict} results in {merged}"
                            )
                        else:
                            ckpt_kwargs.pop(intersecting_kwarg)
                kwargs.update(ckpt_kwargs)
                # noinspection PyBroadException
                try:
                    logging.info(f"postprocessed checkpoint kwargs:\n{yaml.safe_dump(kwargs, sort_keys=False)[:-1]}")
                except:
                    logging.warning(f"couldnt parse kwargs: {kwargs}")

        assert kind is not None, "model has no kind (maybe use_checkpoint_kwargs=True is missing in the initializer?)"

        # rename optim to optim_ctor (in yaml it is intuitive to call it optim as the yaml should not bother with the
        # implementation details but the implementation passes a ctor so it should also be called like it)
        optim = kwargs.pop("optim", None)
        # model doesn't need to have an optimizer (e.g. frozen models)
        if optim is not None:
            kwargs["optim_ctor"] = optim

        # filter out modules passed to ctor
        ctor_kwargs_filtered = {k: v for k, v in kwargs.items() if not isinstance(v, nn.Module)}
        ctor_kwargs = deepcopy(ctor_kwargs_filtered)
        ctor_kwargs["kind"] = kind
        ctor_kwargs.pop("input_shape", None)
        ctor_kwargs.pop("output_shape", None)
        ctor_kwargs.pop("optim_ctor", None)
        ctor_kwargs.pop("is_frozen", None)
        ctor_kwargs.pop("freezers", None)
        ctor_kwargs.pop("initializers", None)
        ctor_kwargs.pop("extractors", None)

        # put everything into optional_kwargs to also support models that dont inherit from ksuit.model classes
        if optional_kwargs is None:
            optional_kwargs = dict(
                update_counter=update_counter,
                path_provider=path_provider,
                data_container=data_container,
                static_ctx=static_ctx,
                dynamic_ctx=dynamic_ctx,
                ctor_kwargs=ctor_kwargs,
            )
        else:
            # copy optional_kwargs before altering
            optional_kwargs = {**optional_kwargs}
            # update optional kwargs with the objects that were passed directly to instantiate
            if update_counter is not None:
                optional_kwargs["update_counter"] = update_counter
            if path_provider is not None:
                optional_kwargs["path_provider"] = path_provider
            if data_container is not None:
                optional_kwargs["data_container"] = data_container
            # set kwargs that should not already be registered as optional_kwargs
            assert "static_ctx" not in optional_kwargs
            optional_kwargs["static_ctx"] = static_ctx
            assert "dynamic_ctx" not in optional_kwargs
            optional_kwargs["dynamic_ctx"] = dynamic_ctx
            assert "ctor_kwargs" not in optional_kwargs
            optional_kwargs["ctor_kwargs"] = ctor_kwargs

        return super().instantiate(
            kind=kind,
            optional_kwargs=optional_kwargs,
            **kwargs,
        )
