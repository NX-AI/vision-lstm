import logging
import shutil
from pathlib import Path

import kappaconfig as kc
import yaml

from .resolver_processors.mindata_postprocessor import MinDataPostProcessor
from .resolver_processors.minduration_postprocessor import MinDurationPostProcessor
from .resolver_processors.minmodel_postprocessor import MinModelPostProcessor
from .resolver_processors.minmodel_preprocessor import MinModelPreProcessor
from .resolver_processors.precision_preprocessor import PrecisionPreProcessor
from .resolver_processors.remove_large_collections_postprocessor import RemoveLargeCollectionsProcessor
from .resolver_processors.schedule_template_postprocessor import ScheduleTemplatePostProcessor
from .resolver_processors.testrun_postprocessor import TestrunPostProcessor

TESTRUN_EFFECTIVE_BATCH_SIZE = 2
TESTRUN_EPOCHS = 2
TESTRUN_UPDATES_PER_EPOCH = 3
TESTRUN_UPDATES = TESTRUN_EPOCHS * TESTRUN_UPDATES_PER_EPOCH
TESTRUN_SAMPLES = TESTRUN_UPDATES * TESTRUN_EFFECTIVE_BATCH_SIZE


class Hyperparams:

    @staticmethod
    def _get_hp_file_uri(hp_file):
        file_uri = Path(hp_file).expanduser().with_suffix(".yaml")
        assert file_uri.exists(), f"hp_file '{file_uri}' doesn't exist"
        return file_uri

    @staticmethod
    def save_unresolved_hp(hp_file, out_file_uri):
        file_uri = Hyperparams._get_hp_file_uri(hp_file)
        shutil.copy(file_uri, out_file_uri)
        logging.info(f"copied unresolved hp to {out_file_uri}")

    @staticmethod
    def save_resolved_hp(stage_hp, out_file_uri):
        stage_hp = Hyperparams._remove_large_collections(stage_hp)
        with open(out_file_uri, "w") as f:
            yaml.safe_dump(stage_hp, f, sort_keys=False)
        logging.info(f"dumped resolved hp to {out_file_uri}")

    @staticmethod
    def get_stage_hp(
            hp_file,
            testrun=False,
            minmodelrun=False,
            mindatarun=False,
            mindurationrun=False,
    ):
        file_uri = Hyperparams._get_hp_file_uri(hp_file)
        run_hp = kc.from_file_uri(file_uri)

        # extract template path
        if "template_path" in run_hp:
            template_path = run_hp.pop("template_path").value
        else:
            template_path = None

        resolver = kc.DefaultResolver(template_path=template_path)
        resolver.pre_processors.append(PrecisionPreProcessor())
        resolver.post_processors.append(ScheduleTemplatePostProcessor())
        if testrun:
            resolver.post_processors.append(TestrunPostProcessor())
        if minmodelrun or testrun:
            resolver.pre_processors.append(MinModelPreProcessor())
            resolver.post_processors.append(MinModelPostProcessor())
        if mindatarun or testrun:
            resolver.post_processors.append(
                MinDataPostProcessor(
                    effective_batch_size=TESTRUN_EFFECTIVE_BATCH_SIZE,
                    updates_per_epoch=TESTRUN_UPDATES_PER_EPOCH,
                ),
            )
        if mindurationrun or testrun:
            resolver.post_processors.append(
                MinDurationPostProcessor(
                    effective_batch_size=TESTRUN_EFFECTIVE_BATCH_SIZE,
                    max_epochs=TESTRUN_EPOCHS,
                    max_updates=TESTRUN_UPDATES,
                    max_samples=TESTRUN_SAMPLES,
                ),
            )

        resolved = resolver.resolve(run_hp)
        return resolved

    @staticmethod
    def _remove_large_collections(stage_hp):
        stage_hp = kc.from_primitive(stage_hp)
        resolver = kc.Resolver(post_processors=[RemoveLargeCollectionsProcessor()])
        resolved = resolver.resolve(stage_hp)
        return resolved

    @staticmethod
    def log_stage_hp(stage_hp):
        stage_hp = Hyperparams._remove_large_collections(stage_hp)
        yaml_str = yaml.safe_dump(stage_hp, sort_keys=False)
        # safe_dump appends a trailing newline
        logging.info(f"------------------\n{yaml_str[:-1]}")
