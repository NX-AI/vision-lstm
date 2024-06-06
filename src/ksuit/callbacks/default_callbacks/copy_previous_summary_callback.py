from ksuit.callbacks.base.callback_base import CallbackBase
from ksuit.initializers import PreviousRunInitializer
from ksuit.models import CompositeModel
from ksuit.utils.model_utils import get_named_models


class CopyPreviousSummaryCallback(CallbackBase):
    @staticmethod
    def _should_include_key(key):
        if key.startswith("profiler/") or key.startswith("profiling/") or key.startswith("lr/"):
            return False
        return True

    def _before_training(self, model, **_):
        # collect summaries
        summaries = {}
        for model_name, model in get_named_models(model).items():
            if isinstance(model, CompositeModel):
                continue
            for initializer in model.initializers:
                if not isinstance(initializer, PreviousRunInitializer):
                    continue
                summary = self.summary_provider.get_summary_of_previous_stage(
                    stage_name=initializer.stage_name,
                    stage_id=initializer.stage_id,
                )
                if summary is None:
                    self.logger.info(
                        f"no summary found for initializer of {model_name} (stage_name='{initializer.stage_name}' "
                        f"stage_id={initializer.stage_id}) -> don't copy anything"
                    )
                    continue
                if initializer.stage_name in summaries:
                    self.logger.info(
                        f"duplicate stage_name when copying summaries from {PreviousRunInitializer.__name__} "
                        "-> using first summary"
                    )
                    if summary != summaries[initializer.stage_name]:
                        self.logger.warning(f"summaries are not the same -> only first summary is copied")
                    continue
                summaries[initializer.stage_name] = summary

        # add to summary
        for previous_stage_name, summary in summaries.items():
            # filter unnecessary keys
            summary = {k: v for k, v in summary.items() if self._should_include_key(k)}
            for key, value in summary.items():
                new_key = f"{previous_stage_name}/{key}"
                if new_key in self.summary_provider:
                    self.logger.warning(f"'{new_key}' already exists in summary_provider -> skip")
                    continue
                self.summary_provider[new_key] = value
