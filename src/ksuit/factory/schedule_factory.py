from kappaschedules import object_to_schedule, ConstantSchedule

from ksuit.utils.schedule_wrapper import ScheduleWrapper
from .base import FactoryBase


class ScheduleFactory(FactoryBase):
    def create(self, obj_or_kwargs, collate_fn=None, **kwargs):
        assert collate_fn is None
        if isinstance(obj_or_kwargs, (float, int)):
            schedule = ConstantSchedule(value=obj_or_kwargs)
            update_counter = None
        else:
            update_counter = kwargs.pop("update_counter", None)
            if update_counter is not None:
                assert "batch_size" not in kwargs
                assert "updates_per_epoch" not in kwargs
                kwargs["batch_size"] = update_counter.effective_batch_size
                kwargs["updates_per_epoch"] = update_counter.updates_per_epoch
            schedule = object_to_schedule(obj=obj_or_kwargs, **kwargs)
        if schedule is None:
            return None
        return ScheduleWrapper(schedule=schedule, update_counter=update_counter)

    def instantiate(self, kind, optional_kwargs=None, **kwargs):
        raise RuntimeError
