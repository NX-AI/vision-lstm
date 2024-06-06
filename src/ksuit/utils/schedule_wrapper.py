from kappaschedules import ScheduleBase, ConstantSchedule

from ksuit.utils.update_counter import UpdateCounter


class ScheduleWrapper:
    def __init__(self, schedule: ScheduleBase, update_counter: UpdateCounter = None):
        self.schedule = schedule
        self.update_counter = update_counter

    def get_value(self):
        if self.update_counter is None:
            assert isinstance(self.schedule, ConstantSchedule)
            return self.schedule.get_value(step=0, total_steps=1)
        return self.schedule.get_value(
            step=self.update_counter.cur_checkpoint.update,
            total_steps=self.update_counter.end_checkpoint.update,
        )
