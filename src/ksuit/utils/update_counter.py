from .checkpoint import Checkpoint


class UpdateCounter:
    def __init__(
            self,
            start_checkpoint: Checkpoint,
            end_checkpoint: Checkpoint,
            updates_per_epoch: int,
            effective_batch_size: int,
    ):
        self.updates_per_epoch = updates_per_epoch

        # start_checkpoint should always be fully specified (either E0_U0_S0 or derived from ResumeInitializer)
        self.start_checkpoint = start_checkpoint
        assert self.start_checkpoint.is_fully_specified

        # fully specify end_checkpoint (based on difference between start_checkpoint)
        # resuming with a different batchsize is not supported
        # - batchsize can still be lower than the defined batchsize by only using a subset of the loaded batch
        # - schedules are not adjusted to it
        # - how are schedules such as inverse sqrt schedule handled?
        assert self.start_checkpoint == Checkpoint(epoch=self.start_checkpoint.epoch).to_fully_specified(
            updates_per_epoch=updates_per_epoch,
            effective_batch_size=effective_batch_size,
        )
        assert end_checkpoint.is_minimally_specified
        delta_ckpt = end_checkpoint - self.start_checkpoint.to_target_specification(end_checkpoint)
        fully_specified_delta = delta_ckpt.to_fully_specified(
            updates_per_epoch=updates_per_epoch,
            effective_batch_size=effective_batch_size,
        )
        self.end_checkpoint = self.start_checkpoint + fully_specified_delta
        assert self.end_checkpoint.is_fully_specified

        self.cur_checkpoint = self.start_checkpoint.copy()
        self.effective_batch_size = effective_batch_size

    @property
    def is_full_epoch(self):
        assert self.cur_checkpoint.is_fully_specified
        return self.update % self.updates_per_epoch == 0

    @property
    def epoch_as_float(self):
        return float(self.cur_checkpoint.update) / self.updates_per_epoch

    @property
    def epoch(self):
        return self.cur_checkpoint.epoch

    @property
    def update(self):
        return self.cur_checkpoint.update

    @property
    def sample(self):
        return self.cur_checkpoint.sample

    @property
    def is_finished(self):
        return self.cur_checkpoint.to_target_specification(self.end_checkpoint) >= self.end_checkpoint

    def next_epoch(self):
        self.cur_checkpoint.epoch += 1

    def next_update(self):
        self.cur_checkpoint.update += 1

    def add_samples(self, num_samples):
        self.cur_checkpoint.sample += num_samples
