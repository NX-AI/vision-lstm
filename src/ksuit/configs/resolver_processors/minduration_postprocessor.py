import kappaconfig as kc


class MinDurationPostProcessor(kc.Processor):
    """ limit training duration to a minimum by manipulating the configuration yaml """

    def __init__(self, effective_batch_size, max_epochs, max_updates, max_samples):
        self.effective_batch_size = effective_batch_size
        self.max_epochs = max_epochs
        self.max_updates = max_updates
        self.max_samples = max_samples

    def preorder_process(self, node, trace):
        if len(trace) == 0:
            return
        parent, parent_accessor = trace[-1]
        if isinstance(parent_accessor, str):
            # trainer
            if parent_accessor == "log_every_n_epochs":
                parent[parent_accessor] = 1
            elif parent_accessor == "log_every_n_updates":
                parent[parent_accessor] = 1
            elif parent_accessor == "log_every_n_samples":
                parent[parent_accessor] = min(parent[parent_accessor], self.effective_batch_size)
            elif parent_accessor == "max_epochs":
                parent[parent_accessor] = min(parent[parent_accessor], self.max_epochs)
            elif parent_accessor == "max_updates":
                parent[parent_accessor] = min(parent[parent_accessor], self.max_updates)
            elif parent_accessor == "max_samples":
                parent[parent_accessor] = min(parent[parent_accessor], self.max_samples)
            # set loggers
            elif parent_accessor == "every_n_epochs":
                parent[parent_accessor] = 1
            elif parent_accessor == "every_n_updates":
                parent[parent_accessor] = 1
            elif parent_accessor == "every_n_samples":
                parent[parent_accessor] = self.effective_batch_size
            # initializers
            if parent_accessor == "initializer":
                if parent[parent_accessor]["kind"] == "previous_stage_initializer":
                    self._process_checkpoint(parent[parent_accessor], "checkpoint")
            # schedules
            if "schedule" in parent_accessor:
                for schedule in parent[parent_accessor]:
                    if "start_checkpoint" in schedule:
                        self._process_checkpoint(schedule, "start_checkpoint")
                    if "end_checkpoint" in schedule:
                        self._process_checkpoint(schedule, "end_checkpoint")

    def _process_checkpoint(self, parent, parent_accessor):
        # check if checkpoint is string checkpoint
        if not isinstance(parent[parent_accessor], dict):
            return
        # replace epoch/update/sample checkpoint
        if "epoch" in parent[parent_accessor]:
            parent[parent_accessor] = dict(epoch=self.max_epochs)
        if "update" in parent[parent_accessor]:
            parent[parent_accessor] = dict(update=self.max_updates)
        if "sample" in parent[parent_accessor]:
            parent[parent_accessor] = dict(sample=self.max_samples)
