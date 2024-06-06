import logging


class SummaryProviderBase:
    def __init__(self):
        self.logger = logging.getLogger(type(self).__name__)

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __getitem__(self, key):
        raise NotImplementedError

    def __contains__(self, key):
        raise NotImplementedError

    def keys(self):
        raise NotImplementedError

    def get_summary_of_previous_stage(self, stage_name, stage_id):
        raise NotImplementedError

    def flush(self):
        raise NotImplementedError

    def summarize_logvalues(self):
        raise NotImplementedError
