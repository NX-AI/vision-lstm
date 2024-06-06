import logging
import sys
from collections import defaultdict
from contextlib import contextmanager

from ksuit.distributed.config import is_rank0

def log(log_fn, msg):
    if log_fn is not None:
        log_fn(msg)

def _add_handler(handler, prefix=""):
    logger = logging.getLogger()
    if prefix != "":
        prefix = f"{prefix} "
    handler.setFormatter(logging.Formatter(
        fmt=f"%(asctime)s %(levelname).1s {prefix}%(message)s",
        datefmt="%m-%d %H:%M:%S",
    ))
    logger.handlers.append(handler)
    return handler


def add_stdout_handler(prefix=""):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    _add_handler(logging.StreamHandler(stream=sys.stdout), prefix=prefix)


def add_global_handlers(log_file_uri=None):
    logger = logging.getLogger()
    logger.handlers = []
    # add a stdout logger to all ranks to also allow non-rank0 processes to log to stdout
    add_stdout_handler()
    # add_stdout_handler sets level to logging.INFO
    if is_rank0():
        if log_file_uri is not None:
            _add_handler(logging.FileHandler(log_file_uri, mode="a"))
            logging.info(f"log file: {log_file_uri.as_posix()}")
    else:
        # subprocesses log warnings to stderr --> logging.CRITICAL prevents this
        logger.setLevel(logging.CRITICAL)
    return _add_handler(MessageCounter())


@contextmanager
def log_from_all_ranks():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    yield
    level = logging.INFO if is_rank0() else logging.CRITICAL
    logger.setLevel(level)


class MessageCounter(logging.Handler):
    def __init__(self):
        super().__init__()
        self.min_level = logging.WARNING
        self.counts = defaultdict(int)

    def emit(self, record):
        if record.levelno >= self.min_level:
            self.counts[record.levelno] += 1

    def log(self):
        logging.info("------------------")
        for level in [logging.WARNING, logging.ERROR]:
            logging.info(f"encountered {self.counts[level]} {logging.getLevelName(level).lower()}s")
