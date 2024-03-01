# Adapted from vllm
# https://github.com/skypilot-org/skypilot/blob/86dc0f6283a335e4aa37b3c10716f90999f48ab6/sky/sky_logging.py
"""Logging configuration for llmpq."""
import logging
import sys

import colorlog
from ..config import PROJECT_NAME


# _FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
_FORMAT = colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s",
    log_colors={
        'DEBUG': 'reset',
        'INFO': 'blue',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
)
_DATE_FORMAT = "%m-%d %H:%M:%S"


# class NewLineFormatter(logging.Formatter):
#     """Adds logging prefix to newlines to align multi-line messages."""

#     def __init__(self, fmt, datefmt=None):
#         logging.Formatter.__init__(self, fmt, datefmt)

#     def format(self, record):
#         msg = logging.Formatter.format(self, record)
#         if record.message != "":
#             parts = msg.split(record.message)
#             msg = msg.replace("\n", "\r\n" + parts[0])
#         return msg


_root_logger = logging.getLogger(PROJECT_NAME)
_default_handler = None


def _setup_logger():
    _root_logger.setLevel(logging.INFO)
    global _default_handler
    if _default_handler is None:
        _default_handler = logging.StreamHandler(sys.stdout)
        _default_handler.flush = sys.stdout.flush  # type: ignore
        _default_handler.setLevel(logging.INFO)
        _root_logger.addHandler(_default_handler)
    # fmt = NewLineFormatter(_FORMAT, datefmt=_DATE_FORMAT)
    _default_handler.setFormatter(_FORMAT)
    # Setting this will avoid the message
    # being propagated to the parent logger.
    _root_logger.propagate = False


# The logger is initialized when the module is imported.
# This is thread-safe as the module is only imported once,
# guaranteed by the Python GIL.
_setup_logger()

def init_logger(name: str):
    # Use the same settings as above for root logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(_default_handler)
    logger.propagate = False
    return logger

def assert_log(condition, msg):
    if not condition:
        _root_logger.error(msg)
        exit()