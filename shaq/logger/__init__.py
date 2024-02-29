import logging
import colorlog

from ..config import PROJECT_NAME

# configure logger
logger = logging.getLogger(PROJECT_NAME)
logger.setLevel(logging.INFO)
# configure formatter
formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)s:%(name)s:%(message)s",
    log_colors={
        'DEBUG': 'reset',
        'INFO': 'blue',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
)

# configure file handler
# file_handler = logging.FileHandler('example.log')
# file_handler.setLevel(logging.INFO)
# file_handler.setFormatter(formatter)

# configure console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

# add handlers to logger
# logger.addHandler(file_handler)
logger.addHandler(console_handler)



def assert_log(condition, msg):
    if not condition:
        logger.error(msg)
        exit()