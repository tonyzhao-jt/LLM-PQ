import logging

# configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# configure formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# configure file handler
file_handler = logging.FileHandler('example.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# configure console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

# add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)
