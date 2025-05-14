import pandas as pd
import logging 

def setup_logger():
    logger = logging.getLogger('ACA')
    handler = logging.StreamHandler()
    logger.addHandler(handler)

    return logger

def set_logger_level(logger):
    level = logging.DEBUG
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
    

logger = setup_logger()