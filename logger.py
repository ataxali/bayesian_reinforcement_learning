import logging
from enum import Enum
import threading


# global threading lock for all logging operations
active_loggers = []
active_loggers_lk = threading.Lock()


class Level(Enum):

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    ERROR = logging.ERROR
    WARN = logging.WARN
    FATAL = logging.FATAL

    def get_value(self):
        return self.value

    def __str__(self):
        return self.name


def log(message, level=Level.DEBUG):

    with active_loggers_lk:
        times_logged = 0
        for logger in active_loggers:
            if level is Level.DEBUG:
                logger.debug(message)
                times_logged += 1
            elif level is Level.INFO:
                logger.info(message)
                times_logged += 1
            elif level is Level.ERROR:
                logger.error(message)
                times_logged += 1
            elif level is Level.WARN:
                logger.warn(message)
                times_logged += 1
            elif level is Level.FATAL:
                logger.fatal(message)
                times_logged += 1

        if times_logged != len(active_loggers):
            raise Exception("Failed to log to all loggers! Logged:" + str(times_logged) +
                            " Active_Loggers:" + str(len(active_loggers)) + " Level:" + str(level))


def append_active_logger(logger):

    with active_loggers_lk:
        active_loggers.append(logger)


class FileLogger:

    def __init__(self, filename="tetris.log", level=Level.DEBUG, name=''):
        logger = logging.getLogger("FileLogger" + name)
        handler = logging.FileHandler(name + "-" + filename)
        formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s]  %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level.get_value())
        self.logger = logger
        append_active_logger(logger)
        logger.info('==============================')
        logger.info('Tetris File-Logger Started...')


class ConsoleLogger:

    def __init__(self, level=Level.DEBUG, name=''):
        logger = logging.getLogger("ConsoleLogger" + name)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [logger: " + name + "]  %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level.get_value())
        self.logger = logger
        append_active_logger(logger)
        logger.info('================================')
        logger.info('Tetris Console-Logger Started...')


class DataLogger:

    pass
