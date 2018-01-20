import logging
from enum import Enum
import threading


# global threading lock for all logging operations
ACTIVE_LOGGERS = []
ACTIVE_LOGGERS_LK = threading.Lock()


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


def log(message, level=Level.DEBUG, logger=None):
    if logger is None:
        __log_all(message, level)
    else:
        with ACTIVE_LOGGERS_LK:
            if isinstance(logger, list):
                # in python3 map is lazily evaluated, need list wrapper
                list(map(lambda lg: __send_to_logger(lg, message, level), logger))
            else:
                __send_to_logger(logger, message, level)


def __log_all(message, level):
    times_logged = 0
    with ACTIVE_LOGGERS_LK:
        for logger in ACTIVE_LOGGERS:
            times_logged += __send_to_logger(logger, message, level)

    if times_logged is not len(ACTIVE_LOGGERS):
        raise Exception("Failed to write to all loggers!")


def __send_to_logger(logger, message, level):
    log = logger.get_logger()
    if level is Level.DEBUG:
        log.debug(message)
        return 1
    elif level is Level.INFO:
        log.info(message)
        return 1
    elif level is Level.ERROR:
        log.error(message)
        return 1
    elif level is Level.WARN:
        log.warn(message)
        return 1
    elif level is Level.FATAL:
        log.fatal(message)
        return 1
    return 0


def append_active_logger(logger):
    with ACTIVE_LOGGERS_LK:
        ACTIVE_LOGGERS.append(logger)


class FileLogger:

    def __init__(self, filename="tetris.log", level=Level.DEBUG, name=''):
        logger = logging.getLogger("FileLogger" + name)
        handler = logging.FileHandler(name + "-" + filename)
        formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s]  %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level.get_value())
        self.logger = logger
        self.name = name
        self.filename = filename
        append_active_logger(self)
        logger.info('==============================')
        logger.info('Tetris File-Logger Started...')

    def get_logger(self):
        return self.logger

    def __str__(self):
        return "Name:" + self.name + "  Filename:" + self.filename + "  logger:"\
               + str(self.logger)


class ConsoleLogger:

    def __init__(self, level=Level.DEBUG, name=''):
        logger = logging.getLogger("ConsoleLogger" + name)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [logger: " + name + "]  %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level.get_value())
        self.logger = logger
        self.name = name
        append_active_logger(self)
        logger.info('================================')
        logger.info('Tetris Console-Logger Started...')

    def get_logger(self):
        return self.logger

    def __str__(self):
        return "Name:" + self.name + "  Filename:" + "Console" + "  logger:"\
               + str(self.logger)


class DataLogger:

    pass
