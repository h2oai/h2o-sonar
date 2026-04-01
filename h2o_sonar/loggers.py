# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import abc
import logging
import sys
import threading


"""Basic logging for H2O Sonar used instead of Python's logging."""

CRITICAL = logging.CRITICAL
ERROR = logging.ERROR
WARNING = logging.WARNING
INFO = logging.INFO
DEBUG = logging.DEBUG
NOTSET = logging.NOTSET

_logger = None
_logger_lock = threading.Lock()


def get_level():
    return _get_logger().level


def log(level, msg, *args, **kwargs):
    _get_logger().log(level, msg, *args, **kwargs)


def debug(msg, *args, **kwargs):
    _get_logger().debug(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    _get_logger().error(msg, *args, **kwargs)


def fatal(msg, *args, **kwargs):
    _get_logger().fatal(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    _get_logger().info(msg, *args, **kwargs)


def warn(msg, *args, **kwargs):
    # for compatibility
    _get_logger().warn(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    _get_logger().warning(msg, *args, **kwargs)


def setLevel(level=NOTSET):
    _get_logger().setLevel(level)


def _get_logger():
    global _logger

    if _logger:
        return _logger

    with _logger_lock:
        if _logger:
            return _logger

        _logger = logging.getLogger("h2o_sonar")

        _logger.propagate = False

        msg_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(msg_formatter)
        _logger.addHandler(stream_handler)

        return _logger


"""H2O Sonar loggers for methods and explainers."""


class SonarLogger(abc.ABC):
    """Abstract logger base class to be extended by runtime specific
    logging implementations.

    """

    FILE_NAME_H2O_SONAR_LOG = "h2o-sonar.log"

    def info(self, msg, *args, **kwargs):
        raise NotImplementedError

    def debug(self, msg, *args, **kwargs):
        raise NotImplementedError

    def warning(self, msg, *args, **kwargs):
        raise NotImplementedError

    def error(self, msg, *args, **kwargs):
        raise NotImplementedError

    # functional

    def data(self, data, *args, **kwargs):
        self.info(msg=data, *args, **kwargs)


class SonarPrintLogger(SonarLogger):
    """Print logger prints log messages using print() method."""

    def __init__(self):
        pass

    def info(self, msg, *args, **kwargs):
        print(msg)

    def debug(self, msg, *args, **kwargs):
        print(msg)

    def warning(self, msg, *args, **kwargs):
        print(msg, file=sys.stderr)

    def error(self, msg, *args, **kwargs):
        print(msg, file=sys.stderr)

    def data(self, data, *args, **kwargs):
        self.info(msg=data, *args, **kwargs)


class SonarFileLogger(SonarLogger):
    """File logger saves log messages to log files."""

    FORMATTER = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    def __init__(self, logger_name: str, log_file: str, log_level=logging.WARNING):
        handler = logging.FileHandler(str(log_file))
        handler.setFormatter(SonarFileLogger.FORMATTER)

        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(log_level)
        self.logger.addHandler(handler)

    @staticmethod
    def __drop_flush(**kwargs):
        if kwargs:
            if "flush" in kwargs:
                kwargs.pop("flush")
        return kwargs

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **SonarFileLogger.__drop_flush(**kwargs))

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **SonarFileLogger.__drop_flush(**kwargs))

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **SonarFileLogger.__drop_flush(**kwargs))

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **SonarFileLogger.__drop_flush(**kwargs))

    def data(self, data, *args, **kwargs):
        self.info(msg=data, *args, **SonarFileLogger.__drop_flush(**kwargs))
