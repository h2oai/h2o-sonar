# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import logging
import os.path

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar import loggers


# constants
FILE_PROC_MEMINFO = "/proc/meminfo"


def get_mem_profile():
    if os.path.isfile(FILE_PROC_MEMINFO):
        with open(os.path.join(FILE_PROC_MEMINFO)) as f:
            lines = f.readlines()
            vm_available = [la for la in lines if la.startswith("MemAvailable")][0]
            vm_total = [lt for lt in lines if lt.startswith("MemTotal")][0]

        # available and total memory normalized to MB
        return (
            int(int(vm_available.split()[1]) * 1024 / 1000000),
            int(int(vm_total.split()[1]) * 1024 / 1000000),
        )

    return None, None


class SonarProfilingFormatter(logging.Formatter):
    KEY_VM_AVAILABLE = "vm_available"
    KEY_VM_TOTAL = "vm_total"

    def format(self, record):
        if isinstance(record.args, dict):
            record.vm_available = record.args.get(
                SonarProfilingFormatter.KEY_VM_AVAILABLE
            )
            record.vm_total = record.args.get(SonarProfilingFormatter.KEY_VM_TOTAL)
        else:
            record.vm_available = -1
            record.vm_total = -1

        return super().format(record)


# CPU and memory profiling utilities
class SonarProfilingLogger(loggers.SonarLogger):
    """Print logger prints log messages using print() method."""

    FORMATTER = SonarProfilingFormatter(
        "%(asctime)s VM:%(vm_available)s/%(vm_total)s %(levelname)s %(message)s"
    )

    KEY_CPU = "cpu"

    @staticmethod
    def _inject_cpu_mem_kwargs() -> dict:
        if h2o_sonar_config.config.enable_profiler:
            (vm_available, vm_total) = get_mem_profile()
            return {
                SonarProfilingFormatter.KEY_VM_AVAILABLE: vm_available,
                SonarProfilingFormatter.KEY_VM_TOTAL: vm_total,
            }

        return {
            SonarProfilingFormatter.KEY_VM_AVAILABLE: -1,
            SonarProfilingFormatter.KEY_VM_TOTAL: -1,
        }

    def __init__(self, logger_name: str, log_file: str, log_level=logging.WARNING):
        handler = logging.FileHandler(str(log_file))
        handler.setFormatter(SonarProfilingLogger.FORMATTER)

        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(log_level)
        self.logger.addHandler(handler)

        self.fallback_logger = loggers.SonarFileLogger(
            logger_name=f"Fallback {logger_name}",
            log_file=log_file,
            log_level=log_level,
        )

    def info(self, msg, *args, **kwargs):
        try:
            self.logger.info(
                msg,
                SonarProfilingLogger._inject_cpu_mem_kwargs(),
                *args,
                **kwargs,
            )
        except Exception as ex:
            self.fallback_logger.info(
                f"{msg}\n(profiling logger failure: {ex})", *args, **kwargs
            )

    def debug(self, msg, *args, **kwargs):
        try:
            self.logger.debug(
                msg, SonarProfilingLogger._inject_cpu_mem_kwargs(), *args, **kwargs
            )
        except Exception as ex:
            self.fallback_logger.debug(
                f"{msg}\n(profiling logger failure: {ex})", *args, **kwargs
            )

    def warning(self, msg, *args, **kwargs):
        try:
            self.logger.warning(
                msg, SonarProfilingLogger._inject_cpu_mem_kwargs(), *args, **kwargs
            )
        except Exception as ex:
            self.fallback_logger.warning(
                f"{msg}\n(profiling logger failure: {ex})", *args, **kwargs
            )

    def error(self, msg, *args, **kwargs):
        try:
            self.logger.error(
                msg, SonarProfilingLogger._inject_cpu_mem_kwargs(), *args, **kwargs
            )
        except Exception as ex:
            self.fallback_logger.error(
                f"{msg}\n(profiling logger failure: {ex})", *args, **kwargs
            )

    def data(self, data, *args, **kwargs):
        self.info(msg=data, *args, **kwargs)
