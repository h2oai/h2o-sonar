# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
from logging import INFO
from logging import WARNING
from unittest import TestCase

from h2o_sonar import loggers as logging


class TestLogging(TestCase):
    def test_basic_logging(self):
        with self.assertLogs("h2o_sonar", level="INFO") as cm:
            logging.setLevel(WARNING)
            logging.debug("debug message")
            logging.info("info message")
            logging.warn("warn message")
            self.assertFalse("DEBUG:h2o_sonar:debug message" in cm.output)
            self.assertFalse("INFO:h2o_sonar:info message" in cm.output)
            self.assertEqual(cm.output, ["WARNING:h2o_sonar:warn message"])

            logging.setLevel(INFO)
            logging.debug("debug message")
            logging.info("info message")
            self.assertFalse("DEBUG:h2o_sonar:debug message" in cm.output)
            self.assertEqual(
                cm.output,
                ["WARNING:h2o_sonar:warn message", "INFO:h2o_sonar:info message"],
            )
