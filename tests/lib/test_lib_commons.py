# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import pytest

from h2o_sonar import loggers
from h2o_sonar.lib.api import commons


@pytest.mark.skip(reason="This is an ad hoc test for the local development.")
@pytest.mark.parametrize(
    "host, port, expected",
    [
        ("localhost", 12345, False),
        ("127.0.0.1", 12345, False),
        ("192.168.0.1", 12345, True),
        ("192.168.0.1", 10, False),
        ("111.111.111.111", 12345, False),
    ],
)
@pytest.mark.h2o_sonar
def test_is_port_used(host: str, port: int, expected: bool):
    #
    # GIVEN
    #

    #
    # WHEN
    #
    is_used = commons.is_port_used(
        host, port, timeout=3, logger=loggers.SonarPrintLogger()
    )

    #
    # THEN
    #
    assert is_used == expected
