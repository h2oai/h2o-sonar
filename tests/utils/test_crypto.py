# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import os

import pytest

from h2o_sonar.utils import crypto
from tests import test_utils


@pytest.mark.skipif(
    not test_utils.are_python_modules_installed({"cryptography"}),
    reason="'cryptography' Python package is not installed",
)
@pytest.mark.h2o_sonar
@pytest.mark.parametrize("primary_key", ["primary encryption key", ""])
def test_encryption_key_resolution(primary_key):
    #
    # GIVEN
    #
    is_present = (
        True if os.getenv(crypto.ENV_VAR_H2O_SONAR_ENCRYPTION_KEY, "") else False
    )
    print(f"\nEncryption key is present in the environment: {is_present}")

    #
    # WHEN
    #
    resolved_key = ""
    try:
        resolved_key = crypto.resolve_encryption_key(encryption_key=primary_key)
    except ValueError as e:
        if not primary_key and not is_present:
            print(f"Expected exception: {e}")

    #
    # THEN
    #
    print(f"Resolved encryption key: '{resolved_key}'")
    if primary_key:
        assert resolved_key == primary_key
    else:
        if is_present:
            assert resolved_key != primary_key


@pytest.mark.skipif(
    not test_utils.are_python_modules_installed({"cryptography"}),
    reason="'cryptography' Python package is not installed",
)
@pytest.mark.h2o_sonar
def test_encryption():
    #
    # GIVEN
    #
    encryption_key = "secret encryption key"
    data = "data to encrypt"
    print(f"\nData to encrypt: '{data}'")

    #
    # WHEN
    #
    encrypted_data = crypto.encrypt(encryption_key=encryption_key, data=data)

    #
    # THEN
    #
    print(f"Encrypted data : '{encrypted_data}'")
    assert data
    decrypted_data = crypto.decrypt(encryption_key=encryption_key, data=encrypted_data)
    print(f"Decrypted data : '{decrypted_data}'")
    assert decrypted_data == data


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
