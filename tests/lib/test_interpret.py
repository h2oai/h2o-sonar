# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import pytest

from h2o_sonar import config
from h2o_sonar import interpret
from tests import test_utils


@pytest.mark.skipif(
    not test_utils.are_python_modules_installed({"cryptography"}),
    reason="'cryptography' Python package is not installed",
)
@pytest.mark.h2o_sonar
def test_config_management(tmp_path):
    #
    # GIVEN
    #
    cfg_path = tmp_path / "test_config.json"
    given_cfg = config.H2oSonarConfig()
    given_cfg.save(str(cfg_path))
    refresh_token = "SECRET REFRESH TOKEN!"
    encryption_key = "SECRET ENCRYPTION KEY"
    my_connection = config.ConnectionConfig(
        connection_type=config.ConnectionConfigType.DRIVERLESS_AI.name,
        name="My connection name",
        description="My connection description.",
        server_url="http://localhost:12345",
        auth_server_url="https://localhost:443/auth",
        realm_name="my_realm",
        client_id="my_client_id",
        token=refresh_token,
    )
    license_value = "SECRET LICENSE!"
    my_license = config.LicenseConfig(
        product=config.ProductLicenseConfig.DRIVERLESS_AI.name,
        name="My license name",
        description="My license description.",
        license=license_value,
        license_file="/tmp/license-of-my-product.txt",
    )

    #
    # WHEN
    #
    my_connection_dict = my_connection.to_dict(encrypt=False)
    print(f"\nmy_connection_dict:\n{my_connection_dict}")
    interpret.add_config_item(
        h2o_sonar_config_path=str(cfg_path),
        config_type=config.ConfigItemType.CONNECTION.name,
        config_value=my_connection_dict,
        encryption_key=encryption_key,
    )
    interpret.add_config_item(
        h2o_sonar_config_path=str(cfg_path),
        config_type=config.ConfigItemType.LICENSE.name,
        config_value=my_license.to_dict(encrypt=False),
        encryption_key=encryption_key,
    )

    #
    # THEN
    #
    with open(cfg_path) as file:
        json_as_str_w_encrypted_data = file.read()
    print(f"\nJSON with new encrypted data:\n{json_as_str_w_encrypted_data}")
    assert json_as_str_w_encrypted_data
    assert my_connection.server_url in json_as_str_w_encrypted_data
    assert my_license.description in json_as_str_w_encrypted_data
