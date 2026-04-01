# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import os
import subprocess

import pytest

from h2o_sonar import config
from h2o_sonar import config as h2o_sonar_config
from h2o_sonar.lib.api import commons
from tests import test_utils
from tests.lib import test_cli as test_cli_test
from tests.lib.given_generative import DAI_WORKER_CONNECTION


@pytest.mark.skipif(
    not test_utils.are_python_modules_installed({"cryptography"}),
    reason="'cryptography' Python package is not installed",
)
def test_save_load(tmp_path):
    #
    # GIVEN
    #
    encryption_key = "SECRET ENCRYPTION KEY"
    refresh_token = "SECRET REFRESH TOKEN"
    my_connection = config.ConnectionConfig(
        connection_type=config.ConnectionConfigType.H2O_3.name,
        name="My connection name",
        description="My connection description.",
        server_url="http://localhost:8080",
        auth_server_url="http://localhost:8080/auth",
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
    json_config_path = tmp_path / "test_config.json"

    #
    # WHEN
    #
    sonar_cfg = config.H2oSonarConfig(
        connections=[my_connection],
        licenses=[my_license],
    )
    sonar_cfg.save(config_path=str(json_config_path), encryption_key=encryption_key)

    #
    # THEN
    #
    with open(json_config_path) as file:
        json_as_str_w_encrypted_data = file.read()
    print(f"\nJSON with encrypted data:\n{json_as_str_w_encrypted_data}")
    assert json_as_str_w_encrypted_data

    # load and assert
    loaded_cfg_dict = config.H2oSonarConfig.load(
        config_path=str(json_config_path),
        encryption_key=encryption_key,
    )
    print(f"Loaded config w/ decrypted data:\n{loaded_cfg_dict}")
    print(
        f"Decrypted connection refresh token: {loaded_cfg_dict['connections'][0].token}"
    )
    assert refresh_token == loaded_cfg_dict["connections"][0].token
    print(f"Decrypted license: {loaded_cfg_dict['licenses'][0].license}")
    assert license_value == loaded_cfg_dict["licenses"][0].license

    # load&override and assert
    sonar_cfg.connections[0].token = "...another refresh token..."
    sonar_cfg.licenses[0].license = "...another license..."
    sonar_cfg.load_and_override(
        config_path=str(json_config_path),
        encryption_key=encryption_key,
    )
    print(f"OBJ: Decrypted connection refresh token: {sonar_cfg.connections[0].token}")
    assert refresh_token == sonar_cfg.connections[0].token
    print(f"OBJ: Decrypted license: {sonar_cfg.licenses[0].license}")
    assert license_value == sonar_cfg.licenses[0].license


@pytest.mark.skipif(
    not test_utils.are_python_modules_installed({"cryptography"}),
    reason="'cryptography' Python package is not installed",
)
def test_cli_config_default():
    """Show default H2O Sonar configuration so that it can be saved to a file."""
    #
    # GIVEN
    #

    #
    # WHEN
    #
    (cmd, child_env) = test_utils.given_base_cli_cmd()
    cmd = cmd + [
        "show",
        "config",
    ]
    print(f"\nRunning H2O Sonar:\n{cmd}\n")
    process = subprocess.Popen(cmd, env=child_env, stdout=subprocess.PIPE, stderr=None)
    process.wait()
    stdout_output, stderr_output = process.communicate()

    #
    # THEN
    #
    stdout_output_str = stdout_output.decode("utf-8")
    print(f"STDOUT:\n{stdout_output_str}")
    for required in ["h2o_host", "h2o_port", "connections"]:
        assert required in stdout_output_str
    assert stderr_output is None
    assert process.wait() == 0


@pytest.mark.skipif(
    not test_utils.are_python_modules_installed({"cryptography"}),
    reason="'cryptography' Python package is not installed",
)
def test_cli_config_file():
    #
    # GIVEN
    #
    h2o_sonar_config_path = test_utils.find_locally("data/config/h2o-sonar-config.json")
    assert os.path.isfile(h2o_sonar_config_path)

    #
    # WHEN
    #
    (cmd, child_env) = test_utils.given_base_cli_cmd()
    cmd = cmd + [
        "show",
        "config",
        "--config-path",
        h2o_sonar_config_path,
        "--encryption-key",
        "my-secret-enc-key",
    ]
    print(f"\nRunning H2O Sonar:\n{cmd}\n")
    process = subprocess.Popen(cmd, env=child_env, stdout=subprocess.PIPE, stderr=None)
    process.wait()
    stdout_output, stderr_output = process.communicate()

    #
    # THEN
    #
    stdout_output_str = stdout_output.decode("utf-8")
    print(f"STDOUT:\n{stdout_output_str}")
    for required in ["h2o_host", "h2o_port", "connections", "ML_API"]:
        assert required in stdout_output_str
    for not_desired in ["encrypted"]:
        assert not_desired not in stdout_output_str
    assert stderr_output is None
    assert process.wait() == 0


@pytest.mark.skipif(
    not test_utils.are_python_modules_installed({"cryptography"}),
    reason="'cryptography' Python package is not installed",
)
def test_cli_config_add_connection(tmp_path):
    #
    # GIVEN
    #
    h2o_sonar_config_path = tmp_path / "h2o-sonar-config.json"
    encryption_key = "my-secret-enc-key"

    # GET & SAVE the default config
    (cmd, child_env) = test_utils.given_base_cli_cmd()
    cmd = cmd + [
        "show",
        "config",
        "--encryption-key",
        encryption_key,
    ]
    print(f"\nRunning H2O Sonar:\n{cmd}\n")
    process = subprocess.Popen(cmd, env=child_env, stdout=subprocess.PIPE, stderr=None)
    process.wait()
    stdout_output, stderr_output = process.communicate()
    assert stderr_output is None
    assert "connections" in stdout_output.decode("utf-8")
    with open(h2o_sonar_config_path, "w") as text_file:
        text_file.write(stdout_output.decode("utf-8"))

    connection_config_dict = DAI_WORKER_CONNECTION.to_dict(encrypt=False)

    #
    # WHEN add connection
    #
    (cmd, child_env) = test_cli_test.given_base_cli_cmd()
    cmd = cmd + [
        "add",
        "config",
        "--config-type",
        "CONNECTION",
        "--config-value",
        json.dumps(connection_config_dict),
        "--config-path",
        str(h2o_sonar_config_path),
        "--encryption-key",
        encryption_key,
    ]
    print(f"\nRunning H2O Sonar to add DAI connection configuration:\n{cmd}\n")
    p = subprocess.Popen(cmd, env=child_env)
    p.wait()
    p_dump = str(os.popen(f"cat {h2o_sonar_config_path}").read())
    print(f"Config file:\n{p_dump}")
    assert DAI_WORKER_CONNECTION.key in p_dump
    assert "encrypted" in p_dump

    #
    # THEN
    #
    (cmd, child_env) = test_utils.given_base_cli_cmd()
    cmd = cmd + [
        "show",
        "config",
        "--config-path",
        str(h2o_sonar_config_path),
        "--encryption-key",
        encryption_key,
    ]
    print(f"\nRunning H2O Sonar:\n{cmd}\n")
    process = subprocess.Popen(cmd, env=child_env, stdout=subprocess.PIPE, stderr=None)
    process.wait()
    stdout_output, stderr_output = process.communicate()
    assert stderr_output is None
    assert DAI_WORKER_CONNECTION.key in stdout_output.decode("utf-8")


def test_dai_in_steam_connection():
    #
    # GIVEN
    #
    dai_steam_connection = h2o_sonar_config.ConnectionConfig(
        connection_type=h2o_sonar_config.ConnectionConfigType.DRIVERLESS_AI_STEAM.name,
        name="My Steam hosted DAI",
        description="Driverless AI server hosted by H2O Enterprise Steam.",
        # H2O Enterprise Steam server URL
        server_url="https://steam.h2o.ai/",
        # name of the Driverless AI server in H2O Enterprise Steam
        server_id="model-validation",
        # H2O Enterprise Steam client access token which might be obtained from:
        # Enterprise Steam > Configurations > Personal Access Token > Get token
        token="f00_1234567890abcdefghijklmnopqrstuvwxyz",
        # token use type: personal access token, refresh token, ...
        token_use_type=h2o_sonar_config.TokenUseType.ACCESS_TOKEN.name,
    )

    #
    # WHEN
    #
    dai_steam_connection_dict = dai_steam_connection.to_dict(encrypt=False)

    #
    # THEN
    #
    assert dai_steam_connection_dict["connection_type"] == "DRIVERLESS_AI_STEAM"


@pytest.mark.skip("H2O AIEM hosted Driverless AI is not yet supported")
def test_dai_in_aiem_connection():
    #
    # GIVEN
    #
    dai_aiem_connection = h2o_sonar_config.ConnectionConfig(
        connection_type=h2o_sonar_config.ConnectionConfigType.DRIVERLESS_AI_AIEM.name,
        # name of the Driverless AI server in H2O AIEM
        name="My AIEM hosted DAI",
        description="Driverless AI server hosted by H2O AIEM.",
        # H2O AIEM server URL
        server_id="new-dai-engine-42",
        # H2O AIEM environment name
        environment_url="https://cloud.h2o.ai/",
        # H2O AIEM client refresh token
        token="f00_1234567890abcdefghijklmnopqrstuvwxyz",
        # token use type: personal access token, refresh token, ...
        token_use_type=h2o_sonar_config.TokenUseType.REFRESH_TOKEN.name,
    )

    #
    # WHEN
    #
    dai_aiem_connection_dict = dai_aiem_connection.to_dict(encrypt=False)

    #
    # THEN
    #
    assert dai_aiem_connection_dict["connection_type"] == "DRIVERLESS_AI_AIEM"


def test_env_var_cfg():
    #
    # GIVEN
    #

    test_cfg = h2o_sonar_config.H2oSonarConfig()

    #
    # WHEN
    #

    test_cfg.env_and_override()

    #
    # THEN
    #
    print(f"\nH2O Sonar config:\n{json.dumps(test_cfg.to_dict(), indent=2)}")
    assert test_cfg


#
# tests
#


@pytest.mark.parametrize(
    "str_handle, expected",
    [
        (
            (
                "resource:connection:my-driverless-ai"
                ":key:1234567890-1234-1234-1234-123456789012"
                ":version:1"
            ),
            ("my-driverless-ai", "1234567890-1234-1234-1234-123456789012", "1"),
        ),
        (
            ("resource:connection:your-dai:key:1234567890-1234-1234-1234-123456789012"),
            ("your-dai", "1234567890-1234-1234-1234-123456789012", ""),
        ),
        (
            "resource:connection:my-connection:key:becede:version:1.10.5alpha",
            ("my-connection", "becede", "1.10.5alpha"),
        ),
        (
            (
                "resource:connection:my-connection-key"
                ":key:resource-key-version"
                ":version:version-1.10.5alpha"
            ),
            ("my-connection-key", "resource-key-version", "version-1.10.5alpha"),
        ),
    ],
    ids=["my-driverless-ai", "your-dai", "my-connection", "my-connection-key"],
)
@pytest.mark.h2o_sonar
def test_resource_handle_cli_arg_parsing(str_handle, expected):
    #
    # GIVEN
    #

    #
    # WHEN
    #
    (connection_name, key, version) = commons.ResourceHandle.parse_string_handle(
        str_handle
    )

    #
    # THEN
    #
    assert connection_name == expected[0]
    assert key == expected[1]
    assert version == expected[2]


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
