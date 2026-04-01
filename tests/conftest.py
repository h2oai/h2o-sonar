# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import pytest

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar import loggers
from h2o_sonar.methods.utils import h2o_utils
from tests import test_utils


try:
    import h2o  # noqa: F401

    HAS_H2O = True
except ImportError:
    HAS_H2O = False

# avoid DEADLOCKS in case of multiprocessing used in tests (see configuration)
h2o_sonar_config.config.mp_start_method = (
    h2o_sonar_config.H2oSonarConfig.MP_START_METHOD_SPAWN
)

# H2O-3 configuration singleton
h2o3_config = None
# control H2O Sonar CPU and memory profiler
h2o_sonar_config.config.enable_profiler = False


def get_h2o3_config():
    return h2o3_config


@pytest.fixture(scope="session", autouse=True)
def h2o3_init_fixture():
    global h2o3_config
    # skip H2O-3 initialization if it is not available (optional dependency)
    if not HAS_H2O:
        return None
    h2o_sonar_config.config.h2o_auto_start = False
    h2o3_config = h2o_utils.start_h2o3(
        test_utils.GitHubActions.get_h2o3_config()
        if test_utils.GitHubActions.is_in_gha()
        else None
    )
    return h2o3_config


@pytest.fixture(scope="session", autouse=True)
def h2o3_shutdown_at_end():
    """Ensure H2O-3 cluster is shut down at end of test session."""
    yield
    # after all tests complete, shut down any running clusters
    if HAS_H2O:
        try:
            h2o_utils.kill_h2o3()
        except Exception as ex:
            loggers.SonarPrintLogger().warning(
                f"WARNING: H2O-3 cluster was not shut down at end of test session: {ex}"
            )
            pass  # best-effort shutdown


@pytest.fixture
def h2o3_cleanup_fixture():
    """Ensure H2O-3 cleanup after test execution.

    This fixture guarantees H2O-3 frames and models are cleaned up after
    each test to prevent memory accumulation in the Java heap.

    Use this for tests that use the session-scoped h2o3_init_fixture
    and don't need custom H2O-3 configuration.

    The fixture follows pytest's yield-based teardown pattern, ensuring
    cleanup runs even if the test fails or raises an exception.
    """
    # STEP 1: pytest SETUP phase - nothing needed > use session-scoped h2o3-init-fixture

    # STEP 2: pytest waits here while the test which has this fixture as arg is running
    yield

    # STEP 3: teardown - after test w/ fixture arg finishes - guaranteed to run
    h2o_utils.clean_up_h2o3()


@pytest.fixture
def h2ogpte_connection_fixture():
    """Lazy-evaluated h2oGPTe connection fixture.

    This fixture provides safe lazy evaluation of h2oGPTe connection for use in tests,
    preventing errors during pytest collection phase when API keys are not available.

    Returns
    -------
    h2o_sonar_config.ConnectionConfig
        h2oGPTe connection configuration.

    Raises
    ------
    pytest.skip
        If test services config is not available or connection cannot be established.

    """
    from tests.lib import given_generative

    if not given_generative.is_config():
        pytest.skip("Test services config not available")

    connection = test_utils.health.get_h2ogpte()
    if connection is None:
        pytest.skip("h2oGPTe connection could not be established")

    return connection


@pytest.fixture
def h2ogpt_connection_fixture():
    """Lazy-evaluated h2oGPT connection fixture.

    This fixture provides safe lazy evaluation of h2oGPT connection for use in tests,
    preventing errors during pytest collection phase when API keys are not available.

    Returns
    -------
    h2o_sonar_config.ConnectionConfig
        h2oGPT connection configuration.

    Raises
    ------
    pytest.skip
        If test services config is not available or connection cannot be established.

    """
    from tests.lib import given_generative

    if not given_generative.is_config():
        pytest.skip("Test services config not available")

    connection = test_utils.health.get_h2ogpt()
    if connection is None:
        pytest.skip("h2oGPT connection could not be established")

    return connection


@pytest.fixture
def openai_azure_connection_fixture():
    """Lazy-evaluated Azure OpenAI connection fixture.

    This fixture provides safe lazy evaluation of Azure OpenAI connection for use in
    tests, preventing errors during pytest collection phase when API keys are not
    available.

    Returns
    -------
    h2o_sonar_config.ConnectionConfig
        Azure OpenAI connection configuration.

    Raises
    ------
    pytest.skip
        If Azure OpenAI is not configured or connection cannot be established.

    """
    if not test_utils.health.is_azure_openai():
        pytest.skip("Azure OpenAI not configured")

    connection = test_utils.health.get_openai_azure()
    if connection is None:
        pytest.skip("Azure OpenAI connection could not be established")

    return connection


@pytest.fixture
def anthropic_connection_fixture():
    """Lazy-evaluated Anthropic Claude connection fixture.

    This fixture provides safe lazy evaluation of Anthropic connection for use in tests,
    preventing errors during pytest collection phase when API keys are not available.

    Returns
    -------
    h2o_sonar_config.ConnectionConfig
        Anthropic connection configuration.

    Raises
    ------
    pytest.skip
        If Anthropic is not configured or connection cannot be established.

    """
    if not test_utils.health.is_anthropic():
        pytest.skip("Anthropic not configured")

    connection = test_utils.health.get_anthropic()
    if connection is None:
        pytest.skip("Anthropic connection could not be established")

    return connection


@pytest.fixture
def ollama_connection_fixture():
    """Lazy-evaluated ollama connection fixture.

    This fixture provides safe lazy evaluation of ollama connection for use in tests,
    preventing errors during pytest collection phase when ollama is not available.

    Returns
    -------
    h2o_sonar_config.ConnectionConfig
        ollama connection configuration.

    Raises
    ------
    pytest.skip
        If ollama is not configured or connection cannot be established.

    """
    if not test_utils.health.is_ollama():
        pytest.skip("ollama not configured")

    connection = test_utils.health.get_ollama()
    if connection is None:
        pytest.skip("ollama connection could not be established")

    return connection
