# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import logging
import time
from concurrent import futures

import pytest

from h2o_sonar import interpret
from h2o_sonar.explainers import dia_explainer
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import interpretations
from tests import conftest
from tests import test_utils
from tests.explainers.templates.template_pd_explainer import (
    TemplatePartialDependenceExplainer,
)
from tests.lib import test_containers


i_dict_keys = interpretations.Interpretation


@pytest.mark.h2o_sonar
def test_list_explainers():
    # GIVEN
    expected_explainers = 5

    # WHEN
    explainers = interpret.list_explainers()

    # THEN
    print(f"Explainers:\n{explainers}")
    assert explainers
    assert expected_explainers <= len(explainers)
    explainers_ids = [e.id for e in explainers]
    print(f"Explainers IDs:\n{explainers_ids}")
    str_explainers = [str(e) for e in explainers]
    print(f"Explainers descriptors:\n{str_explainers}")


@pytest.mark.h2o_sonar
def test_default_explainers(tmpdir):
    """Test DEFAULT explainers execution by the local container."""

    # GIVEN
    dataset_path = test_utils.find_locally("data/predictive/creditcard.csv")
    target_col = "SEX"
    mock_model = test_containers.SimpleMockModel(
        dataset_path=dataset_path, target_col=target_col
    )
    print(f"H2O-3 configuration: {conftest.get_h2o3_config()}")

    # WHEN
    i = interpret.run_interpretation(
        dataset=dataset_path,
        model=mock_model,
        target_col=target_col,
        results_location=tmpdir,
        log_level=logging.DEBUG,
    )

    # THEN
    print(f"Interpretation: {i}")
    i_dict = i.to_dict()
    assert i
    expected = 5
    assert expected == len(
        i_dict[i_dict_keys.KEY_RESULT][i_dict_keys.KEY_SCHEDULED_EXPLAINERS]
    )
    assert expected == len(i.get_scheduled_explainer_ids())
    assert expected == len(i.get_finished_explainer_ids())
    assert expected == len(i.get_successful_explainer_ids())


@pytest.mark.h2o_sonar
def test_all_explainers(tmpdir):
    """Test ALL explainers (including non-default) execution by the local container."""

    # GIVEN
    dataset_path = test_utils.find_locally("data/predictive/creditcard.csv")
    target_col = "SEX"
    mock_model = test_containers.SimpleMockModel(
        dataset_path=dataset_path, target_col=target_col
    )

    # WHEN
    i = interpret.run_interpretation(
        dataset=dataset_path,
        model=mock_model,
        target_col=target_col,
        explainer_keywords=[interpret.KEYWORD_FILTER_ALL],
        results_location=tmpdir,
        log_level=logging.DEBUG,
    )

    # THEN
    print(f"Interpretation: {i}")
    i_dict = i.to_dict()
    assert i
    expected = 3
    assert expected <= len(
        i_dict[i_dict_keys.KEY_RESULT][i_dict_keys.KEY_SCHEDULED_EXPLAINERS]
    )
    assert expected <= len(i.get_scheduled_explainer_ids())
    assert expected <= len(i.get_finished_explainer_ids())
    assert expected <= len(i.get_successful_explainer_ids())


@pytest.mark.h2o_sonar
def test_keyword_explainers(tmpdir):
    """Test explainers with specific KEYWORD execution by the local container."""

    # GIVEN
    dataset_path = test_utils.find_locally("data/predictive/creditcard.csv")
    target_col = "SEX"
    mock_model = test_containers.SimpleMockModel(
        dataset_path=dataset_path, target_col=target_col
    )

    # WHEN
    i = interpret.run_interpretation(
        dataset=dataset_path,
        model=mock_model,
        target_col=target_col,
        explainer_keywords=["surrogate"],
        results_location=tmpdir,
        log_level=logging.DEBUG,
    )

    # THEN
    print(f"Interpretation: {i}")
    i_dict = i.to_dict()
    assert i
    expected = 2
    assert expected == len(
        i_dict[i_dict_keys.KEY_RESULT][i_dict_keys.KEY_SCHEDULED_EXPLAINERS]
    )
    assert expected == len(i.get_scheduled_explainer_ids())
    assert expected == len(i.get_finished_explainer_ids())
    assert expected == len(i.get_successful_explainer_ids())


@pytest.mark.h2o_sonar
@pytest.mark.parametrize(
    "explainer_to_run",
    [
        # specify explainer by ID
        TemplatePartialDependenceExplainer.explainer_id(),
        # specify explainer w/ params
        commons.ExplainerToRun(
            explainer_id=TemplatePartialDependenceExplainer.explainer_id(),
            params="{'parameter': 'value'}",
        ),
    ],
    ids=[
        "TemplatePartialDependenceExplainer",
        "TemplatePartialDependenceExplainer_with_params",
    ],
)
def test_1_explainer(tmpdir, explainer_to_run):
    """Test explicitly explainers to be run."""

    # GIVEN
    dataset_path = test_utils.find_locally("data/predictive/creditcard.csv")
    target_col = "SEX"
    mock_model = test_containers.SimpleMockModel(
        dataset_path=dataset_path, target_col=target_col
    )

    # WHEN
    i = interpret.run_interpretation(
        dataset=dataset_path,
        model=mock_model,
        target_col=target_col,
        explainers=[explainer_to_run],
        results_location=tmpdir,
        log_level=logging.DEBUG,
    )

    # THEN
    print(f"Interpretation: {i}")


@pytest.mark.h2o_sonar
@pytest.mark.skipif(not test_utils.is_mojo_supported(), reason="MOJO is not supported")
def test_async_1_explainer(tmpdir):
    """Test asynchronous interpretation execution."""

    def print_future(i_future):
        print(f"  Running? {i_future.running()}")
        print(f"  Done? {i_future.done()}")
        print(f"  Cancelled? {i_future.cancelled()}")

    # GIVEN
    dataset_path = test_utils.find_locally("data/predictive/creditcard_str_10k.csv")
    model_path = test_utils.find_locally(
        "data/predictive/models/creditcard-binomial-str.mojo"
    )
    target_col = "DEFAULT_NEXT_MONTH"
    timeout = 10 * 60  # seconds for the whole interpretation run

    # WHEN
    interpretation = interpret.run_interpretation(
        run_asynchronously=True,
        dataset=dataset_path,
        model=model_path,
        target_col=target_col,
        explainers=[
            commons.ExplainerToRun(
                explainer_id=dia_explainer.DiaExplainer.explainer_id(),
                params="{'parameter': 'value'}",
            )
        ],
        results_location=tmpdir,
        log_level=logging.DEBUG,
    )

    # THEN
    print(f"Interpretation dump: {interpretation}")
    assert interpretation
    print(f"Interpretation future: {interpretation.future}")
    assert interpretation.future
    assert isinstance(interpretation.future, futures.Future)
    print_future(interpretation.future)

    countdown = timeout * 2
    while interpretation.future.running() and countdown > 0:
        countdown -= 1
        print(f"{countdown}")
        time.sleep(0.5)
    assert timeout > 0, (
        f"Async interpretation run timed out - it did not finish in {countdown}s"
    )

    print("Finished:")
    print_future(interpretation.future)
    assert interpretation.future.done()
    result = interpretation.future.result()
    print(f"Result: {result}")
    assert result
    assert isinstance(result, dict)
    assert interpretation.key == result.get(interpretations.Interpretation.KEY_I_KEY)

    # THEN: cleanup
    interpret.do_gc()


@pytest.mark.h2o_sonar
@pytest.mark.skipif(not test_utils.is_mojo_supported(), reason="MOJO is not supported")
def test_async_1_explainer_crash(tmpdir):
    """Test asynchronous explainer execution failure: status, progress and error
    message reporting.

    """

    def print_future(i_future):
        print(f"  Running? {i_future.running()}")
        print(f"  Done? {i_future.done()}")
        print(f"  Cancelled? {i_future.cancelled()}")

    # GIVEN
    dataset_path = test_utils.find_locally("data/predictive/creditcard_str_10k.csv")
    model_path = test_utils.find_locally(
        "data/predictive/models/creditcard-binomial-str.mojo"
    )
    # crash the interpretation
    target_col = "WRONG_TARGET"
    timeout = 10 * 60  # seconds for the whole interpretation run

    # WHEN
    interpretation = interpret.run_interpretation(
        run_asynchronously=True,
        dataset=dataset_path,
        model=model_path,
        target_col=target_col,
        explainers=[
            commons.ExplainerToRun(
                explainer_id=dia_explainer.DiaExplainer.explainer_id(),
                params="{'parameter': 'value'}",
            )
        ],
        results_location=tmpdir,
        log_level=logging.DEBUG,
    )

    # THEN
    print(f"Interpretation dump: {interpretation}")
    assert interpretation
    print(f"Interpretation future: {interpretation.future}")
    assert interpretation.future
    assert isinstance(interpretation.future, futures.Future)
    print_future(interpretation.future)

    countdown = timeout * 2
    while interpretation.future.running() and countdown > 0:
        countdown -= 1
        print(f"{countdown}")
        time.sleep(0.5)
    assert timeout > 0, (
        f"Async interpretation run timed out - it did not finish in {countdown}s"
    )

    interpret.do_gc()

    print("Finished:")
    print_future(interpretation.future)
    assert interpretation.future.done()
    exception_raised = ""
    try:
        interpretation.future.result()
    except Exception as ex:
        exception_raised = str(ex)

    assert target_col in exception_raised


@pytest.mark.h2o_sonar
def test_list_interpretations(tmpdir):
    # GIVEN
    interpretations_count = 2
    for _ in range(interpretations_count):
        test_keyword_explainers(tmpdir)

    # WHEN
    interpretations_keys = interpret.list_interpretations(
        results_location=tmpdir,
        log_level=logging.DEBUG,
    )

    # THEN
    print(f"Interpretations:\n{interpretations_keys}")
    assert interpretations_keys
    assert interpretations_count == len(interpretations_keys)
