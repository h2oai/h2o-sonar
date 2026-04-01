# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import logging

import datatable
import pytest

from h2o_sonar import interpret
from h2o_sonar.lib.api import persistences
from tests import test_utils
from tests.lib import test_containers


@pytest.mark.skipif(not test_utils.is_mojo_supported(), reason="MOJO is not supported")
@pytest.mark.h2o_sonar
def test_test_container_mock_model_dataset_path(tmpdir):
    # GIVEN
    dataset_path = test_utils.find_locally("data/predictive/creditcard.csv")
    target_col = "SEX"
    mock_model = test_containers.SimpleMockModel(
        dataset_path=dataset_path, target_col=target_col
    )
    container = test_containers.ExplainerExamplesAndTemplatesTestContainer()
    container.setup(
        results_location=tmpdir,
        log_level=logging.DEBUG,
    )

    # WHEN
    i = interpret.run_interpretation(
        dataset=dataset_path,
        model=mock_model,
        target_col=target_col,
        explainer_keywords=[interpret.KEYWORD_FILTER_ALL],
        results_location=tmpdir,
        log_level=logging.DEBUG,
        container=container,
    )

    # THEN
    print(f"Interpretation:\n{i}")
    assert i
    failed_explainer = i.get_failed_explainer_ids()
    assert not failed_explainer, f"Failed explainers: {failed_explainer}"


@pytest.mark.skip(reason="To be fixed: filesystem used instead of in-memory")
@pytest.mark.h2o_sonar
def test_test_container_with_in_memory_persistence(tmpdir):
    # GIVEN
    dataset_path = test_utils.find_locally("data/predictive/creditcard.csv")
    target_col = "SEX"
    mock_model = test_containers.SimpleMockModel(
        dataset_path=dataset_path, target_col=target_col
    )
    container = test_containers.ExplainerExamplesAndTemplatesTestContainer()
    container.setup(
        results_location=tmpdir,
        persistence_type=persistences.PersistenceType.in_memory,
        log_level=logging.DEBUG,
    )

    # WHEN
    interpretation = interpret.run_interpretation(
        dataset=dataset_path,
        model=mock_model,
        target_col=target_col,
        explainer_keywords=[interpret.KEYWORD_FILTER_ALL],
        results_location=tmpdir,
        log_level=logging.DEBUG,
        container=container,
    )

    # THEN
    print(f"Interpretation:\n{interpretation}")
    assert interpretation

    test_utils.dump_in_memory_persistence(interpretation.persistence.store)
    assert list(interpretation.persistence.store.keys())

    failed_explainers = interpretation.get_failed_explainer_ids()
    assert not failed_explainers, f"Failed explainers: {failed_explainers}"


@pytest.mark.h2o_sonar
def test_interpret_mock_b_model_on_datatable(tmpdir):
    # GIVEN
    target_col = "binomial_target"
    train_dataset = datatable.Frame(
        {
            "id": [1, 2, 3, 4, 5],
            "int_feature": [1, 2, 3, 2, 3],
            "float_feature": [0.2, 0.1, 0.3, 0.1, 0.2],
            "str_feature": ["cat", "dog", "badger", "cat", "badger"],
            "regression_target": [0.1, 0.2, 0.3, 0.4, 0.5],
            target_col: [1, 0, 1, 0, 1],
            "multinomial_target": [
                "positive",
                "negative",
                "neutral",
                "positive",
                "negative",
            ],
        }
    )
    mock_model = test_containers.SimpleMockModel(
        target_col=target_col, dataset=train_dataset
    )
    container = test_containers.ExplainerExamplesAndTemplatesTestContainer()
    container.setup(
        results_location=tmpdir,
        persistence_type=persistences.PersistenceType.in_memory,
        log_level=logging.DEBUG,
    )

    # WHEN
    i = interpret.run_interpretation(
        dataset=train_dataset,
        model=mock_model,
        target_col=target_col,
        explainer_keywords=[interpret.KEYWORD_FILTER_ALL],
        results_location=tmpdir,
        log_level=logging.DEBUG,
        container=container,
    )

    # THEN
    print(f"Interpretation:\n{i}")
