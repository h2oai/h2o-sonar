# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import logging

import pytest

from h2o_sonar import interpret
from tests import test_utils
from tests.lib import test_containers


try:
    import h2o
    from h2o.estimators.gbm import H2OGradientBoostingEstimator

    HAS_H2O = True
except ImportError:
    HAS_H2O = False


@pytest.mark.skipif(
    not HAS_H2O,
    reason="H2O-3 Python package is not installed",
)
@pytest.mark.skipif(
    test_utils.GitHubActions.is_in_gha(),
    reason="Skipped on GHA as this test has high memory usage (ran on MMC/self-hosted)",
)
@pytest.mark.parametrize(
    "use_explainable_model,use_path",
    [
        [True, True],
        [False, False],
        [False, True],
        [True, False],
    ],
)
@pytest.mark.h2o_sonar
def test_h2o_all_examples_and_templates(tmpdir, use_explainable_model, use_path):
    #
    # GIVEN
    #
    # connect to H2O-3 cluster
    test_utils.h2o3_init_for_tests()
    # container
    container = test_containers.ExplainerExamplesAndTemplatesTestContainer()
    container.setup(
        results_location=tmpdir,
        log_level=logging.DEBUG,
    )
    # dataset
    dataset_path = test_utils.find_locally("data/predictive/creditcard.csv")
    target_col = "default payment next month"
    df = h2o.import_file(dataset_path)
    df[target_col] = df[target_col].asfactor()
    X = list(df.names)
    X.remove(target_col)

    # h2o model
    model = H2OGradientBoostingEstimator(ntrees=1, seed=1234)
    model.train(x=X, y=target_col, training_frame=df)

    # explainable model
    if use_explainable_model:
        model = container.model_api.create_model(
            model_src=model,
            target_col=target_col,
            used_features=X,
        )
    #
    # WHEN
    #
    interpretation = interpret.run_interpretation(
        dataset=df if not use_path else dataset_path,
        model=model,
        target_col=target_col,
        explainer_keywords=[interpret.KEYWORD_FILTER_ALL],
        container=container,
        results_location=tmpdir,
        log_level=logging.DEBUG,
    )
    #
    # THEN
    #
    test_utils.assert_interpretation(interpretation)
