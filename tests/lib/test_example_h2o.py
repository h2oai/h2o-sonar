# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import logging

import pytest

from h2o_sonar import interpret
from tests import test_utils


try:
    import h2o
    from h2o.estimators import H2OGradientBoostingEstimator

    HAS_H2O = True
except ImportError:
    HAS_H2O = False


@pytest.mark.skipif(
    not HAS_H2O,
    reason="H2O-3 Python package is not installed",
)
@pytest.mark.h2o_sonar
def test_h2o_docs_example(tmpdir, h2o3_cleanup_fixture):
    test_utils.h2o3_init_for_tests()

    # GIVEN
    dataset_path = test_utils.find_locally("data/predictive/creditcard.csv")
    target_col = "default payment next month"
    df = h2o.import_file(dataset_path)
    X = list(df.names)
    X.remove(target_col)

    gradient_booster = H2OGradientBoostingEstimator(ntrees=1, seed=1234)
    gradient_booster.train(
        x=X,
        y=target_col,
        training_frame=df,
        verbose=True,
    )

    # WHEN
    interpretation = interpret.run_interpretation(
        dataset=dataset_path,
        model=gradient_booster,
        target_col=target_col,
        results_location=tmpdir,
        explainer_keywords=[interpret.KEYWORD_FILTER_ALL],
        log_level=logging.DEBUG,
    )

    # THEN
    test_utils.assert_interpretation(interpretation)
