# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import numpy as np
import pytest

from h2o_sonar.methods.core import _mli
from h2o_sonar.methods.surrogates import _random_forest_h2o
from h2o_sonar.methods.surrogates import _surrogate_tree_h2o
from tests.conftest import get_h2o3_config


@pytest.mark.skipif(
    not _surrogate_tree_h2o.HAS_H2O,
    reason="H2O-3 Python package is not installed",
)
def test_rf_instantiation(h2o3_cleanup_fixture):
    # GIVEN + WHEN
    rf = _surrogate_tree_h2o.TreeSurrogateH2O.instantiate(
        backend=_surrogate_tree_h2o.H2OTreeBackend.RANDOMFOREST
    )

    # THEN
    assert isinstance(rf, _random_forest_h2o.RandomForestH2O)


@pytest.mark.skipif(
    not _surrogate_tree_h2o.HAS_H2O,
    reason="H2O-3 Python package is not installed",
)
def test_rf_instantiation_fail(h2o3_cleanup_fixture):
    # GIVEN + WHEN + THEN
    with pytest.raises(ValueError):
        _surrogate_tree_h2o.TreeSurrogateH2O.instantiate(
            backend="none-existing-backend"
        )


@pytest.mark.skipif(
    not _surrogate_tree_h2o.HAS_H2O,
    reason="H2O-3 Python package is not installed",
)
def test_rf_save_load(tmpdir, h2o3_cleanup_fixture):
    # GIVEN
    mli = _mli.MLI(work_dir=str(tmpdir), seed=1234, config=get_h2o3_config())
    # Model as data
    data = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    labels = [1, 2, 3]
    model = mli.wrap("test_model_basic", data, labels)

    # WHEN
    rf = _surrogate_tree_h2o.TreeSurrogateH2O.instantiate(
        _surrogate_tree_h2o.H2OTreeBackend.RANDOMFOREST, max_depth=3, nfolds=0
    )
    rf.fit(model, response_column=3)
    predictions = rf.predict(data)

    # THEN
    np.testing.assert_almost_equal(predictions, labels, decimal=1)

    rf.save()

    loaded_rf = _surrogate_tree_h2o.TreeSurrogateH2O.load(
        mli.work_dir, "test_model_basic"
    )
    np.testing.assert_equal(predictions, loaded_rf.predict(data))
