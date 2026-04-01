# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import numpy as np
import pytest

from h2o_sonar.methods.core import _mli
from h2o_sonar.methods.surrogates import _decision_tree_h2o
from h2o_sonar.methods.surrogates import _surrogate_tree_h2o
from tests.conftest import get_h2o3_config


@pytest.mark.skipif(
    not _surrogate_tree_h2o.HAS_H2O,
    reason="H2O-3 Python package is not installed",
)
def test_dt_instantiation(h2o3_cleanup_fixture):
    # GIVEN + WHEN
    dt = _surrogate_tree_h2o.TreeSurrogateH2O.instantiate(
        backend=_surrogate_tree_h2o.H2OTreeBackend.DECISIONTREE
    )

    # THEN
    assert isinstance(dt, _decision_tree_h2o.DecisionTreeH2O)


@pytest.mark.skipif(
    not _surrogate_tree_h2o.HAS_H2O,
    reason="H2O-3 Python package is not installed",
)
def test_dt_instantiation_fail(h2o3_cleanup_fixture):
    # GIVEN + WHEN + THEN
    with pytest.raises(ValueError):
        _surrogate_tree_h2o.TreeSurrogateH2O.instantiate(
            backend="none-existing-backend"
        )


@pytest.mark.skipif(
    not _surrogate_tree_h2o.HAS_H2O,
    reason="H2O-3 Python package is not installed",
)
def test_dt_save_load(tmpdir, h2o3_cleanup_fixture):
    # GIVEN
    mli = _mli.MLI(work_dir=str(tmpdir), seed=1234, config=get_h2o3_config())
    # Model as data
    data = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    labels = [1, 2, 3]
    model = mli.wrap("test_model_basic", data, labels)

    # WHEN
    dt = _surrogate_tree_h2o.TreeSurrogateH2O.instantiate(
        _surrogate_tree_h2o.H2OTreeBackend.DECISIONTREE, max_depth=3, nfolds=0
    )
    dt.fit(model, response_column=3)
    predictions = dt.predict(data)

    # THEN
    np.testing.assert_almost_equal(predictions, labels, decimal=1)

    dt.save()

    loaded_dt = _surrogate_tree_h2o.TreeSurrogateH2O.load(
        mli.work_dir, "test_model_basic"
    )
    np.testing.assert_equal(predictions, loaded_dt.predict(data))


@pytest.mark.skipif(
    not _surrogate_tree_h2o.HAS_H2O,
    reason="H2O-3 Python package is not installed",
)
def test_dt_const_col(tmpdir, h2o3_cleanup_fixture):
    # GIVEN
    mli = _mli.MLI(work_dir=str(tmpdir), seed=1234, config=get_h2o3_config())
    # Model as data
    data = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    labels = [3, 3, 3]
    model = mli.wrap("test_model_basic", data, labels)

    # WHEN
    dt = _surrogate_tree_h2o.TreeSurrogateH2O.instantiate(
        _surrogate_tree_h2o.H2OTreeBackend.DECISIONTREE,
        max_depth=3,
        nfolds=0,
        check_constant_response=False,
    )
    dt.fit(model, response_column=3)
    predictions = dt.predict(data)

    # THEN
    np.testing.assert_almost_equal(predictions, labels, decimal=1)
