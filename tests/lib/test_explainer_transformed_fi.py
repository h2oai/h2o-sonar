# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import logging
import os.path

import pytest

from h2o_sonar import interpret
from h2o_sonar.explainers import transformed_fi_shapley_explainer as explainer
from h2o_sonar.lib.api import commons
from tests import test_utils


@pytest.mark.h2o_sonar
@pytest.mark.parametrize(
    "experiment_type,mojo_path,dataset_path,target_col",
    [
        (
            commons.ExperimentType.regression,
            "data/predictive/models/creditcard-regression.mojo",
            "data/predictive/pd_ice_creditcard_10_rows.csv",
            "LIMIT_BAL",
        ),
        (
            commons.ExperimentType.binomial,
            "data/predictive/models/creditcard-binomial.mojo",
            "data/predictive/creditcard.csv",
            "default payment next month",
        ),
        (
            commons.ExperimentType.multinomial,
            "data/predictive/models/iris-multinomial.mojo",
            "data/predictive/iris.csv",
            "class",
        ),
    ],
    ids=["CC-regression", "CC-binomial", "iris-multinomial"],
)
def test_dai_mojo(tmpdir, experiment_type, mojo_path, dataset_path, target_col):
    import daimojo

    #
    # GIVEN
    #
    # dataset
    dataset_path = test_utils.find_locally(dataset_path)
    mojo_path = test_utils.find_locally(mojo_path)
    # DAI model
    mojo_model = daimojo.model(mojo_path)
    # container
    container = interpret.resolve_container()
    # explainable model
    explainable_model = container.model_api.create_model(
        model_src=mojo_model,
        target_col=target_col,
        used_features=list(mojo_model.feature_names),
    )
    print(f"Explainable model: {explainable_model}")

    #
    # WHEN
    #
    interpretation = interpret.run_interpretation(
        dataset=dataset_path,
        model=explainable_model,
        target_col=target_col,
        explainers=[
            explainer.ShapleyMojoTransformedFeatureImportanceExplainer.explainer_id()
        ],
        results_location=tmpdir,
        log_level=logging.DEBUG,
        container=container,
    )

    #
    # THEN
    #
    print(f"\n{interpretation}")
    # find failed explainers
    assert interpretation
    assert interpretation.result.explainers
    assert len(interpretation.result.explainers) == 1
    failed_explainers = interpretation.get_failed_explainer_ids()
    assert not failed_explainers, f"Failed explainers: {failed_explainers}"

    # result
    result = interpretation.get_explainer_result(
        explainer.ShapleyMojoTransformedFeatureImportanceExplainer.explainer_id()
    )
    print(f"Explainer result: {result}")
    assert result
    print(f"Explainer result SUMMARY: {result.summary()}")
    assert result.summary()
    print(f"Explainer result PARAMS: {result.params()}")
    assert result.params()
    print(f"Explainer result DATA: {result.data().to_dict()}")
    assert result.data()
    print("Explainer result PLOT...")
    result.plot()
    print("Explainer result LOG...")
    result.log(path=os.path.join(tmpdir, "feature-importance-demo.log"))


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
