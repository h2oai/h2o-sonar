# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import logging
import os.path

import datatable
import pytest

from h2o_sonar import interpret
from h2o_sonar.explainers.fi_naive_shapley_explainer import (
    NaiveShapleyMojoFeatureImportanceExplainer as explainer,
)
from h2o_sonar.lib.api import commons
from tests import test_utils


@pytest.mark.h2o_sonar
@pytest.mark.parametrize(
    (
        "experiment_type,mojo_path,dataset_path,target_col,"
        "problem_threshold,problem_should_appear"
    ),
    [
        (
            commons.ExperimentType.regression,
            "data/predictive/models/creditcard-regression.mojo",
            "data/predictive/pd_ice_creditcard_10_rows.csv",
            "LIMIT_BAL",
            0.999,
            False,
        ),
        (
            commons.ExperimentType.binomial,
            "data/predictive/models/creditcard-binomial.mojo",
            "data/predictive/creditcard.csv",
            "default payment next month",
            0.3,
            True,
        ),
        (
            commons.ExperimentType.multinomial,
            "data/predictive/models/iris-multinomial.mojo",
            "data/predictive/iris.csv",
            "class",
            0.9,
            True,
        ),
    ],
    ids=["cc_regression", "cc_binomial", "iris_multinomial"],
)
def test_dai_mojo_original_fi(
    tmpdir,
    experiment_type,
    mojo_path,
    dataset_path,
    target_col,
    problem_threshold,
    problem_should_appear,
):
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
        dataset=dataset_path,
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
            commons.ExplainerToRun(
                explainer.explainer_id(),
                {explainer.PARAM_LEAKAGE_WARN_THRESHOLD: problem_threshold},
            )
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
    result = interpretation.get_explainer_result(explainer.explainer_id())

    if problem_should_appear:
        if experiment_type == commons.ExperimentType.multinomial:
            assert len(interpretation.result.problems) > 0
        else:
            assert len(interpretation.result.problems) == 1
    else:
        assert len(interpretation.result.problems) == 0

    print(f"Explainer result: {result}")
    assert result
    print(f"Explainer result SUMMARY: {result.summary()}")
    assert result.summary()
    print(f"Explainer result DATA: {result.data().to_dict()}")
    assert result.data()
    # assert multinomial
    if target_col == "class":
        assert "feature" in result.data().to_dict()
        assert ["petal_len", "sepal_len"] == result.data().to_dict()["feature"]
    print("Explainer result PLOT...")
    result.plot()
    print("Explainer result LOG...")
    result.log(path=os.path.join(tmpdir, "feature-importance-demo.log"))


@pytest.mark.skip("C++ MOJO introspection experiment")
@pytest.mark.h2o_sonar
@pytest.mark.parametrize(
    "experiment_type,mojo_path,dataset_path",
    [
        (
            commons.ExperimentType.regression,
            "data/predictive/models/creditcard-regression.mojo",
            "data/predictive/pd_ice_creditcard_10_rows.csv",
        ),
        (
            commons.ExperimentType.binomial,
            "data/predictive/models/creditcard-binomial.mojo",
            "data/predictive/creditcard.csv",
        ),
        (
            commons.ExperimentType.multinomial,
            "data/predictive/models/iris-multinomial.mojo",
            "data/predictive/iris.csv",
        ),
    ],
    ids=["cc_regression", "cc_binomial", "iris_multinomial"],
)
def test_dai_mojo_introspection(experiment_type, mojo_path, dataset_path):
    import daimojo

    #
    # GIVEN
    #
    dataset_path = test_utils.find_locally(dataset_path)
    x = datatable.fread(dataset_path)
    mojo_path = test_utils.find_locally(mojo_path)
    model = daimojo.model(mojo_path)

    #
    # WHEN
    #
    print(f"Dataset cols : {x.names}")
    print(f"Feature names: {model.feature_names}")
    print(f"Feature types: {model.feature_types}")
    print(f"Output names : {model.output_names}")
    print(f"Output types : {model.output_types}")
    print(f"Has tree SHAP: {model.has_treeshap}")
    assert model.has_treeshap
    # Shapley contributions for ORIGINAL features
    original_shapleys = model.predict(
        x, pred_contribs=True, pred_contribs_original=True
    )
    # Shapley contributions for TRANSFORMED features
    transformed_shapleys = model.predict(x, pred_contribs=True)

    #
    # WHEN
    #
    # ORIGINAL: contrib_<original feature name>, ..., contrib_bias
    print(f"Original Shapley values:\n{original_shapleys}")
    assert original_shapleys
    # ORIGINAL: contrib_<transformed feature name>, ..., contrib_bias
    print(f"Transformed Shapley values:\n{transformed_shapleys}")
    assert transformed_shapleys


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
