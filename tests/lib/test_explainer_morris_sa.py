# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import os

import pytest

from h2o_sonar import interpret
from h2o_sonar import loggers
from h2o_sonar.explainers import morris_sa_explainer
from h2o_sonar.lib.api import commons
from tests import test_utils
from tests.lib import test_containers


# constants
MorrisSaExplainer = morris_sa_explainer.MorrisSensitivityAnalysisExplainer


@pytest.mark.skipif(
    not test_utils.are_python_modules_installed({"interpret"}),
    reason="ML 'interpret' Python package is not installed",
)
@pytest.mark.h2o_sonar
@pytest.mark.parametrize(
    "model_type,problem_threshold,problem_should_appear",
    [
        (commons.ExperimentType.regression, 0.999, False),
        (commons.ExperimentType.binomial, 0.001, True),
    ],
    ids=["regression", "binomial"],
)
def test_mock(
    tmpdir, model_type, problem_threshold: float, problem_should_appear: bool
):
    # GIVEN
    dataset_path = test_utils.find_locally("data/predictive/creditcard.csv")
    if commons.ExperimentType.regression == model_type:
        target_col = "AGE"
    elif commons.ExperimentType.binomial == model_type:
        target_col = "SEX"
    else:
        raise ValueError(f"Unsupported model type: '{model_type}'")

    mock_model = test_containers.SimpleMockModel(
        dataset_path=dataset_path,
        target_col=target_col,
        model_type=model_type,
    )
    container = test_containers.ExplainerExamplesAndTemplatesTestContainer()
    container.setup(
        results_location=tmpdir,
        log_level=loggers.DEBUG,
    )
    container.explainers_registry.register(
        explainer_class=MorrisSaExplainer,
    )

    # WHEN
    try:
        assert interpret.describe_explainer(MorrisSaExplainer)
        interpretation = interpret.run_interpretation(
            dataset=dataset_path,
            model=mock_model,
            target_col=target_col,
            explainers=[
                commons.ExplainerToRun(
                    MorrisSaExplainer.explainer_id(),
                    {MorrisSaExplainer.PARAM_LEAKAGE_WARN_THRESHOLD: problem_threshold},
                )
            ],
            results_location=tmpdir,
            container=container,
            log_level=loggers.DEBUG,
        )

        # THEN
        print(f"Interpretation:\n{interpretation}")

        assert interpretation
        assert interpretation.is_explainer_scheduled()
        assert interpretation.is_explainer_finished()
        assert interpretation.is_explainer_successful()
        assert not interpretation.is_explainer_failed()
        assert interpretation.get_scheduled_explainer_ids()
        assert interpretation.get_finished_explainer_ids()
        assert interpretation.get_successful_explainer_ids()
        assert not interpretation.get_failed_explainer_ids()
        assert interpretation.get_explainer_result_metadata(
            MorrisSaExplainer.explainer_id()
        )
        result = interpretation.get_explainer_result(MorrisSaExplainer.explainer_id())
        assert result, "Explainer Result cannot be None"
        data = result.data()
        print(data)
        assert data

        if problem_should_appear:
            assert len(interpretation.result.problems) == 1
        else:
            assert len(interpretation.result.problems) == 0

        assert result.summary()
        # why was here not supposed to be a summary?

    finally:
        container.explainers_registry.unregister(MorrisSaExplainer.explainer_id())


@pytest.mark.skipif(
    not test_utils.are_python_modules_installed({"interpret"}),
    reason="ML 'interpret' Python package is not installed",
)
@pytest.mark.h2o_sonar
@pytest.mark.parametrize(
    "dataset, target_col",
    [
        (
            "creditcard.csv",
            "LIMIT_BAL",
        ),
        (
            "creditcard.csv",
            "default payment next month",
        ),
    ],
    ids=["CC-regression", "CC-binomial"],
)
def test_sklearn(tmpdir, dataset, target_col):
    # GIVEN
    (dataset_path, explainable_model, target_col) = test_utils.create_sklearn_model(
        dataset_name=dataset, target_col=target_col
    )

    # WHEN
    interpretation = interpret.run_interpretation(
        dataset=dataset_path,
        model=explainable_model,
        target_col=target_col,
        explainers=[MorrisSaExplainer.explainer_id()],
        results_location=tmpdir,
        log_level=loggers.DEBUG,
    )

    # THEN
    print(f"Interpretation:\n{interpretation}")
    assert interpretation
    assert interpretation.is_explainer_scheduled()
    assert interpretation.is_explainer_finished()
    assert interpretation.is_explainer_successful()
    assert not interpretation.is_explainer_failed()
    assert interpretation.get_scheduled_explainer_ids()
    assert interpretation.get_finished_explainer_ids()
    assert interpretation.get_successful_explainer_ids()
    assert not interpretation.get_failed_explainer_ids()
    assert interpretation.get_explainer_result(MorrisSaExplainer.explainer_id())


@pytest.mark.skipif(
    not test_utils.are_python_modules_installed({"interpret"}),
    reason="ML 'interpret' Python package is not installed",
)
@pytest.mark.h2o_sonar
@pytest.mark.parametrize(
    "experiment_type,mojo_path,dataset_path,target_col",
    [
        (
            commons.ExperimentType.regression,
            "data/predictive/models/creditcard-regression.mojo",
            # has categorical features > label encoder is tested
            "data/predictive/pd_ice_creditcard_train.csv",
            "LIMIT_BAL",
        ),
        (
            commons.ExperimentType.binomial,
            "data/predictive/models/creditcard-binomial.mojo",
            "data/predictive/creditcard.csv",
            "default payment next month",
        ),
    ],
    ids=["CC-regression", "CC-binomial"],
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
        explainers=[MorrisSaExplainer.explainer_id()],
        results_location=tmpdir,
        log_level=loggers.DEBUG,
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
    result = interpretation.get_explainer_result(MorrisSaExplainer.explainer_id())
    print(f"Explainer result: {result}")
    assert result
    print(f"Explainer result SUMMARY: {result.summary()}")
    assert result.summary()
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
