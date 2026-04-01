# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
import logging
import os.path

import pytest

from h2o_sonar import interpret
from h2o_sonar.explainers import friedman_h_statistic_explainer as explainer
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import models
from tests import test_utils
from tests.lib import test_containers


@pytest.mark.parametrize(
    "experiment_type,model_path,dataset_path,target_col",
    [
        # MOJO models
        pytest.param(
            commons.ExperimentType.regression,
            "data/predictive/models/creditcard-regression.mojo",
            "data/predictive/pd_ice_creditcard_10_rows.csv",
            "LIMIT_BAL",
            marks=pytest.mark.skipif(
                not test_utils.is_mojo_supported(), reason="MOJO is not supported"
            ),
            id="CC-regression",
        ),
        pytest.param(
            commons.ExperimentType.binomial,
            "data/predictive/models/creditcard-binomial.mojo",
            "data/predictive/creditcard.csv",
            "default payment next month",
            marks=pytest.mark.skipif(
                not test_utils.is_mojo_supported(), reason="MOJO is not supported"
            ),
            id="CC-binomial",
        ),
        # sklearn model(s)
        pytest.param(
            commons.ExperimentType.binomial,
            "./data/predictive/models/creditcard-binomial-sklearn-gbm.pkl",
            "data/predictive/creditcard.csv",
            "SEX",
            marks=pytest.mark.skipif(
                not test_utils.is_sklearn_1_1_2(),
                reason=(
                    "scikit-learn version cannot be enforced in case of Driverless AI "
                    "tests and this test is run only with particular scikit-learn "
                    "version"
                ),
            ),
            id="CC-binomial-sklearn",
        ),
        # mock model(s)
        (
            commons.ExperimentType.binomial,
            "",
            "data/predictive/creditcard.csv",
            "default payment next month",
        ),
    ],
)
@pytest.mark.h2o_sonar
def test_models(tmpdir, experiment_type, model_path, dataset_path, target_col):
    if test_utils.is_mojo_supported():
        import daimojo
    else:
        daimojo = None

    #
    # GIVEN
    #
    # container
    container = interpret.resolve_container()
    # dataset
    dataset_path = test_utils.find_locally(dataset_path)
    if model_path:
        model_path = test_utils.find_locally(model_path)
        # DAI model
        if daimojo and model_path.endswith(models.DriverlessAiModel.EXT_MOJO):
            model_src = daimojo.model(model_path)
            used_features = list(model_src.feature_names)
            # explainable model
            explainable_model = container.model_api.create_model(
                model_src=model_src,
                target_col=target_col,
                used_features=used_features,
            )
        elif model_path.endswith(models.PickleFileModel.EXT_PICKLE):
            explainable_model = model_path
        else:
            raise ValueError(
                f"Unsupported model type: {model_path} (must be MOJO or pickle)"
            )
    else:
        # mock model
        explainable_model = test_containers.SimpleMockModel(
            dataset_path=dataset_path, target_col=target_col
        )
    print(f"Explainable model: {explainable_model}")

    #
    # WHEN
    #
    interpretation = interpret.run_interpretation(
        dataset=dataset_path,
        model=explainable_model,
        target_col=target_col,
        explainers=[explainer.FriedmanHStatisticExplainer.explainer_id()],
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
        explainer.FriedmanHStatisticExplainer.explainer_id()
    )
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


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
