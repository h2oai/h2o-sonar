# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import logging
import os

import pandas
import pytest
from sklearn import ensemble

from h2o_sonar import interpret
from h2o_sonar.explainers import dia_explainer as explainer
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import interpretations as i13s
from h2o_sonar.lib.api import persistences
from h2o_sonar.lib.api import results
from h2o_sonar.lib.api.commons import ExplainerToRun
from h2o_sonar.lib.api.models import ModelApi
from h2o_sonar.methods.core import method
from h2o_sonar.utils import preprocessing
from tests import test_utils
from tests.lib import test_containers


try:
    import h2o

    HAS_H2O = True
except ImportError:
    HAS_H2O = False


@pytest.mark.skipif(
    not HAS_H2O,
    reason="H2O-3 Python package is not installed",
)
@pytest.mark.parametrize(
    "persistence_type, dataset, target_col, model_type, classification, params",
    [
        (
            persistences.PersistenceType.file_system,
            "creditcard.csv",
            "default payment next month",
            "sklearn",
            True,
            dict(cut_off=0.2),
        ),
        (
            persistences.PersistenceType.file_system,
            "creditcard.csv",
            "LIMIT_BAL",
            "sklearn",
            False,
            "",
        ),
        (
            persistences.PersistenceType.in_memory,
            "creditcard.csv",
            "LIMIT_BAL",
            "sklearn",
            False,
            "",
        ),
        pytest.param(
            persistences.PersistenceType.file_system,
            "creditcard.csv",
            "LIMIT_BAL",
            "h2o3",
            False,
            "",
            marks=pytest.mark.skipif(
                not HAS_H2O, reason="H2O-3 Python package is not installed"
            ),
        ),
        pytest.param(
            persistences.PersistenceType.in_memory,
            "creditcard.csv",
            "LIMIT_BAL",
            "h2o3",
            False,
            "",
            marks=pytest.mark.skipif(
                not HAS_H2O, reason="H2O-3 Python package is not installed"
            ),
        ),
        (
            persistences.PersistenceType.file_system,
            "creditcard.csv",
            "default payment next month",
            "sklearn",
            True,
            "",
        ),
        (
            persistences.PersistenceType.in_memory,
            "creditcard.csv",
            "default payment next month",
            "sklearn",
            True,
            "",
        ),
        pytest.param(
            persistences.PersistenceType.file_system,
            "creditcard.csv",
            "default payment next month",
            "h2o3",
            True,
            "",
            marks=pytest.mark.skipif(
                not HAS_H2O, reason="H2O-3 Python package is not installed"
            ),
        ),
        pytest.param(
            persistences.PersistenceType.in_memory,
            "creditcard.csv",
            "default payment next month",
            "h2o3",
            True,
            "",
            marks=pytest.mark.skipif(
                not HAS_H2O, reason="H2O-3 Python package is not installed"
            ),
        ),
        pytest.param(
            persistences.PersistenceType.file_system,
            "creditcard.csv",
            "default payment next month",
            "dai",
            True,
            "",
            marks=pytest.mark.skipif(
                not test_utils.is_mojo_supported(), reason="MOJO is not supported"
            ),
        ),
        pytest.param(
            persistences.PersistenceType.in_memory,
            "pd_ice_creditcard_train.csv",
            "LIMIT_BAL",
            "dai",
            False,
            "",
            marks=pytest.mark.skipif(
                not test_utils.is_mojo_supported(), reason="MOJO is not supported"
            ),
        ),
        pytest.param(
            persistences.PersistenceType.in_memory,
            "creditcard.csv",
            "default payment next month",
            "dai",
            True,
            "",
            marks=pytest.mark.skipif(
                not test_utils.is_mojo_supported(), reason="MOJO is not supported"
            ),
        ),
    ],
    ids=[
        "filesystem_binomial_sklearn",
        "filesystem_regression_sklearn",
        "inmemory_regression_sklearn",
        "filesystem_regression_h2o3",
        "inmemory_regression_h2o3",
        "filesystem_binomial_sklearn_no_params",
        "inmemory_binomial_sklearn_no_params",
        "filesystem_binomial_h2o3_no_params",
        "inmemory_binomial_h2o3_no_params",
        "filesystem_binomial_dai",
        "inmemory_regression_dai",
        "inmemory_binomial_dai",
    ],
)
@pytest.mark.h2o_sonar
def test_dia_dataset_path(
    tmpdir,
    h2o3_cleanup_fixture,
    persistence_type,
    dataset,
    target_col,
    model_type,
    classification,
    params,
):
    # GIVEN

    # connect to H2O-3 cluster
    test_utils.h2o3_init_for_tests()

    if model_type == "sklearn":
        dataset_path, explainable_model, target_col = setup_sklearn(
            dataset=dataset, target_col=target_col, classification=classification
        )
    elif model_type == "h2o3":
        dataset_path, explainable_model, target_col = setup_h2o(
            dataset=dataset, target_col=target_col, classification=classification
        )
    elif model_type == "dai":
        dataset_path, explainable_model, target_col = setup_dai(
            dataset=dataset, classification=classification
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    # WHEN
    assert interpret.describe_explainer(explainer.DiaExplainer)
    interpretation = interpret.run_interpretation(
        dataset=dataset_path,
        model=explainable_model,
        target_col=target_col,
        results_location=tmpdir,
        persistence_type=persistence_type,
        log_level=logging.DEBUG,
        explainers=(
            [
                ExplainerToRun(
                    explainer_id=explainer.DiaExplainer.explainer_id(),
                    params=params,
                )
            ]
            if params
            else [explainer.DiaExplainer.explainer_id()]
        ),
    )

    # THEN
    print(f"Interpretation:\n{interpretation}")
    if params:
        assert interpretation.explainers[0].params == params
        assert (
            params.items()
            <= interpretation.get_explainer_result(
                explainer.DiaExplainer.explainer_id()
            )
            .params()
            .items()
        )
    if persistences.PersistenceType.in_memory == persistence_type:
        test_utils.dump_in_memory_persistence(
            interpretation.persistence.store, do_assert=True
        )

    assert interpretation
    assert interpretation.get_scheduled_explainer_ids()
    assert interpretation.get_finished_explainer_ids()
    assert interpretation.get_successful_explainer_ids()
    assert not interpretation.get_failed_explainer_ids()
    assert interpretation.get_explainer_result_metadata(
        explainer.DiaExplainer.explainer_id()
    )
    assert_dia_explainer_result(
        result=interpretation.get_explainer_result(
            explainer.DiaExplainer.explainer_id()
        ),
        classification=classification,
        persistence_type=persistence_type,
    )
    assert interpretation.get_explainer_result(
        explainer.DiaExplainer.explainer_id()
    ).summary()


@pytest.mark.h2o_sonar
def test_metadata_builder():
    # GIVEN
    raw_meta = {
        method.FeaturesMetadata.KEY_NUMERIC_FEATURES: ["AGE"],
        method.FeaturesMetadata.KEY_CATEGORICAL_FEATURES: ["PAY_1"],
    }

    # WHEN
    feature_metadata = method.FeaturesMetadata(raw_meta)

    # THEN
    assert feature_metadata
    print(f"Features metadata: {feature_metadata.to_dict()}")
    assert feature_metadata.to_dict()


def setup_sklearn(dataset, target_col, classification=True):
    dataset_path = test_utils.find_locally(f"data/predictive/{dataset}")
    if classification:
        model = ensemble.GradientBoostingClassifier()
    else:
        model = ensemble.GradientBoostingRegressor()
    df = pandas.read_csv(dataset_path)
    (x, y) = df.drop(target_col, axis=1), df[target_col]
    (x, _, _) = preprocessing.categorical_encoder(x)
    # scikit-learn model
    model.fit(x, y)
    explainable_model = ModelApi().create_model(
        model_src=model,
        target_col=target_col,
        used_features=list(x.columns),
    )
    return dataset_path, explainable_model, target_col


def setup_h2o(dataset, target_col, classification=True):
    from h2o.estimators.gbm import H2OGradientBoostingEstimator

    dataset_path = test_utils.find_locally(f"data/predictive/{dataset}")
    df = h2o.import_file(dataset_path)
    if classification:
        df[target_col] = df[target_col].asfactor()
    else:
        df["default payment next month"] = df["default payment next month"].asfactor()
    x = list(df.names)
    x.remove(target_col)
    # h2o model
    gradient_booster = H2OGradientBoostingEstimator(ntrees=1, seed=1234)
    gradient_booster.train(x=x, y=target_col, training_frame=df)
    # explainable model
    explainable_model = ModelApi().create_model(
        model_src=gradient_booster,
        target_col=target_col,
        used_features=x,
    )
    return dataset_path, explainable_model, target_col


def setup_dai(dataset, classification):
    import daimojo

    if classification:
        dataset_path = test_utils.find_locally(f"data/predictive/{dataset}")
        mojo_path = test_utils.find_locally(
            "data/predictive/models/creditcard-binomial.mojo"
        )
        target_col = "default payment next month"
    else:
        dataset_path = test_utils.find_locally(f"data/predictive/{dataset}")
        mojo_path = test_utils.find_locally(
            "data/predictive/models/creditcard-regression.mojo"
        )
        target_col = "LIMIT_BAL"

    # DAI model
    dai_model = daimojo.model(mojo_path)
    # explainable model
    explainable_model = ModelApi().create_model(
        model_src=dai_model,
        target_col=target_col,
        used_features=list(dai_model.feature_names),
    )
    return dataset_path, explainable_model, target_col


def assert_dia_explainer_result(
    result: results.DiaResult | None,
    classification: bool,
    persistence_type: persistences.PersistenceType,
):
    assert result, "Result cannot be None"
    if persistence_type != persistences.PersistenceType.file_system:
        return
    data = result.data(feature_name="MARRIAGE", category=result.DiaCategory.DIA_METRICS)
    assert (
        data.names
        == (
            "Groups",
            "N",
            "Adverse Impact",
            "Accuracy",
            "True Positive Rate",
            "Precision",
            "Specificity",
            "Negative Predicted Value",
            "False Positive Rate",
            "False Discovery Rate",
            "False Negative Rate",
            "False Omissions Rate",
        )
        if classification
        else (
            "Groups",
            "N",
            "Mean Prediction",
            "Std.Dev Prediction",
            "Maximum Prediction",
            "Minimum Prediction",
            "R2",
            "RMSE",
        )
    )
    if classification:
        data = result.data(
            feature_name="PAY_0", category=result.DiaCategory.DIA_CATEGORY_CM
        )
        assert data.names == ("actual1", "actual0")

    data = result.data(
        feature_name="PAY_0",
        category=result.DiaCategory.DIA_CATEGORY_ME_SMD,
        ref_level=2,
    )
    assert (
        data.names
        == (
            "Groups",
            "N",
            "Marginal Error",
            "Standardized Mean Difference",
        )
        if classification
        else ("Groups", "N", "Standardized Mean Difference")
    )
    data = result.data(
        feature_name="PAY_3",
        category=result.DiaCategory.DIA_CATEGORY_PARITY,
        ref_level="5",
    )
    assert (
        data.names
        == (
            "Groups",
            "N",
            "Adverse Impact Parity",
            "Accuracy Parity",
            "True Positive Rate Parity",
            "Precision Parity",
            "Specificity Parity",
            "Negative Predicted Value Parity",
            "False Positive Rate Parity",
            "False Discovery Rate Parity",
            "False Negative Rate Parity",
            "False Omissions Rate Parity",
            "Type I Parity",
            "Type II Parity",
            "Equalized Odds",
            "Supervised Fairness",
            "Overall Fairness",
        )
        if classification
        else (
            "Groups",
            "N",
            "Mean Prediction Parity",
            "Std.Dev Prediction Parity",
            "Maximum Prediction Parity",
            "Minimum Prediction Parity",
            "R2 Parity",
            "RMSE Parity",
            "Overall Fairness",
        )
    )

    data = result.data(
        feature_name="PAY_3",
        category=result.DiaCategory.DIA_CATEGORY_DISPARITY,
        ref_level=1,
    )
    assert (
        data.names
        == (
            "Groups",
            "N",
            "Adverse Impact Disparity",
            "Marginal Error",
            "Standardized Mean Difference",
            "Accuracy Disparity",
            "True Positive Rate Disparity",
            "Precision Disparity",
            "Specificity Disparity",
            "Negative Predicted Value Disparity",
            "False Positive Rate Disparity",
            "False Discovery Rate Disparity",
            "False Negative Rate Disparity",
            "False Omissions Rate Disparity",
        )
        if classification
        else (
            "Groups",
            "N",
            "Standardized Mean Difference",
            "Mean Prediction Disparity",
            "Std.Dev Prediction Disparity",
            "Maximum Prediction Disparity",
            "Minimum Prediction Disparity",
            "R2 Disparity",
            "RMSE Disparity",
        )
    )


@pytest.mark.h2o_sonar
def test_explainer_params(tmpdir):
    # GIVEN
    dataset_path = test_utils.find_locally("data/predictive/creditcard.csv")
    target_col = "SEX"
    mock_model = test_containers.SimpleMockModel(
        dataset_path=dataset_path, target_col=target_col
    )
    param_max_metric = "MCC"
    param_cut_off = 0.54321
    interpretation = None
    iterations = 1  # ad hoc HTML index sorting test

    # WHEN
    for _ in range(iterations):
        interpretation = interpret.run_interpretation(
            dataset=dataset_path,
            model=mock_model,
            target_col=target_col,
            explainers=[
                commons.ExplainerToRun(
                    explainer_id=explainer.DiaExplainer.explainer_id(),
                    params={
                        explainer.DiaExplainer.PARAM_MAXIMIZE_METRIC: param_max_metric,
                        explainer.DiaExplainer.PARAM_CUT_OFF: param_cut_off,
                    },
                )
            ],
            results_location=tmpdir,
            log_level=logging.DEBUG,
        )

    # THEN
    print(f"{interpretation}")
    assert interpretation
    failed_explainers = interpretation.get_failed_explainer_ids()
    assert not failed_explainers
    result_meta = interpretation.get_explainer_result_metadata(
        explainer.DiaExplainer().explainer_id()
    )
    print(f"Result metadata ({type(result_meta)}):\n{result_meta}")
    assert result_meta
    result = interpretation.get_explainer_result(
        explainer.DiaExplainer().explainer_id()
    )
    print(f"Result ({type(result)}):\n{result}")
    assert isinstance(result, results.DiaResult)

    # assert CM
    feature_data = result.data(
        feature_name="EDUCATION", category=result.DiaCategory.DIA_CATEGORY_CM
    )
    print(f"Data ({feature_data.shape}):\n{feature_data}")
    assert 2 == feature_data.shape[0]
    assert 2 == feature_data.shape[1]

    # assert explainer params
    params = result_meta[i13s.ExplainerJob.KEY_EXPLAINER_DESCRIPTOR]["parameters"]
    print(f"Parameters:\n{params}")
    assert params
    args = interpretation.explainers[0].params
    print(f"Explainer arguments: {args}")
    assert explainer.DiaExplainer.PARAM_CUT_OFF in args
    assert explainer.DiaExplainer.PARAM_MAXIMIZE_METRIC in args

    # assert JSon
    job_path = result_meta[i13s.ExplainerJob.KEY_JOB_LOCATION]
    print(f"Job path:\n{job_path}")
    # assert cut-off via JSon
    json_path = os.path.join(job_path, "work")
    idx_dict = persistences.FilesystemPersistence().load_json(
        os.path.join(json_path, "dia_entity.json")
    )
    assert idx_dict
    print(f"DIA entity:\n{idx_dict}")
    assert param_cut_off == idx_dict["summary"]["cut_off"]
    assert param_max_metric == idx_dict["summary"]["max_metric"]
