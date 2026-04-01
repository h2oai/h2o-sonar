# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import logging
import os

import datatable
import pandas
import pytest
from sklearn import ensemble

from h2o_sonar import interpret
from h2o_sonar.explainers import pd_ice_explainer
from h2o_sonar.explainers import residual_pd_ice_explainer as explainer
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import formats
from h2o_sonar.lib.api import interpretations as i13s
from h2o_sonar.lib.api import persistences
from h2o_sonar.lib.api import results
from tests import test_utils
from tests.lib import test_containers


# constants
ResidualPdExplainer = explainer.ResidualPdIceExplainer


@pytest.mark.h2o_sonar
@pytest.mark.parametrize(
    "model_type,persistence_type,original_features,residuals",
    [
        # R/B/M @ original features RESIDUALS @ file-system
        [
            commons.ExperimentType.regression,
            persistences.PersistenceType.file_system,
            True,
            True,
        ],
        [
            commons.ExperimentType.binomial,
            persistences.PersistenceType.file_system,
            True,
            True,
        ],
        [
            commons.ExperimentType.multinomial,
            persistences.PersistenceType.file_system,
            True,
            True,
        ],
    ],
    ids=["CC-regression", "CC-binomial", "CC-multinomial"],
)
def test_mock_model(
    tmpdir,
    model_type: commons.ExperimentType,
    persistence_type: persistences.PersistenceType,
    original_features: bool,
    residuals: bool,
):
    # GIVEN
    dataset_path = test_utils.find_locally("data/predictive/creditcard.csv")
    if commons.ExperimentType.regression == model_type:
        target_col = "AGE"
    elif commons.ExperimentType.binomial == model_type:
        target_col = "SEX"
    elif commons.ExperimentType.multinomial == model_type:
        target_col = "EDUCATION"
    else:
        raise ValueError(f"Unsupported model type: '{model_type}'")
    mock_model = test_containers.SimpleMockModel(
        dataset_path=dataset_path,
        target_col=target_col,
        model_type=model_type,
        used_features=[
            "LIMIT_BAL",
            "SEX",
            "EDUCATION",
            "MARRIAGE",
            "AGE",
            "PAY_0",
            "PAY_2",
            "PAY_3",
            "PAY_4",
            "PAY_5",
            "PAY_6",
            "BILL_AMT1",
        ],
    )
    container = test_containers.ExplainerExamplesAndTemplatesTestContainer()
    container.setup(
        results_location=tmpdir,
        log_level=logging.DEBUG,
    )
    container.explainers_registry.register(
        explainer_class=ResidualPdExplainer,
    )

    # WHEN
    try:
        assert interpret.describe_explainer(ResidualPdExplainer)
        interpretation = interpret.run_interpretation(
            dataset=dataset_path,
            model=mock_model,
            target_col=target_col,
            explainers=[ResidualPdExplainer.explainer_id()],
            results_location=tmpdir,
            persistence_type=persistence_type,
            container=container,
            log_level=logging.DEBUG,
        )

        # THEN
        print(f"Interpretation:\n{interpretation}")
        if persistences.PersistenceType.in_memory == persistence_type:
            test_utils.dump_in_memory_persistence(
                interpretation.persistence.store, do_assert=True
            )

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
            ResidualPdExplainer().explainer_id()
        )
        assert_pd_explainer(
            interpretation.get_explainer_result(ResidualPdExplainer.explainer_id())
        )
        assert interpretation.get_explainer_result(
            ResidualPdExplainer.explainer_id()
        ).summary()

        # TODO assert PD specific files existence @ file-system

    finally:
        container.explainers_registry.unregister(ResidualPdExplainer.explainer_id())


@pytest.mark.h2o_sonar
@pytest.mark.parametrize(
    "model_source",
    [
        # scikit-learn model built in runtime
        None,
        # pickled scikit-learn model
        "./data/predictive/models/creditcard-binomial-sklearn-gbm.pkl",
    ],
)
def test_sklearn_model(tmpdir, model_source):
    #
    # GIVEN
    #
    dataset_path = test_utils.find_locally("data/predictive/creditcard.csv")
    target_col = "default payment next month"
    df = pandas.read_csv(dataset_path)
    (X_train, y) = df.drop(target_col, axis=1), df[target_col]
    # scikit-learn model
    gradient_booster = ensemble.GradientBoostingClassifier(learning_rate=0.1)
    gradient_booster.fit(X_train, y)
    # local container
    container = interpret.resolve_container()
    # explainable model
    explainable_model = container.model_api.create_model(
        model_src=gradient_booster,
        target_col=target_col,
        used_features=list(X_train.columns),
    )

    #
    # WHEN
    #
    interpretation = interpret.run_interpretation(
        # TODO no longer needed - to be converted by the library
        dataset=datatable.Frame(df),
        model=explainable_model,
        target_col=target_col,
        explainers=[ResidualPdExplainer.explainer_id()],
        results_location=tmpdir,
        log_level=logging.DEBUG,
    )

    #
    # THEN
    #
    print(f"\n{interpretation}")
    # find failed explainers
    assert interpretation
    assert interpretation.result.explainers
    assert len(interpretation.result.explainers)
    failed_explainers = interpretation.get_failed_explainer_ids()
    assert not failed_explainers, f"Failed explainers: {failed_explainers}"
    assert_pd_explainer(
        interpretation.get_explainer_result(ResidualPdExplainer.explainer_id())
    )


def assert_pd_explainer(
    result: results.PdResult | None,
):
    assert result, "Result cannot be None"
    assert result.summary()
    assert result.data(feature_name="PAY_0")


@pytest.mark.h2o_sonar
def test_explainer_params(tmpdir):
    # GIVEN
    dataset_path = test_utils.find_locally("data/predictive/creditcard.csv")
    target_col = "LIMIT_BAL"
    mock_model = test_containers.SimpleMockModel(
        dataset_path=dataset_path, target_col=target_col
    )
    param_grid_resolution = 3

    # WHEN
    interpretation = interpret.run_interpretation(
        dataset=dataset_path,
        model=mock_model,
        target_col=target_col,
        explainers=[
            commons.ExplainerToRun(
                explainer_id=ResidualPdExplainer.explainer_id(),
                params={
                    ResidualPdExplainer.PARAM_GRID_RESOLUTION: param_grid_resolution,
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
        ResidualPdExplainer().explainer_id()
    )
    print(f"Result metadata ({type(result_meta)}):\n{result_meta}")
    assert result_meta

    # assert explainer params
    params = result_meta[i13s.ExplainerJob.KEY_EXPLAINER_DESCRIPTOR]["parameters"]
    print(f"Parameters:\n{params}")
    assert params
    args = interpretation.explainers[0].params
    print(f"Explainer arguments: {args}")
    assert ResidualPdExplainer.PARAM_GRID_RESOLUTION in args
    assert param_grid_resolution == next(iter(args.values()))

    # assert JSon
    job_path = result_meta[i13s.ExplainerJob.KEY_JOB_LOCATION]
    print(f"Job path:\n{job_path}")
    # assert depth via JSon
    tree_json = persistences.FilesystemPersistence().load_json(
        os.path.join(
            job_path,
            "global_partial_dependence",
            "application_json",
            "pd_feature_0_class_0.json",
        )
    )
    assert tree_json
    print(f"PD data as JSon:\n{tree_json[formats.ExplanationFormat.KEY_DATA]}")
    # assert PD bin count
    assert len(tree_json[formats.ExplanationFormat.KEY_DATA]) == 3


@pytest.mark.h2o_sonar
def test_residual_not_regular_pd(tmpdir):
    # GIVEN
    dataset_path = test_utils.find_locally("data/predictive/creditcard.csv")
    target_col = "LIMIT_BAL"
    mock_model = test_containers.SimpleMockModel(
        dataset_path=dataset_path, target_col=target_col
    )

    # WHEN
    interpretation = interpret.run_interpretation(
        dataset=dataset_path,
        model=mock_model,
        target_col=target_col,
        explainers=[
            ResidualPdExplainer.explainer_id(),
            pd_ice_explainer.PdIceExplainer.explainer_id(),
        ],
        results_location=tmpdir,
        log_level=logging.DEBUG,
    )

    # THEN
    print(f"{interpretation}")
    assert interpretation
    failed_explainers = interpretation.get_failed_explainer_ids()
    assert not failed_explainers

    # residual PD
    rpd_data = _then_get_pd_data(interpretation, ResidualPdExplainer().explainer_id())
    pd_data = _then_get_pd_data(
        interpretation, pd_ice_explainer.PdIceExplainer.explainer_id()
    )
    assert rpd_data[0]["bin"] == pd_data[0]["bin"]
    assert rpd_data[0]["pd"] != pd_data[0]["pd"]
    assert rpd_data[0]["sd"] != pd_data[0]["sd"]


def _then_get_pd_data(interpretation, explainer_id: str) -> dict:
    rpd_result_meta = interpretation.get_explainer_result_metadata(explainer_id)
    assert rpd_result_meta
    job_path = rpd_result_meta[i13s.ExplainerJob.KEY_JOB_LOCATION]
    print(f"{explainer_id} job path:\n{job_path}")
    rpd_tree_json = persistences.FilesystemPersistence().load_json(
        os.path.join(
            job_path,
            "global_partial_dependence",
            "application_json",
            "pd_feature_0_class_0.json",
        )
    )
    assert rpd_tree_json
    print(
        f"{explainer_id} data as JSon:\n"
        f"{rpd_tree_json[formats.ExplanationFormat.KEY_DATA]}"
    )

    return rpd_tree_json[formats.ExplanationFormat.KEY_DATA]
