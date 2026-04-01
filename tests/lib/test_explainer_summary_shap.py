# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import logging
import os

import datatable
import pandas
import pytest
from sklearn import ensemble

from h2o_sonar import interpret
from h2o_sonar.explainers import summary_shap_explainer
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import formats
from h2o_sonar.lib.api import interpretations as i13s
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import persistences
from h2o_sonar.lib.api import results
from h2o_sonar.utils import preprocessing
from tests import test_utils
from tests.lib import test_containers


# constants
ShapleyExplainer = summary_shap_explainer.SummaryShapleyExplainer


@pytest.mark.h2o_sonar
@pytest.mark.parametrize(
    "model_type",
    [
        commons.ExperimentType.regression,
        commons.ExperimentType.binomial,
        commons.ExperimentType.multinomial,
    ],
)
def test_mock_model(tmpdir, model_type):
    # GIVEN
    dataset_path = test_utils.find_locally("data/predictive/creditcard.csv")
    if commons.ExperimentType.regression == model_type:
        target_col = "AGE"
        classes = [formats.GlobalSummaryFeatImpJsonFormat.LABEL_REGRESSION]
    elif commons.ExperimentType.binomial == model_type:
        target_col = "SEX"
        classes = [1, 2]
    elif commons.ExperimentType.multinomial == model_type:
        target_col = "EDUCATION"
        classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    else:
        raise ValueError(f"Unsupported model type: '{model_type}'")

    mock_model = test_containers.SimpleMockModel(
        dataset_path=dataset_path,
        target_col=target_col,
        labels=classes,
        model_type=model_type,
    )
    container = test_containers.ExplainerExamplesAndTemplatesTestContainer()
    container.setup(
        results_location=tmpdir,
        log_level=logging.DEBUG,
    )
    container.explainers_registry.register(
        explainer_class=summary_shap_explainer.SummaryShapleyExplainer,
    )

    # WHEN
    try:
        assert interpret.describe_explainer(
            summary_shap_explainer.SummaryShapleyExplainer
        )
        interpretation = interpret.run_interpretation(
            dataset=dataset_path,
            model=mock_model,
            target_col=target_col,
            explainers=[summary_shap_explainer.SummaryShapleyExplainer.explainer_id()],
            results_location=tmpdir,
            persistence_type=persistences.PersistenceType.file_system,
            container=container,
            log_level=logging.DEBUG,
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
            summary_shap_explainer.SummaryShapleyExplainer.explainer_id()
        )
        _then_summary_shap_explainer(
            interpretation.get_explainer_result(
                summary_shap_explainer.SummaryShapleyExplainer.explainer_id()
            ),
            clazz=str(classes[-1]),
        )
        assert interpretation.get_explainer_result(
            summary_shap_explainer.SummaryShapleyExplainer.explainer_id()
        ).summary()
        assert interpretation.get_explainer_result(
            summary_shap_explainer.SummaryShapleyExplainer.explainer_id()
        ).params()

    finally:
        container.explainers_registry.unregister(
            summary_shap_explainer.SummaryShapleyExplainer.explainer_id()
        )


@pytest.mark.h2o_sonar
@pytest.mark.parametrize(
    "persistence_type, dataset, target_col, labels",
    [
        (
            persistences.PersistenceType.file_system,
            "creditcard.csv",
            "default payment next month",
            [0, 1],
        ),
        (
            persistences.PersistenceType.in_memory,
            "creditcard.csv",
            "default payment next month",
            [0, 1],
        ),
        (
            persistences.PersistenceType.file_system,
            "iris.csv",
            "class",
            ["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
        ),
        (
            persistences.PersistenceType.in_memory,
            "iris.csv",
            "class",
            ["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
        ),
    ],
    ids=[
        "CC-binomial-filesystem",
        "CC-binomial-in-memory",
        "iris-multinomial-filesystem",
        "iris-multinomial-in-memory",
    ],
)
def test_sklearn(tmpdir, persistence_type, dataset, target_col, labels):
    # GIVEN
    dataset_path, explainable_model, target_col = _given_sklearn(
        dataset=dataset, target_col=target_col
    )
    explainable_model.meta.labels = labels
    # WHEN
    interpretation = interpret.run_interpretation(
        dataset=dataset_path,
        model=explainable_model,
        target_col=target_col,
        explainers=[summary_shap_explainer.SummaryShapleyExplainer.explainer_id()],
        results_location=tmpdir,
        persistence_type=persistence_type,
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
    assert interpretation.get_explainer_result(
        summary_shap_explainer.SummaryShapleyExplainer.explainer_id()
    )


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
        dataset=dataset_path,
    )
    print(f"Explainable model: {explainable_model}")

    # WHEN
    interpretation = interpret.run_interpretation(
        dataset=dataset_path,
        model=explainable_model,
        target_col=target_col,
        explainers=[summary_shap_explainer.SummaryShapleyExplainer.explainer_id()],
        results_location=tmpdir,
        log_level=logging.DEBUG,
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
    assert interpretation.get_explainer_result(
        summary_shap_explainer.SummaryShapleyExplainer.explainer_id()
    )


def _given_sklearn(dataset, target_col):
    dataset_path = test_utils.find_locally(f"data/{dataset}")
    model = ensemble.GradientBoostingClassifier()
    df = pandas.read_csv(dataset_path)
    (X, y) = df.drop(target_col, axis=1), df[target_col]
    used_features = X.columns.to_list()
    (X, _, _) = preprocessing.categorical_encoder(X)
    # scikit-learn model
    model.fit(X, y)
    explainable_model = models.ModelApi().create_model(
        model_src=model,
        target_col=target_col,
        used_features=used_features,
    )
    return dataset_path, explainable_model, target_col


def _then_summary_shap_explainer(
    result: results.SummaryShapResult | None,
    clazz: str,
):
    assert result, "Result cannot be None"
    data = result.data(feature_names="PAY_0", clazz=clazz)
    assert data.names == ("PAY_0",) or data.names == ("PAY_0", "bias")
    data = result.data(feature_names=["PAY_0", "PAY_4"], clazz=clazz)
    assert data.names == ("PAY_0", "PAY_4") or data.names == ("PAY_0", "PAY_4", "bias")


@pytest.mark.parametrize(
    "max_features,create_drilldown_charts",
    [
        (3, True),
        (2, False),
    ],
)
@pytest.mark.h2o_sonar
def test_explainer_params(tmpdir, max_features: int, create_drilldown_charts: bool):
    # GIVEN
    dataset_path = test_utils.find_locally("data/predictive/creditcard.csv")
    target_col = "SEX"
    mock_model = test_containers.SimpleMockModel(
        dataset_path=dataset_path, target_col=target_col
    )
    explainer = summary_shap_explainer
    param_max_features: int = max_features
    param_drilldown: bool = create_drilldown_charts

    # WHEN
    interpretation = interpret.run_interpretation(
        dataset=dataset_path,
        model=mock_model,
        target_col=target_col,
        explainers=[
            commons.ExplainerToRun(
                explainer_id=explainer.SummaryShapleyExplainer.explainer_id(),
                params={
                    ShapleyExplainer.PARAM_MAX_FEATURES: param_max_features,
                    ShapleyExplainer.PARAM_DRILLDOWN_CHARTS: param_drilldown,
                },
            )
        ],
        results_location=tmpdir,
        log_level=logging.DEBUG,
    )

    # THEN
    print(f"{interpretation}")
    assert interpretation

    interpretation_dict = interpretation.to_dict()
    assert interpretation_dict
    # assert h2o-sonar.html
    assert os.path.isfile(
        os.path.join(
            interpretation_dict[i13s.Interpretation.KEY_RESULT][
                i13s.Interpretation.KEY_RESULTS_LOCATION
            ],
            persistences.InterpretationPersistence.FILE_H2O_SONAR_HTML,
        )
    )
    assert os.path.isfile(
        os.path.join(
            interpretation_dict[i13s.Interpretation.KEY_RESULT][
                i13s.Interpretation.KEY_INTERPRETATION_LOCATION
            ],
            persistences.InterpretationPersistence.FILE_INTERPRETATION_HTML,
        )
    )
    assert os.path.isfile(
        os.path.join(
            interpretation_dict[i13s.Interpretation.KEY_RESULT][
                i13s.Interpretation.KEY_INTERPRETATION_LOCATION
            ],
            persistences.InterpretationPersistence.FILE_INTERPRETATION_JSON,
        )
    )

    failed_explainers = interpretation.get_failed_explainer_ids()
    assert not failed_explainers
    result_meta = interpretation.get_explainer_result_metadata(
        explainer.SummaryShapleyExplainer().explainer_id()
    )
    print(f"Result metadata ({type(result_meta)}):\n{result_meta}")
    assert result_meta
    explainer_result = interpretation.get_explainer_result(
        explainer.SummaryShapleyExplainer().explainer_id()
    )
    print(f"Explainer result ({type(explainer_result)}):\n{explainer_result}")
    assert explainer_result

    # assert data
    feature_data = explainer_result.data()
    print(f"Data ({feature_data.shape}):\n{feature_data}")
    # IMPROVE: data provided by the explainer are not trimmed

    # assert explainer params
    params = result_meta[i13s.ExplainerJob.KEY_EXPLAINER_DESCRIPTOR]["parameters"]
    print(f"Parameters:\n{params}")
    assert params
    args = interpretation.explainers[0].params
    print(f"Explainer arguments: {args}")
    assert explainer.SummaryShapleyExplainer.PARAM_MAX_FEATURES in args
    assert explainer.SummaryShapleyExplainer.PARAM_DRILLDOWN_CHARTS in args

    # assert datatable explanation
    job_path = result_meta[i13s.ExplainerJob.KEY_JOB_LOCATION]
    print(f"Job path:\n{job_path}")
    # assert max features via JSon
    dt_path = os.path.join(
        job_path,
        "global_summary_feature_importance",
        "application_vnd_h2oai_json_datatable_jay",
        "summary_feature_importance_class_0.jay",
    )
    dt_data = datatable.fread(dt_path)
    assert param_max_features >= dt_data[:, "feature"].nunique1(), (
        f"Features: {dt_data[:, 'feature'].to_list()}"
    )

    # assert main chart existence
    file_path = os.path.join(
        job_path,
        "work",
        "shapley-class-0.png",
    )
    assert os.path.isfile(file_path), f"Main chart file {file_path} is missing"

    # assert scatter plots existence (only num features have scatter plot, cat do not)
    json_format_dir = os.path.join(
        job_path,
        "global_summary_feature_importance",
        "application_json",
    )
    json_format_idx = persistences.FilesystemPersistence().load_json(
        os.path.join(json_format_dir, "explanation.json")
    )
    assert json_format_idx
    features_with_scatter = json_format_idx.get("files_details", {}).get(
        formats.ExplanationFormat.LABEL_REGRESSION, {}
    )
    for f in features_with_scatter:
        file_path = os.path.join(
            json_format_dir,
            features_with_scatter[f],
        )
        if create_drilldown_charts:
            assert os.path.isfile(file_path), f"Chart file {file_path} missing"
        else:
            assert not os.path.isfile(file_path), f"Unexpected chart file {file_path}"


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
