# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import logging
import os

import datatable
import pytest

from h2o_sonar import interpret
from h2o_sonar.explainers import pd_ice_explainer as explainer
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import formats
from h2o_sonar.lib.api import interpretations as i13s
from h2o_sonar.lib.api import persistences
from h2o_sonar.lib.api import results
from h2o_sonar.methods.core import method
from tests import test_utils
from tests.lib import test_containers


DATASET_CATS_MISSING = "data/predictive/creditcard_cats_missing.csv"
DATASET_MISSING = "data/predictive/creditcard_missing.csv"
CAT_MISSING_VALUE = "UNSEEN"


@pytest.mark.h2o_sonar
@pytest.mark.parametrize(
    "persistence_type, dataset, oor",
    [
        [persistences.PersistenceType.file_system, DATASET_MISSING, True],
        [persistences.PersistenceType.in_memory, DATASET_MISSING, True],
        [
            persistences.PersistenceType.file_system,
            DATASET_CATS_MISSING,
            True,
        ],
        [
            persistences.PersistenceType.in_memory,
            DATASET_CATS_MISSING,
            True,
        ],
        [
            persistences.PersistenceType.file_system,
            DATASET_MISSING,
            False,
        ],
        [persistences.PersistenceType.in_memory, DATASET_MISSING, False],
        [
            persistences.PersistenceType.file_system,
            DATASET_CATS_MISSING,
            False,
        ],
        [
            persistences.PersistenceType.in_memory,
            DATASET_CATS_MISSING,
            False,
        ],
    ],
    ids=[
        "fs_missing_true",
        "im_missing_true",
        "fs_cats_missing_true",
        "im_cats_missing_true",
        "fs_missing_false",
        "im_missing_false",
        "fs_cats_missing_false",
        "im_cats_missing_false",
    ],
)
def test_pd_mock_model_dataset_path_missing_values(
    tmpdir, persistence_type, dataset, oor
):
    # GIVEN
    dataset_path = test_utils.find_locally(dataset)
    target_col = "default.payment.next.month"
    raw_meta = {
        method.FeaturesMetadata.KEY_NUMERIC_FEATURES: ["LIMIT_BAL"],
        method.FeaturesMetadata.KEY_CATEGORICAL_FEATURES: ["PAY_0"],
    }

    # WHEN

    mock_model = test_containers.SimpleMockModel(
        dataset_path=dataset_path, target_col=target_col, feature_meta=raw_meta
    )

    # WHEN
    params = {
        explainer.PdIceExplainer.PARAM_FEATURES: ["LIMIT_BAL", "PAY_0"],
        explainer.PdIceExplainer.PARAM_NUMCAT_THRESHOLD: 20,
    }
    if oor:
        params[explainer.PdIceExplainer.PARAM_OOR_GRID_RESOLUTION] = 2
    assert interpret.describe_explainer(explainer.PdIceExplainer)
    interpretation = interpret.run_interpretation(
        dataset=dataset_path,
        model=mock_model,
        target_col=target_col,
        explainers=[
            commons.ExplainerToRun(
                explainer_id=explainer.PdIceExplainer.explainer_id(),
                params=params,
            )
        ],
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
    assert interpretation.get_scheduled_explainer_ids()
    assert interpretation.get_finished_explainer_ids()
    assert interpretation.get_successful_explainer_ids()

    failed_explainers = interpretation.get_failed_explainer_ids()
    assert not failed_explainers
    assert interpretation.get_explainer_result_metadata(
        explainer.PdIceExplainer().explainer_id()
    )
    assert interpretation.get_explainer_result(
        explainer.PdIceExplainer.explainer_id()
    ).summary()
    assert interpretation.get_explainer_result(
        explainer.PdIceExplainer.explainer_id()
    ).params()

    data_lm = interpretation.get_explainer_result(
        explainer.PdIceExplainer.explainer_id()
    ).data(feature_name="LIMIT_BAL")
    data_pay = interpretation.get_explainer_result(
        explainer.PdIceExplainer.explainer_id()
    ).data(feature_name="PAY_0")
    assert None in data_lm["bin"].to_list()[0], "None should be a bin"
    assert None in data_pay["bin"].to_list()[0], "None should be a bin"
    if oor:
        assert data_lm["bin"].to_list()[0][-3] is None, (
            "None should be last item before OOR values in bins list"
        )
        if dataset is DATASET_CATS_MISSING:
            idx = -1
        else:
            idx = -3
        assert data_pay["bin"].to_list()[0][idx] == (
            CAT_MISSING_VALUE if idx == -1 else None
        ), (
            f"{CAT_MISSING_VALUE if idx == -1 else 'None'} "
            f"should be last item before OOR values in bins list"
        )
    else:
        assert data_lm["bin"].to_list()[0][-1] is None, (
            "None should be last item in bins list"
        )
        assert data_pay["bin"].to_list()[0][-1] is None, (
            "None should be last item in bins list"
        )


@pytest.mark.h2o_sonar
@pytest.mark.parametrize(
    "persistence_type",
    [
        persistences.PersistenceType.file_system,
        persistences.PersistenceType.in_memory,
    ],
)
def test_pd_mock_model_dataset_path(tmpdir, persistence_type):
    # GIVEN
    dataset_path = test_utils.find_locally("data/predictive/creditcard.csv")
    target_col = "SEX"
    mock_model = test_containers.SimpleMockModel(
        dataset_path=dataset_path, target_col=target_col
    )

    # WHEN
    assert interpret.describe_explainer(explainer.PdIceExplainer)
    interpretation = interpret.run_interpretation(
        dataset=dataset_path,
        model=mock_model,
        target_col=target_col,
        explainers=[explainer.PdIceExplainer.explainer_id()],
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
    assert interpretation.get_scheduled_explainer_ids()
    assert interpretation.get_finished_explainer_ids()
    assert interpretation.get_successful_explainer_ids()

    failed_explainers = interpretation.get_failed_explainer_ids()
    assert not failed_explainers
    assert interpretation.get_explainer_result_metadata(
        explainer.PdIceExplainer().explainer_id()
    )
    assert_pd_ice_explainer_result(
        interpretation.get_explainer_result(explainer.PdIceExplainer.explainer_id())
    )
    assert interpretation.get_explainer_result(
        explainer.PdIceExplainer.explainer_id()
    ).summary()
    assert interpretation.get_explainer_result(
        explainer.PdIceExplainer.explainer_id()
    ).params()


@pytest.mark.h2o_sonar
def test_pd_feat_date_type(tmpdir):
    # GIVEN
    date_feature = "contact_date"
    dataset_path = test_utils.find_locally(
        "data/predictive/bank_marketing_with_dates.csv"
    )
    target_col = "subscribed"
    raw_meta = {
        method.FeaturesMetadata.KEY_DATE_FEATURES: [date_feature],
    }
    mock_model = test_containers.SimpleMockModel(
        dataset_path=dataset_path, target_col=target_col, feature_meta=raw_meta
    )

    datatable.options.fread.parse_dates = False
    datatable.options.fread.parse_times = False
    df = datatable.fread(dataset_path)

    columns = list(df.names)
    drop_cols = [x for x in columns if x != date_feature]

    # WHEN
    assert interpret.describe_explainer(explainer.PdIceExplainer)
    interpretation = interpret.run_interpretation(
        dataset=df,
        model=mock_model,
        target_col=target_col,
        explainers=[explainer.PdIceExplainer.explainer_id()],
        results_location=tmpdir,
        log_level=logging.DEBUG,
        drop_cols=drop_cols,
    )

    # THEN
    print(f"Interpretation:\n{interpretation}")

    pd_result_meta = interpretation.get_explainer_result_metadata(
        explainer.PdIceExplainer().explainer_id()
    )
    # assert JSon
    job_path = pd_result_meta[i13s.ExplainerJob.KEY_JOB_LOCATION]
    print(f"Job path:\n{job_path}")
    # assert max features via JSon
    pd_json_path = os.path.join(
        job_path, "global_partial_dependence", "application_json"
    )
    idx_dict = persistences.FilesystemPersistence().load_json(
        os.path.join(pd_json_path, "explanation.json")
    )
    feat_type = idx_dict["features"][date_feature]["feature_type"][0]
    assert feat_type == "categorical", (
        f"Feature type should be `categorical`, not {feat_type}"
    )


@pytest.mark.h2o_sonar
def test_explainer_params(tmpdir):
    # GIVEN
    dataset_path = test_utils.find_locally("data/predictive/creditcard.csv")
    target_col = "SEX"
    mock_model = test_containers.SimpleMockModel(
        dataset_path=dataset_path, target_col=target_col
    )
    param_max_features = 3
    param_bins = 3

    # WHEN
    interpretation = interpret.run_interpretation(
        dataset=dataset_path,
        model=mock_model,
        target_col=target_col,
        explainers=[
            commons.ExplainerToRun(
                explainer_id=explainer.PdIceExplainer.explainer_id(),
                params={
                    explainer.PdIceExplainer.PARAM_MAX_FEATURES: param_max_features,
                    explainer.PdIceExplainer.PARAM_GRID_RESOLUTION: param_bins,
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
    pd_result_meta = interpretation.get_explainer_result_metadata(
        explainer.PdIceExplainer().explainer_id()
    )
    print(f"Result metadata ({type(pd_result_meta)}):\n{pd_result_meta}")
    assert pd_result_meta
    pd_result = interpretation.get_explainer_result(
        explainer.PdIceExplainer().explainer_id()
    )
    print(f"Result ({type(pd_result)}):\n{pd_result}")
    assert pd_result

    # assert bins
    pd_feature_data = pd_result.data(feature_name="LIMIT_BAL")
    print(f"Data ({pd_feature_data.shape}):\n{pd_feature_data}")
    assert param_bins == pd_feature_data.shape[0]

    # assert explainer params
    pd_params = pd_result_meta[i13s.ExplainerJob.KEY_EXPLAINER_DESCRIPTOR]["parameters"]
    print(f"Parameters:\n{pd_params}")
    assert pd_params
    pd_args = interpretation.explainers[0].params
    print(f"Explainer arguments: {pd_args}")
    assert explainer.PdIceExplainer.PARAM_MAX_FEATURES in pd_args
    assert explainer.PdIceExplainer.PARAM_GRID_RESOLUTION in pd_args
    assert param_max_features == next(iter(pd_args.values()))

    # assert JSon
    job_path = pd_result_meta[i13s.ExplainerJob.KEY_JOB_LOCATION]
    print(f"Job path:\n{job_path}")
    # assert max features via JSon
    pd_json_path = os.path.join(
        job_path, "global_partial_dependence", "application_json"
    )
    idx_dict = persistences.FilesystemPersistence().load_json(
        os.path.join(pd_json_path, "explanation.json")
    )
    assert idx_dict
    print(f"Index:\n{idx_dict}")
    assert param_max_features == len(idx_dict[formats.ExplanationFormat.KEY_FEATURES])

    # assert bins via JSon
    data_dict = persistences.FilesystemPersistence().load_json(
        os.path.join(pd_json_path, "pd_feature_0_class_0.json")
    )
    assert param_bins == len(data_dict["data"])


def assert_pd_ice_explainer_result(result: results.PdResult | None) -> None:
    assert result, "Result cannot be None"
    data = result.data(feature_name="EDUCATION")
    assert data.names == ("bin", "frequency", "pd", "sd", "oor")
    # TODO: LABEL_REGRESSION is used as the class name for both binomial and regression
    data = result.data(feature_name="PAY_0", clazz=result.format.LABEL_REGRESSION)
    assert data.names == ("bin", "frequency", "pd", "sd", "oor")


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
