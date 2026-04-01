# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import logging
import os

import pytest

from h2o_sonar import interpret
from h2o_sonar.explainers import pd_2_features_explainer as explainer
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import interpretations as i13s
from h2o_sonar.lib.api import persistences
from h2o_sonar.lib.api import plots
from tests import test_utils
from tests.lib import test_containers


@pytest.mark.h2o_sonar
@pytest.mark.parametrize(
    "persistence_type",
    [
        persistences.PersistenceType.file_system,
        persistences.PersistenceType.in_memory,
    ],
)
def test_mock_model_persistences(tmpdir, persistence_type):
    #
    # GIVEN
    #
    dataset_path = test_utils.find_locally("data/predictive/creditcard.csv")
    target_col = "SEX"
    mock_model = test_containers.SimpleMockModel(
        dataset_path=dataset_path, target_col=target_col
    )

    #
    # WHEN
    #
    assert interpret.describe_explainer(explainer.PdFor2FeaturesExplainer)
    interpretation = interpret.run_interpretation(
        dataset=dataset_path,
        model=mock_model,
        target_col=target_col,
        explainers=[explainer.PdFor2FeaturesExplainer.explainer_id()],
        results_location=tmpdir,
        persistence_type=persistence_type,
        log_level=logging.DEBUG,
    )

    #
    # THEN
    #
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
        explainer.PdFor2FeaturesExplainer().explainer_id()
    )

    result = interpretation.get_explainer_result(
        explainer.PdFor2FeaturesExplainer.explainer_id()
    )
    assert result, "Result cannot be None"
    data = result.data(feature_names="'LIMIT_BAL' and 'EDUCATION'")
    print(data)
    assert data

    assert interpretation.get_explainer_result(
        explainer.PdFor2FeaturesExplainer.explainer_id()
    ).summary()
    assert interpretation.get_explainer_result(
        explainer.PdFor2FeaturesExplainer.explainer_id()
    ).params()


@pytest.mark.skipif(not test_utils.is_mojo_supported(), reason="MOJO is not supported")
@pytest.mark.h2o_sonar
@pytest.mark.parametrize(
    "dataset_path,mojo_path,target_col",
    [
        (
            "data/predictive/creditcard_str_10k.csv",
            "data/predictive/models/creditcard-regression-str.mojo",
            "LIMIT_BAL",
        ),
        (
            "data/predictive/creditcard_str_10k.csv",
            "data/predictive/models/creditcard-binomial-str.mojo",
            "DEFAULT_NEXT_MONTH",
        ),
        # M: not supported, but explainer must report incompatibility w/o failing
        (
            "data/predictive/creditcard_str_10k.csv",
            "data/predictive/models/creditcard-multinomial-str.mojo",
            "EDUCATION",
        ),
    ],
    ids=["CC-str-regression", "CC-str-binomial", "CC-str-multinomial"],
)
def test_mojo_str(tmpdir, dataset_path, mojo_path, target_col):
    """Test the explainer on R/B/M MOJO and (CC) dataset with **string** levels."""
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
    assert interpret.describe_explainer(explainer.PdFor2FeaturesExplainer)
    t_explainer = explainer.PdFor2FeaturesExplainer
    interpretation = interpret.run_interpretation(
        dataset=dataset_path,
        model=explainable_model,
        target_col=target_col,
        explainers=[
            commons.ExplainerToRun(
                explainer_id=explainer.PdFor2FeaturesExplainer.explainer_id(),
                params={
                    t_explainer.PARAM_PLOT_TYPE: plots.Data3dPlot.PLOT_TYPE_SURFACE,
                    # t_explainer.PARAM_PLOT_TYPE:
                    #   plots.Data3dPlot.PLOT_TYPE_HEATMAP,
                    # t_explainer.PARAM_PLOT_TYPE:
                    #   plots.Data3dPlot.PLOT_TYPE_CONTOUR,
                },
            )
        ],
        results_location=tmpdir,
        log_level=logging.DEBUG,
    )

    #
    # THEN
    #
    print(f"Interpretation:\n{interpretation}")
    assert interpretation
    assert not interpretation.is_explainer_failed()
    if not target_col == "EDUCATION":
        assert interpretation.is_explainer_scheduled()
        assert interpretation.is_explainer_finished()
        assert interpretation.is_explainer_successful()
        assert interpretation.get_scheduled_explainer_ids()
        assert interpretation.get_finished_explainer_ids()
        assert interpretation.get_successful_explainer_ids()

        failed_explainers = interpretation.get_failed_explainer_ids()
        assert not failed_explainers
        assert interpretation.get_explainer_result_metadata(
            explainer.PdFor2FeaturesExplainer().explainer_id()
        )

        result = interpretation.get_explainer_result(
            explainer.PdFor2FeaturesExplainer.explainer_id()
        )
        assert result, "Result cannot be None"


@pytest.mark.h2o_sonar
@pytest.mark.parametrize(
    "plot_type",
    [
        plots.Data3dPlot.PLOT_TYPE_HEATMAP,
        plots.Data3dPlot.PLOT_TYPE_SURFACE,
        plots.Data3dPlot.PLOT_TYPE_CONTOUR,
    ],
)
def test_explainer_params(tmpdir, plot_type):
    #
    # GIVEN
    #
    dataset_path = test_utils.find_locally("data/predictive/creditcard.csv")
    target_col = "SEX"
    mock_model = test_containers.SimpleMockModel(
        dataset_path=dataset_path, target_col=target_col
    )
    param_max_features = 4
    param_bins = 7

    #
    # WHEN
    #
    t_explainer = explainer.PdFor2FeaturesExplainer
    interpretation = interpret.run_interpretation(
        dataset=dataset_path,
        model=mock_model,
        target_col=target_col,
        explainers=[
            commons.ExplainerToRun(
                explainer_id=explainer.PdFor2FeaturesExplainer.explainer_id(),
                params={
                    t_explainer.PARAM_PLOT_TYPE: plot_type,
                    t_explainer.PARAM_MAX_FEATURES: param_max_features,
                    t_explainer.PARAM_GRID_RESOLUTION: param_bins,
                },
            )
        ],
        results_location=tmpdir,
        log_level=logging.DEBUG,
    )

    #
    # THEN
    #
    print(f"{interpretation}")
    assert interpretation
    failed_explainers = interpretation.get_failed_explainer_ids()
    assert not failed_explainers
    pd_result_meta = interpretation.get_explainer_result_metadata(
        explainer.PdFor2FeaturesExplainer().explainer_id()
    )
    print(f"Result metadata ({type(pd_result_meta)}):\n{pd_result_meta}")
    assert pd_result_meta
    pd_result = interpretation.get_explainer_result(
        explainer.PdFor2FeaturesExplainer().explainer_id()
    )
    print(f"Result ({type(pd_result)}):\n{pd_result}")
    assert pd_result

    # assert bins
    pd_feature_data = pd_result.data(feature_names="'ID' and 'LIMIT_BAL'")
    print(f"Data:\n{pd_feature_data}")
    assert pd_feature_data

    # assert explainer params
    pd_params = pd_result_meta[i13s.ExplainerJob.KEY_EXPLAINER_DESCRIPTOR]["parameters"]
    print(f"Parameters:\n{pd_params}")
    assert pd_params
    pd_args = interpretation.explainers[0].params
    print(f"Explainer arguments: {pd_args}")
    assert explainer.PdFor2FeaturesExplainer.PARAM_MAX_FEATURES in pd_args
    assert explainer.PdFor2FeaturesExplainer.PARAM_GRID_RESOLUTION in pd_args

    # assert Markdown report and images
    job_path = pd_result_meta[i13s.ExplainerJob.KEY_JOB_LOCATION]
    print(f"Job path:\n{job_path}")
    # assert max features via JSon
    pd_report_dir_path = os.path.join(job_path, "global_report", "text_markdown")
    os.path.isfile(os.path.join(pd_report_dir_path, "explanation.md"))
    for i in range(5):
        os.path.isfile(os.path.join(pd_report_dir_path, f"image-{i}.png"))


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
