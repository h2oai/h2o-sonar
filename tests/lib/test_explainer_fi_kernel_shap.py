# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import logging
import os

import pandas
import pytest
from sklearn import ensemble

from h2o_sonar import interpret
from h2o_sonar.explainers import fi_kernel_shap_explainer as explainer
from h2o_sonar.lib.api import commons
from h2o_sonar.methods import _shap
from tests import test_utils
from tests.lib import test_containers


# constants
KSExplainer = explainer.KernelShapFeatureImportanceExplainer


@pytest.mark.h2o_sonar
@pytest.mark.parametrize(
    "model_type,problem_threshold,problem_should_appear",
    [
        (commons.ExperimentType.regression, 0.999, False),
        (commons.ExperimentType.binomial, 0.001, True),
        (commons.ExperimentType.multinomial, 0.001, True),
    ],
)
def test_mock_model(
    tmpdir,
    model_type: commons.ExperimentType,
    problem_threshold: float,
    problem_should_appear: bool,
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
    )
    container = test_containers.ExplainerExamplesAndTemplatesTestContainer()
    container.setup(
        results_location=tmpdir,
        log_level=logging.DEBUG,
    )
    container.explainers_registry.register(
        explainer_class=KSExplainer,
    )

    # WHEN
    try:
        assert interpret.describe_explainer(KSExplainer)
        interpretation = interpret.run_interpretation(
            dataset=dataset_path,
            model=mock_model,
            target_col=target_col,
            explainers=[
                commons.ExplainerToRun(
                    KSExplainer.explainer_id(),
                    {KSExplainer.PARAM_LEAKAGE_WARN_THRESHOLD: problem_threshold},
                )
            ],
            results_location=tmpdir,
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
        assert interpretation.get_explainer_result_metadata(KSExplainer.explainer_id())
        # result
        result = interpretation.get_explainer_result(
            explainer.KernelShapFeatureImportanceExplainer.explainer_id()
        )

        if problem_should_appear:
            if model_type == commons.ExperimentType.multinomial:
                assert len(interpretation.result.problems) > 0
            else:
                assert len(interpretation.result.problems) == 1
        else:
            assert len(interpretation.result.problems) == 0

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

    finally:
        container.explainers_registry.unregister(
            explainer.KernelShapFeatureImportanceExplainer.explainer_id()
        )


@pytest.mark.h2o_sonar
@pytest.mark.parametrize(
    "model_type",
    [
        commons.ExperimentType.binomial,
    ],
)
def test_sklearn_model(tmpdir, model_type):
    #
    # GIVEN
    #
    dataset_path = test_utils.find_locally("data/predictive/creditcard.csv")
    if commons.ExperimentType.binomial == model_type:
        target_col = "default payment next month"
    else:
        raise RuntimeError(f"Unsupported model type to be tested: {model_type}")
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
        dataset=df,
        model=explainable_model,
        target_col=target_col,
        explainers=[KSExplainer.explainer_id()],
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
    # result
    result = interpretation.get_explainer_result(
        explainer.KernelShapFeatureImportanceExplainer.explainer_id()
    )
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
    ids=["creditcard-regression", "creditcard-binomial", "iris-multinomial"],
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
        explainers=[explainer.KernelShapFeatureImportanceExplainer.explainer_id()],
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
        explainer.KernelShapFeatureImportanceExplainer.explainer_id()
    )
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


@pytest.mark.h2o_sonar
def test_bug_547_548(tmpdir):
    #
    # GIVEN
    #
    dataset_path = test_utils.find_locally("data/predictive/prostate.csv")
    target_col = "DPROS"
    df = pandas.read_csv(dataset_path)
    (X_train, y) = df.drop(target_col, axis=1), df[target_col]
    # scikit-learn model
    model = ensemble.GradientBoostingClassifier(learning_rate=0.1)
    model.fit(X_train, y)
    # local container
    container = interpret.resolve_container()
    # explainable model
    explainable_model = container.model_api.create_model(
        model_src=model,
        target_col=target_col,
        dataset=df,
    )

    #
    # WHEN
    #
    interpretation = interpret.run_interpretation(
        dataset=df,
        model=explainable_model,
        target_col=target_col,
        explainers=[KSExplainer.explainer_id()],
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
    # result
    result = interpretation.get_explainer_result(
        explainer.KernelShapFeatureImportanceExplainer.explainer_id()
    )
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


@pytest.mark.h2o_sonar
def test_bug_579_m_per_label_contribs():
    """This test verifies per-label filtering of kernel SHAP contributions frame."""
    #
    # GIVEN
    #
    # feature names (from ...)
    # labels from self.labels / self.model.meta.labels - SANITIZED:
    labels = ["graduate", "other", "střední ško/la", "university"]
    # contributions from shap.explain(dataset):
    raw_shap_contribs_col_names = (
        "A.G.E.graduate",
        "BILL.AMT6.graduate",
        "BILL[AMT1.graduate",
        "BILL\\\\nAMT3.graduate",
        "BILL\\\\nAMT4.graduate",
        "BILL\\\\tAMT2.graduate",
        "BILL\\\\tAMT5.graduate",
        "L.I.M.I.T.BAL.graduate",
        "MARRIAGE.graduate",
        "PAY=2.graduate",
        "PAY=5.graduate",
        "PAY]6.graduate",
        "PAY_AMT1.graduate",
        "PAY_AMT2.graduate",
        "PAY_AMT3.graduate",
        "PAY_AMT4.graduate",
        "PAY_AMT5.graduate",
        "PAY_AMT6.graduate",
        "PAY|0.graduate",
        "PAY~3.graduate",
        "PAY~4.graduate",
        "SEX.graduate",
        "default.payment.next.month.graduate",
        "bias.graduate",
        "A.G.E.other",
        "BILL.AMT6.other",
        "BILL[AMT1.other",
        "BILL\\\\nAMT3.other",
        "BILL\\\\nAMT4.other",
        "BILL\\\\tAMT2.other",
        "BILL\\\\tAMT5.other",
        "L.I.M.I.T.BAL.other",
        "MARRIAGE.other",
        "PAY=2.other",
        "PAY=5.other",
        "PAY]6.other",
        "PAY_AMT1.other",
        "PAY_AMT2.other",
        "PAY_AMT3.other",
        "PAY_AMT4.other",
        "PAY_AMT5.other",
        "PAY_AMT6.other",
        "PAY|0.other",
        "PAY~3.other",
        "PAY~4.other",
        "SEX.other",
        "default.payment.next.month.other",
        "bias.other",
        "A.G.E.střední ško/la",
        "BILL.AMT6.střední ško/la",
        "BILL[AMT1.střední ško/la",
        "BILL\\\\nAMT3.střední ško/la",
        "BILL\\\\nAMT4.střední ško/la",
        "BILL\\\\tAMT2.střední ško/la",
        "BILL\\\\tAMT5.střední ško/la",
        "L.I.M.I.T.BAL.střední ško/la",
        "MARRIAGE.střední ško/la",
        "PAY=2.střední ško/la",
        "PAY=5.střední ško/la",
        "PAY]6.střední ško/la",
        "PAY_AMT1.střední ško/la",
        "PAY_AMT2.střední ško/la",
        "PAY_AMT3.střední ško/la",
        "PAY_AMT4.střední ško/la",
        "PAY_AMT5.střední ško/la",
        "PAY_AMT6.střední ško/la",
        "PAY|0.střední ško/la",
        "PAY~3.střední ško/la",
        "PAY~4.střední ško/la",
        "SEX.střední ško/la",
        "default.payment.next.month.střední ško/la",
        "bias.střední ško/la",
        "A.G.E.university",
        "BILL.AMT6.university",
        "BILL[AMT1.university",
        "BILL\\\\nAMT3.university",
        "BILL\\\\nAMT4.university",
        "BILL\\\\tAMT2.university",
        "BILL\\\\tAMT5.university",
        "L.I.M.I.T.BAL.university",
        "MARRIAGE.university",
        "PAY=2.university",
        "PAY=5.university",
        "PAY]6.university",
        "PAY_AMT1.university",
        "PAY_AMT2.university",
        "PAY_AMT3.university",
        "PAY_AMT4.university",
        "PAY_AMT5.university",
        "PAY_AMT6.university",
        "PAY|0.university",
        "PAY~3.university",
        "PAY~4.university",
        "SEX.university",
        "default.payment.next.month.university",
        "bias.university",
    )

    #
    # WHEN
    #
    shap_sorter = _shap.ShapContribsSorter(
        raw_shap_contribs_col_names=raw_shap_contribs_col_names,
        labels=labels,
    )

    #
    # THEN
    #
    print(shap_sorter)

    print(shap_sorter.get_bias_col_for_label("graduate"))
    assert "bias.graduate" == shap_sorter.get_bias_col_for_label("graduate")

    print(shap_sorter.get_cols_for_label("graduate"))
    assert [
        "A.G.E.graduate",
        "BILL.AMT6.graduate",
        "BILL[AMT1.graduate",
        "BILL\\\\nAMT3.graduate",
        "BILL\\\\nAMT4.graduate",
        "BILL\\\\tAMT2.graduate",
        "BILL\\\\tAMT5.graduate",
        "L.I.M.I.T.BAL.graduate",
        "MARRIAGE.graduate",
        "PAY=2.graduate",
        "PAY=5.graduate",
        "PAY]6.graduate",
        "PAY_AMT1.graduate",
        "PAY_AMT2.graduate",
        "PAY_AMT3.graduate",
        "PAY_AMT4.graduate",
        "PAY_AMT5.graduate",
        "PAY_AMT6.graduate",
        "PAY|0.graduate",
        "PAY~3.graduate",
        "PAY~4.graduate",
        "SEX.graduate",
        "default.payment.next.month.graduate",
    ] == shap_sorter.get_cols_for_label("graduate", strip_label=False)

    print(shap_sorter.get_cols_for_label("graduate", strip_label=True))
    assert [
        "A.G.E",
        "BILL.AMT6",
        "BILL[AMT1",
        "BILL\\\\nAMT3",
        "BILL\\\\nAMT4",
        "BILL\\\\tAMT2",
        "BILL\\\\tAMT5",
        "L.I.M.I.T.BAL",
        "MARRIAGE",
        "PAY=2",
        "PAY=5",
        "PAY]6",
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
        "PAY|0",
        "PAY~3",
        "PAY~4",
        "SEX",
        "default.payment.next.month",
    ] == shap_sorter.get_cols_for_label("graduate", strip_label=True)


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
