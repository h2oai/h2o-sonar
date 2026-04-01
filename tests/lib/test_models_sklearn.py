# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import logging
import os
import pickle

import datatable
import pandas
import pytest
import sklearn
from sklearn import ensemble

from h2o_sonar import interpret
from h2o_sonar.explainers import pd_ice_explainer
from tests import test_utils
from tests.lib import test_containers
from tests.test_utils import assert_interpretation


@pytest.mark.xfail(
    os.getenv("XFAIL_SCIKIT_LEARN") == "true",
    reason=(
        "There is an issue when scikit-learn model is used with SHAP module "
        "version <= 0.39.0"
    ),
)
@pytest.mark.h2o_sonar
@pytest.mark.parametrize(
    "use_explainable_model",
    [
        True,
        False,
    ],
)
@pytest.mark.h2o_sonar
def test_sklearn_all_examples_and_templates(tmpdir, use_explainable_model):
    #
    # GIVEN
    #
    # container
    container = test_containers.ExplainerExamplesAndTemplatesTestContainer()
    container.setup(
        results_location=tmpdir,
        log_level=logging.DEBUG,
    )
    # dataset
    dataset_path = test_utils.find_locally("data/predictive/creditcard.csv")
    target_col = "default payment next month"
    x_train = pandas.read_csv(dataset_path)
    (X, y) = x_train.drop(target_col, axis=1), x_train[target_col]
    # scikit-learn model
    model = ensemble.GradientBoostingClassifier(learning_rate=0.1)
    model.fit(X, y)
    if use_explainable_model:
        # explainable model
        model = container.model_api.create_model(
            model_src=model,
            target_col=target_col,
            used_features=list(X.columns),
        )
    #
    # WHEN
    #
    interpretation = interpret.run_interpretation(
        dataset=datatable.Frame(x_train),  # must have target - surrogates need it
        model=model,
        target_col=target_col,
        explainer_keywords=[interpret.KEYWORD_FILTER_ALL],
        results_location=tmpdir,
        log_level=logging.DEBUG,
        container=container,
        used_features=list(X.columns),
    )

    #
    # THEN
    #
    assert_interpretation(interpretation)


@pytest.mark.skip(reason="New sklearn pickles are not backward compatible")
@pytest.mark.xfail(
    os.getenv("XFAIL_SCIKIT_LEARN") == "true",
    reason=(
        "There is an issue when scikit-learn model is used with SHAP module "
        "version <= 0.39.0"
    ),
)
@pytest.mark.h2o_sonar
def test_pickled_sklearn_model(tmpdir):
    # GIVEN
    dataset_path = test_utils.find_locally("data/predictive/creditcard.csv")
    target_col = "default payment next month"
    x = datatable.fread(dataset_path)
    model_pickle_path = test_utils.find_locally(
        test_utils.get_version_specific_scikit_model(
            "data/predictive/models/creditcard-binomial-sklearn-gbm.pkl"
        )
    )
    guessed_user_features = list(x.names)
    guessed_user_features.remove(target_col)

    # WHEN
    interpretation = interpret.run_interpretation(
        dataset=datatable.Frame(x),
        model=model_pickle_path,
        target_col=target_col,
        results_location=tmpdir,
        log_level=logging.DEBUG,
    )

    # THEN
    print(f"Interpretation:\n{interpretation}")
    successful = interpretation.get_successful_explainer_ids()
    print(f"Successful explainers: {successful}")
    for e in [
        "h2o_sonar.explainers.dia_explainer.DiaExplainer",
        "h2o_sonar.explainers.dt_surrogate_explainer.DecisionTreeSurrogateExplainer",
        "h2o_sonar.explainers.summary_shap_explainer.SummaryShapleyExplainer",
        "h2o_sonar.explainers.pd_ice_explainer.PdIceExplainer",
    ]:
        assert e in successful


@pytest.mark.h2o_sonar
def test_bug_431(tmpdir):
    #
    # GIVEN
    #
    # dataset
    dataset_path = test_utils.find_locally(
        "data/predictive/creditcard_const_target.csv"
    )
    target_col = "MARRIAGE"
    x_train = pandas.read_csv(dataset_path)
    (X, y) = x_train.drop(target_col, axis=1), x_train[target_col]
    # scikit-learn model
    model = ensemble.GradientBoostingClassifier(learning_rate=0.1)
    model.fit(X, y)

    #
    # WHEN
    #
    interpretation = interpret.run_interpretation(
        dataset=datatable.Frame(x_train),
        model=model,
        target_col=target_col,
        explainers=[pd_ice_explainer.PdIceExplainer.explainer_id()],
        results_location=tmpdir,
        log_level=logging.DEBUG,
        used_features=list(X.columns),
    )

    #
    # THEN
    #
    print(f"Interpretation:\n{interpretation}")
    assert interpretation, "Interpretation cannot be None"
    assert interpretation.result, "Interpretation result cannot be None"
    assert interpretation.result.explainers
    assert len(interpretation.result.explainers) == 1
    failed_explainers = interpretation.get_failed_explainer_ids()
    assert not failed_explainers, f"Failed explainers: {failed_explainers}"


@pytest.mark.skip("This is a manual test used to create pickled sklearn models")
def test_pickle_sklearn_model(tmpdir):
    #
    # GIVEN
    #
    # dataset
    dataset_path = test_utils.find_locally("data/predictive/creditcard.csv")
    target_col = "default payment next month"
    x_train = pandas.read_csv(dataset_path)
    (X, y) = x_train.drop(target_col, axis=1), x_train[target_col]

    # scikit-learn model
    print(f"Sklearn version: {sklearn.__version__}")
    model = ensemble.GradientBoostingClassifier(learning_rate=0.1)
    model.fit(X, y)

    # model path
    model_path = os.path.join(
        tmpdir, f"creditcard-binomial-sklearn-{sklearn.__version__}-gbm.pkl"
    )

    #
    # WHEN
    #
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    #
    # THEN
    #
    print(f"Pickle model path: {model_path}")
    assert os.path.isfile(model_path)
