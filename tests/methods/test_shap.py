# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import logging

import datatable
import pandas
import pytest
from sklearn import ensemble
from sklearn import tree

from h2o_sonar.lib.api.models import ExplainableModelType
from h2o_sonar.methods._shap import Shap
from h2o_sonar.utils import preprocessing
from tests import test_utils
from tests.lib import test_containers


try:
    from h2o.estimators.deeplearning import H2ODeepLearningEstimator

    HAS_H2O = True
except ImportError:
    HAS_H2O = False


@pytest.mark.skipif(
    not HAS_H2O,
    reason="H2O-3 Python package is not installed",
)
def test_shap_h2o_dl_cc_reg(tmpdir, h2o3_cleanup_fixture):
    model = H2ODeepLearningEstimator()
    run_h2o_model(
        tmpdir,
        model=model,
        sample=True,
        dataset="creditcard.csv",
        target="default payment next month",
        regression=True,
    )


@pytest.mark.skipif(
    not HAS_H2O,
    reason="H2O-3 Python package is not installed",
)
def test_shap_h2o_gam_cc_reg(tmpdir, h2o3_cleanup_fixture):
    from h2o.estimators.gam import H2OGeneralizedAdditiveEstimator

    model = H2OGeneralizedAdditiveEstimator(gam_columns=["LIMIT_BAL"])
    run_h2o_model(
        tmpdir,
        model=model,
        sample=True,
        dataset="creditcard.csv",
        target="default payment next month",
        regression=True,
    )


@pytest.mark.skipif(
    not HAS_H2O,
    reason="H2O-3 Python package is not installed",
)
def test_shap_h2o_glm_cc_reg(tmpdir, h2o3_cleanup_fixture):
    from h2o.estimators.glm import H2OGeneralizedLinearEstimator

    model = H2OGeneralizedLinearEstimator()
    run_h2o_model(
        tmpdir,
        model=model,
        sample=True,
        dataset="creditcard.csv",
        target="default payment next month",
        regression=True,
    )


@pytest.mark.skipif(
    not HAS_H2O,
    reason="H2O-3 Python package is not installed",
)
def test_shap_h2o_gbm_cc_reg(tmpdir, h2o3_cleanup_fixture):
    from h2o.estimators.gbm import H2OGradientBoostingEstimator

    model = H2OGradientBoostingEstimator(ntrees=1, seed=1234)
    run_h2o_model(
        tmpdir,
        model=model,
        dataset="creditcard.csv",
        target="default payment next month",
        regression=True,
    )


@pytest.mark.skipif(
    not HAS_H2O,
    reason="H2O-3 Python package is not installed",
)
def test_shap_h2o_drf_cc_reg(tmpdir, h2o3_cleanup_fixture):
    from h2o.estimators.random_forest import H2ORandomForestEstimator

    model = H2ORandomForestEstimator()
    run_h2o_model(
        tmpdir,
        model=model,
        dataset="creditcard.csv",
        target="default payment next month",
        regression=True,
    )


@pytest.mark.skipif(
    not HAS_H2O,
    reason="H2O-3 Python package is not installed",
)
def test_shap_h2o_dl_iris(tmpdir, h2o3_cleanup_fixture):
    from h2o.estimators.deeplearning import H2ODeepLearningEstimator

    model = H2ODeepLearningEstimator()
    run_h2o_model(
        tmpdir,
        model=model,
        sample=True,
        dataset="iris.csv",
        target="class",
    )


@pytest.mark.skipif(
    not HAS_H2O,
    reason="H2O-3 Python package is not installed",
)
def test_shap_h2o_gam_iris(tmpdir, h2o3_cleanup_fixture):
    from h2o.estimators.gam import H2OGeneralizedAdditiveEstimator

    model = H2OGeneralizedAdditiveEstimator(gam_columns=["sepal_wid"])
    run_h2o_model(
        tmpdir,
        model=model,
        sample=True,
        dataset="iris.csv",
        target="class",
    )


@pytest.mark.skipif(
    not HAS_H2O,
    reason="H2O-3 Python package is not installed",
)
def test_shap_h2o_glm_iris(tmpdir, h2o3_cleanup_fixture):
    from h2o.estimators.glm import H2OGeneralizedLinearEstimator

    model = H2OGeneralizedLinearEstimator()
    run_h2o_model(
        tmpdir,
        model=model,
        sample=True,
        dataset="iris.csv",
        target="class",
    )


@pytest.mark.skipif(
    not HAS_H2O,
    reason="H2O-3 Python package is not installed",
)
def test_shap_h2o_gbm_iris(tmpdir, h2o3_cleanup_fixture):
    from h2o.estimators.gbm import H2OGradientBoostingEstimator

    model = H2OGradientBoostingEstimator(ntrees=1, seed=1234)
    run_h2o_model(
        tmpdir,
        model=model,
        dataset="iris.csv",
        target="class",
    )


@pytest.mark.skipif(
    not HAS_H2O,
    reason="H2O-3 Python package is not installed",
)
def test_shap_h2o_drf_iris(tmpdir, h2o3_cleanup_fixture):
    from h2o.estimators.random_forest import H2ORandomForestEstimator

    model = H2ORandomForestEstimator()
    run_h2o_model(
        tmpdir,
        model=model,
        dataset="iris.csv",
        target="class",
    )


@pytest.mark.skipif(
    not HAS_H2O,
    reason="H2O-3 Python package is not installed",
)
def test_shap_h2o_dl_cc_cat(tmpdir, h2o3_cleanup_fixture):
    from h2o.estimators.deeplearning import H2ODeepLearningEstimator

    model = H2ODeepLearningEstimator()
    run_h2o_model(
        tmpdir,
        model=model,
        sample=True,
        dataset="creditcard_cat_train_int_target.csv",
        target="DEFAULT_PAYMENT_NEXT_MONTH",
    )


@pytest.mark.skipif(
    not HAS_H2O,
    reason="H2O-3 Python package is not installed",
)
def test_shap_h2o_gam_cc_cat(tmpdir, h2o3_cleanup_fixture):
    from h2o.estimators.gam import H2OGeneralizedAdditiveEstimator

    model = H2OGeneralizedAdditiveEstimator(gam_columns=["LIMIT_BAL"])
    run_h2o_model(
        tmpdir,
        model=model,
        sample=True,
        dataset="creditcard_cat_train_int_target.csv",
        target="DEFAULT_PAYMENT_NEXT_MONTH",
    )


@pytest.mark.skipif(
    not HAS_H2O,
    reason="H2O-3 Python package is not installed",
)
def test_shap_h2o_glm_cc_cat(tmpdir, h2o3_cleanup_fixture):
    from h2o.estimators.glm import H2OGeneralizedLinearEstimator

    model = H2OGeneralizedLinearEstimator()
    run_h2o_model(
        tmpdir,
        model=model,
        sample=True,
        dataset="creditcard_cat_train_int_target.csv",
        target="DEFAULT_PAYMENT_NEXT_MONTH",
    )


@pytest.mark.skipif(
    not HAS_H2O,
    reason="H2O-3 Python package is not installed",
)
def test_shap_h2o_gbm_cc_cat(tmpdir, h2o3_cleanup_fixture):
    from h2o.estimators.gbm import H2OGradientBoostingEstimator

    model = H2OGradientBoostingEstimator(ntrees=1, seed=1234)
    run_h2o_model(
        tmpdir,
        model=model,
        dataset="creditcard_cat_train_int_target.csv",
        target="DEFAULT_PAYMENT_NEXT_MONTH",
    )


@pytest.mark.skipif(
    not HAS_H2O,
    reason="H2O-3 Python package is not installed",
)
def test_shap_h2o_drf_cc_cat(tmpdir, h2o3_cleanup_fixture):
    from h2o.estimators.random_forest import H2ORandomForestEstimator

    model = H2ORandomForestEstimator()
    run_h2o_model(
        tmpdir,
        model=model,
        dataset="creditcard_cat_train_int_target.csv",
        target="DEFAULT_PAYMENT_NEXT_MONTH",
    )


@pytest.mark.skipif(
    not HAS_H2O,
    reason="H2O-3 Python package is not installed",
)
def test_shap_h2o_dl_cc(tmpdir, h2o3_cleanup_fixture):
    from h2o.estimators.deeplearning import H2ODeepLearningEstimator

    model = H2ODeepLearningEstimator()
    run_h2o_model(
        tmpdir,
        model=model,
        sample=True,
        dataset="creditcard.csv",
        target="default payment next month",
    )


@pytest.mark.skipif(
    not HAS_H2O,
    reason="H2O-3 Python package is not installed",
)
def test_shap_h2o_gam_cc(tmpdir, h2o3_cleanup_fixture):
    from h2o.estimators.gam import H2OGeneralizedAdditiveEstimator

    model = H2OGeneralizedAdditiveEstimator(gam_columns=["LIMIT_BAL"])
    run_h2o_model(
        tmpdir,
        model=model,
        sample=True,
        dataset="creditcard.csv",
        target="default payment next month",
    )


@pytest.mark.skipif(
    not HAS_H2O,
    reason="H2O-3 Python package is not installed",
)
def test_shap_h2o_glm_cc(tmpdir, h2o3_cleanup_fixture):
    from h2o.estimators.glm import H2OGeneralizedLinearEstimator

    model = H2OGeneralizedLinearEstimator()
    run_h2o_model(
        tmpdir,
        model=model,
        sample=True,
        dataset="creditcard.csv",
        target="default payment next month",
    )


@pytest.mark.skipif(
    not HAS_H2O,
    reason="H2O-3 Python package is not installed",
)
def test_shap_h2o_gbm_cc(tmpdir, h2o3_cleanup_fixture):
    from h2o.estimators.gbm import H2OGradientBoostingEstimator

    model = H2OGradientBoostingEstimator(ntrees=1, seed=1234)
    run_h2o_model(
        tmpdir,
        model=model,
        dataset="creditcard.csv",
        target="default payment next month",
    )


@pytest.mark.skipif(
    not HAS_H2O,
    reason="H2O-3 Python package is not installed",
)
def test_shap_h2o_drf_cc(tmpdir, h2o3_cleanup_fixture):
    from h2o.estimators.random_forest import H2ORandomForestEstimator

    model = H2ORandomForestEstimator()
    run_h2o_model(
        tmpdir,
        model=model,
        dataset="creditcard.csv",
        target="default payment next month",
    )


def test_shap_sklearn_tree_iris(tmpdir):
    model = tree.DecisionTreeClassifier()
    run_sklearn_model(
        tmpdir,
        model=model,
        sample=True,
        dataset="iris.csv",
        target="class",
    )


def test_shap_sklearn_gbm_iris(tmpdir):
    model = ensemble.GradientBoostingClassifier()
    run_sklearn_model(
        tmpdir,
        model=model,
        dataset="iris.csv",
        target="class",
    )


def test_shap_sklearn_tree_cc_cat(tmpdir):
    model = tree.DecisionTreeClassifier()
    run_sklearn_model(
        tmpdir,
        model=model,
        sample=True,
        dataset="creditcard_cat_train_int_target.csv",
        target="DEFAULT_PAYMENT_NEXT_MONTH",
    )


def test_shap_sklearn_gbm_cc_cat(tmpdir):
    model = ensemble.GradientBoostingClassifier()
    run_sklearn_model(
        tmpdir,
        model=model,
        dataset="creditcard_cat_train_int_target.csv",
        target="DEFAULT_PAYMENT_NEXT_MONTH",
    )


def test_shap_sklearn_tree_cc(tmpdir):
    model = tree.DecisionTreeClassifier()
    run_sklearn_model(
        tmpdir,
        model=model,
        sample=True,
        dataset="creditcard.csv",
        target="default payment next month",
    )


def test_shap_sklearn_gbm_cc(tmpdir):
    model = ensemble.GradientBoostingClassifier()
    run_sklearn_model(
        tmpdir,
        model=model,
        dataset="creditcard.csv",
        target="default payment next month",
    )


@pytest.mark.skipif(not test_utils.is_mojo_supported(), reason="MOJO is not supported")
def test_shap_dai_cc_kernel_shap(tmpdir):
    import daimojo

    mojo_path = test_utils.find_locally(
        "data/predictive/models/creditcard-binomial.mojo"
    )
    dai_model = daimojo.model(mojo_path)
    dai_model.has_treeshap = False  # Forces Kernel Explainer
    run_dai_model(
        tmpdir,
        model=dai_model,
        dataset="creditcard.csv",
        target="default payment next month",
        sample=True,
    )


@pytest.mark.skipif(not test_utils.is_mojo_supported(), reason="MOJO is not supported")
def test_shap_dai_iris(tmpdir):
    import daimojo

    mojo_path = test_utils.find_locally("data/predictive/models/iris-multinomial.mojo")
    dai_model = daimojo.model(mojo_path)
    run_dai_model(
        tmpdir,
        model=dai_model,
        dataset="iris.csv",
        target="class",
    )


@pytest.mark.skipif(not test_utils.is_mojo_supported(), reason="MOJO is not supported")
def test_shap_dai_cc(tmpdir):
    import daimojo

    mojo_path = test_utils.find_locally(
        "data/predictive/models/creditcard-binomial.mojo"
    )
    dai_model = daimojo.model(mojo_path)
    run_dai_model(
        tmpdir,
        model=dai_model,
        dataset="creditcard.csv",
        target="default payment next month",
        sample=True,
    )


def run_h2o_model(tmpdir, model, dataset, target, sample=False, regression=False):
    import h2o

    # connect to H2O-3 cluster
    test_utils.h2o3_init_for_tests()
    # container
    container = test_containers.ExplainerExamplesAndTemplatesTestContainer()
    container.setup(
        results_location=tmpdir,
        log_level=logging.DEBUG,
    )
    # dataset
    dataset_path = test_utils.find_locally(f"data/{dataset}")
    df = h2o.import_file(dataset_path)
    X = list(df.names)
    X.remove(target)
    if not regression:
        df[target] = df[target].asfactor()
    # h2o model
    model.train(x=X, y=target, training_frame=df)
    # explainable model
    explainable_model = container.model_api.create_model(
        model_src=model,
        target_col=target,
    )

    # SHAP
    data_k_exp = df[1:10, X] if sample else df[X]
    check_shap(
        explainable_model,
        data_k_exp,
        X,
        model_type=explainable_model.model_type,
    )


def run_sklearn_model(tmpdir, model, dataset, target, sample=False):
    # container
    container = test_containers.ExplainerExamplesAndTemplatesTestContainer()
    container.setup(
        results_location=tmpdir,
        log_level=logging.DEBUG,
    )
    # dataset
    dataset_path = test_utils.find_locally(f"data/{dataset}")
    x_train = pandas.read_csv(dataset_path)
    (X_raw, y) = x_train.drop(target, axis=1), x_train[target]
    (X, _, _) = preprocessing.categorical_encoder(X_raw)
    # scikit-learn model
    model.fit(X, y)
    # explainable model
    explainable_model = container.model_api.create_model(
        model_src=model, target_col=target, used_features=list(X_raw.columns)
    )
    # SHAP
    data_k_exp = X[1:10] if sample else X
    check_shap(
        explainable_model,
        data_k_exp,
        list(data_k_exp.columns),
        model_type=explainable_model.model_type,
    )


def run_dai_model(tmpdir, model, dataset, target, sample=False):
    # container
    container = test_containers.ExplainerExamplesAndTemplatesTestContainer()
    container.setup(
        results_location=tmpdir,
        log_level=logging.DEBUG,
    )
    # dataset
    dataset_path = test_utils.find_locally(f"data/{dataset}")
    df = datatable.fread(dataset_path)
    del df[:, target]
    # explainable model
    explainable_model = container.model_api.create_model(
        model_src=model,
        target_col=target,
    )
    # SHAP
    df = df[1:10, :] if sample else df
    check_shap(
        explainable_model,
        df,
        explainable_model.model_src.feature_names,
        model_type=explainable_model.model_type,
    )


def check_shap(
    explainable_model,
    df,
    feature_names,
    model_type=ExplainableModelType.scikit_learn,
):
    if model_type is ExplainableModelType.driverless_ai:
        # subset frame to features used by model for DAI MOJO
        df = df[:, explainable_model.model_src.feature_names]
    # SHAP
    shap = Shap(explainable_model=explainable_model, output_names=feature_names)
    shap_values = shap.explain(df)
    x_shape = list(df.shape)
    shap_shape = list(shap_values.shape)

    if model_type is ExplainableModelType.scikit_learn:
        shap_shape[1] = (
            list(shap_values.shape)[1] - 1
            if len(explainable_model.model_src.classes_) <= 2
            else list(shap_values.shape)[1]
        )  # Shapley adds on a bias column

        if len(explainable_model.model_src.classes_) > 2:
            assert [
                df.shape[0],
                len(explainable_model.model_src.classes_) + len(list(df.columns)) * 3,
            ] == shap_shape, (
                f"Shapley values shape should match input datasets shape. Shapley "
                f"value shap: {shap_values.shape}. Input dataset shape {df.shape}"
            )
        else:
            assert x_shape == shap_shape, (
                f"Shapley values shape should match input datasets shape. Shapley "
                f"value shap: {shap_values.shape}. Input dataset shape {df.shape}"
            )
    elif model_type is ExplainableModelType.h2o3:
        shap_shape[1] = (
            list(shap_values.shape)[1] - 1
            if explainable_model.model_src._model_json["output"]["model_category"]
            != "Multinomial"
            else list(shap_values.shape)[1]
        )

        if (
            explainable_model.model_src._model_json["output"]["model_category"]
            == "Multinomial"
        ):
            assert [
                df.shape[0],
                len(
                    explainable_model.model_src._model_json["output"]["domains"][
                        explainable_model.model_src._model_json["output"][
                            "names"
                        ].index(explainable_model.meta.target_col)
                    ]
                )
                + len(list(df.columns)) * 3,
            ] == shap_shape, (
                f"Shapley values shape should match input datasets shape. Shapley "
                f"value shap: {shap_values.shape}. Input dataset shape {df.shape}"
            )
        else:
            assert x_shape == shap_shape, (
                f"Shapley values shape should match input datasets shape. Shapley "
                f"value shap: {shap_values.shape}. Input dataset shape {df.shape}"
            )
    elif explainable_model.model_type is ExplainableModelType.driverless_ai:
        if len(explainable_model.model_src.output_names) > 2:
            assert [
                df.shape[0],
                len(explainable_model.model_src.output_names) + len(list(df.names)) * 3,
            ] == shap_shape, (
                f"Shapley values shape should match input datasets shape. Shapley "
                f"value shap: {shap_values.shape}. Input dataset shape {df.shape}"
            )
        else:
            x_shape[1] = x_shape[1] + 1
            assert x_shape == shap_shape, (
                f"Shapley values shape should match input datasets shape. Shapley "
                f"value shap: {shap_values.shape}. Input dataset shape {df.shape}"
            )
    else:
        raise ValueError(
            f"Model type, {explainable_model.model_type}, is not recognized."
        )
