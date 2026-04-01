# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import functools
import random
from typing import Any

import datatable
import numpy
import pandas
import pytest

from h2o_sonar import interpret
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import datasets
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import persistences
from h2o_sonar.lib.container import explainer_container
from tests import test_utils
from tests.explainers.examples import example_compatibility_check_explainer
from tests.explainers.examples import example_custom_explanation_explainer
from tests.explainers.examples import example_eda_explainer
from tests.explainers.examples import example_hello_world_explainer
from tests.explainers.examples import example_logging_explainer
from tests.explainers.examples import example_metadata_explainer
from tests.explainers.examples import example_params_explainer
from tests.explainers.examples import example_persistence_explainer
from tests.explainers.examples import example_score_explainer
from tests.explainers.templates import template_dt_explainer
from tests.explainers.templates import template_featimp_explainer
from tests.explainers.templates import template_md_explainer
from tests.explainers.templates import template_md_featimp_summary_explainer
from tests.explainers.templates import template_md_vega_explainer
from tests.explainers.templates import template_pd_explainer
from tests.explainers.templates import template_scatter_plot_explainer
from tests.explainers.templates import template_summary_featimp_explainer


TemplateMarkdownFeatImpSummaryExplainer = (
    template_md_featimp_summary_explainer.TemplateMarkdownFeatImpSummaryExplainer
)


class SimpleMockModel(models.ExplainableModel):
    ARG_SHAPLEY_CONTRIBS = "pred_contribs_original"

    def __init__(
        self,
        target_col: str,
        dataset_path: str = "",
        dataset=None,
        used_features: list | None = None,
        labels: list | None = None,
        model_type: commons.ExperimentType = commons.ExperimentType.binomial,
        feature_meta: dict | None = None,
    ):
        # dataset
        self.dataset = datasets.ExplainableDatatableDataset(
            dataset or datatable.fread(dataset_path)
        )
        # used features
        if not used_features:
            used_features = list(self.dataset.data.names)
        if target_col in used_features:
            used_features.remove(target_col)
        # target column
        self.target_col = target_col

        if commons.ExperimentType.regression == model_type:
            predict_method = _simple_r_mock_model_predict_method
            labels = labels or []
        elif commons.ExperimentType.binomial == model_type:
            predict_method = _simple_b_mock_model_predict_method
            labels = labels or [1]
        elif commons.ExperimentType.multinomial == model_type:
            labels = labels or [1, 2, 3, 4, 5]
            predict_method = functools.partial(
                _simple_m_mock_model_predict_method, labels, used_features
            )
        else:
            raise ValueError(f"Unsupported model type: '{model_type}'")

        models.ExplainableModel.__init__(
            self,
            predict_method=predict_method,
            model_type=models.ExplainableModelType.mock,
            model_src=None,
        )

        # model metadata
        self._meta = models.ExplainableModelMeta(
            used_features=used_features,
            dataset=self.dataset,
            target_col=self.target_col,
            feature_meta=feature_meta,
        )

        # labels
        self._meta.labels = labels


def register_template_explainers(
    container: explainer_container.ExplainerContainer,
):
    container.explainers_registry.register(
        template_dt_explainer.TemplateDecisionTreeExplainer
    )
    container.explainers_registry.register(
        template_featimp_explainer.TemplateFeatureImportanceExplainer
    )
    container.explainers_registry.register(TemplateMarkdownFeatImpSummaryExplainer)
    container.explainers_registry.register(
        template_md_explainer.TemplateMarkdownExplainer
    )
    container.explainers_registry.register(
        template_md_vega_explainer.TemplateMarkdownVegaExplainer
    )
    container.explainers_registry.register(
        template_pd_explainer.TemplatePartialDependenceExplainer
    )
    container.explainers_registry.register(
        template_scatter_plot_explainer.TemplateScatterPlotExplainer
    )
    container.explainers_registry.register(
        template_summary_featimp_explainer.TemplateShapleySummaryOrigFeatExplainer
    )


def register_example_explainers(
    container: explainer_container.ExplainerContainer,
):
    container.explainers_registry.register(
        example_compatibility_check_explainer.ExampleCompatibilityCheckExplainer
    )
    container.explainers_registry.register(
        example_custom_explanation_explainer.ExampleCustomExplanationExplainer
    )
    container.explainers_registry.register(example_eda_explainer.ExampleEdaExplainer)
    container.explainers_registry.register(
        example_hello_world_explainer.ExampleHelloWorldExplainer
    )
    container.explainers_registry.register(
        example_logging_explainer.ExampleLoggingExplainer
    )
    container.explainers_registry.register(
        example_metadata_explainer.ExampleMetaAndAttrsExplainer
    )
    container.explainers_registry.register(
        example_params_explainer.ExampleParamsExplainer
    )
    container.explainers_registry.register(
        example_persistence_explainer.ExamplePersistenceExplainer
    )
    container.explainers_registry.register(
        example_score_explainer.ExampleScoreExplainer
    )


class ExplainerExamplesAndTemplatesTestContainer(
    explainer_container.LocalExplainerContainer
):
    TYPE_ID = "EXAMPLES_N_TEMPLATES_TEST_CONTAINER"

    def __init__(self):
        explainer_container.LocalExplainerContainer.__init__(self)

    def setup(
        self,
        results_location: str | Any = "",
        persistence_api: persistences.PersistenceApi | None = None,
        persistence_type: (
            persistences.PersistenceType | None
        ) = persistences.PersistenceType.file_system,
        logger=None,
        log_level: int | None = None,
    ):
        explainer_container.LocalExplainerContainer.setup(
            self,
            results_location=results_location,
            persistence_api=persistence_api,
            persistence_type=persistence_type,
            logger=logger,
            log_level=log_level,
        )

        # register explainers (OOTB and BYOE are registered by local container)
        register_example_explainers(self)
        register_template_explainers(self)


def _simple_r_mock_model_predict_method(X, **kwargs):
    data = X.data if isinstance(X, datasets.ExplainableDataset) else X
    predictions = [
        i + _simple_r_mock_model_predict_method.seed for i in range(0, data.shape[0])
    ]
    _simple_r_mock_model_predict_method.seed += data.shape[0]
    return datatable.Frame(pandas.Series(predictions))


def _simple_b_mock_model_predict_method(X, **kwargs):
    data = X.data if isinstance(X, datasets.ExplainableDataset) else X
    p_predictions = [random.random() for i in range(0, data.shape[0])]
    return datatable.Frame(
        {
            # negative class
            # "P0": [1 - p for p in p_predictions],
            # positive class of interest
            "P1": p_predictions,
        }
    )


def _simple_m_mock_model_predict_method(
    classes: int | list,
    features: list[str] | None,
    X,
    **kwargs,
):
    """Mock predict method:

    - regression: returns one column with predictions
    - binomial: returns one column with predictions for the positive class of interest
    - multinomial: returns the number of columns which is the cartesian product of
      classes and features used by the model

    Parameters
    ----------
    classes : int | list
      Number of classes or the list of classes.
    features : list[str] | None
      Optional list of features used by the mock model (needed for multinomial).
    X :
      Dataset.

    """
    # IMPROVE: rows should sum to 1 (classes probabilities)
    data = X.data if isinstance(X, datasets.ExplainableDataset) else X
    frame_dict = {}

    if SimpleMockModel.ARG_SHAPLEY_CONTRIBS in kwargs and kwargs.get(
        SimpleMockModel.ARG_SHAPLEY_CONTRIBS, False
    ):
        # Shapley contributions
        if not features:
            for i in range(0, classes):
                frame_dict[f"P{i}"] = []
        elif isinstance(classes, (list, int)):
            if isinstance(classes, int):
                classes = [str(c) for c in range(classes)]
            frame_dict = dict()
            for c in classes:
                for f in features:
                    frame_dict[f"{f}.{c}"] = []
        else:
            if not features:
                raise ValueError(
                    "The list of features used by the model is required for the "
                    "multinomial predict function"
                )
            else:
                raise ValueError(
                    f"Unsupported type of predict method classes: {type(classes)}"
                )

        for _ in range(0, data.shape[0]):
            # generate classes random numbers which sum to 1.0
            predictions = numpy.random.dirichlet(
                numpy.ones(len(frame_dict.keys())), size=1
            ).tolist()[0]
            for i, k in enumerate(frame_dict):
                frame_dict[k].append(predictions[i])
    else:
        # predictions
        num_classes = (
            len(classes)
            if isinstance(classes, list)
            else (classes if isinstance(classes, int) else None)
        )
        for i in range(0, num_classes):
            frame_dict[f"P{i}"] = []

        for _ in range(0, data.shape[0]):
            # generate classes random numbers which sum to 1.0
            predictions = numpy.random.dirichlet(
                numpy.ones(num_classes), size=1
            ).tolist()[0]
            for i in range(0, num_classes):
                frame_dict[f"P{i}"].append(predictions[i])

    return datatable.Frame(frame_dict)


# initialization of predict function static variable
_simple_r_mock_model_predict_method.seed = 0


@pytest.mark.h2o_sonar
@pytest.mark.parametrize(
    "keyword_args", [{}, {SimpleMockModel.ARG_SHAPLEY_CONTRIBS: True}]
)
def test_mock_predict_methods(keyword_args):
    # GIVEN
    x = datatable.Frame(
        {
            "F1": [1, 2, 3, 4, 5],
            "F2": ["A", "B", "C", "D", "E"],
            "F3": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    print(f"X shape: {x}")

    # WHEN: regression
    y_hat = _simple_r_mock_model_predict_method(x, **keyword_args)

    # THEN: regression
    print(f"REGRESSION predictions:\n{y_hat}")

    # WHEN: binomial
    y_hat = _simple_b_mock_model_predict_method(x, **keyword_args)

    # THEN: binomial
    print(f"BINOMIAL predictions:\n{y_hat}")

    # WHEN: multinomial
    classes = 3
    m_predict = functools.partial(
        _simple_m_mock_model_predict_method,
        classes,
        x.names,
        **keyword_args,
    )
    y_hat = m_predict(x)

    # THEN: multinomial
    if keyword_args:
        print(f"MULTINOMIAL Shapley values:\n{y_hat}")
        assert classes * len(x.names) == len(y_hat.names)
    else:
        print(f"MULTINOMIAL predictions:\n{y_hat}")
        assert classes == len(y_hat.names)


@pytest.mark.h2o_sonar
def test_simple_mock_model():
    # GIVEN
    dataset_path = test_utils.find_locally("data/predictive/creditcard.csv")
    x = datatable.fread(dataset_path)
    target_col = "SEX"
    mock_model = SimpleMockModel(dataset_path=dataset_path, target_col=target_col)

    # WHEN
    predictions = mock_model.predict(x)

    # THEN
    print(f"Predictions ({type(predictions)}): {predictions}")
    assert isinstance(predictions, datatable.Frame)
    assert predictions.shape[0] == 10_000


@pytest.mark.h2o_sonar
def test_simple_mock_model_pickle(tmp_path):
    # GIVEN
    dataset_path = test_utils.find_locally("data/predictive/creditcard.csv")
    x = datatable.fread(dataset_path)
    target_col = "SEX"
    mock_model = SimpleMockModel(dataset_path=dataset_path, target_col=target_col)
    pickle_path = str(tmp_path / "simple_mock_model.pickle")

    # WHEN
    print(
        f"ORIGINAL Model:\n  target={mock_model.target_col}\n  meta={mock_model.meta}"
    )
    mock_model.save(pickle_path)
    unpickled_mock_model = SimpleMockModel.load(pickle_path)
    predictions = unpickled_mock_model.predict(x)

    # THEN
    print(
        f"PICKLE Model:"
        f"\n  target={unpickled_mock_model.target_col}"
        f"\n  meta={unpickled_mock_model.meta}"
    )
    print(f"Predictions ({type(predictions)}):\n{predictions}")
    assert isinstance(predictions, datatable.Frame)
    assert predictions.shape[0] == 10_000


@pytest.mark.parametrize(
    "explainer_class,data_args",
    [
        (template_dt_explainer.TemplateDecisionTreeExplainer, None),
        (template_featimp_explainer.TemplateFeatureImportanceExplainer, dict()),
        (
            TemplateMarkdownFeatImpSummaryExplainer,
            None,
        ),
        (template_md_explainer.TemplateMarkdownExplainer, None),
        (template_md_vega_explainer.TemplateMarkdownVegaExplainer, None),
        (
            template_pd_explainer.TemplatePartialDependenceExplainer,
            dict(feature_name="feature_1"),
        ),
        (template_scatter_plot_explainer.TemplateScatterPlotExplainer, None),
        (
            template_summary_featimp_explainer.TemplateShapleySummaryOrigFeatExplainer,
            None,
        ),
        (example_score_explainer.ExampleScoreExplainer, None),
        (example_params_explainer.ExampleParamsExplainer, None),
        (example_logging_explainer.ExampleLoggingExplainer, None),
        (
            example_compatibility_check_explainer.ExampleCompatibilityCheckExplainer,
            None,
        ),
        (example_hello_world_explainer.ExampleHelloWorldExplainer, None),
        (example_metadata_explainer.ExampleMetaAndAttrsExplainer, None),
        (example_eda_explainer.ExampleEdaExplainer, None),
        (example_persistence_explainer.ExamplePersistenceExplainer, None),
        (example_custom_explanation_explainer.ExampleCustomExplanationExplainer, None),
    ],
    ids=[
        "template_dt_explainer",
        "template_featimp_explainer",
        "template_md_featimp_summary_explainer",
        "template_md_explainer",
        "template_md_vega_explainer",
        "template_pd_explainer",
        "template_scatter_plot_explainer",
        "template_summary_featimp_explainer",
        "example_score_explainer",
        "example_params_explainer",
        "example_logging_explainer",
        "example_compatibility_check_explainer",
        "example_hello_world_explainer",
        "example_metadata_explainer",
        "example_eda_explainer",
        "example_persistence_explainer",
        "example_custom_explanation_explainer",
    ],
)
def test_template_examples_explainer(tmpdir, explainer_class, data_args):
    # GIVEN
    dataset_path = test_utils.find_locally("data/predictive/creditcard.csv")
    target_col = "AGE"
    mock_model = SimpleMockModel(
        dataset_path=dataset_path,
        target_col=target_col,
        model_type=commons.ExperimentType.regression,
    )
    # WHEN
    interpret.register_explainer(explainer_class)
    interpretation = interpret.run_interpretation(
        dataset=dataset_path,
        model=mock_model,
        target_col=target_col,
        explainers=[
            commons.ExplainerToRun(
                explainer_id=explainer_class.explainer_id(),
            )
        ],
        results_location=tmpdir,
    )
    # THEN
    assert interpretation
    failed_explainers = interpretation.get_failed_explainer_ids()
    assert not failed_explainers
    result = interpretation.get_explainer_result(explainer_class.explainer_id())
    assert result
    assert result.summary()
    assert result.params() is not None
    if data_args is not None:
        assert result.data(**data_args)


@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_not_registered_explainer(tmpdir):
    #
    # GIVEN
    #
    explainer_id = "wrong-explainer-id"

    #
    # WHEN
    #

    evaluation = interpret.run_interpretation(
        dataset="",
        model=None,
        target_col="",
        explainers=[explainer_id],
        results_location=tmpdir,
    )

    # THEN
    print(f"Evaluation:\n{evaluation}")
    assert evaluation.status == commons.ExplainerJobStatus.FAILED
    assert len(evaluation.result.problems) > 0

    assert evaluation
    assert not evaluation.is_explainer_failed()
    print(f"HTML report:\nfile://{evaluation.result.get_html_report_location()}")


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
