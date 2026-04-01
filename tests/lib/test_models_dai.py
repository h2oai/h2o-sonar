# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import pytest

from h2o_sonar import interpret
from h2o_sonar import loggers
from tests import test_utils
from tests.lib import test_containers


@pytest.mark.skipif(not test_utils.is_mojo_supported(), reason="MOJO is not supported")
@pytest.mark.parametrize(
    "dataset_path,mojo_path,target_col, use_explainable_model",
    [
        # CC string dataset & MOJO
        (
            "data/predictive/creditcard_str_10k.csv",
            "data/predictive/models/creditcard-regression-str.mojo",
            "LIMIT_BAL",
            True,
        ),
        (
            "data/predictive/creditcard_str_10k.csv",
            "data/predictive/models/creditcard-binomial-str.mojo",
            "DEFAULT_NEXT_MONTH",
            False,
        ),
        (
            "data/predictive/creditcard_str_10k.csv",
            "data/predictive/models/creditcard-multinomial-str.mojo",
            "EDUCATION",
            True,
        ),
        # CC numeric dataset & MOJO
        (
            "data/predictive/creditcard.csv",
            "data/predictive/models/creditcard-binomial.mojo",
            "default payment next month",
            True,
        ),
    ],
    ids=[
        "CC-str-regression",
        "CC-str-binomial",
        "CC-str-multinomial",
        "CC-num-binomial",
    ],
)
@pytest.mark.h2o_sonar
def test_all_examples_and_templates(
    tmpdir, dataset_path, mojo_path, target_col, use_explainable_model
):
    """Comprehensive test of ALL explainers + examples + templates on:

    - string columns CC dataset&MOJO: R + B + M
    - numerical columns CC dataset&MOJO: B
    - MOJOs and explainable models created from MOJO

    """
    import daimojo

    #
    # GIVEN
    #
    # container
    container = test_containers.ExplainerExamplesAndTemplatesTestContainer()
    container.setup(
        results_location=tmpdir,
        log_level=loggers.DEBUG,
    )
    # dataset
    dataset_path = test_utils.find_locally(dataset_path)
    mojo_path = test_utils.find_locally(mojo_path)
    # DAI model
    model = daimojo.model(mojo_path)
    # explainable model
    if use_explainable_model:
        model = container.model_api.create_model(
            model_src=model,
            target_col=target_col,
            used_features=list(model.feature_names),
            dataset=dataset_path,
        )

    #
    # WHEN
    #
    interpretation = interpret.run_interpretation(
        dataset=dataset_path,
        model=model,
        target_col=target_col,
        explainer_keywords=[interpret.KEYWORD_FILTER_ALL],
        results_location=tmpdir,
        log_level=loggers.DEBUG,
        container=container,
    )

    #
    # THEN
    #
    print(f"\n{interpretation}")
    # find failed explainers
    assert interpretation
    assert interpretation.result.explainers
    assert len(interpretation.result.explainers) > 5
    failed_explainers = interpretation.get_failed_explainer_ids()
    assert not failed_explainers, f"Failed explainers: {failed_explainers}"


@pytest.mark.skip("MOJO introspection")
@pytest.mark.parametrize(
    "mojo_path",
    [
        "data/predictive/models/bug-20901-daimojo.mojo",
        "data/predictive/models/creditcard-regression-str.mojo",
        "data/predictive/models/creditcard-binomial-str.mojo",
        "data/predictive/models/creditcard-multinomial-str.mojo",
        "data/predictive/models/creditcard-regression-raw.mojo",
        "data/predictive/models/creditcard-binomial.mojo",
        "data/predictive/models/creditcard-multinomial-raw.mojo",
    ],
)
@pytest.mark.h2o_sonar
def test_mojo_introspection(mojo_path):
    import daimojo

    #
    # GIVEN
    #
    mojo_path = test_utils.find_locally(mojo_path)
    model = daimojo.model(mojo_path)

    #
    # WHEN
    #
    print(f"MOJO: {mojo_path}")
    print(f"Feature names: {model.feature_names}")
    print(f"Feature types: {model.feature_types}")
    print(f"Output names : {model.output_names}")
    print(f"Output types : {model.output_types}")
    print(f"Has tree SHAP: {model.has_treeshap}")
