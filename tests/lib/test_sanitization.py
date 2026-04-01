# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import pytest

from h2o_sonar import interpret
from h2o_sonar import loggers
from h2o_sonar.utils import sanitization
from tests import test_utils
from tests.lib import test_containers


@pytest.mark.h2o_sonar
def test_sanitization_map():
    # GIVEN

    # WHEN
    sanitization_map = sanitization.SanitizationMap(
        raw_names=[
            "raw 1",
            "raw.1",
            "raw\n1",
            "raw/1",
            "raw+1",
            "raw@1",
        ],
        sanitized_names=[
            "raw_1",
            "raw_1_",
            "raw_1__",
            "raw_1___",
            "raw_1____",
            "raw_1_____",
        ],
    )

    # THEN
    print(sanitization_map)
    print("SANITIZATION > RAW:")
    assert sanitization_map.to_raw("raw_1") == "raw 1"
    assert sanitization_map.to_raw("raw_1_") == "raw.1"
    assert sanitization_map.to_raw("raw_1_____") == "raw@1"
    assert sanitization_map.to_raw("UNKNOWN") == "UNKNOWN"

    print("RAW > SANITIZATION:")
    assert sanitization_map.to_sanitized("raw 1") == "raw_1"
    assert sanitization_map.to_sanitized("raw.1") == "raw_1_"
    assert sanitization_map.to_sanitized("raw@1") == "raw_1_____"
    assert sanitization_map.to_sanitized("UNKNOWN") == "UNKNOWN"


@pytest.mark.h2o_sonar
def test_sanitize_names():
    # GIVEN
    names = [
        "raw.1",
        "raw\n1",
        "raw/1",
        "raw+1",
        "raw@1",
    ]
    sanitization_map = sanitization.SanitizationMap(
        raw_names=names.copy(),
        sanitized_names=[
            "raw_1",
            "raw_1_",
            "raw_1__",
            "raw_1___",
            "raw_1____",
        ],
    )

    # WHEN
    sanitized_names = sanitization.sanitize_names(
        names=names + ["UN_KNOWN"],
        sanitization_map=sanitization_map,
    )

    # THEN
    assert sanitized_names == [
        "raw_1",
        "raw_1_",
        "raw_1__",
        "raw_1___",
        "raw_1____",
        "UN_KNOWN",
    ]


@pytest.mark.h2o_sonar
def test_sanitize_strings():
    # GIVEN
    strings = [
        "raw.1",
        "raw\n1",
        "raw/1",
        "raw+1",
        "raw@1",
    ]
    e = "*"

    # WHEN
    sanitized_strings = sanitization.sanitize_strings(
        strings=strings,
        replace_with=e,
    )

    # THEN
    print(f"Sanitization:\n  {strings} > {sanitized_strings}")
    assert sanitized_strings == [
        "raw*1*",
        "raw*1",
        "raw/1",
        "raw+1",
        "raw@1",
    ]


@pytest.mark.skip(reason="Feature to be implemented")
@pytest.mark.h2o_sonar
def test_sanitize_frame():
    # GIVEN
    # WHEN
    # THEN
    raise NotImplementedError


@pytest.mark.skipif(not test_utils.is_mojo_supported(), reason="MOJO is not supported")
@pytest.mark.h2o_sonar
@pytest.mark.parametrize(
    "dataset_path,mojo_path,target_col",
    [
        # regression
        (
            "data/predictive/creditcard_train_BAD_COL_NAMES_BAD_EDU_VALUES.csv",
            "data/predictive/models/creditcard-regression-raw.mojo",
            "L.I.M.I.T.BAL",
        ),
        # binomial
        (
            "data/predictive/creditcard_train_BAD_COL_NAMES_BAD_EDU_VALUES.csv",
            "data/predictive/models/creditcard-binomial-raw.mojo",
            "default.payment.next.month",
        ),
        # multinomial
        (
            "data/predictive/creditcard_train_BAD_COL_NAMES_BAD_EDU_VALUES.csv",
            "data/predictive/models/creditcard-multinomial-raw.mojo",
            "E.D.U.C.A.T.I.O.N",
        ),
    ],
)
def test_all_explainers_sanitization(tmpdir, dataset_path, mojo_path, target_col):
    """Run all explainers on raw dataset with hostile characters and (
    categorical features values).

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
    model_path = test_utils.find_locally(mojo_path)
    # model
    model = daimojo.model(model_path)
    # explainers
    explainers = None
    # from h2o_sonar.explainers import fi_kernel_shap_explainer
    # explainers = [
    #    fi_kernel_shap_explainer.KernelShapFeatureImportanceExplainer.explainer_id()
    # ]

    #
    # WHEN
    #
    interpretation = interpret.run_interpretation(
        dataset=dataset_path,
        model=model,
        target_col=target_col,
        explainer_keywords=[interpret.KEYWORD_FILTER_ALL] if not explainers else None,
        explainers=explainers,
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


@pytest.mark.h2o_sonar
@pytest.mark.parametrize(
    "text_in,text_out",
    [
        (
            (
                "<   html   ></script>[](  javascript: ) "
                "dfs [Click me|](https://mywebsite.com)"
            ),
            " </script>`[](  javascript: )` dfs `[Click me ](https://mywebsite.com)`",
        ),
        (
            "\n\r<   script   >javascript</script>[Click here](https://mywebsite.com)",
            "   javascript</script>`[Click here](https://mywebsite.com)`",
        ),
    ],
)
def test_markdown_sanitization(tmp_path, text_in, text_out):
    output = sanitization.sanitize_markdown(text_in)
    assert output == text_out
