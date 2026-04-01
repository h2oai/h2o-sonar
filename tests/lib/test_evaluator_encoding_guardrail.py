# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import pytest

from h2o_sonar import evaluate
from h2o_sonar.evaluators import encoding_guardrail_evaluator
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import persistences
from h2o_sonar.lib.integrations import genai
from h2o_sonar.utils import encoding
from h2o_sonar.utils import perturbations
from h2o_sonar.utils import testing
from tests import test_utils
from tests.lib import given_generative
from tests.lib import then_eval


# shortcuts
EncodingEvaluator = encoding_guardrail_evaluator.EncodingGuardrailEvaluator


@pytest.mark.parametrize(
    "s,expected_enc_substr",
    [
        (
            "The given string is hex-encoded. Let's decode it first:\n\n57 68 61 74 "
            "20 69 73 20 74 68 65 20 63 61 70 69 74 61 6C 20 6F 66 20 46 72 61 6E 63 "
            "65 3F\n\nDecoding each hex pair:\n- 57 = W\n- 68 = h\n- 61 = a\n- 74 = t"
            "\n- 20 = (space)\n- 69 = i\n- 73 = s\n- 20 = (space)\n- 74 = t\n- 68 = h"
            "\n- 65 = e\n- 20 = (space)\n- 63 = c\n- 61 = a\n- 70 = p\n- 69 = i"
            "\n- 74 = t\n- 61 = a\n- 6C = l\n- 20 = (space)\n- 6F = o\n- 66 = f"
            "\n- 20 = (space)\n- 46 = F\n- 72 = r\n- 61 = a\n- 6E = n\n- 63 = c"
            "\n- 65 = e\n- 3F = ?\n\nPutting it together: "
            '"What is the capital of France?"\n\nNow, let\'s answer the question '
            'in the same base16_spaces encoding. The answer is "Paris".'
            '\n\nConverting "Paris" to hex:\n- P = 50\n- a = 61\n- r = 72'
            "\n- i = 69\n- s = 73\n\nSo the encoded answer is: 50 61 72 69 73",
            [
                "57 68 61 74 20 69 73 20 74 68 65 20 63 61 70 69 74 61 6C 20 6F 66 20 "
                "46 72 61 6E 63 65 3F",
                "50 61 72 69 73",
            ],
        ),
        (
            "50 61 72 69 73",
            ["50 61 72 69 73"],
        ),
        (
            "The answer is 50 61 72 69 73.",
            ["50 61 72 69 73"],
        ),
        (
            "Encoded answer: 50 61 72 69 73. Decoded: Paris.",
            ["50 61 72 69 73"],
        ),
        (
            "Negative case where are no hex pairs.",
            [],
        ),
        (
            "Too short as extract requires 3 pairs: 5a 61",
            [],
        ),
        (
            "Just enough with 3 pairs: 5a 61 4c",
            ["5A 61 4C"],
        ),
    ],
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_enc_substr_from_any_str(s: str, expected_enc_substr: str):
    # GIVEN
    assert isinstance(s, str)

    # WHEN
    actual_enc_substr = EncodingEvaluator._extract_encoded_substring(s=s)

    # THEN
    assert actual_enc_substr == expected_enc_substr, (
        f"Expected encoded substring '{expected_enc_substr}' "
        f"but got '{actual_enc_substr}' for string '{s}'"
    )


@pytest.mark.parametrize(
    "test_suite_path",
    [
        "data/generative/eval_llm/pii_test_suite.json",
    ],
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_genai_perturb_suite(tmp_path, h2ogpte_connection_fixture, test_suite_path):
    """Build Test Lab for h2oGPTe hosted LLM models."""

    # test SUITE
    llm_test_suite = testing.RagTestSuiteConfig.load_from_json(
        test_utils.find_locally(test_suite_path)
    )

    # optional SMALLER TEST (for debugging)
    # llm_test_suite.test_cases = llm_test_suite.test_cases[:3]

    perturbed_suite = evaluate.perturb(
        content=llm_test_suite,
        perturbators=[
            commons.PerturbatorToRun(
                perturbator_id=perturbations.EncodingPerturbatorBase16.perturbator_id(),
                params={
                    "prompt_type": (
                        perturbations.EncodingPerturbator.TYPE_PROMPT_ENCODED
                    ),
                    "answer_type": (
                        perturbations.EncodingPerturbator.TYPE_ANSWER_ENCODED
                    ),
                    "encoding_type": encoding.EncodingType.BASE16_SPACES,
                },
            )
        ],
        in_place=False,
        raised_errors=None,
    )
    perturbed_suite.save_as_json(tmp_path / "perturbed_test_suite.json")

    print(perturbed_suite)


@pytest.mark.skipif(
    not test_utils.health.is_h2ogpte(),
    reason="Test requires h2oGPTe server whose connection is not available",
)
@pytest.mark.parametrize(
    "test_suite_path",
    ["data/generative/eval_llm/encoded_perturbed_pii.json"],
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_build_lab(tmp_path, h2ogpte_connection_fixture, test_suite_path):
    """Build Test Lab for h2oGPTe hosted LLM models."""
    #
    # GIVEN
    #

    h2ogpte_connection = h2ogpte_connection_fixture
    # LLM models to evaluate
    llm_model_names = genai.H2oGpteRagClient(h2ogpte_connection).list_llm_model_names()

    # test SUITE
    llm_test_suite = testing.RagTestSuiteConfig.load_from_json(
        test_utils.find_locally(test_suite_path)
    )

    # optional SMALLER TEST (for debugging)
    llm_model_names = llm_model_names[:1]
    # llm_test_suite.test_cases = llm_test_suite.test_cases[:3]

    test_lab = testing.RagTestLab.from_llm_test_suite(
        llm_host_connection=h2ogpte_connection,
        llm_test_suite=llm_test_suite,
        llm_model_type=models.ExplainableModelType.h2ogpte_llm,
        llm_model_names=llm_model_names,
    )
    # deploy the test lab configuration to the h2oGPTe server
    test_lab.build()

    # complete dataset w/ actual values from the h2oGPTe server (answer, duration, ...)
    test_lab.complete_dataset(
        save_as_you_go=tmp_path / "wip_testlab.json",
        parallelize=testing.TestLab.PARALLEL_RUN,
    )
    test_lab.save_as_json(tmp_path / "test_lab.json")

    #
    # THEN
    #

    print(test_lab)


@pytest.mark.skipif(
    not test_utils.health.is_h2ogpte(),
    reason="Test requires h2oGPTe server whose connection is not available",
)
@pytest.mark.parametrize(
    "test_lab_path,expect_compatible",
    [
        (
            "data/generative/eval_llm/encoded_pii_perturbed_test_lab.json",
            #
            # Expected compatibility: True (all rows have AA)
            #
            True,
        ),
        (
            "data/generative/eval_llm/encoded_pii_perturbed_test_lab_1p.json",
            #
            # Expected compatibility: True (all rows have AA)
            #
            True,
        ),
        (
            "data/generative/eval_llm/encoded_bug_1415_test_lab.json",
            #
            # Expected compatibility: True (all rows have AA)
            #
            True,
        ),
        (
            "data/generative/toxicity_test_lab_2x3p_EMPTY_AA.json",
            #
            # Expected compatibility: False (NO row has AA)
            #
            False,
        ),
    ],
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_eval_lab_load(
    tmp_path, h2ogpte_connection_fixture, test_lab_path, expect_compatible
):
    #
    # GIVEN
    #

    h2ogpte_connection = h2ogpte_connection_fixture
    # test lab (load cfg w/ actual values - build/chat not needed)
    test_lab = testing.RagTestLab.load_from_json(
        llm_host_connection=h2ogpte_connection,
        file_path=test_lab_path,
        docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
    )

    #
    # WHEN
    #

    evaluator_id = EncodingEvaluator.evaluator_id()

    evaluation = evaluate.run_evaluation(
        # dataset w/ prompts, constraints, and model keys
        dataset=test_lab.dataset,
        # models to be evaluated / compared to get leaderboard
        models=list(test_lab.evaluated_models.values()),
        # evaluators
        evaluators=[
            commons.EvaluatorToRun(
                evaluator_id=evaluator_id,
                params={
                    EncodingEvaluator.PARAM_METRIC_THRESHOLD: 0.9,
                },
            )
        ],
        # where to save the report
        results_location=tmp_path,
    )

    #
    # THEN
    #

    print(f"Evaluation:\n{evaluation}")
    if expect_compatible:
        print(f"HTML:\nfile://{evaluation.result.get_html_report_location()}")
    assert evaluation

    # check compatibility based on expected result
    incompatible_ids = evaluation.get_incompatible_evaluator_ids()
    print(f"Incompatible evaluator IDs: {incompatible_ids}")

    if expect_compatible:
        # evaluator should be compatible and execute successfully
        assert evaluator_id not in incompatible_ids, (
            f"Expected {evaluator_id} to be compatible, but was incompatible"
        )
        assert not evaluation.is_explainer_failed()

        # assert result
        result = evaluation.get_evaluator_result(evaluator_id)
        print(result)
        assert result

        # result: data
        data = result.data()
        print(f"Data:\n{data}")
        assert data

        # result: summary
        summary = result.summary()
        print(f"Summary:\n{summary}")
        assert summary

        # result: plot / log / zip
        result.plot(file_path=tmp_path / "my_plot.png")
        result.log(path=tmp_path / "my_log.txt")
        result.zip(file_path=tmp_path / "my_result.zip")

        # assert leaderboard JSon representation data and meta
        then_eval.then_leaderboard_json(evaluation, evaluator_id)

        # THEN get_evaluation() w/ problems
        loaded_evaluation = _then_evaluation_load_from_json(
            evaluation_key=evaluation.key,
            results_location=evaluation.result.results_location,
        )
        # check problems deserialization
        print(f"Problems:\n{len(loaded_evaluation.result.problems)}")
        if loaded_evaluation.result.problems:
            assert loaded_evaluation.result.problems
            for p in loaded_evaluation.result.problems:
                print(p.to_dict())

        # load Markdown leaderboard
        ep = persistences.ExplainerPersistence(
            data_dir=evaluation.result.results_location,
            mli_key=evaluation.key,
            username=commons.DEFAULT_USER,
            explainer_id=evaluator_id,
            explainer_job_key=next(iter(evaluation.result.explainers)),
        )
        md_path = ep.get_explanation_file_path(
            explanation_type=e10s.LlmBoolLeaderboardExplanation.explanation_type(),
            explanation_format=f5s.MarkdownFormat.mime,
        )

        print(
            f"Explanations:\n"
            f"  HTML: file://{evaluation.result.get_html_report_location()}\n"
            f"  MD  : file://{md_path}\n"
        )
    else:
        # evaluator should be marked as incompatible
        assert evaluator_id in incompatible_ids, (
            f"Expected {evaluator_id} to be incompatible, but got: {incompatible_ids}"
        )

        # verify that a problem was generated
        problems = evaluation.get_explainer_problems(evaluator_id)
        print(f"Problems: {problems}")
        assert len(problems) > 0, "Expected at least one problem to be reported"

        # verify the problem description contains the expected message
        problem_descriptions = [p.description for p in problems]
        problem_found = any(
            ("does not contain" in desc and "actual answer" in desc)
            for desc in problem_descriptions
        )
        assert problem_found, (
            f"Expected problem about missing actual answers not found. "
            f"Problems: {problem_descriptions}"
        )


def _then_evaluation_load_from_json(evaluation_key: str, results_location: str):
    loaded_evaluation = evaluate.get_evaluation(
        evaluation_key=evaluation_key,
        results_location=results_location,
    )
    assert loaded_evaluation
    assert loaded_evaluation.key
    assert loaded_evaluation.created
    assert loaded_evaluation.progress
    assert loaded_evaluation.result
    assert loaded_evaluation.result.results_location
    assert loaded_evaluation.result.interpretation_location

    return loaded_evaluation


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
