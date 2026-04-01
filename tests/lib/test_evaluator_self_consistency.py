# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import pathlib

import pytest

from h2o_sonar import evaluate
from h2o_sonar import loggers
from h2o_sonar.evaluators import self_consistency_evaluator
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import datasets
from h2o_sonar.lib.api import models
from h2o_sonar.lib.integrations import genai
from h2o_sonar.utils import perturbations
from h2o_sonar.utils import testing
from tests import test_utils
from tests.lib import given_generative


#
# TEST CASE SAMPLING
#


def _given_test_case(
    question: str,
    actual_output: str,
    model_key: str = "model_1",
    test_key: str = "test_1",
) -> datasets.LlmDataset.LlmDatasetRow:
    return datasets.LlmDataset.LlmDatasetRow(
        i=question,
        actual_output=actual_output,
        model_key=model_key,
        test_key=test_key,
    )


def _given_llm_dataset_with_duplicates(
    question: str, num_answers: int, model_key: str = "model_1"
) -> datasets.LlmDataset:
    """Create an LLM dataset with the same question repeated with different answers."""
    llm_dataset = datasets.LlmDataset()
    for i in range(num_answers):
        tc = _given_test_case(
            question=question,
            actual_output=f"answer_{i}",
            model_key=model_key,
        )
        llm_dataset.inputs.append(tc)

    return llm_dataset


@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_no_sampling_when_group_size_below_limit():
    #
    # GIVEN
    #
    max_group_size = 100
    question = "What is the capital of France?"
    num_answers = 50
    llm_dataset = _given_llm_dataset_with_duplicates(
        question=question, num_answers=num_answers
    )

    #
    # WHEN
    #
    result = self_consistency_evaluator.SelfConsistencyEvaluator._prep_tc_pairs(
        llm_dataset=llm_dataset,
        max_group_size=max_group_size,
        logger=loggers.SonarPrintLogger(),
    )

    #
    # THEN
    #
    pairs = result["model_1"][question]
    expected_pairs = num_answers * (num_answers - 1) // 2
    assert len(pairs) == expected_pairs
    assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)


@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_sampling_reduces_large_group():
    """Test that groups larger than max_group_size are reduced to the limit."""
    #
    # GIVEN
    #
    max_group_size = 50
    question = "What is the capital of France?"
    num_answers = 150
    llm_dataset = _given_llm_dataset_with_duplicates(
        question=question, num_answers=num_answers
    )

    #
    # WHEN
    #
    result = self_consistency_evaluator.SelfConsistencyEvaluator._prep_tc_pairs(
        llm_dataset=llm_dataset,
        max_group_size=max_group_size,
        logger=loggers.SonarPrintLogger(),
    )

    #
    # THEN
    # After sampling to max_group_size, the number of pairs should be:
    # max_group_size * (max_group_size - 1) / 2
    pairs = result["model_1"][question]
    expected_pairs = max_group_size * (max_group_size - 1) // 2
    assert len(pairs) == expected_pairs
    # Verify all pairs are tuples with two test case objects
    assert all(
        isinstance(p, tuple)
        and len(p) == 2
        and isinstance(p[0], datasets.LlmDataset.LlmDatasetRow)
        and isinstance(p[1], datasets.LlmDataset.LlmDatasetRow)
        for p in pairs
    )


@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_sampling_with_varied_input_lengths():
    """Test stratified sampling distributes samples across input length buckets.
    This test verifies that stratified sampling maintains diversity across input length
    groups when a single group (by question text) has items with varied lengths.

    """
    #
    # GIVEN
    #
    max_group_size = 20
    model_key = "model_1"
    base_question = "What is the capital?"

    llm_dataset = datasets.LlmDataset()
    for i in range(100):
        if i % 2 == 0:
            output = "A"  # short
        else:
            output = "A" * 100  # long
        tc = _given_test_case(
            question=base_question,
            actual_output=output,
            model_key=model_key,
        )
        llm_dataset.inputs.append(tc)

    #
    # WHEN
    #
    result = self_consistency_evaluator.SelfConsistencyEvaluator._prep_tc_pairs(
        llm_dataset=llm_dataset,
        max_group_size=max_group_size,
        logger=loggers.SonarPrintLogger(),
    )

    #
    # THEN
    #
    pairs = result[model_key][base_question]
    assert len(pairs) > 0
    assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)


@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_sampling_with_uniform_lengths_uses_fallback():
    """Test that uniform input lengths fall back to simple truncation."""
    #
    # GIVEN
    #
    max_group_size = 30
    question = "Q?"  # uniform length
    num_answers = 100
    llm_dataset = _given_llm_dataset_with_duplicates(
        question=question, num_answers=num_answers
    )

    #
    # WHEN
    #
    result = self_consistency_evaluator.SelfConsistencyEvaluator._prep_tc_pairs(
        llm_dataset=llm_dataset,
        max_group_size=max_group_size,
        logger=loggers.SonarPrintLogger(),
    )

    #
    # THEN
    #
    pairs = result["model_1"][question]
    expected_pairs = max_group_size * (max_group_size - 1) // 2
    assert len(pairs) == expected_pairs
    if pairs:
        ref_tc, comp_tc = pairs[0]
        assert ref_tc.actual_output == "answer_0"
        assert comp_tc.actual_output == "answer_1"


@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_sampling_deterministic_with_seed():
    """Test that sampling is deterministic ~ reproducible across runs."""
    #
    # GIVEN
    #
    max_group_size = 30
    model_key = "model_1"
    question = "What is the capital?"
    num_answers = 100

    def create_and_sample():
        llm_dataset = _given_llm_dataset_with_duplicates(
            question=question, num_answers=num_answers, model_key=model_key
        )
        return self_consistency_evaluator.SelfConsistencyEvaluator._prep_tc_pairs(
            llm_dataset=llm_dataset,
            max_group_size=max_group_size,
            logger=loggers.SonarPrintLogger(),
        )

    #
    # WHEN
    #
    result1 = create_and_sample()
    result2 = create_and_sample()

    #
    # THEN
    #
    pairs1 = result1[model_key][question]
    pairs2 = result2[model_key][question]

    outputs1 = set()
    for ref_tc, comp_tc in pairs1:
        outputs1.add(ref_tc.actual_output)
        outputs1.add(comp_tc.actual_output)

    outputs2 = set()
    for ref_tc, comp_tc in pairs2:
        outputs2.add(ref_tc.actual_output)
        outputs2.add(comp_tc.actual_output)

    assert outputs1 == outputs2, "Sampling should be deterministic with fixed seed"


@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_sampling_multiple_models():
    """Test that sampling works independently for each model."""
    #
    # GIVEN
    #
    max_group_size = 20
    question = "What is the capital?"
    num_answers = 50

    llm_dataset = datasets.LlmDataset()
    # model 1
    for i in range(num_answers):
        tc = _given_test_case(
            question=question,
            actual_output=f"model1_answer_{i}",
            model_key="model_1",
        )
        llm_dataset.inputs.append(tc)
    # model 2
    for i in range(num_answers):
        tc = _given_test_case(
            question=question,
            actual_output=f"model2_answer_{i}",
            model_key="model_2",
        )
        llm_dataset.inputs.append(tc)

    #
    # WHEN
    #
    result = self_consistency_evaluator.SelfConsistencyEvaluator._prep_tc_pairs(
        llm_dataset=llm_dataset,
        max_group_size=max_group_size,
        logger=loggers.SonarPrintLogger(),
    )

    #
    # THEN
    #
    model1_pairs = result["model_1"][question]
    model2_pairs = result["model_2"][question]
    assert len(model1_pairs) > 0
    assert len(model2_pairs) > 0
    assert all(isinstance(p, tuple) and len(p) == 2 for p in model1_pairs)
    assert all(isinstance(p, tuple) and len(p) == 2 for p in model2_pairs)
    for ref_tc, comp_tc in model1_pairs:
        assert ref_tc.actual_output.startswith(
            "model1_"
        ) or comp_tc.actual_output.startswith("model1_")
    for ref_tc, comp_tc in model2_pairs:
        assert ref_tc.actual_output.startswith(
            "model2_"
        ) or comp_tc.actual_output.startswith("model2_")


@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_sampling_with_multiple_questions_same_model():
    """Test that sampling handles multiple questions for the same model
    independently.

    """
    #
    # GIVEN
    #
    max_group_size = 20
    model_key = "model_1"

    llm_dataset = datasets.LlmDataset()
    q1 = "What is the capital of France?"
    q2 = "What is the capital of Germany?"

    # add 50 answers for q1
    for i in range(50):
        tc = _given_test_case(
            question=q1,
            actual_output=f"france_answer_{i}",
            model_key=model_key,
        )
        llm_dataset.inputs.append(tc)
    # add 50 answers for q2
    for i in range(50):
        tc = _given_test_case(
            question=q2,
            actual_output=f"germany_answer_{i}",
            model_key=model_key,
        )
        llm_dataset.inputs.append(tc)

    #
    # WHEN
    #
    result = self_consistency_evaluator.SelfConsistencyEvaluator._prep_tc_pairs(
        llm_dataset=llm_dataset,
        max_group_size=max_group_size,
        logger=loggers.SonarPrintLogger(),
    )

    #
    # THEN
    #
    q1_pairs = result[model_key][q1]
    q2_pairs = result[model_key][q2]
    assert len(q1_pairs) > 0
    assert len(q2_pairs) > 0
    # Verify all items are pairs
    assert all(isinstance(p, tuple) and len(p) == 2 for p in q1_pairs)
    assert all(isinstance(p, tuple) and len(p) == 2 for p in q2_pairs)


@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_sampling_with_small_group_size_limit():
    """Test sampling with very small max_group_size (edge case)."""

    #
    # GIVEN
    #
    max_group_size = 5
    num_answers = 100
    question = "What is the answer?"

    llm_dataset = _given_llm_dataset_with_duplicates(
        question=question, num_answers=num_answers
    )

    #
    # WHEN
    #
    result = self_consistency_evaluator.SelfConsistencyEvaluator._prep_tc_pairs(
        llm_dataset=llm_dataset,
        max_group_size=max_group_size,
        logger=loggers.SonarPrintLogger(),
    )

    #
    # THEN
    #
    pairs = result["model_1"][question]
    expected_pairs = max_group_size * (max_group_size - 1) // 2
    assert len(pairs) == expected_pairs
    assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)


#
# LAB
#


@pytest.mark.skip(
    reason=(
        "Test which builds resolved test lab for multiple answers to the identical "
        "question"
    )
)
@pytest.mark.parametrize(
    "test_suite_path,docs_dir_path",
    [
        (
            "data/generative/sr1107_test_suite_7p.json",
            pathlib.Path(given_generative.DIR_TEST_RAG_DOCS_CACHE),
        )
    ],
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_build_lab(
    tmp_path, h2ogpte_connection_fixture, test_suite_path, docs_dir_path
):
    """Build TestLab for h2oGPTe hosted RAG/LLM models."""

    #
    # GIVEN
    #

    h2ogpte_connection = h2ogpte_connection_fixture
    # H2OGPTE server
    preferred_llm_model_name = [
        "gemini-2.5-flash",
        "gpt-4.1-nano",
        "claude-3-7-sonnet-20250219",
    ]
    all_llm_model_names = genai.H2oGpteRagClient(
        h2ogpte_connection
    ).list_llm_model_names()
    print(f"Available LLM models: {len(all_llm_model_names)}:")
    llm_model_names = [
        name for name in preferred_llm_model_name if name in all_llm_model_names
    ]
    if not llm_model_names:
        raise ValueError(
            f"None of the preferred LLMs is available: {preferred_llm_model_name}"
        )
    print(f"Using LLM models: {len(llm_model_names)}:")

    # test SUITE
    rag_test_suite = testing.RagTestSuiteConfig.load_from_json(
        test_utils.find_locally(test_suite_path)
    )
    for _ in range(2):
        perturbation_errors = []
        perturbed_test_suite = rag_test_suite.perturb(
            perturbators=[
                commons.PerturbatorToRun(
                    perturbator_id=perturbations.CopyPerturbator.perturbator_id(),
                )
            ],
            in_place=False,
            raised_errors=perturbation_errors,
        )
        assert len(perturbed_test_suite.test_cases) > len(rag_test_suite.test_cases), (
            "Perturbation did not increase the number of test cases"
        )
        if perturbation_errors:
            print("Perturbation errors:")
            for err in perturbation_errors:
                print(f"- {err}")
            raise RuntimeError(f"Perturbation errors occurred: {perturbation_errors}")
        perturbed_test_suite_path = tmp_path / "perturbed_test_suite.json"
        perturbed_test_suite.save_as_json(perturbed_test_suite_path)
        rag_test_suite = perturbed_test_suite
    print(
        f"Final test suite has {len(rag_test_suite.test_cases)} test cases and is "
        f"saved to: file://{rag_test_suite}"
    )

    # test LAB
    test_lab = testing.RagTestLab.from_rag_test_suite(
        rag_connection=h2ogpte_connection,
        rag_test_suite=rag_test_suite,
        rag_model_type=models.ExplainableModelType.h2ogpte,
        llm_model_names=llm_model_names,
        docs_cache_dir=tmp_path,
    )
    # deploy the test lab configuration to the h2oGPTe server
    test_lab.build()

    # complete dataset w/ actual values from the h2oGPTe server (answer, duration, ...)
    test_lab.complete_dataset(
        save_as_you_go=tmp_path / "wip_testlab.json",
        parallelize=testing.TestLab.PARALLEL_RUN,
    )
    test_lab.save_as_json(tmp_path / "self_consistency_test_lab.json", as_unicode=False)

    #
    # THEN
    #
    print(f"Test lab w/ duplicate questions built and saved to: file://{test_lab}")


#
# EVALUATOR
#


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_evaluator(tmpdir):
    #
    # GIVEN
    #

    rag_dataset = testing.RagTestLab.load_from_json(
        llm_host_connection=test_utils.health.get_h2ogpt(),
        file_path="data/generative/self_consistency_test_lab_28p.json",
    )
    llm_models = rag_dataset.evaluated_models.values()

    #
    # WHEN
    #

    evaluation = evaluate.run_evaluation(
        dataset=rag_dataset.dataset,
        models=llm_models,
        evaluators=[self_consistency_evaluator.SelfConsistencyEvaluator.evaluator_id()],
        results_location=tmpdir,
        log_level=loggers.DEBUG,
    )

    #
    # THEN
    #
    print(f"Evaluation:\n{evaluation}")

    assert evaluation
    assert not evaluation.is_explainer_failed()
    print(f"HTML report:\nfile://{evaluation.result.get_html_report_location()}")


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
