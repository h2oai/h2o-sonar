# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import pytest

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar import evaluate
from h2o_sonar import loggers
from h2o_sonar.evaluators import (
    answer_semantic_similarity_per_sentence_evaluator as asspse,
)
from h2o_sonar.lib.api import commons
from h2o_sonar.utils import progress
from h2o_sonar.utils import testing
from tests import test_utils
from tests.lib import given_generative
from tests.lib import then_eval


try:
    import sentence_transformers

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


# constants
AnswerSimilarityPerSentence = asspse.AnswerSemanticSimilarityPerSentenceEvaluator


@pytest.mark.skipif(
    not test_utils.health.is_h2ogpte(),
    reason="h2oGPTE service is not reachable",
)
@pytest.mark.skipif(
    not test_utils.are_python_modules_installed({"optimum"}),
    reason="Package 'optimum' is not installed",
)
@pytest.mark.skipif(
    not test_utils.are_python_modules_installed({"sentence_transformers"}),
    reason="Package 'sentence-transformers' is not installed",
)
@pytest.mark.parametrize(
    "test_lab_path,evaluator_class,expect_compatible",
    [
        (
            #
            # RAG test labs:
            #
            # "data/generative/kaggle_llm_science_exam_test_lab_2x_small_3.json",
            # "data/generative/h2ogpte_benchmark_test_lab_small.json",
            "data/generative/kaggle_llm_science_exam_test_lab_2x_small_25.json",
            #
            # EVALUATORS: AS fastest (3s @ cosine)
            #
            AnswerSimilarityPerSentence,
            #
            # Expected compatibility: True (all rows have both AA and EA)
            #
            True,
        ),
        (
            #
            # RAG test lab with 1 missing AA
            #
            "data/generative/toxicity_test_lab_2x3p_1x_EMPTY_AA.json",
            #
            # EVALUATOR: Answer Similarity Per Sentence
            #
            AnswerSimilarityPerSentence,
            #
            # Expected compatibility: True (at least 1 row has both AA and EA)
            #
            True,
        ),
        (
            #
            # RAG test lab with ALL missing AAs
            #
            "data/generative/toxicity_test_lab_2x3p_EMPTY_AA.json",
            #
            # EVALUATOR: Answer Similarity Per Sentence
            #
            AnswerSimilarityPerSentence,
            #
            # Expected compatibility: False (NO row has both AA and EA)
            #
            False,
        ),
    ],
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_sanity(
    tmpdir,
    h2ogpte_connection_fixture: h2o_sonar_config.ConnectionConfig,
    test_lab_path,
    evaluator_class,
    expect_compatible,
):
    #
    # GIVEN
    #

    test_lab = testing.RagTestLab.load_from_json(
        llm_host_connection=h2ogpte_connection_fixture,
        file_path=test_lab_path,
        docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
    )
    the_dataset = test_lab.dataset
    the_models = list(test_lab.evaluated_models.values())

    # progress: 1 stage - evaluate
    progress_callback_name = "[TEST E2E progress callback]"
    progress_callback = progress.LoggingProgressCallbackContext(
        logger=test_lab.logger,
        prefix=progress_callback_name,
        name=progress_callback_name,
    )

    #
    # WHEN
    #
    assert evaluate.describe_evaluator(evaluator_class.evaluator_id())

    evaluation = evaluate.run_evaluation(
        dataset=the_dataset,
        models=the_models,
        evaluators=[
            commons.EvaluatorToRun(
                evaluator_id=evaluator_class.evaluator_id(),
                params={evaluator_class.PARAM_METRIC_THRESHOLD: 0.986},
            )
        ],
        results_location=tmpdir,
        log_level=loggers.DEBUG,
        progress_callback=progress_callback,
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
        assert evaluator_class.evaluator_id() not in incompatible_ids, (
            f"Expected {evaluator_class.evaluator_id()} to be compatible, "
            f"but was incompatible"
        )
        assert not evaluation.is_explainer_failed()

        # assert result
        result = evaluation.get_evaluator_result(evaluator_class.evaluator_id())
        print(result)
        assert result

        # assert leaderboard JSon representation data and meta
        then_eval.then_leaderboard_json(evaluation, evaluator_class.evaluator_id())
    else:
        # evaluator should be marked as incompatible
        assert evaluator_class.evaluator_id() in incompatible_ids, (
            f"Expected {evaluator_class.evaluator_id()} to be incompatible, "
            f"but got: {incompatible_ids}"
        )

        # verify that a problem was generated
        problems = evaluation.get_explainer_problems(evaluator_class.evaluator_id())
        print(f"Problems: {problems}")
        assert len(problems) > 0, "Expected at least one problem to be reported"

        # verify the problem description contains the expected message
        problem_descriptions = [p.description for p in problems]
        problem_found = any(
            "does not contain at least one row with actual answers and expected answers"
            in desc
            for desc in problem_descriptions
        )
        assert problem_found, (
            f"Expected problem about missing fields not found. "
            f"Problems: {problem_descriptions}"
        )


@pytest.mark.skipif(
    not test_utils.are_python_modules_installed({"optimum"}),
    reason="Package 'optimum' is not installed",
)
@pytest.mark.skipif(
    not HAS_SENTENCE_TRANSFORMERS,
    reason="Required Python package 'sentence-transformers' is not installed",
)
@pytest.mark.generative
def test_empty_embeddings_edge_case():
    """Test that empty actual or expected output doesn't produce -Infinity values."""
    #
    # GIVEN
    #
    import numpy as np

    from h2o_sonar.utils import caching

    # create a mock row with empty expected output
    class MockRow:
        def __init__(self, actual_output, expected_output):
            self.key = "test_key"
            self.model_key = "test_model"
            self.i = "test prompt"
            self.actual_output = actual_output
            self.expected_output = expected_output

    # create a mock model
    class MockModel:
        def __init__(self):
            self.key = "test_model"
            self.llm_model_name = "test-llm-model"

    # initialize evaluator
    evaluator = AnswerSimilarityPerSentence()
    evaluator.logger = loggers.SonarPrintLogger()
    evaluator.models = [MockModel()]
    evaluator.args = {}
    evaluator.problems = []

    # initialize embedding model
    caching.cache_nltk_punkt(evaluator.logger)
    device = h2o_sonar_config.config.resolve_gpu_cpu_device(result_format="str")
    embedding_model = sentence_transformers.SentenceTransformer(
        AnswerSimilarityPerSentence._e_model_baai_bge,
        device=device,
        revision=caching.REVISIONS_FOR_MODEL.get(
            AnswerSimilarityPerSentence._e_model_baai_bge, "main"
        ),
    )

    #
    # WHEN
    #
    # test case 1: empty expected output
    row1 = MockRow(actual_output="This is a test.", expected_output="")
    mean1, min1, meta1 = evaluator._calculate_answer_similarity(embedding_model, row1)

    # test case 2: empty actual output
    row2 = MockRow(actual_output="", expected_output="This is a test.")
    mean2, min2, meta2 = evaluator._calculate_answer_similarity(embedding_model, row2)

    # test case 3: both empty
    row3 = MockRow(actual_output="", expected_output="")
    mean3, min3, meta3 = evaluator._calculate_answer_similarity(embedding_model, row3)

    #
    # THEN
    #
    # all should return NaN, not -Infinity
    assert np.isnan(mean1), f"Expected NaN for empty expected output, got {mean1}"
    assert np.isnan(min1), f"Expected NaN for empty expected output, got {min1}"
    assert meta1 is None

    assert np.isnan(mean2), f"Expected NaN for empty actual output, got {mean2}"
    assert np.isnan(min2), f"Expected NaN for empty actual output, got {min2}"
    assert meta2 is None

    assert np.isnan(mean3), f"Expected NaN for both empty, got {mean3}"
    assert np.isnan(min3), f"Expected NaN for both empty, got {min3}"
    assert meta3 is None

    # verify that -inf is NOT in the results (this is the main fix)
    assert not np.isinf(mean1), f"Got -Infinity for mean1: {mean1}"
    assert not np.isinf(min1), f"Got -Infinity for min1: {min1}"
    assert not np.isinf(mean2), f"Got -Infinity for mean2: {mean2}"
    assert not np.isinf(min2), f"Got -Infinity for min2: {min2}"
    assert not np.isinf(mean3), f"Got -Infinity for mean3: {mean3}"
    assert not np.isinf(min3), f"Got -Infinity for min3: {min3}"


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
