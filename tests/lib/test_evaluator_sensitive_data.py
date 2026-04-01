# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import pytest

from h2o_sonar import evaluate
from h2o_sonar import loggers
from h2o_sonar.evaluators import bleu_evaluator as b_e
from h2o_sonar.evaluators import rag_tokens_presence_evaluator as strstr_evaluator
from h2o_sonar.evaluators import sensitive_data_leakage_evaluator
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import persistences
from h2o_sonar.lib.integrations import genai
from h2o_sonar.utils import perturbations
from h2o_sonar.utils import testing
from tests import test_utils
from tests.lib import given_generative
from tests.lib import then_eval


@pytest.mark.skip(
    reason=(
        "Test which builds fully resolved LLM benchmark test lab which can be "
        "subsequently loaded to perform fast evaluators testing"
    )
)
@pytest.mark.parametrize(
    "host_type,test_suite_path,llm_model_names",
    [
        # SA test
        (
            models.ExplainableModelType.h2ogpte_llm,
            "data/generative/eval_llm/sensitive_data_test_suite.json",
            None,
        ),
        # BUG test: #1138
        (
            models.ExplainableModelType.h2ogpte,
            "data/generative/sr1107_test_suite_7p_perturbed.json",
            [
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                given_generative.H2OGPTE_JUDGE_LLM_MODEL_NAME,
            ],
        ),
    ],
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_build_lab(
    tmp_path, h2ogpte_connection_fixture, host_type, test_suite_path, llm_model_names
):
    """Build Test Lab for h2oGPTe hosted LLM models."""
    #
    # GIVEN
    #

    h2ogpte_connection = h2ogpte_connection_fixture
    # LLM models to evaluate
    if not llm_model_names:
        llm_model_names = genai.H2oGpteRagClient(
            h2ogpte_connection
        ).list_llm_model_names()
        # models (NOT) working
        broken_llm_model_names = ["Yukang/LongAlpaca-70B", "gemini-pro"]
        for broken_llm_model_name in broken_llm_model_names:
            if broken_llm_model_name in llm_model_names:
                llm_model_names.remove(broken_llm_model_name)

    # test SUITE
    llm_test_suite = testing.RagTestSuiteConfig.load_from_json(
        test_utils.find_locally(test_suite_path)
    )

    # optional SMALLER TEST (for debugging)
    # llm_model_names = llm_model_names[:3]
    # llm_test_suite.test_cases = llm_test_suite.test_cases[:5]

    if host_type == models.ExplainableModelType.h2ogpte:
        test_lab = testing.RagTestLab.from_rag_test_suite(
            rag_connection=h2ogpte_connection,
            rag_test_suite=llm_test_suite,
            rag_model_type=host_type,
            llm_model_names=llm_model_names,
            docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
        )
    else:
        test_lab = testing.RagTestLab.from_llm_test_suite(
            llm_host_connection=h2ogpte_connection,
            llm_test_suite=llm_test_suite,
            llm_model_type=host_type,
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
            #
            # h2oGPTe server connections
            #
            #
            # test labs: LLM
            #
            "data/generative/eval_llm/sensitive_data_test_lab.json",
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
        # NOTE: 1x_EMPTY case currently fails due to strict compatibility check
        # TODO: Investigate _check_llm_dataset_compatibility() behavior
    ],
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_eval_lab_load(
    tmp_path, h2ogpte_connection_fixture, test_lab_path, expect_compatible
):
    """h2oGPTe benchmark test which loads RESOLVED lab from the filesystem:

    - h2oGPTe server is not needed (dataset w/ actual data is loaded from filesystem)

    """
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

    evaluator_id = (
        sensitive_data_leakage_evaluator.SensitiveDataLeakageEvaluator.evaluator_id()
    )

    evaluation = evaluate.run_evaluation(
        # dataset w/ prompts, constraints and model keys
        dataset=test_lab.dataset,
        # models to be evaluated / compared to get leaderboard
        models=list(test_lab.evaluated_models.values()),
        # evaluators
        evaluators=[
            evaluator_id,
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

        # THEN get_evaluation()
        _then_evaluation_load_from_json(
            evaluation_key=evaluation.key,
            results_location=evaluation.result.results_location,
        )

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
    # print(f"Loaded evaluation:\n{loaded_evaluation}")
    assert loaded_evaluation
    # print(loaded_evaluation.key)
    assert loaded_evaluation.key
    # print(loaded_evaluation.created)
    assert loaded_evaluation.created
    # print(loaded_evaluation.status)
    # print(loaded_evaluation.progress)
    assert loaded_evaluation.progress
    # print(loaded_evaluation.result)
    assert loaded_evaluation.result
    # print(loaded_evaluation.result.results_location)
    assert loaded_evaluation.result.results_location
    # print(loaded_evaluation.result.interpretation_location)
    assert loaded_evaluation.result.interpretation_location


@pytest.mark.skip("Test for the sensitivity analysis evaluation data preparation")
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_sa_bug_1138_perturb(tmp_path):
    """Test for the sensitivity analysis bug - step 1 perturbation."""
    #
    # GIVEN
    #
    test_suite_path = "data/generative/sr1107_test_suite_7p.json"
    test_suite = testing.RagTestSuiteConfig.load_from_json(
        test_utils.find_locally(test_suite_path)
    )

    perturbators = [
        commons.PerturbatorToRun(
            perturbator_id=perturbations.CommaPerturbator.perturbator_id(),
            intensity=commons.PerturbationIntensity.MEDIUM.name,
        ),
        commons.PerturbatorToRun(
            perturbator_id=perturbations.WordSwapPerturbator.perturbator_id(),
            intensity=commons.PerturbationIntensity.MEDIUM.name,
        ),
    ]

    #
    # WHEN
    #
    perturbed_suite = evaluate.perturb(
        content=test_suite, perturbators=perturbators, in_place=False
    )

    #
    # THEN
    #
    perturbed_suite.save_as_json(tmp_path / "sr1107_test_suite_7p_perturbed.json")


@pytest.mark.parametrize(
    "evaluator_ids",
    [
        # bool leaderboard
        [
            strstr_evaluator.RagStrStrEvaluator.evaluator_id(),
        ],
        # heatmap leaderboard
        [
            b_e.BleuEvaluator.evaluator_id(),
        ],
    ],
)
@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_sa_bug_1138(tmp_path, evaluator_ids: list):
    """Test for the sensitivity analysis bug."""
    #
    # GIVEN
    #
    h2ogpte_connection = test_utils.health.get_h2ogpte()
    test_lab_path = "data/generative/sr1107_test_lab_7p_perturbed.json"

    test_lab = testing.RagTestLab.load_from_json(
        llm_host_connection=h2ogpte_connection,
        file_path=test_lab_path,
        docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
    )

    #
    # WHEN
    #
    evaluation = evaluate.run_evaluation(
        dataset=test_lab.dataset,
        models=list(test_lab.evaluated_models.values()),
        evaluators=evaluator_ids,
        results_location=tmp_path,
        log_level=loggers.DEBUG,
    )

    #
    # THEN
    #
    print(
        f"Explanations:\n"
        f"  HTML: file://{evaluation.result.get_html_report_location()}\n"
    )

    # sensitivity analysis asserts
    assert not evaluation.get_failed_evaluator_ids()
    assert evaluation.result.problems

    if evaluator_ids[0] == strstr_evaluator.RagStrStrEvaluator.evaluator_id():
        sa_problem = evaluation.result.problems[0]
        assert sa_problem.problem_type == "robustness"
        assert "robustness" in sa_problem.description
        assert "Mixtral-8x7B-Instruct-v0.1" in sa_problem.description
    elif evaluator_ids[0] == b_e.BleuEvaluator.evaluator_id():
        sa_prolems_count = 0
        for p in evaluation.result.problems:
            if p.problem_type == "robustness":
                sa_prolems_count += 1
                assert "Mixtral-8x7B-Instruct-v0.1" in p.description
            assert sa_prolems_count > 0


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
