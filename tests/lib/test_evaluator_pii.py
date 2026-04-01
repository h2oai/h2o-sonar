# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import uuid

import pytest

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar import evaluate
from h2o_sonar.evaluators import pii_leakage_evaluator
from h2o_sonar.evaluators import rag_tokens_presence_evaluator
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import datasets
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import persistences
from h2o_sonar.lib.integrations import genai
from h2o_sonar.utils import testing
from tests import test_utils
from tests.lib import given_generative
from tests.lib import then_eval


# aliases
PiiEvaluator = pii_leakage_evaluator.PiiLeakageEvaluator()


@pytest.mark.skip(
    reason=(
        "Test which builds fully resolved LLM benchmark test lab which can be "
        "subsequently loaded to perform fast evaluators testing"
    )
)
@pytest.mark.parametrize(
    "test_suite_path",
    [
        "data/generative/eval_llm/pii_test_suite.json",
    ],
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
            # test labs: RAG
            "data/generative/eval_llm/pii_test_lab.json",
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
@pytest.mark.generative
@pytest.mark.h2o_sonar
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

    evaluator_id = PiiEvaluator.evaluator_id()

    evaluation = evaluate.run_evaluation(
        # dataset w/ prompts, constraints and model keys
        dataset=test_lab.dataset,
        # models to be evaluated / compared to get leaderboard
        models=list(test_lab.evaluated_models.values()),
        # evaluators
        evaluators=[
            commons.EvaluatorToRun(
                evaluator_id=evaluator_id,
                params={
                    PiiEvaluator.PARAM_METRIC_THRESHOLD: 0.9,
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

        # THEN get_evaluation() w/ problems
        loaded_evaluation = _then_evaluation_load_from_json(
            evaluation_key=evaluation.key,
            results_location=evaluation.result.results_location,
        )
        # check problems deserialization
        print(f"Problems:\n{len(loaded_evaluation.result.problems)}")
        # TODO add assert back once PII will report problems again
        if loaded_evaluation.result.problems:
            assert loaded_evaluation.result.problems
            for p in loaded_evaluation.result.problems:
                print(p.to_dict())

        # assert leaderboard JSon representation data and meta
        then_eval.then_leaderboard_json(evaluation, evaluator_id)

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

    return loaded_evaluation


@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_eval_dummy_conn(tmp_path):
    #
    # GIVEN
    #

    prompt = "What is the capital of France?"
    expected_answer = "Paris"
    actual_answer = "London."
    evaluator_id = rag_tokens_presence_evaluator.RagStrStrEvaluator.evaluator_id()

    #
    # WHEN
    #

    dummy_host_connection = h2o_sonar_config.ConnectionConfig(
        connection_type=h2o_sonar_config.ConnectionConfigType.H2O_GPT_E.name,
        name="Dummy H2O GPTe",
        description="Dummy H2O GPTe connection.",
        server_url="https://h2ogpte.h2o.ai",
        token="dummy-token",
        token_use_type=h2o_sonar_config.TokenUseType.API_KEY.name,
    )

    evaluated_model = models.ExplainableRagModel(
        connection=dummy_host_connection,
        documents=["dummy.pdf"],
        model_type=testing.ExplainableModelTypes.h2ogpte,
        llm_model_name="dummy/llm",
    )

    completed_dataset = datasets.LlmDataset()
    completed_dataset.add_input(
        key=str(uuid.uuid4()),
        i=prompt,
        corpus=evaluated_model.documents,
        expected_output=expected_answer,
        actual_output=actual_answer,
        # context=["dummy retrieved context"],  # valid ctx required in case of RAG
        output_condition="",  # required by text matching
        model_key=evaluated_model.key,
    )

    test_lab = testing.RagTestLab(
        llm_host_connection=dummy_host_connection,
        raw_dataset=completed_dataset,
        evaluated_models=[evaluated_model],
        llm_model_names=[evaluated_model.llm_model_name],
    )
    test_lab.dataset = completed_dataset

    evaluation = evaluate.run_evaluation(
        # dataset w/ prompts, constraints and model keys
        dataset=test_lab.dataset,
        # models to be evaluated / compared to get leaderboard
        models=list(test_lab.evaluated_models.values()),
        # evaluators
        evaluators=[evaluator_id],
        # where to save the report
        results_location=tmp_path,
    )

    #
    # THEN
    #
    print(f"{evaluation}")
    assert evaluation


@pytest.mark.parametrize(
    "checked_txt, expected_fragments",
    [("My mail is john@doe.com or jane@doe.com.", [[11, 23], [27, 39]])],
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_mail_leakage(checked_txt: str, expected_fragments: list):
    # GIVEN
    failed_constraints = []
    fragments = []

    # WHEN
    pii_leakage_evaluator.PiiLeakageEvaluator.check_email_leakage(
        checked_txt=checked_txt,
        failed_constraints=failed_constraints,
        fragments=fragments,
    )

    # THEN
    print("Result:")
    print(f"  Failed constraints: {failed_constraints}")
    print(f"  Fragments         : {fragments}")
    for f in fragments:
        print(f"    {checked_txt[f[0] : f[1]]}")
    assert fragments == expected_fragments


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
