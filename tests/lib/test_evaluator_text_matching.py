# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import logging
import pathlib

import pytest

from h2o_sonar import config
from h2o_sonar import evaluate
from h2o_sonar import loggers
from h2o_sonar.evaluators import rag_tokens_presence_evaluator as evaluator
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import persistences
from h2o_sonar.lib.integrations import genai
from h2o_sonar.utils import progress as progress_utils
from h2o_sonar.utils import testing
from tests import test_utils
from tests.lib import given_generative
from tests.lib import then_eval


# test suites
_TS_COLA = (
    "data/generative/eval_s3/bug-1549-cmp-diff-parse/test-suite-coca-cola-MERGED.json"
)


@pytest.mark.skip(
    reason=(
        "Test which builds fully resolved RAG benchmark test lab which can be "
        "subsequently loaded to perform fast evaluators testing"
    )
)
@pytest.mark.parametrize(
    "test_suite_path,docs_dir_path,llm_model_names",
    [
        pytest.param(
            # ### connection
            # ### suites
            pathlib.Path(_TS_COLA),
            # ### docs
            pathlib.Path(given_generative.DIR_TEST_RAG_DOCS_CACHE),
            # ### LLMs to complete (if empty, then the test uses a default set)
            [
                given_generative.LLM_GPT_4O_MINI,
                given_generative.LLM_CLAUDE_SONNET_37_LITE,
            ],
            marks=(
                pytest.mark.skipif(
                    not test_utils.is_private_test_data_available(),
                    reason="Skipped as S3 data are needed",
                ),
            ),
        )
    ],
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_build_lab(
    tmp_path: pathlib.Path,
    h2ogpte_connection_fixture,
    test_suite_path: pathlib.Path,
    docs_dir_path: pathlib.Path,
    llm_model_names: list[str],
):
    """Build TestLab for h2oGPTe hosted RAG/LLM models."""

    #
    # GIVEN
    #

    h2ogpte_connection = h2ogpte_connection_fixture
    # test SUITE
    rag_test_suite = testing.RagTestSuiteConfig.load_from_json(test_suite_path)
    # optional SMALLER TEST (for debugging)
    # rag_test_suite.test_cases = rag_test_suite.test_cases[:1]

    # LLMs to complete the test lab
    if not llm_model_names:
        # H2OGPTE server
        llm_model_names = genai.H2oGpteRagClient(
            h2ogpte_connection
        ).list_llm_model_names()
        # optional SMALLER TEST (for debugging)
        llm_model_names = llm_model_names[:3]

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
    test_lab.save_as_json(tmp_path / "test_lab.json", as_unicode=False)

    #
    # THEN
    #
    print(test_lab)


@pytest.mark.skip(
    reason=(
        "Test which builds fully resolved LLM benchmark test lab which can be "
        "subsequently loaded to perform fast evaluators testing"
    )
)
@pytest.mark.parametrize(
    "test_suite_path",
    [
        (
            # ### connection
            test_utils.health.get_h2ogpt(),
            # ### suites
            # "data/generative/h2ogpte_benchmark_test_suite.json",
            # "data/generative/h2ogpte_benchmark_test_suite_top.json",
            # "data/generative/h2ogpte_benchmark_test_suite_small.json",
            "data/generative/eval_llm/bank_teller_test_suite.json",
        ),
    ],
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_build_h2ogpte_benchmark_lab(
    tmp_path, h2ogpte_connection_fixture, test_suite_path
):
    """Build Test Lab for h2oGPTe hosted LLM models."""
    #
    # GIVEN
    #
    h2ogpte_connection = h2ogpte_connection_fixture
    llm_model_type = (
        models.ExplainableModelType.h2ogpte_llm
        if h2ogpte_connection.connection_type
        == config.ConnectionConfigType.H2O_GPT_E.name
        else models.ExplainableModelType.h2ogpt
    )

    # H2OGPTE server as LLM host
    if llm_model_type == models.ExplainableModelType.h2ogpte_llm:
        llm_model_names = genai.H2oGpteRagClient(
            h2ogpte_connection
        ).list_llm_model_names()
    else:
        llm_model_names = genai.H2oGptLlmClient(
            h2ogpte_connection
        ).list_llm_model_names()
    # models (NOT) working
    # broken_llm_model_names = ["Yukang/LongAlpaca-70B", "gemini-pro"]
    # for broken_llm_model_name in broken_llm_model_names:
    #     if broken_llm_model_name in llm_model_names:
    #         llm_model_names.remove(broken_llm_model_name)

    # test SUITE
    llm_test_suite = testing.RagTestSuiteConfig.load_from_json(
        test_utils.find_locally(test_suite_path)
    )

    # optional SMALLER TEST (for debugging)
    # llm_model_names = llm_model_names[:5]
    # llm_test_suite.test_cases = llm_test_suite.test_cases[:5]

    test_lab = testing.RagTestLab.from_llm_test_suite(
        llm_host_connection=h2ogpte_connection,
        llm_test_suite=llm_test_suite,
        llm_model_type=llm_model_type,
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


@pytest.mark.skip("Test requires h2oGPTe server")
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_h2ogpte_benchmark_api(tmp_path):
    """Minimal h2oGPTe benchmark test: API-based, 3 tests/1 doc, 1 base model:

    - test config is created using API
    - test lab is build (documents are uploaded)
    - actual dataset values are resolved (chat)
    - evaluation is run

    """
    #
    # GIVEN
    #

    # h2oGPTe server
    h2ogpte_connection = test_utils.health.get_h2ogpte()

    # test config
    test_config = testing.RagTestConfig(
        documents=[
            "https://eval-studio-artifacts.s3.amazonaws.com"
            "/h2o-eval-studio-suite-library/corpus-h2ogpte-benchmark"
            "/Coca-Cola-FEMSA-Results-1Q23-vf-2.pdf"
        ]
    )

    # test lab
    test_lab = testing.RagTestLab.from_rag_test_suite(
        rag_connection=h2ogpte_connection,
        rag_test_suite=testing.RagTestSuiteConfig(
            [
                testing.RagTestCaseConfig(
                    prompt="What was the revenue of Brazil?",
                    constraints=["15,969", "million"],
                    config=test_config,
                ),
                testing.RagTestCaseConfig(
                    prompt="What was the revenue of Mexico?",
                    constraints=["27,229", "million"],
                    config=test_config,
                ),
                testing.RagTestCaseConfig(
                    prompt="How did gross profit change YoY for South America?",
                    constraints=["11%"],
                    config=test_config,
                ),
            ]
        ),
        rag_model_type=models.ExplainableModelType.h2ogpte,
        llm_model_names=["h2oai/h2ogpt-4096-llama2-70b-chat"],
        docs_cache_dir=tmp_path,
    )
    # deploy the test lab configuration to the h2oGPTe server
    test_lab.build()

    # complete dataset w/ actual values from the h2oGPTe server (answer, duration, ...)
    test_lab.complete_dataset()
    test_lab.save_as_json(
        tmp_path / "h2ogpte_benchmark_test_lab_with_actual_values.json"
    )

    #
    # WHEN
    #

    evaluation = evaluate.run_evaluation(
        # dataset w/ prompts, constraints and model keys
        dataset=test_lab.dataset,
        # models to be evaluated / compared to get leaderboard
        models=test_lab.evaluated_models,
        # evaluators
        evaluators=[evaluator.RagStrStrEvaluator.evaluator_id()],
        # where to save the report
        results_location=tmp_path,
    )

    #
    # THEN
    #

    print(f"{evaluation}")
    assert not evaluation.get_failed_evaluator_ids()
    # result: leaderboard
    result = evaluation.get_evaluator_result(
        evaluator.RagStrStrEvaluator().evaluator_id()
    )
    assert result
    result.data()
    print(result.data())


@pytest.mark.parametrize(
    "constraint,expected_condition",
    [
        ([], ""),
        (["REGEXP:^C.*"], 'regexp("^C.*")'),
        (
            ["conceptual soundness", "ongoing monitoring", "outcomes analysis"],
            '"conceptual soundness" AND "ongoing monitoring" AND "outcomes analysis"',
        ),
        (
            ["15,969", "REGEXP:15?,969 [Mm]illion", ["BRAZIL", "REGEXP:[B]razil"]],
            (
                '"15,969" AND regexp("15?,969 [Mm]illion") AND ('
                '"BRAZIL" OR regexp("[B]razil")'
                ")"
            ),
        ),
    ],
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_legacy_constraints_to_conditions(constraint, expected_condition):
    #
    # GIVEN
    #
    print(f"Constraint:\n{constraint}")

    #
    # WHEN
    #
    condition = evaluator.constraints_to_condition(constraint)
    print(f"Condition:\n'{condition}'")

    #
    # THEN
    #
    assert expected_condition == condition


@pytest.mark.parametrize(
    "expected_output,actual_output",
    [
        (
            '""ةدوجب" signifies the result or outcome of "ةيرارمتلا"."',
            'According to the document text provided, "ةدوجب" (quality) and "ةيرارمتلا"'
            " (continuity) are presented "
            "as complementary concepts in the context of customer service and business"
            " success.\n\nIn the document"
            ' "arabic-crm.pdf" on page 8, there is a section about the importance of'
            " customer service"
            " (أهمية خدمة العملاء) that "
            "establishes the relationship between these concepts:\n\n1. One point"
            " mentions"
            ' "استمرارية نجاح المنظمة والمحافظة على ربحيتها" '
            "(continuity of the organization's success and "
            "maintaining its profitability) as a key benefit of good customer "
            "service.\n\n2. Another "
            'point refers to "التحسين المستمر لجودة الخدمة المقدمة" '
            "(continuous improvement of the quality of service provided).\n\nThe "
            "relationship between these concepts suggests that maintaining service"
            " quality (جودة) is an essential "
            "component for ensuring business continuity and success (استمرارية). In "
            "the customer service framework presented in this document, quality is "
            "positioned as a driver of "
            "continuity - meaning that consistent quality in customer service leads "
            "to business sustainability and "
            "ongoing success.\n\nAdditionally, on page 9, the document mentions that"
            " service quality (ةدوجب) is "
            "something that attracts new customers who become aware of it, further"
            " reinforcing the idea "
            "that quality is a prerequisite for business continuity and growth.\n\nThis"
            " relationship aligns with "
            "modern customer relationship management (CRM) principles where service"
            " quality is considered fundamental"
            " to building long-term customer relationships and ensuring business "
            "sustainability.",
        ),
    ],
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_condition_eval_bugs(expected_output: str, actual_output: str):
    #
    # GIVEN
    #
    print(f"Expected output:\n{expected_output}")
    output_constraints = [expected_output]

    #
    # WHEN
    #
    (
        ok,
        failed_retrieval,
        failed_answer,
        err_msg,
        err_msg_html,
    ) = evaluator.RagStrStrEvaluator._eval_input_condition(
        llm_host=commons.LlmModelHostType.RAG,
        actual_output=actual_output,
        context="",
        condition="",
        constraints=output_constraints,
        do_eval_rc=False,
        logger=loggers.SonarPrintLogger(),
    )

    #
    # THEN
    #
    assert not ok, f"OK must be false: {ok}"
    assert not failed_retrieval, f"Failed retrieval must be false: {failed_retrieval}"
    assert failed_answer, f"Failed answer must be true: {failed_answer}"
    assert err_msg, f"Err message must be NON empty: {err_msg}"
    assert err_msg_html, f"HTML err message must be NON empty: {err_msg_html}"


@pytest.mark.skipif(
    not test_utils.health.is_h2ogpte(),
    reason="Test requires h2oGPTe server whose connection is not available",
)
@pytest.mark.parametrize(
    "test_lab_path,test_custom_progress",
    [
        # test labs: RAG
        (
            "data/generative/conferences/atlanta-2024/sr1107_test_lab_large.json",
            True,
        ),
        # test labs: LLM
        (
            "data/generative/eval_llm/bank_teller_h2ogpt_test_lab.json",
            False,
        ),
        (
            "data/generative/eval_llm/bank_teller_h2ogpt_test_lab.json",
            True,
        ),
        (
            # perturbation + conditions (not constraints)
            "data/generative/eval_llm/perturbed_test_lab_2p.json",
            False,
        ),
        (
            "data/generative/nist-ai-600-1--test-lab--30p-5m.json",
            False,
        ),
        (
            "data/generative/bugs/82a80332-b363-4cf2-8735-44fb69cbabb2_test_lab.json",
            False,
        ),
        # BUG: test rounding
        (
            "data/generative/h2ogpte_benchmark_test_lab_3x5_actuals.json",
            False,
        ),
        # tokenization w/ visualizations
        (
            "data/generative/tokens-presence-tokenization-eval.json",
            False,
        ),
        # BUG: text to SQL
        (
            "data/generative/eval_llm/bug-1315-test-lab.json",
            False,
        ),
        # internal model host error testing
        (
            "data/generative/sr1107_test_lab_171.json",
            False,
        ),
        # arabic i18n #1370 bug testing
        (
            "data/generative/eval_llm/arabic-bug-1370-test-lab-3p.json",
            False,
        ),
        # docstring and .rst examples
        (
            "data/generative/doc_text_matching_regexps.json",
            False,
        ),
    ],
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_evaluator_lab_load(
    tmp_path, h2ogpte_connection_fixture, test_lab_path, test_custom_progress
):
    """h2oGPTe benchmark test which loads RESOLVED lab from the filesystem:

    - h2oGPTe server is not needed (dataset w/ actual data is loaded from filesystem)

    """
    #
    # GIVEN
    #
    h2ogpte_connection = h2ogpte_connection_fixture
    threshold = 0.543

    # progress callback
    if test_custom_progress:

        class MyCustomCallback(progress_utils.ProgressCallbackContext):
            def __init__(
                self,
                min_progress: float = 0.25,
                max_progress: float = 0.75,
            ):
                progress_utils.ProgressCallbackContext.__init__(
                    self,
                    min_progress=min_progress,
                    max_progress=max_progress,
                    name="[TEST custom progress callback]",
                )
                self.debug_progress_log: list = []

            def set_progress(
                self, progress: float, message: str | None = None
            ) -> float:
                print(f"{self.name}: progress={progress}")

                # normalize progress reported by child to the range of this callback
                progress = progress_utils.ProgressCallbackContext.set_progress(
                    self, progress, message
                )
                # record progress
                self.debug_progress_log.append(self.relative_progress)

                return progress

        progress_callback = MyCustomCallback()
    else:
        progress_callback = None

    # test lab (load cfg w/ actual values - build/chat not needed)
    test_lab = testing.RagTestLab.load_from_json(
        llm_host_connection=h2ogpte_connection,
        file_path=test_lab_path,
        docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
    )

    #
    # WHEN
    #

    evaluation = evaluate.run_evaluation(
        # dataset w/ prompts, constraints and model keys
        dataset=test_lab.dataset,
        # models to be evaluated / compared to get leaderboard
        models=list(test_lab.evaluated_models.values()),
        # evaluators
        evaluators=[
            commons.EvaluatorToRun(
                evaluator_id=evaluator.RagStrStrEvaluator().evaluator_id(),
                params={
                    evaluator.RagStrStrEvaluator.PARAM_METRIC_THRESHOLD: threshold,
                },
            ),
        ],
        # where to save the report
        results_location=tmp_path,
        # progress callback
        progress_callback=progress_callback,
        # ensure progress is printed to the log
        log_level=logging.INFO,
    )

    #
    # THEN
    #

    print(f"{evaluation}")
    assert not evaluation.get_failed_evaluator_ids()

    # assert evaluation type
    is_rag = models.ExplainableModelType.is_rag(
        evaluation.result.models[0].model_type if evaluation.result.models else None
    )
    print(f"Is RAG evaluation? {is_rag}")

    # assert leaderboard JSon
    then_eval.then_leaderboard_json(
        evaluation=evaluation,
        evaluator_id=evaluator.RagStrStrEvaluator().evaluator_id(),
    )

    # load Markdown leaderboard
    ep = persistences.ExplainerPersistence(
        data_dir=evaluation.result.results_location,
        mli_key=evaluation.key,
        username=commons.DEFAULT_USER,
        explainer_id=evaluator.RagStrStrEvaluator().evaluator_id(),
        explainer_job_key=next(iter(evaluation.result.explainers)),
    )
    md_path = ep.get_explanation_file_path(
        explanation_type=e10s.LlmBoolLeaderboardExplanation.explanation_type(),
        explanation_format=f5s.MarkdownFormat.mime,
    )
    md_es_path = ep.get_explanation_file_path(
        explanation_type=e10s.LlmBoolLeaderboardExplanation.explanation_type(),
        explanation_format=f5s.EvalStudioMarkdownFormat.mime,
    )
    json_leaderboard_path = ep.get_explanation_file_path(
        explanation_type=e10s.LlmBoolLeaderboardExplanation.explanation_type(),
        explanation_format=f5s.LlmLeaderboardJSonFormat.mime,
    )
    json_eval_result_path = ep.get_explanation_file_path(
        explanation_type=e10s.LlmEvalResultsExplanation.explanation_type(),
        explanation_format=f5s.CustomJsonFormat.mime,
    )

    # result: leaderboard
    result = evaluation.get_explainer_result(
        evaluator.RagStrStrEvaluator().evaluator_id()
    )
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
    # result: JSon leaderboard
    with open(json_leaderboard_path) as f:
        json_leaderboard = f.read()
    print(f"Leaderboard:\n{json_leaderboard}")
    assert json_leaderboard
    # progress
    assert evaluation.progress == 1.0
    assert evaluation.progress_callback.progress == 1.0
    if test_custom_progress:
        print(progress_callback.debug_progress_log)
        assert progress_callback.debug_progress_log
        assert min(progress_callback.debug_progress_log) >= 0.25
        assert max(progress_callback.debug_progress_log) <= 0.75

    print(f"Problems [{len(evaluation.result.problems)}]")
    for p in evaluation.result.problems:
        print(f"  {json.dumps(p.to_dict(), indent=2)}")

    json_leaderboard_data_path = json_leaderboard_path.replace(
        "explanation.json", "leaderboard_0.json"
    )
    with open(json_leaderboard_data_path) as f:
        leaderboard_data_dict = json.load(f)
    print(f"Leaderboard data:\n{leaderboard_data_dict}")
    for m in leaderboard_data_dict[f5s.ExplanationFormat.KEY_METADATA]:
        actual_threshold = leaderboard_data_dict[f5s.ExplanationFormat.KEY_METADATA][m][
            commons.MetricMeta.KEY_THRESHOLD
        ]
        assert actual_threshold == threshold, (
            f"Expected {threshold} vs. actual {actual_threshold}"
        )

    print(
        f"Explanations:\n"
        f"  HTML : file://{evaluation.result.get_html_report_location()}\n"
        f"  MD ES: file://{md_es_path}\n"
        f"  MD   : file://{md_path}\n"
        f"  JSon : file://{json_leaderboard_path}\n"
        f"  Data : file://{json_leaderboard_data_path}\n"
        f"  Eval : file://{json_eval_result_path}\n"
    )

    #
    # THEN get_evaluation()
    #
    _then_evaluation_load_from_json(
        evaluation_key=evaluation.key,
        results_location=evaluation.result.results_location,
    )

    #
    # THEN assert that for every test lab's dataset row key
    #      there is a corresponding evaluation result row key
    #

    # TODO technical debt: test case keys are not UNIQUE in the evaluation result
    with open(json_eval_result_path) as f:
        json_eval_result_dict = json.load(f)
    json_eval_result_keys = []
    json_eval_result_tc_keys = set()
    for i in json_eval_result_dict["results"]:
        json_eval_result_tc_keys.add(i["test_case_key"])
        if i["key"] not in json_eval_result_keys:
            json_eval_result_keys.append(i["key"])
        else:
            # assert that the evaluation result items has unique keys
            raise Exception(
                f"Duplicate key in the results of {test_lab_path}: {i['test_case_key']}"
            )
    # check that all test lab's dataset row keys are in the evaluation result
    for lab_row in test_lab.dataset.inputs:
        assert lab_row.key in json_eval_result_tc_keys, (
            f"Missing test LAB key {lab_row.key} in the evaluation result keys:\n"
            f"{json_eval_result_tc_keys}"
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


@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.skip(reason="Test requires h2oGPTe server")
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_evaluator(tmp_path):
    """Test evalGPT evaluator ~ h2oGPTe benchmark ~ to get leaderboard as .md.

    - loads test configuration
    - creates test lab configuration
       - builds test lab (uploads documents)
       - resolves actual dataset columns (chat w/ h2oGPTe)
    - runs evaluation

    Nothing, except the test configuration, is loaded.

    """
    bug_2024_08_02 = False

    #
    # GIVEN
    #

    # OPENAI RAG SERVER
    # h2ogpte_connection = given_generative.OPENAI_RAG
    # llm_model_names = genai.OpenAiAssistantsRagClient(
    #     h2ogpte_connection
    # ).list_llm_model_names()

    if bug_2024_08_02:
        connection = test_utils.health.get_h2ogpte()
        llm_model_names = ["mistralai/Mixtral-8x7B-Instruct-v0.1"]
        test_suite_path = "data/generative/eval_llm/bank_teller_test_suite.json"
        evaluators_to_run = [evaluator.RagStrStrEvaluator().evaluator_id()]
    else:
        connection = test_utils.health.get_h2ogpte()
        # default test scenario
        test_suite_path = "data/generative/h2ogpte_benchmark_test_suite_min.json"
        # "data/generative/h2ogpte_benchmark_test_suite_no_corpus.json"
        # "data/generative/h2ogpte_benchmark_test_suite_top.json"
        # "data/generative/h2ogpte_benchmark_test_suite_demo.json"
        # "data/generative/h2ogpte_benchmark_test_suite.json"
        # "data/generative/h2ogpte_benchmark_test_suite_no_constraints.json"
        llm_model_names = genai.H2oGpteRagClient(connection).list_llm_model_names()

        # OPTIONAL SMALLER TEST (to finish quickly - for debugging)
        llm_model_names = llm_model_names[:1]
        # rag_test_suite.test_cases = rag_test_suite.test_cases[:3]

        evaluators_to_run = [evaluator.RagStrStrEvaluator().evaluator_id()]

    # TEST CONFIGURATION
    rag_test_suite = testing.RagTestSuiteConfig.load_from_json(
        test_utils.find_locally(test_suite_path)
    )

    # TEST LAB
    if bug_2024_08_02:
        test_lab = testing.RagTestLab.from_llm_test_suite(
            llm_host_connection=connection,
            llm_test_suite=rag_test_suite,
            llm_model_type=models.ExplainableModelType.h2ogpte,
            llm_model_names=llm_model_names,
            work_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
        )
    else:
        test_lab = testing.RagTestLab.from_rag_test_suite(
            rag_connection=connection,
            rag_test_suite=rag_test_suite,
            rag_model_type=models.ExplainableModelType.h2ogpte,
            llm_model_names=llm_model_names,
            docs_cache_dir=given_generative.DIR_TEST_RAG_DOCS_CACHE,
        )
    # deploy the test lab configuration to the h2oGPTe server
    test_lab.build()

    # complete dataset w/ actual values from the h2oGPTe server (answer, duration, ...)
    test_lab.complete_dataset(
        save_as_you_go=tmp_path / "wip_testlab.json",
        parallelize=testing.TestLab.PARALLEL_RUN,
    )
    # backup fully resolved dataset
    test_lab.save_as_json(tmp_path / "test_lab.json")

    #
    # WHEN
    #

    evaluation = evaluate.run_evaluation(
        # dataset w/ prompts, constraints and model keys
        dataset=test_lab.dataset,
        # models to be evaluated / compared to get leaderboard
        models=test_lab.evaluated_models.values(),
        # evaluators
        evaluators=evaluators_to_run,
        # where to save the report
        results_location=tmp_path,
    )

    #
    # THEN
    #
    print(f"{evaluation}")
    assert evaluation
    assert not evaluation.get_failed_evaluator_ids()
    # assert result
    result = evaluation.get_evaluator_result(
        evaluator.RagStrStrEvaluator().evaluator_id()
    )
    print(result)
    assert result

    print(
        f"Explanations:\n"
        f"  HTML: file://{evaluation.result.get_html_report_location()}\n"
    )


@pytest.mark.skip(reason="This is a documentation demo")
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_evaluator_demo(tmp_path):
    from h2o_sonar import config as h2o_sonar_config
    from h2o_sonar.evaluators import rag_hallucination_evaluator

    # LLM models to be evaluated

    model_host = h2o_sonar_config.ConnectionConfig(
        connection_type=h2o_sonar_config.ConnectionConfigType.H2O_GPT_E.name,
        name="H2O GPT Enterprise",
        description="H2O GPT Enterprise model host.",
        server_url="https://h2ogpte.h2o.ai/",
        token="YOUR_API_TOKEN_HERE",
        token_use_type=h2o_sonar_config.TokenUseType.API_KEY.name,
    )
    llm_models = genai.H2oGpteRagClient(model_host).list_llm_model_names()

    # evaluation dataset

    # test suite: RAG corpus, prompts, expected answers
    rag_test_suite = testing.RagTestSuiteConfig.load_from_json(
        test_utils.find_locally("data/generative/demo_doc_test_suite.json")
    )
    # test lab: resolved test suite w/ actual values from the LLM models host
    test_lab = testing.RagTestLab.from_rag_test_suite(
        rag_connection=model_host,
        rag_test_suite=rag_test_suite,
        rag_model_type=models.ExplainableModelType.h2ogpte,
        llm_model_names=llm_models,
        docs_cache_dir=tmp_path,
    )
    # deploy the test lab: upload corpus and create RAG collections/knowledge bases
    test_lab.build()
    # complete the test lab: actual values - answers, duration, cost, ...
    test_lab.complete_dataset()

    # EVALUATION

    evaluation = evaluate.run_evaluation(
        # test lab as the evaluation dataset (prompts, expected and actual answers)
        dataset=test_lab.dataset,
        # models to be evaluated ~ compared in the evaluation leaderboard
        models=test_lab.evaluated_models.values(),
        # evaluators
        evaluators=[
            rag_hallucination_evaluator.RagHallucinationEvaluator().evaluator_id()
        ],
        # where to save the report
        results_location=tmp_path,
    )

    # HTML report and the evaluation data (JSon, CSV, data frames, ...)

    print(f"HTML report: file://{evaluation.result.get_html_report_location()}")


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
