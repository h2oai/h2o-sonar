# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import uuid

import pytest

from h2o_sonar import evaluate
from h2o_sonar import loggers
from h2o_sonar.evaluators import rouge_evaluator
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import datasets
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import persistences
from h2o_sonar.utils import testing
from tests import test_utils
from tests.lib import given_generative


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
        file_path="data/generative/kaggle_llm_science_exam_test_lab_4x_25.json",
    )
    llm_models = rag_dataset.evaluated_models.values()

    #
    # WHEN
    #

    evaluation = evaluate.run_evaluation(
        dataset=rag_dataset.dataset,
        models=llm_models,
        evaluators=[rouge_evaluator.RougeEvaluator.evaluator_id()],
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


@pytest.mark.skipif(
    test_utils.GitHubActions.is_in_gha(),
    reason="Skipped on GHA as this test has high resource usage",
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_reproducibility_diverse_inputs(tmp_path):
    """Test the reproducibility of the ROUGE scores when run on the SAME data.

    The purpose of this test is to experiment with VARIOUS input types:

    - small/medium/large texts
    - character/sentence/paragraph level
    - similar/dissimilar texts

    ... and see if the ROUGE scores are reproducible. The idea is that shorter the
    compared texts are, the more likely they are to produce non-reproducible results
    in case that a sampling based approach is used to compute the ROUGE scores and/or
    the confidence interval estimates (central limit theorem).

    The test MUST be run multiple times and Python runtime must be restarted between
    the runs to ensure that they are not influenced:

    - use associated SH script to run it

    """
    #
    # GIVEN
    #
    logger = loggers.SonarPrintLogger()

    evaluation_key = "eeeeeeee-613b-4063-bac6-1f869d175e02"
    job_key = "3d1bca71-0000-0000-0000-1f869d175e02"
    model_key = "model-1"
    test_key = "test-1"

    # tuples of expected + actual answers
    answers = [
        # single character / similar
        ("A", "A"),
        ("A", "A."),
        # single character / dissimilar
        ("A", "D"),
        ("A", "Answer: D."),
        # single sentence / similar
        (
            "Disabling Bootstrapping (If possible/desired): If you only need the raw, "
            "deterministic score (not the confidence interval), you might be able to "
            "disable the bootstrapping mechanism.",
            "Bootstrapping Disabling (If possible/desired): zou If only the need raw, "
            "score deterministic (not confidence the interval), might zou be to able "
            "disable bootstrapping the mechanism.",
        ),
        # single sentence / dissimilar
        (
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod "
            "tempor incididunt ut labore et dolore magna aliqua.",
            "Lpo(r}eAm~ ipsum do-lor siOt amet, conYsecHtqe]tur aIdipisciKn+g el9it. "
            "SWedd wdo qeiusamo:d tjempaorY incidiFdFuntV u]t labo<re et zd!ojlWore "
            ")mRagna aliqWu/aL.",
        ),
        # multi sentence / similar
        (
            "Disabling Bootstrapping (If possible/desired): If you only need the raw, "
            "deterministic score (not the confidence interval), you might be able to "
            "disable the bootstrapping mechanism. In libraries that wrap rouge_score "
            "(like huggingface/evaluate), you can often pass a parameter like "
            "use_aggregator=False or similar to get the raw, deterministic per-example "
            "scores or a simple non-bootstrapped mean.",
            "Bootstrapping Disabling (If possible/desired): zou If only the need raw, "
            "score deterministic (not confidence the interval), might zou be to able "
            "disable bootstrapping the mechanism. libraries In that rouge_score wrap "
            "(like huggingface/evaluate), can you often a pass parameter "
            "use_aggregator=False like or to similar get the raw, per-example "
            "deterministic scores a or simple non-bootstrapped mean.",
        ),
        # medium sentence / dissimilar
        (
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod "
            "tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim "
            "veniam, "
            "quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo "
            "consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse "
            "cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat "
            "non proident, sunt in culpa qui officia deserunt mollit anim id est "
            "laborum.",
            "Lpo(r}eAm~ ipsum do-lor siOt amet, conYsecHtqe]tur aIdipisciKn+g el9it. "
            "SWedd wdo qeiusamo:d tjempaorY incidiFdFuntV u]t labo<re et zd!ojlWore "
            ")mRagna aliqWu/aL. Ut eni[m ad; mipnwim vAeniam$, q-uRi]s nmostUruYd "
            "gexlercitastion ^ullcaqmco lRaboriRs nisi ut Ialiquip eIx ea| commodos "
            "cionsequat. Duis aut%e irJure dolor in irebpr(e3heFn5deHrit in( "
            "voluoprtateX HvVelitK es*se lci(lluym kdolo|rwe_ Beu fug0iat nulla "
            "p(a-riiatur. ExceptHeur sXingt occaecat? oc(u,pVidatat no{n prgoid~eznta, "
            "sunt Rin /culdpa q'ui /off5icyiva d;es|er1un{t molwlit aani}m' id _est# "
            "ulaborum.",
        ),
    ]
    # long similar / dissimilar input is tested by test_evaluator_reproducibility.py

    llm_testset = datasets.LlmDataset()
    for i in range(len(answers)):
        print(f"Expected answer {i}:\n{answers[i][0]}")
        print(f"Actual answer {i}:\n{answers[i][1]}\n\n")
        llm_testset.inputs.append(
            datasets.LlmDataset.LlmDatasetRow(
                i=f"Input {i}",
                actual_output=answers[i][0],
                expected_output=answers[i][1],
                model_key=model_key,
                test_key=test_key,
                key=str(i),
            )
        )
    # DEBUG: print(f"LLM testset:\n{json.dumps(llm_testset.to_dict(),indent=2)}")
    dt_frame_llm_testset = llm_testset.to_datatable()

    e = rouge_evaluator.RougeEvaluator()
    e.setup(
        model=models.ExplainableModel(
            model_src="mock",
            predict_method=lambda x: x,
        ),
        persistence=persistences.ExplainerPersistence(
            data_dir=str(tmp_path),
            username=commons.DEFAULT_USER,
            explainer_id=e.explainer_id(),
            explainer_job_key=job_key,
            mli_key=evaluation_key,
        ),
    )
    e_model = models.ExplainableLlmModel(
        connection=test_utils.health.get_h2ogpt(),
        model_type=models.ExplainableModelType.h2ogpte_llm,
        name="Mock",
        llm_model_name="mock/llm",
        logger=logger,
        key=str(uuid.uuid4()),
    )
    e.models = [e_model]

    #
    # WHEN
    #

    e.evaluate(llm_testset=dt_frame_llm_testset)

    #
    # THEN
    #

    ep = persistences.ExplainerPersistence(
        data_dir=str(tmp_path),
        mli_key=evaluation_key,
        username=commons.DEFAULT_USER,
        explainer_id=e.explainer_id(),
        explainer_job_key=job_key,
    )
    json_leaderboard_path = ep.get_explanation_file_path(
        explanation_type=e10s.LlmHeatmapLeaderboardExplanation.explanation_type(),
        explanation_format=f5s.LlmLeaderboardJSonFormat.mime,
    )
    json_eval_result_path = ep.get_explanation_file_path(
        explanation_type=e10s.LlmEvalResultsExplanation.explanation_type(),
        explanation_format=f5s.CustomJsonFormat.mime,
    )

    with open(json_leaderboard_path) as f:
        json_leaderboard = f.read()
    print(f"Leaderboard:\n{json_leaderboard}")
    assert json_leaderboard
    json_leaderboard_data_path = json_leaderboard_path.replace(
        "explanation.json", "leaderboard_3.json"
    )
    print(f"Leaderboard data path: {json_leaderboard_data_path}")
    with open(json_leaderboard_data_path) as f:
        json_leaderboard_data = f.read()
    print(f"Leaderboard data:\n{json_leaderboard_data}")
    assert json_leaderboard_data

    # reproducibility check @ average ROUGE scores
    with open(json_leaderboard_data_path) as f:
        jlbd = json.load(f)
    rouge_scores = jlbd[f5s.ExplanationFormat.KEY_DATA]["model"]
    # Why is ROUGE-2 so low in contrast to ROUGE-1 - even 10x?
    # - EXPECTED and common - uni vs. bi-gram.
    # Why is ROUGE-L is not as low as ROUGE-2?
    # - EXPECTED ROUGE-L is longest common subsequence of NON consecutive words.
    assert rouge_scores[e.METRIC_ROUGE_1] == 0.5370597078226901
    assert rouge_scores[e.METRIC_ROUGE_2] == 0.053825757575757575
    assert rouge_scores[e.METRIC_ROUGE_L] == 0.472819690065015

    print("\nREPRODUCIBILITY test summary:")
    print(f"- test cases : {len(answers)}")
    print(f"- shortest EA: {min([len(a[1]) for a in answers]):,}")
    print(f"- longest EA : {max([len(a[1]) for a in answers]):,}")
    print(f"- shortest AA: {min([len(a[0]) for a in answers]):,}")
    print(f"- longest AA : {max([len(a[0]) for a in answers]):,}")

    print(f"Evaluation results path: file://{json_eval_result_path}")


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
