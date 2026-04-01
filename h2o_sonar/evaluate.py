# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import logging
import pathlib
import time
from typing import Any

from h2o_sonar import config
from h2o_sonar import interpret
from h2o_sonar import loggers
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import datasets
from h2o_sonar.lib.api import evaluations
from h2o_sonar.lib.api import evaluators as ev7s
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import interpretations
from h2o_sonar.lib.api import models as m4s
from h2o_sonar.lib.api import persistences as persist
from h2o_sonar.lib.container import explainer_container
from h2o_sonar.utils import caching
from h2o_sonar.utils import perturbations
from h2o_sonar.utils import progress
from h2o_sonar.utils import testing


"""H2O Sonar public API for LLM/RAG evaluations."""

KEYWORD_LLM = ev7s.KEYWORD_LLM
KEYWORD_FILTER_ALL = commons.KEYWORD_FILTER_ALL
KEYWORD_FILTER_ALL_ASTERISK = commons.KEYWORD_FILTER_ALL_ASTERISK

#
# EVALUATORS
#


def list_evaluators(
    keywords: list[str] | None = None,
    evaluator_filter: list[commons.FilterEntry] | None = None,
    args_as_json_location: pathlib.Path | str | None = None,
    portable: bool = False,
    extra_params: dict | None = None,
) -> list[ev7s.EvaluatorDescriptor]:
    """List evaluators by supported experiment types, scores, keywords and other
    criteria. Parameters are used with ``AND`` logical operator.

    Parameters
    ----------
    keywords : list[str] | None
      Filter evaluators by keywords.
    evaluator_filter : list[commons.FilterEntry] | None
      Filter evaluators by generic filter (forward compatible filtering). See
      ``FilterEntry`` for supported search types.
    args_as_json_location : pathlib.Path | str | None
      Load all positional arguments and keyword arguments from JSon file.
      This is useful when input is generated, persisted, repeated and used from CLI
      (which doesn't support all the options).
      IMPORTANT: if this argument is specified, then all other function parameters
      are ignored.
    portable : bool
      If ``True``, then floats (infinity, NaN) and tuples are converted to be
      portable - from strings to max/min values of respective types.
    extra_params : dict | None
      Extra parameters.

    Returns
    -------
    list[evaluators.EvaluatorDescriptor] :
      Evaluators compliant with provided filters.

    """
    es = interpret.list_explainers(
        keywords=keywords or [ev7s.KEYWORD_LLM],
        explainer_filter=evaluator_filter,
        args_as_json_location=args_as_json_location,
        portable=portable,
        extra_params=extra_params,
    )

    results = []
    for e in es:
        results.append(
            ev7s.EvaluatorDescriptor(
                id=e.id,
                name=e.name,
                display_name=e.display_name,
                tagline=e.tagline,
                description=e.description,
                brief_description=e.brief_description,
                model_types=e.model_types,
                can_explain=e.can_explain,
                explanation_scopes=e.explanation_scopes,
                explanations=e.explanations,
                parameters=e.parameters,
                keywords=e.keywords,
                metrics_meta=e.metrics_meta,
            )
        )
    return results


def describe_evaluator(
    evaluator: str | Any,
    portable: bool = False,
) -> dict:
    """Get evaluator description.

    Parameters
    ----------
    evaluator : str | Any
       Evaluator to describe.
    portable : bool
        If ``True``, then floats (infinity, NaN) and tuples are converted
        to be portable - from strings to max/min values of respective types.

    Returns
    -------
    dict :
     Dictionary with evaluator name and parameters.

    """
    return interpret.describe_explainer(explainer=evaluator, portable=portable)


#
# EVALUATIONS
#


def run_evaluation(
    dataset: datasets.LlmDataset | Any,
    models: (
        list[
            m4s.ExplainableLlmModel | m4s.ExplainableRagModel | m4s.OpenAiRagModel | Any
        ]
        | None
    ) = None,
    evaluators: list[str | commons.EvaluatorToRun] | None = None,
    evaluator_keywords: list[str] | None = None,
    results_location: pathlib.Path | str | dict | Any | None = None,
    persistence_type: persist.PersistenceType = persist.PersistenceType.file_system,
    run_asynchronously: bool = False,
    progress_callback: progress.AbstractProgressCallbackContext | None = None,
    logger: loggers.SonarLogger | None = None,
    log_level: int = logging.WARNING,
    args_as_json_location: pathlib.Path | str | None = None,
    upload_to: str | config.ConnectionConfig | None = None,
    key: str = "",
    extra_params: dict | None = None,
) -> evaluations.Evaluation:
    """Run interpretation.

    Parameters
    ----------
    dataset : datasets.LlmDataset | Any
      Dataset source.
    models : list[m4s.ExplainableLlmModel | m4s.ExplainableRagModel
    | m4s.OpenAiRagModel | Any] | None
      Models to be evaluated.
      LLM/RAG models.
    evaluators : list[str | commons.EvaluatorToRun] | None
      Evaluator IDs to run within the evaluation or ``EvaluatorToRun`` instances
      with evaluator parameters. In case of ``None`` or empty list are run all
      default compatible evaluators.
    evaluator_keywords: list[str] | None
      Run compatible evaluators which have given keyword (AND). This setting is used
      *only* in case that ``evaluators`` parameter is empty list (or ``None``).
    results_location : pathlib.Path | str | dict | Any | None
      Where to store evaluation results - filesystem (path as string or ``Path``),
      memory (dictionary) or DB. If ``None``, then results are stored to the current
      directory.
    persistence_type : persist.PersistenceType
      Optional choice of the persistence type: file-system (default), in-memory
      or database. This option does not override persistence type in case that
      container is provided.
    run_asynchronously : bool
      ``True`` to run the evaluation asynchronously - the evaluation is run
      synchronously by default (``False``). Use evaluation's JSon report to determine
      progress, success/failure and results.
    progress_callback : progress.AbstractProgressCallbackContext | None
        Optional progress callback context.
    logger : loggers.SonarLogger | None
        Optional custom logger which implements ``loggers.SonarLogger`` interface.
        If ``logger`` is provided, then ``log_level`` is ignored.
    log_level : int
      Optional container and explainers log level.
      If ``logger`` is provided, then ``log_level`` is ignored.
    args_as_json_location : pathlib.Path | str | None
      Load all positional arguments and keyword arguments from JSon file.
      This is useful when input is generated, persisted, repeated and used from CLI
      (which doesn't support all the options).
      IMPORTANT: if this argument is specified, then all other function parameters
      are ignored.
    upload_to : str | config.ConnectionConfig | None
      Upload the interpretation report to the H2O GPT Enterprise in order to talk to
      the report.
    key : str
      Custom evaluation key which must be valid UUID4 string.
    extra_params : dict | None
      Extra parameters.

    Returns
    -------
    evaluations.Evaluation :
      Evaluation instance with the evaluator results (references).

    """
    i = interpret.run_interpretation(
        dataset=dataset,
        models=models,
        explainers=evaluators,
        explainer_keywords=evaluator_keywords,
        results_location=results_location,
        persistence_type=persistence_type,
        progress_callback=progress_callback,
        logger=logger,
        log_level=log_level,
        args_as_json_location=args_as_json_location,
        upload_to=upload_to,
        run_asynchronously=run_asynchronously,
        key=key,
        extra_params={"is_evaluation": True, **(extra_params or {})},
    )

    return evaluations.Evaluation.from_interpretation(i) if i else i


def wait_for_evaluation(
    evaluation_key: str,
    results_location: pathlib.Path | str | dict | Any | None = None,
    wait_steps: int = 10,
    wait_step_seconds: float = 1,
    logger: loggers.SonarLogger | None = None,
):
    """Actively wait for asynchronously executed evaluation to finish.

    Parameters
    ----------
    evaluation_key : str
      Evaluation key.
    results_location : pathlib.Path | str | dict | Any | None
      Where to store evaluation results.,
    wait_steps : int
      Number of steps to wait.
    wait_step_seconds : float
      Number of seconds to wait between steps.
    logger : loggers.SonarLogger | None
        Optional custom logger which implements ``loggers.SonarLogger`` interface
        to optionally log the progress.

    Returns
    -------
    evaluations.Evaluation :
      Evaluation instance (partially instantiated) with the evaluators results
      (references).

    """
    evaluation = None
    for _ in range(wait_steps):
        evaluation = get_evaluation(
            evaluation_key=evaluation_key,
            results_location=results_location,
        )
        if evaluation:
            if logger:
                logger.info(
                    f"WAITING for evaluation {evaluation.key}:"
                    f"\n  progress            : "
                    f"{evaluation.progress}"
                    f"\n  status              : "
                    f"{evaluation.status}"
                    f"\n  progress message    : "
                    f"'{evaluation.progress_message}'"
                    f"\n  scheduled explainers: "
                    f"'{evaluation.get_scheduled_explainer_ids()}'"
                    f"\n  finished explainers : "
                    f"'{evaluation.get_finished_explainer_ids()}'"
                    f"\n  failed explainers   : "
                    f"'{evaluation.get_failed_evaluator_ids()}'"
                    f"\n  progress callback   : "
                    f"'{evaluation.progress_callback}'"
                )
            if commons.ExplainerJobStatus.is_job_finished(evaluation.status):
                return evaluation
        time.sleep(wait_step_seconds)

    raise RuntimeError(
        f"Evaluation timeout {wait_steps} * {wait_step_seconds}s exceeded - "
        f"evaluation did not finish. Status: "
        f"{'UNKNOWN' if not evaluation else evaluation.status}"
    )


def get_evaluation(
    evaluation_key: str,
    results_location: pathlib.Path | str | dict | Any | None = None,
) -> evaluations.Evaluation | None:
    """Get evaluation by key.

    Parameters
    ----------
    evaluation_key : str
      Evaluation key.
    results_location : pathlib.Path | str | dict | Any | None
      Where the evaluation results are stored.

    Returns
    -------
    evaluations.Evaluation | None :
      PARTIALLY initialized ``Evaluation`` instance with status, progress, evaluators
      results (if finished, failed or running) or ``None`` if the evaluation not
      started/found.

    """
    i_persistence = persist.InterpretationPersistence(
        data_dir=results_location,
        username=commons.DEFAULT_USER,
        mli_key=evaluation_key,
        logger=loggers.SonarPrintLogger(),
    )
    i_json_path = i_persistence.get_json_path()
    if not pathlib.Path(i_json_path).exists():
        return None

    # load interpretation JSon > Interpretation > Evaluation
    i = interpretations.Interpretation.load_from_json(i_json_path)
    e = evaluations.Evaluation.from_interpretation(i)

    return e


def _compare_evaluation_load_explanation_from_evaluation(
    evaluation: evaluations.Evaluation | str,
    evaluator_ids: list[str],
    results_location: pathlib.Path | str | dict | Any | None,
) -> dict[str, e10s.LlmEvalResultsExplanation]:
    """Load LlmEvalResultsExplanation from evaluation or evaluation key:

    - Load the first evaluator's evaluation results explanation in JSON format.

    Parameters
    ----------
    evaluation : evaluations.Evaluation | str
        Evaluation or evaluation key.
    evaluator_ids | list[str]
        Evaluators to compare.
    results_location : pathlib.Path | str | dict | Any | None
        Where the evaluation results are stored.

    Returns
    -------
    dict[str, e10s.LlmEvalResultsExplanation] :
        Map of explainer ID to explanations created by the evaluation.

    """
    load_from_file = False

    if isinstance(evaluation, str):
        evaluation = get_evaluation(
            evaluation_key=evaluation,
            results_location=results_location,
        )
        if not evaluation:
            raise ValueError(
                f"Evaluation with key '{evaluation}' not found at "
                f"location '{results_location}'"
            )
        load_from_file = True

    # evaluators IDs
    successful_evaluator_ids = evaluation.get_successful_evaluator_ids()
    if not successful_evaluator_ids:
        successful_evaluator_ids = evaluation.get_finished_evaluator_ids()
        if not successful_evaluator_ids:
            raise ValueError(
                f"No successfully finished evaluators in the evaluation: "
                f"'{evaluation.key}'"
            )

    # find the available evaluators
    target_evaluator_ids = []
    if evaluator_ids:
        # filter only required evaluator IDs
        for evaluator_id in evaluator_ids:
            if evaluator_id in successful_evaluator_ids:
                target_evaluator_ids.append(evaluator_id)
    else:
        target_evaluator_ids = successful_evaluator_ids
    if not target_evaluator_ids:
        raise ValueError(
            f"No requested evaluators found among successful evaluators: "
            f"{successful_evaluator_ids}"
        )

    result: dict[str, e10s.LlmEvalResultsExplanation] = {}
    for evaluator_id in target_evaluator_ids:
        explanation = None
        evaluator_result = evaluation.get_evaluator_result(evaluator_id)

        # try to get explanation from memory first
        if not load_from_file:
            if evaluator_result and evaluator_result.explanation:
                if isinstance(
                    evaluator_result.explanation, e10s.LlmEvalResultsExplanation
                ):
                    explanation = evaluator_result.explanation

        # if not in memory, try to load from file
        if explanation is None:
            # construct the pattern to search for explanation files
            # the explainer job key UUID may vary, so we use a glob pattern
            base_path = pathlib.Path(evaluation.result.results_location)
            mli_dir = base_path / "h2o-sonar" / f"mli_experiment_{evaluation.key}"

            # find explainer directory matching the evaluator_id
            # convert dots to underscores for filesystem path
            evaluator_id_path = evaluator_id.replace(".", "_")
            explainer_pattern = f"explainer_{evaluator_id_path}_*"
            explainer_dirs = list(mli_dir.glob(explainer_pattern))

            explanation_path = None
            if explainer_dirs:
                # use the first matching directory
                explainer_dir = explainer_dirs[0]
                explanation_path = (
                    explainer_dir
                    / "global_llm_eval_results"
                    / "application_json"
                    / "explanation.json"
                )

            if not explanation_path or not explanation_path.exists():
                raise FileNotFoundError(
                    f"Explanation file not found for evaluator '{evaluator_id}' "
                    f"in evaluation '{evaluation.key}' at location "
                    f"'{evaluation.result.results_location}'"
                )

            with open(explanation_path) as f:
                explanation_dict = json.load(f)

            # explainers map source
            container = explainer_container.LocalExplainerContainer()
            container.setup()

            explanation = e10s.LlmEvalResultsExplanation.from_dict(
                explainers_map=container.explainers_registry.list_explainers(),
                explanation_dict=explanation_dict,
            )

        if explanation:
            result[evaluator_id] = explanation

    return result


def compare_evaluations(
    baseline_evaluation: evaluations.Evaluation | str,
    current_evaluation: evaluations.Evaluation | str,
    evaluators: list[str | commons.EvaluatorToRun] | None = None,
    baseline_llm_model: str = "",
    current_llm_model: str = "",
    results_location: pathlib.Path | str | dict | Any | None = None,
    comparison_method: e10s.SentenceComparisonMethod = (
        e10s.SentenceComparisonMethod.COSINE_DISTANCE
    ),
    sentence_similarity_threshold: float = 0.9,
) -> e10s.EvalResultsDiff:
    """Compare evaluations.

    Parameters
    ----------
    baseline_evaluation : evaluations.Evaluation | str
        Baseline evaluation or evaluation key to compare.
    current_evaluation : evaluations.Evaluation | str
        Current evaluation or evaluation key to compare.
    evaluators : list[str | commons.EvaluatorToRun] | None = None
        List of evaluators to compare. ``None`` or empty list to compare all evaluators.
    baseline_llm_model: str
        The LLM model name used for filtering and equals of explainable models. Either
        both baseline and current LLM model names must be specified or none. The use
        case is comparison of explainable models on different host types (like
        h2oGPTe vs.AWS Bedrock) where LLM model names are different
        (names fixed by provider), collection IDs are (very) different, connections
        are different, ... but user know what they want to compare
    current_llm_model: str
        The LLM model name used for filtering and equals of explainable models.
    results_location : pathlib.Path | str | dict | Any | None
        Where the evaluation results are stored. If not specified, then the current
        working directory is used.
    comparison_method : SentenceComparisonMethod
        The method to use for comparing sentences:
        - EXACT_MATCH: exact string matching
        - COSINE_DISTANCE: cosine distance of sentence embeddings (default)
        - BERT_SCORE: BERTScore contextual embeddings similarity
    sentence_similarity_threshold : float
        Threshold for determining if sentences are "common" (high similarity).
        Sentences with similarity >= threshold are considered common.
        Default is 0.9.

    Returns
    -------
    e10s.EvalResultsDiff :
        Difference between ``baseline_evaluation`` and ``current_evaluation``.

    """
    evaluator_ids = []
    if evaluators:
        for e in evaluators:
            if isinstance(e, commons.EvaluatorToRun):
                evaluator_ids.append(e.id)
            elif isinstance(e, str):
                evaluator_ids.append(e)
            else:
                raise ValueError(f"Evaluator '{e}' is not a recognized evaluator type.")

    baseline_explanations = _compare_evaluation_load_explanation_from_evaluation(
        evaluation=baseline_evaluation,
        evaluator_ids=evaluator_ids,
        results_location=results_location,
    )
    current_explanations = _compare_evaluation_load_explanation_from_evaluation(
        evaluation=current_evaluation,
        evaluator_ids=evaluator_ids,
        results_location=results_location,
    )

    e_intersection = list(baseline_explanations.keys() & current_explanations.keys())
    if not e_intersection:
        raise ValueError(
            f"Evaluations cannot be compared as sets of their explainers have empty "
            f"intersection: {baseline_explanations.keys()} vs "
            f"{current_explanations.keys()}."
        )

    # merge metrics of all explanations into 1 for given evaluation
    if len(baseline_explanations) > 1:
        # get the first baseline explanation, merge others into it
        baseline_explanations_list = list(baseline_explanations.values())
        baseline_explanation = baseline_explanations_list[0].merge_metrics(
            explanations=baseline_explanations_list[1:],
            evaluator_ids=e_intersection,
        )
    else:
        baseline_explanation = list(baseline_explanations.values())[0]

    # merge current explanations the same way as baseline
    if len(current_explanations) > 1:
        current_explanations_list = list(current_explanations.values())
        current_explanation = current_explanations_list[0].merge_metrics(
            explanations=current_explanations_list[1:],
            evaluator_ids=e_intersection,
        )
    else:
        current_explanation = list(current_explanations.values())[0]

    return baseline_explanation.compare(
        other=current_explanation,
        baseline_llm_model=baseline_llm_model,
        current_llm_model=current_llm_model,
        comparison_method=comparison_method,
        sentence_similarity_threshold=sentence_similarity_threshold,
    )


#
# PERTURBATIONS
#


def list_perturbators(
    keywords: list[str] | None = None,
) -> list[perturbations.PerturbatorDescriptor]:
    """List perturbators and optionally filter them by keywords. If no keywords are
    provided, then all perturbators are listed. If more than one keyword is provided,
    then the perturbators must have all the keywords to be listed.

    """

    return [
        p.as_descriptor()
        for p in perturbations.PerturbatorRegistry.registry().list_perturbators(
            keywords=keywords
        )
    ]


def describe_perturbator(
    perturbator_id: str,
) -> perturbations.PerturbatorDescriptor | None:
    return perturbations.PerturbatorRegistry.registry().describe_perturbator(
        perturbator_id
    )


def perturb(
    content: (
        str
        | testing.RagTestCaseConfig
        | testing.RagTestSuiteConfig
        | datasets.LlmDataset
    ),
    perturbators: list[str | commons.PerturbatorToRun],
    in_place: bool = True,
    raised_errors: list | None = None,
) -> str | testing.RagTestCaseConfig | testing.RagTestSuiteConfig | datasets.LlmDataset:
    """Perturb the content using the perturbator.

    Parameters
    ----------
    content : str |  testing.RagTestCaseConfig | testing.RagTestSuiteConfig
    | datasets.LlmDataset
        Content to be perturbed.
    perturbators : list[str | commons.PerturbatorToRun]
        Perturbators to be used for perturbation.
    in_place : bool
        If ``True``, then the content is perturbed in place (given content is
        modified), otherwise the perturbator(s) output is added to
        given content.
        In case of `string`, the perturbed string is returned - always not in place
        as the string is immutable and passed by value.
        In case of ``RagTestCaseConfig`` and in place perturbation, the test case's
        prompt is perturbed in place, else whole test case is cloned, perturbed and
        returned.
        In case of ``RagTestSuiteConfig`` and ``LlmDataset``, the content (test
        cases) are perturbed either in place or new copy is created. Which means
        that in place perturbation keeps the number of test cases / rows the same,
        while non-in-place perturbation creates new test cases / rows i.e. the
        result will have 2x number of test cases / rows.
    raised_errors : list | None
        If ``None``, then raise error(s) if the perturbator(s) fail(s,
        otherwise do not raise exceptions and store them in the (empty) list provided
        by the caller.

    Returns
    -------
    str | testing.RagTestCaseConfig | testing.RagTestSuiteConfig | datasets.LlmDataset :
        Perturbed content.

    """
    if not perturbators:
        raise ValueError("No perturbators provided.")

    # check perturbation IDs validity
    registry = perturbations.PerturbatorRegistry.registry()
    perturbators_2_run = []
    for p in perturbators:
        if isinstance(p, str):
            p = commons.PerturbatorToRun(perturbator_id=p)
            perturbators_2_run.append(p)
        elif isinstance(p, commons.PerturbatorToRun):
            p.params = p.params or {}
            perturbators_2_run.append(p)
        else:
            raise ValueError(f"Unsupported perturbator to run type: {type(p)}")

        if not p.perturbator_id:
            raise ValueError(f"Perturbator '{p.perturbator_id}' not found.")

    # perturb PRIMITIVE ~ IMMUTABLE content
    if isinstance(content, str):
        perturbed_content = None
        for p2r in perturbators_2_run:
            perturbator = registry.get_perturbator(p2r.perturbator_id)

            perturbed_content = perturbator.perturb(
                text=content,
                intensity=p2r.intensity,
                raised_errors=raised_errors,
                **p2r.params,
            )

    # perturb COMPLEX content
    elif isinstance(content, testing.RagTestCaseConfig):
        perturbed_content = content.perturb(
            perturbators=perturbators_2_run,
            in_place=in_place,
            raised_errors=raised_errors,
        )

    elif isinstance(content, testing.RagTestSuiteConfig):
        perturbed_content = content.perturb(
            perturbators=perturbators_2_run,
            in_place=in_place,
            raised_errors=raised_errors,
        )

    elif isinstance(content, datasets.LlmDataset):
        perturbed_content = content.perturb(
            perturbators=perturbators_2_run,
            in_place=in_place,
            raised_errors=raised_errors,
        )

    else:
        raise ValueError(f"Unsupported content type to be perturbed: {type(content)}")

    return perturbed_content


def cache_models(logger: loggers.SonarLogger | None = None) -> None:
    """Download and cache all the third-party models required for evaluations.

    Parameters
    ----------
    logger : logging.Logger | None
        Logger to use.

    """
    logger = logger or loggers.SonarPrintLogger()
    caching.cache_all_models(logger)
