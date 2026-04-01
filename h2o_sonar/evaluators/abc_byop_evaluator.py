# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import abc
import collections
import re
import traceback

from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import datasets
from h2o_sonar.lib.api import evaluators
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats
from h2o_sonar.lib.api import judges
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import problems
from h2o_sonar.lib.api import results as r5s


class AbcByopEvaluator(abc.ABC, evaluators.Evaluator):
    """Abstract base class for Bring Your Own Prompt (BYOP) evaluators."""

    _display_name = "Bring Your Own Prompt (BYOP) Evaluator"
    _tagline = "BYOP Evaluator for LLM and RAG models."
    _brief_description = (
        "Bring Your Own Prompt (BYOP) Evaluator for LLM and RAG models."
    )
    _description = _brief_description

    _metrics_meta = e10s.LlmBoolLeaderboardExplanation.LEADERBOARD_METRICS_META

    # COMPATIBILITY: RAG evaluation
    _llm = True
    _rag = True

    # GLOBAL: leaderboard as global explanation
    _global_explanation = True

    # EXPLANATION TYPES created by the evaluator
    _explanation_types = [
        e10s.LlmEvalResultsExplanation,
        e10s.LlmBoolLeaderboardExplanation,
    ]

    _keywords = [
        evaluators.KEYWORD_LLM,
        evaluators.KEYWORD_EVALUATES_LLM,
        evaluators.KEYWORD_EVALUATES_RAG,
        evaluators.KEYWORD_RQ_J,
        evaluators.KEYWORD_METHOD_TYPE_NON_DETERMINISTIC,
        evaluators.KEYWORD_METHOD_JUDGE,
    ]

    PARAM_JUDGE_MODEL: str = "judge_model"
    PARAM_JUDGE_HOST: str = "judge_host"

    KEY_PROMPT: str = "prompt"
    KEY_ANSWER: str = "answer"
    KEY_PARSED_ANSWER: str = "parsed_answer"
    KEY_ERROR: str = "error"

    # BYOP Identifiers
    IDENTIFIER_ACTUAL_OUTPUT = "{ACTUAL_OUTPUT}"
    IDENTIFIER_EXPECTED_OUTPUT = "{EXPECTED_OUTPUT}"
    IDENTIFIER_INPUT = "{INPUT}"
    IDENTIFIER_CONTEXT = "{CONTEXT}"

    # EVALUATOR PARAMETERS
    _parameters = [
        evaluators.Evaluator._PARAM_SAVE_LLM_RESULT,
        evaluators.Evaluator._get_custom_param_metric_threshold(
            _metrics_meta.get_primary_metric()
        ),
        evaluators.Evaluator._PARAM_EVAL_JUDGE,
        evaluators.Evaluator._get_custom_param_min_test_case(),
    ]

    Classes = collections.namedtuple("Classes", ["failure", "success"])

    _CLASSES = Classes(failure="false", success="true")

    _PROBLEM_THRESHOLD_PROTO: problems.ProblemAndAction | None = None

    def __init__(self):
        evaluators.Evaluator.__init__(self)

        self.args = None
        self.problems = []
        self.log_name = "Bring Your Own Prompt (BYOP) evaluator"
        self.prompt_template: str | None = None
        self._classes_regex = re.compile(
            f"({'|'.join(sorted(self._CLASSES, key=len, reverse=True))})"
        )
        self._judge = None
        self._judge_cfg_key = ""
        self._judge_cfg = None

    def check_compatibility(
        self,
        params: commons.CommonInterpretationParams | None = None,
        **evaluator_params,
    ) -> bool:
        evaluators.Evaluator.check_compatibility(self, params, **evaluator_params)

        if not self.models:
            self.logger.warning(
                f"{self.log_name}: no RAG/LLM models found for evaluation: "
                f"{[m.key for m in self.models]} - NOT COMPATIBLE"
            )
            return False

        if not evaluators.Evaluator._check_llm_dataset_compatibility(
            self, params=params, evaluator_keywords=self._keywords
        ):
            return False

        return True

    def setup(self, model, persistence, **kwargs):
        evaluators.Evaluator.setup(self, model, persistence, **kwargs)

        self._resolve_evaluator_params()

        self.log_name = (
            f"Bring Your Own Prompt (BYOP) evaluator {self.mli_key}/{self.key}"
        )

        self.prompt_template = self._initialize_prompt_template()

        # prompt template sanity check
        if not (
            AbcByopEvaluator.IDENTIFIER_CONTEXT in self.prompt_template
            or AbcByopEvaluator.IDENTIFIER_INPUT in self.prompt_template
            or AbcByopEvaluator.IDENTIFIER_ACTUAL_OUTPUT in self.prompt_template
            or AbcByopEvaluator.IDENTIFIER_EXPECTED_OUTPUT in self.prompt_template
        ):
            raise ValueError(
                "Prompt template has to contain at least one of the "
                f"following keys: '{AbcByopEvaluator.IDENTIFIER_CONTEXT}', "
                f"'{AbcByopEvaluator.IDENTIFIER_INPUT}', "
                f"'{AbcByopEvaluator.IDENTIFIER_ACTUAL_OUTPUT}', "
                f"'{AbcByopEvaluator.IDENTIFIER_EXPECTED_OUTPUT}'"
            )

    @property
    def judge(self):
        if self._judge is None:
            self._judge_cfg_key = self._resolve_judge_key()
            if self._judge_cfg_key:
                # LLM judge materialization
                self._judge_cfg = self.config.get_evaluation_judge(
                    judge_key=self._judge_cfg_key
                )
                if not self._judge_cfg:
                    valid_judge_keys = [j.key for j in self.config.evaluation_judges]
                    raise ValueError(
                        f"Custom LLM judge for key: "
                        f"'{self._judge_cfg_key}' not found in H2O Sonar "
                        f"configuration. Valid keys are: {valid_judge_keys}"
                    )
                self._judge = judges.get_evaluation_judge_for_config(self._judge_cfg)
                # run custom judge health check
                try:
                    self._judge.health_check()
                except Exception as ex:
                    raise ValueError(
                        f"Custom LLM judge '{self._judge_cfg.name}' health check "
                        f"failed: {ex}\n{traceback}"
                    )

        # fallback to default judge
        if self._judge is None:
            # default judge is OpenAI GPT-4
            self._judge = judges.get_default_evaluation_judge(logger=self.logger)

        return self._judge

    @abc.abstractmethod
    def _initialize_prompt_template(self) -> str:
        pass

    def _prepare_prompts(
        self, rows: list[datasets.LlmDataset.LlmDatasetRow]
    ) -> list[str]:
        return [
            self.prompt_template.format(
                INPUT=row.i,
                EXPECTED_OUTPUT=row.expected_output,
                ACTUAL_OUTPUT=row.actual_output,
                CONTEXT="\n".join(row.context),
            )
            for row in rows
        ]

    def _eval_prompts(self, prompts: list[str]) -> list[dict[str, str]]:
        outputs = self.judge.evaluate(prompts)
        return [
            {self.KEY_ANSWER: outputs[i].answer, self.KEY_PROMPT: prompts[i]}
            for i in range(len(prompts))
        ]

    def _coerce_answer(self, prompt_and_answer: dict[str, str]) -> bool | None:
        answer = prompt_and_answer[self.KEY_ANSWER].lower()
        matched_classes = list(set(self._classes_regex.findall(answer)))

        # did we get the answer we wanted? and is it not ambiguous?
        if len(matched_classes) != 1:
            return None

        return self._CLASSES.success == matched_classes[0]

    def _parse_answers(self, prompts_and_answers: list[dict]) -> list[dict[str, str]]:
        return [
            dict(**pna, **{self.KEY_PARSED_ANSWER: self._coerce_answer(pna)})
            for pna in prompts_and_answers
        ]

    def evaluate(self, llm_testset, **kwargs) -> list:
        # RAG models: key -> model
        key_2_evaluated_model = {m.key: m for m in self.models}
        # LLM host: RAG or service
        llm_host = (
            commons.LlmModelHostType.RAG
            if isinstance(
                next(iter(key_2_evaluated_model.values())), models.ExplainableRagModel
            )
            else commons.LlmModelHostType.SERVICE
        )
        llm_host_str = "RAG" if llm_host == commons.LlmModelHostType.RAG else "LLM"
        self.report_progress(0.01, "Loading dataset...")
        llm_dataset = datasets.LlmDataset.from_datatable_dict(llm_testset.to_dict())
        self.report_progress(0.011, "Preparing prompts...")
        prompts = self._prepare_prompts(llm_dataset.inputs)
        self.report_progress(0.012, "Evaluating prompts...")
        prompts_and_answers = self._eval_prompts(prompts)
        self.report_progress(0.98, "Parsing answers...")
        results = self._parse_answers(prompts_and_answers)
        self.report_progress(0.99, "Preparing report...")
        eval_results = datasets.LlmEvalResults()

        #
        # EVALUATION
        #
        t_bool_leaderboard = e10s.LlmBoolLeaderboardExplanation
        for index, row in enumerate(llm_dataset.inputs):
            # handle actual answer retrieval error ~ RAG/LLM client crash
            if evaluators.Evaluator._is_internal_err_answer(row.actual_output):
                # IMPROVE: do NOT send row to LLM judge if AA is internal error
                row_2_add = datasets.LlmEvalResults.LlmEvalResultRow(
                    dataset_row=row,
                    metrics={
                        t_bool_leaderboard.KEY_RESULT_CHECK_OK: 0.0,
                        t_bool_leaderboard.KEY_RESULT_CHECK_FAIL: 1.0,
                        t_bool_leaderboard.KEY_RESULT_CHECK_FAIL_R: 0.0,
                        t_bool_leaderboard.KEY_RESULT_CHECK_FAIL_A: 1.0,
                        t_bool_leaderboard.KEY_RESULT_CHECK_FAIL_P: 0.0,
                        t_bool_leaderboard.KEY_RESULT_CHECK_ERR_MSG: (
                            evaluators.Evaluator._internal_err_answer_msg(
                                err_msg=row.actual_output
                            )
                        ),
                    },
                )
            else:
                parsed_ans = results[index][self.KEY_PARSED_ANSWER]
                answer = f"Judge's Answer: {results[index][self.KEY_ANSWER]}"
                # add to result
                ok = float(bool(parsed_ans))
                row_2_add = datasets.LlmEvalResults.LlmEvalResultRow(
                    dataset_row=row,
                    metrics={
                        t_bool_leaderboard.KEY_RESULT_CHECK_OK: ok,
                        t_bool_leaderboard.KEY_RESULT_CHECK_FAIL: 0.0 if ok else 1.0,
                        t_bool_leaderboard.KEY_RESULT_CHECK_FAIL_R: 0.0,
                        t_bool_leaderboard.KEY_RESULT_CHECK_FAIL_A: 0.0,
                        t_bool_leaderboard.KEY_RESULT_CHECK_FAIL_P: (
                            1.0 if parsed_ans is None else 0.0
                        ),
                        t_bool_leaderboard.KEY_RESULT_CHECK_ERR_MSG: answer,
                    },
                )

            # result row
            eval_results.add_result(row_2_add)

        #
        # NORMALIZATION of the evaluation RESULTS
        #

        # EXPLANATIONS
        explanations = []

        # EXPLANATION: all data (per prompt metrics)
        if self.args.get(
            evaluators.Evaluator.PARAM_SAVE_LLM_RESULT,
            evaluators.Evaluator.DEFAULT_SAVE_LLM_RESULT,
        ):
            eval_results_explanation = e10s.LlmEvalResultsExplanation(
                evaluator=self,
                display_name=self._display_name,
                display_category=e10s.Explanation.DISPLAY_CAT_LLM,
                eval_results=eval_results,
            )
            # FORMATS of the explanation: JSon, CSV, DataTable
            eval_results_explanation.add_json_format()
            eval_results_explanation.add_csv_format()
            eval_results_explanation.add_datatable_format()
            explanations.append(eval_results_explanation)

        # EXPLANATION: ok/fail leaderboard
        leaderboard_explanation = t_bool_leaderboard.from_eval_results(
            evaluator=self,
            eval_results=eval_results,
            metrics_meta=self._metrics_meta,
            metric_id_success=t_bool_leaderboard.KEY_RESULT_CHECK_OK,
            metric_id_failure_message=t_bool_leaderboard.KEY_RESULT_CHECK_ERR_MSG,
            key_2_evaluated_model=key_2_evaluated_model,
            display_name="RAG/LLM benchmark leaderboard",
            display_category=e10s.GlobalSummaryFeatImpExplanation.DISPLAY_CAT_LLM,
            llm_host=(
                commons.LlmModelHostType.RAG
                if isinstance(
                    next(iter(key_2_evaluated_model.values())),
                    models.ExplainableRagModel,
                )
                else commons.LlmModelHostType.SERVICE
            ),
            logger=self.logger,
        )

        # FORMAT of the explanation: Markdown
        leaderboard_explanation.add_markdown_format(title=f"{llm_host_str} Benchmarks")
        leaderboard_explanation.add_evalstudio_markdown_format(title="Summary")
        leaderboard_explanation.add_json_format(
            llm_host=llm_host,
            metrics_meta=self._metrics_meta,
            threshold=self.args.get(
                evaluators.Evaluator.PARAM_METRIC_THRESHOLD,
                leaderboard_explanation.METRIC_META_MODEL_PASSES.threshold,
            ),
        )
        explanations.append(leaderboard_explanation)

        # PROBLEMS for alerts and actionability
        self._diagnose_problems(
            eval_results=eval_results,
            key_2_evaluated_model=key_2_evaluated_model,
            leaderboard_explanation=leaderboard_explanation,
        )

        # INSIGHTS
        self._diagnose_insights(
            leaderboard_explanation=leaderboard_explanation,
        )

        # EXPLANATION: HTML fragment
        if self.config and self.config.create_html_representations:
            try:
                html_explanation = e10s.GlobalHtmlFragmentExplanation(
                    evaluator=self,
                    display_name=f"{llm_host_str} benchmark leaderboard as HTML",
                    display_category=e10s.Explanation.DISPLAY_CAT_COMPLIANCE,
                )
                html_explanation.add_html_format(
                    str(
                        leaderboard_explanation.as_html(
                            additional_details={
                                "Judge used by evaluator": (
                                    "Default OpenAI GPT LLM."
                                    if not self._judge_cfg_key
                                    else (
                                        f"Custom '{self._judge_cfg.name}' with "
                                        f" '{self._judge_cfg.llm_model_name}' LLM."
                                    )
                                ),
                                "Prompt template": (
                                    leaderboard_explanation.AdditionalDetails(
                                        formatting="pre",
                                        text=self.prompt_template,
                                    )
                                ),
                            },
                        )
                    )
                )
                explanations.append(html_explanation)
            except Exception as ex:
                self.logger.warning(
                    f"{self.log_name}: HTML fragment explanation creation failed: "
                    f"{ex}\n{traceback.format_exc()}",
                )

        # EXPLANATION: ZIP archive of all artifacts created by the evaluator
        explanations.append(
            self.create_explanation_workdir_archive(
                display_name=f"Archive of {self._display_name} artifacts",
                display_category=e10s.Explanation.DISPLAY_CAT_COMPLIANCE,
            )
        )

        return explanations

    def _diagnose_problems(
        self,
        eval_results: datasets.LlmEvalResults,
        key_2_evaluated_model: dict,
        leaderboard_explanation: e10s.LlmBoolLeaderboardExplanation,
    ):
        # perturbation problems
        self._diagnose_perturbation_problems(
            eval_results=eval_results,
            key_2_evaluated_model=key_2_evaluated_model,
            # row metrics are different from aggregated model metrics
            metrics_meta=e10s.LlmBoolLeaderboardExplanation.LEADERBOARD_METRICS_META,
        )

        # low test case count
        self._diagnose_low_test_case_problem(
            eval_results=eval_results,
            models=self.models,
            test_case_minimum=self.args.get(evaluators.Evaluator.PARAM_MIN_TEST_CASES),
        )

        # threshold failures
        problems.problems_for_bool_leaderboard(
            evaluator=self,
            leaderboard=leaderboard_explanation,
            metric_threshold=self.args.get(
                evaluators.Evaluator.PARAM_METRIC_THRESHOLD,
                leaderboard_explanation.METRIC_META_MODEL_PASSES.threshold,
            ),
            primary_metric_meta=self._metrics_meta.get_primary_metric(),
            severity=(
                self._PROBLEM_THRESHOLD_PROTO.severity
                if self._PROBLEM_THRESHOLD_PROTO
                else None
            ),
            problem_type=(
                self._PROBLEM_THRESHOLD_PROTO.problem_type
                if self._PROBLEM_THRESHOLD_PROTO
                else "accuracy"
            ),
            actions_description=(
                self._PROBLEM_THRESHOLD_PROTO.actions_description
                if self._PROBLEM_THRESHOLD_PROTO
                else ""
            ),
            explanation_type=e10s.GlobalHtmlFragmentExplanation.explanation_type(),
            explanation_name=e10s.GlobalHtmlFragmentExplanation.__name__,
            explanation_mime=formats.HtmlFormat.mime,
        )

    def _diagnose_insights(
        self, leaderboard_explanation: e10s.LlmBoolLeaderboardExplanation
    ):
        t_html_fragment = e10s.GlobalHtmlFragmentExplanation

        leaderboard_explanation.get_insights(
            explanation_type=t_html_fragment.explanation_type(),
            explanation_name=t_html_fragment.__name__,
            explanation_mime=formats.HtmlFormat.mime,
        )

    def get_result(self) -> r5s.LeaderboardResult:
        return r5s.LeaderboardResult(
            persistence=self.persistence,
            explainer_id=self.explainer_id(),
            h2o_sonar_config=self.config,
            logger=self.logger,
            explanation=e10s.LlmBoolLeaderboardExplanation,
            explanation_format=formats.CustomJsonFormat,
        )
