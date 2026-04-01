# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import re
import traceback

import airium
import datatable

from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import datasets
from h2o_sonar.lib.api import evaluators
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import problems
from h2o_sonar.lib.api import results
from h2o_sonar.utils import parsing
from h2o_sonar.utils import progress as progress_utils
from h2o_sonar.utils import tokenization


class RagStrStrEvaluator(evaluators.Evaluator):
    _display_name = "Text matching"
    _tagline = (
        "Evaluate the presence of specific strings in the answers and retrieved "
        "contexts."
    )

    # COMPATIBILITY: LLM/RAG evaluation
    _rag = True
    _llm = True  # evaluator does NOT require context

    # GLOBAL EXPLANATIONS support
    _global_explanation = True

    # EXPLANATION TYPES created by the evaluator
    _explanation_types = [
        e10s.LlmEvalResultsExplanation,
        e10s.LlmBoolLeaderboardExplanation,
        e10s.WorkDirArchiveExplanation,
    ]

    # KEYWORDS to find this evaluator when listing / filtering evaluators
    _keywords = [
        evaluators.KEYWORD_LLM,
        evaluators.KEYWORD_EVALUATES_RAG,
        evaluators.KEYWORD_EVALUATES_LLM,
        evaluators.KEYWORD_RQ_P,
        evaluators.KEYWORD_RQ_C,
        evaluators.KEYWORD_SR_11_7_CS,
        evaluators.KEYWORD_SR_11_7_OGA,
        evaluators.KEYWORD_SR_11_7_OA,
        evaluators.KEYWORD_NIST_AI_RMF_S,
        evaluators.KEYWORD_NIST_AI_RMF_SR,
        evaluators.KEYWORD_NIST_AI_RMF_PE,
        evaluators.KEYWORD_NIST_AI_RMF_F,
        evaluators.KEYWORD_NIST_AI_RMF_AT,
        evaluators.KEYWORD_NIST_AI_RMF_VR,
        evaluators.KEYWORD_PROBLEM_TYPE_IR,
        evaluators.KEYWORD_PROBLEM_TYPE_QA,
        evaluators.KEYWORD_PROBLEM_TYPE_SUM,
        evaluators.KEYWORD_PROBLEM_TYPE_REG,
        evaluators.KEYWORD_PROBLEM_TYPE_CLS,
        evaluators.KEYWORD_PROBLEM_TYPE_BIN,
        evaluators.KEYWORD_PROBLEM_TYPE_MUL,
        evaluators.KEYWORD_ES_GENERATE,  # required for H2O Eval Studio eval eye
        evaluators.KEYWORD_METHOD_RULE_BASED,
        evaluators.KEYWORD_METHOD_TYPE_DETERMINISTIC,
        evaluators.KEYWORD_CAP_CH,
    ]

    _metrics_meta = e10s.LlmBoolLeaderboardExplanation.LEADERBOARD_METRICS_META

    # EVALUATOR PARAMETERS
    PARAM_EVAL_RC = "evaluate_retrieved_context"
    DEFAULT_EVAL_RC = False
    _parameters = [
        evaluators.Evaluator._get_custom_param_metric_threshold(
            _metrics_meta.get_primary_metric()
        ),
        evaluators.Evaluator._PARAM_SAVE_LLM_RESULT,
        evaluators.EvaluatorParam(
            param_name=PARAM_EVAL_RC,
            description=(
                "Control whether to evaluate also retrieved context - conditions "
                "to check whether it contains or does not contained specific strings."
            ),
            param_type=commons.EvaluatorParamType.bool,
            default_value=DEFAULT_EVAL_RC,
            src=evaluators.EvaluatorParam.SRC_EVALUATOR_PARAMS,
        ),
    ]

    _brief_description = """Text Matching Evaluator assesses whether both
the retrieved context (in the case of RAG hosted models) and the generated answer
**contain/match** a specified set of required strings. The evaluation is based on an
boolean expression (condition) that can be used to define the required strings presence:

- operands are **strings** or **regular expressions**
- operators are `AND`, `OR`, and `NOT`
- **parentheses** can be used to group expressions"""
    _description = evaluators.Evaluator._description_builder(
        brief=f"""{_brief_description}

- **Example 1: Simple string matching**
   - Expression: `"15,969"`
   - The evaluator will check if the retrieved context and the actual answer
     contain the string `15,969`. If the condition is satisfied, the test case
     passes.

- **Example 2: Flexible regex patterns**
   - Expression: `regexp("15,?969")`
   - What if the number `15,969` might be expressed as `15969` or `15,969`?
     The boolean expression can be extended to use a regular expression. The
     evaluator will check if the retrieved context and the actual answer contain
     the string `15,969` or `15969`. If the condition is satisfied, the test
     case passes.

- **Example 3: Combining string and regex**
   - Expression: `"15,969" AND regexp("[Mm]illion")`
   - The evaluator will check if the retrieved context and the actual answer
     contain the string `15,969` **and** match the regular expression
     `[Mm]illion`. If the condition is satisfied, the test case passes.

- **Example 4: Complex boolean logic**
   - Expression: `("Rio" OR "rio") AND regexp("15,?969 [Mm]il") AND NOT "Real"`
   - The evaluator will check if the retrieved context and the actual answer
     contain either `Rio` or `rio` **and** match the regular expression
     `15,969 [Mm]il` **and** do not contain the string `Real`. If the
     condition is satisfied, the test case passes.

- **Example 5: Exact matching with regex anchors**
   - Expression: `regexp("^Brazil revenue was 15,969 million$")`
   - The evaluator will check if the retrieved context and the actual answer
     **exactly** match the regular expression
     `^Brazil revenue was 15,969 million$`. If the condition is satisfied, the
     test case passes.

- **Example 6: Case-insensitive matching**
   - Expression: `regexp("(?i)python")`
   - The `(?i)` flag enables case-insensitive matching. The evaluator will match
     `python`, `Python`, `PYTHON`, `PyThOn`, etc. This is useful when the
     capitalization in the output is unpredictable.

- **Example 7: OR within regular expressions**
   - Expression: `regexp("(cat|dog|bird)")`
   - Using the pipe `|` operator inside a group allows matching multiple
     alternatives. The evaluator will match any of: `cat`, `dog`, or `bird`.
     This is more concise than using multiple `OR` operators in the boolean
     expression.

- **Example 8: Capturing groups and word boundaries**
   - Expression: `regexp("\\b(error|warning|failure)\\b")`
   - The `\\b` word boundary ensures exact word matching (not as part of a larger
     word). The regex will match `error`, `warning`, or `failure` as complete
     words. Parentheses capture the matched text for reference.

- **Example 9: Repeated patterns and quantifiers**
   - Expression: `regexp("\\d{3}-\\d{3}-\\d{4}")`
   - Quantifiers specify repetition: `\\d{3}` matches exactly 3 digits, `+`
     matches one or more, `*` matches zero or more. This example matches phone
     numbers in the format `123-456-7890`. Use `\\d` for digits, `\\w` for
     word characters, `\\s` for whitespace.

- **Example 10: Lookahead and combining patterns**
   - Expression: `regexp("(?i)(success|completed).*\\d+%")`
   - This combines case-insensitive matching `(?i)`, an OR group
     `(success|completed)`, `.*` to match any characters, and `\\d+%` to
     match one or more digits followed by a percent sign. Useful for matching
     complex patterns like progress messages.

**Method**:

- The evaluator parses the boolean expression and checks if the retrieved context
  and the generated answer contain the required strings.
- The evaluator uses Python `re` module for regular expression matching (`re.search`
  function). See https://docs.python.org/3/howto/regex.html#regex-howto

""",
        metrics_meta=_metrics_meta,
        keywords=_keywords,
        parameters=_parameters,
        leaderboard_type=e10s.LlmBoolLeaderboardExplanation.explanation_type(),
    )

    def __init__(self):
        evaluators.Evaluator.__init__(self)

        self.args = None
        self.problems = []
        self.log_name = "Text Matching evaluator"

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

        # IMPROVE check dataset columns presence & actual values (non-empty)
        # IMPROVE document column names in the evaluator description

        if not evaluators.Evaluator._check_llm_dataset_compatibility(
            self, params=params, evaluator_keywords=self._keywords
        ):
            return False

        return True

    def setup(self, model, persistence, **kwargs):
        evaluators.Evaluator.setup(self, model, persistence, **kwargs)

        self._resolve_evaluator_params()

        self.log_name = f"Text Matching evaluator {self.mli_key}/{self.key}"

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

        #
        # EVALUATION
        #
        eval_results = self._eval_conditions(
            llm_host=llm_host,
            llm_dataset_as_dt=llm_testset,
            key_2_evaluated_model=key_2_evaluated_model,
        )

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
                display_name="Text Matching evaluation results",
                display_category=e10s.Explanation.DISPLAY_CAT_LLM,
                eval_results=eval_results,
            )
            # FORMATS of the explanation: JSon, CSV, DataTable
            eval_results_explanation.add_json_format()
            eval_results_explanation.add_csv_format()
            eval_results_explanation.add_datatable_format()
            explanations.append(eval_results_explanation)

        # EXPLANATION: ok/fail leaderboard
        t_bool_leaderboard = e10s.LlmBoolLeaderboardExplanation
        leaderboard_explanation = t_bool_leaderboard.from_eval_results(
            evaluator=self,
            eval_results=eval_results,
            metrics_meta=self._metrics_meta,
            metric_id_success=t_bool_leaderboard.KEY_RESULT_CHECK_OK,
            metric_id_failure_message=t_bool_leaderboard.KEY_RESULT_CHECK_ERR_MSG,
            display_name="RAG benchmark leaderboard",
            display_category=e10s.GlobalSummaryFeatImpExplanation.DISPLAY_CAT_LLM,
            key_2_evaluated_model=key_2_evaluated_model,
            llm_host=llm_host,
            do_eval_rc=self.args.get(
                RagStrStrEvaluator.PARAM_EVAL_RC, RagStrStrEvaluator.DEFAULT_EVAL_RC
            ),
            logger=self.logger,
        )
        # FORMAT of the explanation: Markdown
        leaderboard_explanation.add_markdown_format(title=f"{llm_host_str} Benchmarks")
        leaderboard_explanation.add_evalstudio_markdown_format(title="Summary")
        leaderboard_explanation.add_json_format(
            llm_host=llm_host,
            threshold=self.args.get(
                evaluators.Evaluator.PARAM_METRIC_THRESHOLD,
                leaderboard_explanation.METRIC_META_MODEL_PASSES.threshold,
            ),
        )
        explanations.append(leaderboard_explanation)

        # PROBLEMS for alerts and actionability
        self._diagnose_problems(
            leaderboard_explanation=leaderboard_explanation,
            eval_results=eval_results,
            key_2_evaluated_model=key_2_evaluated_model,
        )

        # INSIGHTS
        RagStrStrEvaluator._diagnose_insights(
            leaderboard_explanation=leaderboard_explanation
        )

        # EXPLANATION: HTML fragment
        if self.config and self.config.create_html_representations:
            try:
                html_explanation = e10s.GlobalHtmlFragmentExplanation(
                    evaluator=self,
                    display_name=f"{llm_host_str} benchmark leaderboard as HTML",
                    display_category=e10s.Explanation.DISPLAY_CAT_COMPLIANCE,
                )
                html_explanation.add_html_format(str(leaderboard_explanation.as_html()))
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

    @staticmethod
    def _eval_input_condition(
        llm_host: commons.LlmModelHostType,
        actual_output: str,
        context: str,
        condition: str,
        constraints: list[str] | None,
        do_eval_rc: bool,
        logger,
    ) -> tuple[float, float, float, str, airium.Airium]:
        """Evaluate condition for particular actual answer row.

        Returns
        -------
        Tuple[bool, bool, bool, str]
          Return OK (1.0 or 0.0), retrieval failure (1.0 or 0.0),
          generation failure (1.0 or 0.0) and error message.

        """
        # handle actual answer retrieval error ~ RAG/LLM client crash
        if evaluators.Evaluator._is_internal_err_answer(actual_output):
            # set WORST metrics values
            return (
                0.0,  # OK
                0.0,  # failed retrieval
                1.0,  # failed answer
                evaluators.Evaluator._internal_err_answer_msg(err_msg=actual_output),
                evaluators.Evaluator._internal_err_answer_msg_html(
                    err_msg=actual_output
                ),
            )

        retrieval_result = True
        answer_result = True

        err_msg = ""

        # constraints are used ONLY if condition is not provided
        if not condition and constraints:
            condition = constraints_to_condition(constraints)

        err_msg_html = None
        if condition:
            condition_evaluator = ConditionEvaluator(c=condition, logger=logger)

            # context to be evaluated only in case of RAG models
            fail_ctx_sub_condition = None
            if do_eval_rc and llm_host == commons.LlmModelHostType.RAG:
                # compact retrieval context
                if isinstance(context, list):
                    context_str = " ".join(context)
                else:
                    context_str = context

                # evaluate CONTEXT
                (
                    retrieval_result,
                    fail_ctx_sub_condition,
                ) = condition_evaluator.evaluate(
                    s=context_str, failed_sub_conditions_as_str=True
                )

            # evaluate ANSWER
            (
                answer_result,
                fail_answer_sub_condition,
            ) = condition_evaluator.evaluate(
                s=actual_output, failed_sub_conditions_as_str=True
            )

            # build error message
            err_msg_html = airium.Airium()
            err_msg_sfx = "did not satisfy the condition"
            err_color = "#f6bbd0"
            if llm_host == commons.LlmModelHostType.RAG:
                if not retrieval_result:
                    is_sub_condition = (
                        fail_ctx_sub_condition and fail_ctx_sub_condition != condition
                    )

                    err_msg = f"Retrieved context {err_msg_sfx}: {condition}."
                    # html
                    err_msg_html(f"Retrieved context {err_msg_sfx}: ")
                    with err_msg_html.span(
                        style=(
                            f"background-color: {err_color}"
                            if not is_sub_condition
                            else ""
                        )
                    ):
                        err_msg_html(f"&nbsp;{condition}&nbsp;")
                    err_msg_html(".")
                    if is_sub_condition:
                        err_msg = (
                            f"{err_msg} The following part of the retrieved context "
                            f"condition did not match: {fail_ctx_sub_condition}."
                        )
                        # html
                        err_msg_html(
                            " The following part of the condition did not match: "
                        )
                        with err_msg_html.span(style=f"background-color: {err_color}"):
                            err_msg_html(f"&nbsp;{fail_ctx_sub_condition}&nbsp;")
                        err_msg_html(".")

                if not answer_result:
                    is_sub_condition = (
                        fail_answer_sub_condition
                        and fail_answer_sub_condition != condition
                    )

                    if err_msg:
                        err_msg = f"{err_msg} "
                    err_msg = (
                        f"{err_msg}Generated actual answer {err_msg_sfx}: {condition}."
                    )
                    # html
                    err_msg_html(f" Generated actual answer {err_msg_sfx}: ")
                    with err_msg_html.span(
                        style=(
                            f"background-color: {err_color}"
                            if not is_sub_condition
                            else ""
                        )
                    ):
                        err_msg_html(f"&nbsp;{condition}&nbsp;")
                    err_msg_html(".")
                    if is_sub_condition:
                        err_msg = (
                            f"{err_msg} The following part of the actual answer "
                            f"condition did not match: {fail_answer_sub_condition}."
                        )
                        # html
                        err_msg_html(
                            " The following part of the condition did not match: "
                        )
                        with err_msg_html.span(style=f"background-color: {err_color}"):
                            err_msg_html(f"&nbsp;{fail_answer_sub_condition}&nbsp;")
                        err_msg_html(".")

            else:  # LLM service
                if not answer_result:
                    is_sub_condition = (
                        fail_answer_sub_condition
                        and fail_answer_sub_condition != condition
                    )

                    err_msg = (
                        f"{err_msg}Generated actual answer {err_msg_sfx}: {condition}."
                    )
                    # html
                    err_msg_html(f" Generated actual answer {err_msg_sfx}: ")
                    with err_msg_html.span(
                        style=(
                            f"background-color: {err_color}"
                            if not is_sub_condition
                            else ""
                        )
                    ):
                        err_msg_html(f"&nbsp;{condition}&nbsp;")
                    err_msg_html(".")
                    if is_sub_condition:
                        err_msg = (
                            f"{err_msg} The following part of the condition did not "
                            f"match: {fail_answer_sub_condition}."
                        )
                        # html
                        err_msg_html(
                            " The following part of the condition did not match: "
                        )
                        with err_msg_html.span(style=f"background-color: {err_color}"):
                            err_msg_html(f"&nbsp;{fail_answer_sub_condition}&nbsp;")
                        err_msg_html(".")

        ok = 1.0 if retrieval_result and answer_result else 0.0
        failed_retrieval = 1.0 if not retrieval_result else 0.0
        failed_answer = 1.0 if not answer_result else 0.0

        return (
            ok,
            failed_retrieval,
            failed_answer,
            err_msg,
            err_msg_html if err_msg else None,
        )

    def _eval_conditions(
        self,
        llm_host: commons.LlmModelHostType,
        llm_dataset_as_dt: datatable,
        key_2_evaluated_model: dict,
    ) -> datasets.LlmEvalResults:
        llm_dataset = datasets.LlmDataset.from_datatable_dict(
            llm_dataset_as_dt.to_dict()
        )
        eval_results = datasets.LlmEvalResults()

        self.report_progress(
            0.01, f"Checking conditions for {len(llm_dataset.inputs)} test cases"
        )

        #
        # EVALUATION
        #
        for e, r in enumerate(llm_dataset.inputs):
            # evaluate conditions for the test case
            RagStrStrEvaluator.eval_tc_conditions(
                row=r,
                evaluator=self,
                evaluator_id=self.key,
                evaluator_display_name=self._display_name,
                eval_results=eval_results,
                do_eval_rc=self.args.get(
                    RagStrStrEvaluator.PARAM_EVAL_RC,
                    RagStrStrEvaluator.DEFAULT_EVAL_RC,
                ),
                key_2_evaluated_model=key_2_evaluated_model,
                llm_host=llm_host,
                logger=self.logger,
            )

            # progress
            self.report_progress(
                progress=progress_utils.ProgressCallbackContext.progress_for_steps(
                    e + 1, len(llm_dataset.inputs)
                ),
                message=(
                    f"Checked {e + 1}/{len(llm_dataset.inputs)} test cases conditions"
                ),
            )

        return eval_results

    @staticmethod
    def eval_tc_conditions(
        row: datasets.LlmDataset.LlmDatasetRow,
        evaluator: evaluators.Evaluator,
        evaluator_id: str,
        evaluator_display_name: str,
        eval_results: datasets.LlmEvalResults,
        key_2_evaluated_model: dict,
        llm_host: commons.LlmModelHostType,
        do_eval_rc: bool,
        logger,
    ):
        t_bool_leaderboard = e10s.LlmBoolLeaderboardExplanation

        is_rag_evaluation = (llm_host == commons.LlmModelHostType.RAG,)
        evaluated_model = key_2_evaluated_model[row.model_key]
        # if neither condition nor constraints are provided,
        # then use expected_output as the condition (exact match)
        if (
            not row.output_condition
            and not row.output_constraints
            and row.expected_output
        ):
            row.output_constraints = [row.expected_output]
        if row.output_condition or row.output_constraints:
            (
                ok,
                failed_retrieval,
                failed_answer,
                err_msg,
                err_msg_html,
            ) = RagStrStrEvaluator._eval_input_condition(
                llm_host=llm_host,
                actual_output=row.actual_output,
                context=row.context,
                condition=row.output_condition,
                constraints=row.output_constraints,
                do_eval_rc=do_eval_rc,
                logger=logger,
            )
        else:
            ok = 1.0
            failed_retrieval = 0.0
            failed_answer = 0.0
            err_msg = ""
            err_msg_html = None
        # PROBLEMS
        if is_rag_evaluation and bool(failed_retrieval) and not bool(failed_answer):
            t_html_fragment = e10s.GlobalHtmlFragmentExplanation

            html = airium.Airium()
            html("Retrieved ")
            with html.b(klass="w3-black"):
                html("&nbsp;context&nbsp;")
            html("&nbsp; provided by evaluated RAG for the prompt ")
            with html.b():
                with html.i():
                    html(f"'{row.i}'")
            html(" did ")
            with html.b(klass="w3-black"):
                html("&nbsp;not satisfy&nbsp;")
            html("&nbsp; the output conditions checked by ")
            with html.code():
                html(f"{RagStrStrEvaluator._display_name}")
            html("evaluator, but the answer generated by the LLM model ")
            with html.code():
                html(f"{evaluated_model.llm_model_name} ")
            with html.b(klass="w3-black"):
                html("&nbsp;did&nbsp;")
            html(".")

            problem = problems.ProblemAndAction(
                description=(
                    f"Retrieved context provided by RAG for the evaluation did not "
                    f"satisfy the output conditions for the prompt '{row.i}', "
                    f"but the answer generated by the LLM model "
                    f"{evaluated_model.llm_model_name} did."
                ),
                description_html=html,
                problem_type="retrieval",
                problem_code=problems.AVIDProblemCode.P0100_DATA,
                problem_attrs={
                    problems.ProblemAndAction.ATTR_MODEL_NAME: (
                        evaluated_model.llm_model_name
                    ),
                    problems.ProblemAndAction.ATTR_ROW_KEYS: [(row.key, row.model_key)],
                    # input dataset ~ test lab ~ key is the test case key
                    problems.ProblemAndAction.ATTR_TEST_CASE_KEYS: [row.key],
                    problems.ProblemAndAction.ATTR_EVALUATOR_NAME: (
                        RagStrStrEvaluator._display_name
                    ),
                },
                severity=problems.ProblemSeverity.medium,
                actions_description=(
                    "The answer generated by the LLM model is correct, which means "
                    "that RAG either did not provide the context or LLM did not "
                    "use the context to generate the answer "
                    "(possibly hallucinated)."
                ),
                evaluator_id=evaluator_id,
                evaluator_name=evaluator_display_name,
                explanation_type=t_html_fragment.explanation_type(),
                explanation_name=t_html_fragment.__name__,
                explanation_mime=f5s.HtmlFormat.mime,
                resources=[],
            )
            evaluator.add_problem(problem)
        # tokenization / fragments for the evaluation results:
        # - highlighting of the actual answer (and context) would NOT
        #   be useful in this case as the reason of the failure is that
        #   the conditions did NOT match the actual answer
        # - therefore the actual answer meta will NOT highlight anything,
        #   indicating it by no metrics values in the fragment which contains
        #   the actual answer
        # - HOWEVER, `meta` parameter will be used to pass the ERROR MESSAGE,
        #   with the error message explaining which part of the condition
        #   was not met by the actual answer (represented as string and HTML)
        meta = {}
        if err_msg:
            meta[tokenization.META_ERR_MSG] = err_msg
        if err_msg_html:
            meta[tokenization.META_ERR_MSG_HTML] = str(err_msg_html)
        actual_answer_meta = [
            tokenization.Tokenization(
                tokenization=tokenization.TOKENIZATION_TYPE_F,
                data=[
                    tokenization.TextFragment(
                        text=row.actual_output,
                        metrics={},
                        meta=meta,
                    )
                ],
            )
        ]
        # add to result
        eval_results.add_result(
            datasets.LlmEvalResults.LlmEvalResultRow(
                dataset_row=row,
                metrics={
                    t_bool_leaderboard.KEY_RESULT_CHECK_OK: ok,
                    t_bool_leaderboard.KEY_RESULT_CHECK_FAIL: 0.0 if ok else 1.0,
                    t_bool_leaderboard.KEY_RESULT_CHECK_FAIL_R: failed_retrieval,
                    t_bool_leaderboard.KEY_RESULT_CHECK_FAIL_A: failed_answer,
                    t_bool_leaderboard.KEY_RESULT_CHECK_FAIL_P: 0.0,
                    t_bool_leaderboard.KEY_RESULT_CHECK_ERR_MSG: err_msg,
                },
                actual_output_meta=actual_answer_meta,
            )
        )

    @staticmethod
    def _diagnose_insights(
        leaderboard_explanation: e10s.LlmBoolLeaderboardExplanation,
    ):
        t_html_fragment = e10s.GlobalHtmlFragmentExplanation

        leaderboard_explanation.get_insights(
            extra_description_actions=(
                "Check the prompt, expected answer and condition - are they "
                "correct? Check models answers in failed cases and look for a "
                "common denominator and/or root cause of these failures."
            ),
            explanation_type=t_html_fragment.explanation_type(),
            explanation_name=t_html_fragment.__name__,
            explanation_mime=f5s.HtmlFormat.mime,
        )

    def _diagnose_problems(
        self,
        leaderboard_explanation: e10s.LlmBoolLeaderboardExplanation,
        eval_results: datasets.LlmEvalResults,
        key_2_evaluated_model: dict,
    ):
        # perturbation flips
        self._diagnose_perturbation_problems(
            eval_results=eval_results,
            key_2_evaluated_model=key_2_evaluated_model,
        )

        # threshold failures
        problems.problems_for_bool_leaderboard(
            evaluator=self,
            leaderboard=leaderboard_explanation,
            metric_threshold=self.args.get(
                evaluators.Evaluator.PARAM_METRIC_THRESHOLD,
                leaderboard_explanation.METRIC_META_MODEL_PASSES.threshold,
            ),
            primary_metric_meta=leaderboard_explanation.METRIC_META_MODEL_PASSES,
            problem_type="accuracy",
            problem_code=problems.AVIDProblemCode.P0200_MODEL,
            actions_description=(
                "For all failed test cases, check the prompt, expected answer, and "
                "condition to see if they are correct. Then, examine the model's "
                "answers in the failed cases and look for a common denominator or "
                "root cause of these failures."
            ),
            explanation_type=e10s.GlobalHtmlFragmentExplanation.explanation_type(),
            explanation_name=e10s.GlobalHtmlFragmentExplanation.__name__,
            explanation_mime=f5s.HtmlFormat.mime,
        )

    def get_result(
        self,
    ) -> results.LeaderboardResult:
        return results.LeaderboardResult(
            persistence=self.persistence,
            explainer_id=RagStrStrEvaluator.explainer_id(),
            h2o_sonar_config=self.config,
            logger=self.logger,
            explanation=e10s.LlmBoolLeaderboardExplanation,
            explanation_format=f5s.CustomJsonFormat,
        )


class ConditionEvaluator:
    """Condition evaluator for the AIP-160 syntax subset."""

    def __init__(self, c: str, logger):
        """Constructor.

        Parameters
        ----------
        c : str
            The condition to be used for the input data evaluation.
        logger : loggers.SonarLogger
            The logger instance.

        """
        self.logger = logger
        self.err_prefix = "Condition evaluation error: "

        self.lexer = parsing.ConditionLexer(logger=logger)
        self.parser = parsing.ConditionRDParser(logger=logger)

        # descent first evaluation function pointers
        self.dfe = {
            parsing.ConditionSymbol.AND.value: self._e_and,
            parsing.ConditionSymbol.OR.value: self._e_or,
            parsing.ConditionSymbol.NOT.value: self._e_not,
            parsing.ConditionSymbol.PARENS.value: self._e_parens,
        }

        # build condition AST to be subsequently used for evaluation
        self.c_ast = self.parser.parse(self.lexer.tokenize(c))

    def _e_descent(self, ast: list, s: str) -> tuple[bool, list]:
        """Evaluate the AST bottom-up (post-order)."""
        self.logger.debug(f"    V: {ast}")
        if ast:
            return self.dfe.get(ast[parsing.SYMBOL], self._e_op)(ast, s)

        # empty AST or None (leaf) evaluates to True
        return True, []

    def _e_parens(self, ast: list, i: str) -> tuple[bool, list]:
        """Evaluate the PARENTHESIS operator."""
        if not ast:
            raise ValueError(
                f"{self.err_prefix} expected PARENTHESES operator, but AST is empty "
                f"at offset: {i}, AST: {ast}"
            )
        if not ast[parsing.SYMBOL] == parsing.ConditionSymbol.PARENS.value:
            raise ValueError(
                f"{self.err_prefix} expected PARENTHESIS operator, but unexpected "
                f"AST root {ast} detected at offset: {i}"
            )
        if not len(ast[parsing.OPERANDS]) == 1:
            raise ValueError(
                f"{self.err_prefix} PARENTHESIS operator must have exactly one "
                f"operand at offset: {i}, AST: {ast}"
            )

        operand = ast[parsing.OPERANDS][0]

        return self._e_descent(operand, i)

    def _e_not(self, ast: list, i: str) -> tuple[bool, list]:
        """Evaluate the NOT operator."""
        if not ast:
            raise ValueError(
                f"{self.err_prefix} expected NOT operator, but AST is empty at offset: "
                f"{i}, AST: {ast}"
            )
        if not ast[parsing.SYMBOL] == parsing.ConditionSymbol.NOT.value:
            raise ValueError(
                f"{self.err_prefix} expected NOT operator, but unexpected "
                f"AST root {ast} detected at offset: {i}"
            )
        if not len(ast[parsing.OPERANDS]) == 2 and ast[parsing.OPERANDS][1] is None:
            raise ValueError(
                f"{self.err_prefix} NOT operator must have exactly one operand "
                f"at offset: {i}, AST: {ast}"
            )

        operand = ast[parsing.OPERANDS]

        (_value, value_ast) = self._e_descent(operand, i)
        value = not _value
        return (True, []) if value else (False, value_ast)

    def _e_and(self, ast: list, i: str) -> tuple[bool, list]:
        if not ast:
            raise ValueError(
                f"{self.err_prefix} expected AND operator, but AST is empty at offset: "
                f"{i}, AST: {ast}"
            )
        if not ast[parsing.SYMBOL] == parsing.ConditionSymbol.AND.value:
            raise ValueError(
                f"{self.err_prefix} expected AND operator, but unexpected "
                f"AST root {ast} detected at offset: {i}"
            )
        if not len(ast[parsing.OPERANDS]) == 2:
            raise ValueError(
                f"{self.err_prefix} AND operator must have exactly two operands "
                f"at offset: {i}, AST: {ast}"
            )

        # partial evaluation
        (left_operand, left_ast) = self._e_descent(ast[parsing.OPERANDS][0], i)
        if not left_operand:
            return False, left_ast
        return self._e_descent(ast[parsing.OPERANDS][1], i)

    def _e_or(self, ast: list, i: str) -> tuple[bool, list]:
        if not ast:
            raise ValueError(
                f"{self.err_prefix} expected OR operator, but AST is empty at offset: "
                f"{i}, AST: {ast}"
            )
        if not ast[parsing.SYMBOL] == parsing.ConditionSymbol.OR.value:
            raise ValueError(
                f"{self.err_prefix} expected OR operator, but unexpected "
                f"AST root {ast} detected at offset: {i}"
            )
        if not len(ast[parsing.OPERANDS]) == 2:
            raise ValueError(
                f"{self.err_prefix} OR operator must have exactly two operands: "
                f"at offset: {i}, AST: {ast}"
            )

        # partial evaluation
        (left_operand_value, left_operand_ast) = self._e_descent(
            ast[parsing.OPERANDS][0], i
        )
        if left_operand_value:
            return True, []

        (right_operand_value, right_operand_ast) = self._e_descent(
            ast[parsing.OPERANDS][1], i
        )
        return (True, []) if right_operand_value else (False, ast)

    def _e_op(self, ast: list, i: str) -> tuple[bool, list]:
        if not ast:
            raise ValueError(
                f"{self.err_prefix} expected operand, but AST is empty at offset: {i}"
            )

        if ast[parsing.SYMBOL].startswith(parsing.ConditionSymbol.FN_REGEXP.value):
            # regular expression
            if not len(ast[parsing.OPERANDS]) == 2:
                raise ValueError(
                    f"{self.err_prefix} expected regular expression operand node with "
                    f"regexp and terminator, but detected unexpected AST: {ast} at "
                    f"offset: {i}"
                )
            if ast[parsing.OPERANDS][1] is not None:
                raise ValueError(
                    f"{self.err_prefix} expected operand node with regular expression "
                    f"and terminator, but detected unexpected AST: {ast} at offset: {i}"
                )
            raw_regexp = ast[parsing.OPERANDS][0]
            regexp = raw_regexp[1:-1] if len(raw_regexp) > 1 else ""
            return (
                (True, [])
                if not regexp or re.search(regexp, i) is not None
                else (False, ast)
            )
        else:
            # string
            if not len(ast) == 2:
                raise ValueError(
                    f"{self.err_prefix} expected operand node with value and "
                    f"terminator - detected unexpected AST: {ast} at offset: {i}"
                )
            if ast[parsing.OPERANDS] is not None:
                raise ValueError(
                    f"{self.err_prefix} expected operand node with value and "
                    f"terminator - detected unexpected AST: {ast} at offset: {i}"
                )
            # strip leading and trailing " from the string
            value = (
                ast[parsing.SYMBOL][1:-1] in i if len(ast[parsing.SYMBOL]) > 1 else True
            )
            return (True, []) if value else (False, ast)

    def evaluate(
        self,
        s: str,
        c_ast: list | None = None,
        failed_sub_conditions_as_str: bool = False,
    ) -> tuple[bool, list]:
        """Evaluate the condition.

        Parameters
        ----------
        s : str
            The string to be evaluated.
        c_ast : list | None
            Optional custom condition AST.
        failed_sub_conditions_as_str : bool
            If ``True``, return the failed sub-conditions as string, otherwise as AST.

        Returns
        -------
        Tuple[bool, list] :
            The evaluation result and the list of failed sub-conditions.

        """
        self.logger.debug("Evaluating condition:")
        c_ast = c_ast or self.c_ast

        # evaluate the condition by traversing the AST bottom-up (post-order)
        (result, failed_sub_conditions) = self._e_descent(ast=c_ast, s=s)
        if failed_sub_conditions_as_str and failed_sub_conditions:
            condition_ast = parsing.ConditionAst()
            failed_sub_conditions = condition_ast.to_string(failed_sub_conditions)
        return result, failed_sub_conditions


def constraints_to_condition(constraint: list | None) -> str:
    """Convert constraints to more powerful AIP-160 syntax based expression.
    The main motivation is KISS - use one evaluator for all types of constraints.

    Parameters
    ----------
    constraint : list | None
        Constraints structure to be converted to the condition.

    Returns
    -------
    str
        The condition string.

    """

    def _regexp_to_condition(regexp: str) -> str:
        prefix_regexp = "REGEXP:"

        if regexp.startswith(prefix_regexp):
            # strip the prefix
            regexp = regexp[len(prefix_regexp) :]
            return (
                f'{parsing.ConditionLexer.S_REGEXP_PREFIX}"{regexp}"'
                f"{parsing.ConditionLexer.S_RIGHT_PAREN}"
            )
        return ""

    def _escape_operand(op: str) -> str:
        if op and "'" in op or '"' in op:
            return re.sub(r"(?<!\\)(['\"])", r"\\\1", op)
        return op

    condition = ""

    if not constraint:
        return condition

    for and_operand in constraint:
        if isinstance(and_operand, list):
            or_condition = ""
            for or_operand in and_operand:
                if isinstance(or_operand, str):
                    operand = (
                        _regexp_to_condition(or_operand)
                        or f'"{_escape_operand(or_operand)}"'
                    )
                    or_condition += (
                        f" {parsing.ConditionLexer.S_OR} {operand}"
                        if or_condition
                        else f"{operand}"
                    )
                else:
                    raise ValueError(f"Invalid OR operand type: '{type(or_operand)}'")
            condition += (
                f" {parsing.ConditionLexer.S_AND} ({or_condition})"
                if condition
                else f"({or_condition})"
            )
        elif isinstance(and_operand, str):
            operand = (
                _regexp_to_condition(and_operand) or f'"{_escape_operand(and_operand)}"'
            )
            condition += (
                f" {parsing.ConditionLexer.S_AND} {operand}"
                if condition
                else f"{operand}"
            )
        else:
            raise ValueError(f"Invalid AND operand type: '{type(and_operand)}'")

    return condition
