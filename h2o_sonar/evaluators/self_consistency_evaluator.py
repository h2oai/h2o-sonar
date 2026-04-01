# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import random
import traceback

import numpy as np

from h2o_sonar import loggers
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import datasets
from h2o_sonar.lib.api import evaluators
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import problems
from h2o_sonar.lib.api import results


try:
    import rouge_score.rouge_scorer

    HAS_ROUGE_SCORE = True
except ImportError:
    HAS_ROUGE_SCORE = False


class SelfConsistencyEvaluator(evaluators.Evaluator):
    _display_name = "Self-consistency"
    _tagline = "Assess the self-consistency of generated actual answers."

    METRIC_ROUGE_1 = "rouge_1"
    METRIC_ROUGE_L = "rouge_l"
    METRIC_BERT_SCORE = "bert_score"

    _metrics_meta = commons.MetricsMeta(
        metrics=[
            commons.MetricMeta(
                key=METRIC_ROUGE_1,
                display_name="ROUGE-1",
                description=(
                    "ROUGE-1 metric measures the overlap of 1-grams (individual words) "
                    "between the generated and the reference actual answers. It "
                    "measures how much of the reference actual answer(s) is present in "
                    "the other actual answers. Rewards content coverage and capturing "
                    "key information."
                ),
                higher_is_better=True,
                threshold=evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
                is_primary_metric=False,
            ),
            commons.MetricMeta(
                key=METRIC_ROUGE_L,
                display_name="ROUGE-L",
                description=(
                    "ROUGE-L metric considers the longest common subsequence (LCS) "
                    "between the generated and reference actual answers."
                ),
                higher_is_better=True,
                threshold=evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
                is_primary_metric=True,
            ),
            # TODO: add sentence embedding similarity - will make the evaluator slower
            # TODO: add BERT Score - will make the evaluator (much) slower
            # commons.MetricMeta(
            #     key=METRIC_BERT_SCORE,
            #     display_name="BERT Score",
            #     description=(
            #         "BERT Score is a metric that uses BERT embeddings to evaluate "
            #         "the similarity between the generated and reference texts. It "
            #         "captures SEMANTIC meaning and context."
            #     ),
            #     higher_is_better=True,
            #     threshold=evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
            #     is_primary_metric=False,
            # ),
        ]
    )

    # COMPATIBILITY: LLM model explanations only
    _llm = True
    _rag = True

    # GLOBAL: leaderboard as global explanation
    _global_explanation = True

    # EXPLANATION TYPES created by the evaluator
    _explanation_types = [
        e10s.LlmEvalResultsExplanation,
        e10s.LlmHeatmapLeaderboardExplanation,
    ]

    PARAM_MAX_TC_GROUP_SIZE = "max_tc_group_size"
    # ROUGE estimate: 100 items, 4950 pairs, ~0.01 sec per pair = 50 sec
    DEFAULT_MAX_TC_GROUP_SIZE = 100

    _parameters = [
        evaluators.EvaluatorParam(
            param_name=PARAM_MAX_TC_GROUP_SIZE,
            description=(
                f"Maximum number of test cases in the group of actual answers to the"
                f"same question which can be used for metrics calculation. This "
                f"parameter is used to prevent combinatorial explosion of comparisons "
                f"when there are large number of actual answers for the same "
                f"question as the method creates all pairs of actual answers within "
                f"the group to compare them. If the number of actual answers for the "
                f"same question exceeds this limit, then only the first "
                f"{PARAM_MAX_TC_GROUP_SIZE} actual answers are used for the "
                f"comparison."
            ),
            param_type=commons.EvaluatorParamType.int,
            default_value=DEFAULT_MAX_TC_GROUP_SIZE,
            src=evaluators.EvaluatorParam.SRC_EVALUATOR_PARAMS,
        ),
        evaluators.Evaluator._get_custom_param_metric_threshold(
            _metrics_meta.get_primary_metric()
        ),
        evaluators.Evaluator._PARAM_SAVE_LLM_RESULT,
        evaluators.Evaluator._get_custom_param_min_test_case(),
    ]

    _keywords = [
        evaluators.KEYWORD_LLM,
        evaluators.KEYWORD_EVALUATES_LLM,
        evaluators.KEYWORD_EVALUATES_RAG,
        evaluators.KEYWORD_RQ_AA,
        evaluators.KEYWORD_SR_11_7_CS,
        evaluators.KEYWORD_SR_11_7_OA,
        evaluators.KEYWORD_NIST_AI_RMF_S,
        evaluators.KEYWORD_PROBLEM_TYPE_QA,
        evaluators.KEYWORD_ES_GENERATE,
        evaluators.KEYWORD_METHOD_NGRAM,
        evaluators.KEYWORD_METHOD_TYPE_DETERMINISTIC,
    ]

    _modules_needed_by_name = ["rouge_score==0.1.2"]

    _brief_description = """Self-consistency Evaluator assesses the consistency
of generated actual answers to the identical question by comparing them using
`ROUGE` metrics. The purpose of this evaluator is to measure the stability of
a specific model's answer when it is prompted multiple times with the identical
question.

- Compatibility: RAG and LLM models."""
    _description = evaluators.Evaluator._description_builder(
        brief=f"""{_brief_description}

**Method**:

- The evaluator groups actual answers by their corresponding question and compares
  them using `ROUGE` metrics to determine their consistency.
- To function correctly, the test must contain multiple actual answers for each
  identical question.
  The `CopyPerturbator` can be used to create these test cases, as it duplicates
  existing test cases in a given test. If the question is present exactly once,
  then it gets `1.0` score for all metrics as it is perfectly consistent with itself.
- `ROUGE-1` measures the overlap of 1-grams (individual words) between the generated
  and the reference summaries.
- `ROUGE-L` considers the longest common subsequence (LCS) between the generated and
  reference summaries.
- These ROUGE metrics provide a quantitative evaluation of the similarity between
  the generated and reference texts to assess how much of the reference text
  (the first generated actual answer) is present in the generated text (subsequent
  actual answers).

See also:

- 3rd party library ROUGE: https://pypi.org/project/rouge-score/
- 3rd party ROUGE source code:
  https://github.com/google-research/google-research/tree/master/rouge""",
        metrics_meta=_metrics_meta,
        keywords=_keywords,
        parameters=_parameters,
        leaderboard_type=e10s.LlmHeatmapLeaderboardExplanation.explanation_type(),
    )

    def __init__(self):
        evaluators.Evaluator.__init__(self)

        self.args = None
        self.problems = []
        self.log_name = "ROUGE evaluator"

    def check_compatibility(
        self,
        params: commons.CommonInterpretationParams | None = None,
        **evaluator_params,
    ) -> bool:
        if not HAS_ROUGE_SCORE:
            self.logger.warning(self._check_compatibility_pckg_err_msg("rouge-score"))
            return False

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

        # check that at least one row has actual answer
        if not self._check_llm_dataset_field_presence(
            params=params,
            require_actual_answer=True,
            require_expected_answer=False,
        ):
            return False

        return True

    def setup(self, model, persistence, **kwargs):
        evaluators.Evaluator.setup(self, model, persistence, **kwargs)

        self._resolve_evaluator_params()

        self.log_name = f"Self-consistency evaluator {self.mli_key}/{self.key}"

    @staticmethod
    def _prep_tc_pairs(
        llm_dataset: datasets.LlmDataset,
        max_group_size: int,
        logger: loggers.SonarLogger,
    ) -> dict:
        """Prepares test case pairs for the evaluation:

        - In order to avoid problems when evaluator would compare a (randomly)
          chosen referential actual answer which may be an outlier (different from
          all other, much longer then all others, ...), we create all pairs of actual
          answers within the group and compute metrics as average over all pairs.
        - To avoid combinatorial explosion when there are large number of actual
          answers for the same question, there is limit the group size
          `max_group_size`.
        - If the group size exceeds `max_group_size`, stratified sampling based
          on input length buckets to select `max_group_size` items from the group.

        Parameters
        ----------
        llm_dataset : datasets.LlmDataset
            LLM dataset with test cases to be grouped.
        max_group_size : int
            Maximum size of the group of actual answers for the particular model and
            the same question.
        logger : loggers.SonarLogger
            Logger for logging messages.

        Returns
        -------
        dict :
            Grouped test case pairs: model_key > question > list of dataset rows pairs.

        """

        # GROUP TCs by model and question (within model by design)
        # m: model_key > question > list of dataset rows
        llm_dataset_grouped = {}
        for group_tc in llm_dataset.inputs:
            model_key = group_tc.model_key
            if model_key not in llm_dataset_grouped:
                llm_dataset_grouped[model_key] = {}
            if group_tc.i not in llm_dataset_grouped[model_key]:
                llm_dataset_grouped[model_key][group_tc.i] = []
            llm_dataset_grouped[model_key][group_tc.i].append(group_tc)

        # REDUCE group size if exceeds max_group_size using stratified sampling
        for model_key in llm_dataset_grouped:
            for q in list(llm_dataset_grouped[model_key].keys()):
                q_group = llm_dataset_grouped[model_key][q]
                if len(q_group) <= max_group_size:
                    continue

                # compute input lengths (number of characters)
                lengths = [len(tc.i) if tc.i is not None else 0 for tc in q_group]

                # number of buckets - keep it reasonably small
                num_buckets = min(10, max_group_size)

                # if all lengths equal -> just take first N (stable fallback)
                if min(lengths) == max(lengths):
                    sampled = q_group[:max_group_size]
                    llm_dataset_grouped[model_key][q] = sampled
                    continue

                # percentile-based bucket edges
                try:
                    pct = np.linspace(0, 100, num_buckets + 1)
                    edges = np.percentile(lengths, pct).tolist()
                except Exception:
                    # fallback to equal-width buckets
                    minl, maxl = min(lengths), max(lengths)
                    edges = [
                        minl + (maxl - minl) * i / num_buckets
                        for i in range(num_buckets + 1)
                    ]

                # assign items to buckets
                buckets: list[list] = [[] for _ in range(num_buckets)]
                for idx, length in enumerate(lengths):
                    # find first bucket where length <= upper edge
                    placed = False
                    for b in range(num_buckets):
                        upper = edges[b + 1]
                        if length <= upper or b == num_buckets - 1:
                            buckets[b].append(q_group[idx])
                            placed = True
                            break
                    if not placed:
                        buckets[-1].append(q_group[idx])

                # sample from buckets
                base = max_group_size // num_buckets
                remainder = max_group_size % num_buckets
                rng = random.Random(42)
                sampled: list = []
                for b_idx, bucket in enumerate(buckets):
                    k = base + (1 if b_idx < remainder else 0)
                    if not bucket:
                        continue
                    if len(bucket) <= k:
                        sampled.extend(bucket)
                    else:
                        sampled.extend(rng.sample(bucket, k))

                # fill up if we sampled fewer items due to empty buckets
                if len(sampled) < max_group_size:
                    remaining = [tc for tc in q_group if tc not in sampled]
                    need = max_group_size - len(sampled)
                    if remaining:
                        if len(remaining) <= need:
                            sampled.extend(remaining)
                        else:
                            sampled.extend(rng.sample(remaining, need))

                # final safety: if still too many/too few, truncate or extend
                if len(sampled) > max_group_size:
                    sampled = sampled[:max_group_size]

                llm_dataset_grouped[model_key][q] = sampled

        # PAIRS GENERATION within each group
        # m: model_key > question > list of (reference_tc, comparison_tc) tuples
        pairs_grouped = {}
        for model_key in llm_dataset_grouped:
            pairs_grouped[model_key] = {}
            for q in llm_dataset_grouped[model_key]:
                q_group = llm_dataset_grouped[model_key][q]

                # n group members =  N*(N-1)/2 unique pairs
                pairs = []
                for i in range(len(q_group)):
                    for j in range(i + 1, len(q_group)):
                        pairs.append((q_group[i], q_group[j]))

                pairs_grouped[model_key][q] = pairs

        return pairs_grouped

    def evaluate(self, llm_testset, **kwargs) -> list:
        save_llm_result = self.args.get(
            evaluators.Evaluator.PARAM_SAVE_LLM_RESULT,
            evaluators.Evaluator.DEFAULT_SAVE_LLM_RESULT,
        )

        metrics_threshold = self.args.get(
            evaluators.Evaluator.PARAM_METRIC_THRESHOLD,
            evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
        )
        self._metrics_meta.set_threshold(metrics_threshold)

        # RAG models: key -> model
        key_2_evaluated_model = {m.key: m for m in self.models}

        llm_dataset = datasets.LlmDataset.from_datatable_dict(llm_testset.to_dict())
        llm_dataset_grouped = SelfConsistencyEvaluator._prep_tc_pairs(
            llm_dataset=llm_dataset,
            max_group_size=self.args.get(
                SelfConsistencyEvaluator.PARAM_MAX_TC_GROUP_SIZE,
                SelfConsistencyEvaluator.DEFAULT_MAX_TC_GROUP_SIZE,
            ),
            logger=self.logger,
        )

        # EVALUATE within each group of pairs
        eval_results = datasets.LlmEvalResults()
        for model_key in llm_dataset_grouped:
            for q in llm_dataset_grouped[model_key]:
                q_pairs = llm_dataset_grouped[model_key][q]

                self.logger.info(
                    f"{20 * 'X'}\n"
                    f"{len(q_pairs)} pairs for group model={model_key} "
                    f"question='{q}'"
                )

                # EMPTY group
                if not q_pairs:
                    continue

                # m: tc.key -> {metric_key -> [list of metric values]}
                tc_metrics = {}

                # ROUGE score cache for the group:
                # (reference_text, comparison_text) -> {rouge1, rougeL}
                rouge_cache = {}

                # EVALUATE each pair
                for reference_tc, comparison_tc in q_pairs:
                    if reference_tc.key not in tc_metrics:
                        tc_metrics[reference_tc.key] = {
                            self.METRIC_ROUGE_1: [],
                            self.METRIC_ROUGE_L: [],
                        }
                    if comparison_tc.key not in tc_metrics:
                        tc_metrics[comparison_tc.key] = {
                            self.METRIC_ROUGE_1: [],
                            self.METRIC_ROUGE_L: [],
                        }

                    # handle RAG/LLM internal errors
                    if evaluators.Evaluator._is_internal_err_answer(
                        reference_tc.actual_output
                    ) or evaluators.Evaluator._is_internal_err_answer(
                        comparison_tc.actual_output
                    ):
                        # set worst scores
                        tc_metrics[reference_tc.key][self.METRIC_ROUGE_1].append(0.0)
                        tc_metrics[reference_tc.key][self.METRIC_ROUGE_L].append(0.0)
                        tc_metrics[comparison_tc.key][self.METRIC_ROUGE_1].append(0.0)
                        tc_metrics[comparison_tc.key][self.METRIC_ROUGE_L].append(0.0)
                        continue

                    # handle empty actual output
                    if (
                        not reference_tc.actual_output
                        or not isinstance(reference_tc.actual_output, str)
                    ) or (
                        not comparison_tc.actual_output
                        or not isinstance(comparison_tc.actual_output, str)
                    ):
                        # log warning
                        self.logger.warning(
                            f"{self.log_name}: Empty actual output detected for test "
                            f"case(s) in model={model_key}. Setting worst scores."
                        )
                        # set worst scores
                        tc_metrics[reference_tc.key][self.METRIC_ROUGE_1].append(0.0)
                        tc_metrics[reference_tc.key][self.METRIC_ROUGE_L].append(0.0)
                        tc_metrics[comparison_tc.key][self.METRIC_ROUGE_1].append(0.0)
                        tc_metrics[comparison_tc.key][self.METRIC_ROUGE_L].append(0.0)
                        continue

                    cache_key = (
                        reference_tc.actual_output,
                        comparison_tc.actual_output,
                    )

                    # CACHE check
                    if cache_key in rouge_cache:
                        rouge_1_score = rouge_cache[cache_key]["rouge1"]
                        rouge_l_score = rouge_cache[cache_key]["rougeL"]
                    else:
                        rouge = rouge_score.rouge_scorer.RougeScorer(
                            ["rouge1", "rougeL"], use_stemmer=True
                        ).score(comparison_tc.actual_output, reference_tc.actual_output)

                        rouge_1_score = rouge["rouge1"].fmeasure
                        rouge_l_score = rouge["rougeL"].fmeasure

                        rouge_cache[cache_key] = {
                            "rouge1": rouge_1_score,
                            "rougeL": rouge_l_score,
                        }

                    tc_metrics[reference_tc.key][self.METRIC_ROUGE_1].append(
                        rouge_1_score
                    )
                    tc_metrics[reference_tc.key][self.METRIC_ROUGE_L].append(
                        rouge_l_score
                    )
                    tc_metrics[comparison_tc.key][self.METRIC_ROUGE_1].append(
                        rouge_1_score
                    )
                    tc_metrics[comparison_tc.key][self.METRIC_ROUGE_L].append(
                        rouge_l_score
                    )

                # RESULT: compute average metrics per test case
                for tc_key, metrics in tc_metrics.items():
                    tc_obj = None
                    for reference_tc, comparison_tc in q_pairs:
                        if reference_tc.key == tc_key:
                            tc_obj = reference_tc
                            break
                        if comparison_tc.key == tc_key:
                            tc_obj = comparison_tc
                            break

                    if tc_obj is None:
                        continue

                    avg_rouge_1 = (
                        sum(metrics[self.METRIC_ROUGE_1])
                        / len(metrics[self.METRIC_ROUGE_1])
                        if metrics[self.METRIC_ROUGE_1]
                        else 0.0
                    )
                    avg_rouge_l = (
                        sum(metrics[self.METRIC_ROUGE_L])
                        / len(metrics[self.METRIC_ROUGE_L])
                        if metrics[self.METRIC_ROUGE_L]
                        else 0.0
                    )

                    # add RESULT for row
                    eval_results.add_result(
                        datasets.LlmEvalResults.LlmEvalResultRow(
                            dataset_row=tc_obj,
                            metrics={
                                self.METRIC_ROUGE_1: avg_rouge_1,
                                self.METRIC_ROUGE_L: avg_rouge_l,
                            },
                        )
                    )

                # FLUSH metric score cache
                rouge_cache.clear()

        #
        # NORMALIZATION of the evaluation RESULTS
        #

        sort_by_metric = self._metrics_meta.get_primary_metric().key

        # EXPLANATIONS
        explanations = []

        # EXPLANATION: all data (per prompt metrics)
        if save_llm_result:
            eval_results_explanation = e10s.LlmEvalResultsExplanation(
                evaluator=self,
                display_name="Rouge evaluation results",
                display_category=e10s.Explanation.DISPLAY_CAT_LLM,
                eval_results=eval_results,
            )
            # FORMATS of the explanation: JSon, CSV, DataTable
            eval_results_explanation.add_json_format()
            eval_results_explanation.add_csv_format()
            eval_results_explanation.add_datatable_format()
            explanations.append(eval_results_explanation)

        # EXPLANATION: heatmap leaderboard
        heatmap_explanation = e10s.LlmHeatmapLeaderboardExplanation.from_eval_results(
            evaluator=self,
            eval_results=eval_results,
            metrics_meta=self._metrics_meta,
            key_2_evaluated_model=key_2_evaluated_model,
            display_name=f"{self._display_name} leaderboard",
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
        heatmap_explanation.add_json_format(
            threshold=self.args.get(
                evaluators.Evaluator.PARAM_METRIC_THRESHOLD,
                self._metrics_meta.get_primary_metric().threshold,
            )
        )
        heatmap_explanation.add_markdown_format(sort_by_metric_id=sort_by_metric)
        heatmap_explanation.add_evalstudio_markdown_format(
            sort_by_metric_id=self._metrics_meta.get_primary_metric().key
        )
        explanations.append(heatmap_explanation)

        # PROBLEMS for alerts and actionability
        self._diagnose_problems(
            eval_results=eval_results,
            key_2_evaluated_model=key_2_evaluated_model,
            leaderboard_explanation=heatmap_explanation,
        )

        # INSIGHTS
        self._diagnose_insights(
            leaderboard_explanation=heatmap_explanation,
        )

        # EXPLANATION: HTML fragment
        if self.config and self.config.create_html_representations:
            try:
                html_explanation = e10s.GlobalHtmlFragmentExplanation(
                    evaluator=self,
                    display_name=f"{self._display_name} leaderboard as HTML",
                    display_category=e10s.Explanation.DISPLAY_CAT_COMPLIANCE,
                )
                html_explanation.add_html_format(
                    str(
                        heatmap_explanation.as_html(
                            sort_by_metric_id=sort_by_metric,
                        )
                    )
                )
                explanations.append(html_explanation)
            except Exception as ex:
                self.logger.warning(
                    f"{self.log_name}: HTML fragment explanation creation failed: "
                    f"{ex}\n{traceback.format_exc()}"
                )

        return explanations

    def _diagnose_problems(
        self,
        eval_results: datasets.LlmEvalResults,
        key_2_evaluated_model: dict,
        leaderboard_explanation: e10s.LlmHeatmapLeaderboardExplanation,
    ):
        # low test case count
        self._diagnose_low_test_case_problem(
            eval_results=eval_results,
            models=self.models,
            test_case_minimum=self.args.get(evaluators.Evaluator.PARAM_MIN_TEST_CASES),
        )

        # threshold failures
        problems.problems_for_heat_leaderboard(
            evaluator=self,
            leaderboard=leaderboard_explanation,
            metric_threshold=self.args.get(
                evaluators.Evaluator.PARAM_METRIC_THRESHOLD,
                self._metrics_meta.get_primary_metric().threshold,
            ),
            primary_metric_meta=self._metrics_meta.get_primary_metric(),
            problem_type="consistency",
            problem_code=problems.AVIDProblemCode.P0200_MODEL,
            actions_description=(
                "Consistency in RAG/LLM actual answers can be primarily achieved by "
                "reducing the randomness in the generation process. Adjust "
                "the system prompt so that it provides clear instructions on role, "
                "output format and required behavior, while tuning decoding parameters "
                "directly controls the output's predictability. The most critical "
                "parameter is temperature, where a low value (e.g., 0.0 to 0.3) should "
                "be used for consistency-critical tasks to make the model more "
                "deterministic by sharpening the probability distribution, thus "
                "favoring the most likely tokens. Pair it with constrained sampling "
                "methods like low top_p` and top_k, to ensure that for the same "
                "input, the model will repeatedly generate the most probable and "
                "therefore consistent answers."
            ),
            explanation_type=e10s.GlobalHtmlFragmentExplanation.explanation_type(),
            explanation_name=e10s.GlobalHtmlFragmentExplanation.__name__,
            explanation_mime=f5s.HtmlFormat.mime,
        )

    def _diagnose_insights(
        self, leaderboard_explanation: e10s.LlmHeatmapLeaderboardExplanation
    ):
        t_html_fragment = e10s.GlobalHtmlFragmentExplanation

        leaderboard_explanation.get_insights(
            metrics_meta=self._metrics_meta,
            metric_name_protection=True,
            extra_description_best=(
                "This model produces responses that are the most consistent based on "
                "the ROUGE metric, which is typically used to measure the quality "
                "of machine-generated text."
            ),
            insight_type="consistency",
            explanation_type=t_html_fragment.explanation_type(),
            explanation_name=t_html_fragment.__name__,
            explanation_mime=f5s.HtmlFormat.mime,
        )

    def get_result(
        self,
    ) -> results.LeaderboardResult:
        return results.LeaderboardResult(
            persistence=self.persistence,
            explainer_id=self.explainer_id(),
            h2o_sonar_config=self.config,
            logger=self.logger,
            explanation=e10s.LlmHeatmapLeaderboardExplanation,
            explanation_format=f5s.CustomJsonFormat,
        )
