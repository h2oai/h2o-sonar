# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
import traceback

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import datasets
from h2o_sonar.lib.api import evaluators
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import problems
from h2o_sonar.lib.api import results
from h2o_sonar.utils import caching
from h2o_sonar.utils import progress as progress_utils


try:
    import nltk

    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False


def _plural(num, plural="s", singular=""):
    if num > 1:
        return plural
    return singular


class LoopingDetectionEvaluator(evaluators.Evaluator):
    _modules_needed_by_name = [h2o_sonar_config.DEP_NLTK]

    _display_name = "Looping detection"
    _tagline = "Detect looping in the generated answers."

    METRIC_UNIQUE_SENTENCES = "unique_sentences"
    METRIC_LONGEST_REPEATED_SUBSTRING = "longest_repeated_substring"
    METRIC_COMPRESSION_RATIO = "compression_ratio"
    _metrics_meta = commons.MetricsMeta(
        metrics=[
            commons.MetricMeta(
                key=METRIC_UNIQUE_SENTENCES,
                display_name="Unique Sentences",
                data_type="int",
                display_format=",d",  # int like 123,456
                description=(
                    "Unique sentences metric is a ratio "
                    "`number of unique sequences / number of all sentences`, "
                    "where sentences shorter than 10 characters are omitted."
                ),
                higher_is_better=True,
                threshold=evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
                is_primary_metric=True,
            ),
            commons.MetricMeta(
                key=METRIC_LONGEST_REPEATED_SUBSTRING,
                display_name="Longest Repeated Substring",
                data_type="int",
                display_format=",d",  # int like 123,456
                description=(
                    "Longest repeated substring metric is a ratio "
                    "`longest repeated substring * frequency of this substring / "
                    "length of the text`."
                ),
                higher_is_better=False,
                threshold=evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
                is_primary_metric=False,
            ),
            commons.MetricMeta(
                key=METRIC_COMPRESSION_RATIO,
                display_name="Compression Ratio",
                description=(
                    "Ratio `length in bytes of compressed string / length "
                    "in bytes of original string`. Compression is done using "
                    "python's zlib and using maximum compression level (9)."
                ),
                higher_is_better=True,
                threshold=evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
                is_primary_metric=False,
            ),
        ]
    )

    # COMPATIBILITY: LLM/RAG evaluation
    _rag = True
    _llm = True
    # GLOBAL: metric value for all dataset rows
    _global_explanation = True
    # LOCAL: metric value for particular row
    _local_explanation = True
    _explanation_types = [
        e10s.LlmEvalResultsExplanation,
        e10s.LlmHeatmapLeaderboardExplanation,
        e10s.WorkDirArchiveExplanation,
    ]

    _parameters = [
        evaluators.Evaluator._get_custom_param_metric_threshold(
            _metrics_meta.get_primary_metric()
        ),
        evaluators.Evaluator._PARAM_SAVE_LLM_RESULT,
    ]

    _keywords = [
        evaluators.KEYWORD_LLM,
        evaluators.KEYWORD_EVALUATES_LLM,
        evaluators.KEYWORD_EVALUATES_RAG,
        evaluators.KEYWORD_RQ_AA,
        evaluators.KEYWORD_ES_GENERATE,
    ]

    _brief_description = """Looping detection evaluator tries to find out whether the
    LLM generation went into a loop."""
    _description = evaluators.Evaluator._description_builder(
        brief=f"""{_brief_description}

**Method**:

* This evaluator provides three metrics:

```math
                     number of unique sequences
unique sentences =  ----------------------------
                      number of all sentences

                              longest repeated substring * frequency of this substring
longest repeated substring = ----------------------------------------------------------
                                               length of the text

                     length in bytes of compressed string
compression ratio = --------------------------------------
                      length in bytes of original string
```

Where:

- `unique sentences` omits sentences shorter than 10 characters.
- `compression ratio` is calculated using python's ``zlib`` and using maximum
  compression level (9).
""",
        metrics_meta=_metrics_meta,
        keywords=_keywords,
        parameters=_parameters,
        leaderboard_type=e10s.LlmHeatmapLeaderboardExplanation.explanation_type(),
    )

    COL_INPUT = datasets.LlmDataset.KEY_INPUT
    COL_CONTEXT = datasets.LlmDataset.KEY_CONTEXT
    COL_EXPECTED_OUTPUT = datasets.LlmDataset.KEY_EXPECTED_OUTPUT
    COL_ACTUAL_OUTPUT = datasets.LlmDataset.KEY_ACTUAL_OUTPUT
    COL_MODEL = "model"
    COL_SCORE = "score"

    def __init__(self):
        evaluators.Evaluator.__init__(self)

        self.args = None
        self.problems = []
        self.log_name = "Looping Detection"
        self._embedding_model = None

    def check_compatibility(
        self,
        params: commons.CommonInterpretationParams | None = None,
        **evaluator_params,
    ) -> bool:
        evaluators.Evaluator.check_compatibility(self, params, **evaluator_params)

        if not HAS_NLTK:
            self.logger.error(self._check_compatibility_pckg_err_msg(["nltk"]))
            return False

        if not evaluators.Evaluator._check_llm_dataset_compatibility(
            self,
            params=params,
            evaluator_keywords=self.keywords,
            check_empty_contexts=evaluators.KEYWORD_RQ_RC in self.keywords,
            fail_on_all_empty_contexts=evaluators.KEYWORD_RQ_RC in self.keywords,
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

        caching.cache_nltk_punkt()

    def evaluate(self, llm_testset, explanations_types=None, **kwargs) -> list:
        save_llm_result = self.args.get(
            evaluators.Evaluator.PARAM_SAVE_LLM_RESULT,
            evaluators.Evaluator.DEFAULT_SAVE_LLM_RESULT,
        )

        return self._evaluate(
            llm_testset=llm_testset,
            save_llm_result=save_llm_result,
        )

    def _evaluate(
        self,
        llm_testset,
        save_llm_result: bool,
    ) -> list:
        #
        # EVALUATION
        #

        key_2_evaluated_model = {m.key: m for m in self.models}

        # LLM host: RAG or service
        llm_host = (
            commons.LlmModelHostType.RAG
            if isinstance(
                next(iter(key_2_evaluated_model.values())), models.ExplainableRagModel
            )
            else commons.LlmModelHostType.SERVICE
        )

        eval_results = self._calculate_metrics(llm_testset=llm_testset)

        #
        # NORMALIZATION of the evaluation results to the common EXPLANATIONS/FORMAT(s)
        #

        # EXPLANATIONS
        explanations = []

        # EXPLANATION: all data (per prompt metrics)
        if save_llm_result:
            eval_results_explanation = e10s.LlmEvalResultsExplanation(
                evaluator=self,
                display_name="Evaluation metrics data",
                display_category=e10s.Explanation.DISPLAY_CAT_LLM,
                eval_results=eval_results,
            )
            # FORMATS of the explanation: JSon, CSV, DataTable
            eval_results_explanation.add_json_format()
            eval_results_explanation.add_csv_format()
            eval_results_explanation.add_datatable_format()
            explanations.append(eval_results_explanation)

        # THRESHOLD for the metric
        metrics_threshold = self.args.get(
            evaluators.Evaluator.PARAM_METRIC_THRESHOLD,
            evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
        )
        self._metrics_meta.set_threshold(metrics_threshold)

        # EXPLANATION: heatmap leaderboard
        heatmap_explanation = e10s.LlmHeatmapLeaderboardExplanation.from_eval_results(
            evaluator=self,
            eval_results=eval_results,
            metrics_meta=self._metrics_meta,
            key_2_evaluated_model=key_2_evaluated_model,
            llm_host=llm_host,
            display_name="LLM heatmap leaderboard",
            display_category=e10s.GlobalSummaryFeatImpExplanation.DISPLAY_CAT_LLM,
            logger=self.logger,
        )
        heatmap_explanation.add_json_format(
            threshold=self.args.get(
                evaluators.Evaluator.PARAM_METRIC_THRESHOLD,
                self._metrics_meta.get_primary_metric().threshold,
            )
        )
        heatmap_explanation.add_markdown_format(
            sort_by_metric_id=self.METRIC_UNIQUE_SENTENCES
        )
        heatmap_explanation.add_evalstudio_markdown_format(
            sort_by_metric_id=self.METRIC_UNIQUE_SENTENCES
        )
        explanations.append(heatmap_explanation)

        # EXPLANATION: HTML fragment
        if self.config and self.config.create_html_representations:
            try:
                html_explanation = e10s.GlobalHtmlFragmentExplanation(
                    evaluator=self,
                    display_name="LLM heatmap leaderboard as HTML",
                    display_category=e10s.Explanation.DISPLAY_CAT_COMPLIANCE,
                )
                html_explanation.add_html_format(
                    str(
                        heatmap_explanation.as_html(
                            sort_by_metric_id=self.METRIC_UNIQUE_SENTENCES
                        )
                    ),
                )
                explanations.append(html_explanation)
            except Exception as ex:
                self.logger.warning(
                    f"{self.log_name}: HTML fragment explanation creation failed: "
                    f"{ex}\n{traceback.format_exc()}"
                )

        # EXPLANATION: ZIP archive of all artifacts created by the evaluator
        explanations.append(
            self.create_explanation_workdir_archive(
                display_name=f"Archive of {self._display_name} artifacts",
                display_category=e10s.Explanation.DISPLAY_CAT_COMPLIANCE,
            )
        )

        return explanations

    # split docs into sentences
    @staticmethod
    def split_sentences(text: str) -> list[str]:
        """Split the data into sentences"""
        if not HAS_NLTK:
            commons.raise_opt_import_err("nltk")

        sentences = nltk.tokenize.sent_tokenize(text)
        return [s for s in sentences if len(s) >= 10]

    def _calculate_unique_sentences(self, row):
        from collections import Counter

        output_sentences = self.split_sentences(row.actual_output)
        cnt = Counter(output_sentences)
        unique_sentences = len(cnt.keys())
        if unique_sentences <= 1:
            return 0
        sentence, repetitions = cnt.most_common(1)[0]
        if repetitions > 1:
            explanation_type = e10s.GlobalHtmlFragmentExplanation.explanation_type()
            description = (
                f"The most common sentence has {repetitions} repetitions. "
                f'The sentence is: "{sentence}"'
            )
            self.add_problem(
                problems.ProblemAndAction(
                    description=description,
                    evaluator_id=self.evaluator_id(),
                    problem_attrs={
                        problems.ProblemAndAction.ATTR_ROW_KEYS: [
                            (row.key, row.model_key)
                        ],
                        # input dataset ~ test lab ~ key is the test case key
                        problems.ProblemAndAction.ATTR_TEST_CASE_KEYS: [row.key],
                        problems.ProblemAndAction.ATTR_EVALUATOR_NAME: (
                            self._display_name
                        ),
                    },
                    evaluator_name=self._display_name,
                    explanation_type=explanation_type,
                    explanation_name=e10s.GlobalHtmlFragmentExplanation.__name__,
                    explanation_mime=f5s.HtmlFormat.mime,
                    resources=[],
                )
            )
        return unique_sentences / len(output_sentences)

    def _calculate_longest_repeated_substring(self, row):
        s = row.actual_output

        n = len(s)
        longest = ""

        for i in range(n):
            for j in range(i + len(longest), n):
                k = 0
                while i + k < j and j + k < n and s[i + k] == s[j + k]:
                    k += 1
                if k > len(longest):
                    longest = s[i : i + k]

        max_length = len(longest)
        if max_length > 0:
            repetitions = s.count(longest)
            metric = repetitions * max_length / len(s)
            metrics_threshold = self.args.get(
                evaluators.Evaluator.PARAM_METRIC_THRESHOLD,
                evaluators.Evaluator.DEFAULT_METRIC_THRESHOLD,
            )
            if metric > metrics_threshold:
                explanation_type = e10s.GlobalHtmlFragmentExplanation.explanation_type()
                description = (
                    f"There {_plural(repetitions, 'are', 'is')} "
                    f"{repetitions} instance{_plural(repetitions)} of the following "
                    f'text: "{longest}"'
                )
                self.add_problem(
                    problems.ProblemAndAction(
                        description=description,
                        evaluator_id=self.evaluator_id(),
                        problem_attrs={
                            problems.ProblemAndAction.ATTR_ROW_KEYS: [
                                (row.key, row.model_key)
                            ],
                            # input dataset ~ test lab ~ key is the test case key
                            problems.ProblemAndAction.ATTR_TEST_CASE_KEYS: [row.key],
                            problems.ProblemAndAction.ATTR_EVALUATOR_NAME: (
                                self._display_name
                            ),
                        },
                        evaluator_name=self._display_name,
                        explanation_type=explanation_type,
                        explanation_name=e10s.GlobalHtmlFragmentExplanation.__name__,
                        explanation_mime=f5s.HtmlFormat.mime,
                        resources=[],
                    )
                )

            return metric
        else:
            return 0

    @staticmethod
    def _calculate_compression_ratio(row):
        import zlib

        encoded_output = row.actual_output.encode("utf-8")
        compressed = zlib.compress(encoded_output, level=9)
        return min(1.0, len(compressed) / len(encoded_output))

    def _calculate_metrics(self, llm_testset):
        llm_dataset = datasets.LlmDataset.from_datatable_dict(llm_testset.to_dict())

        # evaluator runs only ONE metric at a time
        self.report_progress(0.01, "Configuring metrics...")

        eval_results = datasets.LlmEvalResults()
        # for every test case run metric (row by row)
        for e, r in enumerate(llm_dataset.inputs):
            # progress
            self.report_progress(
                progress=progress_utils.ProgressCallbackContext.progress_for_steps(
                    3 * e + 1, 3 * len(llm_dataset.inputs)
                ),
                message=(
                    f"Building, configuring and running "
                    f"'{self.METRIC_UNIQUE_SENTENCES}' "
                    f"evaluation for input {e + 1}/{len(llm_dataset.inputs)} "
                ),
            )

            # handle actual answer retrieval error ~ RAG/LLM client crash
            if evaluators.Evaluator._is_internal_err_answer(r.actual_output):
                # set WORST metrics values
                eval_results.add_result(
                    datasets.LlmEvalResults.LlmEvalResultRow(
                        dataset_row=r,
                        metrics={
                            self.METRIC_UNIQUE_SENTENCES: 0.0,
                            self.METRIC_LONGEST_REPEATED_SUBSTRING: 1.0,
                            self.METRIC_COMPRESSION_RATIO: 0.0,
                        },
                    )
                )
                continue

            # handle empty or invalid actual_output at runtime
            if not r.actual_output or not isinstance(r.actual_output, str):
                description = (
                    f"Empty or invalid actual output detected in row "
                    f"{e + 1}. Looping detection evaluator requires actual "
                    "output to be a non-empty string."
                )
                self.logger.warning(description)

                self.add_problem_for_row(
                    eval_row=r,
                    description=description,
                    evaluator_id=self.evaluator_id(),
                    evaluator_name=self._display_name,
                    severity=problems.ProblemSeverity.low,
                    explanation_type=(
                        e10s.GlobalHtmlFragmentExplanation.explanation_type()
                    ),
                    explanation_name=e10s.GlobalHtmlFragmentExplanation.__name__,
                    explanation_mime=f5s.HtmlFormat.mime,
                )

                # set WORST metrics values and continue
                eval_results.add_result(
                    datasets.LlmEvalResults.LlmEvalResultRow(
                        dataset_row=r,
                        metrics={
                            self.METRIC_UNIQUE_SENTENCES: 0.0,
                            self.METRIC_LONGEST_REPEATED_SUBSTRING: 1.0,
                            self.METRIC_COMPRESSION_RATIO: 0.0,
                        },
                    )
                )
                continue

            uniq = self._calculate_unique_sentences(r)

            self.report_progress(
                progress=progress_utils.ProgressCallbackContext.progress_for_steps(
                    3 * e + 2, 3 * len(llm_dataset.inputs)
                ),
                message=(
                    f"Building, configuring and running "
                    f"'{self.METRIC_LONGEST_REPEATED_SUBSTRING}' "
                    f"evaluation for input {e + 1}/{len(llm_dataset.inputs)} "
                ),
            )
            longest = self._calculate_longest_repeated_substring(r)

            self.report_progress(
                progress=progress_utils.ProgressCallbackContext.progress_for_steps(
                    3 * e + 3, 3 * len(llm_dataset.inputs)
                ),
                message=(
                    f"Building, configuring and running "
                    f"'{self.METRIC_COMPRESSION_RATIO}' "
                    f"evaluation for input {e + 1}/{len(llm_dataset.inputs)} "
                ),
            )
            compression_ratio = LoopingDetectionEvaluator._calculate_compression_ratio(
                r
            )
            eval_results.add_result(
                datasets.LlmEvalResults.LlmEvalResultRow(
                    dataset_row=r,
                    metrics={
                        self.METRIC_UNIQUE_SENTENCES: uniq,
                        self.METRIC_LONGEST_REPEATED_SUBSTRING: longest,
                        self.METRIC_COMPRESSION_RATIO: compression_ratio,
                    },
                )
            )

        return eval_results

    def get_result(
        self,
    ) -> results.LeaderboardResult:
        return results.LeaderboardResult(
            persistence=self.persistence,
            explainer_id=LoopingDetectionEvaluator.explainer_id(),
            h2o_sonar_config=self.config,
            logger=self.logger,
            explanation=e10s.LlmHeatmapLeaderboardExplanation,
            explanation_format=f5s.CustomJsonFormat,
        )
