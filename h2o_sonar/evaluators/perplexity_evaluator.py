# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
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
from h2o_sonar.utils import resource_mgmt


try:
    import lmppl
    import torch
    import transformers

    HAS_REQUIRED_PACKAGES = True
except ImportError:
    HAS_REQUIRED_PACKAGES = False


class PerplexityEvaluator(evaluators.Evaluator):
    _display_name = "Perplexity"
    _tagline = "Evaluate coherence, fluency, and certainty of actual answers."

    METRIC_PERPLEXITY = "perplexity"

    # models used by the evaluator
    _e_model_bert = caching.MODEL_DISTILGPT2

    _metrics_meta = commons.MetricsMeta(
        metrics=[
            commons.MetricMeta(
                key=METRIC_PERPLEXITY,
                display_name="Perplexity",
                description=(
                    f"Perplexity measures how well a model predicts the next word "
                    f"based on what came before (sliding window). The lower the "
                    f"perplexity score, the better the model is at predicting "
                    f"the next word. "
                    f"Perplexity is calculated as exp(mean(-log likelihood)), "
                    f"where log likelihood is computed using the "
                    f"'{_e_model_bert}' language model as probability of "
                    f"predicting the next word."
                ),
                higher_is_better=False,
                value_range=(0, float("inf")),
                is_primary_metric=True,
            )
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

    _parameters = [
        evaluators.EvaluatorParam(
            param_name=evaluators.Evaluator.PARAM_METRIC_THRESHOLD,
            description=(
                "Evaluated metric threshold - values above/below this threshold are "
                "considered problematic."
            ),
            param_type=commons.EvaluatorParamType.float,
            default_value=float("inf"),
            src=evaluators.EvaluatorParam.SRC_EVALUATOR_PARAMS,
        ),
        evaluators.Evaluator._PARAM_SAVE_LLM_RESULT,
        evaluators.Evaluator._get_custom_param_min_test_case(),
    ]

    _keywords = [
        evaluators.KEYWORD_GPU_OPT,
        evaluators.KEYWORD_LLM,
        evaluators.KEYWORD_EVALUATES_LLM,
        evaluators.KEYWORD_EVALUATES_RAG,
        evaluators.KEYWORD_RQ_AA,
        evaluators.KEYWORD_SR_11_7_CS,
        evaluators.KEYWORD_SR_11_7_OA,
        evaluators.KEYWORD_NIST_AI_RMF_S,
        evaluators.KEYWORD_PROBLEM_TYPE_IR,
        evaluators.KEYWORD_PROBLEM_TYPE_QA,
        evaluators.KEYWORD_ES_GENERATE,
        evaluators.KEYWORD_METHOD_TYPE_DETERMINISTIC,
        evaluators.KEYWORD_METHOD_NLI,
    ]

    _modules_needed_by_name = ["lmppl==0.0.1"]

    _brief_description = f"""Perplexity measures how well a model
**predicts the next word** based on what came before. The lower the perplexity score,
the better the model is at predicting the next word.

Perplexity can be interpreted as **the average number of choices** a model has to
consider when predicting the next word.

A **lower** perplexity indicates that the model is **more certain** about its
predictions. In comparison, higher perplexity suggests the model is more uncertain.
Perplexity is a crucial metric for evaluating the performance of language models in
tasks like machine translation, speech recognition, and text generation.

- Evaluator uses [{_e_model_bert}](https://huggingface.co/{_e_model_bert}) language
  model to calculate perplexity of the actual answer using
  [lmppl](https://github.com/asahi417/lmppl) package.
- Compatibility: RAG and LLM models."""
    _description = evaluators.Evaluator._description_builder(
        brief=f"""{_brief_description}

**Method**:

- Evaluator utilizes [{_e_model_bert}](https://huggingface.co/{_e_model_bert})
  language model to calculate perplexity of the actual answer using
  [lmppl](https://github.com/asahi417/lmppl) library. The calculation is as follows:

```math
perplexity = exp(mean(cross-entropy loss))
```

- Where:
    - `cross-entropy loss` is a measure of the **difference** between the predicted
       probability distribution of the next token and the true probability distribution
       of [{_e_model_bert}](https://huggingface.co/{_e_model_bert}) calculated on
       the actual answer.
    - `mean()` is the average cross-entropy loss over all the words in a sequence.
    - `exp()` is the exponential function which takes the mean cross-entropy loss
      as an input and returns a value that represents the perplexity.

See also:

- 3rd party library used: https://github.com/asahi417/lmppl
- 3rd party model used: https://huggingface.co/{_e_model_bert}""",
        metrics_meta=_metrics_meta,
        keywords=_keywords,
        parameters=_parameters,
        leaderboard_type=e10s.LlmHeatmapLeaderboardExplanation.explanation_type(),
    )

    def __init__(self):
        evaluators.Evaluator.__init__(self)

        self.args = None
        self.problems = []
        self.log_name = "Perplexity evaluator"
        self._scorer_constructor = None

    def check_compatibility(
        self,
        params: commons.CommonInterpretationParams | None = None,
        **evaluator_params,
    ) -> bool:
        evaluators.Evaluator.check_compatibility(self, params, **evaluator_params)

        if not HAS_REQUIRED_PACKAGES:
            self.logger.warning(
                self._check_compatibility_pckg_err_msg(
                    ["lmppl", "torch", "transformers"]
                )
            )
            return False

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

        self.log_name = f"Perplexity evaluator {self.mli_key}/{self.key}"

        logging = self.logger

        # The following class is basically copy-paste lmppl.LM __init__
        # with modifications to allow usage of particular revision
        # and non-interactively trust_remote_code
        class PatchedLM(lmppl.LM):
            """Language Model."""

            def __init__(  # noqa
                self,
                model: str = PerplexityEvaluator._e_model_bert,
                use_auth_token: bool = False,
                max_length: int | None = None,
                device: str | None = None,
                num_gpus: int | None = None,
            ):
                """Language Model.

                @param model: Model alias or path to local model file.
                @param use_auth_token: HF transformers argument of `use_auth_token`
                @param device: Device name to load the models.
                @param num_gpus: Number of gpus to be used.
                """
                logging.info(f"Loading model '{model}' on device '{device}' ...")

                # load model
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model,
                    use_auth_token=use_auth_token,
                    revision=caching.REVISIONS_FOR_MODEL.get(model, "main"),
                    trust_remote_code=True,
                )
                self.config = transformers.AutoConfig.from_pretrained(
                    model,
                    use_auth_token=use_auth_token,
                    revision=caching.REVISIONS_FOR_MODEL.get(model, "main"),
                    trust_remote_code=True,
                )
                self.model = transformers.AutoModelForCausalLM.from_pretrained(
                    model,
                    config=self.config,
                    use_auth_token=use_auth_token,
                    revision=caching.REVISIONS_FOR_MODEL.get(model, "main"),
                    trust_remote_code=True,
                )
                self.max_length = (
                    max_length
                    if max_length is not None
                    else self.tokenizer.model_max_length
                )
                self.tokenizer.pad_token = self.tokenizer.eos_token
                assert self.max_length <= self.tokenizer.model_max_length, (
                    f"{self.max_length} > {self.tokenizer.model_max_length}"
                )

                # loss function
                self.loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

                # GPU setup
                if device is None:
                    self.device = "cuda" if torch.cuda.device_count() > 0 else "cpu"
                else:
                    self.device = device

                logging.info(f"Loaded model '{model}' on device '{self.device}' ...")

                if "cuda" in self.device.lower():
                    num_gpus = (
                        torch.cuda.device_count() if num_gpus is None else num_gpus
                    )
                    if num_gpus > 1:
                        self.parallel = True
                        self.model = torch.nn.DataParallel(self.model)
                    self.model.to(self.device)
                self.model.eval()
                logging.info(f"\t * Num of GPU in use: {torch.cuda.device_count()}")

        self._scorer_constructor = lambda: PatchedLM(
            PerplexityEvaluator._e_model_bert,
            device=h2o_sonar_config.config.resolve_gpu_cpu_device(
                result_format="str",
            ),
        )

    def evaluate(self, llm_testset, **kwargs) -> list:
        save_llm_result = self.args.get(
            evaluators.Evaluator.PARAM_SAVE_LLM_RESULT,
            evaluators.Evaluator.DEFAULT_SAVE_LLM_RESULT,
        )

        metrics_threshold = self.args.get(
            evaluators.Evaluator.PARAM_METRIC_THRESHOLD,
            float("inf"),
        )
        self._metrics_meta.set_threshold(metrics_threshold)

        # RAG models: key -> model
        key_2_evaluated_model = {m.key: m for m in self.models}

        llm_dataset = datasets.LlmDataset.from_datatable_dict(llm_testset.to_dict())

        eval_results = datasets.LlmEvalResults()

        raw_scorer = self._scorer_constructor()
        with resource_mgmt.PytorchModelLifeCycleManager(raw_scorer) as scorer:
            for e, r in enumerate(llm_dataset.inputs):
                self.report_progress(
                    progress=progress_utils.ProgressCallbackContext.progress_for_steps(
                        e + 1, len(llm_dataset.inputs)
                    ),
                    message=evaluators.Evaluator._eval_row_progress_msg(
                        metric_name=self.METRIC_PERPLEXITY,
                        device=scorer.device if scorer else "cpu",
                        row=e + 1,
                        total_rows=len(llm_dataset.inputs),
                    ),
                )

                # handle actual answer retrieval error ~ RAG/LLM client crash
                if evaluators.Evaluator._is_internal_err_answer(r.actual_output):
                    # set WORST metrics values
                    eval_results.add_result(
                        datasets.LlmEvalResults.LlmEvalResultRow(
                            dataset_row=r,
                            metrics={
                                # avoid INF, set "effective infinity" score instead
                                self.METRIC_PERPLEXITY: (
                                    commons.MetricMeta.EFFECTIVE_INF_FLOAT
                                ),
                            },
                        )
                    )
                    continue

                # handle empty or invalid actual_output at runtime
                if not r.actual_output or not isinstance(r.actual_output, str):
                    description = (
                        f"Empty or invalid actual output detected in row "
                        f"{e + 1}. Perplexity evaluator requires actual "
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
                                # avoid INF, set "effective infinity" score instead
                                self.METRIC_PERPLEXITY: (
                                    commons.MetricMeta.EFFECTIVE_INF_FLOAT
                                ),
                            },
                        )
                    )
                    continue

                perplexity = scorer.get_perplexity(r.actual_output)
                # add to result
                eval_results.add_result(
                    datasets.LlmEvalResults.LlmEvalResultRow(
                        dataset_row=r,
                        metrics={
                            self.METRIC_PERPLEXITY: perplexity,
                        },
                    )
                )

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
                display_name="Perplexity evaluation results",
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
        self._diagnose_insights(leaderboard_explanation=heatmap_explanation)

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
        # perturbation problems
        self._diagnose_perturbation_problems(
            eval_results=eval_results, key_2_evaluated_model=key_2_evaluated_model
        )

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
            problem_type="accuracy",
            problem_code=problems.AVIDProblemCode.P0200_MODEL,
            actions_description=(
                "While perplexity is a common metric, it doesn't directly guarantee "
                "meaningful answers. To improve perplexity score focus on training "
                "Data - prioritize high-quality training data that includes diverse "
                "responses and factual information. This broadens the LLM's "
                "understanding and ability to generate informative answers. "
                "Also train the LLM on tasks that require informative responses, "
                "such as question answering or summarization. This helps the model "
                "understand the importance of providing relevant and comprehensive "
                "information. During training, incorporate reward mechanisms that "
                "favor informative outputs over surprising but irrelevant ones "
                "(reward). This encourages the LLM to prioritize generating meaningful "
                "content. Also explore ways to integrate external knowledge sources "
                "during generation. This can provide the LLM with additional context "
                "to support its answers with factual information."
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
            extra_description_best=(
                "This model produces responses that most closely resemble the expected "
                "responses based on the perplexity metric which is a measure of the "
                "confidence model has in its next token predictions. Lower perplexity "
                "indicates that the model is more certain about its predictions. "
                "Perplexity is a important metric for evaluating the performance of "
                "RAGs and LLMs in tasks like text generation or translation."
            ),
            insight_type="accuracy",
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
