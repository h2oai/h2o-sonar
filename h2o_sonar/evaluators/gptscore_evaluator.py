# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import abc
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
    import torch
    from transformers import AutoModelForSeq2SeqLM
    from transformers import AutoTokenizer
    from transformers import GPT2LMHeadModel
    from transformers import GPT2Tokenizer
    from transformers import GPTJForCausalLM
    from transformers import OPTForCausalLM

    HAS_TORCH_AND_TRANSFORMERS = True
except ImportError:
    HAS_TORCH_AND_TRANSFORMERS = False


class OPTScorer:
    def __init__(self, device: str, model: str):
        if not HAS_TORCH_AND_TRANSFORMERS:
            raise ImportError(
                "The 'torch' or 'transformers' package is required, but not installed."
            )

        self._device = device
        self._max_length = 2048
        self._tokenizer = GPT2Tokenizer.from_pretrained(
            model,
            trust_remote_code=True,
            revision=caching.REVISIONS_FOR_MODEL.get(model, "main"),
        )
        if "gpt2" in model:
            self._model = GPT2LMHeadModel.from_pretrained(
                model,
                # trust_remote_code=True, # ignored
                revision=caching.REVISIONS_FOR_MODEL.get(model, "main"),
            ).to(self._device)
            self._max_length = 1024
        elif "gpt-j" in model:
            self._model = GPTJForCausalLM.from_pretrained(
                model,
                # trust_remote_code=True, # ignored
                revision=caching.REVISIONS_FOR_MODEL.get(model, "main"),
            ).to(self._device)
        else:
            self._model = OPTForCausalLM.from_pretrained(
                model,
                # trust_remote_code=True, # ignored
                revision=caching.REVISIONS_FOR_MODEL.get(model, "main"),
            ).to(self._device)

        self._model.eval()

    def score(
        self, inputs: list[str], predictions: list[str]
    ) -> tuple[list[float], bool]:
        scores = []
        ignore_index = torch.nn.CrossEntropyLoss().ignore_index
        too_long = False
        for i, (in_, out) in enumerate(zip(inputs, predictions, strict=False)):
            tok_in = self._tokenizer.encode(in_)
            tok_out = self._tokenizer.encode(out)
            if "facebook/opt" in self._tokenizer.name_or_path:
                tok_out = tok_out[1:]  # remove bos token
            total_len = len(tok_in) + len(tok_out)
            if total_len > self._max_length:
                tok_in = tok_in[: -total_len - self._max_length - 1]
                too_long = True

            tok_in.extend(tok_out)
            labels = [ignore_index] * len(tok_in)
            labels[-len(tok_out) :] = tok_out
            labels = torch.LongTensor(labels).unsqueeze(0).to(self._device)
            with torch.no_grad():
                outputs = self._model(
                    input_ids=torch.LongTensor(tok_in).unsqueeze(0).to(self._device),
                    labels=labels,
                    output_hidden_states=False,
                )
            scores.append(-outputs[0].item())
        return scores, too_long


class FLANScorer:
    def __init__(self, device: str, model: str):
        if not HAS_TORCH_AND_TRANSFORMERS:
            raise ImportError(
                "The 'torch' or 'transformers' package is required, but not installed."
            )

        # set up model
        self._device = device
        self._max_length = 1024
        self._tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=True,
            revision=caching.REVISIONS_FOR_MODEL.get(model, "main"),
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            model,
            trust_remote_code=True,
            revision=caching.REVISIONS_FOR_MODEL.get(model, "main"),
        ).to(device)
        self._model.eval()
        # set up loss
        self._loss_fn = torch.nn.NLLLoss(
            reduction="none", ignore_index=self._model.config.pad_token_id
        )
        self._log_softmax = torch.nn.LogSoftmax(dim=1)

    def score(
        self, inputs: list[str], predictions: list[str]
    ) -> tuple[list[float], bool]:
        scores = []
        too_long = False
        for i, (in_, out) in enumerate(zip(inputs, predictions, strict=False)):
            with torch.no_grad():
                tok_in = self._tokenizer(
                    in_,
                    max_length=self._max_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    return_length=True,
                )
                tok_out = self._tokenizer(
                    out,
                    max_length=self._max_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    return_length=True,
                )

                total_len = float(tok_in["length"] + tok_out["length"])
                too_long |= total_len > self._max_length
                tok_in_in = tok_in["input_ids"].to(self._device)
                tok_in_att = tok_in["attention_mask"].to(self._device)
                tok_out_in = tok_out["input_ids"].to(self._device)
                tok_out_att = tok_out["attention_mask"].to(self._device)
                out_len = tok_out_att.sum(dim=1).to(self._device)

                output = self._model(
                    input_ids=tok_in_in,
                    attention_mask=tok_in_att,
                    labels=tok_out_in,
                )
                logits = output.logits.view(-1, self._model.config.vocab_size)
                loss = (
                    self._loss_fn(self._log_softmax(logits), tok_out_in.view(-1))
                    .view(tok_out_in.shape[0], -1)
                    .sum(dim=1)
                ) / out_len
                scores.extend([-x.item() for x in loss])

        return scores, too_long


def _get_scorer(device: str, model: str) -> OPTScorer | FLANScorer:
    if model in [
        caching.MODEL_GOOGLE_FLAN_T5_SMALL,
        caching.MODEL_GOOGLE_FLAN_T5_BASE,
        caching.MODEL_GOOGLE_FLAN_T5_LARGE,
        caching.MODEL_GOOGLE_FLAN_T5_XL,
        caching.MODEL_GOOGLE_FLAN_T5_XXL,
    ]:
        return FLANScorer(device=device, model=model)

    return OPTScorer(device=device, model=model)


class GptScoreEvaluator(abc.ABC, evaluators.Evaluator):
    _display_name = "GPTScore"
    _tagline = (
        "Assess generated answers for fluency, style, and grammatical correctness."
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

    _ASPECT_DEFINITIONS = dict()

    PARAM_EVAL_GPT_SCORE_MODEL = "gpt_score_model"
    DEFAULT_METRIC_THRESHOLD = float("inf")
    _ALLOWED_MODELS = [
        caching.MODEL_GOOGLE_FLAN_T5_SMALL,
        caching.MODEL_GOOGLE_FLAN_T5_BASE,
        caching.MODEL_GOOGLE_FLAN_T5_LARGE,
        caching.MODEL_GOOGLE_FLAN_T5_XL,
        caching.MODEL_GOOGLE_FLAN_T5_XXL,
        caching.MODEL_FACEBOOK_OPT_125M,
        caching.MODEL_FACEBOOK_OPT_350M,
        caching.MODEL_FACEBOOK_OPT_1_3B,
        caching.MODEL_FACEBOOK_OPT_2_7B,
        caching.MODEL_FACEBOOK_OPT_6_7B,
        caching.MODEL_FACEBOOK_OPT_13B,
        caching.MODEL_FACEBOOK_OPT_66B,
        caching.MODEL_GPT2_MEDIUM,
        caching.MODEL_GPT2_LARGE,
        caching.MODEL_GPT2_XL,
        caching.MODEL_ELEUTHERAI_GPT_J_6B,
    ]

    # models used by the evaluator
    _e_model_gpt = caching.MODEL_GPT2_MEDIUM

    _parameters = [
        evaluators.Evaluator._PARAM_SAVE_LLM_RESULT,
        evaluators.EvaluatorParam(
            param_name=PARAM_EVAL_GPT_SCORE_MODEL,
            description=(
                f"Model to use for GPTScore evaluation. Default model is "
                f'"{_e_model_gpt}".'
                "\nSupported models:" + ", ".join([f'"{m}"' for m in _ALLOWED_MODELS])
            ),
            param_type=commons.EvaluatorParamType.str,
            default_value=_e_model_gpt,
            src=evaluators.EvaluatorParam.SRC_EVALUATOR_PARAMS,
            predefined=_ALLOWED_MODELS,
        ),
        evaluators.Evaluator._get_custom_param_min_test_case(),
    ]

    _keywords = [
        evaluators.KEYWORD_GPU_OPT,
        evaluators.KEYWORD_LLM,
        evaluators.KEYWORD_EVALUATES_LLM,
        evaluators.KEYWORD_EVALUATES_RAG,
        evaluators.KEYWORD_PROBLEM_TYPE_SUM,
        evaluators.KEYWORD_METHOD_TYPE_DETERMINISTIC,
        evaluators.KEYWORD_METHOD_NLI,
    ]

    _modules_needed_by_name = ["transformers==4.38.2"]

    _brief_description = """GPT Score evaluator is based on a novel evaluation
framework specifically designed for RAGs and LLMs. It utilizes the inherent
abilities of LLMs, particularly their ability to understand and respond to
instructions, to assess the quality of generated text.

See also:

- Paper _"GPTScore: Evaluate as You Desire"_: https://arxiv.org/abs/2302.04166"""
    _description = evaluators.Evaluator._description_builder(
        brief=_brief_description,
        metrics_meta=commons.MetricsMeta(),
        keywords=_keywords,
        parameters=_parameters,
        leaderboard_type=e10s.LlmHeatmapLeaderboardExplanation.explanation_type(),
    )

    def __init__(self):
        evaluators.Evaluator.__init__(self)

        self.args = None
        self.problems = []
        self.log_name = f"{self._display_name} evaluator"

        configured_device = h2o_sonar_config.config.resolve_gpu_cpu_device(
            result_format="torch",
        )
        self._device = (
            h2o_sonar_config.H2oSonarConfig.VALUE_CPU
            if not configured_device
            else configured_device
        )

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

        self.log_name = f"{self._display_name} evaluator {self.mli_key}/{self.key}"

        assert self.args.get(self.PARAM_EVAL_GPT_SCORE_MODEL) in self._ALLOWED_MODELS

    def evaluate(self, llm_testset, **kwargs) -> list:
        save_llm_result = self.args.get(
            evaluators.Evaluator.PARAM_SAVE_LLM_RESULT,
            evaluators.Evaluator.DEFAULT_SAVE_LLM_RESULT,
        )

        metrics_threshold = self.args.get(
            evaluators.Evaluator.PARAM_METRIC_THRESHOLD,
            self.DEFAULT_METRIC_THRESHOLD,
        )
        self._metrics_meta.set_threshold(metrics_threshold)

        # RAG models: key -> model
        key_2_evaluated_model = {m.key: m for m in self.models}

        llm_dataset = datasets.LlmDataset.from_datatable_dict(llm_testset.to_dict())

        eval_results = datasets.LlmEvalResults()
        with resource_mgmt.PytorchModelLifeCycleManager(
            _get_scorer(
                device=self._device,
                model=self.args.get(self.PARAM_EVAL_GPT_SCORE_MODEL),
            )
        ) as scorer:
            for e, r in enumerate(llm_dataset.inputs):
                # handle actual answer retrieval error ~ RAG/LLM client crash
                if evaluators.Evaluator._is_internal_err_answer(r.actual_output):
                    # set WORST metrics values
                    eval_results.add_result(
                        datasets.LlmEvalResults.LlmEvalResultRow(
                            dataset_row=r,
                            metrics={k: 0.0 for k in self._ASPECT_DEFINITIONS.keys()},
                        )
                    )
                    continue

                # add to result
                eval_results.add_result(
                    datasets.LlmEvalResults.LlmEvalResultRow(
                        dataset_row=r,
                        metrics={
                            k: self._run(
                                scorer=scorer,
                                row=r,
                                aspect=k,
                                i=e + 1,
                                n=len(llm_dataset.inputs),
                                device=self._device,
                            )
                            for k in self._ASPECT_DEFINITIONS.keys()
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
                display_name="GPTScore evaluation results",
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
                "Try to indirectly improve GPTScore by focusing on two areas: training "
                "and evaluation. High-quality training data with well-labeled "
                "generations demonstrating desired qualities and curriculum learning "
                "techniques for gradual LLM challenge will enhance output quality. "
                "Additionally, crafting clear instructions and providing examples "
                "of high-quality RAG outputs during GPTScore evaluation can guide "
                "the LLM towards valuing aspects important for the task, ultimately "
                "leading to better GPTScore results."
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

    def _run(
        self,
        scorer,
        row: datasets.LlmDataset.LlmDatasetRow,
        aspect: str,
        i: int,
        n: int,
        device,
    ) -> float:
        self.report_progress(
            progress=progress_utils.ProgressCallbackContext.progress_for_steps(i, n),
            message=evaluators.Evaluator._eval_row_progress_msg(
                metric_name=aspect,
                device=device,
                row=i,
                total_rows=n,
            ),
        )
        prefixes, completions = self._prepare_inputs(row, aspect)
        scores, too_long = scorer.score(prefixes, completions)
        if too_long:
            self.add_problem_for_row(
                problems.ProblemSeverity.low, "Input was too long.", row
            )
        # return -(scores[0] * scores[1]) / (scores[0] + scores[1])
        return -sum(scores) / len(scores)

    @abc.abstractmethod
    def _prepare_inputs(
        self, row: datasets.LlmDataset.LlmDatasetRow, aspect: str
    ) -> tuple[list[str], list[str]]:
        """Returns Tuple (list of prefixes, list of completions to_evaluate) for single
        input i.e. when there is a task that requires ref->hypo and hypo->ref this
        method will return tuple with lists of length 2 - one element for ref->hypo and
        one for hypo->ref."""

    def add_problem_for_row(
        self,
        severity: problems.ProblemSeverity,
        message: str,
        row: datasets.LlmDataset.LlmDatasetRow,
    ):
        explanation_type = e10s.GlobalHtmlFragmentExplanation.explanation_type()
        self.add_problem(
            problems.ProblemAndAction(
                description=message,
                severity=severity,
                evaluator_id=self.evaluator_id(),
                evaluator_name=self._display_name,
                problem_attrs={
                    problems.ProblemAndAction.ATTR_ROW_KEYS: [(row.key, row.model_key)],
                    # input dataset ~ test lab ~ key is the test case key
                    problems.ProblemAndAction.ATTR_TEST_CASE_KEYS: [row.key],
                    problems.ProblemAndAction.ATTR_EVALUATOR_NAME: self._display_name,
                },
                explanation_type=explanation_type,
                explanation_name=e10s.GlobalHtmlFragmentExplanation.__name__,
                explanation_mime=f5s.HtmlFormat.mime,
                resources=[],
            )
        )
