# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json

from h2o_sonar import config
from h2o_sonar import loggers
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import datasets
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import models as m4s
from h2o_sonar.lib.api import persistences
from h2o_sonar.lib.api.explanations import _explanations_base
from h2o_sonar.lib.api.explanations import _explanations_cmp


class LlmEvalResultsExplanation(_explanations_base.Explanation):
    _explanation_type = "llm-eval-results"
    _is_global = True

    KEY_RESULTS = "results"
    KEY_MODELS = "models"
    KEY_EVALUATOR = "evaluator"
    KEY_ID = "id"

    def __init__(
        self,
        evaluator,
        eval_results,
        display_name: str = None,
        display_category: str = None,
    ) -> None:
        _explanations_base.Explanation.__init__(
            self,
            explainer=evaluator,
            display_name=display_name,
            display_category=display_category,
        )

        self.eval_results = eval_results
        self.explainable_models = evaluator.models

    def validate(self) -> bool:
        return self._formats is not None

    def add_datatable_format(self):
        """Add datatable format."""
        self.add_format(
            f5s.DatatableCustomExplanationFormat(
                explanation=self,
                frame=self.eval_results.to_datatable(),
                frame_file="",
                persistence=self.explainer.persistence.store,
            )
        )

    def add_csv_format(self):
        """Add CSV format."""
        self.add_format(
            f5s.CustomCsvFormat(
                explanation=self,
                frame=self.eval_results.to_datatable(),
                persistence=self.explainer.persistence.store,
            )
        )

    def add_json_format(self):
        """Add JSon format."""
        this = LlmEvalResultsExplanation
        results_dict = self.eval_results.to_dict()
        if self.explainable_models:
            results_dict[this.KEY_MODELS] = [
                m.to_dict() for m in self.explainable_models if hasattr(m, "to_dict")
            ]
        results_dict[this.KEY_EVALUATOR] = self.explainer.as_descriptor().dump()
        self.add_format(
            f5s.CustomJsonFormat(
                explanation=self,
                json_data=json.dumps(
                    results_dict, indent=4, cls=persistences.NanEncoder
                ),
                persistence=self.explainer.persistence.store,
            )
        )

    @staticmethod
    def from_dict(
        explainers_map: dict,
        explanation_dict: dict,
        display_name: str = "",  # transient > need to be set
        display_category: str = "",  # transient > need to be set
        logger=None,
    ) -> "LlmEvalResultsExplanation":
        """Create LlmEvalResultsExplanation from dictionary.

        Parameters
        ----------
        explainers_map : dict[str, Explainer]
            Map of explainer ID to Explainer.
        explanation_dict : dict
            Dictionary representation of ``LlmEvalResultsExplanation``.
        display_name : str
            Display name of the explanation.
        display_category : str
            Display category of the explanation.
        logger : logging.SonarLogger
            Logger to use.

        Returns
        -------
        LlmEvalResultsExplanation :
            LLM eval results explanation object.

        """
        logger = logger or loggers.SonarPrintLogger()

        if not explainers_map:
            raise ValueError("Explainers dictionary is empty!")
        if not explanation_dict:
            raise ValueError("Explanation dictionary is empty!")

        # EVALUATOR - get it first to access default metrics_meta
        evaluator_id = explanation_dict.get(
            LlmEvalResultsExplanation.KEY_EVALUATOR, {}
        ).get(LlmEvalResultsExplanation.KEY_ID, None)
        if evaluator_id is None:
            raise ValueError(
                f"Evaluator ID '{evaluator_id}' not found in explanation dictionary!"
            )
        evaluator_cls = explainers_map[evaluator_id]
        if not evaluator_cls:
            raise ValueError(
                f"Explainer with id '{evaluator_id}' not found in explainers map!"
            )
        evaluator = evaluator_cls()

        # METRICS META: load it from the dict, if dict empty, then fallback to evaluator
        # metrics meta w/ CUSTOMIZED thresholds from JSON
        custom_metrics_meta_dict = explanation_dict.get(
            LlmEvalResultsExplanation.KEY_EVALUATOR, {}
        ).get(datasets.LlmEvalResults.LlmEvalResultRow.KEY_METRICS_META, {})
        if not custom_metrics_meta_dict:
            logger.warning(
                "Explanation result dictionary has NO metrics meta with customized "
                "threshold values - using fallback to evaluator's metrics meta with "
                "DEFAULT thresholds"
            )
        custom_metrics_meta = commons.MetricsMeta.from_dict(custom_metrics_meta_dict)
        evaluator_cls._metrics_meta = custom_metrics_meta  # ENSURE custom thresholds

        # models (injected into the evaluator)
        evaluator.models = []
        if explanation_dict.get(LlmEvalResultsExplanation.KEY_MODELS) is None:
            raise ValueError("Models not found in the explanation dictionary!")

        models_json = explanation_dict.get(LlmEvalResultsExplanation.KEY_MODELS, [])
        if models_json:
            for m_json in models_json:
                try:
                    evaluator.models.append(
                        m4s.explainable_rag_llm_model_from_json(m_json)
                    )
                except Exception as ex:
                    logger.warning(
                        f"WARNING: Unable to load model from JSon: {ex}\n'{m_json}'"
                    )

        # eval results
        if explanation_dict.get(LlmEvalResultsExplanation.KEY_RESULTS) is None:
            raise ValueError("Results not found in the explanation dictionary!")
        eval_results = datasets.LlmEvalResults.from_dict(explanation_dict)

        return LlmEvalResultsExplanation(
            evaluator=evaluator,
            eval_results=eval_results,
            display_name=display_name,
            display_category=display_category,
        )

    def merge_metrics(
        self,
        explanations: list["LlmEvalResultsExplanation"],
        evaluator_ids: list[str] | None = None,
    ) -> "LlmEvalResultsExplanation":
        """Merge metric scores of multiple explanation results into the one by
        finding corresponding test cases of corresponding models.

        Parameters
        ----------
            explanations : list["LlmEvalResultsExplanation"]
                Explanations to be merged.
            evaluator_ids : list[str] | None
                Evaluators whose metrics should be merged. If None,
                all evaluator metrics will be merged.

        Returns
        -------
        LlmEvalResultsExplanation :
            New explanation with merged metrics from all provided explanations.

        """
        if not explanations:
            raise ValueError("No explanations to merge")

        # build a lookup dict: (test_case_key, model_key) -> result_row
        merged_results_map = {}
        for result_row in self.eval_results.results:
            key = (result_row.dataset_row.key, result_row.dataset_row.model_key)
            merged_results_map[key] = result_row

        # iterate through all explanations to merge
        for explanation in explanations:
            # check if we should merge metrics from this evaluator
            evaluator_id = explanation.explainer.as_descriptor().id
            if evaluator_ids and evaluator_id not in evaluator_ids:
                continue

            # iterate through all result rows in this explanation
            for result_row in explanation.eval_results.results:
                key = (
                    result_row.dataset_row.key,
                    result_row.dataset_row.model_key,
                )

                # find matching result row in merged results
                if key in merged_results_map:
                    # merge metrics from this result row into the existing one
                    merged_row = merged_results_map[key]
                    for metric_id, metric_value in result_row.metrics.items():
                        merged_row.metrics[metric_id] = metric_value

                    # merge actual_output_meta if not already present
                    if result_row.actual_output_meta:
                        for meta in result_row.actual_output_meta:
                            if meta not in merged_row.actual_output_meta:
                                merged_row.actual_output_meta.append(meta)

                    # merge metrics_meta if not already present
                    if result_row.metrics_meta:
                        merged_row.metrics_meta.update(result_row.metrics_meta)

        # create a new LlmEvalResults with the merged data
        merged_eval_results = datasets.LlmEvalResults()
        merged_eval_results.results = list(merged_results_map.values())

        # merge metrics metadata from all evaluators
        merged_metrics_meta = commons.MetricsMeta()
        # add metrics from base evaluator
        if hasattr(self.explainer, "_metrics_meta"):
            for (
                metric_key,
                metric,
            ) in self.explainer._metrics_meta.key_to_metric.items():
                merged_metrics_meta.add_metric(metric)
        # add metrics from all other evaluators
        for explanation in explanations:
            evaluator_id = explanation.explainer.as_descriptor().id
            if evaluator_ids and evaluator_id not in evaluator_ids:
                continue
            if hasattr(explanation.explainer, "_metrics_meta"):
                for (
                    metric_key,
                    metric,
                ) in explanation.explainer._metrics_meta.key_to_metric.items():
                    if metric_key not in merged_metrics_meta.key_to_metric:
                        merged_metrics_meta.add_metric(metric)

        # create a new explanation with the merged results
        merged_explanation = LlmEvalResultsExplanation(
            evaluator=self.explainer,
            eval_results=merged_eval_results,
            display_name=self.display_name,
            display_category=self.display_category,
        )
        # set the merged metrics metadata on the evaluator
        merged_explanation.explainer._metrics_meta = merged_metrics_meta

        return merged_explanation

    def compare(
        self,
        other: "LlmEvalResultsExplanation",
        baseline_llm_model: str = "",
        current_llm_model: str = "",
        comparison_method: _explanations_base.SentenceComparisonMethod = (
            _explanations_base.SentenceComparisonMethod.COSINE_DISTANCE
        ),
        sentence_similarity_threshold: float = 0.9,
    ) -> _explanations_cmp.EvalResultsDiff:
        """Compare this with other LLM evaluation results.

        Parameters
        ----------
        other : LlmEvalResultsExplanation
            Other LLM evaluation results to compare with.
        baseline_llm_model: str
            The LLM model name used for filtering and equals of explainable models.
        current_llm_model: str
            The LLM model name used for filtering and equals of explainable models.
        comparison_method : SentenceComparisonMethod
            The method to use for comparing sentences:
            - EXACT_MATCH: exact string matching
            - COSINE_DISTANCE: cosine distance of sentence embeddings (default)
        sentence_similarity_threshold : float
            Threshold for determining if sentences are "common" (high similarity).
            Sentences with similarity >= threshold are considered common.
            Default is 0.9.

        Returns
        -------
        EvalResultsDiff
            Difference between the two LLM evaluation results.

        """
        # get branding from config
        branding = commons.Branding.EVAL_STUDIO
        if config.config.branding:
            if config.config.branding == commons.Branding.H2O_SONAR.name:
                branding = commons.Branding.H2O_SONAR
            elif config.config.branding == commons.Branding.EVAL_STUDIO.name:
                branding = commons.Branding.EVAL_STUDIO

        comparator = _explanations_cmp.EvalResultsExplanationsComparator(
            baseline_explanation=self,
            current_explanation=other,
            comparison_method=comparison_method,
            sentence_similarity_threshold=sentence_similarity_threshold,
            branding=branding,
        )

        return comparator.compare(
            baseline_llm_model=baseline_llm_model,
            current_llm_model=current_llm_model,
        )
