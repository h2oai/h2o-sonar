# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.

from h2o_sonar import loggers
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import datasets
from h2o_sonar.lib.api import insights
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api.explanations import _explanations_leaderboards as t_leads


class FlippedPerturbedTestCase:
    """Represents a flipped perturbed test case serialized as LLM dataset row or
    evaluation result row.

    """

    @staticmethod
    def is_flipped() -> bool:
        pass

    @staticmethod
    def resolve_metrics(
        metrics: dict,
        metrics_meta: commons.MetricsMeta,
    ) -> dict[str, tuple]:
        """Resolve metrics values and pass/fail status for given set of ``metrics``.

        Parameters
        ----------
        metrics : dict
            Dictionary with metrics.
        metrics_meta : commons.MetricsMeta
            Metrics metadata.

        Returns
        -------
        dict[str, Tuple[MetricMeta, float, bool]] :
            Dictionary which maps metric ID to a tuple with metric meta, metric value,
            and metric pass/fail status (based on the threshold and higher is
            better/worse determine from the metadata).

        """
        # map: metric ID -> (metric meta, metric value, metric passed)
        result = {}

        if t_leads.LlmBoolLeaderboardExplanation.KEY_RESULT_CHECK_OK in metrics:
            # BOOL leaderboards metrics
            metric_meta = t_leads.LlmBoolLeaderboardExplanation.METRIC_META_MODEL_PASSES
            if metrics[t_leads.LlmBoolLeaderboardExplanation.KEY_RESULT_CHECK_OK]:
                metric_value = 1.0
                metric_passed = True
            else:
                metric_value = 0.0
                metric_passed = False

            # add to the result
            result[metric_meta.key] = (metric_meta, metric_value, metric_passed)
        elif (
            t_leads.LlmClassifierLeaderboardExplanation.METRIC_ACCURACY in metrics
            and t_leads.LlmClassifierLeaderboardExplanation.METRIC_RECALL in metrics
            and t_leads.LlmClassifierLeaderboardExplanation.METRIC_F1 in metrics
        ):
            # CLASSIFICATION leaderboards metrics
            metric_meta = (
                t_leads.LlmClassifierLeaderboardExplanation.METRIC_META_ACCURACY
            )
            if metrics[metric_meta.key]:
                metric_value = 1.0
                metric_passed = True
            else:
                metric_value = 0.0
                metric_passed = False

            # add to the result
            result[metric_meta.key] = (metric_meta, metric_value, metric_passed)
        else:
            # HEATMAP leaderboard metrics
            for metric_id in metrics:
                metric_meta = metrics_meta.get_metric(metric_id)
                metric_value = metrics[metric_id]
                metric_passed = metrics_meta.is_metric_passed(metric_id, metric_value)
                # add to the result
                result[metric_meta.key] = (metric_meta, metric_value, metric_passed)

        return result

    @property
    def is_flip(self) -> bool | None:
        if self.orig_pass is None or self.perturbed_pass is None:
            return None

        if (self.orig_pass and not self.perturbed_pass) or (
            not self.orig_pass and self.perturbed_pass
        ):
            return True

        return False

    @property
    def good_to_bad(self) -> bool | None:
        """``True`` if the perturbation flipped the test case from PASSING
        the metric to FAILING it, else ``False``.

        """
        if self.is_flip is None:
            return None

        return True if self.orig_pass and not self.perturbed_pass else False

    @property
    def llm_model_name(self) -> str:
        return self.model.llm_model_name if self.model else "MODEL"

    def __init__(
        self,
        explainable_model_key: str,
        explainable_model: (
            models.ExplainableRagModel | models.ExplainableLlmModel | None
        ),
        metric_meta: commons.MetricMeta,
        orig_row: datasets.LlmDataset.LlmDatasetRow | None = None,
        orig_metric_value: float = 0.0,
        orig_pass: bool = False,
        perturbed_row: datasets.LlmDataset.LlmDatasetRow | None = None,
        perturbed_metric_value: float = 0.0,
        perturbed_pass: bool = False,
        heat_threshold: float | None = None,
    ):
        self.model_key = explainable_model_key
        self.model = explainable_model
        self.metric_meta = metric_meta
        self.orig_row = orig_row
        self.orig_metric_value = orig_metric_value
        self.orig_metric_value_str = insights.r(self.orig_metric_value)
        self.orig_pass = orig_pass
        self.perturbed_row = perturbed_row
        self.perturbed_metric_value = perturbed_metric_value
        self.perturbed_metric_value_str = insights.r(self.perturbed_metric_value)
        self.perturbed_pass = perturbed_pass
        self.heat_threshold = heat_threshold

    def copy(self) -> "FlippedPerturbedTestCase":
        return FlippedPerturbedTestCase(
            explainable_model_key=self.model_key,
            explainable_model=self.model,
            metric_meta=self.metric_meta,
            orig_row=self.orig_row,
            orig_metric_value=self.orig_metric_value,
            orig_pass=self.orig_pass,
            perturbed_row=self.perturbed_row,
            perturbed_metric_value=self.perturbed_metric_value,
            perturbed_pass=self.perturbed_pass,
            heat_threshold=self.heat_threshold,
        )


def diagnose_perturbation_flips(
    eval_results: datasets.LlmEvalResults,
    metrics_meta: commons.MetricsMeta,
    key_2_evaluated_model: dict,
    logger=None,
) -> dict[str, dict[str, FlippedPerturbedTestCase]]:
    """Diagnose perturbation flips.

    Returns
    -------
    dict :
        Map: original row key -> perturbed row key -> ``FlippedPerturbedTestCase``.

    """
    logger = logger or loggers.SonarPrintLogger()

    # PASS 1: get list of rows which were perturbed
    # map: orig row key
    #        -> metric ID
    #           -> model key (prompt is answered by multiple models)
    #              -> FlippedPerturbedTestCase w/ pert. fields only
    perturbed_dict = {}
    for p_r in eval_results.results:
        if p_r.dataset_row.relationships:
            for rel in p_r.dataset_row.relationships:
                if rel.rel_type == datasets.LlmInputRelType.perturbation_source.name:
                    logger.info(
                        f"Row {p_r.dataset_row.key} was perturbed by {rel.target}"
                    )

                    # perturbed row metric(s):
                    #   map: metric ID -> (metric meta, metric value, metric passed)
                    perturbed_metrics_dict = FlippedPerturbedTestCase.resolve_metrics(
                        metrics=p_r.metrics,
                        metrics_meta=metrics_meta,
                    )

                    for metric_id in perturbed_metrics_dict:
                        (
                            metric_meta,
                            perturbed_metric_value,
                            perturbed_metric_passed,
                        ) = perturbed_metrics_dict[metric_id]

                        perturbed_test_case = FlippedPerturbedTestCase(
                            explainable_model_key=p_r.dataset_row.model_key,
                            explainable_model=key_2_evaluated_model.get(
                                p_r.dataset_row.model_key, None
                            ),
                            metric_meta=metric_meta,
                            perturbed_row=p_r.dataset_row,
                            perturbed_metric_value=perturbed_metric_value,
                            perturbed_pass=perturbed_metric_passed,
                        )

                        if not perturbed_dict.get(rel.target):
                            perturbed_dict[rel.target] = {}
                        if not perturbed_dict[rel.target].get(
                            p_r.dataset_row.model_key
                        ):
                            perturbed_dict[rel.target][p_r.dataset_row.model_key] = {}
                        perturbed_dict[rel.target][p_r.dataset_row.model_key][
                            metric_id
                        ] = perturbed_test_case

    # PASS 2: find original rows for perturbed rows and detect FLIPS
    #   orig row key -> metric ID -> FlippedPerturbedTestCase
    flips_dict = {}
    for o_r in eval_results.results:
        o_key = o_r.dataset_row.key
        if (
            o_key in perturbed_dict
            and o_r.dataset_row.model_key in perturbed_dict[o_key]
        ):
            for metric_id in perturbed_dict[o_key][o_r.dataset_row.model_key]:
                flipped_row_candidate = perturbed_dict[o_key][
                    o_r.dataset_row.model_key
                ][metric_id]

                # original row metric(s)
                #   map: metric ID -> (metric meta, metric value, metric passed)
                orig_metrics_dict = FlippedPerturbedTestCase.resolve_metrics(
                    metrics=o_r.metrics, metrics_meta=metrics_meta
                )
                orig_metric_tuple = orig_metrics_dict.get(metric_id, None)
                if orig_metric_tuple:
                    (
                        metric_meta,
                        orig_metric_value,
                        orig_metric_passed,
                    ) = orig_metric_tuple

                    if (
                        t_leads.LlmBoolLeaderboardExplanation.KEY_RESULT_CHECK_OK
                        not in o_r.metrics
                    ):
                        flipped_row_candidate.heat_threshold = (
                            flipped_row_candidate.metric_meta.threshold
                        )

                    flipped_row_candidate.orig_row = o_r.dataset_row
                    flipped_row_candidate.orig_metric_value = orig_metric_value
                    flipped_row_candidate.orig_metric_value_str = insights.r(
                        flipped_row_candidate.orig_metric_value
                    )
                    flipped_row_candidate.orig_pass = orig_metric_passed

                    if flipped_row_candidate.is_flip:
                        logger.info(
                            f"Flip detected: original row {o_key} vs. "
                            f"perturbed row {flipped_row_candidate.perturbed_row.key}"
                        )

                        # add new flip to the flips dictionary
                        if not flips_dict.get(o_key):
                            flips_dict[o_key] = {}
                        flips_dict[o_key][metric_id] = flipped_row_candidate.copy()
                else:
                    raise ValueError(
                        f"Metric ID {metric_id} not found in the original "
                        f"row {o_key}, available keys are: "
                        f"{orig_metrics_dict.keys()}"
                    )

    return flips_dict
