# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
from h2o_sonar import loggers
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import explainers as e8s
from h2o_sonar.lib.api import interpretations
from h2o_sonar.utils import progress
from h2o_sonar.utils import sampling


class Evaluation(interpretations.Interpretation):
    def __init__(
        self,
        common_params: commons.CommonInterpretationParams,
        created: float,
        evaluators: list[str | commons.ExplainerToRun] | None,
        evaluator_keywords: list[str] | None = None,
        key: str = "",
        sampler: sampling.DatasetSampler | None = None,
        branding: commons.Branding = commons.Branding.H2O_SONAR,
        progress_callback: progress.AbstractProgressCallbackContext | None = None,
        logger=None,
        extra_params: list | None = None,
    ):
        progress_callback = (
            progress_callback
            or progress.LoggingProgressCallbackContext(
                logger=logger or loggers.SonarPrintLogger(),
                prefix="Evaluation progress",
                do_update=[self],  # auto update self.progress
            )
        )

        interpretations.Interpretation.__init__(
            self,
            common_params=common_params,
            created=created,
            explainers=evaluators,
            explainer_keywords=evaluator_keywords,
            key=key,
            sampler=sampler,
            branding=branding,
            progress_callback=progress_callback,
            logger=logger,
            extra_params=extra_params,
        )

        self.evaluators = []

    def get_evaluator_result(self, evaluator_id: str) -> e8s.ExplainerResult | None:
        return self.explainer_results.get(evaluator_id)

    def get_all_evaluator_ids(self) -> list[str]:
        """Return IDs of evaluators which user requested to run."""
        return self.get_all_explainer_ids()

    def get_scheduled_evaluator_ids(self) -> list[str]:
        """Return IDs of evaluators which were not discarded by compatibility checks
        and were used to build the execution plan.

        """
        return self.get_scheduled_explainer_ids()

    def get_incompatible_evaluator_ids(self):
        """Return IDs of evaluators which were discarded by the compatibility checks."""
        return self.get_incompatible_explainer_ids()

    def get_finished_evaluator_ids(self):
        """Return IDs of evaluators which finished."""
        return self.get_finished_explainer_ids()

    def get_failed_evaluator_ids(self) -> list[str]:
        """Return IDs of evaluators which finished, but failed."""
        return self.get_failed_explainer_ids()

    def get_successful_evaluator_ids(self) -> list[str]:
        """Return IDs of evaluators which successfully finished."""
        return self.get_successful_explainer_ids()

    def get_evaluator_ids_by_status(self, status: int) -> list[str]:
        return self.get_explainer_ids_by_status(status)

    def get_evaluator_jobs_by_status(
        self, status: int
    ) -> list[interpretations.ExplainerJob]:
        return self.get_explainer_jobs_by_status(status)

    @staticmethod
    def from_interpretation(
        interpretation: interpretations.Interpretation,
    ) -> "Evaluation":
        if not interpretation:
            raise ValueError("Interpretation is None")

        e = Evaluation(
            common_params=interpretation.common_params,
            created=interpretation.created,
            evaluators=interpretation.explainers,
            evaluator_keywords=interpretation.explainer_keywords,
            key=interpretation.key,
            sampler=interpretation.sampler,
            branding=interpretation.branding,
            logger=interpretation.logger,
            extra_params=interpretation.extra_params,
        )

        # copy fields which are not in the constructor
        e.progress = interpretation.progress
        e.progress_message = interpretation.progress_message
        e.progress_callback = interpretation.progress_callback
        e.status = interpretation.status
        e.error = interpretation.error
        e.explainers = interpretation.explainers
        e.evaluators = e.explainers
        e.explainer_id_to_e2run = interpretation.explainer_id_to_e2run
        e.persistence = interpretation.persistence
        e.future = interpretation.future
        e.html_format = interpretation.html_format
        e.pdf_format = interpretation.pdf_format
        e.result = interpretation.result
        e.explainer_results = interpretation.explainer_results

        return e
