# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import datetime
import enum
import inspect
import json
import os
import pathlib
import random
import subprocess
import time
import uuid

import airium
import markdown

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar import loggers
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import datasets as d6s
from h2o_sonar.lib.api import explainers as e8s
from h2o_sonar.lib.api import explanations as e9s
from h2o_sonar.lib.api import formats
from h2o_sonar.lib.api import htmls
from h2o_sonar.lib.api import insights
from h2o_sonar.lib.api import models as m4s
from h2o_sonar.lib.api import persistences
from h2o_sonar.lib.api import problems
from h2o_sonar.lib.api.insights import AbcProblemInsight as abc_ins
from h2o_sonar.utils import progress as progress_utils
from h2o_sonar.utils import sampling


class ExplainerJob:
    """Explainer job."""

    KEY_KEY = "key"
    KEY_EXPLAINER_DESCRIPTOR = "explainer"
    KEY_RESULT_DESCRIPTOR = "result_descriptor"
    KEY_PROGRESS = "progress"
    KEY_STATUS = "status"
    KEY_ERROR = "error"
    KEY_MESSAGE = "message"
    KEY_CREATED = "created"
    KEY_DURATION = "duration"
    KEY_CHILD_KEYS = "child_explainer_job_keys"
    KEY_JOB_LOCATION = "job_location"

    def __init__(
        self,
        key: str = "",
        created: float = 0.0,
        duration: float = 0.0,
        progress: float = 0.0,
        status: commons.ExplainerJobStatus = commons.ExplainerJobStatus.UNKNOWN,
        message: str = "",
        error: str = "",
        explainer_persistence=None,
        explainer_descriptor: e8s.ExplainerDescriptor | None = None,
        result_descriptor=None,
        child_explainer_job_keys: list[str] | None = None,
        job_location: str = "",
    ):
        """Explainer job constructor."""
        self.key = key
        self.created = created
        self.duration = duration
        self.progress = progress
        self.status = status
        self.error = error
        self.message = message
        self.explainer_persistence = explainer_persistence
        self.explainer_descriptor = explainer_descriptor
        self.result_descriptor = result_descriptor
        self.child_explainer_job_keys = child_explainer_job_keys
        self.job_location = job_location

    def explainer_id(self) -> str:
        return self.explainer_descriptor.id

    def evaluator_id(self) -> str:
        return self.explainer_descriptor.id

    def to_dict(self) -> dict:
        return {
            ExplainerJob.KEY_KEY: self.key,
            ExplainerJob.KEY_PROGRESS: self.progress,
            ExplainerJob.KEY_STATUS: (
                self.status.value if self.status is not None else 0
            ),
            ExplainerJob.KEY_ERROR: self.error,
            ExplainerJob.KEY_MESSAGE: self.message,
            ExplainerJob.KEY_EXPLAINER_DESCRIPTOR: (
                self.explainer_descriptor.dump() if self.explainer_descriptor else {}
            ),
            ExplainerJob.KEY_CREATED: self.created,
            ExplainerJob.KEY_DURATION: self.duration,
            ExplainerJob.KEY_CHILD_KEYS: self.child_explainer_job_keys,
            ExplainerJob.KEY_JOB_LOCATION: str(self.job_location),
            ExplainerJob.KEY_RESULT_DESCRIPTOR: (
                self.result_descriptor if self.result_descriptor else {}
            ),
        }

    @staticmethod
    def from_dict(explainer_job_dict: dict):
        return ExplainerJob(
            key=explainer_job_dict.get(ExplainerJob.KEY_KEY, ""),
            created=explainer_job_dict.get(ExplainerJob.KEY_CREATED, 0.0),
            duration=explainer_job_dict.get(ExplainerJob.KEY_DURATION, 0.0),
            progress=explainer_job_dict.get(ExplainerJob.KEY_PROGRESS, 0.0),
            status=commons.ExplainerJobStatus.from_int(
                explainer_job_dict.get(ExplainerJob.KEY_STATUS, 0)
            ),
            message=explainer_job_dict.get(ExplainerJob.KEY_MESSAGE, ""),
            error=explainer_job_dict.get(ExplainerJob.KEY_ERROR, ""),
            explainer_descriptor=e8s.ExplainerDescriptor.load(
                explainer_job_dict.get(ExplainerJob.KEY_EXPLAINER_DESCRIPTOR, {})
            ),
            result_descriptor=explainer_job_dict.get(
                ExplainerJob.KEY_RESULT_DESCRIPTOR, {}
            ),
            child_explainer_job_keys=explainer_job_dict.get(
                ExplainerJob.KEY_CHILD_KEYS, []
            ),
            job_location=explainer_job_dict.get(ExplainerJob.KEY_JOB_LOCATION, ""),
        )

    def tick(self, msg: str = "", progress_increment: float = 0.1):
        self.message = msg
        self.progress += progress_increment
        self.duration = time.time() - self.created

    def is_finished(self) -> bool:
        if self.status.value >= 0 or self.progress >= 1.0:
            return True
        return False

    def success(self):
        self.status = commons.ExplainerJobStatus.SUCCESS
        self.message = "DONE"
        self.progress = 1.0
        self.duration = time.time() - self.created


class OverallResult(enum.Enum):
    """Overall evaluation/interpretation result in the traffic light style."""

    no_problem = enum.auto()  # green
    low_severity_problems = enum.auto()  # yellow
    medium_severity_problems = enum.auto()  # orange
    high_severity_problems = enum.auto()  # red


class Interpretation:
    """Interpretation is request to interpret a model using explainers.
    Interpretation instance serves also as execution context, however, interpretation
    instance does not execute explainers itself  - it's purpose is to be
    prescription (of what is requested) and stateful data holder. Interpretation
    result (referenced by the instance) is a set of explanations which were created
    by explainers.

    Attributes
    ----------
    key : str
      Interpretation key.
    common_params : commons.CommonInterpretationParams
      Interpretation parameters specified by the user.
    explainers : list
      Explainers to be run (if no explainers specified, then all compatible explainers
      are run).
    persistence : persistences.Persistence
      Persistence store - file-system, in-memory, DB - where were stored interpretation
      results and from where it might be loaded using the persistence instance.

    """

    KEY_CREATED = "created"
    KEY_PROGRESS = "progress"
    KEY_PROGRESS_MESSAGE = "progress_message"
    KEY_STATUS = "status"
    KEY_ERROR = "error"
    KEY_I_KEY = "interpretation_key"
    KEY_I_PARAMS = "interpretation_parameters"
    KEY_EXPLAINERS = "explainers"
    KEY_E_PARAMS = persistences.InterpretationPersistence.KEY_E_PARAMS
    KEY_RESULT = persistences.InterpretationPersistence.KEY_RESULT
    KEY_DATASET = "dataset"
    KEY_TESTSET = "testset"
    KEY_VALIDSET = "validset"
    KEY_MODEL = "model"
    KEY_MODELS = "models"
    KEY_TARGET_COL = "target_col"
    KEY_ALL_EXPLAINERS = "all_explainer_ids"
    KEY_INCOMPATIBLE_EXPLAINERS = "incompatible_explainer_ids"
    KEY_INCOMPATIBLE_EXPLAINERS_DS = "incompatible_explainers"
    KEY_SCHEDULED_EXPLAINERS = "scheduled_explainers"
    KEY_EXECUTED_EXPLAINERS = "executed_explainers"
    KEY_PROBLEMS = "problems"
    KEY_INSIGHTS = "insights"
    KEY_OVERALL_RESULT = "overall_result"
    KEY_RESULTS_LOCATION = "results_location"
    KEY_INTERPRETATION_LOCATION = "interpretation_location"

    def __init__(
        self,
        common_params: commons.CommonInterpretationParams,
        created: float,
        explainers: list[str | commons.ExplainerToRun] | None,
        explainer_keywords: list[str] | None = None,
        key: str = "",
        sampler: sampling.DatasetSampler | None = None,
        branding: commons.Branding = commons.Branding.H2O_SONAR,
        results_formats: list[str] | None = None,
        progress_callback: progress_utils.AbstractProgressCallbackContext | None = None,
        logger=None,
        extra_params: list | None = None,
    ):
        """Create interpretation.

        Parameters
        ----------
        common_params :
          Common interpretation parameters.
        created : float
          Interpretation creation time.
        explainers :
          Explainer explainers to run. If empty, then run all compatible explainers.
        explainer_keywords : list[str] | None
          Explainer keywords.
        key : str
          User defined interpretation key.
        logger :
          Logger.

        """
        self.created = created or time.time()
        self.logger = logger or loggers.SonarPrintLogger()
        self.progress: float = 0.0
        self.progress_message: str = ""
        self.progress_callback = progress_utils.LoggingProgressCallbackContext(
            logger=self.logger,
            prefix=(
                "Interpretation progress"
                if branding == commons.Branding.H2O_SONAR
                else "Evaluation progress"
            ),
            do_update=[self],  # auto update self.progress
            parent_callback=progress_callback,
            name=(
                "Interpretation progress callback"
                if branding == commons.Branding.H2O_SONAR
                else "Evaluation progress callback"
            ),
        )
        self.status: commons.ExplainerJobStatus = commons.ExplainerJobStatus.UNKNOWN
        # error message which is set only in case of failed interpretation ^
        self.error: str = ""

        self.branding = branding
        self.results_formats = results_formats or []

        # IN: parameters
        self.common_params = common_params
        self.explainers = Interpretation._normalize_explainers_spec(explainers)
        self.explainer_id_to_e2run = {e.id: e for e in self.explainers}
        self.explainer_keywords = (
            explainer_keywords
            if explainer_keywords
            else [e8s.Explainer.KEYWORD_DEFAULT]
        )
        self.sampler = sampler
        self.extra_params = extra_params

        # RUNTIME
        if key and commons.is_valid_key(key):  # interpretation aka MLI key
            self.key = key
        else:
            self.key = commons.generate_key()
        self.persistence: persistences.InterpretationPersistence | None = None
        self.future = None  # concurrent.futures.Future in case of the async run
        self.html_format: HtmlInterpretationFormat = HtmlInterpretationFormat(
            self, branding=branding, logger=logger
        )
        self.pdf_format: PdfInterpretationFormat = PdfInterpretationFormat(
            interpretation=self, branding=branding, logger=logger
        )

        # OUT: result
        self.result = InterpretationResult()
        self.explainer_results: dict[str, e8s.ExplainerResult] = {}

    def __str__(self) -> str:
        return f"Interpretation:\n{self.to_json(2)}"

    def set_progress(self, progress: float, message: str | None = None) -> float:
        return self.progress_callback.set_progress(progress, message)

    def is_finished(self) -> bool:
        """Check if interpretation is finished.

        Returns True if status indicates completion (success, failed, finished)
        or progress has reached 100%.

        Returns
        -------
        bool :
            True if interpretation is finished, False otherwise.

        """
        if self.status.value >= 0 or self.progress >= 1.0:
            return True
        return False

    def get_explainer_problems(self, explainer_id: str) -> list:
        model_problems = []
        if explainer_id and self.result and self.result.problems:
            model_problems = [
                p for p in self.result.problems if p.explainer_id == explainer_id
            ]
        return model_problems

    def get_explainer_insights(self, explainer_id: str) -> list:
        models_insights = []
        if explainer_id and self.result and self.result.insights:
            models_insights = [
                i for i in self.result.insights if i.explainer_id == explainer_id
            ]
        return models_insights

    def get_model_insights(self, model_name: str) -> list:
        models_insights = []
        if model_name and self.result and self.result.insights:
            for i in self.result.insights:
                if i.insight_attrs and i.insight_attrs.get("model_name") == model_name:
                    models_insights.append(i)
        return models_insights

    def get_model_problems(self, model_name: str) -> list:
        models_problems = []
        if model_name and self.result and self.result.problems:
            for i in self.result.problems:
                if i.problem_attrs and i.problem_attrs.get("model_name") == model_name:
                    models_problems.append(i)
        return models_problems

    def get_problems_by_severity(self, severity: problems.ProblemSeverity) -> list:
        model_problems = []
        if severity and self.result and self.result.problems:
            model_problems = [p for p in self.result.problems if p.severity == severity]
        return model_problems

    def update_overall_result(self) -> OverallResult:
        if self.result and self.result.problems:
            highest_severity = self.result.problems[0].severity
            for p in self.result.problems:
                if problems.ProblemSeverity.compare(highest_severity, p.severity) > 0:
                    highest_severity = p.severity

            if highest_severity == problems.ProblemSeverity.high:
                self.result.overall_result = OverallResult.high_severity_problems
            elif highest_severity == problems.ProblemSeverity.medium:
                self.result.overall_result = OverallResult.medium_severity_problems
            else:
                self.result.overall_result = OverallResult.low_severity_problems
        else:
            self.result.overall_result = OverallResult.no_problem

        return self.result.overall_result

    def get_insights(self) -> list:
        return self.result.insights.copy()

    @staticmethod
    def _normalize_explainers_spec(explainers: list) -> list[commons.ExplainerToRun]:
        """Normalize explainer-to-run specification."""
        explainers_normalized: list = []
        if explainers:
            for e in explainers:
                if isinstance(e, str):
                    e_normalized = commons.ExplainerToRun(explainer_id=e)
                elif isinstance(e, commons.ExplainerToRun):
                    e_normalized = e
                elif inspect.isclass(e) and "explainer_id" in dict(
                    inspect.getmembers(e, inspect.isfunction)
                ):
                    e_normalized = commons.ExplainerToRun(explainer_id=e.explainer_id())
                else:
                    raise ValueError(
                        f"Unsupported type '{type(e)}' to specify explainer to run"
                    )
                explainers_normalized.append(e_normalized)

        return explainers_normalized

    def validate_and_normalize_params(
        self,
    ):
        """Validate and check interpretation parameters."""
        if self.common_params:
            # need this check to handle pandas.DataFrame check ...
            if self.common_params.dataset is None:
                raise ValueError(
                    "Invalid interpretation parameters: missing required parameter "
                    "'dataset' value"
                )

            # validate results location
            if self.common_params.results_location and isinstance(
                self.common_params.results_location, str
            ):
                if not os.path.isdir(self.common_params.results_location):
                    try:
                        persistences.ExplainerPersistence.makedirs(
                            self.common_params.results_location
                        )
                    except Exception as e:
                        raise ValueError(
                            f"Interpretation result directory does not exist: "
                            f"'{self.common_params.results_location}' and attempt to "
                            f"create the directory failed: {e}"
                        )
            self.result.results_location = self.common_params.results_location

            # normalize explainer to commons.ExplainerToRun w/ ID and key
            if self.explainers:
                normalized_explainers: list[commons.ExplainerToRun] = []
                for r in self.explainers:
                    if isinstance(r, commons.ExplainerToRun):
                        normalized_explainers.append(r)
                    elif isinstance(r, str):
                        normalized_explainers.append(
                            commons.ExplainerToRun(
                                explainer_id=r,
                                params={},
                            )
                        )
                    else:
                        raise ValueError(
                            f"Unknown type of explainer to be executed: {type(r)} ({r})"
                        )
                self.explainers = normalized_explainers
        else:
            raise ValueError(
                "No or invalid interpretation parameters - required parameters "
                "(like results location) are not set"
            )

    @staticmethod
    def dict_to_digest(i_json: dict):
        i_created = i_json.get(Interpretation.KEY_CREATED, 0.0)
        if i_created:
            created_str = str(datetime.datetime.fromtimestamp(i_created))
            created_str = created_str[: created_str.index(".")]
            created_str = f"({created_str} T{time.strftime('%z')})"
        else:
            created_str = ""

        return {
            Interpretation.KEY_CREATED: created_str,
            Interpretation.KEY_MODEL: i_json.get(Interpretation.KEY_I_PARAMS, {}).get(
                Interpretation.KEY_MODEL, ""
            ),
            Interpretation.KEY_TARGET_COL: i_json.get(
                Interpretation.KEY_I_PARAMS, {}
            ).get(Interpretation.KEY_TARGET_COL, ""),
            Interpretation.KEY_DATASET: i_json.get(Interpretation.KEY_I_PARAMS, {}).get(
                Interpretation.KEY_DATASET, ""
            ),
            Interpretation.KEY_ALL_EXPLAINERS: len(
                i_json.get(Interpretation.KEY_RESULT, {}).get(
                    Interpretation.KEY_ALL_EXPLAINERS, {}
                )
            ),
            Interpretation.KEY_INCOMPATIBLE_EXPLAINERS: len(
                i_json.get(Interpretation.KEY_RESULT, {}).get(
                    Interpretation.KEY_INCOMPATIBLE_EXPLAINERS, {}
                )
            ),
            Interpretation.KEY_SCHEDULED_EXPLAINERS: len(
                i_json.get(Interpretation.KEY_RESULT, {}).get(
                    Interpretation.KEY_SCHEDULED_EXPLAINERS, {}
                )
            ),
            Interpretation.KEY_PROBLEMS: len(
                i_json.get(Interpretation.KEY_RESULT, {}).get(
                    Interpretation.KEY_PROBLEMS, {}
                )
            ),
            Interpretation.KEY_INSIGHTS: len(
                i_json.get(Interpretation.KEY_RESULT, {}).get(
                    Interpretation.KEY_INSIGHTS, {}
                )
            ),
        }

    def is_explainer_scheduled(self) -> bool:
        """Indicate whether there was at least one explainer which was ran."""
        return True if self.get_scheduled_explainer_ids() else False

    def is_explainer_finished(self) -> bool:
        """Indicate whether an explainer successfully finished or failed."""
        return True if self.get_finished_explainer_ids() else False

    def is_explainer_successful(self) -> bool:
        """Indicate whether an explainer successfully finished."""
        return True if self.get_successful_explainer_ids() else False

    def is_explainer_failed(self) -> bool:
        """Indicate whether an explainer failed."""
        return True if self.get_failed_explainer_ids() else False

    def is_evaluator_scheduled(self) -> bool:
        return self.is_explainer_failed()

    def is_evaluator_finished(self) -> bool:
        return self.is_explainer_finished()

    def is_evaluator_successful(self) -> bool:
        return self.is_explainer_successful()

    def is_evaluator_failed(self) -> bool:
        return self.is_explainer_failed()

    def get_all_explainer_ids(self) -> list[str]:
        e_ids = []
        if self.explainers:
            for e in self.explainers:
                if isinstance(e, commons.ExplainerToRun):
                    e_ids.append(e.id)
                elif isinstance(e, str):
                    e_ids.append(e)

        return e_ids

    def get_incompatible_explainer_ids(self) -> list[str]:
        incompatible_e_ids = self.get_all_explainer_ids()
        for e_id in self.get_finished_explainer_ids():
            if e_id in incompatible_e_ids:
                incompatible_e_ids.remove(e_id)

        return incompatible_e_ids

    def get_scheduled_explainer_ids(self) -> list[str]:
        if self.result:
            return self.result.to_dict().get(
                Interpretation.KEY_SCHEDULED_EXPLAINERS, []
            )
        return []

    def get_finished_explainer_ids(self) -> list[str]:
        if self.result:
            executed_explainers = self.result.to_dict().get(
                Interpretation.KEY_EXECUTED_EXPLAINERS, []
            )
            return [
                r.get(ExplainerJob.KEY_EXPLAINER_DESCRIPTOR, {}).get("id", "")
                for r in executed_explainers
            ]
        return []

    def get_jobs_for_evaluator_id(self, explainer_id: str) -> list[ExplainerJob]:
        return self.get_jobs_for_explainer_id(explainer_id)

    def get_explainer_ids_by_status(self, status: int) -> list[str]:
        if self.result:
            executed_explainers = self.result.to_dict().get(
                Interpretation.KEY_EXECUTED_EXPLAINERS, []
            )
            return [
                r.get(ExplainerJob.KEY_EXPLAINER_DESCRIPTOR, {}).get("id", "")
                for r in executed_explainers
                if r.get("status", -123_456) == status
            ]
        return []

    def get_explainer_jobs_by_status(self, status: int) -> list[ExplainerJob]:
        e_ids_by_status = self.get_explainer_ids_by_status(status)
        e_jobs = []
        for e_id in e_ids_by_status:
            e_jobs.extend(self.get_jobs_for_explainer_id(e_id))
        return e_jobs

    def get_jobs_for_explainer_id(self, explainer_id: str) -> list[ExplainerJob]:
        e_jobs = []
        if self.result:
            executed_explainers = self.result.to_dict().get(
                Interpretation.KEY_EXECUTED_EXPLAINERS, []
            )
            for r in executed_explainers:
                if (
                    r.get(ExplainerJob.KEY_EXPLAINER_DESCRIPTOR, {}).get("id", "")
                    == explainer_id
                ):
                    e_job = self.result.get_explainer_job(
                        r.get(ExplainerJob.KEY_KEY, "")
                    )
                    e_jobs.append(e_job)

        return e_jobs

    def get_successful_explainer_ids(self) -> list[str]:
        return self.get_explainer_ids_by_status(0)

    def get_failed_explainer_ids(self) -> list[str]:
        return self.get_explainer_ids_by_status(2)

    def get_explainer_result_metadata(self, explainer_id: str) -> dict | None:
        matching_explainers = None
        if self.result:
            executed_explainers = self.result.to_dict().get(
                Interpretation.KEY_EXECUTED_EXPLAINERS, []
            )
            matching_explainers = [
                r
                for r in executed_explainers
                if r.get(ExplainerJob.KEY_EXPLAINER_DESCRIPTOR, {}).get("id", "")
                == explainer_id
            ]
        return matching_explainers[0] if matching_explainers else None

    def get_explainer_result(self, explainer_id: str) -> e8s.ExplainerResult | None:
        return self.explainer_results.get(explainer_id)

    def register_explainer_result(self, explainer_id: str, result: e8s.ExplainerResult):
        self.explainer_results[explainer_id] = result

    def get_explanation_file_path(
        self,
        explanation_type: str,
        explanation_format: str,
        explainer_id: str = "",
        evaluator_id: str = "",
    ):
        """Get explanation (index) file path.

        Parameters
        ----------
        explainer_id : str
          Explainer ID - either explainer or evaluator ID must be specified.
        evaluator_id : str
          Evaluator ID - either explainer or evaluator ID must be specified.
        explanation_type : str
          Explanation type as string.
        explanation_format : str
          Explanation (MIME) format.

        Returns
        -------

        str :
          Path to the explanation file.

        """
        explainer_id = explainer_id or evaluator_id
        e_jobs = self.get_jobs_for_explainer_id(explainer_id)
        if not e_jobs:
            raise ValueError(
                f"Explainer/evaluator {explainer_id} was not run in this "
                f"interpretation/evaluation"
            )

        e_persistence = persistences.ExplainerPersistence(
            data_dir=self.result.results_location,
            mli_key=self.key,
            username=commons.DEFAULT_USER,
            explainer_id=explainer_id,
            explainer_job_key=e_jobs[0].key,
        )

        return e_persistence.get_explanation_file_path(
            explanation_type=explanation_type,
            explanation_format=explanation_format,
        )

    def to_dict(self) -> dict:
        return {
            Interpretation.KEY_CREATED: self.created,
            Interpretation.KEY_PROGRESS: self.progress,
            Interpretation.KEY_PROGRESS_MESSAGE: self.progress_message,
            Interpretation.KEY_STATUS: self.status.value,
            Interpretation.KEY_ERROR: self.error,
            Interpretation.KEY_I_KEY: self.key,
            Interpretation.KEY_I_PARAMS: self.common_params.to_dict(),
            Interpretation.KEY_EXPLAINERS: [{r.id: r.params} for r in self.explainers],
            Interpretation.KEY_RESULT: self.result.to_dict(),
        }

    def to_json(self, indent=None) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_html(self) -> str:
        return self.html_format.to_html()

    def to_html_4_pdf(self) -> str:
        return self.pdf_format.to_html_4_pdf()

    def to_pdf(self, input_path: str, output_path: str):
        self.pdf_format.to_pdf(input_path, output_path)

    @staticmethod
    def load_from_json(
        interpretation_json_path: str | pathlib.Path,
    ) -> "Interpretation":
        """Load interpretation from JSON."""
        logger = loggers.SonarPrintLogger()

        interpretation_json_path = pathlib.Path(interpretation_json_path)
        if not interpretation_json_path.exists():
            raise ValueError(
                f"Interpretation JSon file '{interpretation_json_path}' does not exist"
            )
        i_json = persistences.FilesystemPersistence().load_json(
            interpretation_json_path
        )

        # JSon to Interpretation instance
        common_params = commons.CommonInterpretationParams(
            model=None,
            models=None,
            dataset=None,
            target_col="",
        )
        if Interpretation.KEY_I_PARAMS in i_json:
            try:
                common_params = commons.CommonInterpretationParams.load(
                    i_json.get(Interpretation.KEY_I_PARAMS, {})
                )
            except Exception as ex:
                logger.warning(f"WARNING: Unable to load common parameters: {ex}")
        explainers = []
        if Interpretation.KEY_EXPLAINERS in i_json:
            for e_id in i_json.get(Interpretation.KEY_EXPLAINERS, []):
                # TODO IMPROVE load explainer parameters / ExplainerToRun structure
                explainers.append(commons.ExplainerToRun(explainer_id=next(iter(e_id))))

        # interpretation
        i = Interpretation(
            common_params=common_params,
            created=i_json.get(Interpretation.KEY_CREATED, 0.0),
            explainers=explainers,
            key=i_json.get(Interpretation.KEY_I_KEY, ""),
        )
        i.progress = i_json.get(Interpretation.KEY_PROGRESS, 0.0)
        i.status = commons.ExplainerJobStatus.from_int(
            i_json.get(
                Interpretation.KEY_STATUS, commons.ExplainerJobStatus.UNKNOWN.value
            )
        )
        i.error = i_json.get(Interpretation.KEY_ERROR, "")

        # interpretation result
        i.result = InterpretationResult(
            results_location=i_json.get(Interpretation.KEY_RESULT, {}).get(
                Interpretation.KEY_RESULTS_LOCATION, ""
            ),
            interpretation_location=i_json.get(Interpretation.KEY_RESULT, {}).get(
                Interpretation.KEY_INTERPRETATION_LOCATION, ""
            ),
        )
        # models
        models_json = i_json.get(Interpretation.KEY_RESULT, {}).get(
            Interpretation.KEY_MODELS
        )
        i.result.models = []
        if models_json:
            for m_json in models_json:
                try:
                    i.result.models.append(
                        m4s.explainable_rag_llm_model_from_json(m_json)
                    )
                except Exception as ex:
                    logger.warning(
                        f"WARNING: Unable to load model from JSon: {ex}\n'{m_json}'"
                    )

        # map: job key -> job
        i.result.explainers = {}
        for j_dict in i_json.get(Interpretation.KEY_RESULT, {}).get(
            Interpretation.KEY_EXECUTED_EXPLAINERS, []
        ):
            e_job = ExplainerJob.from_dict(j_dict)
            i.result.explainers[e_job.key] = e_job
        # problems
        problems_dicts = i_json.get(Interpretation.KEY_RESULT, {}).get(
            Interpretation.KEY_PROBLEMS, []
        )
        i.result.problems = []
        for p_dict in problems_dicts:
            i.result.problems.append(problems.ProblemAndAction.from_dict(p_dict))
        # insights
        insights_dicts = i_json.get(Interpretation.KEY_RESULT, {}).get(
            Interpretation.KEY_INSIGHTS, []
        )
        i.result.insights = []
        for p_dict in insights_dicts:
            i.result.insights.append(insights.InsightAndAction.from_dict(p_dict))
        # overall result
        i.result.overall_result = OverallResult[
            i_json.get(Interpretation.KEY_RESULT, {}).get(
                Interpretation.KEY_OVERALL_RESULT, OverallResult.no_problem.name
            )
        ]

        # async interpretation progress: try to load more detailed progress
        # in case of async interpretations/evaluations
        if i.progress < 1.0:
            progress_json_path = (
                interpretation_json_path.parent
                / persistences.InterpretationPersistence.FILE_PROGRESS_JSON
            )
            if progress_json_path.exists():
                try:
                    progress_json = persistences.FilesystemPersistence().load_json(
                        progress_json_path
                    )
                    i.progress = progress_json.get(progress_utils.KEY_PROGRESS, 0.0)
                    i.progress_message = progress_json.get(
                        progress_utils.KEY_MESSAGE, ""
                    )
                except Exception as ex:
                    logger.warning(
                        f"WARNING: Unable to load interpretation progress: {ex}"
                    )

        # TODO IMPROVE load the remaining content of the JSon file to data structures

        return i

    def load(self, persistence, logger=None) -> "Interpretation":
        """Load persistence interpretation using given persistence.

        Parameters
        ----------
        persistence : persistences.InterpretationPersistence
          Interpretation persistence which can be used to load the interpretation
          from file-system, memory or DB.
        logger :
          Logger.

        Returns
        -------
        Interpretation :
          Interpretation instance.

        """
        # TODO PRE: load common parameters
        # TODO PRE: load requested explainers
        # TODO POST: load interpretation result (via JSon)
        # TODO PRE: load resolved/.../run explainers

        raise NotImplementedError


class InterpretationResult:
    """Result of the interpretation run."""

    def __init__(
        self,
        results_location: str = "",
        interpretation_location: str = "",
    ):
        # results location (directory, DB, dict key); in case that user did not specify
        # the location in interpretation parameter, then it is resolved by container
        # (to a default store type) - this field is always specified and contains
        # valid location
        self.results_location = results_location
        # location of the interpretation in results store
        self.interpretation_location = interpretation_location
        # location of the interpretation HTML report
        self.html_location = None
        # location of the interpretation JSon report
        self.json_location = None
        # explainable dataset created from the dataset provided by the user
        self.dataset = None
        # explainable testset created from the dataset provided by the user
        self.testset = None
        # explainable validset created from the dataset provided by the user
        self.validset = None
        # explainable model created from the model provided by the user
        self.model = None
        # explainable (LLM) models
        self.models: list = []
        # explainers requested to run by the user
        self.all_explainer_ids: list = []
        # explainers incompatible with the model, dataset, etc.
        self.incompatible_explainer_ids: list = []
        # incompatible explainers descriptors for reporting: ID -> descriptor
        self.incompatible_explainers: dict = {}
        # explainers scheduled for run (search + compatibility check, prior actual run)
        self.explainer_ids: list = []
        # resolved explainers parameters
        self.explainers_params = {}
        # explainers: list of executed explainers (ID, status, explanations)
        self.explainers: dict[str, ExplainerJob] = {}
        # problems: list of model problems
        self.problems: list[problems.ProblemAndAction] = []
        # insights: list of insights
        self.insights: list[insights.InsightAndAction] = []
        # location / URL where was the interpretation (report) uploaded
        self.upload_url: str = ""
        # overall interpretation/evaluation result is a traffic light style status
        # which indicates the severity of problems which were found - overall result
        # is one of the following values:
        self.overall_result: OverallResult = OverallResult.no_problem

    def get_explainer_job(self, explainer_job_id: str) -> ExplainerJob | None:
        return self.explainers.get(explainer_job_id, None)

    def get_evaluator_job(self, evaluator_job_id: str) -> ExplainerJob | None:
        return self.explainers.get(evaluator_job_id, None)

    def get_explainer_jobs(self) -> list[ExplainerJob]:
        return list(self.explainers.values())

    def get_evaluator_jobs(self) -> list[ExplainerJob]:
        return list(self.explainers.values())

    def get_results_dir_location(self, absolute_path: bool = True) -> str:
        results_location = str(self.results_location) or ""
        return os.path.abspath(results_location) if absolute_path else results_location

    def get_interpretations_html_index_location(self, absolute_path: bool = True):
        return os.path.join(
            self.get_results_dir_location(absolute_path),
            persistences.InterpretationPersistence.FILE_H2O_SONAR_HTML,
        )

    def get_interpretation_dir_location(self, absolute_path: bool = True) -> str:
        interpretation_location = str(self.interpretation_location) or ""
        return (
            os.path.abspath(interpretation_location)
            if absolute_path
            else interpretation_location
        )

    def get_html_report_location(self, absolute_path: bool = True) -> str:
        return os.path.join(
            self.get_interpretation_dir_location(absolute_path),
            persistences.InterpretationPersistence.FILE_INTERPRETATION_HTML,
        )

    def get_pdf_report_location(self, absolute_path: bool = True) -> str:
        return os.path.join(
            self.get_interpretation_dir_location(absolute_path),
            persistences.InterpretationPersistence.FILE_INTERPRETATION_PDF,
        )

    def get_json_report_location(self, absolute_path: bool = True) -> str:
        return os.path.join(
            self.get_interpretation_dir_location(absolute_path),
            persistences.InterpretationPersistence.FILE_INTERPRETATION_JSON,
        )

    def get_progress_location(self, absolute_path: bool = True) -> str:
        return os.path.join(
            self.get_interpretation_dir_location(absolute_path),
            persistences.InterpretationPersistence.FILE_PROGRESS_JSON,
        )

    def to_dict(self) -> dict:
        explainers_dict = [self.explainers[j].to_dict() for j in self.explainers]
        if self.models:
            models_list = [m.to_dict() for m in self.models if hasattr(m, "to_dict")]
        else:
            models_list = []
        return {
            Interpretation.KEY_DATASET: (
                self.dataset.to_dict()
                if self.dataset and hasattr(self.dataset, "to_dict")
                else {}
            ),
            Interpretation.KEY_TESTSET: (
                self.testset.to_dict()
                if self.testset and hasattr(self.testset, "to_dict")
                else {}
            ),
            Interpretation.KEY_VALIDSET: (
                self.validset.to_dict()
                if self.validset and hasattr(self.validset, "to_dict")
                else {}
            ),
            Interpretation.KEY_MODEL: (
                self.model.to_dict()
                if self.model and hasattr(self.model, "to_dict")
                else {}
            ),
            Interpretation.KEY_MODELS: models_list,
            Interpretation.KEY_ALL_EXPLAINERS: self.all_explainer_ids,
            Interpretation.KEY_INCOMPATIBLE_EXPLAINERS: self.incompatible_explainer_ids,
            Interpretation.KEY_INCOMPATIBLE_EXPLAINERS_DS: (
                self.incompatible_explainers
            ),
            Interpretation.KEY_SCHEDULED_EXPLAINERS: self.explainer_ids,
            Interpretation.KEY_E_PARAMS: self.explainers_params,
            Interpretation.KEY_EXECUTED_EXPLAINERS: explainers_dict,
            Interpretation.KEY_PROBLEMS: [p.to_dict() for p in self.problems],
            Interpretation.KEY_INSIGHTS: [i.to_dict() for i in self.insights],
            Interpretation.KEY_OVERALL_RESULT: self.overall_result.name,
            Interpretation.KEY_RESULTS_LOCATION: str(self.results_location),
            Interpretation.KEY_INTERPRETATION_LOCATION: str(
                self.interpretation_location
            ),
        }

    def to_json(self, indent=None) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def make_zip_archive(self, zip_filename):
        persistences.FilesystemPersistence().make_dir_zip_archive(
            src_key=self.results_location, zip_key=zip_filename
        )

    def remove_duplicate_insights(self):
        seen_insights = {
            abc_ins.ATTR_CHEAPEST_MODEL_NAME: False,
            abc_ins.ATTR_MOST_EXPENSIVE_MODEL_NAME: False,
            abc_ins.ATTR_SLOWEST_MODEL_NAME: False,
            abc_ins.ATTR_FASTEST_MODEL_NAME: False,
        }
        for key in seen_insights:
            self._remove_duplicate_insights_by_attr(seen_insights, key)

    def _remove_duplicate_insights_by_attr(self, seen_insights: dict, attr_name: str):
        for i in self.insights:
            if i.insight_attrs.get(attr_name):
                if not seen_insights[attr_name]:
                    seen_insights[attr_name] = True
                else:
                    self.insights.remove(i)

    def __str__(self):
        return f"InterpretationResult:\n{self.to_json(2)}"


class HtmlInterpretationFormat:
    """HTML representation of the interpretation."""

    def __init__(
        self,
        interpretation: Interpretation,
        branding: commons.Branding = commons.Branding.H2O_SONAR,
        logger: loggers.SonarLogger | None = None,
    ):
        self.i = interpretation
        self.logger = logger or loggers.SonarPrintLogger()

        # branding: H2O Sonar vs. Eval Studio
        self.branding = branding
        if self.branding == commons.Branding.H2O_SONAR:
            self.brand_h2o_sonar = "H2O Sonar"
            self.brand_report = "Interpretation Report"
            self.brand_m_i_report = "Model Interpretation Report"
            self.brand_interpretation = "Interpretation"
            self.brand_interpreted = "interpre"
            self.brand_explainer = "Explainer"
        else:
            self.brand_h2o_sonar = "Eval Studio"
            self.brand_report = "Evaluation Report"
            self.brand_m_i_report = "LLM Evaluation Report"
            self.brand_interpretation = "Evaluation"
            self.brand_interpreted = "evalua"
            self.brand_explainer = "Evaluator"

    class Context:
        """Context with data which are needed to create HTML."""

        def __init__(self):
            # explainer purpose (keyword) > explainer job
            self.explainers_by_purpose: dict[str, list[ExplainerJob]] = {}
            # 1 representative explainer job for every purpose: purpose > explainer job
            self.purpose_representative: dict[str, ExplainerJob | None] = {}
            # image for purpose representative: job key > path to image
            self.purpose_image: dict[str, str] = {}
            # purposes w/ a representative
            self.purposes_w_repre: list[str] = []
            # UUID (anchor) > HTML fragment (HTML w/ image URLs)
            self.explanations_html_fragments = {}

        def clear(self):
            self.explainers_by_purpose = {}
            self.purpose_representative = {}
            self.purpose_image = {}
            self.purposes_w_repre = []
            self.explanations_html_fragments = {}

        def get_purpose_representatives_job_keys(self) -> list[str]:
            return [self.purpose_representative[p].key for p in self.purposes_w_repre]

    @staticmethod
    def _html_parse_1st_img_path(html_representation: str) -> str:
        if html_representation:
            try:
                lexem = ' src="'
                begin_idx = html_representation.index(lexem)
                end_idx = html_representation.index('"', begin_idx + len(lexem))
                path = html_representation[begin_idx + len(lexem) : end_idx]
                return path
            except ValueError:
                pass

        return ""

    def _prepare_data(self) -> Context:
        """Prepare data to be used for HTML report creation."""
        ctx = HtmlInterpretationFormat.Context()

        for p in e8s.Explainer.EXPLAINERS_PURPOSES:
            ctx.explainers_by_purpose[p] = []
            ctx.purpose_representative[p] = None

        for e_id in self.i.get_successful_explainer_ids():
            e_jobs = self.i.get_jobs_for_explainer_id(e_id)
            for e_job in e_jobs:
                if e_job:
                    is_sorted = False
                    if e_job.explainer_descriptor.keywords:
                        for p in e8s.Explainer.EXPLAINERS_PURPOSES:
                            if p in e_job.explainer_descriptor.keywords:
                                # all: explainer purpose > explainer jobs
                                ctx.explainers_by_purpose[p].append(e_job)
                                # representative: explainer purpose > explainer job
                                if not ctx.purpose_representative[p]:
                                    ctx.purpose_representative[p] = e_job
                                    ctx.purposes_w_repre.append(p)
                                is_sorted = True
                                break

                    if not is_sorted:
                        # all: explainer purpose > explainer jobs
                        ctx.explainers_by_purpose[
                            e8s.Explainer.KEYWORD_EXPLAINS_UNKNOWN
                        ].append(e_job)

        # index & save explanations (MIME) and images ~ available to overviews
        for explainer_id in self.i.get_finished_explainer_ids():
            r = self.i.get_explainer_result_metadata(explainer_id)
            # TODO OPTIMIZE the code below and avoid duplication w/ explainer section
            if r:
                try:
                    job_key = r.get("key", None)
                    e_persistence = persistences.ExplainerPersistence(
                        data_dir=self.i.persistence.store.base_path,
                        username=commons.DEFAULT_USER,
                        explainer_id=explainer_id,
                        explainer_job_key="" if not job_key else str(job_key),
                        mli_key=str(self.i.key),
                        store_persistence=self.i.persistence,
                    )
                    t_explainer_persistence = persistences.ExplainerPersistence
                    e_file = e_persistence.get_explanation_file_path(
                        explanation_type=(
                            e9s.GlobalHtmlFragmentExplanation.explanation_type()
                        ),
                        explanation_format=formats.HtmlFormat.mime,
                        explanation_file=(
                            f"{t_explainer_persistence.FILE_EXPLANATION}.html"
                        ),
                    )
                    include_mark = str(uuid.uuid4())
                    ctx.explanations_html_fragments[include_mark] = (
                        self.i.persistence.store.load(
                            key=e_file,
                            data_type=persistences.PersistenceDataType.text,
                        )
                    )

                    # parse path to the first image from HTML for representatives
                    if job_key:
                        ctx.purpose_image[job_key] = (
                            HtmlInterpretationFormat._html_parse_1st_img_path(
                                ctx.explanations_html_fragments[include_mark]
                            )
                        )

                except Exception as ex:
                    msg = (
                        f"Unable to index explanations HTML fragment of explainer "
                        f"{explainer_id}: {ex}"
                    )
                    self.logger.warning(msg)

        return ctx

    def to_html(
        self,
        include_left_navigation: bool = True,
        report_style: str = "HTML",
    ) -> str:
        """Get HTML report for the interpretation."""
        ctx = self._prepare_data()

        html = airium.Airium()
        html("<!DOCTYPE html>")
        with html.html(lang="en"):
            self._html_head(html)

            with html.body(klass="w3-light-grey w3-content", style="max-width:1600px"):
                if include_left_navigation:
                    self._html_left_navigation(ctx, html)
                self._html_right_main(
                    ctx=ctx,
                    html=html,
                    report_style=report_style,
                )
                self._html_javascript(html)

        html_str = str(html)
        for m in ctx.explanations_html_fragments:
            # set HTML fragment in every explainer section
            html_str = html_str.replace(m, ctx.explanations_html_fragments[m])

        return html_str

    KEYWORD_ID_2_NAME = {
        e8s.Explainer.KEYWORD_EXPLAINS_APPROX_BEHAVIOR: "Approximate model behavior",
        e8s.Explainer.KEYWORD_EXPLAINS_O_FEATURE_IMPORTANCE: (
            "Original feature importance"
        ),
        e8s.Explainer.KEYWORD_EXPLAINS_T_FEATURE_IMPORTANCE: (
            "Transformed feature importance"
        ),
        e8s.Explainer.KEYWORD_EXPLAINS_FEATURE_BEHAVIOR: "Feature behavior",
        e8s.Explainer.KEYWORD_EXPLAINS_FAIRNESS: "Fairness",
        e8s.Explainer.KEYWORD_EXPLAINS_MODEL_DEBUGGING: "Model debugging",
        e8s.Explainer.KEYWORD_EXPLAINS_UNKNOWN: "Model explanations",
    }

    @staticmethod
    def _purpose_id_to_name(keyword: str) -> str:
        return HtmlInterpretationFormat.KEYWORD_ID_2_NAME.get(
            keyword, "Model explanations"
        )

    @staticmethod
    def html_safe_str_field(field):
        import html

        return html.escape(field) if field and isinstance(field, str) else field

    def _html_head(self, a):
        """Create HTML head for this interpretation.

        Parameters
        ----------
        a : airium.Airium
          Airium HTML instance.

        """

        htmls.evaluation_report_html_head(
            a,
            title=(
                f"{self.brand_h2o_sonar} Model {self.brand_interpretation} "
                f"Report {self.i.key}"
            ),
        )

    @staticmethod
    def _html_javascript(html):
        """Inject JavaScript code into Airium HTML instance.

        Parameters
        ----------
        html : airium.Airium
          Airium HTML instance.

        """
        with html.script():
            html(
                "// toggle explainers overview: ALL vs. REPRESENTATIVES\n"
                "function showExplainersPreviewAll() {\n"
                "  // show ALL panel\n"
                '  document.getElementById("explainers-overview-representatives")'
                '.style.display = "block";\n'
                "  // hide REPRESENTATIVES panel\n"
                '  document.getElementById("explainers-overview-all")'
                '.style.display = "none";\n'
                "  // toggle button\n"
                "  document.getElementById("
                '"explainers-overview-button-all")'
                '.className = "w3-tag w3-black w3-margin-bottom";\n'
                "  document.getElementById("
                '"explainers-overview-button-representatives")'
                '.className = "w3-tag w3-light-gray w3-margin-bottom";\n'
                "}\n"
                "\n"
                "function showExplainersPreviewRepresentatives() {\n"
                "  // hide REPRESENTATIVES panel\n"
                '  document.getElementById("explainers-overview-all")'
                '.style.display = "block";\n'
                "  // show ALL panel\n"
                '  document.getElementById("explainers-overview-representatives")'
                '.style.display = "none";\n'
                "  // toggle button\n"
                "  document.getElementById("
                '"explainers-overview-button-representatives")'
                '.className = "w3-tag w3-black w3-margin-bottom";\n'
                "  document.getElementById("
                '"explainers-overview-button-all")'
                '.className = "w3-tag w3-light-gray w3-margin-bottom";\n'
                "}\n"
                "\n"
            )

    def _html_left_navigation_content(
        self, ctx: Context, html, navi_style: str = "TOC"
    ):
        with html.div(klass="w3-bar-block"):
            if navi_style == "TOC":
                with html.p(klass="w3-text-black"):
                    html.b(_t="Summary")

            if self.branding == commons.Branding.H2O_SONAR:
                with html.a(
                    klass="w3-bar-item w3-button w3-padding w3-margin-left",
                    href="#explainers-overview",
                ):
                    html(f"{self.brand_explainer}s overview")

            with html.a(
                klass="w3-bar-item w3-button w3-padding w3-margin-left",
                href="#problems",
            ):
                html("Problems (")
                with html.span(klass="w3-text-red"):
                    html(
                        len(
                            self.i.get_problems_by_severity(
                                problems.ProblemSeverity.high
                            )
                        )
                    )
                html("/")
                with html.span(klass="w3-text-orange"):
                    html(
                        len(
                            self.i.get_problems_by_severity(
                                problems.ProblemSeverity.medium
                            )
                        )
                    )
                html("/")
                with html.span(klass="w3-text-yellow"):
                    html(
                        len(
                            self.i.get_problems_by_severity(
                                problems.ProblemSeverity.low
                            )
                        )
                    )
                html(")")

            with html.a(
                klass="w3-bar-item w3-button w3-padding w3-margin-left",
                href="#insights",
            ):
                html("Insights (")
                with html.span(klass="w3-text-green"):
                    html(len(self.i.get_insights()))
                html(")")

            with html.a(
                klass="w3-bar-item w3-button w3-padding w3-margin-left",
                href="#interpretation",
            ):
                html(f"{self.brand_explainer}s ({len(self.i.get_all_explainer_ids())}/")
                with html.span(klass="w3-text-yellow"):
                    html(f"{len(self.i.get_scheduled_explainer_ids())}")
                html("/")
                with html.span(klass="w3-text-green"):
                    html(f"{len(self.i.get_successful_explainer_ids())}")
                html("/")
                with html.span(klass="w3-text-red"):
                    html(f"{len(self.i.get_failed_explainer_ids())}")
                html(")")

        # explainers by PURPOSE
        if self.i.is_explainer_scheduled():
            for p in ctx.explainers_by_purpose:
                if ctx.explainers_by_purpose[p]:
                    with html.div(klass="w3-container"):
                        if navi_style == "TOC":
                            with html.p(klass="w3-text-black"):
                                html.b(_t=f"{self._purpose_id_to_name(p)}")
                        else:
                            html.p(
                                klass="w3-text-yellow",
                                _t=f"{self._purpose_id_to_name(p)}",
                            )

                    for e_job in ctx.explainers_by_purpose[p]:
                        with html.div(klass="w3-bar-block"):
                            e_display_name = e_job.explainer_descriptor.display_name
                            with html.a(
                                klass=(
                                    "w3-bar-item w3-button w3-padding w3-margin-left"
                                ),
                                href=f"#{e_job.explainer_descriptor.id}",
                                title=e_display_name,
                            ):
                                html(
                                    HtmlInterpretationFormat._html_txt_ellipsis(
                                        text=e_display_name,
                                        lng=30,
                                        strategy="middle",
                                    )
                                )

            with html.div(klass="w3-container"):
                if navi_style == "TOC":
                    html.p(klass="w3-text-black", _t=self.brand_interpretation)
                else:
                    html.p(klass="w3-text-yellow", _t=self.brand_interpretation)

            with html.div(klass="w3-bar-block"):
                with html.a(
                    klass="w3-bar-item w3-button w3-padding w3-margin-left",
                    href="#dataset",
                ):
                    html("Dataset")
                if self.i.common_params.testset is not None:
                    with html.a(
                        klass="w3-bar-item w3-button w3-padding w3-margin-left",
                        href="#testset",
                    ):
                        html("Test dataset")
                if self.i.common_params.validset is not None:
                    with html.a(
                        klass="w3-bar-item w3-button w3-padding w3-margin-left",
                        href="#validset",
                    ):
                        html("Validation dataset")
                with html.a(
                    klass="w3-bar-item w3-button w3-padding w3-margin-left",
                    href="#model",
                ):
                    if self.i.result.to_dict().get("models", {}):
                        html("Models")
                    else:
                        html("Model")
                with html.a(
                    klass="w3-bar-item w3-button w3-padding w3-margin-left",
                    href="#parameters",
                ):
                    html("Configuration and parameters")
                with html.a(
                    klass="w3-bar-item w3-button w3-padding w3-margin-left",
                    href="#artifacts",
                ):
                    html("Directories, files and logs")

        with html.div(klass="w3-panel w3-large"):
            html.i(klass="fa fa-facebook-official w3-hover-opacity")
            html.i(klass="fa fa-instagram w3-hover-opacity")
            html.i(klass="fa fa-snapchat w3-hover-opacity")
            html.i(klass="fa fa-pinterest-p w3-hover-opacity")
            html.i(klass="fa fa-twitter w3-hover-opacity")
            html.i(klass="fa fa-linkedin w3-hover-opacity")

    def _html_right_main_toc(self, ctx: Context, html):
        """Create Table of Contents.

        Parameters
        ----------
        ctx : Context
          This interpretation context.
        html : airium.Airium
          Airium HTML instance.

        """
        with html.div(
            klass="w3-container w3-padding-large",
            style="margin-bottom:32px",
        ):
            html.h2(id="toc", _t="Table of Contents")

            self._html_left_navigation_content(ctx, html, navi_style="TOC")

    def _html_left_navigation(self, ctx: Context, html):
        """Create left navigation panel of HTML representation.

        Parameters
        ----------
        ctx : Context
          This interpretation context.
        html : airium.Airium
          Airium HTML instance.

        """
        with html.nav(
            klass="w3-sidebar w3-collapse w3-black w3-animate-left",
            id="mySidebar",
            style="z-index:3;width:300px;",
        ):
            html.br()
            with html.div(klass="w3-container"):
                with html.a(klass="w3-left w3-margin-right", href="#"):
                    htmls.html_svg_h2oai_logo(html)
                with html.h3():
                    html.b(
                        _t=self.brand_h2o_sonar,
                        title=HtmlInterpretationFormat.html_h2o_sonar_pitch(
                            self.brand_h2o_sonar
                        ),
                        style="color: #fec925",
                    )
                html.h4(_t=self.brand_m_i_report)
                html.p(klass="w3-text-yellow", _t="Summary")

            self._html_left_navigation_content(ctx, html, navi_style="HTML")

    def _html_cfg_items(
        self,
        params: dict,
        description_src,
        html,
        as_table: bool = False,
        as_cfg_item: bool = False,
    ):
        """Inject configuration to Airium HTML.

        Parameters
        ----------
        params : dict
          Dictionary of all parameters.
        description_src :
          From where to load descriptions of parameters.
        html : airium.Airium
          Airium HTML instance.
        as_table : bool
          Render parameters as table or unordered list.
        as_cfg_item : bool
          Render as configuration item.

        """
        with html.ul():
            if as_table and params.keys():
                with html.table(klass="w3-table-all"):
                    with html.tr():
                        html.th(_t="Config item" if as_cfg_item else "Config parameter")
                        html.th(_t="Value")
                        html.th(_t="Description")
                        html.th(_t="Type")
                        html.th(_t="Default value")

                    for param_name in params.keys():
                        param_doc = description_src.describe_config_item(param_name)
                        self._html_cfg_item(
                            param_name=param_name,
                            param_doc=param_doc,
                            params=params,
                            html=html,
                            as_table_row=as_table,
                        )
            else:
                for param_name in params.keys():
                    param_doc = description_src.describe_config_item(param_name)
                    self._html_cfg_item(
                        param_name=param_name,
                        param_doc=param_doc,
                        params=params,
                        html=html,
                        as_table_row=as_table,
                    )

    def _html_cfg_item(
        self,
        param_name: str,
        param_doc,
        params: dict,
        html,
        as_table_row: bool = False,
    ):
        """Inject one configuration item to Airium HTML.

        Parameters
        ----------
        param_name : str,
          Parameter name.
        param_doc
          Parameter documentation.
        params : dict
          Parameters dictionary.
        html : airium.Airium
          Airium HTML instance.
        as_table_row : bool
          Whether to render it as table row, or unordered list item.

        """
        if as_table_row:
            with html.tr():
                with html.td():
                    html(f"{param_name}")
                with html.td():
                    param_value = self.html_safe_str_field(params.get(param_name, ""))
                    with html.code():
                        html(f"{param_value}")
                with html.td():
                    html(param_doc.description if param_doc else "")
                with html.td():
                    with html.code():
                        html(param_doc.param_type.name if param_doc else "")
                with html.td():
                    with html.code():
                        html(param_doc.default_value if param_doc else "")

        else:
            with html.li():
                with html.b():
                    html(f"{param_name} = ")
                param_value = self.html_safe_str_field(params.get(param_name, ""))
                with html.code():
                    html(f"{param_value}")

                with html.ul():
                    with html.li():
                        with html.i():
                            html(param_doc.description if param_doc else "")

                    with html.ul():
                        with html.li():
                            html("Type: ")
                            with html.code():
                                html(param_doc.param_type.name if param_doc else "")
                        with html.li():
                            html("Default value: ")
                            with html.code():
                                html(param_doc.default_value if param_doc else "")

    def _html_e_param(
        self,
        explainer_id: str,
        e_result_descriptor,
        html,
        as_table: bool = False,
    ):
        """Inject explainer parameter documentation to Airium HTML.

        Parameters
        ----------
        explainer_id : str
          Explainer ID.
        e_result_descriptor :
          Explainer result descriptor.
        html : airium.Airium
          Airium HTML instance.
        as_table : bool
          Whether to render it as table row or unordered list item.

        """
        es = self.i.result.explainers_params or {}
        params = es.get(explainer_id, {})
        if params:
            e_params_doc_list = e_result_descriptor.get("parameters", [])
            e_params_doc = {}
            for p in e_params_doc_list:
                e_params_doc[p.get("name", "")] = p

            if as_table:
                with html.table(klass="w3-table-all"):
                    with html.tr():
                        html.th(_t="Parameter")
                        html.th(_t="Value")
                        html.th(_t="Description")
                        html.th(_t="Type")
                        html.th(_t="Default value")

                    for param_name in params.keys():
                        p_doc_dict = e_params_doc.get(param_name, {})

                        with html.tr():
                            with html.td():
                                html(f"{param_name}")
                            with html.td():
                                param_value = self.html_safe_str_field(
                                    params.get(param_name, "")
                                )
                                with html.code():
                                    html(f"{param_value}")
                            with html.td():
                                html(p_doc_dict.get("description", ""))
                            with html.td():
                                with html.code():
                                    html(p_doc_dict.get("type", ""))
                            with html.td():
                                with html.code():
                                    html(p_doc_dict.get("val", ""))

            else:
                with html.ul():
                    for param_name in params.keys():
                        with html.li():
                            with html.b():
                                html(f"{param_name} = ")
                            param_value = HtmlInterpretationFormat.html_safe_str_field(
                                params.get(param_name, "")
                            )
                            with html.code():
                                html(f"{param_value}")
                            p_doc_dict = e_params_doc.get(param_name, {})
                            if p_doc_dict:
                                with html.ul():
                                    with html.li():
                                        with html.i():
                                            html(p_doc_dict.get("description", ""))
                                    with html.ul():
                                        with html.li():
                                            html("Type: ")
                                            with html.code():
                                                html(p_doc_dict.get("type", ""))
                                        with html.li():
                                            html("Default value: ")
                                            with html.code():
                                                html(p_doc_dict.get("val", ""))

    def _html_problems_table(
        self,
        problem_list: list[problems.ProblemAndAction],
        html,
        plural: bool = True,
        root_problem_list: bool = True,
    ):
        """Inject problems table into Airium HTML.

        Parameters
        ----------
        problem_list: list[problems.ProblemAndAction],
          List of problems.
        html : airium.Airium
          Airium HTML instance.
        plural : bool
          Do use plural.

        """

        if problem_list:
            with html.p():
                html(
                    f"{self.brand_explainer}{'s' if plural else ''} "
                    f"identified the following problems:"
                )
            with html.table(klass="w3-table-all"):
                with html.tr():
                    html.th(_t="Severity")
                    html.th(_t="Type")
                    html.th(_t="Problem")
                    html.th(_t="Suggested actions")
                    html.th(_t=self.brand_explainer)
                    html.th(_t="Resources")

                for p in problem_list:
                    with html.tr():
                        with html.td():
                            severity_color = (
                                "red"
                                if p.severity.name == problems.ProblemSeverity.high.name
                                else (
                                    "orange"
                                    if p.severity.name
                                    == problems.ProblemSeverity.medium.name
                                    else "yellow"
                                )
                            )
                            html.b(
                                klass=f"w3-{severity_color}",
                                _t=(f"&nbsp;{str(p.severity.name).upper()}&nbsp;"),
                            )
                        html.td(_t=p.problem_type)
                        with html.td():
                            if p.description_html:
                                html.append(p.description_html)
                            else:
                                html.b(_t=p.description)
                        html.td(_t=p.actions_description)
                        with html.td():
                            html.a(href=f"#{p.explainer_id}", _t=p.explainer_name)
                        with html.td():
                            if p.explanation_name or p.explanation_mime:
                                html.a(
                                    href=f"#e-explanations-data-{p.explainer_id}",
                                    _t=f"{p.explanation_name} / {p.explanation_mime}",
                                )
        elif root_problem_list:
            html(
                f"Explainer{'s' if plural else ''} did not find any problems with the "
                f"{self.brand_interpreted}ted model(s)."
            )

    def _html_insights_table(
        self,
        insight_list: list[insights.InsightAndAction],
        html,
        plural: bool = True,
        root_problem_list: bool = True,
    ):
        """Inject insights table into Airium HTML.

        Parameters
        ----------
        insight_list: list[problems.ProblemAndAction],
          List of problems.
        html : airium.Airium
          Airium HTML instance.
        plural : bool
          Do use plural.

        """

        if insight_list:
            with html.p():
                html(
                    f"{self.brand_explainer}{'s' if plural else ''} "
                    f"identified the following insights:"
                )
            with html.table(klass="w3-table-all"):
                with html.tr():
                    html.th(_t="Type")
                    html.th(_t="Insight")
                    html.th(_t="Suggested actions")
                    html.th(_t=self.brand_explainer)
                    html.th(_t="Resources")

                for i in insight_list:
                    with html.tr():
                        html.td(_t=i.insight_type)
                        with html.td():
                            if i.description_html:
                                html.append(i.description_html)
                            else:
                                html.b(_t=i.description)
                        html.td(_t=i.actions_description)
                        with html.td():
                            html.a(href=f"#{i.explainer_id}", _t=i.explainer_name)
                        with html.td():
                            html.a(
                                href=f"#e-explanations-data-{i.explainer_id}",
                                _t=f"{i.explanation_name} / {i.explanation_mime}",
                            )
        elif root_problem_list:
            html(
                f"Explainer{'s' if plural else ''} did not find any insights related "
                f"to the {self.brand_interpreted}ted model(s)."
            )

    @staticmethod
    def _html_dataset_sampling(dataset_dict, html):
        sampled = dataset_dict.get("metadata", {}).get(
            "original_dataset_sampled", False
        )
        if sampled:
            self_type = HtmlInterpretationFormat

            file_path = dataset_dict.get("metadata", {}).get(
                "original_dataset_path", ""
            )
            with html.li():
                html.b(_t="Original dataset sampled")
                html(" = ")
                with html.code():
                    html("True")
            if file_path:
                with html.li():
                    html.b(_t="Original dataset file")
                    html(" = ")
                    with html.code():
                        html(self_type.html_safe_str_field(file_path))
                    with html.ul():
                        with html.li():
                            with html.i():
                                html("Path to the dataset.")
            file_size = dataset_dict.get("metadata", {}).get("original_dataset_size", 0)
            if file_size:
                with html.li():
                    html.b(_t="Original dataset size")
                    html(" = ")
                    with html.code():
                        html(f"{self_type.html_safe_str_field(file_size)}B")
                    with html.ul():
                        with html.li():
                            with html.i():
                                html("Size of the dataset.")

            dataset_shape = dataset_dict.get("metadata", {}).get(
                "original_dataset_shape", ""
            )
            if dataset_shape:
                with html.li():
                    html.b(_t="Original dataset frame shape")
                    html(" = ")
                    with html.code():
                        html(
                            HtmlInterpretationFormat.html_safe_str_field(dataset_shape)
                        )
                    with html.ul():
                        with html.li():
                            with html.i():
                                html("Dataset data frame shape.")

    @staticmethod
    def _html_col_meta(c_meta, html):
        fields = [
            "data_type",
            "logical_types",
            "values_format",
            "is_id",
            "is_numeric",
            "is_categorical",
            "count",
            "frequency",
            "unique",
            "max_value",
            "min_value",
            "mean",
            "std",
            "histogram_counts",
            "histogram_ticks",
        ]
        labels = [
            "Data type",
            "Logical types",
            "Values format",
            "ID column",
            "Numeric column",
            "Categorical column",
            "Count",
            "Frequency",
            "Unique",
            "Max value",
            "Min value",
            "Mean",
            "Standard deviation",
            "Histogram counts",
            "Histogram ticks",
        ]
        value = c_meta.get("name", "")
        if value:
            with html.li():
                with html.code():
                    html(HtmlInterpretationFormat.html_safe_str_field(value))
                with html.ul():
                    for i, field in enumerate(fields):
                        value = c_meta.get(field, "")
                        if value:
                            with html.li():
                                html.b(_t=f"{labels[i]}")
                                html(" = ")
                                with html.code():
                                    html(
                                        HtmlInterpretationFormat.html_safe_str_field(
                                            value
                                        )
                                    )

    def _html_right_main_dataset(self, html, dataset_type: str = "Dataset"):
        # shortcuts
        self_type = HtmlInterpretationFormat

        if self.i.common_params.dataset is not None and isinstance(
            self.i.common_params.dataset, d6s.LlmDataset
        ):
            prompts = self.i.common_params.dataset.prompts()
            if prompts:
                with html.p():
                    html(f"{self.brand_interpretation} test suite details:")

                    prompts.sort()
                    with html.table(klass="w3-table-all"):
                        with html.tr():
                            html.th(_t=f"Prompts ({len(prompts)})")
                        limit = 100
                        for e, i in enumerate(prompts):
                            with html.tr():
                                html.td(_t=i)
                            if e > limit:
                                break

        # dataset / testset/ validset
        html(f"{dataset_type} description:")
        with html.ul():
            dataset_dict = self.i.result.to_dict().get(dataset_type.lower(), {})
            if dataset_dict:
                with html.li():
                    html.b(_t=dataset_type)
                    html(" = ")
                    with html.code():
                        html(
                            HtmlInterpretationFormat.html_safe_str_field(
                                dataset_dict.get("data", "")
                            )
                        )

                    file_path = dataset_dict.get("metadata", {}).get("file_path", "")
                    if file_path:
                        with html.li():
                            html.b(_t=f"{dataset_type} file path")
                            html(" = ")
                            with html.code():
                                html(self_type.html_safe_str_field(file_path))
                            with html.ul():
                                with html.li():
                                    with html.i():
                                        html(f"Path to the {dataset_type.lower()}.")
                    file_name = dataset_dict.get("metadata", {}).get("file_name", "")
                    if file_name:
                        with html.li():
                            html.b(_t="Dataset name")
                            html(" = ")
                            with html.code():
                                html(self_type.html_safe_str_field(file_name))
                            with html.ul():
                                with html.li():
                                    with html.i():
                                        html("Name of the dataset.")
                    file_size = dataset_dict.get("metadata", {}).get("file_size", 0)
                    if file_size:
                        with html.li():
                            html.b(_t="Dataset size")
                            html(" = ")
                            with html.code():
                                fsize = self_type.html_safe_str_field(file_size)
                                html(f"{fsize}B")
                            with html.ul():
                                with html.li():
                                    with html.i():
                                        html("Size of the dataset.")
                    with html.ul():
                        with html.li():
                            with html.i():
                                html("Dataset data frame type.")
                dataset_shape = dataset_dict.get("metadata", {}).get("shape", "")
                if dataset_shape:
                    with html.li():
                        html.b(_t="Dataset frame shape")
                        html(" = ")
                        with html.code():
                            html(
                                HtmlInterpretationFormat.html_safe_str_field(
                                    dataset_shape
                                )
                            )
                        with html.ul():
                            with html.li():
                                with html.i():
                                    html("Dataset data frame shape.")
                # sampling: show original dataset metadata if sampled
                HtmlInterpretationFormat._html_dataset_sampling(dataset_dict, html)
                with html.li():
                    html.b(_t="Row count")
                    html(" = ")
                    with html.code():
                        html(
                            HtmlInterpretationFormat.html_safe_str_field(
                                dataset_dict.get("metadata", {}).get("row_count", "")
                            )
                        )
                    with html.ul():
                        with html.li():
                            with html.i():
                                html("The number of the dataset rows.")
                with html.li():
                    html.b(_t="Column names")
                    html(" = ")
                    with html.code():
                        html(
                            HtmlInterpretationFormat.html_safe_str_field(
                                dataset_dict.get("metadata", {}).get("column_names", "")
                            )
                        )
                    with html.ul():
                        with html.li():
                            with html.i():
                                html("Dataset column names.")
                with html.li():
                    html.b(_t="Unique values")
                    html(" = ")
                    with html.code():
                        html(
                            HtmlInterpretationFormat.html_safe_str_field(
                                dataset_dict.get("metadata", {}).get(
                                    d6s.ExplainableDatasetMeta.KEY_COLUMN_UNIQUES, ""
                                )
                            )
                        )
                    with html.ul():
                        with html.li():
                            with html.i():
                                html("The number of unique values for dataset columns.")
                with html.li():
                    html.b(_t="Column types")
                    html(" = ")
                    with html.code():
                        html(
                            HtmlInterpretationFormat.html_safe_str_field(
                                dataset_dict.get("metadata", {}).get("column_types", "")
                            )
                        )
                    with html.ul():
                        with html.li():
                            with html.i():
                                html("Dataset column types.")
                dataset_cats = dataset_dict.get("metadata", {}).get("columns_cat", "")
                if dataset_cats:
                    with html.li():
                        html.b(_t="Categorical columns")
                        html(" = ")
                        with html.code():
                            html(
                                HtmlInterpretationFormat.html_safe_str_field(
                                    dataset_cats
                                )
                            )
                        with html.ul():
                            with html.li():
                                with html.i():
                                    html("The list of categorical columns.")
                dataset_nums = dataset_dict.get("metadata", {}).get("columns_num", "")
                if dataset_nums:
                    with html.li():
                        html.b(_t="Numeric columns")
                        html(" = ")
                        with html.code():
                            html(
                                HtmlInterpretationFormat.html_safe_str_field(
                                    dataset_nums
                                )
                            )
                        with html.ul():
                            with html.li():
                                with html.i():
                                    html("The list of numeric columns.")
                dataset_cmeta = dataset_dict.get("metadata", {}).get("columns_meta", [])
                if dataset_cmeta:
                    with html.li():
                        html.b(_t="Dataset columns:")
                        with html.ul():
                            for c_meta in dataset_cmeta:
                                HtmlInterpretationFormat._html_col_meta(c_meta, html)
                            with html.li():
                                with html.i():
                                    html("Dataset columns metadata.")
            else:
                with html.li():
                    html.code(_t=f"{self.i.common_params.dataset}")

    def _html_right_main_content(self, ctx: Context, html):
        with html.div(
            klass="w3-container w3-padding-large", style="margin-bottom:32px"
        ):
            if self.branding == commons.Branding.H2O_SONAR:
                html.h2(
                    id="explainers-overview", _t=f"{self.brand_explainer}s overview"
                )

                with html.div(
                    id="explainers-overview-button-all",
                    klass="w3-tag w3-black w3-margin-bottom",
                    style="cursor: pointer;",
                    onclick="showExplainersPreviewAll()",
                ):
                    html("Representatives")

                with html.span(
                    id="explainers-overview-button-representatives",
                    klass="w3-tag w3-light-gray w3-margin-bottom fa fa-hand-pointer-o",
                    style="cursor: pointer;",
                    onclick="showExplainersPreviewRepresentatives()",
                ):
                    html(f"All {self.brand_explainer.lower()}s")

                # VIEW 1) hideable paragraph w/ per-explainer type REPRESENTATIVE
                with html.div(id="explainers-overview-representatives"):
                    data_e_overview = []
                    sorted_purposes = [
                        p
                        for p in e8s.Explainer.EXPLAINERS_PURPOSES
                        if p in ctx.purposes_w_repre
                    ]
                    for i, e_job_id in enumerate(sorted_purposes):
                        if (i % 3) == 0:
                            data_e_overview.append([])
                        data_e_overview[int(i / 3)].append(e_job_id)

                    for row in data_e_overview:
                        with html.div(klass="w3-row-padding"):
                            for purpose_key in row:
                                e_job = ctx.purpose_representative.get(purpose_key)
                                if e_job:
                                    self._html_explainer_overview_tile(
                                        e_job=e_job,
                                        purpose_key=purpose_key,
                                        ctx=ctx,
                                        html=html,
                                    )
                # VIEW 2) hideable paragraph w/ ALL explainers preview
                with html.div(id="explainers-overview-all", style="display: none;"):
                    data_e_overview = []
                    for i, e_id in enumerate(self.i.get_successful_explainer_ids()):
                        e_jobs = self.i.get_jobs_for_explainer_id(e_id)
                        if e_jobs:
                            if (i % 3) == 0:
                                data_e_overview.append([])
                            data_e_overview[int(i / 3)].append(e_jobs[0])

                    for row in data_e_overview:
                        with html.div(klass="w3-row-padding"):
                            for e_job in row:
                                if e_job:
                                    self._html_explainer_overview_tile(
                                        e_job=e_job,
                                        purpose_key="",
                                        ctx=ctx,
                                        html=html,
                                    )

        if self.i.result.problems:
            with html.div(
                klass="w3-container w3-padding-large", style="margin-bottom:32px"
            ):
                html.h2(id="problems", _t="Problems")
                self._html_problems_table(
                    problem_list=self.i.result.problems,
                    html=html,
                    root_problem_list=True,
                )

        if self.i.result.insights:
            with html.div(
                klass="w3-container w3-padding-large", style="margin-bottom:32px"
            ):
                html.h2(id="insights", _t="Insights")
                self._html_insights_table(
                    insight_list=self.i.result.insights,
                    html=html,
                    root_problem_list=True,
                )

        # section: interpretation
        with html.div(
            klass="w3-container w3-padding-large", style="margin-bottom:32px"
        ):
            html.h2(id="interpretation", _t=f"{self.brand_explainer}s")

            b_explainers = f"{self.brand_explainer}s".lower()
            if self.i.is_explainer_scheduled():
                all_cnt = len(self.i.get_all_explainer_ids())
                incompatible_cnt = len(self.i.get_incompatible_explainer_ids())
                scheduled_cnt = len(self.i.get_scheduled_explainer_ids())
                success_cnt = len(self.i.get_successful_explainer_ids())
                failed_cnt = len(self.i.get_failed_explainer_ids())
                successful_pct = (
                    int(float(success_cnt) / (float(scheduled_cnt) / 100.0))
                    if failed_cnt
                    else 100
                )

                with html.div(klass="w3-red"):
                    html.div(
                        klass="w3-container w3-green w3-padding w3-center",
                        style=f"width:{successful_pct}%",
                        _t=f"{successful_pct}%",
                    )

                with html.p():
                    if all_cnt:
                        html.b(_t="All")
                        html(f"{b_explainers} ({all_cnt}):")
                        self._html_explainer_ids_2_ul(
                            self.i.get_all_explainer_ids(), html
                        )
                    if incompatible_cnt:
                        html.b(_t="Incompatible")
                        html(f"{b_explainers} ({incompatible_cnt}):")
                        self._html_explainer_ids_2_ul(
                            self.i.get_incompatible_explainer_ids(), html
                        )
                    if self.i.is_explainer_scheduled():
                        html.b(_t="Scheduled")
                        html(f"{b_explainers} ({scheduled_cnt}):")
                        self._html_explainer_ids_2_ul(
                            self.i.get_scheduled_explainer_ids(), html
                        )
                    if self.i.is_explainer_finished():
                        html.b(_t="Finished")
                        html(
                            f"{b_explainers} "
                            f"({len(self.i.get_finished_explainer_ids())}):"
                        )
                        self._html_explainer_ids_2_ul(
                            self.i.get_finished_explainer_ids(), html
                        )
                    if self.i.is_explainer_successful():
                        html.b(klass="w3-green", _t="Successful")
                        html(f"{b_explainers} ({success_cnt}):")
                        self._html_explainer_ids_2_ul(
                            self.i.get_successful_explainer_ids(), html
                        )
                    if self.i.is_explainer_failed():
                        html.b(klass="w3-red", _t="Failed")
                        html(f"{b_explainers} ({failed_cnt}):")
                        self._html_explainer_ids_2_ul(
                            self.i.get_failed_explainer_ids(), html
                        )
            else:
                html(f"No {b_explainers}s were run.")

                incompatible_cnt = len(self.i.get_incompatible_explainer_ids())
                if incompatible_cnt:
                    with html.p():
                        html.b(_t="Incompatible")
                        html(f"{b_explainers} ({incompatible_cnt}):")
                        self._html_explainer_ids_2_ul(
                            self.i.get_incompatible_explainer_ids(), html
                        )

        # section: per-explainer section
        for explainer_id in self.i.get_finished_explainer_ids():
            r = self.i.get_explainer_result_metadata(explainer_id)
            r_descriptor = r.get(
                ExplainerJob.KEY_RESULT_DESCRIPTOR,
                r.get(ExplainerJob.KEY_EXPLAINER_DESCRIPTOR, {}),
            )

            if r:
                r_name = self._html_explainer_id_2_name(explainer_id) or explainer_id

                with html.div(
                    klass="w3-container w3-padding-large",
                    style="margin-bottom:32px",
                ):
                    explainer_problems = self.i.get_explainer_problems(explainer_id)
                    explainer_insights = self.i.get_explainer_insights(explainer_id)

                    with html.h2(id=f"{explainer_id}"):
                        html(f"{self.brand_explainer}: {r_name}")
                    with html.ul():
                        if explainer_problems:
                            with html.li():
                                html.a(
                                    href=f"#e-model-problems-{explainer_id}",
                                    _t="Problems",
                                )
                        if explainer_insights:
                            with html.li():
                                html.a(
                                    href=f"#e-model-insights-{explainer_id}",
                                    _t="Insights",
                                )

                        with html.li():
                            html.a(
                                href=f"#e-description-{explainer_id}",
                                _t=f"{self.brand_explainer} description",
                            )
                        m_ms = r_descriptor.get(
                            e8s.ExplainerDescriptor.KEY_METRICS_META, None
                        )
                        if m_ms:
                            with html.li():
                                html.a(
                                    href=f"#ms-description-{explainer_id}",
                                    _t="Metrics description",
                                )
                        with html.li():
                            html.a(
                                href=f"#e-explanations-details-{explainer_id}",
                                _t="Explanations",
                            )
                        with html.li():
                            html.a(
                                href=f"#e-explanations-data-{explainer_id}",
                                _t="Explanations data",
                            )
                        with html.li():
                            html.a(
                                href=f"#e-parameters-{explainer_id}",
                                _t="Parameters",
                            )
                        with html.li():
                            html.a(href=f"#e-metadata-{explainer_id}", _t="Metadata")
                        with html.li():
                            html.a(href=f"#e-run-{explainer_id}", _t="Run")
                        with html.li():
                            html.a(href=f"#e-log-{explainer_id}", _t="Log")

                    if explainer_problems:
                        with html.div(klass="w3-container w3-padding-small"):
                            with html.h5(id=f"e-model-problems-{explainer_id}"):
                                html.b(_t="Problems")

                            self._html_problems_table(
                                problem_list=self.i.get_explainer_problems(
                                    explainer_id
                                ),
                                html=html,
                                plural=False,
                                root_problem_list=False,
                            )

                    if explainer_insights:
                        with html.div(klass="w3-container w3-padding-small"):
                            with html.h5(id=f"e-model-problems-{explainer_id}"):
                                html.b(_t="Insights")

                            self._html_insights_table(
                                insight_list=self.i.get_explainer_insights(
                                    explainer_id
                                ),
                                html=html,
                                plural=False,
                                root_problem_list=False,
                            )

                    with html.div(klass="w3-container w3-padding-small"):
                        with html.h5(id=f"e-description-{explainer_id}"):
                            html.b(_t=f"{self.brand_explainer} description")
                        with html.p():
                            if r_descriptor:
                                str_descr = r_descriptor.get("description", "")
                            else:
                                str_descr = r.get("explainer", {}).get(
                                    "description", ""
                                )
                            # Markdown to HTML:
                            # - https://daringfireball.net/projects/markdown/syntax
                            # - https://pypi.org/project/Markdown/
                            # - https://python-markdown.github.io/extensions/
                            md_2_html_description = markdown.markdown(
                                text=str_descr,
                                extensions=[
                                    "markdown.extensions.tables",
                                    "markdown.extensions.fenced_code",
                                ],
                            )
                            # force style for tables
                            md_2_html_description = md_2_html_description.replace(
                                "<table>", '<table class="w3-table-all">'
                            )
                            html(f"{md_2_html_description}")

                    # metrics metadata are part of evaluator's description

                    # explainer persistence & files
                    type_e_persistence = persistences.ExplainerPersistence
                    e_persistence = persistences.ExplainerPersistence(
                        data_dir=self.i.persistence.store.base_path,
                        username=commons.DEFAULT_USER,
                        explainer_id=explainer_id,
                        explainer_job_key=r.get("key", ""),
                        mli_key=str(self.i.key),
                        store_persistence=self.i.persistence.store,
                    )
                    e_log_path = e_persistence.get_relative_path(
                        e_persistence.get_explainer_log_path()
                    )
                    t_html_fragment = e9s.GlobalHtmlFragmentExplanation
                    e_file = e_persistence.get_explanation_file_path(
                        explanation_type=(t_html_fragment.explanation_type()),
                        explanation_format=formats.HtmlFormat.mime,
                        explanation_file=f"{type_e_persistence.FILE_EXPLANATION}.html",
                    )

                    if r_descriptor.get("explanations"):
                        with html.div(klass="w3-container w3-padding-small"):
                            with html.h5(id=f"e-explanations-details-{explainer_id}"):
                                html.b(_t="Explanations")

                            try:
                                include_mark = str(uuid.uuid4())
                                ctx.explanations_html_fragments[include_mark] = (
                                    self.i.persistence.store.load(
                                        key=e_file,
                                        data_type=persistences.PersistenceDataType.text,
                                    )
                                )
                                with html.p():
                                    html(include_mark)
                            except Exception as ex:
                                msg = (
                                    f"Unable to include explanations HTML fragment for "
                                    f"explainer {explainer_id}: {ex}"
                                )
                                self.logger.warning(msg)

                    if r_descriptor.get("explanations"):

                        def _is_file_loc():
                            return (
                                False
                                if self.branding == commons.Branding.H2O_SONAR
                                else True
                            )

                        with html.div(klass="w3-container w3-padding-small"):
                            with html.h5(id=f"e-explanations-data-{explainer_id}"):
                                html.b(_t="Explanations")

                            html(
                                f"Model explanations created by the "
                                f"{self.brand_explainer.lower()} "
                                f"organized by explanation types with its formats "
                                f"(representations) identified by"
                            )
                            html.a(
                                href=(
                                    "https://www.iana.org/assignments/media-types/"
                                    "media-types.xhtml"
                                ),
                                _t="media types",
                            )
                            html(":")

                            with html.ul():
                                for t in r_descriptor.get("explanations", []):
                                    with html.li():
                                        with html.b():
                                            html(f"{t.get('name', '')}")
                                        html.br()
                                        with html.i():
                                            html(f"{t.get('explanation_type', '')}")
                                        with html.ul():
                                            for format_mime in t.get("formats", []):
                                                with html.li():
                                                    f_dir = self._html_format_dir(
                                                        explainer_job_id=r["key"],
                                                        explanation_type=t.get(
                                                            "explanation_type"
                                                        ),
                                                        format_mime=format_mime,
                                                        file=_is_file_loc(),
                                                    )
                                                    with html.a(href=f"{f_dir}"):
                                                        html(f"{format_mime}")

                    with html.div(klass="w3-container w3-padding-small"):
                        with html.h5(id=f"e-parameters-{explainer_id}"):
                            html.b(_t=f"{self.brand_explainer} parameters")

                        if self.i.result and self.i.result.explainers_params:
                            html(
                                f"{self.brand_explainer} was run with the "
                                f"following parameters: "
                            )
                            self._html_e_param(
                                explainer_id=explainer_id,
                                e_result_descriptor=r_descriptor,
                                html=html,
                                as_table=True,
                            )
                        else:
                            html("The explainer has no parameters.")

                    if r_descriptor:
                        with html.div(klass="w3-container w3-padding-small"):
                            with html.h5(id=f"e-metadata-{explainer_id}"):
                                html.b(_t=f"{self.brand_explainer} metadata")

                            can_explain_m = r_descriptor.get("can_explain", [])
                            if can_explain_m:
                                html(
                                    f"The {self.brand_explainer.lower()} can explain "
                                    f"model types:"
                                )
                                with html.ul():
                                    for t in can_explain_m:
                                        with html.li():
                                            with html.b():
                                                html(f"{t}")

                            e_keywords = r_descriptor.get("keywords", [])
                            if e_keywords:
                                html(f"{self.brand_explainer} keywords:")
                                with html.ul():
                                    for t in e_keywords:
                                        with html.li():
                                            with html.b():
                                                html(f"{t}")

                    with html.div(klass="w3-container w3-padding-small"):
                        with html.h5(id=f"e-run-{explainer_id}"):
                            html.b(_t=f"{self.brand_explainer} run")

                        html(f"{self.brand_explainer} run details:")
                        with html.ul():
                            with html.li():
                                r_status = self._html_status_2_str(
                                    r.get(ExplainerJob.KEY_STATUS, 0)
                                )
                                html("Status code:")
                                klass = (
                                    "w3-red" if r.get("status", 0) != 0 else "w3-green"
                                )
                                with html.b(klass=klass):
                                    html(f"&nbsp;{r_status}&nbsp;")
                            if r.get("status", 0) != 0:
                                with html.li():
                                    r_error = r.get(ExplainerJob.KEY_ERROR, "")
                                    html("Error:")
                                    if r_error:
                                        with html.pre():
                                            html(f"{r_error}")
                            with html.li():
                                r_progress = int(
                                    r.get(ExplainerJob.KEY_PROGRESS, 0) * 100.0
                                )
                                html("Progress:")
                                with html.b():
                                    html(f"{r_progress}%")
                            with html.li():
                                started_str = str(
                                    datetime.datetime.fromtimestamp(
                                        r.get(ExplainerJob.KEY_CREATED, 0)
                                    )
                                )
                                started_str = started_str[: started_str.index(".")]
                                html("Started:")
                                with html.b():
                                    html(f"{started_str} T{time.strftime('%z')}")
                            with html.li():
                                r_duration = round(
                                    r.get(ExplainerJob.KEY_DURATION, 0), 3
                                )
                                html("Duration:")
                                with html.b():
                                    html(f"{r_duration}s")

                    with html.div(klass="w3-container w3-padding-small"):
                        with html.h5(id=f"e-log-{explainer_id}"):
                            html.b(_t=f"{self.brand_explainer} log")
                        html(f"{self.brand_explainer} log file:")
                        with html.ul():
                            with html.li():
                                html.a(
                                    href=f"{e_log_path}",
                                    _t=f"{self.brand_explainer.lower()}.log",
                                )

        # section: dataset
        with html.div(
            klass="w3-container w3-padding-large", style="margin-bottom:32px"
        ):
            html.h2(id="dataset", _t="Dataset")

            self._html_right_main_dataset(html=html, dataset_type="Dataset")

        # section: testset
        if self.i.common_params.testset is not None:
            with html.div(
                klass="w3-container w3-padding-large", style="margin-bottom:32px"
            ):
                html.h2(id="testset", _t="Test dataset")

                self._html_right_main_dataset(html=html, dataset_type="Testset")

        # section: validset
        if self.i.common_params.validset is not None:
            with html.div(
                klass="w3-container w3-padding-large", style="margin-bottom:32px"
            ):
                html.h2(id="validset", _t="Validation dataset")

                self._html_right_main_dataset(html=html, dataset_type="Validset")

        # section: model/models
        model_dict = self.i.result.to_dict().get("model", {})
        model_dict_list = self.i.result.to_dict().get("models", [])
        if model_dict:
            with html.div(
                klass="w3-container w3-padding-large", style="margin-bottom:32px"
            ):
                html.h2(id="model", _t="Model")
                html("Model description:")
                with html.ul():
                    model_dict = self.i.result.to_dict().get("model", {})
                    if model_dict:
                        with html.li():
                            html.b(_t="Model type")
                            html(" = ")
                            with html.code():
                                html(
                                    HtmlInterpretationFormat.html_safe_str_field(
                                        str(model_dict.get("model_type", ""))
                                    )
                                )
                            with html.ul():
                                with html.li():
                                    with html.i():
                                        html("Model type.")

                        with html.li():
                            html.b(_t="Experiment type")
                            html(" = ")
                            with html.code():
                                html(
                                    HtmlInterpretationFormat.html_safe_str_field(
                                        str(model_dict.get("experiment_type", ""))
                                    )
                                )
                            with html.ul():
                                with html.li():
                                    with html.i():
                                        html("Experiment type.")

                        with html.li():
                            html.b(_t="Target column")
                            html(" = ")
                            with html.code():
                                html(
                                    HtmlInterpretationFormat.html_safe_str_field(
                                        str(
                                            model_dict.get("metadata", {}).get(
                                                "target_col", ""
                                            )
                                        )
                                    )
                                )
                            with html.ul():
                                with html.li():
                                    with html.i():
                                        html("Target column.")

                        with html.li():
                            html.b(_t="Labels count")
                            html(" = ")
                            with html.code():
                                html(
                                    HtmlInterpretationFormat.html_safe_str_field(
                                        str(
                                            model_dict.get("metadata", {}).get(
                                                "num_labels", ""
                                            )
                                        )
                                    )
                                )
                            with html.ul():
                                with html.li():
                                    with html.i():
                                        html("The number of interpreted model labels.")

                        with html.li():
                            html.b(_t="Labels")
                            html(" = ")
                            with html.code():
                                html(
                                    HtmlInterpretationFormat.html_safe_str_field(
                                        str(
                                            model_dict.get("metadata", {}).get(
                                                "labels", ""
                                            )
                                        )
                                    )
                                )
                            with html.ul():
                                with html.li():
                                    with html.i():
                                        html("Interpret model labels.")

                        with html.li():
                            html.b(_t="Used features")
                            html(" = ")
                            with html.code():
                                html(
                                    HtmlInterpretationFormat.html_safe_str_field(
                                        str(
                                            model_dict.get("metadata", {}).get(
                                                "used_features", ""
                                            )
                                        )
                                    )
                                )
                            with html.ul():
                                with html.li():
                                    with html.i():
                                        html(
                                            "Features (dataset columns) used by "
                                            "the interpreted model."
                                        )

                        with html.li():
                            html.b(_t="Transformed features")
                            html(" = ")
                            with html.code():
                                html(
                                    HtmlInterpretationFormat.html_safe_str_field(
                                        str(
                                            model_dict.get("metadata", {}).get(
                                                "transformed_features", ""
                                            )
                                        )
                                    )
                                )
                            with html.ul():
                                with html.li():
                                    with html.i():
                                        html(
                                            "Transformed features created by/used "
                                            "by the interpreted model."
                                        )

                    else:
                        with html.li():
                            html.code(_t=f"{self.i.common_params.model}")
        if model_dict_list:
            with html.div(
                klass="w3-container w3-padding-large", style="margin-bottom:32px"
            ):
                html.h2(id="model", _t=f"Models ({len(model_dict_list)})")

                s = (
                    "Interpre"
                    if self.branding == commons.Branding.H2O_SONAR
                    else "Evalua"
                )
                with html.p():
                    html(
                        f"{s}ted models - LLM and corpus (in case of RAG) -  overview:"
                    )

                # map: model name -> TOC ID
                toc_anchors = {}
                for m_dict in model_dict_list:
                    if not isinstance(m_dict, dict):
                        continue
                    llm_model_name = m_dict.get("llm_model_name", "")
                    model_name = m_dict.get("name", llm_model_name)

                    # TOC anchor
                    section_id = str(uuid.uuid4())
                    toc_anchors[model_name] = section_id

                # model request stats table
                if (
                    model_dict_list
                    and model_dict_list[0].get("llm_model_meta")
                    and model_dict_list[0].get("llm_model_meta").get("duration_stats")
                ):
                    HtmlInterpretationFormat._html_per_llm_request_table(
                        html, model_dict_list, toc_anchors
                    )
                else:
                    with html.ul():
                        for m_dict in model_dict_list:
                            with html.li():
                                llm_model_name = m_dict.get("llm_model_name", "")
                                model_name = m_dict.get("name", llm_model_name)

                                # TOC anchor
                                with html.a(href=f"#{toc_anchors[model_name]}"):
                                    html(model_name)

                for m_dict in model_dict_list:
                    if not isinstance(m_dict, dict):
                        continue
                    llm_model_name = m_dict.get("llm_model_name", "")
                    model_name = m_dict.get("name", llm_model_name)

                    html.br()
                    html.br()
                    with html.h3(id=toc_anchors[model_name]):
                        html(model_name)

                    html(f"{s}ted model details:")

                    with html.ul():
                        with html.li():
                            html("Evaluated model ID: ")
                            html.br()
                            with html.a(
                                title=(
                                    "Evaluation model ID which is used in "
                                    "the evaluation results, JSon evaluation"
                                    "report representation and in other "
                                    "evaluation artifacts."
                                ),
                                href="./interpretation.json",
                            ):
                                html(m_dict.get("key", ""))
                        with html.li():
                            html("LLM model name: ")
                            html.br()
                            with html.code():
                                html(m_dict.get("llm_model_name", ""))
                        with html.li():
                            html("Model host: ")
                            html.br()
                            with html.code():
                                html(m_dict.get("model_type", ""))
                        v = m_dict.get("collection_id", "")
                        if v:
                            with html.li():
                                html("Collection ID:")
                                html.br()
                                with html.code():
                                    html(v)
                        v = m_dict.get("collection_name", "")
                        if v:
                            with html.li():
                                html("Collection name:")
                                html.br()
                                with html.code():
                                    html(v)
                        v = m_dict.get("documents", [])
                        if v:
                            with html.li():
                                html("Documents (corpus):")
                                with html.ul():
                                    for d in v:
                                        with html.li():
                                            with html.a(href=d):
                                                html(d)
                        v = m_dict.get("model_cfg", "")
                        if v:
                            with html.li():
                                html("Model configuration:")
                                with html.pre(style="white-space:pre-wrap"):
                                    html(json.dumps(v, indent=2))

                    # model problems
                    model_problems = self.i.get_model_problems(llm_model_name)
                    if model_problems:
                        self._html_problems_table(
                            problem_list=model_problems,
                            html=html,
                            root_problem_list=True,
                        )

                    # model insights
                    model_insights = self.i.get_model_insights(llm_model_name)
                    if model_insights:
                        self._html_insights_table(
                            insight_list=model_insights,
                            html=html,
                            root_problem_list=True,
                        )

        # section: config & parameters
        with html.div(
            klass="w3-container w3-padding-large", style="margin-bottom:32px"
        ):
            html.h2(id="parameters", _t="Configuration and parameters")
            with html.p():
                if (
                    self.i.result
                    and self.i.common_params
                    and self.i.common_params.to_dict()
                ):
                    html(f"{self.brand_interpretation} parameters:")

                    params = self.i.common_params.to_dict()
                    self._html_cfg_items(
                        params=params,
                        description_src=self.i.common_params,
                        html=html,
                        as_table=False,
                    )

                html(f"{self.brand_h2o_sonar} library configuration:")
                params = h2o_sonar_config.config.to_dict(
                    encrypt=True,
                    # just make sensitive fields unreadable
                    encryption_key=f"{random.random()}",
                )
                self._html_cfg_items(
                    params=params,
                    description_src=h2o_sonar_config.config,
                    html=html,
                    as_table=True,
                )

        # section: dirs, files and logs
        with html.div(
            klass="w3-container w3-padding-large", style="margin-bottom:32px"
        ):
            html.h2(id="artifacts", _t="Directories, files and logs")
            with html.p():
                html("Directories and files:")
                with html.ul():
                    with html.li():
                        html.a(
                            href="./interpretation.html",
                            _t=f"{self.brand_interpretation} summary in HTML format",
                        )
                    with html.li():
                        html.a(
                            href="./interpretation.json",
                            _t=f"{self.brand_interpretation} summary in JSon format",
                        )
                    with html.li():
                        html.a(
                            href="./",
                            _t=f"This model {self.brand_interpretation.lower()} "
                            f"directory",
                        )
                    with html.li():
                        html.a(
                            href="../..",
                            _t=f"{self.brand_h2o_sonar} library results directory",
                        )
                    with html.li():
                        html.a(
                            href="../../h2o-sonar.log",
                            _t=f"{self.brand_h2o_sonar} library log",
                        )

    @staticmethod
    def _html_per_llm_request_table(html, model_dict_list, toc_anchors):
        with html.table(klass="w3-table-all"):
            with html.thead():
                with html.tr():
                    html.th(_t="Request Model", rowspan=2)
                    html.th(_t="Request Count", colspan=5)
                    html.th(_t="Request Time", colspan=4)
                with html.tr():
                    html.th(_t="Total")
                    html.th(_t="Successful")
                    html.th(_t="Failed")
                    html.th(_t="Retries")
                    html.th(_t="Timeouts")
                    html.th(_t="Fastest")
                    html.th(_t="Slowest")
                    html.th(_t="Average")
                    html.th(_t="Total")
            with html.tbody():
                # create table rows
                for model in model_dict_list:
                    llm_meta = model["llm_model_meta"]
                    duration_stats = llm_meta["duration_stats"]
                    # ensure statistics are present
                    duration_stats = duration_stats or {}
                    duration_stats[e9s.DurationStatsKey.N] = duration_stats.get(
                        e9s.DurationStatsKey.N, 1
                    )
                    duration_stats[e9s.DurationStatsKey.MIN] = duration_stats.get(
                        e9s.DurationStatsKey.MIN, 0.0
                    )
                    duration_stats[e9s.DurationStatsKey.MAX] = duration_stats.get(
                        e9s.DurationStatsKey.MAX, 0.0
                    )
                    duration_stats[e9s.DurationStatsKey.AVG] = duration_stats.get(
                        e9s.DurationStatsKey.AVG, 0.0
                    )
                    duration_stats[e9s.DurationStatsKey.SUM] = duration_stats.get(
                        e9s.DurationStatsKey.SUM, 0.0
                    )
                    with html.tr():
                        with html.td():
                            html.a(
                                _t=model["name"],
                                href=f"#{toc_anchors[model['name']]}",
                            )
                        html.td(_t=f"{duration_stats['n']}")
                        norm_count = (
                            llm_meta["success_count"] / duration_stats["n"]
                            if duration_stats["n"]
                            else 0.0
                        )
                        html.td(_t=f"{llm_meta['success_count']} ({norm_count:.0%})")
                        norm_count = (
                            llm_meta["failure_count"] / duration_stats["n"]
                            if duration_stats["n"]
                            else 0.0
                        )
                        html.td(_t=f"{llm_meta['failure_count']} ({norm_count:.0%})")
                        norm_count = (
                            llm_meta["retry_count"] / duration_stats["n"]
                            if duration_stats["n"]
                            else 0.0
                        )
                        html.td(_t=f"{llm_meta['retry_count']} ({norm_count:.0%})")
                        norm_count = (
                            llm_meta["timeout_count"] / duration_stats["n"]
                            if duration_stats["n"]
                            else 0.0
                        )
                        html.td(_t=f"{llm_meta['timeout_count']} ({norm_count:.0%})")
                        html.td(_t=f"{duration_stats['min']:.2f}s")
                        html.td(_t=f"{duration_stats['max']:.2f}s")
                        html.td(_t=f"{duration_stats['avg']:.2f}s")
                        html.td(_t=f"{duration_stats['sum']:.2f}s")

    @staticmethod
    def _html_semaphore(html, highest_severity: problems.ProblemSeverity | None):
        with html.div(
            title=(
                "Overall evaluation result represented as traffic "
                "light color - based on the highest severity of problem(s) "
                "which were identified."
            ),
            klass="semaphore-device",
        ):
            # LEFT
            light_color = (
                "red"
                if highest_severity == problems.ProblemSeverity.high
                else "dark-gray"
            )
            html.span(klass=f"semaphore-light w3-{light_color}")
            # MID
            light_color = (
                "orange"
                if highest_severity == problems.ProblemSeverity.medium
                else (
                    "yellow"
                    if highest_severity == problems.ProblemSeverity.low
                    else "dark-gray"
                )
            )
            html.span(klass=f"semaphore-light w3-{light_color}")
            # RIGHT
            light_color = "green" if highest_severity is None else "dark-gray"
            html.span(klass=f"semaphore-light w3-{light_color}")

    def _html_right_main_summary_table(self, html):
        with html.table(klass="w3-table-all"):
            with html.tr():
                # result ~ SEMAPHORE made based on the problems severity
                html.td(_t=f"{self.brand_interpretation} result:")
                with html.td():
                    overall_result = self.i.update_overall_result()
                    if overall_result == OverallResult.high_severity_problems:
                        HtmlInterpretationFormat._html_semaphore(
                            html, problems.ProblemSeverity.high
                        )
                    elif overall_result == OverallResult.medium_severity_problems:
                        HtmlInterpretationFormat._html_semaphore(
                            html, problems.ProblemSeverity.medium
                        )
                    elif overall_result == OverallResult.low_severity_problems:
                        HtmlInterpretationFormat._html_semaphore(
                            html, problems.ProblemSeverity.low
                        )
                    else:
                        HtmlInterpretationFormat._html_semaphore(html, None)

            with html.tr():
                html.td(_t="Problems:")
                with html.td():
                    problems_count = len(self.i.result.problems)
                    if problems_count:
                        p_severity_count = len(
                            self.i.get_problems_by_severity(
                                problems.ProblemSeverity.high
                            )
                        )
                        if p_severity_count > 0:
                            with html.b(klass="w3-red"):
                                html("&nbsp;")
                                html.a(href="#problems", _t=f"{p_severity_count}")
                                html("&nbsp;")

                            html("&nbsp;")

                        p_severity_count = len(
                            self.i.get_problems_by_severity(
                                problems.ProblemSeverity.medium
                            )
                        )
                        if p_severity_count > 0:
                            with html.b(klass="w3-orange"):
                                html("&nbsp;")
                                html.a(href="#problems", _t=f"{p_severity_count}")
                                html("&nbsp;")

                            html("&nbsp;")

                        p_severity_count = len(
                            self.i.get_problems_by_severity(
                                problems.ProblemSeverity.low
                            )
                        )
                        if p_severity_count > 0:
                            with html.b(klass="w3-yellow"):
                                html("&nbsp;")
                                html.a(href="#problems", _t=f"{p_severity_count}")
                                html("&nbsp;")

                    else:
                        with html.b(klass="w3-green"):
                            html("&nbsp;")
                            html.a(href="#problems", _t=f"{problems_count}")
                            html("&nbsp;")
            with html.tr():
                html.td(_t="Insights:")
                with html.td():
                    insights_count = len(self.i.result.insights)
                    if insights_count:
                        with html.b(klass="w3-green"):
                            html("&nbsp;")
                            html.a(href="#insights", _t=f"{insights_count}")
                            html("&nbsp;")
                    else:
                        with html.b(klass="w3-white"):
                            html("&nbsp;")
                            html.a(href="#insights", _t=f"{insights_count}")
                            html("&nbsp;")
            with html.tr():
                plural = self.i.common_params.models and len(
                    self.i.common_params.models
                )
                html.td(_t=f"Model{'s' if plural else ''}:")
                with html.td():
                    with html.a(href="#model"):
                        models_str = ""
                        if self.i.common_params.model is not None:
                            models_str = str(self.i.common_params.model)
                        elif self.i.common_params.models is not None:
                            models_count = len(self.i.common_params.models)
                            models_plural = "s" if models_count > 0 else ""
                            models_str = f"{models_count} LLM/RAG model{models_plural}"
                        model = self.html_safe_str_field(models_str)
                        html.code(_t=f"{model}")
            if self.i.common_params.target_col:
                with html.tr():
                    html.td(_t="Target column:")
                    with html.td():
                        t_c_str = HtmlInterpretationFormat.html_safe_str_field(
                            self.i.common_params.target_col
                        )
                        html.code(_t=f"{t_c_str}")
            if self.i.common_params.dataset is not None:
                with html.tr():
                    html.td(_t="Dataset:")
                    with html.td():
                        with html.a(href="#dataset"):
                            html.code(_t=f"{self.i.common_params.dataset}")
            if self.i.common_params.testset is not None:
                with html.tr():
                    html.td(_t="Test&nbsp;dataset:")
                    with html.td():
                        with html.a(href="#dataset-test"):
                            html.code(_t=f"{self.i.common_params.testset}")
            if self.i.common_params.validset is not None:
                with html.tr():
                    html.td(_t="Validation&nbsp;dataset:")
                    with html.td():
                        with html.a(href="#dataset-validation"):
                            html.code(_t=f"{self.i.common_params.validset}")
            with html.tr():
                html.td(_t=f"{self.brand_interpretation}&nbsp;status:")
                with html.td():
                    if self.i and self.i.status:
                        if self.i.status.value == 0:
                            html.b(
                                klass="w3-green", _t=f"&nbsp;{self.i.status.name}&nbsp;"
                            )
                        elif self.i.status.value < 0:
                            html.b(_t=f"&nbsp;{self.i.status.name}&nbsp;")
                        else:
                            html.b(
                                klass="w3-red", _t=f"&nbsp;{self.i.status.name}&nbsp;"
                            )
                    else:
                        html.code(_t="&nbsp;UNKNOWN&nbsp;")
            with html.tr():
                html.td(_t=f"{self.brand_interpretation}&nbsp;ID:")
                with html.td():
                    html.code(_t=f"{self.i.key}")
            with html.tr():
                html.td(_t="Created:")
                with html.td():
                    created_str = str(datetime.datetime.fromtimestamp(self.i.created))
                    created_str = created_str[: created_str.index(".")]
                    html.code(_t=f"{created_str}")
            if self.i and self.i.result and self.i.result.upload_url:
                with html.tr():
                    html.td(_t="Uploaded&nbsp;to:")
                    with html.td():
                        with html.a(href=f"{self.i.result.upload_url}"):
                            html.code(_t="H2O GPT Enterprise collection")

    def _html_right_main_summary(self, html):
        with html.div(
            klass="w3-container w3-padding-large", style="margin-bottom:32px"
        ):
            html.h2(id="report-summary", _t="Summary")
            self._html_right_main_summary_table(html)

    def _html_right_main(self, ctx: Context, html, report_style: str = "HTML"):
        """Inject right main panel to Airium HTML.

        Parameters
        ----------
        ctx : Context
          Interpretation context.
        html : airium.Airium
          Airium HTML instance.

        """
        # document BODY options:
        # 1) HTML-style - accounts for left navigation panel
        # 2) (PDF) Document-style - no left navigation panel & doc w/ intro, TOC, etc.
        if report_style == "DOCUMENT":
            with html.div(klass="w3-main", style="margin-left:50px;margin-right:50px"):
                with html.header(id="model-interpretation-report"):
                    with html.div(klass="w3-container"):
                        with html.h1():
                            with html.a(klass="w3-left w3-margin-right", href="#"):
                                htmls.html_svg_h2oai_logo(html)
                            html.b(_t=f"{self.brand_h2o_sonar} {self.brand_m_i_report}")
                        html(
                            "This report presents the results of the machine learning "
                            "model interpretation. "
                            "The report summarizes the results of the explainers "
                            "that were run to interpret the model in order to "
                            "describe various model properties such as "
                            "model problems, approximate model behavior, "
                            "the most important model features and their behavior, "
                            "fairness model metrics and issues, model insights, "
                            "and recommendations for model debugging."
                        )

                # report summary
                self._html_right_main_summary(html)
                # table of contents
                self._html_right_main_toc(ctx=ctx, html=html)
                # document main body
                self._html_right_main_content(ctx=ctx, html=html)

                # section: footer
                HtmlInterpretationFormat.html_footer(
                    html, brand_h2o_sonar=self.brand_h2o_sonar, branding=self.branding
                )

            html.div(klass="w3-black w3-center w3-padding-24")
        else:
            html.div(
                klass="w3-overlay w3-hide-large w3-animate-opacity",
                id="myOverlay",
                style="cursor:pointer",
                title="close side menu",
            )
            with html.div(klass="w3-main", style="margin-left:300px"):
                with html.header(id="model-interpretation-report"):
                    with html.div(klass="w3-container"):
                        with html.h1():
                            html.b(_t=self.brand_m_i_report)
                        self._html_right_main_summary_table(html=html)

                # document main body
                self._html_right_main_content(ctx=ctx, html=html)

                # section: footer
                HtmlInterpretationFormat.html_footer(
                    html, brand_h2o_sonar=self.brand_h2o_sonar, branding=self.branding
                )

            html.div(klass="w3-black w3-center w3-padding-24")

    @staticmethod
    def _html_txt_ellipsis(text: str, lng: int = 100, strategy: str = "end") -> str:
        if text and len(text) > lng:
            if strategy == "end":
                return f"{text[:lng]}..."
            else:
                half = int((lng - 3) / 2)
                return f"{text[:half]}...{text[-half:]}"

        return text

    def _html_explainer_overview_tile(
        self, e_job: ExplainerJob, purpose_key: str, ctx: Context, html
    ):
        """Inject explainer overview tile into right panel of Airium HTML.

        Parameters
        ----------
        e_job : ExplainerJob
          Explainer job.
        purpose_key : str
          Explainer function (if available).
        ctx : Context
          Interpretation context.
        html : airium.Airium
          Airium HTML instance.

        """
        with html.div(klass="w3-third w3-container w3-margin-bottom"):
            with html.a(href=f"#{e_job.explainer_descriptor.id}"):
                img_path = ctx.purpose_image.get(e_job.key, "")
                if img_path:
                    html.img(
                        alt="explanation chart",
                        klass="w3-hover-opacity",
                        src=img_path,
                        style="width:100%",
                    )
                else:
                    html.div(style="height: 200px", klass="w3-white")
            with html.div(klass="w3-container w3-white"):
                if purpose_key:
                    purpose_display = self._purpose_id_to_name(purpose_key)
                    with html.p():
                        html.b(_t=f"{purpose_display}")
                    html.p(_t=f"{e_job.explainer_descriptor.display_name}")
                else:
                    with html.p():
                        html.b(_t=f"{e_job.explainer_descriptor.display_name}")

                # TODO print explainer_purpose, NOT description
                purpose_txt = HtmlInterpretationFormat._html_txt_ellipsis(
                    e_job.explainer_descriptor.description, 107
                )
                html.p(
                    klass="w3-small",
                    title=e_job.explainer_descriptor.description,
                    _t=f"{purpose_txt}",
                )

    @staticmethod
    def html_h2o_sonar_pitch(brand_h2o_sonar: str) -> str:
        return (
            f"{brand_h2o_sonar} is Python package that enables a holistic, "
            f"low-risk, human-interpretable, fair, and trustable "
            f"approach to machine learning by implementing various "
            f"facets of Responsible AI. "
        )

    @staticmethod
    def html_footer(html, brand_h2o_sonar: str, branding: commons.Branding):
        """Inject footer into Airium HTML.

        Parameters
        ----------
        html : airium.Airium
          Airium HTML instance.
        brand_h2o_sonar : str
            H2O Sonar branding.
        branding : commons.Branding
            Branding.

        """
        with html.footer(klass="w3-container w3-padding-32 w3-dark-grey"):
            with html.div(klass="w3-row-padding"):
                with html.div(klass="w3-third"):
                    html.h3(_t=brand_h2o_sonar)
                    if branding == commons.Branding.H2O_SONAR:
                        with html.p():
                            html(
                                HtmlInterpretationFormat.html_h2o_sonar_pitch(
                                    brand_h2o_sonar
                                )
                            )
                with html.div(klass="w3-third"):
                    if branding == commons.Branding.H2O_SONAR:
                        html.h3(_t="Resources")
                    with html.ul(klass="w3-ul w3-hoverable"):
                        if branding == commons.Branding.H2O_SONAR:
                            with html.li(klass="w3-padding-16"):
                                with html.div(
                                    klass="w3-left w3-margin-right", style="width:50px"
                                ):
                                    htmls.html_svg_h2oai_logo(html)
                                html.span(klass="w3-large", _t=brand_h2o_sonar)
                                html.br()
                                with html.span():
                                    html.a(
                                        href="https://github.com/h2oai/h2o-sonar",
                                        _t="GitHub&nbsp;repository",
                                    )
                        with html.li(klass="w3-padding-16"):
                            with html.div(
                                klass="w3-left w3-margin-right", style="width:50px"
                            ):
                                htmls.html_svg_h2oai_logo(html)
                            html.span(klass="w3-large", _t="H2O.ai")
                            html.br()
                            with html.span():
                                html("Democratize AI with")
                                html.a(href="https://h2o.ai/", _t="H2O.ai")

    def _html_explainer_id_2_name(self, explainer_id) -> str:
        fallback_name = f"UNKNOWN ({explainer_id})"

        if explainer_id:
            explainer_meta = self.i.get_explainer_result_metadata(explainer_id)
            if explainer_meta:
                explainer_descr = explainer_meta.get(
                    ExplainerJob.KEY_EXPLAINER_DESCRIPTOR, {}
                )
                if explainer_descr:
                    return explainer_descr.get(
                        e8s.ExplainerDescriptor.KEY_DISPLAY_NAME, explainer_id
                    )
            descr = self.i.result.incompatible_explainers.get(explainer_id)
            if descr:
                return descr.get(
                    e8s.ExplainerDescriptor.KEY_DISPLAY_NAME, fallback_name
                )

        self.logger.warning(
            f"Unable to get explainer name for ID '{explainer_id}' from the result "
            f"metadata - explainer ID missing or invalid - using explainer ID as "
            f"fallback name."
        )
        # return fallback display name
        return fallback_name

    def _html_explainer_ids_2_ul(self, explainer_ids: list, html):
        """Convert IDs to unordered HTML list.

        Parameters
        ----------
        explainer_ids : list
          Explainers.
        html : airium.Airium
          Airium HTML instance.

        """
        with html.ul():
            for explainer_id in explainer_ids:
                with html.li():
                    with html.a(href=f"#{explainer_id}"):
                        html(f"{self._html_explainer_id_2_name(explainer_id)}")

    @staticmethod
    def _html_status_2_str(status: int) -> str:
        return str(commons.ExplainerJobStatus(status).name)

    def _html_format_dir(
        self,
        explainer_job_id: str,
        explanation_type: str,
        format_mime: str,
        file: bool = False,
    ) -> str:
        dir_path = ""
        if self.i.result:
            job = self.i.result.get_explainer_job(explainer_job_id)
            if job.explainer_persistence:
                if file:
                    return job.explainer_persistence.get_relative_path(
                        job.explainer_persistence.get_explanation_file_path(
                            explanation_type=explanation_type,
                            explanation_format=format_mime,
                        )
                    )
                else:
                    return job.explainer_persistence.get_relative_path(
                        job.explainer_persistence.get_explanation_dir_path(
                            explanation_type=explanation_type,
                            explanation_format=format_mime,
                        )
                    )
        return dir_path


class Interpretations:
    """Interpretations created by H2O Sonar in ``results`` location."""

    def __init__(
        self,
        interpretations_paths: list[str],
        persistence,
        branding: commons.Branding = commons.Branding.H2O_SONAR,
        logger=None,
    ):
        self.interpretations_paths = interpretations_paths
        self.persistence = persistence
        self.h2o_sonar_path = persistence.base_path
        self.interpretations_keys = []
        self._init_keys_for_paths()
        self.branding = branding
        self.logger = logger or loggers.SonarPrintLogger()

    def _init_keys_for_paths(self):
        self.interpretations_keys = [
            path[path.rfind("_") + 1 :] for path in self.interpretations_paths
        ]

    def load_interpretation_meta(self, i_path: str, digest: bool = True) -> dict:
        if i_path:
            i_json_path = os.path.join(
                self.h2o_sonar_path,
                i_path,
                persistences.InterpretationPersistence.FILE_INTERPRETATION_JSON,
            )
            if os.path.isfile(i_json_path):
                return (
                    Interpretation.dict_to_digest(
                        self.persistence.load_json(i_json_path) or {}
                    )
                    if digest
                    else self.persistence.load_json(i_json_path) or {}
                )

        return {}

    def count(self) -> int:
        return len(self.interpretations_paths) if self.interpretations_paths else 0

    def to_html(self, branding: commons.Branding = commons.Branding.EVAL_STUDIO) -> str:
        return HtmlInterpretationsFormat(
            self, branding=branding, logger=self.logger
        ).to_html()


class PdfInterpretationFormat(HtmlInterpretationFormat):
    """PDF (via HTML) representation of the interpretation."""

    def __init__(
        self,
        interpretation: Interpretation,
        logger: loggers.SonarLogger,
        branding: commons.Branding = commons.Branding.H2O_SONAR,
    ):
        HtmlInterpretationFormat.__init__(
            self, interpretation=interpretation, branding=branding, logger=logger
        )

    def to_html_4_pdf(self) -> str:
        """To HTML which can be used to generate PDF."""
        return HtmlInterpretationFormat.to_html(
            self,
            include_left_navigation=False,
            report_style="DOCUMENT",
        )

    @staticmethod
    def to_pdf(input_path: str, output_path: str):
        # to make this method work the following packages need to be installed:
        #
        # 1) pandoc
        #    sudo apt install pandoc
        # 2) LaTeX (for PDF generation)
        #    sudo apt-get install texlive-latex-base texlive-fonts-recommended \
        #    texlive-fonts-extra texlive-latex-extra
        #
        # Pandoc can use various engines to generate PDF
        # (like latex|beamer|context|ms|html5), latex looks good:
        #
        #  pandoc -t latex -o interpretation-detailed.pdf interpretation-detailed.html
        #
        if not os.path.isfile(input_path):
            raise RuntimeError(
                f"Unable to generate PDF from HTML: input HTML '{input_path}' "
                f"does NOT exist"
            )

        cmd = [
            "pandoc",
            "-t",
            "latex",
            "-o",
            f"{output_path}",
            f"{input_path}",
        ]
        p = subprocess.Popen(cmd, env=os.environ.copy())
        p.wait()

        if not os.path.isfile(output_path):
            raise RuntimeError(
                f"WARNING: Unable to generate PDF from HTML: input HTML '{input_path}',"
                f" output PDF '{output_path}' and command '{cmd}' - please verify "
                f"`pandoc` installation"
            )


class HtmlInterpretationsFormat:
    """HTML representation of an interpretations list."""

    def __init__(
        self,
        interpretations: Interpretations,
        branding: commons.Branding = commons.Branding.H2O_SONAR,
        logger: loggers.SonarLogger | None = None,
    ):
        self.intepretations = interpretations
        self.logger = logger or loggers.SonarPrintLogger()

        self.branding = branding
        if self.branding == commons.Branding.H2O_SONAR:
            self.brand_h2o_sonar = "H2O Sonar"
            self.brand_report = "Interpretation Report"
            self.brand_m_i_report = "Model Interpretation Report"
            self.brand_interpretation = "Interpretation"
            self.brand_explainer = "Explainer"
            self.brand_ms_is = "Models Interpretations"
        else:
            self.brand_h2o_sonar = "Eval Studio"
            self.brand_report = "Evaluation Report"
            self.brand_m_i_report = "LLM Evaluation Report"
            self.brand_interpretation = "Evaluation"
            self.brand_explainer = "Evaluator"
            self.brand_ms_is = "LLMs Evaluations"

    def _brand_product_name(self) -> str:
        return (
            "H2O Sonar"
            if self.branding == commons.Branding.H2O_SONAR
            else "Eval Studio"
        )

    def _html_head(self, a):
        """Inject head to Airium HTML.

        Parameters
        ----------
        a : airium.Airium
          Airium HTML instance.

        """
        htmls.evaluation_report_html_head(
            a,
            title=(
                f"{self.brand_h2o_sonar} {self.brand_interpretation}s "
                f"({self.intepretations.count()})"
            ),
        )

    def _html_left_navigation(self, html):
        """Inject left navigation panel to Airium HTML.

        Parameters
        ----------
        html : airium.Airium
          Airium HTML instance.

        """
        with html.nav(
            klass="w3-sidebar w3-collapse w3-black w3-animate-left",
            id="mySidebar",
            style="z-index:3;width:300px;",
        ):
            html.br()
            with html.div(klass="w3-container"):
                with html.a(klass="w3-left w3-margin-right", href="#"):
                    htmls.html_svg_h2oai_logo(html)
                with html.h3():
                    html.b(
                        _t=self.brand_h2o_sonar,
                        title=HtmlInterpretationFormat.html_h2o_sonar_pitch(
                            self.brand_h2o_sonar
                        ),
                    )
                if commons.Branding.H2O_SONAR == self.branding:
                    html.h4(_t="Responsible AI library")
                else:
                    html.h4(_t="LLM Evaluation library")

                html.p(klass="w3-text-yellow", _t=f"{self.brand_interpretation}s")
                with html.div(klass="w3-bar-block"):
                    with html.a(
                        klass="w3-bar-item w3-button w3-padding w3-margin-left",
                        href="#models-interpretations",
                    ):
                        html(f"{self.brand_ms_is} ({self.intepretations.count()})")

                html.p(klass="w3-text-yellow", _t="Diagnostics")
                with html.div(klass="w3-bar-block"):
                    with html.a(
                        klass="w3-bar-item w3-button w3-padding w3-margin-left",
                        href="#h2o-sonar-log",
                    ):
                        html(f"{self.brand_h2o_sonar} log")

                with html.div(klass="w3-panel w3-large"):
                    html.i(klass="fa fa-facebook-official w3-hover-opacity")
                    html.i(klass="fa fa-instagram w3-hover-opacity")
                    html.i(klass="fa fa-snapchat w3-hover-opacity")
                    html.i(klass="fa fa-pinterest-p w3-hover-opacity")
                    html.i(klass="fa fa-twitter w3-hover-opacity")
                    html.i(klass="fa fa-linkedin w3-hover-opacity")

    def _html_right_main(self, html):
        """Inject the right main panel to Airium HTML.

        Parameters
        ----------
        html : airium.Airium
          Airium HTML instance.

        """
        # aliases
        i_p = persistences.InterpretationPersistence
        safe_str = HtmlInterpretationFormat.html_safe_str_field

        # data class CANNOT be used to get automatically sorted list of interpretations
        # (cython compiler does NOT support data classes)
        class HtmlIndexEntry:
            def __init__(
                self,
                ts: int,
                ts_str: str,
                key: str,
                path: str,
                model: str,
                target_col: str,
                dataset: str,
                testset: str,
                validset: str,
                num_scheduled_es: int,
                failed_es: list,
            ):
                self.ts = ts
                self.ts_str = ts_str
                self.key = key
                self.path = path
                self.model = model
                self.target_col = target_col
                self.dataset = dataset
                self.testset = testset
                self.validset = validset
                self.num_scheduled_es = num_scheduled_es
                self.failed_es = failed_es

        index_entries: list[HtmlIndexEntry] = []

        for i, i_path in enumerate(self.intepretations.interpretations_paths):
            i_meta_digest = self.intepretations.load_interpretation_meta(i_path)
            i_meta = self.intepretations.load_interpretation_meta(i_path, digest=False)
            es = i_meta.get(Interpretation.KEY_RESULT, {}).get(
                Interpretation.KEY_EXECUTED_EXPLAINERS, []
            )
            i_created = i_meta.get(Interpretation.KEY_CREATED, 0.0)

            entry = HtmlIndexEntry(
                ts=i_created,
                ts_str=f"{i_meta_digest.get(Interpretation.KEY_CREATED, '')}",
                key=f"{self.intepretations.interpretations_keys[i]}",
                path=f"{i_path}/{i_p.FILE_INTERPRETATION_HTML}",
                model=safe_str(i_meta_digest.get(Interpretation.KEY_MODEL, "")),
                target_col=safe_str(
                    i_meta_digest.get(Interpretation.KEY_TARGET_COL, "")
                ),
                dataset=safe_str(i_meta_digest.get(Interpretation.KEY_DATASET, "")),
                testset=safe_str(i_meta_digest.get(Interpretation.KEY_TESTSET, "")),
                validset=safe_str(i_meta_digest.get(Interpretation.KEY_VALIDSET, "")),
                num_scheduled_es=safe_str(
                    i_meta_digest.get(Interpretation.KEY_SCHEDULED_EXPLAINERS, [])
                ),
                failed_es=[
                    e for e in es if e.get(Interpretation.KEY_STATUS, None) != 0
                ],
            )

            index_entries.append(entry)
        index_entries.sort(key=lambda a: a.ts, reverse=True)

        # HTML
        html.div(
            klass="w3-overlay w3-hide-large w3-animate-opacity",
            id="myOverlay",
            style="cursor:pointer",
            title="close side menu",
        )
        with html.div(klass="w3-main", style="margin-left:300px"):
            with html.header(id="model-interpretation-report"):
                with html.div(klass="w3-container"):
                    with html.h1():
                        html.b(_t=self.brand_h2o_sonar)

                    with html.p():
                        html(
                            f"{self.brand_h2o_sonar} is a Python package for the "
                            f"introspection of machine learning models by enabling "
                            f"various facets of Responsible AI."
                        )

            with html.div(
                klass="w3-container w3-padding-large", style="margin-bottom:16px"
            ):
                html.h2(id="models-interpretations", _t=self.brand_ms_is)
                html(
                    f"{self.brand_interpretation}s with model explanations "
                    f"created by {self.brand_explainer.lower()}s:"
                )
                with html.ul():
                    for entry in index_entries:
                        with html.li():
                            html(f"{self.brand_interpretation} ")
                            with html.b():
                                with html.a(href=f"{entry.path}"):
                                    html(f"{entry.key}")
                            html(f"  {entry.ts_str}")
                            with html.ul():
                                safe_str = HtmlInterpretationFormat.html_safe_str_field
                                if entry.model:
                                    with html.li():
                                        html.b(_t="Model")
                                        html(f": {entry.model}")
                                if entry.target_col:
                                    with html.li():
                                        html.b(_t="Target column")
                                        html(f": '{entry.target_col}'")
                                with html.li():
                                    html.b(_t="Dataset")
                                    html(f": {entry.dataset}")
                                with html.li():
                                    i_f = safe_str(entry.num_scheduled_es)
                                    html.b(_t="Explainer(s)")
                                    if not i_f:
                                        with html.b(klass="w3-red"):
                                            html("&nbsp;")
                                            html(f"{i_f}")
                                            html("&nbsp;")
                                    else:
                                        html(f": {i_f}")

                                    if entry.failed_es and len(entry.failed_es) > 0:
                                        failed_es_ids = [
                                            e.get(
                                                ExplainerJob.KEY_EXPLAINER_DESCRIPTOR,
                                                {},
                                            ).get("name", "")
                                            for e in entry.failed_es
                                        ]
                                        failed_es_str = ", ".join(failed_es_ids)
                                        html("&nbsp;")
                                        with html.b(
                                            klass="w3-red",
                                            title=failed_es_str,
                                        ):
                                            html("&nbsp;")
                                            html(f"{len(entry.failed_es)}&nbsp;failed")
                                            html("&nbsp;")

            with html.div(
                klass="w3-container w3-padding-large", style="margin-bottom:32px"
            ):
                html.h2(id="h2o-sonar-log", _t=f"{self.brand_h2o_sonar} log")
                html(f"{self.brand_h2o_sonar} log with library runtime messages:")

                with html.ul():
                    with html.li():
                        log_file_name = loggers.SonarLogger.FILE_NAME_H2O_SONAR_LOG
                        with html.a(href=f"./{log_file_name}"):
                            html(f"{log_file_name}")

                html(
                    f"Please not that every {self.brand_explainer.lower()} has its own "
                    f"log and {self.brand_explainer.lower()} messages are not logged "
                    f"to the library log by default (check configuration documentation "
                    f"for more details)."
                )

            HtmlInterpretationFormat.html_footer(
                html, brand_h2o_sonar=self.brand_h2o_sonar, branding=self.branding
            )

    def to_html(self, branding: commons.Branding = commons.Branding.EVAL_STUDIO) -> str:
        """Get HTML for the interpretations list."""
        html = airium.Airium()
        html("<!DOCTYPE html>")
        with html.html(lang="en"):
            self._html_head(html)

            with html.body(klass="w3-light-grey w3-content", style="max-width:1600px"):
                self._html_left_navigation(html)
                self._html_right_main(html)

        return str(html)
