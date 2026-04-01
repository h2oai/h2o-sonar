# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import importlib
import json
import logging
import os.path
import pathlib
import time
import traceback
from abc import ABC
from abc import abstractmethod
from concurrent import futures
from typing import Any

import datatable
import pandas
from matplotlib import pyplot

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar import errors
from h2o_sonar import loggers
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import datasets
from h2o_sonar.lib.api import explainers as e8s
from h2o_sonar.lib.api import interpretations
from h2o_sonar.lib.api import models as m4s
from h2o_sonar.lib.api import persistences as p10s
from h2o_sonar.lib.api import problems
from h2o_sonar.methods.utils import h2o_utils
from h2o_sonar.utils import _profiling
from h2o_sonar.utils import progress as progress_utils
from h2o_sonar.utils import sampling


def _resolve_branding(models: list | None = None) -> commons.Branding:
    """Resolve branding based on config and model types.

    Parameters
    ----------
    models : list | None
        List of models to be evaluated. If provided and not empty,
        defaults to EVAL_STUDIO branding.

    Returns
    -------
    commons.Branding :
        Resolved branding enum value.

    """
    # check config first
    if h2o_sonar_config.config.branding:
        # use configured branding if it's set (convert string to enum)
        if h2o_sonar_config.config.branding == commons.Branding.H2O_SONAR.name:
            return commons.Branding.H2O_SONAR
        elif h2o_sonar_config.config.branding == commons.Branding.EVAL_STUDIO.name:
            return commons.Branding.EVAL_STUDIO
        # fallback to default logic if invalid value
    # default logic when config branding is not set (empty string) or invalid
    return commons.Branding.EVAL_STUDIO if models else commons.Branding.H2O_SONAR


class ExplainerContainer(ABC):
    """Explainer container interface provides execution runtime independent APIs
    and functions which enable explainers to run "anywhere" via  explainer container
    implementations for various runtimes.

    Explainer gets reference to explainer container on its instantiation. Explainer
    container in turn contains references to public dataset, model and persistence APIs.

    Explainer container can be instantiated with various combinations
    of mock/real/DB-based/... model/dataset/persistence APIs.

    """

    TYPE_ID = ""

    @property
    def explainers_registry(self):
        """Registry of explainers registered within the container."""
        return self._explainers_registry

    def __init__(self):
        # Model API can be used to load any model, determine model metadata
        # (target, features used by the model, model type (regression, binomial,
        # multinomial)) and get predict function (scorer).
        self.dataset_api = datasets.DatasetApi()
        self.model_api = m4s.ModelApi()
        self.persistence_api = p10s.PersistenceApi()

        # default persistence: filesystem @ current directory
        self.persistence = self.persistence_api.create_persistence(
            persistence_type=p10s.PersistenceType.file_system,
        )
        self.persistence_type: p10s.PersistenceType = self.persistence.type

        self.log_level = None
        self.logger: loggers.SonarLogger = loggers.SonarPrintLogger()
        # registry of OOTB explainers
        self._explainers_registry = e8s.ExplainerRegistry.registry()
        self._executor = None

    def hot_deploy_explainer(self, explainer_descriptor: str):
        """Import custom BYOE recipe explainer **class** from given Python module.
        It is expected that the module is installed and/or on ``PYTHONPATH``.

        Parameters
        ----------
        explainer_descriptor : str
          Explainer descriptor with format ``...`` for example ``...``.

        Returns
        -------
        Type[Explainer] :
          New explainer type.

        """
        if not isinstance(explainer_descriptor, str):
            raise ValueError(
                f"Configured custom explainer descriptor '{explainer_descriptor}' is "
                f"invalid - it must be string, but it is: {type(explainer_descriptor)}"
            )
        if not explainer_descriptor:
            raise ValueError(
                f"Configured custom explainer descriptor '{explainer_descriptor}' "
                f"cannot be empty."
            )
        if "::" not in explainer_descriptor:
            raise ValueError(
                f"Configured custom explainer descriptor '{explainer_descriptor}' "
                f"must have format <module>::<explainer class name> for example "
                f"pkg1.pkg2.module1::MyExplainer"
            )

        (e_module_name, e_class_name) = explainer_descriptor.split("::")

        self.logger.debug(
            f"Hot deploying BYOE recipe explainer: {e_module_name} > {e_class_name}"
        )

        # dynamic explainer class import
        # e_module = __import__(e_module_name, fromlist=e_class_name)
        # e_module = __import__(e_module_name, globals(), locals(), [e_class_name], 0)
        e_module = importlib.__import__(e_module_name, fromlist=e_class_name)

        e_type = getattr(e_module, e_class_name)

        self.logger.debug(f"BYOE recipe explainer successfully hot deployed: {e_type}")

        return e_type

    def register_configured_explainers(self):
        """Register custom BYOE recipe explainers which are configured in H2O Sonar
        configuration.

        """
        if h2o_sonar_config.config.custom_explainers:
            for e_descriptor in h2o_sonar_config.config.custom_explainers:
                e_type = self.hot_deploy_explainer(e_descriptor)
                e_id = self._explainers_registry.register(e_type)
                self.logger.info(
                    f"Successfully registered BYOE recipe explainer '{e_type}' "
                    f"under ID '{e_id}'"
                )

    def _create_logger(self, results_location):
        log_path = os.path.join(
            results_location,
            loggers.SonarLogger.FILE_NAME_H2O_SONAR_LOG,
        )

        if h2o_sonar_config.config.enable_profiler:
            return _profiling.SonarProfilingLogger(
                logger_name="H2oSonarProfilingLogger",
                log_file=str(log_path),
                log_level=self.log_level,
            )

        return loggers.SonarFileLogger(
            logger_name="H2oSonarLogger",
            log_file=str(log_path),
            log_level=self.log_level,
        )

    @abstractmethod
    def setup(
        self,
        results_location: str | Any = "",
        persistence_api: p10s.PersistenceApi | None = None,
        persistence_type: (
            p10s.PersistenceType | None
        ) = p10s.PersistenceType.file_system,
        logger: loggers.SonarLogger | None = None,
        log_level: int | None = None,
    ):
        """Setup explainer container.

        Parameters
        ----------
        results_location : str | Any
          Optional directory where can container (interpretation, explainers, ...)
          store the data. If not specified, then in-memory store is used and logging to
          stdout/stderr is used.
        persistence_api : p10s.PersistenceApi | None
          Instance of the persistence API allowing to create various persistence types
          (like file-system or DB)
        persistence_type : p10s.Persistence | None
          Persistence type to be used by this container - default is the file-system
          persistence with the root in the current directory.
        logger : loggers.SonarLogger | None
          Optional custom logger.
        log_level : int | None
          Interpretation log level.

        """
        self.logger = logger or self.logger

        self.persistence_api = (
            persistence_api
            or self.persistence_api
            or p10s.PersistenceApi(logger=self.logger)
        )

        results_location = (
            str(results_location) if results_location is not None else None
        )
        if not results_location:
            results_location = self.persistence_api.get_cwl(persistence_type)
        if not os.path.isdir(results_location):
            try:
                p10s.ExplainerPersistence.makedirs(results_location)
            except Exception as e:
                # IMPROVE might be DB / memory path in the future
                raise ValueError(
                    f"Interpretation result directory does not exist: "
                    f"'{results_location}' and attempt to create the directory "
                    f"failed: {e}"
                )

        # logger to be overwritten by child class if filesystem/DB is available
        self.log_level = log_level or logging.WARNING
        if not logger and persistence_type == p10s.PersistenceType.file_system:
            self.logger = self._create_logger(results_location)

        self._executor = SequentialExplainerExecutor(
            container=self,
            explainers_registry=self._explainers_registry,
            log_level=self.log_level,
            container_logger=self.logger,
        )

        # initialize persistence
        self.persistence = self.persistence_api.create_persistence(
            persistence_type=persistence_type,
            base_path=results_location,
        )
        # preferred persistence type
        self.persistence_type = self.persistence.type

    #
    # API
    #

    def get_explainer(self, explainer_id) -> e8s.ExplainerDescriptor:
        return self.get_explainer_class(explainer_id)().as_descriptor()

    def get_explainer_class(self, explainer_id) -> type[e8s.Explainer]:
        clazz = self.explainers_registry.get_class(explainer_id)
        if not clazz:
            raise ValueError(
                f"Unable to return explainer class - unknown explainer ID "
                f"'{explainer_id}'"
            )

        return clazz

    def list_explainers(
        self,
        experiment_types: list[str] | None = None,
        explanation_scopes: list[str] | None = None,
        model_meta: m4s.ExplainableModelMeta | None = None,
        keywords: list[str] | None = None,
        explainer_filter: list[commons.FilterEntry] | None = None,
        portable: bool = False,
    ) -> list[e8s.ExplainerDescriptor]:
        """List and filter explainers.

        Returns
        -------
        list[ExplainerDescriptor] :
          List of explainer descriptors which match the filter.

        """
        experiment_types = experiment_types or []
        explanation_scopes = explanation_scopes or []
        keywords = keywords or []
        explainer_filter = explainer_filter or []

        # TODO to be implemented by REUSE
        result = []
        explainers_dict: dict[e8s.Explainer] = (
            self._explainers_registry.list_explainers()
        )
        # match explainers w/ DEFAULT keyword only by default, if ALL/*
        #   > match all explainers
        all_by_keyword: bool = (
            True
            if not keywords
            or (
                keywords
                and (
                    commons.KEYWORD_FILTER_ALL in keywords
                    or commons.KEYWORD_FILTER_ALL_ASTERISK in keywords
                )
            )
            else False
        )
        for explainer_id in explainers_dict:
            explainer = explainers_dict[explainer_id]()
            descriptor = explainer.as_descriptor(portable=portable)

            if not all_by_keyword and keywords:
                if all(k in descriptor.keywords for k in keywords):
                    result.append(explainer.as_descriptor(portable=portable))
            elif all_by_keyword:
                result.append(explainer.as_descriptor(portable=portable))

        return result

    def register_explainer(
        self,
        explainer_class: type[e8s.Explainer],
        explainer_id: str = "",
        extra_params: dict | None = None,
    ):
        del explainer_id
        del extra_params

        return self.explainers_registry.register(explainer_class=explainer_class)

    def unregister_explainer(
        self,
        explainer_id: str,
        extra_params: dict | None = None,
    ) -> str:
        del extra_params

        return self.explainers_registry.unregister(explainer_id=explainer_id)

    def run_interpretation(
        self,
        dataset: (
            str
            | pathlib.Path
            | datasets.ExplainableDataset
            | datatable.Frame
            | pandas.DataFrame
        ),
        model: str | pathlib.Path | m4s.ExplainableModel | Any,
        models: list[str | pathlib.Path | m4s.ExplainableModel | Any],
        target_col: str,
        explainers: list[str | commons.ExplainerToRun] | None = None,
        explainer_keywords: list[str] | None = None,
        validset: (
            str
            | pathlib.Path
            | datasets.ExplainableDataset
            | commons.ResourceHandle
            | datatable.Frame
            | pandas.DataFrame
            | datasets.ExplainableDatasetHandle
            | None
        ) = None,
        testset: (
            str
            | pathlib.Path
            | datasets.ExplainableDataset
            | commons.ResourceHandle
            | datatable.Frame
            | pandas.DataFrame
            | datasets.ExplainableDatasetHandle
            | None
        ) = None,
        use_raw_features: bool = True,
        used_features: list | None = None,
        weight_col: str = "",
        prediction_col: str = "",
        drop_cols: list | None = None,
        sample_num_rows: int | None = 0,
        sampler: sampling.DatasetSampler | None = None,
        results_location: str | pathlib.Path | dict | None = None,
        results_formats: list[str] | None = None,
        progress_callback: progress_utils.AbstractProgressCallbackContext | None = None,
        run_asynchronously: bool = False,
        run_explainers_in_parallel: bool = False,
        key: str = "",
        logger: loggers.SonarLogger | None = None,
        extra_params: list | None = None,
    ) -> interpretations.Interpretation:
        if run_explainers_in_parallel:
            raise NotImplementedError("Parallel run of explainers is not supported yet")

        # determine initial branding - check config first, then use default logic
        initial_branding = _resolve_branding(models)

        interpretation = interpretations.Interpretation(
            common_params=commons.CommonInterpretationParams(
                model=model,
                models=models,
                dataset=dataset,
                target_col=target_col,
                validset=validset,
                testset=testset,
                use_raw_features=use_raw_features,
                weight_col=weight_col,
                prediction_col=prediction_col,
                drop_cols=drop_cols,
                sample_num_rows=sample_num_rows,
                results_location=results_location,
                extra_params=extra_params,
                used_features=used_features,
            ),
            explainers=explainers,
            explainer_keywords=explainer_keywords,
            created=time.time(),
            key=key,
            sampler=sampler,
            branding=initial_branding,
            results_formats=results_formats,
            progress_callback=progress_callback,
            logger=logger or self.logger,
            extra_params=extra_params,
        )

        self._executor.run(interpretation=interpretation)

        return interpretation

    def list_interpretations(
        self,
        username: str = commons.DEFAULT_USER,
    ) -> list[str]:
        return p10s.InterpretationPersistence.list_interpretations(
            data_dir=self.persistence.base_path,
            username=username,
            store_persistence=self.persistence,
            paths=True,
        )

    def load_interpretations(
        self,
        interpretation_key: str,
        username: str = commons.DEFAULT_USER,
    ) -> interpretations.Interpretation:
        i_persistence = p10s.InterpretationPersistence(
            data_dir=self.persistence.base_path,
            username=username,
            mli_key=interpretation_key,
            store_persistence=self.persistence,
        )
        return interpretations.Interpretation.load(
            persistence=i_persistence,
            logger=self.logger,
        )

    def delete_interpretation(self, mli_key: str):
        raise NotImplementedError

    def get_interpretation_params(self, mli_key: str):
        raise NotImplementedError

    def get_interpretation_status(self, mli_key: str):
        """Did interpretation succeeded and failed?"""
        raise NotImplementedError

    def get_explainer_job_statuses(self, mli_key: str):
        """Get explainer job status for every explainer which was run
        within the interpretation.

        """
        raise NotImplementedError

    def get_explainer_job_keys_by_id(self):
        raise NotImplementedError

    def get_explainer_log_path(self):
        raise NotImplementedError

    def get_explainer_result_path(self):
        raise NotImplementedError

    def get_explainer_snapshot_path(self):
        raise NotImplementedError

    def get_explainer_metadata(self):
        raise NotImplementedError

    def update_explainer_result(self):
        raise NotImplementedError

    def get_explainer_local_result(self):
        raise NotImplementedError

    def gc(self):
        """Free system resources:

        - shutdowns process pool(s)
        - runs garbage collector
        - clears temporary files

        """
        pass


#
# Interpretation executor(s)
#


class DependencyTreeNode:
    def __init__(self, explainer_class, *children):
        self.eclass = explainer_class
        self.eid = explainer_class.explainer_id()
        self.children = children

    def __str__(self):
        return "\n".join(self.str_lines())

    def str_lines(self):
        yield self.eid
        last = self.children[-1] if self.children else None
        for child in self.children:
            prefix = "`-" if child is last else "+-"
            for line in child.str_lines():
                yield prefix + line
                prefix = "  " if child is last else "| "

    def to_list(
        self, result: list, queue_result: dict, parent_queue: int = -1
    ) -> tuple[list, dict]:
        result.append(self.eid)
        keywords = self.eclass().keywords or []

        if parent_queue <= 0:
            queue_result[self.eid] = ParallelExplainerExecutor.get_queue_for_explainer(
                keywords
            )

            parent_queue = 0 if parent_queue == -1 else queue_result[self.eid]
        else:
            queue_result[self.eid] = parent_queue

        for child in self.children:
            child.to_list(
                result=result,
                queue_result=queue_result,
                parent_queue=parent_queue,
            )
        return result, queue_result

    def get_node_by_id(self, eid) -> "DependencyTreeNode | None":
        if eid == self.eid:
            return self
        for child in self.children:
            found = child.get_node_by_id(eid)
            if found:
                return found
        return None

    def add_child(self, node):
        self.children = self.children + (node,)


class ExplainerExecutor(ABC):
    """Executor of the interpretation and its explainers."""

    pass


class ParallelExplainerExecutor(ExplainerExecutor):
    """Run interpretation explainers in parallel."""

    class PriorityQueues:
        """Explainer executor priority queues."""

        HIGH_PRIORITY = 1
        MEDIUM_PRIORITY = 2
        LOW_PRIORITY = 3

    @classmethod
    def get_queue_for_explainer(cls, explainer_keywords: list) -> int:
        """Get execution queue (high priority, medium priority, low priority)
        for explainer.

        Parameters
        ----------
        explainer_keywords : list
          Explainer keywords where it requests the queue

        Returns
        -------
          Priority queue ID.

        """
        explainer_keywords = explainer_keywords or []
        if e8s.Explainer.KEYWORD_IS_FAST in explainer_keywords:
            return cls.PriorityQueues.HIGH_PRIORITY
        elif e8s.Explainer.KEYWORD_IS_SLOW in explainer_keywords:
            return cls.PriorityQueues.LOW_PRIORITY
        else:
            return cls.PriorityQueues.MEDIUM_PRIORITY

    @classmethod
    def ensure_valid_queue_plan(
        cls, execution_plan: list, queue_plan: dict, container_logger
    ):
        """In 3 priority queues design, task from a higher priority queue
        cannot depend on task from a lower priority queue.

        Parameters
        ----------
        execution_plan : list
          Execution plan which gives sequence of explainers to be executed.
        queue_plan : dict
          Queue plan to be verified (and eventually fixed).
        container_logger
          Container logger.

        """
        # invariant:
        # - assigned queue numbers must for non-growing sequence
        if execution_plan:
            queue_level = queue_plan[execution_plan[0]]
            for e in execution_plan:
                e_level = queue_plan[e]
                if e_level < queue_level:
                    container_logger.debug(
                        f"Explainer explainers executor: {queue_plan[e]} explainers "
                        f"demoted from priority queue {queue_plan[e]} to "
                        f"{queue_level} to avoid deadlock",
                    )
                    queue_plan[e] = queue_level


def _print_progress_callback(progress: float, message: str = ""):
    """Explainer progress callback which prints progress to standard output."""
    message = message or f"Explainer progress: {progress * 100}%"
    print(f"{message}")


class SequentialExplainerExecutor(ExplainerExecutor):
    """Run interpretation explainers sequentially.

    This is "for loop style" explainers execution executor - no multithreading,
    no multiprocessing, no async - explainers are executed sequentially within
    the same process. This is straightforward, deadlock/race conditions free and
    system resources considerate approach.

    """

    def __init__(
        self,
        container: ExplainerContainer,
        explainers_registry: e8s.ExplainerRegistry,
        log_level,
        container_logger,
    ):
        self.log_level = log_level
        self.container_logger = container_logger
        self.explainers_registry = explainers_registry
        self.container = container
        self.username: str = commons.DEFAULT_USER

    def _build_execution_plan(
        self,
        explainers_to_run: dict[str, commons.ExplainerToRun],
        mli_key: str,
    ) -> tuple[list[str], dict[str, int]]:
        """Create execution plan which ensures valid order of explainers in case
        of (sequential) execution and ensures dependencies availability on the explainer
        run.

        Parameters
        ----------
        explainers_to_run : dict[str, commons.ExplainerToRun]
          Explainers to be run.
        mli_key : str
          Interpretation key.

        Returns
        -------
        Tuple[list[str], dict[str, int]] :
          List of explainer IDs prescribing the order in which explainers must be
          executed to ensure that the dependencies of every explainer will be available
          at the right time.
          Priority queue plan (in case of parallel execution).

        """
        execution_plan: list[str] = []
        queue_plan: dict[str, int] = {}
        if explainers_to_run:
            tree: DependencyTreeNode | None = None

            # sort explainers by priority (higher runs earlier)
            blueprint_class_dict: dict = dict()
            for explainer_id in explainers_to_run:
                blueprint_class_dict[explainer_id] = self.explainers_registry.get_class(
                    explainer_id
                )

            blueprints_by_priority: list = list(blueprint_class_dict.values())
            blueprints_by_priority.sort(key=lambda x: x.priority(), reverse=True)
            explainers_by_priority: list = [
                blueprint.explainer_id() for blueprint in blueprints_by_priority
            ]

            for explainer_id in explainers_by_priority:
                if tree and tree.get_node_by_id(explainer_id):
                    raise errors.MliError(
                        f"Unable to build execution plan for "
                        f"interpretation {mli_key} and explainers "
                        f"{explainers_by_priority} - detected circular "
                        f"dependency in explainers dependencies ({explainer_id})",
                    )

                explainer_cls = self.explainers_registry.get_class(explainer_id)
                if explainer_cls.depends_on():
                    for parent_cls in explainer_cls.depends_on():
                        parent_id: str = parent_cls.explainer_id()
                        p_node = tree.get_node_by_id(parent_id) if tree else None
                        if not p_node:
                            p_node = DependencyTreeNode(parent_cls)
                            if tree:
                                tree.add_child(p_node)
                            else:
                                tree = p_node
                        p_node.add_child(DependencyTreeNode(explainer_cls))
                else:
                    node = DependencyTreeNode(explainer_cls)
                    if tree:
                        tree.add_child(node)
                    else:
                        tree = node
            self.container_logger.info(f"Execution plan tree:\n{tree}")
            tree.to_list(execution_plan, queue_plan)

            ParallelExplainerExecutor.ensure_valid_queue_plan(
                execution_plan=execution_plan,
                queue_plan=queue_plan,
                container_logger=self.container_logger,
            )

            # update indices on explainer dependencies injection
            if len(execution_plan) > len(explainers_to_run):
                for explainer_id in execution_plan:
                    if explainer_id not in explainers_to_run:
                        # injected explainers are run w/ default params - it's up to
                        # user to explicitly invoke them to specify parameters
                        explainers_to_run[explainer_id] = commons.ExplainerToRun(
                            explainer_id=explainer_id, params=""
                        )

        return execution_plan, queue_plan

    def _check_explainer_compatibility(
        self,
        explainer_id: str,
        interpretation: interpretations.Interpretation,
        model: m4s.ExplainableModel,
        models,
        dataset: datasets.ExplainableDataset | datasets.ExplainableDatasetHandle | None,
        testset: datasets.ExplainableDataset | datasets.ExplainableDatasetHandle | None,
        validset: (
            datasets.ExplainableDataset | datasets.ExplainableDatasetHandle | None
        ),
    ) -> bool:
        """Check whether the explainer is compatible with the model and dataset."""
        self.container_logger.info(
            f"Checking compatibility of explainer: '{explainer_id}' ..."
        )
        try:
            # instantiate explainer
            explainer_class = self.explainers_registry.get_class(explainer_id)
            if not explainer_class:
                raise errors.UnknownExplainerError(explainer_id)
            explainer: e8s.Explainer = explainer_class()
            # clone of the global H2O Sonar configuration
            explainer.config = h2o_sonar_config.config
            explainer.logger = self.container_logger

            # check evaluator LLM/RAG compatibility
            if explainer.is_rag() or explainer.is_llm():
                if not models:
                    return False
                if explainer.is_rag() and not explainer.is_llm():
                    for m in models:
                        if not isinstance(m, m4s.ExplainableRagModel):
                            self.container_logger.info(
                                f"Evaluator '{explainer_id}' evaluates RAG models "
                                f"only, however, model '{m}' is not a RAG model (has "
                                f"no corpus)"
                            )
                            return False
                # if explainer.is_llm():
                #   LLM evaluator can test RAG models - they have all required fields
                #   -> continue to evaluator's compatibility check

            # check dataset locator compatibility
            if dataset:
                if commons.ResourceHandle.is_handle(dataset):
                    if not explainer.supports_dataset_locator(
                        commons.ResourceLocatorType.handle
                    ):
                        self.container_logger.info(
                            "  'NOT COMPATIBLE' (models which use datasets "
                            "represented as a remote handles are not supported by "
                            "the explainer)"
                        )
                        return False

            # check model locator compatibility
            if model:
                if commons.ResourceHandle.is_handle(model):
                    if not explainer.supports_model_locator(
                        commons.ResourceLocatorType.handle
                    ):
                        self.container_logger.info(
                            "  'NOT COMPATIBLE' (models represented as a remote "
                            "handles are not supported by the explainer)"
                        )
                        return False
                else:
                    # check R/B/M compatibility:
                    if not explainer.can_explain(model_meta=model.meta):
                        self.container_logger.info(
                            f"  'NOT COMPATIBLE' "
                            f"({model.meta.get_model_type().name.upper()} models are "
                            f"not supported by the explainer)"
                        )
                        return False

            explainer_to_run = interpretation.explainer_id_to_e2run.get(explainer_id)
            explainer_params_as_str = (
                json.dumps(explainer_to_run.params)
                if explainer_to_run and isinstance(explainer_to_run.params, dict)
                else ""
            )
            compatibility: bool = explainer.check_compatibility(
                params=interpretation.common_params,
                explainer_params_as_str=explainer_params_as_str,
                model=model,
                models=models,
                mli_key=interpretation.key,
                X=(
                    dataset
                    if commons.ResourceHandle.is_handle(dataset)
                    else dataset.data
                ),
                dataset_meta=(
                    dataset.meta
                    if dataset and not commons.ResourceHandle.is_handle(dataset)
                    else None
                ),
                validset_meta=(
                    validset.meta
                    if validset and not commons.ResourceHandle.is_handle(validset)
                    else None
                ),
                testset_meta=(
                    testset.meta
                    if testset and not commons.ResourceHandle.is_handle(testset)
                    else None
                ),
                dataset_api=self.container.dataset_api,
                model_meta=(
                    model.meta
                    if model and not isinstance(model, commons.ResourceHandle)
                    else None
                ),
                model_api=self.container.model_api,
                logger=self.container_logger,
                sanitization_map={},  # TODO
                used_features=(
                    model.meta.used_features
                    if model
                    and not isinstance(model, commons.ResourceHandle)
                    and model.meta
                    else None
                ),
                on_demand_params={},  # TODO
            )

            # gather problems from the compatibility check to the interpretation
            for p in explainer.explain_problems():
                interpretation.result.problems.append(p)

            self.container_logger.info(
                f"  {'COMPATIBLE' if compatibility else 'NOT COMPATIBLE'}"
            )
            return compatibility
        except errors.UnknownExplainerError as ex:
            raise ex
        except Exception as ex:
            self.container_logger.error(
                f"Compatibility check of explainer '{explainer_id}' failed: {ex}\n"
                f"{traceback.format_exc()}"
            )

        return False

    def _save_interpretation_json(self, interpretation: interpretations.Interpretation):
        try:
            interpretation.persistence.make_interpretation_sandbox()
            interpretation.persistence.store.save_json(
                key=interpretation.result.json_location,
                data=interpretation.to_dict(),
            )
        except Exception as ex:
            self.container_logger.error(
                f"Unable to save interpretation result interpretation.json: {ex}\n"
                f"{traceback.format_exc()}"
            )

    def _resolve_dataset(
        self,
        dataset_param,
        target_col_param: str,
        sampled_dataset_path: str = "",
        sample_num_rows: int | None = None,
        sampler: sampling.DatasetSampler | None = None,
    ) -> tuple[
        tuple[datasets.ExplainableDataset, pandas.DataFrame] | None,
        str | None,
    ]:
        if not isinstance(dataset_param, pandas.DataFrame) and not dataset_param:
            return None, None

        if commons.ResourceHandle.is_handle(dataset_param):
            dataset_handle = datasets.ExplainableDatasetHandle.from_string(
                str(dataset_param)
            )
            dataset: datasets.ExplainableDataset | None = None
        elif isinstance(dataset_param, datasets.LlmDataset):
            dataset = self.container.dataset_api.create_dataset(
                dataset_src=dataset_param.to_datatable(),
                target_col=target_col_param,
                sampled_dataset_path=sampled_dataset_path,
                sample_num_rows=sample_num_rows,
                sampler=sampler,
            )
            dataset_handle = None
        else:
            dataset = self.container.dataset_api.create_dataset(
                dataset_src=dataset_param,
                target_col=target_col_param,
                sampled_dataset_path=sampled_dataset_path,
                sample_num_rows=sample_num_rows,
                sampler=sampler,
            )
            dataset_handle = None

        return dataset, dataset_handle

    def _resolve_model(
        self, model_param, target_col_param: str, dataset, used_features_param
    ) -> tuple[m4s.ExplainableModel | None, m4s.ExplainableModelHandle | None]:
        if not model_param:
            return None, None

        if commons.ResourceHandle.is_handle(model_param):
            model_handle = m4s.ExplainableModelHandle.from_string(str(model_param))
            model: m4s.ExplainableModel | None = None
        else:
            model: m4s.ExplainableModel = self.container.model_api.create_model(
                model_src=model_param,
                target_col=target_col_param,
                dataset=dataset,
                used_features=used_features_param,
            )
            model_handle = None

        return model, model_handle

    def _get_interpretation_error_msg(
        self, interpretation: interpretations.Interpretation
    ) -> str:
        err_msg = "1 or more explainers/evaluators failed"
        if interpretation.result.problems:
            # try to use message from runtime problems
            for p in interpretation.result.problems:
                if p.problem_type == "runtime":
                    err_msg = p.description
                    break

        self.container_logger.error(
            f"Setting failed interpretation/evaluation error message to: {err_msg}"
        )

        return err_msg

    def _set_interpretation_status(
        self,
        interpretation: interpretations.Interpretation,
        success_if_any_e_succeeded: bool = True,
    ):
        """Set the interpretation status:

        - SUCCESS
          - if at least one explainer succeeded
          - can be controlled by the parameter of this method
        - FAILED
          - if all explainers failed
          - if no explainer was run
          - if exception occurred in the container / container crashed

        """
        interpretation.set_progress(1.0, "DONE")
        interpretation.progress = 1.0  # just for sure

        try:
            if success_if_any_e_succeeded:
                # SUCCESS if at least one explainer succeeded
                interpretation.status = (
                    commons.ExplainerJobStatus.SUCCESS
                    if interpretation.get_successful_explainer_ids()
                    else commons.ExplainerJobStatus.FAILED
                )
            else:
                # FAIL if one explainer failed
                interpretation.status = (
                    commons.ExplainerJobStatus.FAILED
                    if not interpretation.get_successful_explainer_ids()
                    or not interpretation.get_finished_explainer_ids()
                    else commons.ExplainerJobStatus.SUCCESS
                )

            if interpretation.status == commons.ExplainerJobStatus.FAILED:
                interpretation.error = self._get_interpretation_error_msg(
                    interpretation
                )
        except Exception as ex:
            self.container_logger.error(
                f"Unable to set interpretation status: {ex}\n{traceback.format_exc()}",
            )
            # fallback to the default status
            interpretation.status = (
                commons.ExplainerJobStatus.FINISHED
                if int(interpretation.status.value) <= 0
                else interpretation.status
            )

    @staticmethod
    def __log_evalxplainer(explainer_id: str):
        if explainer_id:
            if ".evaluators." in explainer_id:
                return f"evaluator {explainer_id}"
            elif ".explainers." in explainer_id:
                return f"explainer {explainer_id}"
        return f"evaluator/explainer {explainer_id}"

    def _run(
        self,
        interpretation: interpretations.Interpretation,
    ) -> interpretations.Interpretation:
        this = SequentialExplainerExecutor
        interpretation.status = commons.ExplainerJobStatus.RUNNING

        interpretation.set_progress(0.0 + progress_utils.EPSILON, "Started")
        # validate interpretation parameters
        interpretation.validate_and_normalize_params()
        # persistence & paths
        persistence_api = self.container.persistence_api
        interpretation.persistence = persistence_api.create_interpretation_persistence(
            store_persistence=self.container.persistence,
            base_path=interpretation.result.results_location,
            interpretation_key=interpretation.key,
            username=self.username,
        )
        interpretation.result.interpretation_location = (
            interpretation.persistence.base_dir
        )
        # file system sandbox - required by async progress reporting
        try:
            interpretation.persistence.make_interpretation_sandbox()
            # report progress only if the sandbox exists
            parent_callback = interpretation.progress_callback.parent_callback
            if isinstance(parent_callback, progress_utils.CallbackToFileBridge):
                parent_callback.logger = interpretation.logger
                parent_callback.progress_file_path = (
                    interpretation.result.get_progress_location()
                )
        except Exception as ex:
            self.container_logger.error(
                f"Unable to create interpretation sandbox (will retry later) "
                f"and initialize file progress bridge: {ex}\n"
                f"{traceback.format_exc()}"
            )

        interpretation.set_progress(0.01, "Preparation...")
        interpretation.result.html_location = interpretation.persistence.get_html_path()
        interpretation.result.json_location = interpretation.persistence.get_json_path()
        self._save_interpretation_json(interpretation)

        # resolve dataset
        (dataset, dataset_handle) = self._resolve_dataset(
            dataset_param=interpretation.common_params.dataset,
            target_col_param=interpretation.common_params.target_col,
            sampled_dataset_path=interpretation.persistence.create_dataset_path(),
            sample_num_rows=interpretation.common_params.sample_num_rows,
            sampler=interpretation.sampler,
        )

        # resolve testset
        (testset, testset_handle) = self._resolve_dataset(
            dataset_param=interpretation.common_params.testset,
            target_col_param=interpretation.common_params.target_col,
            sampled_dataset_path=interpretation.persistence.create_dataset_path(),
            sample_num_rows=interpretation.common_params.sample_num_rows,
            sampler=interpretation.sampler,
        )

        # resolve validset
        (validset, validset_handle) = self._resolve_dataset(
            dataset_param=interpretation.common_params.validset,
            target_col_param=interpretation.common_params.target_col,
            sampled_dataset_path=interpretation.persistence.create_dataset_path(),
            sample_num_rows=interpretation.common_params.sample_num_rows,
            sampler=interpretation.sampler,
        )

        # ensure valid target column <=> model is specified & target col is non-empty
        if interpretation.common_params.target_col and dataset:
            if interpretation.common_params.target_col not in dataset.data.names:
                raise errors.MliError(
                    f"Interpretation cannot be run - target column "
                    f"'{interpretation.common_params.target_col}' doesn't present "
                    f"among dataset columns: {dataset.data.names}"
                )

        # resolve model
        if isinstance(
            interpretation.common_params.model,
            (m4s.ExplainableLlmModel, m4s.ExplainableRagModel),
        ):
            model = None
            model_handle = None
            if interpretation.common_params.models:
                interpretation.common_params.models.append(
                    interpretation.common_params.model
                )
                interpretation.common_params.model = None
            else:
                interpretation.common_params.models = [
                    interpretation.common_params.model
                ]
                interpretation.common_params.model = None
        else:
            (model, model_handle) = self._resolve_model(
                model_param=interpretation.common_params.model,
                target_col_param=interpretation.common_params.target_col,
                dataset=dataset,
                used_features_param=interpretation.common_params.used_features,
            )

        self.container_logger.info(
            f"Running interpretation of explainable model:"
            f"\nModel: {model} / Model handle: {model_handle}"
            f"\nModels: \n  {interpretation.common_params.models}"
            f"\nDataset:\n  {dataset}"
            f"\nDataset handle:\n  {dataset_handle}"
        )

        # branding: H2O Sonar vs. H2O Eval Studio
        interpretation.branding = _resolve_branding(interpretation.common_params.models)

        # chosen OR all available explainers
        explainers_to_run: dict[str, commons.ExplainerToRun] = {}
        if interpretation.explainers:
            explainers_to_run = {r.id: r for r in interpretation.explainers}
        else:
            default_explainers_descriptors = self.container.list_explainers(
                model_meta=model.meta if model else None,
                keywords=interpretation.explainer_keywords,
            )
            if not default_explainers_descriptors:
                raise errors.MliError(
                    f"No compatible explainers for interpretation: {interpretation}"
                )
            for descriptor in default_explainers_descriptors:
                explainers_to_run[descriptor.id] = commons.ExplainerToRun(
                    explainer_id=descriptor.id, params=""
                )

        # compatibility check
        # - progress reported in range [0.01, 0.09]
        progress_slot_min = interpretation.progress
        progress_slot_items = len(explainers_to_run.keys())
        progress_slot_size = (0.09 - progress_slot_min) / progress_slot_items
        incompatible_explainers = {}
        compatible_explainers_to_run: dict[str, commons.ExplainerToRun] = {}
        for e, explainer_id in enumerate(explainers_to_run.keys()):
            # progress
            e_step_min = progress_slot_min + progress_slot_size * e
            e_step_max = e_step_min + progress_slot_size
            interpretation.set_progress(
                e_step_min + progress_utils.EPSILON,
                message=(
                    f"#{e + 1}/{progress_slot_items} checking compatibility of "
                    f"the  explainer/evaluator {explainer_id} ..."
                ),
            )

            try:
                if self._check_explainer_compatibility(
                    explainer_id=explainer_id,
                    interpretation=interpretation,
                    model=model or model_handle,
                    models=interpretation.common_params.models,
                    dataset=dataset or dataset_handle,
                    testset=testset or testset_handle,
                    validset=validset or validset_handle,
                ):
                    compatible_explainers_to_run[explainer_id] = explainers_to_run[
                        explainer_id
                    ]
                else:
                    incompatible_explainers[explainer_id] = (
                        self.explainers_registry.get_class(explainer_id)()
                        .as_descriptor(runtime_view=True)
                        .dump()
                    )
            except errors.UnknownExplainerError as ex:
                self.container_logger.error(
                    f"Compatibility check of explainer '{explainer_id}' failed: {ex}\n"
                    f"{traceback.format_exc()}"
                )
                interpretation.result.problems.append(
                    problems.ProblemAndAction(
                        description=(
                            f"Unknown explainer/evaluator - cannot run "
                            f"'{explainer_id}' as it is not known to the runtime."
                        ),
                        severity=problems.ProblemSeverity.high,
                        problem_type="runtime",
                        problem_attrs={
                            "explainer_id": explainer_id,
                        },
                        actions_description=(
                            "Please register the explainer/evaluator in the container "
                            "to be able to run it."
                        ),
                        explainer_id=explainer_id,
                    )
                )
                try:
                    incompatible_explainers[explainer_id] = (
                        self.explainers_registry.get_class(explainer_id)()
                        .as_descriptor(runtime_view=True)
                        .dump()
                    )
                except Exception as ex:
                    self.container_logger.error(
                        f"Unable to dump incompatible explainer {explainer_id} "
                        f"descriptor : {ex}\n{traceback.format_exc()}"
                    )
                interpretation.status = commons.ExplainerJobStatus.FAILED

            interpretation.set_progress(
                e_step_max,
                message=(
                    f"#{e + 1}/{progress_slot_items} compatibility check of "
                    f"the {this.__log_evalxplainer(explainer_id)} finished"
                ),
            )
        # progress: 10%
        interpretation.set_progress(0.1, "Prepared")

        # execution plan: sequence of explainers which considers dependencies
        (execution_plan, _) = self._build_execution_plan(
            explainers_to_run=compatible_explainers_to_run,
            mli_key=interpretation.key,
        )
        interpretation.result.explainer_ids = execution_plan
        self._save_interpretation_json(interpretation)

        # execute explainers SEQUENTIALLY
        # - progress will be reported in range [0.1, 0.9]
        (
            progress_slot_min,
            progress_slot_size,
            steps,
        ) = progress_utils.ProgressCallbackContext.step_loop_prepare(
            progress_min=0.1,
            progress_max=0.9,
            steps=len(execution_plan),
        )
        for step, explainer_id in enumerate(execution_plan):
            self.container_logger.debug(f"Running explainer: {explainer_id}")

            # explainer job
            job = interpretations.ExplainerJob(
                key=commons.generate_key(),
                created=time.time(),
                duration=0.0,
                progress=0.0,
                status=commons.ExplainerJobStatus.RUNNING,
                message="Starting explainer...",
                error="",
                child_explainer_job_keys=None,
            )
            interpretation.result.explainers[job.key] = job

            e_persistence = None
            try:
                # instantiate explainer
                explainer: e8s.Explainer = self.explainers_registry.get_class(
                    explainer_id
                )()
                job.explainer_descriptor = explainer.as_descriptor(False)

                # progress: evaluator setup
                (
                    step_slot_min,
                    step_slot_max,
                ) = progress_utils.ProgressCallbackContext.step_loop_get_min_and_max(
                    step=step,
                    progress_slot_min=progress_slot_min,
                    progress_slot_size=progress_slot_size,
                )
                interpretation.set_progress(
                    step_slot_min,
                    (
                        f"{step + 1}/{progress_slot_items} setting up "
                        f"{this.__log_evalxplainer(explainer_id)}"
                    ),
                )
                e_progress_callback = (
                    interpretation.progress_callback.get_sub_callback_for_progress(
                        min_progress=step_slot_min,
                        max_progress=step_slot_max,
                        do_update=[job],
                        name=f"Explainer {explainer_id} progress callback",
                    )
                )
                explainer.progress_callback = e_progress_callback

                # running H2O-3 requirement
                if e8s.Explainer.KEYWORD_REQUIRES_H2O3 in explainer.keywords:
                    h2o_utils.ensure_h2o3_running(logger=self.container_logger)

                # explainer parameters & sandbox preparation
                e_persistence = persistence_api.create_explainer_persistence(
                    store_persistence=self.container.persistence,
                    base_path=interpretation.result.results_location,
                    explainer_id=explainer_id,
                    explainer_job_key=job.key,
                    interpretation_key=interpretation.key,
                    username=self.username,
                )
                e_persistence.make_explainer_sandbox()
                if h2o_sonar_config.config.per_explainer_logger:
                    explainer_logger = loggers.SonarFileLogger(
                        logger_name=f"{explainer_id}Logger",
                        log_level=self.log_level,
                        log_file=e_persistence.get_explainer_log_path(),
                    )
                else:
                    explainer_logger = self.container_logger
                interpretation.result.interpretation_location = e_persistence.base_dir
                job.explainer_persistence = e_persistence
                job.job_location = e_persistence.get_explainer_dir()

                # clone of the global H2O Sonar configuration
                explainer.config = h2o_sonar_config.config

                # explainer params
                explainer_params = explainers_to_run[explainer_id].params
                explainer_params_as_str = (
                    json.dumps(explainer_params)
                    if isinstance(explainer_params, dict)
                    else ""
                )

                # setup
                job.tick("setup()")
                explainer.setup(
                    model=model or model_handle,
                    models=interpretation.common_params.models,
                    persistence=e_persistence,
                    key=job.key,
                    mli_key=interpretation.key,
                    params=interpretation.common_params,
                    explainer_params_as_str=explainer_params_as_str,
                    explainer_dependencies=[],  # TODO
                    dataset_entity=dataset.meta if dataset else None,
                    validset_entity=validset.meta if validset else None,
                    testset_entity=testset.meta if testset else None,
                    dataset_api=self.container.dataset_api,
                    model_entity=model.meta if model else None,
                    model_api=self.container.model_api,
                    progress_callback=_print_progress_callback,
                    logger=explainer_logger,
                    sanitization_map={},  # TODO
                    used_features=(
                        model.meta.used_features if model and model.meta else None
                    ),
                    config=h2o_sonar_config.config,
                    on_demand_params={},  # TODO
                )
                if explainer.args and explainer.args.args:
                    interpretation.result.explainers_params[explainer_id] = (
                        explainer.args.args
                    )
                    self._save_interpretation_json(interpretation)
                # (sanitized) model target column
                s_model_target_col = (
                    model.meta.sanitization_map.to_sanitized(
                        interpretation.common_params.target_col
                    )
                    if model and model.meta and model.meta.sanitization_map
                    else interpretation.common_params.target_col
                )

                # fit
                job.tick("fit()")
                if dataset:
                    explainer.run_fit(
                        X=(
                            dataset.data[
                                :,
                                list(
                                    set(dataset.meta.column_names)
                                    - set(s_model_target_col)
                                ),
                            ]
                            if s_model_target_col
                            and s_model_target_col in list(dataset.meta.column_names)
                            else dataset.data
                        ),
                        y=(
                            dataset.data[:, s_model_target_col]
                            if s_model_target_col
                            and s_model_target_col in list(dataset.meta.column_names)
                            else None
                        ),
                        dataset=dataset,
                    )

                # explain
                job.tick("explain()")
                if dataset:
                    X = (
                        dataset.data[
                            :,
                            list(
                                set(dataset.meta.column_names) - set(s_model_target_col)
                            ),
                        ]
                        if s_model_target_col
                        and s_model_target_col in list(dataset.meta.column_names)
                        else dataset.data
                    )
                    y = (
                        dataset.data[:, s_model_target_col]
                        if s_model_target_col
                        and s_model_target_col in list(dataset.meta.column_names)
                        else None
                    )
                else:  # handle
                    X = dataset_handle
                    y = None

                explanations = explainer.run_explain(
                    X=X,
                    y=y,
                    explanations_types=None,
                    explainable_x=dataset or dataset_handle,
                    testset=testset or testset_handle,
                    validset=validset or validset_handle,
                )

                # finalize
                job.result_descriptor = e_persistence.load_result_descriptor()
                job.success()
                interpretation.register_explainer_result(
                    explainer_id, explainer.get_result()
                )
                interpretation.result.problems.extend(explainer.explain_problems())
                interpretation.update_overall_result()
                interpretation.result.insights.extend(explainer.explain_insights())

                self.container_logger.debug(
                    f"DONE execution of explainer: {explainer_id}\n"
                    f"  - created explanations: {len(explanations)}"
                )

                interpretation.set_progress(
                    step_slot_max,
                    (
                        f"{step + 1}/{progress_slot_items} "
                        f"{this.__log_evalxplainer(explainer_id)} run FINISHED"
                    ),
                )

                self._save_interpretation_json(interpretation)
            except Exception as ex:
                job.result_descriptor = (
                    e_persistence.load_result_descriptor() if e_persistence else {}
                )
                job.error = str(ex)
                job.status = commons.ExplainerJobStatus.FAILED
                self.container_logger.error(
                    f"Explainer {explainer_id} execution failed with: {ex}\n"
                    f"{traceback.format_exc()}"
                )

                # let the evaluation fail if the explainer crashed
                interpretation.result.problems.append(
                    problems.ProblemAndAction(
                        description=(
                            f"Explainer/evaluator '{explainer_id}' crashed during "
                            f"the execution: {ex}"
                        ),
                        severity=problems.ProblemSeverity.high,
                        problem_type="runtime",
                        problem_attrs={
                            "explainer_id": explainer_id,
                        },
                        actions_description=(
                            "Please analyze the error and explainer/evaluator log in "
                            "order to find out the root cause of the problem like code "
                            "defect, invalid dataset, unexpected data or other."
                        ),
                        explainer_id=explainer_id,
                    )
                )
                interpretation.status = commons.ExplainerJobStatus.FAILED
            finally:
                # force release of system resources

                # close the last/current figure
                pyplot.clf()
                # close all matplotlib figures windows
                pyplot.close("all")

        # progress
        interpretation.set_progress(0.9, "Evaluators/explainers run finished")
        interpretation.set_progress(0.91, "Shutting down H2O-3 (if it was started)")

        # H2O-3 cleanup and finalization
        if h2o_sonar_config.config.h2o_auto_cleanup:
            h2o_utils.clean_up_h2o3()
        if h2o_sonar_config.config.h2o_auto_stop:
            h2o_utils.kill_h2o3()

        # finalize interpretation
        try:
            interpretation.result.all_explainer_ids = (
                interpretation.get_all_explainer_ids()
            )
            interpretation.result.incompatible_explainer_ids = (
                interpretation.get_incompatible_explainer_ids()
            )
            interpretation.result.incompatible_explainers = incompatible_explainers
            # sort problems by severity
            if interpretation.result.problems:
                interpretation.result.problems.sort(
                    key=lambda p: p.severity.value, reverse=False
                )
            # remove duplicate cheapest/most expensive model insights
            if interpretation.result.insights:
                interpretation.result.remove_duplicate_insights()
            # sort insights by type
            if interpretation.result.insights:
                interpretation.result.insights.sort(
                    key=lambda i: i.insight_type, reverse=False
                )
        except Exception as ex:
            self.container_logger.error(
                f"Unable to sort interpretation problems: {ex}\n"
                f"{traceback.format_exc()}"
            )

        interpretation.set_progress(
            0.95, "Finalizing - saving datasets, models, and explanations"
        )
        interpretation.result.dataset = dataset
        interpretation.result.testset = testset
        interpretation.result.validset = validset
        interpretation.result.model = model or model_handle
        interpretation.result.models = interpretation.common_params.models
        self._set_interpretation_status(interpretation)
        try:
            interpretation.persistence.make_interpretation_sandbox()
            interpretation.persistence.store.save_json(
                key=interpretation.result.json_location,
                data=interpretation.to_dict(),
            )
        except Exception as ex:
            self.container_logger.error(
                f"Unable to save interpretation result interpretation.json: {ex}\n"
                f"{traceback.format_exc()}"
            )
        try:
            interpretation.persistence.save_as_html(interpretation.to_html())
        except Exception as ex:
            self.container_logger.error(
                f"Unable to save interpretation result interpretation.html: {ex}\n"
                f"{traceback.format_exc()}"
            )
        if commons.MimeType.MIME_PDF in interpretation.results_formats:
            try:
                interpretation.persistence.save_as_pdf(interpretation)
            except Exception as ex:
                self.container_logger.error(
                    f"Unable to save interpretation result "
                    f"interpretation-detailed.html "
                    f"and/or corresponding PDF file: {ex}\n{traceback.format_exc()}"
                )

        # interpretations HTML
        try:
            interpretations_paths = [
                os.path.relpath(path, self.container.persistence.base_path)
                for path in self.container.list_interpretations()
            ]
            interpretation.persistence.store.save(
                key=os.path.join(
                    interpretation.persistence.data_dir,
                    p10s.InterpretationPersistence.FILE_H2O_SONAR_HTML,
                ),
                data=interpretations.Interpretations(
                    interpretations_paths=interpretations_paths,
                    persistence=self.container.persistence,
                    branding=interpretation.branding,
                    logger=self.container_logger,
                ).to_html(branding=interpretation.branding),
            )
        except Exception as ex:
            self.container_logger.error(
                f"Unable to save interpretations listing `interpretations.html`: {ex}\n"
                f"{traceback.format_exc()}"
            )

        return interpretation

    def run(
        self, interpretation: interpretations.Interpretation
    ) -> interpretations.Interpretation:
        try:
            return self._run(interpretation)
        except Exception as ex:
            if interpretation:
                interpretation.status = commons.ExplainerJobStatus.FAILED
                interpretation.set_progress(
                    progress=1.0, message=f"Interpretation FAILED: {ex}"
                )
                interpretation.progress = 1.0  # just for sure
                self._save_interpretation_json(interpretation)

            raise ex


class LocalExplainerContainer(ExplainerContainer):
    """Local explainer container enables explainers to run on local machine."""

    TYPE_ID = "LOCAL_EXPLAINER_CONTAINER"

    def __init__(self):
        ExplainerContainer.__init__(self)

    def setup(
        self,
        results_location: str | Any = "",
        persistence_api: p10s.PersistenceApi | None = None,
        persistence_type: (
            p10s.PersistenceType | None
        ) = p10s.PersistenceType.file_system,
        logger: loggers.SonarLogger | None = None,
        log_level: int | None = None,
    ):
        ExplainerContainer.setup(
            self,
            results_location=results_location,
            persistence_api=persistence_api,
            persistence_type=persistence_type,
            logger=logger,
            log_level=log_level,
        )

        # OOTB explainers
        register_ootb_explainers(self)

        # BYOE explainers
        self.register_configured_explainers()


def _async_resolve_container(
    container: str | None = None,
    results_location: pathlib.Path | str | dict | Any | None = None,
    persistence_type: p10s.PersistenceType = p10s.PersistenceType.file_system,
    log_level: int = logging.WARNING,
):
    container = ContainerRegistry.registry().resolve_container(container)
    container.setup(
        results_location=results_location,
        persistence_type=persistence_type,
        log_level=log_level,
    )

    return container


def _async_interpretation_task(
    dataset: str | pathlib.Path,
    model: str | pathlib.Path,
    models: list[str | pathlib.Path | m4s.ExplainableModel | Any],
    target_col: str,
    explainers: list[str | commons.ExplainerToRun] | None = None,
    explainer_keywords: list[str] | None = None,
    validset: str | datatable.Frame | None = None,
    testset: str | datatable.Frame | None = None,
    use_raw_features: bool = True,
    used_features: list | None = None,
    weight_col: str = "",
    prediction_col: str = "",
    drop_cols: list | None = None,
    sample_num_rows: int | None = None,
    container: str | None = None,
    results_location: pathlib.Path | str | dict | Any | None = None,
    persistence_type: p10s.PersistenceType = p10s.PersistenceType.file_system,
    run_explainers_in_parallel: bool = False,
    key: str = "",
    branding: str = "",
    log_level: int = logging.WARNING,
    extra_params: dict | None = None,
) -> dict:
    # set branding in child process config if provided
    if branding:
        h2o_sonar_config.config.branding = branding

    # progress: bridge will be initialized when the interpretation sandbox is created
    progress_bridge = progress_utils.CallbackToFileBridge()

    container = _async_resolve_container(
        container=container,
        results_location=results_location,
        persistence_type=persistence_type,
        log_level=log_level,
    )
    resolved_results_location = container.persistence.getcwl()

    #
    # logging: - IMPORTANT - per-interpretation LOGGER @ ASYNC required
    #
    # - per-interpretation LOGGER is must - DEADLOCK prevention @ Python logging
    # - interpretation CANNOT log into H2O Sonar wide log > deadlock
    # - SAFE is if the interpretation logs into its own file
    #
    if AsyncLocalContainer.OPT_PER_INTERPRETATION_LOGGER:
        interpretation_dir = p10s.InterpretationPersistence.get_mli_dir_name(
            data_dir=resolved_results_location,
            username=commons.DEFAULT_USER,
            mli_key=key,
        )
        pathlib.Path(interpretation_dir).mkdir(parents=True, exist_ok=True)
        interpretation_file_logger = loggers.SonarFileLogger(
            logger_name=f"H2oSonarInterpretationLogger-{key}",
            log_level=log_level,
            log_file=str(
                pathlib.Path(interpretation_dir)
                / p10s.InterpretationPersistence.get_async_log_file_name(key)
            ),
        )

    else:
        interpretation_file_logger = None

    interpretation = container.run_interpretation(
        dataset=dataset,
        target_col=target_col,
        explainers=explainers,
        model=model,
        models=models,
        validset=validset,
        testset=testset,
        use_raw_features=use_raw_features,
        used_features=used_features,
        weight_col=weight_col,
        prediction_col=prediction_col,
        drop_cols=drop_cols,
        sample_num_rows=sample_num_rows,
        results_location=resolved_results_location,
        explainer_keywords=explainer_keywords,
        run_asynchronously=False,
        run_explainers_in_parallel=run_explainers_in_parallel,
        key=key,
        progress_callback=progress_bridge,
        # override default logging of all interpretations to H2O Sonar wide log
        logger=interpretation_file_logger,
        extra_params=extra_params,
    )

    return interpretation.to_dict()


#
# ASYNC local container
#


class AsyncLocalContainer(ExplainerContainer):
    """Local container which runs interpretations asynchronously."""

    # enable per-interpretation logger to prevent Python logging deadlocks @ async
    OPT_PER_INTERPRETATION_LOGGER: bool = True

    TYPE_ID = "ASYNC_LOCAL_EXPLAINER_CONTAINER"

    @property
    def executor(self):
        if not self._executor:
            self._executor = futures.ProcessPoolExecutor(max_workers=self.max_workers)
        return self._executor

    def __init__(self, max_workers: int | None = 3):
        ExplainerContainer.__init__(self)

        self.max_workers = max_workers

    def setup(
        self,
        results_location: str | Any = "",
        persistence_api: p10s.PersistenceApi | None = None,
        persistence_type: (
            p10s.PersistenceType | None
        ) = p10s.PersistenceType.file_system,
        logger=None,
        log_level: int | None = None,
    ):
        ExplainerContainer.setup(
            self,
            results_location=results_location,
            persistence_api=persistence_api,
            persistence_type=persistence_type,
            logger=logger,
            log_level=log_level,
        )

        # executor is initialized on-demand using the property to save system resources,
        # it can be shut down using gc() if needed
        self._executor = None

    def run_interpretation(
        self,
        dataset: (
            str
            | pathlib.Path
            | datasets.ExplainableDataset
            | datatable.Frame
            | pandas.DataFrame
        ),
        model: str | pathlib.Path | m4s.ExplainableModel | Any,
        models: list[str | pathlib.Path | m4s.ExplainableModel | Any],
        target_col: str,
        explainers: list[str | commons.ExplainerToRun] | None = None,
        explainer_keywords: list[str] | None = None,
        validset: (
            str
            | pathlib.Path
            | datasets.ExplainableDataset
            | datatable.Frame
            | pandas.DataFrame
            | None
        ) = None,
        testset: (
            str
            | pathlib.Path
            | datasets.ExplainableDataset
            | datatable.Frame
            | pandas.DataFrame
            | None
        ) = None,
        use_raw_features: bool = True,
        used_features: list | None = None,
        weight_col: str = "",
        prediction_col: str = "",
        drop_cols: list | None = None,
        sample_num_rows: int | None = 0,
        sampler: sampling.DatasetSampler | None = None,
        results_location: str | pathlib.Path | dict | None = None,
        results_formats: list[str] | None = None,
        progress_callback: progress_utils.AbstractProgressCallbackContext | None = None,
        run_asynchronously: bool = False,
        run_explainers_in_parallel: bool = False,
        key: str = "",
        extra_params: list | None = None,
    ) -> interpretations.Interpretation:
        interpretation = interpretations.Interpretation(
            common_params=commons.CommonInterpretationParams(
                model=model,
                models=models,
                dataset=dataset,
                target_col=target_col,
                validset=validset,
                testset=testset,
                use_raw_features=use_raw_features,
                weight_col=weight_col,
                prediction_col=prediction_col,
                drop_cols=drop_cols,
                sample_num_rows=sample_num_rows,
                results_location=results_location,
                extra_params=extra_params,
                used_features=used_features,
            ),
            explainers=explainers,
            explainer_keywords=explainer_keywords,
            created=time.time(),
            key=commons.generate_key(),
            sampler=sampler,
            results_formats=results_formats,
            progress_callback=progress_callback,
            logger=self.logger,
            extra_params=extra_params,
        )

        # condition: ensure serializable parameters
        if model and not isinstance(model, (str, pathlib.Path)):
            raise ValueError(
                "Path to the model is required in case of asynchronous interpretation."
            )
        if not isinstance(dataset, (str, pathlib.Path)):
            # fallback to JAY serialization
            if isinstance(dataset, datasets.LlmDataset):
                dataset_path = (
                    pathlib.Path(results_location) / "llm_dataset_for_async.bin"
                )
                dataset.to_datatable().to_jay(str(dataset_path))
            else:
                raise ValueError(
                    "Path to the dataset is required in case of asynchronous "
                    "interpretation."
                )
        if testset and not isinstance(testset, (str, pathlib.Path)):
            raise ValueError(
                "Path to the test dataset is required in case of asynchronous "
                "interpretation."
            )
        if validset and not isinstance(validset, (str, pathlib.Path)):
            raise ValueError(
                "Path to the validation dataset is required in case of asynchronous "
                "interpretation."
            )
        if sampler:
            raise ValueError(
                "Sampler cannot be be specified in case of asynchronous interpretation."
            )

        # resolve interpretation fields
        interpretation.container = _async_resolve_container(
            container=None,
            results_location=results_location,
            persistence_type=self.persistence_type,
            log_level=self.log_level,
        )
        interpretation.result.results_location = (
            interpretation.container.persistence.getcwl()
        )

        interpretation.future = self.executor.submit(
            _async_interpretation_task,
            dataset,
            model,
            models,
            target_col,
            explainers=explainers,
            explainer_keywords=explainer_keywords,
            validset=validset,
            testset=testset,
            use_raw_features=use_raw_features,
            used_features=used_features,
            weight_col=weight_col,
            prediction_col=prediction_col,
            drop_cols=drop_cols,
            sample_num_rows=sample_num_rows,
            results_location=results_location,
            persistence_type=self.persistence_type,
            run_explainers_in_parallel=run_explainers_in_parallel,
            key=interpretation.key,
            branding=h2o_sonar_config.config.branding,
            log_level=self.log_level,
            extra_params=extra_params,
        )

        # progress:
        #
        # - reporting from interpretations is done via file system
        #   (global queue was deadlocking)
        # - file system race conditions prevention:
        #   > each interpretation has its own progress file (in its directory)
        #   > progress file is created when the interpretation starts
        #   > progress file is read by exactly one poller thread from
        #     AsyncProgressFileToCallbackPoller
        #   > progress poller thread is daemon
        #
        # AsyncProgressFileToCallbackPoller progress bridge:
        # - poller reads progress from the filesystem and calls progress callback
        #   (if the progress changes)
        # - poller thread stops when interpretation finishes
        # - poller is NOT run if user doesn't ask for progress updates by providing
        #   callback
        try:
            if progress_callback:
                i_c = interpretation.container
                interpretation.persistence = (
                    i_c.persistence_api.create_interpretation_persistence(
                        store_persistence=i_c.persistence,
                        base_path=interpretation.result.results_location,
                        interpretation_key=interpretation.key,
                        username=commons.DEFAULT_USER,
                    )
                )
                interpretation.result.interpretation_location = (
                    interpretation.persistence.base_dir
                )

                interpretation.progress_callback.set_progress_poller(
                    progress_utils.AsyncProgressFileToCallbackPoller(
                        progress_file=interpretation.result.get_progress_location(
                            absolute_path=True
                        ),
                        progress_callback=interpretation.progress_callback,
                        target_entity=interpretation,
                    )
                )
        except Exception as ex:
            interpretation.logger.error(
                f"Unable to initialize progress poller for interpretation: {ex}\n"
                f"{traceback.format_exc()}"
            )

        return interpretation

    def gc(self):
        # THREAD SAFE: test & set (avoids shutdown() to be invoked more than once)
        executor, self._executor = self._executor, None
        if executor:
            executor.shutdown()


class ContainerRegistry:
    """Explainer container factory and registry class is able to instantiate
    containers using symbolic names.

    """

    # SINGLETON container registry
    __registry = None
    # SINGLETON: secret key to prevent instantiation using constructor

    __singleton_secret_key = object()

    @classmethod
    def registry(cls):
        if not ContainerRegistry.__registry:
            ContainerRegistry.__registry = ContainerRegistry(cls.__singleton_secret_key)
            ContainerRegistry.__registry.register_container_type(
                container_type=AsyncLocalContainer,
                container_type_id=AsyncLocalContainer.TYPE_ID,
            )

        return ContainerRegistry.__registry

    def __init__(self, singleton_create_key):
        # singleton: constructor instantiation protection
        assert singleton_create_key == ContainerRegistry.__singleton_secret_key, (
            "Explainer container registry must be created using registry() method"
        )

        # container type symbolic name (string) to container type (class)
        self.container_types: dict = dict()
        # container type symbolic name (string) to container instance (reusability)
        self.containers: dict = dict()

        # OOTB containers
        self.container_types[LocalExplainerContainer.TYPE_ID] = LocalExplainerContainer

    @classmethod
    def register_container_type(
        cls,
        container_type: type[ExplainerContainer],
        container_type_id: str,
    ):
        cls.registry().container_types[container_type_id] = container_type

    def get_container_types(self) -> list:
        return list(self.container_types.values())

    def create_container(self, container_type_id):
        """Get instance of the container for given container type."""
        return self.container_types[container_type_id]()

    def list_containers(self) -> dict:
        raise NotImplementedError

    def resolve_container(
        self,
        container: str | ExplainerContainer | None = None,
    ):
        if container:
            if isinstance(container, str):
                if container in self.containers:
                    container_instance = self.containers[container]
                elif container in self.container_types:
                    container_instance = self.create_container(container)
                    self.containers[container] = container_instance
                else:
                    raise ValueError(f"Unknown explainer container type: '{container}'")
            elif isinstance(container, ExplainerContainer):
                if container.TYPE_ID:
                    if container.TYPE_ID not in self.container_types:
                        self.container_types[container.TYPE_ID] = container.__class__
                    if container.TYPE_ID not in self.containers:
                        self.containers[container.TYPE_ID] = container
                    container_instance = container
                else:
                    raise ValueError(
                        f"Invalid explainer container type ID of : '{container}'"
                    )
            else:
                raise ValueError(
                    f"Unsupported explainer container type: {type(container)}"
                )
        else:
            if LocalExplainerContainer.TYPE_ID not in self.container_types:
                self.container_types[LocalExplainerContainer.TYPE_ID] = (
                    LocalExplainerContainer
                )
            if LocalExplainerContainer.TYPE_ID not in self.containers:
                self.containers[LocalExplainerContainer.TYPE_ID] = (
                    LocalExplainerContainer()
                )
            container_instance = self.containers[LocalExplainerContainer.TYPE_ID]

        return container_instance


def register_ootb_explainers(container: ExplainerContainer):
    from h2o_sonar.evaluators import agent_sanity_check_evaluator
    from h2o_sonar.evaluators import agentic_fact_check_evaluator
    from h2o_sonar.evaluators import answer_accuracy_evaluator
    from h2o_sonar.evaluators import (
        answer_semantic_similarity_per_sentence_evaluator as asspse,
    )
    from h2o_sonar.evaluators import bertscore_evaluator
    from h2o_sonar.evaluators import bleu_evaluator
    from h2o_sonar.evaluators import classification_evaluator
    from h2o_sonar.evaluators import contact_information_byop_evaluator
    from h2o_sonar.evaluators import encoding_guardrail_evaluator
    from h2o_sonar.evaluators import fairness_bias_evaluator
    from h2o_sonar.evaluators import gptscore_machine_translation_evaluator
    from h2o_sonar.evaluators import gptscore_question_answering_evaluator
    from h2o_sonar.evaluators import gptscore_summary_with_reference_evaluator
    from h2o_sonar.evaluators import gptscore_summary_without_reference_evaluator as gsw
    from h2o_sonar.evaluators import json_schema_evaluator
    from h2o_sonar.evaluators import language_mismatch_byop_evaluator
    from h2o_sonar.evaluators import looping_detection_evaluator
    from h2o_sonar.evaluators import parameterizable_byop_evaluator
    from h2o_sonar.evaluators import perplexity_evaluator
    from h2o_sonar.evaluators import pii_leakage_evaluator
    from h2o_sonar.evaluators import questions_drift_evaluator
    from h2o_sonar.evaluators import rag_answer_correctness_evaluator
    from h2o_sonar.evaluators import rag_answer_relevancy_evaluator
    from h2o_sonar.evaluators import rag_answer_relevancy_no_judge_evaluator
    from h2o_sonar.evaluators import rag_answer_similarity_evaluator
    from h2o_sonar.evaluators import rag_chunk_relevancy_evaluator
    from h2o_sonar.evaluators import rag_context_mean_reciprocal_rank_evaluator
    from h2o_sonar.evaluators import rag_context_precision_evaluator
    from h2o_sonar.evaluators import rag_context_recall_evaluator
    from h2o_sonar.evaluators import rag_context_relevancy_evaluator
    from h2o_sonar.evaluators import rag_faithfulness_evaluator
    from h2o_sonar.evaluators import rag_groundedness_evaluator
    from h2o_sonar.evaluators import rag_hallucination_evaluator
    from h2o_sonar.evaluators import rag_ragas_evaluator
    from h2o_sonar.evaluators import rag_tokens_presence_evaluator
    from h2o_sonar.evaluators import rouge_evaluator
    from h2o_sonar.evaluators import self_consistency_evaluator
    from h2o_sonar.evaluators import sensitive_data_leakage_evaluator
    from h2o_sonar.evaluators import sexism_byop_evaluator
    from h2o_sonar.evaluators import stereotype_byop_evaluator
    from h2o_sonar.evaluators import summarization_byop_evaluator
    from h2o_sonar.evaluators import toxicity_evaluator
    from h2o_sonar.explainers import adversarial_similarity_explainer
    from h2o_sonar.explainers import backtesting_explainer
    from h2o_sonar.explainers import calibration_score_explainer as cs_explainer
    from h2o_sonar.explainers import dataset_and_model_insights_explainer
    from h2o_sonar.explainers import dia_explainer
    from h2o_sonar.explainers import drift_explainer
    from h2o_sonar.explainers import dt_surrogate_explainer
    from h2o_sonar.explainers import fi_kernel_shap_explainer
    from h2o_sonar.explainers import fi_naive_shapley_explainer
    from h2o_sonar.explainers import friedman_h_statistic_explainer
    from h2o_sonar.explainers import morris_sa_explainer
    from h2o_sonar.explainers import pd_2_features_explainer
    from h2o_sonar.explainers import pd_ice_explainer
    from h2o_sonar.explainers import residual_dt_surrogate_explainer
    from h2o_sonar.explainers import residual_pd_ice_explainer
    from h2o_sonar.explainers import segment_performance_explainer
    from h2o_sonar.explainers import size_dependency_explainer
    from h2o_sonar.explainers import summary_shap_explainer
    from h2o_sonar.explainers import transformed_fi_shapley_explainer as t_fi_explainer

    container.explainers_registry.register(
        agent_sanity_check_evaluator.AgentSanityCheckEvaluator,
    )
    container.explainers_registry.register(
        fi_naive_shapley_explainer.NaiveShapleyMojoFeatureImportanceExplainer,
    )
    container.explainers_registry.register(
        summary_shap_explainer.SummaryShapleyExplainer,
    )
    container.explainers_registry.register(
        dt_surrogate_explainer.DecisionTreeSurrogateExplainer
    )
    container.explainers_registry.register(
        pd_ice_explainer.PdIceExplainer,
    )
    container.explainers_registry.register(dia_explainer.DiaExplainer)
    container.explainers_registry.register(
        t_fi_explainer.ShapleyMojoTransformedFeatureImportanceExplainer
    )
    container.explainers_registry.register(
        residual_dt_surrogate_explainer.ResidualDecisionTreeSurrogateExplainer
    )
    container.explainers_registry.register(
        residual_pd_ice_explainer.ResidualPdIceExplainer
    )
    container.explainers_registry.register(
        fi_kernel_shap_explainer.KernelShapFeatureImportanceExplainer
    )
    container.explainers_registry.register(
        pd_2_features_explainer.PdFor2FeaturesExplainer
    )
    container.explainers_registry.register(
        friedman_h_statistic_explainer.FriedmanHStatisticExplainer
    )
    container.explainers_registry.register(
        morris_sa_explainer.MorrisSensitivityAnalysisExplainer
    )
    container.explainers_registry.register(
        dataset_and_model_insights_explainer.DatasetAndModelInsightsExplainer
    )

    # H2O MV
    container.explainers_registry.register(drift_explainer.DriftDetectionExplainer)
    container.explainers_registry.register(
        adversarial_similarity_explainer.AdversarialSimilarityExplainer
    )
    container.explainers_registry.register(
        size_dependency_explainer.SizeDependencyExplainer
    )
    container.explainers_registry.register(
        segment_performance_explainer.SegmentPerformanceExplainer
    )
    container.explainers_registry.register(cs_explainer.CalibrationScoreExplainer)
    container.explainers_registry.register(backtesting_explainer.BacktestingExplainer)

    # LLM
    container.explainers_registry.register(
        asspse.AnswerSemanticSimilarityPerSentenceEvaluator
    )
    container.explainers_registry.register(rag_ragas_evaluator.RagasEvaluator)
    container.explainers_registry.register(
        rag_tokens_presence_evaluator.RagStrStrEvaluator
    )
    container.explainers_registry.register(
        answer_accuracy_evaluator.AnswerAccuracyEvaluator
    )
    container.explainers_registry.register(
        rag_answer_correctness_evaluator.AnswerCorrectnessEvaluator
    )
    container.explainers_registry.register(
        rag_answer_similarity_evaluator.AnswerSemanticSimilarityEvaluator
    )
    container.explainers_registry.register(
        rag_chunk_relevancy_evaluator.ContextChunkRelevancyEvaluator
    )
    container.explainers_registry.register(
        rag_context_relevancy_evaluator.ContextRelevancyEvaluator
    )
    container.explainers_registry.register(
        rag_answer_relevancy_evaluator.AnswerRelevancyEvaluator,
    )
    container.explainers_registry.register(
        rag_answer_relevancy_no_judge_evaluator.RagAnswerRelevancyNoJudgeEvaluator,
    )
    container.explainers_registry.register(
        rag_context_precision_evaluator.ContextPrecisionEvaluator,
    )
    container.explainers_registry.register(
        rag_context_recall_evaluator.ContextRecallEvaluator,
    )
    container.explainers_registry.register(
        rag_context_mean_reciprocal_rank_evaluator.MeanReciprocalRankEvaluator,
    )
    container.explainers_registry.register(
        rag_faithfulness_evaluator.FaithfulnessEvaluator,
    )
    container.explainers_registry.register(
        pii_leakage_evaluator.PiiLeakageEvaluator,
    )
    container.explainers_registry.register(
        json_schema_evaluator.JSONSchemaEvaluator,
    )
    container.explainers_registry.register(
        sensitive_data_leakage_evaluator.SensitiveDataLeakageEvaluator
    )
    container.explainers_registry.register(toxicity_evaluator.ToxicityEvaluator)
    container.explainers_registry.register(
        fairness_bias_evaluator.FairnessBiasEvaluator
    )
    container.explainers_registry.register(
        contact_information_byop_evaluator.ContactInformationByopEvaluator
    )
    container.explainers_registry.register(
        language_mismatch_byop_evaluator.LanguageMismatchByopEvaluator
    )
    container.explainers_registry.register(
        looping_detection_evaluator.LoopingDetectionEvaluator
    )
    container.explainers_registry.register(
        parameterizable_byop_evaluator.ParameterizableByopEvaluator
    )
    container.explainers_registry.register(perplexity_evaluator.PerplexityEvaluator)
    container.explainers_registry.register(
        questions_drift_evaluator.QuestionsDriftEvaluator
    )
    container.explainers_registry.register(sexism_byop_evaluator.SexismByopEvaluator)
    container.explainers_registry.register(
        stereotype_byop_evaluator.StereotypeByopEvaluator
    )
    container.explainers_registry.register(
        summarization_byop_evaluator.SummarizationByopEvaluator
    )
    container.explainers_registry.register(
        rag_groundedness_evaluator.RagGroundednessEvaluator
    )
    container.explainers_registry.register(
        rag_hallucination_evaluator.RagHallucinationEvaluator
    )
    container.explainers_registry.register(bertscore_evaluator.BertscoreEvaluator)
    container.explainers_registry.register(bleu_evaluator.BleuEvaluator)
    container.explainers_registry.register(rouge_evaluator.RougeEvaluator)
    container.explainers_registry.register(
        self_consistency_evaluator.SelfConsistencyEvaluator
    )
    container.explainers_registry.register(
        classification_evaluator.ClassificationEvaluator
    )
    container.explainers_registry.register(
        gptscore_summary_with_reference_evaluator.GptScoreSummaryWithReferenceEvaluator
    )
    container.explainers_registry.register(gsw.GptScoreSummaryWithoutReferenceEvaluator)
    container.explainers_registry.register(
        gptscore_question_answering_evaluator.GptScoreQuestionAnsweringEvaluator
    )
    container.explainers_registry.register(
        gptscore_machine_translation_evaluator.GptScoreMachineTranslationEvaluator
    )
    container.explainers_registry.register(
        encoding_guardrail_evaluator.EncodingGuardrailEvaluator
    )
    # agent-based evaluators
    container.explainers_registry.register(
        agentic_fact_check_evaluator.FactCheckAgenticEvaluator
    )
