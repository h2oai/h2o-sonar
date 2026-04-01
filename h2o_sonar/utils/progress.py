# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
"""H2O Sonar progress reporting utilities.

- entities with built-in progress reporting capabilities
  - Evaluation and Interpretation
  - Evaluator and Explainer

- CUSTOM progress callback can be set on
  - evaluate.py::run_evaluation()
  - interpret.py::run_interpretation()

- callback stacking:
  custom_callback = MyCustomCallback(AbstractProgressCallbackContext)()
    ... e.g. callback from the user
  parent_callback = ProgressCallback*(total_steps=10, parent_callback=custom_callback)
    ... e.g. callback of the container KNOWING that it has 10 evaluators
    child_callback = ProgressCallbackStackingBridge(parent_callback)
        ... child_callback is created for every evaluator & SET on that evaluator
        child_callback.set_progress(0.01)
          ... evaluator reports its progress to the parent container callback
        child_callback.set_progress(0.5)
          ... evaluator reports its progress to the parent container callback
        child_callback.set_progress(1.0)
          ... evaluator reports its progress to the parent container callback

- DEFAULT progress callback hierarchy chaining (evaluation):
  1. evaluate.py::run_evaluation()
    -> parameter progress_callback ... set to None (default)
  2. interpret.py::run_interpretation()
    -> parameter progress_callback ... None (default)
    -> container.run_interpretation(..., progress_callback, ...) ... None (default)
  3. explainer_container.py::ExplainerContainer::run_interpretation()/run()/_run()
    -> parameter progress_callback ... None (default)
    -> INIT of progress_callback w/ DEFAULT callback:
       LoggingProgressCallbackContext(0.0, 1.0, progress)
      - 0% - 10%  ... reserved for initialization
        - progress is reported manually within this range
      - 10% - 90% ... evenly split for evaluators/explainers e.g. 10x sub-callbacks
        - for every evaluator is created sub-callback for ITS range & set on evaluator
          e.g. container_callback.get_sub_callback_for_progress(0.33, 0.44)
      - 0% - 90%  ... reserved for initialization
        - progress is reported manually within this range
    -> CHAINING of container callback to evaluator/explainer callbacks:
      - every evaluator/explainer:
        - gets progress callback INJECTED from the parent:
          self.progress_callback
        - EITHER it can simply report its progress in 0.0 to 1.0 range using:
          self.progress_callback.set_progress(progress)
        - OR it can create sub-callback(s) based on its requirements if needed
          self.progress_callback.get_sub_callback_for_progress(min_range, max_range)

See evaluation progress reporting test / mock in

- tests/utils/test_progress.py::test_evaluate_default_progress
- tests/utils/test_progress.py::test_evaluate_custom_progress

"""

import abc
import multiprocessing
import pathlib
import queue
import threading
import time
import traceback
from collections.abc import Callable

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar import loggers
from h2o_sonar.lib.api import persistences


# progress quantum
EPSILON = 0.0001
# message which terminates the progress poller
PROGRESS_POLLER_TERMINATE_MESSAGE = "__async_progress_poller_t3rm1nat0r__"
# dict keys
KEY_PROGRESS = "progress"
KEY_MESSAGE = "message"

#
# config
#

if h2o_sonar_config.config.mp_start_method:
    multiprocessing.set_start_method(
        h2o_sonar_config.config.mp_start_method, force=True
    )

#
# CORE progress reporting framework
#


class AbstractProgressCallbackContext(abc.ABC):
    """Abstract progress callback context which can be used as base class on
    implementation of custom callback contexts.

    """

    @property
    def progress(self) -> float:
        """Progress in ``[0, 1]`` range as set using the ``set_progress() method."""
        return self._progress

    @property
    def progress_percent(self) -> int:
        return int(self._progress * 100.0)

    @property
    def relative_progress(self) -> float:
        """Progress normalized to the ``[min_progress, max_progress]`` range
        as specified in the constructor.

        """
        return self._relative_progress

    @property
    def relative_progress_percent(self) -> int:
        return int(self._relative_progress * 100.0)

    @property
    def message(self) -> str:
        """Progress message."""
        return self._message or ""

    def __init__(self):
        self._progress = 0.0
        self._relative_progress = 0.0
        self._message = ""
        self.name = ""

        self.verbose_children = True

        # optional auxiliary poller which pulls progress events from various sources
        # like queues, files, etc.
        self._progress_poller = None

    @abc.abstractmethod
    def set_progress(self, progress: float, message: str | None = None) -> float:
        """Override this method to get progress value in a custom callback context.

        Parameters
        ----------
        progress : float
            Progress of the calculation is expected in the ``[0.0, 1.0]`` range
            (if the ``progress`` is out of the range, then it is normalized to this
            interval).
        message : str
            Optional message to be reported with the progress.
            If ``""``, then previous message is cleared.
            If ``None``, then message is kept.

        Returns
        -------
        float :
            Progress (normalized) of the calculation.

        """
        raise NotImplementedError

    def set_progress_poller(self, poller):
        """Set progress poller to pull progress events from various sources like
        queues, files, etc.

        """
        self._progress_poller = poller


#
# METHOD progress reporting framework
#


class ProgressCallbackContext(AbstractProgressCallbackContext):
    """Progress callback context class used by methods to report progress.

    There are options how this class can evaluate the progress - the option must
    be set at the BEGINNING, and then it cannot be changed:

    1. initialize this class with a **float progress** in ``[0, 1]`` range and then
       use ``set_progress()`` method to report the progress as float
    2. initialize this class with a **total number of calculation steps** (can be any
       positive integer) and then use ``set_steps()`` method to report progress
       as integer in ``[0, total number of steps]`` range

    Float progress might be optionally restricted to a smaller range to enable
    progress report callbacks stacking.

    """

    def __init__(
        self,
        min_progress: float = 0.0,
        max_progress: float = 1.0,
        progress: float = 0.0,
        message: str | None = None,
        total_steps: int = 0,
        do_update: list | None = None,
        verbose_children: bool = True,
        parent_callback: AbstractProgressCallbackContext | None = None,
        name: str = "",
    ):
        """Constructor.

        Parameters
        ----------
        min_progress : float
            Minimum progress value which can be used to specify progress in
            ``[min_progress, max_progress]`` range where ``min_progress`` is bigger
            or equal to 0 *and* smaller or equal to ``max_progress``.
        max_progress : float
            Maximum progress value which can be used to specify progress in
            ``[min_progress, max_progress]`` range where ``max_progress`` is smaller
            or equal to 1 and bigger or equal to ``min_progress``.
        progress : float
            Optional progress initialization of the target methods or algorithm
            in ``[min_progress, max_progress]`` range.
        message : str | None
            Optional message to be reported with the progress.
        total_steps : int
            Progress value can be alternatively calculated by providing total number
            of steps (and finished steps later) - see class documentation.
        do_update : list | None
            List of entities to be updated with the progress. This can be used to
            update the progress of the entities which are not directly connected to
            the progress report context. It is expected that the entity has
            ``progress`` field which might be set.
        verbose_children : bool
            Control whether child callbacks which print to log/standard output/error
            should be muted or not.
        parent_callback : AbstractProgressCallbackContext | None
            Parent callback which can be used to stack callback contexts. Parent
            callback is meant to provide progress from this progress context to
            a parent callback context. This enables progress callback contexts
            stacking.
        name : str
            Name of the progress callback context.

        """
        AbstractProgressCallbackContext.__init__(self)

        # [min, max] progress: if callbacks are stacked, then progress is normalized to
        # the RELATIVE range of the given parent callback [min_progress, max_progress]
        # i.e. the range does NOT have to be [0, 1]
        self._relative_progress = -1.0
        # [0.0, 1.0] progress: as SET using set_progress() method,
        # normalization is made just to ensure that the progress is in [0, 1] range,
        # it IGNORES min_progress and max_progress
        self._progress = -1.0

        self._message = message or ""
        self._parent_callback = parent_callback
        self._total_steps: int = self.set_total_steps(total_steps)
        self._min_progress: float = ProgressCallbackContext._normalize_progress(
            min_progress
        )
        self._max_progress: float = ProgressCallbackContext._normalize_progress(
            max_progress
        )
        if self._min_progress > self._max_progress:
            self._min_progress = 0.0
            self._max_progress = 1.0
        self._range = self._max_progress - self._min_progress
        # entities whose `progress` field will be updated with the progress
        self._do_update = do_update or []
        self.verbose_children = verbose_children

        self.name = (
            name or f"Progress callback context [{min_progress}, {max_progress}]"
        )

        # initialize self._progress and self._relative_progress
        ProgressCallbackContext._set_progress(self, progress=progress)

    @property
    def parent_callback(self) -> AbstractProgressCallbackContext | None:
        return self._parent_callback

    def _relativize_progress(self, progress: float) -> float:
        """Recalculate [0.0, 1.1] progress to the [min_progress, max_progress] range."""

        # if total steps are not set, then normalize progress to the [min, max] range
        relative_progress = (
            self._min_progress + self._range * progress
            if progress
            else self._min_progress
        )
        # ensure [min_progress, max_progress] range
        if relative_progress < self._min_progress:
            relative_progress = self._min_progress
        if relative_progress > self._max_progress:
            relative_progress = self._max_progress

        return relative_progress

    #
    # progress utils
    #

    @staticmethod
    def one_pct_for_steps(total_steps: int) -> float:
        """Calculate one percent of the total steps."""
        return float(float(total_steps) / 100.0) if total_steps else 0.0

    @staticmethod
    def progress_for_steps(steps: int, total_steps: int) -> float:
        """Calculate float progress in [0.0, 1.0] range for given total and
        current steps.

        """
        return (
            (float(steps) / ProgressCallbackContext.one_pct_for_steps(total_steps))
            / 100.0
            if total_steps
            else 0.0
        )

    @staticmethod
    def step_loop_prepare(
        progress_min: float,
        progress_max: float,
        steps: int,
    ) -> tuple[float, float, int]:
        """Helper to prepare variables used to track progress of ``steps`` iterations
        in the [progress_min, progress_max] range.

        Returns
        -------
        Tuple[float, float] :
          Returns the first slot minimum, slot size and progress steps.

        """
        total_slot_size = (
            progress_max - progress_min if progress_max > progress_min else 0.0
        )
        progress_slot_size = total_slot_size / float(steps) if steps else 0.0

        return progress_min + EPSILON, progress_slot_size, steps

    @staticmethod
    def step_loop_get_min_and_max(
        step: int,
        progress_slot_min: float,
        progress_slot_size: float,
    ):
        """Recalculate progress slot min and max for given step."""
        progress_slot_min = progress_slot_min + (progress_slot_size + EPSILON) * float(
            step
        )

        progress_slot_max = progress_slot_min + progress_slot_size

        return progress_slot_min, progress_slot_max

    #
    # core
    #

    @staticmethod
    def _normalize_progress(progress: float) -> float:
        """Normalize progress to the [0.0, 1.0] range."""
        return 0.0 if progress < 0.0 else (1.0 if progress > 1.0 else progress)

    def _set_progress(self, progress: float) -> bool:
        """Set progress and relative progress, return True if the progress changed."""
        last_progress = self._progress

        # ensure [0.0, 1.0] range
        self._progress = ProgressCallbackContext._normalize_progress(progress)
        # ensure [min_progress, max_progress] range
        self._relative_progress = ProgressCallbackContext._normalize_progress(
            self._relativize_progress(progress)
        )

        return True if last_progress == self._progress else False

    def set_progress(self, progress: float, message: str | None = None) -> float:
        if message is not None:
            self._message = message

        # recalculate progress and relative progress,
        # skip parent callback & entities update if progress did NOT change
        if self._set_progress(progress):
            return self.progress

        if self._parent_callback:
            self._parent_callback.set_progress(
                progress=self._relative_progress, message=message
            )

        # set the progres normalized to [0, 1] range to all entities
        for entity in self._do_update:
            try:
                entity.progress = self._progress
                if hasattr(entity, "progress_message"):
                    entity.progress_message = message
                elif hasattr(entity, "message"):
                    entity.message = message
            except Exception as ex:
                print(
                    f"Error: unable to set progress on entity '{entity}': {ex}\n"
                    f"{traceback.format_exc()}"
                )

        return self.progress

    def set_total_steps(self, total_steps: int) -> int:
        """Set total steps to subsequently report progress in integer steps
        (rather than using float progress).

        """
        self._total_steps: int = total_steps if total_steps > 0 else 0
        return total_steps

    def set_steps(self, steps: int, message: str | None = None) -> float:
        """Set current (performed) steps to report progress in integer steps
        (rather than using float progress).

        Returns
        -------
        float :
            Progress in the [0.0, 1.0] range.

        """
        if steps > self._total_steps:
            self._progress = 1.0
            self._relative_progress = self._max_progress
        elif steps <= 0:
            self._progress = 0.0
            self._relative_progress = self._min_progress
        else:
            # recalculate progress to the [0.0, 1.0] range
            pct_1 = float(float(self._total_steps) / 100.0)
            pct = float(float(steps) / pct_1 / 100.0) if pct_1 else 0.0
            self.set_progress(progress=pct, message=message)

        return self.progress

    def get_range_for_step(self) -> float:
        return self._range / float(self._total_steps) if self._total_steps else 0.0

    def get_sub_callback_for_steps(self, total_steps: int):
        """Get sub-callback for progress tracking where will be used (total and
        finished) steps to report progress.

        """
        bottom_step_range = self.progress
        return ProgressCallbackContext(
            total_steps=total_steps,
            min_progress=bottom_step_range,
            max_progress=bottom_step_range + self.get_range_for_step(),
            parent_callback=self,
        )

    def get_sub_callback_for_progress(
        self,
        min_progress: float,
        max_progress: float,
        do_update: list | None = None,
        verbose_children: bool = True,
        name: str = "",
    ):
        """Get sub-callback for progress tracking of the given progress interval.
        Child progress callback of this progress callback will report progress in
        range ``[0.0, 1.0]``, but it will be added to this callback only withing
        ``min_progress`` and ``max_progress`` range.

        """
        return ProgressCallbackContext(
            min_progress=min_progress + EPSILON,
            max_progress=max_progress,
            do_update=do_update,
            verbose_children=verbose_children,
            parent_callback=self,
            name=name,
        )


class CallbackToQueueBridge(ProgressCallbackContext):
    """Utility which provides progress callback API and routes progress updates made
    using ``set_progress() to given queue as messages. This can be used to bridge
    progress updates from a subprocess to the main process.

    """

    def __init__(
        self,
        # TRICK type hints: multiprocessing.Queue is factory function which returns vvv
        progress_queue: queue.Queue | multiprocessing.queues.Queue,
        job_id: str = "",
        worker_id: int = 0,
    ):
        """Constructor.

        Parameters
        ----------
        progress_queue : Union[queue.Queue, multiprocessing.Queue]
            Queue to which the progress updates will be routed.
        job_id : str
            Optional identifier for the job being processed.
        worker_id : int, default=0
            Identifier for the worker processing the job.

        """
        ProgressCallbackContext.__init__(self, name="CallbackToQueueBridge")
        self.progress_queue = progress_queue
        self.job_id = job_id
        self.worker_id = worker_id
        self.logger = loggers.SonarPrintLogger()

    def set_progress(self, progress: float, message: str | None = None) -> float:
        self.logger.debug(
            f"CallbackToQueueBridge: set_progress({progress}, {message})"
            f"\n job_id: {self.job_id} worker_id: {self.worker_id}"
        )
        self.progress_queue.put((self.job_id, self.worker_id, progress, message))
        return progress


class CallbackToFileBridge(ProgressCallbackContext):
    """Utility which provides progress callback API and routes progress updates made
    using ``set_progress() to given file. This can be used to bridge
    progress updates from a subprocess to the main process.

    """

    def __init__(
        self,
        progress_file_path: pathlib.Path | str | None = None,
    ):
        """Constructor.

        Parameters
        ----------
        progress_file_path : str | pathlib.Path
            Path to the progress file.

        """
        ProgressCallbackContext.__init__(self, name="CallbackToFileBridge")
        self.progress_file_path = progress_file_path
        # ASYNC processes CANNOT log using logger - Python logging deadlocks
        self.logger = loggers.SonarPrintLogger()

    def set_progress(self, progress: float, message: str | None = None) -> float:
        self.logger.debug(
            f"CallbackToFileBridge: set_progress({progress}, {message})"
            f"\n file: {self.progress_file_path}"
        )
        try:
            if self.progress_file_path:
                persistences.FilesystemPersistence.save_json(
                    self.progress_file_path,
                    {KEY_PROGRESS: progress, KEY_MESSAGE: message},
                )
        except Exception as ex:
            self.logger.error(
                f"Error: unable to save progress to file "
                f"'{self.progress_file_path}': {ex}\n"
                f"{traceback.format_exc()}"
            )
        return progress


class ProgressCallbackStackingBridge(AbstractProgressCallbackContext):
    """Create callback for THE ONE STEP of the parent callback.

    Progress reporting callback bridge which allows to hierarchically stack report
    progress context from the child callback to parent callback
    in the range of 1 parent's step.

    Example: parent progress callback context tracks progress on PD feature
    granularity while child callback context tracks progress on bin granularity
    for particular feature i.e. feature currently calculated by parent - parent's
    step is feature; child's step is feature's bin.

    """

    def __init__(self, progress_callback: ProgressCallbackContext):
        AbstractProgressCallbackContext.__init__(self)

        self._sub_callback = progress_callback.get_sub_callback_for_progress(
            min_progress=progress_callback.progress,
            max_progress=progress_callback.progress
            + progress_callback.get_range_for_step(),
        )

    def set_progress(self, progress: float, message: str | None = None) -> float:
        return self._sub_callback.set_progress(progress)


class _AsyncProgressQueueThread(threading.Thread):
    def __init__(
        self,
        progress_queue: multiprocessing.Queue,
        progress_callback: AbstractProgressCallbackContext,
        job_id: str,
        worker_id: int,
        src_name: str = "",
    ):
        threading.Thread.__init__(self)
        self.progress_queue = progress_queue
        self.progress_callback = progress_callback
        self.job_id = job_id
        self.worker_id = worker_id
        # ASYNC processes CANNOT log using logger - Python logging deadlocks
        self.logger = loggers.SonarPrintLogger()
        self.src_name = src_name

    def run(self):
        while True:
            try:
                qmsg = self.progress_queue.get(timeout=0.3)

                # ASYNC processes CANNOT log using logger - Python logging deadlocks
                self.logger.debug(
                    f"Async progress queue to callback poller: received progress "
                    f"message from the {self.src_name} sub-process: {qmsg}"
                )

                if qmsg and isinstance(qmsg, tuple) and len(qmsg) == 4:
                    (job_id, worker_id, i_progress, message) = qmsg
                    if job_id == self.job_id and worker_id == self.worker_id:
                        self.progress_callback.set_progress(i_progress, message)
                        if i_progress >= 1.0:
                            break
            except queue.Empty:
                time.sleep(0.3)
                continue

        # ASYNC processes CANNOT log using loggers - Python logging deadlocks
        self.logger.debug(
            f"Async progress queue to callback poller finished for job_id: "
            f"{self.job_id} worker_id: {self.worker_id} name: {self.src_name}"
        )


class AsyncProgressQueueToCallbackPoller:
    """DEPRECATED poller of the progress queue which routes queue progress messages
    to the (sub-process, interpretation, ...) progress callback.

    """

    def __init__(
        self,
        progress_queue: multiprocessing.Queue,
        progress_callback: ProgressCallbackContext,
        job_id: str,
        worker_id: int = 0,
    ):
        self.progress_queue = progress_queue
        self.job_id = job_id
        self.worker_id = worker_id
        self.poller = None

        # run poller
        self.poller = _AsyncProgressQueueThread(
            progress_queue=self.progress_queue,
            progress_callback=progress_callback,
            job_id=self.job_id,
            worker_id=self.worker_id,
        )
        self.poller.start()

        raise DeprecationWarning(
            "AsyncProgressQueueToCallbackPoller is deprecated - do not use it,"
            "its thread is mixing threading and multiprocessing.Queue which leads"
            "to race conditions and deadlocks."
        )


class _AsyncProgressFileThread(threading.Thread):
    def __init__(
        self,
        progress_file: pathlib.Path,
        progress_callback: AbstractProgressCallbackContext,
        target_entity=None,
        src_name: str = "",
    ):
        """Constructor.

        Parameters
        ----------
        progress_file : pathlib.Path
            Path to the progress file.
        progress_callback : AbstractProgressCallbackContext
            Progress callback to which the progress messages will be routed.
        target_entity :
            Target entity whose progress is being tracked by this poller.
            Used to detect when the entity is finished via ``is_finished()`` method.
        src_name : str
            Name of the source of the progress messages.

        """
        # create DAEMON thread (no need to join / block the main thread from exiting)
        threading.Thread.__init__(self, daemon=True)

        self.progress_file = progress_file
        self.progress_callback = progress_callback
        self.terminator = target_entity
        # ASYNC processes CANNOT log using logger - Python logging deadlocks
        self.logger = loggers.SonarPrintLogger()
        self.src_name = src_name
        self.persistence = persistences.FilesystemPersistence()

        # activate waiting: exponential backoff
        self.wait_min = 0.1
        self.wait_backoff_coef = 2
        self.wait_max = 3.5

    def _do_wait(self, wait_limit: float) -> float:
        wait_limit = min(wait_limit * self.wait_backoff_coef, self.wait_max)
        # ASYNC processes CANNOT log using logger - Python logging deadlocks
        self.logger.debug(
            f"AsyncProgressFileThread poller ({id(self)}) will wait for progress "
            f"update:"
            f"\n  wait time    : {wait_limit}s"
            f"\n  progress file: {self.progress_file}"
        )
        time.sleep(wait_limit)
        return wait_limit

    def run(self):
        max_wait_time_seconds = 600  # 10 minutes safety timeout
        start_time = time.time()
        # call progress callback only if the progress has changed
        last_progress = -1.0
        # activate waiting: exponential backoff
        wait_limit = self.wait_min
        while True:  # will not loop indefinitely ~ emergency exit v
            try:
                # EMERGENCY EXIT
                elapsed_time = time.time() - start_time
                if elapsed_time > max_wait_time_seconds:
                    self.logger.warning(
                        f"AsyncProgressFileThread poller ({id(self)}): giving up after "
                        f"{elapsed_time:.1f}s - progress file '{self.progress_file}' "
                        f"NOT found. Progress reporting will be disabled, but "
                        f"the task will continue."
                    )
                    return

                # TERMINATOR: ensure clean exit even if progress file was NOT created
                if (
                    hasattr(self.terminator, "is_finished")
                    and self.terminator.is_finished()
                ):
                    self.logger.debug(
                        f"AsyncProgressFileThread poller ({id(self)}): terminating - "
                        f"target entity finished after {elapsed_time:.1f}s"
                    )
                    return

                if not self.progress_file.exists():
                    self.logger.debug(
                        f"AsyncProgressFileThread poller ({id(self)}): waiting for "
                        f"progress file (elapsed: {elapsed_time:.1f}s): "
                        f"{self.progress_file}"
                    )
                    wait_limit = self._do_wait(wait_limit)
                    continue

                # ATOMICALLY read the progress file
                progress_dict = self.persistence.load_json(self.progress_file)
                # ASYNC processes CANNOT log using logger - Python logging deadlocks
                self.logger.debug(
                    f"Async progress file to callback poller: received progress "
                    f"message from {self.progress_file} task: {progress_dict}"
                )

                if (
                    progress_dict
                    and isinstance(progress_dict, dict)
                    and KEY_PROGRESS in progress_dict
                ) and KEY_MESSAGE in progress_dict:
                    progress = progress_dict[KEY_PROGRESS]
                    if last_progress != progress:
                        message = progress_dict[KEY_MESSAGE]
                        self.progress_callback.set_progress(progress, message)
                        if progress >= 1.0:
                            # stop polling if the interpretation is finished
                            return
                        wait_limit = self.wait_min

                wait_limit = self._do_wait(wait_limit)
            except Exception as ex:
                # ASYNC processes CANNOT log using logger - Python logging deadlocks
                self.logger.error(
                    f"ERROR: Async progress file to callback poller: failed to read "
                    f"progress file {self.progress_file}: {ex}\n"
                    f"{traceback.format_exc()}"
                )
                # IMPORTANT: sleep after ex to prevent tight error loop ~ CPU thrash
                time.sleep(wait_limit)


class AsyncProgressFileToCallbackPoller:
    """Poller of the interpretation/evaluation progress file (JSon) which
    periodically reads the file and routes progress  messages to the given
    progress callback.

    """

    def __init__(
        self,
        progress_file: str | pathlib.Path,
        progress_callback: ProgressCallbackContext,
        target_entity=None,
    ):
        """Constructor.

        Parameters
        ----------
        progress_file : str | pathlib.Path
            Path to the progress file.
        progress_callback : ProgressCallbackContext
            Progress callback to which the progress messages will be routed.
        target_entity :
            Target entity (Interpretation/Evaluation) used to detect completion
            via ``is_finished()`` method.

        """
        # ASYNC processes CANNOT log using logger - Python logging deadlocks
        self.logger = loggers.SonarPrintLogger()
        self.poller = None
        self.progress_file = pathlib.Path(progress_file)

        if not self.progress_file:
            raise FileNotFoundError(
                f"Unable to poll/bridge progress - progress file "
                f"'{self.progress_file}' path is empty"
            )

        # run poller
        self.poller = _AsyncProgressFileThread(
            progress_file=self.progress_file,
            progress_callback=progress_callback,
            target_entity=target_entity,
        )
        self.poller.start()


#
# progress reporting UTILITIES
#


class LoggingProgressCallbackContext(ProgressCallbackContext):
    """Progress reporting callback context which logs progress using a given logger."""

    def __init__(
        self,
        logger: loggers.SonarLogger | None = None,
        prefix: str = "Progress",
        min_progress: float = 0.0,
        max_progress: float = 1.0,
        progress: float = 0.0,
        message: str | None = None,
        total_steps: int = 0,
        do_update: list | None = None,
        parent_callback: AbstractProgressCallbackContext | None = None,
        verbose_children: bool = True,
        name: str = "",
    ):
        """Constructor.

        Parameters
        ----------
        logger : loggers.SonarLogger | None
            Logger to be used for progress reporting. If not provided,
            then ``loggers.SonarPrintLogger`` is used.
        prefix : str
            Prefix to be used for the progress message logged.
        min_progress : float
            Minimum progress value.
        max_progress : float
            Maximum progress value.
        verbose_children : bool
            Optionally mute logging to avoid duplicate messages in case of stacking.

        """
        ProgressCallbackContext.__init__(
            self,
            min_progress=min_progress,
            max_progress=max_progress,
            progress=progress,
            message=message,
            total_steps=total_steps,
            do_update=do_update,
            verbose_children=verbose_children,
            parent_callback=parent_callback,
            name=name,
        )

        self.prefix = prefix
        self.logger = logger or loggers.SonarPrintLogger()

    def set_progress(self, progress: float, message: str | None = None) -> float:
        progress = ProgressCallbackContext.set_progress(self, progress, message)
        message = f"{self.prefix}: {self.progress_percent}% - {message or ''}"

        if self.verbose_children:
            self.logger.info(message)

        return progress


class MethodToExplainerProgressReporter(AbstractProgressCallbackContext):
    """Progress reporting callback context to propagate progress
    from methods to explainer runtime.

    """

    def __init__(self, min_progress: float, max_progress: float, callback: Callable):
        AbstractProgressCallbackContext.__init__(self)
        self._min_progress = min_progress
        self._max_progress = max_progress
        self._range = self._max_progress - self._min_progress
        self._callback = callback

    def set_progress(self, progress: float, message: str | None = None) -> float:
        # recalculate progress to range
        relative_progress = self._min_progress + self._range * progress
        relative_progress = (
            self._min_progress
            if relative_progress < self._min_progress
            else relative_progress
        )
        relative_progress = (
            self._max_progress
            if relative_progress > self._max_progress
            else relative_progress
        )
        # report progress
        self._callback(progress=relative_progress, message=message)
        return relative_progress
