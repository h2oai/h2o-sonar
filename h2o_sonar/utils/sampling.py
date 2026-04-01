# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
"""This module provides the following dataset sampling techniques:

- ``StratifiedDatasetSampling``: (default)
  Dataset sampler which implements both stratified and random sampling. The sampler
  automatically decided which sampling technique to use.

  - CONS:
     - stratified sampling can sample datasets up to 50% of the free RAM
       (sklearn sampler is the bottleneck)
  - PROS:
     - supports stratified (classification models) and random sampling (regression)
     - makes automatic decision of the sampling method (can be overriden w/ parameter)
     - random sampling is able to sample dataset bigger than the free RAM size

- ``NoDatasetSampling``:
  Sampler which is used when the user requests NO sampling. In order to avoid
  OOM/H2O Sonar crash it checks whether the datasets fits in RAM and if it doesn't
  then it raises an exception with a request to sample/use a different dataset.

- ``RandomPandasDatasetSampling``:
  Dataset sampler which implements random sampling using Pandas.

  - CONS:
     - dataset must fit in free RAM (2x)
     - sampler does not support the stratification
  - PROS:
     - enables the use of Pandas sampler seamlessly in the H2O Sonar runtime

- ``HeadOfDatasetSampling``:
  Sampler which does **not** sample, but returns ``sampling_limit`` number of rows
  from the head of the dataset.

  - CONS:
     - sampled dataset will be very likely biased (should not be used in production)
  - PROS:
     - fast
     - handles dataset of any size
     - can be used for splitting and non-functional testing

"""

import abc
import pathlib
import traceback

import datatable
import numpy
import pandas

from h2o_sonar import errors
from h2o_sonar import loggers
from h2o_sonar.lib.api import commons


try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from sklearn import model_selection
    from sklearn import preprocessing

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class DatasetSampler(abc.ABC):
    """The sampler children implementations various dataset sampling techniques.

    H2O Sonar container samples the dataset upfront (based on the interpretation
    parameters) in order to protect the process/runtime (from the crash), the system
    (from OOO and extensive used of resources) and explainers from failures.

    """

    # sampling limit used REGARDLESS users/callers requests to protect the OS/system
    SYSTEM_LIMIT = 1_000_000_000
    # H2O Sonar sampling limit meaningful for running explainers on top of such dataset
    H2O_SONAR_LIMIT = 25_000

    DEFAULT_CAT_NUM_THRESHOLD = 50

    @staticmethod
    def is_dataset_fit_in_memory(dataset_path: str | pathlib.Path):
        """Check whether the dataset file would fit to free RAM and return sizes.

        Parameters
        ----------
        dataset_path : str
          Dataset path.

        Returns
        -------
        Tuple[bool, int, int] :
          Return whether the dataset will fit, dataset size in bytes and RAM size in
          bytes.

        """
        if not HAS_PSUTIL:
            commons.raise_opt_import_err("psutil")

        dataset_path = (
            pathlib.Path(dataset_path)
            if isinstance(dataset_path, str)
            else dataset_path
        )

        if not dataset_path.exists() or not dataset_path.is_file():
            raise ValueError(f"Dataset path is not valid: {dataset_path}")

        ram_free = psutil.virtual_memory().available
        dataset_size = dataset_path.stat().st_size
        if ram_free > dataset_size:
            return True, dataset_size, ram_free
        return False, dataset_size, ram_free

    def __init__(self, system_limit: int = SYSTEM_LIMIT):
        """Constructor.

        Parameters
        ----------
        system_limit : int
          Sampling limit which is used as theoretical maximum REGARDLESS users/callers
          limit requests to protect the OS/interpreter/system.

        """
        self.system_limit = system_limit or DatasetSampler.SYSTEM_LIMIT

    def sample_dataset(
        self,
        dataset: datatable.Frame | str | pathlib.Path,
        sampling_limit: int | None = 0,
        target_col: str = "",
        is_classification: bool = False,
        drop_nan_rows: bool = True,
        drop_1_classes: bool = True,
        classes: list | None = None,
        sampled_dataset_path: datatable.Frame | str | pathlib.Path = "",
        seed: int = 42,  # None to randomize
        logger=None,
    ) -> tuple[bool, datatable.Frame, str]:
        """Sample dataset.

        Parameters
        ----------
        dataset: datatable.Frame | str | pathlib.Path :
          Dataset to be sampled as reference to the frame or a path to the file.
        sampling_limit : int | None = None,
          If ``None``, then automatically sample based on the dataset and RAM size.
          If > 0, then do sample the ``dataset`` to ``sampling_limit`` number of rows.
          If == 0, then do NOT sample.
        target_col : str = "",
          Optional target colum which is required for certain sampling techniques
          (like for stratified sampling).
        is_classification : bool
          If ``None``, then automatically choose stratified or random sampling.
          If ``True``, then force stratified sampling.
          If ``False``, then force random sampling.
        drop_nan_rows : bool
          ``True`` to drop rows with "not a number" value in the ``target_col``
          column in case of classification-friendly techniques.
        drop_1_classes : bool
          ``True`` to drop rows which represent classes with cardinality equal to 1
          (categories which are represented by exactly one row in the dataset)
          in the ``target_col`` column in case of classification-friendly techniques.
        classes : list | None = None
          Optional specification of classes to be used for sampling (all valid classes
          will be used by default). ``classes`` values are expected to be a subset of
          the target column classes.
        sampled_dataset_path : datatable.Frame | str | pathlib.Path
          Optional path to the sampled dataset file to be created (if no path is
          specified, then the method returns the reference to datatable frame).
        seed : int
          Optional random seed for reproducible sampling.
        logger :
          Optional logger.

        Returns
        -------
        datatable.Frame | str :
          Path to the sampled dataset (if the path to ``sampled_dataset_path`` has
          been specified), datatable Frame reference otherwise.

        """
        raise NotImplementedError

    @staticmethod
    def _save_frame(
        frame: datatable.Frame,
        sampled_dataset_path: datatable.Frame | str | pathlib.Path,
    ):
        if sampled_dataset_path:
            frame.to_csv(sampled_dataset_path)
            return sampled_dataset_path
        return frame


class NoDatasetSampling(DatasetSampler):
    """Sampler which does **NO sampling** and can check whether the dataset would
    fit in RAM and thus avoid H2O Sonar OOM crash. Used as default sampling method.

    """

    def __init__(self, check_ram: bool = True):
        """Constructor.

        Parameters
        ----------
        check_ram : bool
          ``True`` to check whether the dataset would fit in memory (RAM), ``False``
          otherwise.

        """
        DatasetSampler.__init__(self)

        self.check_ram = check_ram

    def sample_dataset(
        self,
        dataset: datatable.Frame | str | pathlib.Path,
        sampling_limit: int | None = None,
        target_col: str = "",
        is_classification: bool = False,
        drop_nan_rows: bool = True,
        drop_1_classes: bool = True,
        classes: list | None = None,
        sampled_dataset_path: datatable.Frame | str | pathlib.Path = "",
        seed: int = 42,  # None to randomize
        logger=None,
    ) -> tuple[bool, datatable.Frame, str]:
        del sampled_dataset_path
        del sampling_limit
        del target_col
        del is_classification
        del drop_nan_rows
        del drop_1_classes
        del classes
        del seed
        del logger

        if isinstance(dataset, (str, pathlib.Path)):
            if self.check_ram:
                (is_safe, d_size, ram_size) = DatasetSampler.is_dataset_fit_in_memory(
                    dataset
                )
                if not is_safe:
                    raise errors.DatasetTooBigError(
                        message=(
                            f"The dataset to be used WITHOUT SAMPLING is too big to "
                            f"fit in memory (RAM) (dataset size: {d_size}B, "
                            f"free RAM size: {ram_size}B) therefore it would lead "
                            f"to OOM error/process crash. Please sample the dataset, "
                            f"provide smaller dataset or configure H2O Sonar to sample "
                            f"the dataset for you."
                        ),
                        suggestion=(
                            "As you know the dataset, preferably sample the dataset "
                            "yourself to preserve its important characteristics. "
                            "Alternatively provide smaller dataset or configure "
                            "H2O Sonar to use a sampling method."
                        ),
                    )

            return (
                False,
                datatable.fread(
                    dataset, memory_limit=StratifiedDatasetSampling._dt_can_alloc_ram()
                ),
                str(dataset),
            )

        # datatable.Frame
        return False, dataset, ""


class RandomPandasDatasetSampling(DatasetSampler):
    """Dataset sampler which implements **random sampling** using Pandas
      ``pandas.DataFrame.sample()``.

    - CONS:

       - dataset must fit in free RAM (2x)
       - sampler does not support the stratification

    - PROS:

       - enables the use of Pandas sampler seamlessly in the H2O Sonar runtime

    """

    def __init__(self, logger=None):
        DatasetSampler.__init__(self)

        self.logger = logger or loggers.SonarPrintLogger()

    def sample_dataset(
        self,
        dataset: datatable.Frame | str | pathlib.Path,
        sampling_limit: int | None = None,
        target_col: str = "",
        is_classification: bool = False,
        drop_nan_rows: bool = True,
        drop_1_classes: bool = True,
        classes: list | None = None,
        sampled_dataset_path: datatable.Frame | str | pathlib.Path = "",
        seed: int = 42,  # None to randomize
        logger=None,
    ) -> tuple[bool, datatable.Frame, str]:
        del target_col
        del sampled_dataset_path
        del is_classification
        del classes

        logger = logger or self.logger

        dataset_path = None
        if isinstance(dataset, datatable.Frame):
            # OOM risk: frame 2x @ RAM
            pd_frame = dataset.to_pandas()
        else:
            dataset_path = dataset
            try:
                pd_frame = pandas.read_csv(dataset)
            except Exception as ex:
                logger.warning(
                    f"Pandas samplers is unable to load the dataset using Pandas ("
                    f"will fallback to datatable): {ex}\n{traceback.format_exc()}"
                )
                # fallback w/ OOM risk (if pd fails): frame 2x @ RAM
                pd_frame = datatable.fread(dataset).to_pandas()

        return (
            True,
            pd_frame.sample(
                n=sampling_limit,
                random_state=seed,
                replace=False,
            ),
            dataset_path,
        )


class HeadOfDatasetSampling(DatasetSampler):
    """Sampler which does **not** sample, but returns sampling limit number of
    examples from the head of the dataset.

    PRESUMPTIONS:

    - sampled dataset will fit into free RAM

    CONS:

    - it is **NOT** correct for the data science perspective and should **NOT** be
      used as it does not guarantee anything - the sampled dataset will very likely
      be biased i.e. may have completely different characteristics and statistics
      than the original dataset

    PROS:

    - it can sample dataset of **any** size, therefore enables H2O Sonar to run on
      the dataset of any size - in case that the data science aspect is not a problem,
      this sampler might be a good choice
    - it is relatively fast in comparison to other samplers
    - it is ideal for non-functional testing

    """

    def __init__(self, chunk_size: int = 1_000_000):
        DatasetSampler.__init__(self)

        # number of CSV file rows to read in one chunk
        self.chunk_size = chunk_size
        self.nope = NoDatasetSampling()

    def sample_dataset(
        self,
        dataset: datatable.Frame | str | pathlib.Path,
        sampling_limit: int | None = None,
        target_col: str = "",
        is_classification: bool = False,
        drop_nan_rows: bool = True,
        drop_1_classes: bool = True,
        classes: list | None = None,
        sampled_dataset_path: datatable.Frame | str | pathlib.Path = "",
        seed: int = 42,  # None to randomize
        logger=None,
    ) -> tuple[bool, datatable.Frame, str]:
        del target_col
        del is_classification
        del classes
        del seed

        if not sampled_dataset_path:
            raise ValueError(
                "Pandas head sampler requires a CVS file path where to store "
                "the sampled dataset"
            )

        if not sampling_limit:
            return self.nope.sample_dataset(
                dataset=dataset,
                sampled_dataset_path=sampled_dataset_path,
                logger=logger,
            )

        if isinstance(dataset, datatable.Frame):
            sampled = dataset.shape[0] > sampling_limit
            sampled_frame = dataset[:sampling_limit, :]
            return (
                sampled,
                sampled_frame,
                DatasetSampler._save_frame(
                    frame=sampled_frame, sampled_dataset_path=sampled_dataset_path
                ),
            )

        # head
        remains = sampling_limit
        chunk_size = (
            sampling_limit if sampling_limit < self.chunk_size else self.chunk_size
        )
        # IMPORTANT: it is expected that SAMPLED dataset WILL FIT 2x INTO RAM
        chunks = []
        for chunk in pandas.read_csv(dataset, chunksize=chunk_size):
            print(f"Chunk: {type(chunk)} {chunk.shape}")
            if remains > chunk.shape[0]:
                remains -= chunk.shape[0]
                chunks.append(chunk)
            else:
                chunks.append(chunk.head(n=remains))
                break

        pd_frame = pandas.concat(chunks)
        del chunks
        # safe the frame anyway to avoid having it in the memory even more times
        pd_frame.to_csv(sampled_dataset_path)

        return True, False, sampled_dataset_path


class StratifiedDatasetSampling(DatasetSampler):
    """Dataset sampler which implements both stratified and  random sampling.

    - CONS:

       - stratified sampling can sample datasets up to 50% of the free RAM
         (sklearn sampler is the bottleneck)

    - PROS:

       - supports stratified (classification models) and random sampling (regression)
       - makes automatic decision of the sampling method (can be overriden w/ parameter)
       - random sampling is able to sample dataset bigger than the free RAM size

    """

    def __init__(self):
        DatasetSampler.__init__(self)

    @staticmethod
    def _dt_can_alloc_ram():
        """Determine RAM which can be used by datatable as buffer to read the data."""
        if not HAS_PSUTIL:
            commons.raise_opt_import_err("psutil")

        ram_free = psutil.virtual_memory().available
        return int(ram_free * 0.33)

    def _sample_regression(
        self,
        dataset: datatable.Frame | str | pathlib.Path,
        sampling_limit: int | None,
        sampled_dataset_path: datatable.Frame | str | pathlib.Path,
        seed: int,
        logger,
    ):
        if not HAS_SKLEARN:
            commons.raise_opt_import_err("scikit-learn")

        log_name = "Random sampler:"

        # STEP) create index column to be used for orig frame filtering after sampling
        num_rows = dataset.shape[0]
        np_frame = numpy.zeros(num_rows, dtype=[("row", ">i4")])  # foo 0s frame
        np_frame["row"] = numpy.arange(num_rows)
        logger.debug(f"{log_name} numpy frame: {np_frame.shape}")

        # STEP) RANDOM sampling using sklearn
        logger.debug(f"{log_name} Sampling...")
        (np_row_selector, _) = model_selection.train_test_split(
            np_frame,  # index data ~ column to track which rows to keep/drop
            train_size=sampling_limit,
            shuffle=True,  # shuffle to get the random
            random_state=seed,
        )
        logger.info(f"{log_name} original row selector shape: {np_frame.shape}")
        logger.info(f"{log_name} sampled row selector shape: {np_row_selector.shape}")

        # 5) filter the original frame now
        logger.debug(f"{log_name} filtering original frame using sampled row selector")
        keep_rows = [r[0] for r in np_row_selector[:,].tolist()]
        np_row_selector = None  # GC
        sampled_frame = dataset[keep_rows, :]
        logger.info(f"{log_name} sampled frame shape: {sampled_frame.shape}")
        self._save_frame(frame=sampled_frame, sampled_dataset_path=sampled_dataset_path)

        logger.debug(f"{log_name} DONE")
        return (
            True,
            datatable.fread(
                sampled_dataset_path,
                memory_limit=StratifiedDatasetSampling._dt_can_alloc_ram(),
            ),
            sampled_dataset_path,
        )

    def _sample_classification(
        self,
        dataset: datatable.Frame,
        sampling_limit: int | None,
        target_col: str,
        classes: list | None,
        sampled_dataset_path: datatable.Frame | str | pathlib.Path,
        seed: int,
        logger,
    ):
        """Method:

        1. PREPARE the dataset target column based on sklearn requirements:
           a) If target column is not numeric, then label encode it (including NaNs)
           b) replace NaNs with ``max(target column) + 1`` (if were not label encoded)
           c) remove (backup) rows which represent categories with cardinality == 1
              (keep row numbers, rows will be inserted back later - configurable action)
        2. TARGET COLUMN frame as ``np_target``
        3. FRAME w/ INDEX COLUMN as ``np_frame``
        4. STRATIFIED SAMPLING of ``np_frame`` using ``sklearn.train_test_split()``
           which returns ``np_frame_sampled``
        5. SAMPLE ROWS LIST preparation:
           a) ``np_frame_sampled[INDEX COLUMN]`` are row numbers to keep
           b) add back rows which represent categories w/ cardinality == 1
        5. FILTER original frame ~ keep rows which are in ``np_frame_sampled``

        """
        if not HAS_SKLEARN:
            commons.raise_opt_import_err("scikit-learn")

        log_name = "Stratified sampler:"

        # keep target column only
        frame_y = dataset[:, target_col]
        target_ltype = frame_y.ltypes[frame_y.colindex(target_col)]

        # 1.b) drop rows with NaN value @ the target column (will NOT be added back)
        del frame_y[datatable.f[target_col] == None, :]  # noqa: E711
        # IMPROVE: keep rows, set valid value, sample, set back NaNs instead of value
        # target_frame = numpy.nan_to_num(
        #    np_y,
        #    # use number which is NOT among unique values & remember it to set it back
        #    nan=12345,
        # )
        logger.debug(f"{log_name} frame after dropping NaN rows: {frame_y.shape}")

        # 1.c) drop rows representing classes w/ cardinality == 1
        logger.debug(f"{log_name} classes: {classes}")
        classes_card_1 = []
        for clazz in classes:
            logger.debug(f"{log_name} checking cardinality of the class '{clazz}'")
            count = frame_y[datatable.f[target_col] == clazz, :].shape[0]
            logger.debug(f"  |'{clazz}'| == {count}")
            if count == 1:
                classes_card_1.append(clazz)
        if classes_card_1:
            logger.debug(
                f"{log_name} dropping classes w/ cardinality == 1: {classes_card_1}"
            )
            for clazz in classes_card_1:
                logger.debug(f"{log_name} dropping row for class '{clazz}'")
                del frame_y[datatable.f[target_col] == clazz, :]
            # check that classes were drop
            # classes_card_1 = []
            # for c in classes:
            #    count = frame[datatable.f[target_col] == c, :].shape[0]
            #    print(f"c: {c} count: {count}")
            #    if count == 1:
            #        print("  TO BE REMOVED")
            #        classes_card_1.append(c)
            logger.debug(
                f"{log_name} frame after dropping |class| == 1 rows: {frame_y.shape}"
            )
        classes_card_1 = []  # GC
        logger.debug(f"{log_name} classes: frame fixed to have valid classes only")

        # 2. target column to numpy
        np_y = frame_y.to_numpy()
        frame_y = None  # GC
        logger.debug(f"{log_name} numpy y: {np_y.shape}")

        # 1.a) label encode if target column is not numeric
        if target_ltype not in [datatable.ltype.int, datatable.ltype.real]:
            logger.debug(
                f"{log_name} will label encode the target column '{target_col}' of "
                f"type {target_ltype} as integer"
            )
            le = preprocessing.LabelEncoder()
            np_y = le.fit_transform(np_y)
            # logger.debug(
            #    f"{log_name} frame after label encoding ({np_y.shape}):\n{np_y}"
            # )
            le = None  # GC

        # 3. create index column to be used for orig frame filtering after sampling
        num_rows = np_y.shape[0]
        # TODO get rid of 0s column
        np_frame = numpy.zeros(num_rows, dtype=[("row", ">i4")])  # foo 0s frame
        np_frame["row"] = numpy.arange(num_rows)
        logger.debug(f"{log_name} numpy frame: {np_frame.shape}")

        # 4. STRATIFIED sampling using sklearn
        if not HAS_SKLEARN:
            commons.raise_opt_import_err("scikit-learn")

        logger.debug("Sampling...")
        # sklearn sampler: 1x CPU, high RAM load (dataset must fit to 50% free RAM)
        # (see also StratifiedKFold, StratifiedShuffleSplit)
        (np_row_selector, _) = model_selection.train_test_split(
            np_frame,  # index data ~ column to track which rows to keep/drop
            train_size=sampling_limit,
            # test_size=sampling_limit,
            shuffle=True,  # must shuffle @ stratify
            stratify=np_y,  # cleaned target column instance used for the stratification
            random_state=seed,
        )
        logger.info(f"{log_name} original row selector shape: {np_frame.shape}")
        logger.info(f"{log_name} sampled row selector shape: {np_row_selector.shape}")

        # 5) filter the original frame now
        logger.debug(f"{log_name} filtering original frame using the row selector")
        keep_rows = [r[0] for r in np_row_selector[:,].tolist()]
        np_row_selector = None  # GC
        sampled_frame = dataset[keep_rows, :]
        logger.info(f"{log_name} sampled frame shape: {sampled_frame.shape}")
        self._save_frame(frame=sampled_frame, sampled_dataset_path=sampled_dataset_path)

        logger.debug(f"{log_name} DONE")
        return (
            True,
            datatable.fread(
                sampled_dataset_path,
                memory_limit=StratifiedDatasetSampling._dt_can_alloc_ram(),
            ),
            sampled_dataset_path,
        )

    def sample_dataset(
        self,
        dataset: datatable.Frame | str | pathlib.Path,
        sampling_limit: int | None = None,
        target_col: str = "",
        is_classification: bool = False,
        drop_nan_rows: bool = True,
        drop_1_classes: bool = True,
        classes: list | None = None,
        sampled_dataset_path: datatable.Frame | str | pathlib.Path = "",
        seed: int = 42,  # None to randomize
        logger=None,
    ) -> tuple[bool, datatable.Frame, str]:
        logger = logger or loggers.SonarPrintLogger()

        if sampling_limit is None or sampling_limit < 0:
            sampling_limit = DatasetSampler.H2O_SONAR_LIMIT
        elif sampling_limit == 0:
            return NoDatasetSampling().sample_dataset(dataset)
        # else sampling limit specified by the caller

        if not drop_nan_rows:
            raise NotImplementedError(
                "Stratified sampling does not support keeping of 'not a number' rows "
                "in the sampled dataset"
            )
        if not drop_1_classes:
            raise NotImplementedError(
                "Stratified sampling does not support keeping of dataset rows "
                "representing classes with cardinality equal to 1"
            )
        if classes:
            raise NotImplementedError(
                f"Stratified sampling does not support selection of classes to be used "
                f"in the sampled dataset ('classes' parameter which was set to: "
                f"{classes})"
            )
        # future checks
        if classes and not is_classification:
            is_classification = True
        if classes and not target_col:
            raise ValueError(
                f"Target column must be specified when the list of classes to be used "
                f"was set: {classes}"
            )

        log_name = "Stratified/random sampler:"

        logger.info(
            f"{log_name} loading the original dataset '{dataset}' for sampling..."
        )
        raw_frame = (
            dataset
            if isinstance(dataset, datatable.Frame)
            else datatable.fread(
                dataset,
                # read dataset of any size & prevent OOM
                memory_limit=StratifiedDatasetSampling._dt_can_alloc_ram(),
            )
        )
        # CHECK: is the sampling needed
        if raw_frame.shape[0] <= sampling_limit:
            logger.info(
                f"{log_name}   -> did NO sampling as the sampling limit is smaller "
                f"than the number of rows in the dataset: "
                f"{raw_frame.shape[0]} <= {sampling_limit}"
            )
            # NO need to save the frame if it is not sampled, original can be used
            return (
                False,
                raw_frame,
                (dataset if isinstance(dataset, (str, pathlib.Path)) else None),
            )

        logger.info(
            f"{log_name} will sample the dataset {raw_frame.shape} to sampling limit "
            f"{sampling_limit}"
        )

        # sampling: STRATIFIED or RANDOM?
        if is_classification is None or is_classification:
            logger.debug(f"{log_name} finding classes of the target column...")
            classes = datatable.unique(raw_frame[:, target_col]).to_list()[0]
            logger.debug(f"{log_name}   found {len(classes)} classes")
            if is_classification is None:
                is_classification = (
                    False
                    if len(classes) > DatasetSampler.DEFAULT_CAT_NUM_THRESHOLD
                    else True
                )

        if is_classification:
            return self._sample_classification(
                dataset=raw_frame,
                sampling_limit=sampling_limit,
                target_col=target_col,
                classes=classes,
                sampled_dataset_path=sampled_dataset_path,
                seed=seed,
                logger=logger,
            )

        return self._sample_regression(
            dataset=raw_frame,
            sampling_limit=sampling_limit,
            sampled_dataset_path=sampled_dataset_path,
            seed=seed,
            logger=logger,
        )


def downsample_dataset(
    dataset,
    sample_size: int | None = None,
    runtime_sample_size: int | None = None,
    target_col: str = "",
    is_classification: bool = False,
    classes: list | None = None,
    seed: int = 42,
    logger=None,
):
    """Dataset sampling method used by the explainers in Driverless AI (and
    potentially other container runtimes) to sample the input dataset according to
    their needs.

    This method is not used by the local explainer as it samples the input dataset
    upfront to **protect** all the explainers. This is why this method serves as
    identity - it ensures that H2O Sonar's sampling will not impact Driverless AI and
    other host runtimes.

    Parameters
    ----------
    dataset :
      Dataset to be sampled.
    sample_size : int | None
      Sampling limit to use.
    runtime_sample_size : int | None
      Runtime protection - sample dataset to this size even if ``sample_size`` is
      bigger to protect the runtime and avoid space (memory) / time overloading.
    target_col : str
      Target column to be used for the sampling.
    is_classification : bool
      Sample for regression (``False``) or classification (``True``).
    classes : list | None
      List of classes in case of sampling of classification model dataset.
    seed : int
      Sampling seed.
    logger :
      Logger.
    Returns
    -------
    Any
      Sample dataset.
    """
    del sample_size
    del runtime_sample_size
    del target_col
    del is_classification
    del classes
    del seed
    del logger

    return dataset
