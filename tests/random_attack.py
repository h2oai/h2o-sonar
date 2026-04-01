#!/usr/bin/env python
# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
#
# Random attack example
#
#   make random_attack
#     DATASETS_DIR=/home/user/h/mli/git/h2o-sonar/data
#
#   make random_attack
#     DATASETS_DIR=/home/usr/h/datasets
#     MODELS_INDEX=/home/usr/h2o-sonar/data/predictive/models/models-index.json
#
import os
import pathlib
import random
import shutil
import sys
import traceback
import uuid
from dataclasses import dataclass

import datatable
import pandas
from sklearn import ensemble

from h2o_sonar import interpret
from h2o_sonar import loggers
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import datasets
from h2o_sonar.lib.api import persistences
from h2o_sonar.utils import preprocessing


# options (overridable by CLI arguments and shell environment variables)
OPT_MAX_ATTACKS = 1_000_000  # 0 means indefinite number of attacks
OPT_MIN_DISK_SPACE = 5  # GB
OPT_MAX_DATASET_SIZE = 100_001  # maximum allowed dataset size in B, skip if larger
OPT_SAMPLE_DATASETS_TO = 100_000  # sample datasets to this size, 0 means no sampling
OPT_CAT_NUM_THRESHOLD = 50

# arguments
ARG_MODELS_INDEX = "--models-index="
ARG_DATASETS_DIR = "--datasets-dir="

# shell environment variables
VAR_MAX_ATTACKS = "OPT_ATTACK_MAX_ATTACKS"
VAR_MIN_DISK_SPACE = "OPT_ATTACK_MIN_DISK_SPACE"
VAR_MAX_DATASET_SIZE = "OPT_ATTACK_MAX_DATASET_SIZE"
VAR_CAT_NUM_THRESHOLD = "OPT_ATTACK_CAT_NUM_THRESHOLD"
VAR_SAMPLE_DATASETS_TO = "OPT_SAMPLE_DATASETS_TO"


@dataclass
class DatasetIndexEntry:
    dataset_path: pathlib.Path

    regression_cols: list[str]
    binomial_cols: list[str]
    multinomial_cols: list[str]


@dataclass
class ModelIndexEntry:
    datasets_path: pathlib.Path
    model_path: pathlib.Path
    target_column: str


def ensure_disk_space(min_free_disk_space_gb=5):
    """Check that there is enough disk space (avoid 'No space left on device' error),
    if not, then raise the exception.

    """
    _, _, free = shutil.disk_usage("/")
    free_gib = free // (2**30)
    if free_gib < min_free_disk_space_gb:
        raise RuntimeError(
            f"Random attack will exit as there is less than "
            f"{min_free_disk_space_gb}GiB (configured limit) on the disk: {free}GiB"
        )


def getenv_int(var_name: str, default_value: int):
    try:
        return int(os.getenv(var_name, default_value))
    except Exception as ex:
        print(
            f"FALLBACK to default value while getting INT value of {var_name} due "
            f"to: {ex}"
        )
        return default_value


def get_rand_sequence(range_max: int, count: int) -> list[int]:
    """This function will return the list of ``count`` integers from the range
    [0, max_value-1]) which are random and do NOT repeat. If ``counter`` is greater
    than ``max_value`` then this function raises exception.

    """
    return random.sample(range(range_max), count)


class RandomAttackOptions:
    """Random attack options.

    Precedence from highest to lowest:

    * Python executable argument
    * shell environment variable
    * default

    """

    def __init__(
        self,
        max_attacks_limit: int = 0,
        min_free_disk_space: int = 0,
        max_dataset_size: int = 0,
        cat_num_threshold: int = 0,
        purge_ok_interpretations: bool = False,
        sample_datasets_to: int = 0,
    ):
        """Random attack options.

        Parameters
        ----------
        max_attacks_limit : int
          Maximum number of dataset attacks to run - as for every dataset there must
          be ``1`` - ``3`` attacks (regression, binomial and multinomial), H2O Sonar
          will be run ``max_attacks_limit*3`` times at most. ``0`` to run indefinitely.
        min_free_disk_space : int
          Minimum free disk space (GB) required by random attack to safely run.
        max_dataset_size : int
          Maximum dataset size (bytes) which can be used in random attack.
        cat_num_threshold :
          Columns with cardinality lower than this threshold will be considered
          as categorical, columns with higher than threshold cardinality will be
          considered numeric
        purge_ok_interpretations : bool
          Purge successful interpretations to preserve th disk space.
        sample_datasets_to : int
          Sample datasets to this size (bytes), 0 means no sampling.

        """
        self.max_attacks_limit = max_attacks_limit or getenv_int(
            VAR_MAX_ATTACKS, OPT_MAX_ATTACKS
        )
        self.min_free_disk_space = min_free_disk_space or getenv_int(
            VAR_MIN_DISK_SPACE,
            OPT_MIN_DISK_SPACE,
        )
        self.max_dataset_size = max_dataset_size or getenv_int(
            VAR_MAX_DATASET_SIZE,
            OPT_MAX_DATASET_SIZE,
        )
        self.cat_num_threshold = cat_num_threshold or getenv_int(
            VAR_CAT_NUM_THRESHOLD,
            OPT_CAT_NUM_THRESHOLD,
        )
        self.purge_ok_interpretations = purge_ok_interpretations
        self.sample_dataset_to = sample_datasets_to or getenv_int(
            VAR_SAMPLE_DATASETS_TO,
            OPT_SAMPLE_DATASETS_TO,
        )


class ModelsIndex:
    def __init__(self, logger):
        self.models: list[ModelIndexEntry] = []
        self.logger = logger

    def add_model(self, model: ModelIndexEntry):
        self.models.append(model)

    def add_model_idx_file(self, idx_path: pathlib.Path):
        self.models.extend(self.load(idx_path))

    @staticmethod
    def validate_model(model: ModelIndexEntry):
        if not model:
            raise ValueError("Model cannot be validated as it is None")
        if not model.datasets_path or not model.datasets_path.exists():
            raise ValueError(f"Dataset path does not exist for model {model}")
        if not model.model_path or not model.model_path.exists():
            raise ValueError(f"Model path does not exist for model {model}")
        if not model.target_column:
            raise ValueError(f"Target column is not specified for model {model}")

    def load(self, idx_path: pathlib.Path | None) -> list[ModelIndexEntry]:
        if not idx_path.exists() or not idx_path.is_file():
            raise ValueError(f"Models index file '{idx_path}' does not exist")

        models = []
        idx_dict = persistences.FilesystemPersistence().load_json(str(idx_path))
        for i in idx_dict:
            try:
                dataset_path = i.get("dataset_path", "")
                model_path = i.get("model_path", "")
                model = ModelIndexEntry(
                    datasets_path=pathlib.Path(dataset_path) if dataset_path else None,
                    model_path=pathlib.Path(model_path) if model_path else None,
                    target_column=i.get("target_column", ""),
                )
                ModelsIndex.validate_model(model)
                self.logger.info(f"Indexing model:\n{model}")
                models.append(model)
            except Exception as ex:
                self.logger.warning(f"Unable to index model: '{i}': {ex}")

        return models


class ModelTrainer:
    def __init__(self, enc_datasets_path: pathlib.Path, logger):
        self.enc_datasets_path = enc_datasets_path
        self.logger = logger

    def train_sklearn_gbm(self, dataset: DatasetIndexEntry, target_col: str):
        self.logger.info(
            f"Training sklearn/GradientBoostingClassifier for "
            f"the dataset {dataset} and "
            f"target column '{target_col}'"
        )

        # PREPARE dataset for sklearn/GBM: columns MUST be numeric
        dataset_pd = pandas.read_csv(dataset.dataset_path)
        (leX, mcle, categorical_variables) = preprocessing.categorical_encoder(
            dataset_pd
        )
        if categorical_variables:
            self.logger.info(
                f"The dataset has STRING columns {categorical_variables}, "
                f"therefore its label encoded variant will be used for the model "
                f"training and H2O Sonar interpretation"
            )
            dataset_path = self.enc_datasets_path / f"dataset-{uuid.uuid4()}.csv"
            # IMPORTANT: saving Pandas frame WITHOUT index as for dt it would be column
            leX.to_csv(dataset_path, index=False)
        else:
            dataset_path = dataset.dataset_path
        # SPLIT to train dataset and target column
        (X, y) = dataset_pd.drop(target_col, axis=1), dataset_pd[target_col]

        # MODEL training
        model = ensemble.GradientBoostingClassifier(learning_rate=0.1)
        model.fit(X, y)

        # inject feature names (for debugging purposes)
        model.feature_names = model.feature_names_in_  # GBM provides feature names

        # SUMMARY
        dataset_cols = list(datatable.fread(dataset_path).names)
        dataset_cols.sort()
        model_cols = list(model.feature_names)
        model_cols.sort()
        self.logger.info(
            f"TRAINED new sklearn/GradientBoostingClassifier model:\n"
            f"> target column   : {target_col}\n"
            f"> data columns  ({len(dataset_cols)}): {dataset_cols}\n"
            f"> model features({len(model_cols)}): {model_cols}"
        )
        assert len(dataset_cols) == len(model_cols) + 1, (
            f"Dataset columns and model features do NOT match:\n"
            f"> target column      : {target_col}\n"
            f"> dataset columns ({len(dataset_cols)}): {dataset_cols}\n"
            f"> model features ({len(model_cols)}): {model_cols}"
        )

        return model, dataset_path


class RandAttackReport:
    FORMAT_HTML = commons.MimeType.MIME_HTML
    FORMAT_JSON = commons.MimeType.MIME_JSON
    FORMAT_MARKDOWN = commons.MimeType.MIME_MARKDOWN

    KEY_I_PATH = "interpretation_path"
    KEY_I_KEY = "interpretations_key"
    KEY_FAILED_ES = "failed_explainers"
    KEY_FAILED_IS = "failed_interpretations"
    KEY_FAILED_MS = "failed_models"
    KEY_FAILED_DS = "failed_datasets"
    KEY_DATASET_PATH = "dataset_path"
    KEY_MODEL_PATH = "model_path"
    KEY_MODEL_TYPE = "model_type"
    KEY_TARGET_COL = "target_col"
    KEY_ERR_MSG = "error_message"

    def __init__(self):
        self.report: dict = {
            RandAttackReport.KEY_FAILED_ES: [],
            RandAttackReport.KEY_FAILED_IS: [],
            RandAttackReport.KEY_FAILED_MS: [],
            RandAttackReport.KEY_FAILED_DS: [],
        }
        self.persistence = persistences.FilesystemPersistence()

    def add_failed_explainers(
        self,
        interpretation_path,
        key,
        dataset_path,
        model_path,
        model_type: str,
        target_col: str,
        error_msg: str,
    ):
        self.report[RandAttackReport.KEY_FAILED_ES].append(
            {
                RandAttackReport.KEY_I_PATH: str(interpretation_path),
                RandAttackReport.KEY_I_KEY: str(key),
                RandAttackReport.KEY_MODEL_TYPE: str(model_type),
                RandAttackReport.KEY_MODEL_PATH: str(model_path),
                RandAttackReport.KEY_DATASET_PATH: str(dataset_path),
                RandAttackReport.KEY_TARGET_COL: target_col,
                RandAttackReport.KEY_ERR_MSG: error_msg,
            }
        )

    def add_failed_interpretation(
        self,
        interpretation_path,
        key,
        dataset_path,
        model_path,
        model_type: str,
        target_col: str,
        error_msg: str,
    ):
        self.report[RandAttackReport.KEY_FAILED_IS].append(
            {
                RandAttackReport.KEY_I_PATH: str(interpretation_path),
                RandAttackReport.KEY_I_KEY: str(key),
                RandAttackReport.KEY_MODEL_TYPE: str(model_type),
                RandAttackReport.KEY_MODEL_PATH: str(model_path),
                RandAttackReport.KEY_DATASET_PATH: str(dataset_path),
                RandAttackReport.KEY_TARGET_COL: target_col,
                RandAttackReport.KEY_ERR_MSG: error_msg,
            }
        )

    def add_failed_model(self, dataset_path, target_col: str, error_msg: str):
        self.report[RandAttackReport.KEY_FAILED_MS].append(
            {
                RandAttackReport.KEY_DATASET_PATH: str(dataset_path),
                RandAttackReport.KEY_TARGET_COL: target_col,
                RandAttackReport.KEY_ERR_MSG: error_msg,
            }
        )

    def add_failed_dataset(self, dataset_path, error_msg: str):
        self.report[RandAttackReport.KEY_FAILED_DS].append(
            {
                RandAttackReport.KEY_DATASET_PATH: str(dataset_path),
                RandAttackReport.KEY_ERR_MSG: error_msg,
            }
        )

    def as_dict(self) -> dict:
        return self.report

    def as_markdown(self) -> str:
        """Save Markdown document which can be used to easily create GitHub issues,
        it is the text to be used as GitHub issue content.

        """
        raise NotImplementedError

    def as_html(self) -> str:
        # TODO airium
        return str(self.as_dict())

    def save(self, path: pathlib.Path, report_format=FORMAT_JSON):
        if RandAttackReport.FORMAT_HTML == report_format:
            self.persistence.save(
                key=path,
                data=self.as_html(),
                data_type=persistences.PersistenceDataType.text,
            )
        else:
            self.persistence.save_json(
                key=path,
                data=self.as_dict(),
            )


class RandAttack:
    """Random attack is test framework which tests H2O Sonar by attacking it with
    randomly selected/created models. The goal of the random attack is to perform
    functional, load and performance testing on a rich collection of datasets
    in order to find as many defects as possible.

    """

    FILE_REPORT_JSON = "random_attack_report.json"

    def __init__(
        self,
        attack_path: pathlib.Path,
        datasets_path: pathlib.Path,
        models_index_path: pathlib.Path | None = None,
        options: RandomAttackOptions | None = None,
    ):
        """Constructor.

        Parameters
        ----------
        attack_path :
          Where to store random attack files.
        datasets_path :
          Path to the directory to be scanned for datasets.
        models_index_path :
          Optional path to `models.json` file which provides information about prepared
          models (path to dataset, path to model, target column, ... for every model).

        """
        if not attack_path:
            raise ValueError(f"Attack path must be specified: {attack_path}")
        attack_path.mkdir(parents=True, exist_ok=True)

        self.logger = loggers.SonarFileLogger(
            logger_name="H2oSonarRandomAttack",
            log_file=str(attack_path / "random-attack.log"),
            log_level=loggers.DEBUG,
        )

        self.opts: RandomAttackOptions = options or RandomAttackOptions()

        self.attack_path = attack_path
        self.datasets_path = datasets_path
        self.models_index_path = models_index_path

        # directory to store datasets for models which need label encoding
        self.enc_datasets_path = attack_path / "datasets"
        self.enc_datasets_path.mkdir(parents=True, exist_ok=True)

        self.models_idx: ModelsIndex = ModelsIndex(self.logger)
        if self.models_index_path:
            self.models_idx.add_model_idx_file(self.models_index_path)

        self.datasets_idx: list[DatasetIndexEntry] = []
        self.dataset_api = datasets.DatasetApi(self.logger)
        self.index_datasets()

        self.logger.info(f"Indexed {len(self.datasets_idx)} datasets")
        self.logger.info(f"Indexed {len(self.models_idx.models)} models")

        self.model_trainer = ModelTrainer(self.enc_datasets_path, self.logger)

    def index_datasets(self):
        # find all *.csv files (.jay, .zip, ... might be added later)
        for i in self.datasets_path.glob("**/*.csv"):
            try:
                self.logger.debug(f"Indexing dataset: {i}")
                self.logger.debug(f"  Size: {i.stat().st_size}B")
                if not self.opts.sample_dataset_to:
                    if i.stat().st_size > self.opts.max_dataset_size:
                        self.logger.warning(
                            "  SKIPPING as it's bigger than dataset size limit"
                        )
                        continue
                else:
                    if i.stat().st_size > self.opts.sample_dataset_to:
                        self.logger.warning(
                            f"  dataset will be SAMPLED by H2O Sonar to "
                            f"{self.opts.sample_dataset_to}B"
                        )

                self.index_dataset(i)
            except Exception as eex:
                self.logger.warning(
                    f"Unable to index dataset '{i}': {eex}\n{traceback.format_exc()}"
                )

        if len(self.datasets_idx) == 0:
            raise RuntimeError(f"Unable to index any datasets in {self.datasets_path}")

    def index_dataset(self, dataset_path: pathlib.Path):
        e_dataset = self.dataset_api.create_dataset(
            dataset_src=str(dataset_path),
            # sampling limit is intentionally DISABLED, otherwise all datasets
            # (including those which will not be used) would be sampled - instead,
            # sampling limit is passed later to H2O Sonar to test its sampling
            sample_num_rows=0,
        )
        self.logger.debug("  Explainable dataset:")
        self.logger.debug(f"    original path: {e_dataset.meta.original_dataset_path}")
        self.logger.debug(f"    original size: {e_dataset.meta.original_dataset_size}")
        self.logger.debug(f"    path         : {e_dataset.meta.file_path}")
        self.logger.debug(f"    size         : {e_dataset.meta.file_size}")

        idx_dataset = DatasetIndexEntry(
            dataset_path=dataset_path,
            regression_cols=[],
            binomial_cols=[],
            multinomial_cols=[],
        )

        for c in e_dataset.meta.columns_meta:
            if not c.is_id and c.count:
                if 2 == c.count:
                    idx_dataset.binomial_cols.append(c.name)
                elif c.is_numeric:
                    if c.count > self.opts.cat_num_threshold:
                        idx_dataset.regression_cols.append(c.name)
                    else:
                        idx_dataset.multinomial_cols.append(c.name)
                else:
                    idx_dataset.multinomial_cols.append(c.name)

        if (
            idx_dataset.regression_cols
            or idx_dataset.binomial_cols
            or idx_dataset.multinomial_cols
        ):
            self.datasets_idx.append(idx_dataset)

    def attack(self, explainer_keywords=None, explainers=None):
        """Run random attack:

        - randomly chose dataset
        - randomly chose R column
          > train model
            > run H2O Sonar
        - randomly choose B column ...
          > train model
            > run H2O Sonar
        - randomly choose M column ...
          > train model
            > run H2O Sonar
        - LOOP

        Parameters
        ----------
        explainer_keywords :
          Keywords.
        explainers :
          Explainer IDs/parameters.

        """
        attack_report = RandAttackReport()

        # every dataset will be used at most once - max attacks must reflect it
        self.opts.max_attacks_limit = min(
            self.opts.max_attacks_limit, len(self.datasets_idx)
        )
        # random sequence of datasets (w/o repetitions) to be used for the attack
        rand_dataset_idxs = get_rand_sequence(
            range_max=len(self.datasets_idx), count=self.opts.max_attacks_limit
        )

        for i in range(self.opts.max_attacks_limit):
            ensure_disk_space()

            # dataset
            dataset: DatasetIndexEntry = self.datasets_idx[rand_dataset_idxs[i]]
            dataset_cols = [
                dataset.regression_cols,
                dataset.binomial_cols,
                dataset.multinomial_cols,
            ]
            for columns in dataset_cols:
                if not columns:
                    continue

                # model
                target_col = columns[random.randrange(len(columns))]
                try:
                    (model, dataset_path) = self.model_trainer.train_sklearn_gbm(
                        dataset=dataset,
                        target_col=target_col,
                    )
                except Exception as ex:
                    err_msg = (
                        f"Unable to train model for the dataset {dataset} and target "
                        f"column {target_col}: {ex}\n{traceback.format_exc()}"
                    )
                    attack_report.add_failed_model(
                        dataset_path=dataset.dataset_path,
                        target_col=target_col,
                        error_msg=err_msg,
                    )
                    self.logger.error(err_msg)
                    continue

                # explainer selection
                if explainers:
                    explainer_keywords = None
                elif not explainer_keywords:
                    explainer_keywords = [interpret.KEYWORD_FILTER_ALL]

                # attack
                interpretation = None
                success: bool = True
                try:
                    interpretation = interpret.run_interpretation(
                        dataset=dataset_path,
                        sample_num_rows=self.opts.sample_dataset_to,
                        model=model,
                        target_col=target_col,
                        explainer_keywords=explainer_keywords,
                        explainers=explainers,
                        results_location=self.attack_path,
                        log_level=loggers.DEBUG,
                    )
                    self.logger.info(
                        f"DONE: interpretation successfully finished: {interpretation}"
                    )

                    if interpretation.get_failed_explainer_ids():
                        success = False
                        attack_report.add_failed_explainers(
                            interpretation_path=(
                                interpretation.result.interpretation_location
                                if interpretation
                                else ""
                            ),
                            key=interpretation.key if interpretation else "",
                            dataset_path=dataset_path,
                            model_path="",
                            model_type=f"{model}",
                            target_col=target_col,
                            error_msg=(
                                f"Failed explainers: "
                                f"{interpretation.get_failed_explainer_ids()}"
                            ),
                        )
                except Exception as ex:
                    success = False
                    err_msg = (
                        f"H2O Sonar failed on {dataset} and "
                        f"target column {target_col}: {ex}\n{traceback.format_exc()}"
                    )
                    attack_report.add_failed_interpretation(
                        interpretation_path=(
                            interpretation.result.interpretation_location
                            if interpretation
                            else ""
                        ),
                        key=interpretation.key if interpretation else "",
                        dataset_path=dataset_path,
                        model_path="",
                        model_type=f"{model}",
                        target_col=target_col,
                        error_msg=err_msg,
                    )
                    self.logger.error(err_msg)

                if success and self.opts.purge_ok_interpretations:
                    shutil.rmtree(interpretation.result.interpretation_location)

            attack_report.save(path=self.attack_path / RandAttack.FILE_REPORT_JSON)

        self.logger.info("Random attack successfully finished!")


if __name__ == "__main__":
    """Run random attack.

    Command line arguments:
    -----------------------
    --datasets-dir
        Path to the directory with datasets.
    --models-index
        Path to the models index file.

    Shell environment variables:
    ----------------------------
    OPT_ATTACK_MAX_ATTACKS : int
        Maximum number of attacks to be performed (0 means infinite).
    OPT_ATTACK_MIN_DISK_SPACE : int
           Minimum disk space (in bytes) to be left on the disk while running the
           attack (remaining disk space is periodically checked).
    OPT_ATTACK_MAX_DATASET_SIZE : int
        Maximum size (in bytes) of the dataset to be used for the attack. Larger
        datasets are skipped.
    OPT_ATTACK_CAT_NUM_THRESHOLD : int
        Maximum number of categories in categorical columns. Columns with more
        unique values will be handled as numeric columns.
    OPT_SAMPLE_DATASET_TO : int
        Sample dataset to the specified number of rows. If the value is 0, then it
        will not be sampled.

    """

    print(f"Running random attack: {sys.argv}")

    _datasets_path = pathlib.Path()
    _models_index_path = None
    for a in sys.argv[1:]:
        if a.startswith(ARG_DATASETS_DIR):
            _datasets_path = a.replace(ARG_DATASETS_DIR, "")
            if not _datasets_path:
                raise ValueError("Path to datasets must be specified")
            _datasets_path = pathlib.Path(_datasets_path)
            if not _datasets_path.exists():
                raise ValueError(f"Path to datasets does not exists: {_datasets_path}")
        elif a.startswith(ARG_MODELS_INDEX):
            _models_index_path = a.replace(ARG_MODELS_INDEX, "")
            if _models_index_path:
                _models_index_path = pathlib.Path(_models_index_path)
                if not _models_index_path.exists():
                    raise ValueError(
                        f"Path to models index does not exists: {_models_index_path}"
                    )
                if not _models_index_path.is_file():
                    raise ValueError(
                        "Path to models index must point to the JSon file, but it "
                        "points to a directory"
                    )
        else:
            raise ValueError(f"Unknown argument: '{a}'")

    attacker = RandAttack(
        attack_path=pathlib.Path() / "random-attack",
        datasets_path=_datasets_path,
        models_index_path=_models_index_path,
        options=RandomAttackOptions(),
    )

    attacker.attack()
