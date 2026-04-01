# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
import json
import logging

from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import persistences as p10s
from h2o_sonar.lib.api.models import ExplainableModelMeta


# serialization/deserialization keys
# LIST EXPLAINERS args
KEY_EXPERIMENT_TYPES = "experiment_types"
KEY_EXPLANATION_SCOPES = "explanation_scopes"
KEY_MODEL_META = "model_meta"
KEY_KEYWORDS = "keywords"
KEY_EXPLAINER_FILTER = "explainer_filter"
KEY_EXTRA_PARAMS = "extra_params"

# RUN INTERPRETATION args
KEY_DATASET = "dataset"
KEY_MODEL = "model"
KEY_MODELS = "models"
KEY_TARGET_COL = "target_col"
KEY_EXPLAINERS = "explainers"
KEY_EXPLAINER_KEYWORDS = "explainer_keywords"
KEY_VALIDSET = "validset"
KEY_TESTSET = "testset"
KEY_USE_RAW_FEATURES = "use_raw_features"
KEY_USED_FEATURES = "used_features"
KEY_WEIGHT_COL = "weight_col"
KEY_PREDICTION_COL = "prediction_col"
KEY_DROP_COLS = "drop_cols"
KEY_SAMPLE_NUM_ROWS = "sample_num_rows"
KEY_RESULTS_LOCATION = "results_location"
KEY_RESULTS_FORMATS = "results_formats"
KEY_PERSISTENCE_TYPE = "persistence_type"
KEY_RUN_ASYNCHRONOUSLY = "run_asynchronously"
KEY_RUN_E_IN_PARALLEL = "run_explainers_in_parallel"
KEY_UPLOAD_TO = "upload_to"
KEY_KEY = "key"
KEY_LOG_LEVEL = "log_level"
KEY_CONFIG_PATH = "config_path"
KEY_ENCRYPTION_KEY = "encryption_key"


def from_list_explainers_args_json(args_str: str) -> dict:
    """Deserialize ``interpret.py::list_explainers()`` method arguments from JSon
    string to dictionary which might be used as a Python method kwargs.

    """
    if args_str:
        args_dict = json.loads(args_str)

        if args_dict.get(KEY_MODEL_META, None):
            args_dict[KEY_MODEL_META] = ExplainableModelMeta()

        if args_dict.get(KEY_EXPLAINER_FILTER, None):
            args_dict[KEY_EXPLAINER_FILTER] = [
                commons.FilterEntry(
                    filter_by=f.get("filter_by", None),
                    value=f.get("value", None),
                )
                for f in args_dict[KEY_EXPLAINER_FILTER]
            ]

        return args_dict

    return {}


def to_list_explainers_args_json(
    experiment_types: list[str] | None = None,
    explanation_scopes: list[str] | None = None,
    model_meta: ExplainableModelMeta | None = None,
    keywords: list[str] | None = None,
    explainer_filter: list[commons.FilterEntry] | None = None,
    extra_params: dict | None = None,
) -> str:
    """Serialize ``interpret.py::list_explainers()`` method arguments as JSon."""
    experiment_types = experiment_types or []
    explanation_scopes = explanation_scopes or []
    explainer_filter = explainer_filter or []
    keywords = keywords or []
    extra_params = extra_params or {}

    return json.dumps(
        {
            KEY_EXPERIMENT_TYPES: experiment_types,
            KEY_EXPLANATION_SCOPES: explanation_scopes,
            KEY_MODEL_META: model_meta,
            KEY_KEYWORDS: keywords,
            KEY_EXPLAINER_FILTER: [f.dump() for f in explainer_filter],
            KEY_EXTRA_PARAMS: extra_params,
        }
    )


def to_run_interpretation_args_json(
    dataset: str = "",
    model: str = "",
    target_col: str = "",
    explainers: list[str | commons.ExplainerToRun] | None = None,
    explainer_keywords: list[str] | None = None,
    validset: str = "",
    testset: str = "",
    use_raw_features: bool = True,
    used_features: list | None = None,
    weight_col: str = "",
    prediction_col: str = "",
    drop_cols: list | None = None,
    sample_num_rows: int | None = None,
    log_level: int = logging.WARNING,
    results_location: str = None,
    persistence_type: p10s.PersistenceType = p10s.PersistenceType.file_system,
    run_asynchronously: bool = False,
    run_explainers_in_parallel: bool = False,
    extra_params: dict | None = None,
) -> str:
    """Serialize ``interpret.py::run_interpretation()`` job arguments as JSon."""
    explainers = explainers or []
    explainers_list_dict = [
        e.dump() if isinstance(e, commons.ExplainerToRun) else e for e in explainers
    ]
    explainer_keywords = explainer_keywords or []
    drop_cols = drop_cols or []
    persistence_type = (
        str(persistence_type.name)
        if persistence_type
        else p10s.PersistenceType.file_system.name
    )
    extra_params = extra_params or {}

    return json.dumps(
        {
            KEY_DATASET: dataset,
            KEY_MODEL: model,
            KEY_TARGET_COL: target_col,
            KEY_EXPLAINERS: explainers_list_dict,
            KEY_EXPLAINER_KEYWORDS: explainer_keywords,
            KEY_VALIDSET: validset,
            KEY_TESTSET: testset,
            KEY_USE_RAW_FEATURES: use_raw_features,
            KEY_USED_FEATURES: used_features,
            KEY_WEIGHT_COL: weight_col,
            KEY_PREDICTION_COL: prediction_col,
            KEY_DROP_COLS: drop_cols,
            KEY_SAMPLE_NUM_ROWS: sample_num_rows,
            KEY_LOG_LEVEL: log_level,
            KEY_RESULTS_LOCATION: results_location,
            KEY_PERSISTENCE_TYPE: persistence_type,
            KEY_RUN_ASYNCHRONOUSLY: run_asynchronously,
            KEY_RUN_E_IN_PARALLEL: run_explainers_in_parallel,
            KEY_EXTRA_PARAMS: extra_params,
        }
    )


def from_run_interpretation_args_json(args_str: str) -> dict:
    """Deserialize ``interpret.py::run_interpretation()`` method arguments from JSon
    string to dictionary which might be used as a Python method kwargs.

    """
    if args_str:
        args_dict = json.loads(args_str)

        if not args_dict:
            raise ValueError(
                f"Unable to load run interpretation arguments from JSon string: "
                f"'{args_str}' and dictionary: '{args_dict}' of type: {type(args_dict)}"
            )
        if args_dict.get(KEY_EXPLAINERS, None):
            explainers = []
            for e in args_dict[KEY_EXPLAINERS]:
                if isinstance(e, str):
                    explainers.append(e)
                elif isinstance(e, dict):
                    e2run = commons.ExplainerToRun(
                        explainer_id=e.get("id", None),
                        params=e.get("params", None),
                    )
                    if not e2run.id:
                        raise ValueError(
                            f"Method arguments deserialization detected invalid "
                            f"explainer ID: '{e2run.id}'"
                        )
                    explainers.append(e2run)
                else:
                    raise ValueError(
                        f"Method arguments deserialization detected invalid "
                        f"explainer to run type: {type(e)}"
                    )

            args_dict[KEY_EXPLAINERS] = explainers

        if args_dict.get(KEY_PERSISTENCE_TYPE, None):
            p = args_dict[KEY_PERSISTENCE_TYPE]
            if p == p10s.PersistenceType.in_memory.name:
                args_dict[KEY_PERSISTENCE_TYPE] = p10s.PersistenceType.in_memory
            elif p == p10s.PersistenceType.database.name:
                args_dict[KEY_PERSISTENCE_TYPE] = p10s.PersistenceType.database
            else:
                args_dict[KEY_PERSISTENCE_TYPE] = p10s.PersistenceType.file_system
        else:
            args_dict[KEY_PERSISTENCE_TYPE] = p10s.PersistenceType.file_system

        return args_dict

    return {}


def load_list_explainers_args_json(file_path) -> dict:
    """Load ``list_explainers()`` keyword arguments from file."""
    with open(file_path) as file:
        data = file.read()
    return from_list_explainers_args_json(data)


def load_run_interpretation_args_json(file_path) -> dict:
    """Load ``run_interpretation()`` keyword arguments from file."""
    with open(file_path) as file:
        data = file.read()
    return from_run_interpretation_args_json(data)
