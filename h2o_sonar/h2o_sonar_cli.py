# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
#
# H2O Sonar command line interface.
#
import argparse
import ast
import json
import logging
import os.path
import sys
import traceback

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar import errors
from h2o_sonar import interpret
from h2o_sonar import version
from h2o_sonar.lib.api import commons


# actions
ACTION_ADD = "add"
ACTION_SHOW = "show"
ACTION_LIST = "list"
ACTION_DESCRIBE = "describe"
ACTION_RUN = "run"
ACTION_UPLOAD = "upload"
ACTIONS = [
    ACTION_ADD,
    ACTION_SHOW,
    ACTION_LIST,
    ACTION_DESCRIBE,
    ACTION_RUN,
    ACTION_UPLOAD,
]

# entities
ENTITY_CONFIG = "config"
ENTITY_EXPLAINER = "explainer"
ENTITY_EXPLAINERS = "explainers"
ENTITY_INTERPRETATION = "interpretation"
ENTITY_INTERPRETATIONS = "interpretations"
ENTITY_VERSION = "version"
ENTITIES = [
    ENTITY_CONFIG,
    ENTITY_VERSION,
    ENTITY_EXPLAINER,
    ENTITY_EXPLAINERS,
    ENTITY_INTERPRETATION,
    ENTITY_INTERPRETATIONS,
]

# log levels
LOG_LEVELS = {
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "warn": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


def main() -> int:
    """H2O Sonar Python library for Responsible AI command line interface (CLI).
    See ``argparse`` configuration for more details.

    """
    parser = argparse.ArgumentParser(
        description=(
            """H2O Sonar Python library for Responsible AI.

H2O Sonar is Python package that enables a holistic, low-risk, human-interpretable,
fair, and trustable approach to machine learning by implementing various facets of
Responsible AI.

model, dataset, validset or testset handle schema:

  resource:connection:"<connection_configuration_name>":key:<resource_key>
  [:version:<resource_version>]

optional arguments per action and entity:

  show version:
                      show H2O Sonar version

  add config:
    --config-path     path to JSon or TOML file with H2O Sonar config to be changed
    --config-type     config item type: 'CONNECTION' or 'LICENSE'
    --config-value    config item value (serialized as JSon) to add to the config file
    --encryption-key  secret key to encrypt config fields with sensitive data
                      (alternatively set H2O_SONAR_ENCRYPTION_KEY environment variable)

  show config:
    --config-path     path to JSon or TOML file with H2O Sonar config
    --encryption-key  optional secret key to decrypt config fields with sensitive data
                      (alternatively set H2O_SONAR_ENCRYPTION_KEY environment variable)

  list explainers:
    --detailed        show detailed descriptors (only IDs are shown by default)
    --args-as-json-location
                      optional JSon file which overrides filtering CLI arguments

  describe explainer:
    --explainer       explainer ID

  run interpretation:
    --dataset         path to dataset
    --target-col      target column
    --model           path to the serialized model, URL or locator
    --results-location
                      optional path to the interpretation results location (directory)
    --validset        optional path to validation dataset
    --testset         optional path to test dataset
    --use_raw_features
                      force the use of transformed features in surrogate models
                      with 'false', by default the original (raw) features are used
    --weight-col      optional dataset column name with examples weights
    --drop-cols       optional list of dataset columns to drop
    --sample-num-rows
                      optional number of rows to sample from dataset (default: sample
                      based on the RAM size, 0 do not sample, >0 sample to the specified
                      number of rows)
    --all-explainers  run all explainers (only the most important are run by default)
    --used-features   optional comma separated list of features used by the model
    --model-type      optional model type: 'pickle' or 'mojo'
    --explainers      optional comma separated list of explainer IDs to be run
    --explainers-pars optional dictionary with explainer parameters
    --config-path     path to JSon or TOML file with H2O Sonar config to be changed
    --encryption-key  secret key to encrypt config fields with sensitive data
                      (alternatively set H2O_SONAR_ENCRYPTION_KEY environment variable)
    --upload-to       optional (h2oGPT Enterprise) connection key from the configuration
                      where the report is to be uploaded in order to talk to it
    --args-as-json-location
                      optional JSon file which overrides CLI arguments
    --log-level       optional log level: 'error', 'warning', 'info', 'debug'

  list interpretations:
    --results-location
                      path to directory, URL, location of interpretation results
    --log-level       optional log level: 'error', 'warning', 'info', 'debug'

  upload interpretations:
    --interpretation  path to the interpretation report PDF or HTML file
    --upload-to       (h2oGPT Enterprise) connection key from the configuration
                      where the report is to be uploaded in order to talk to it
    --collection-name optional name of the collection where the report is to be uploaded
    --config-path     path to JSon or TOML file with H2O Sonar config to be changed
    --encryption-key  secret key to encrypt config fields with sensitive data
                      (alternatively set H2O_SONAR_ENCRYPTION_KEY environment variable)
"""
        ),
        epilog=(
            """examples:

  h2o-sonar --help
  h2o-sonar show version
  h2o-sonar list explainers
  h2o-sonar list explainers --detailed
  h2o-sonar describe explainer
    --explainer=h2o_sonar.explainers.dia_explainer.DiaExplainer
  h2o-sonar run interpretation
    --dataset=dataset.csv
    --target-col=PROFIT
    --results-location=/home/user/results
    --model=model.pickle
    --all-explainers
  h2o-sonar run interpretation
    --dataset=dataset.csv
    --target-col=PROFIT
    --results-location=/home/user/results
    --model=model.pickle
    --used-features=FEATURE_1,FEATURE_2,FEATURE_3
    --explainers=h2o_sonar.explainers.dia_explainer.DiaExplainer
    --explainers-pars=
      "{'h2o_sonar.explainers.dia_explainer.DiaExplainer':{'cut_off': 0.5}}"
    --drop_cols=COLUMN_1,COLUMN_2,COLUMN_3
    --config-path=h2o-sonar-config.json
    --upload-to=4cff6fc9-f49a-4dda-aeb5-2c42e9f12807
  h2o-sonar run interpretation
    --args-as-json-location=h2o-sonar-args.json
  h2o-sonar list interpretations --results-location=/home/user/results
  h2o-sonar upload interpretation
    --interpretation=./results/h2o-sonar/mli-experiment/interpretation-detailed.html
    --upload-to=4cff6fc9-f49a-4dda-aeb5-2c42e9f12807

H2O Sonar JSon configuration example:
  {
    "h2o_host": "192.168.0.1",
    "h2o_port": 57561,
    "h2o_auto_start": true,
    "connections": [
      {
        "key": "4cff6fc9-f49a-4dda-aeb5-2c42e9f12807",
        "connection_type": "H2O_GPT_E",
        "name": "H2O GPT Enterprise",
        "description": "H2O GPT Enterprise service.",
        "server_url": "https://h2ogpte.h2o.ai",
        "token": {
          "encrypted": "gAAA3LcKQ7x_X...gnKsBqVdNydYTlk8nyQ=="
        },
        "token_use_type": "API_KEY"
      }
    ]
  }

Interpretation arguments JSon file example - see interpret.py::run_interpretation():
  {
    "dataset": "dataset.csv",
    "model": "model.pickle",
    "target_col": "PROFIT",
    "results_location": "./results"
  }

Explainer listing arguments JSon file example - see interpret.py::list_explainers():
  {
    "experiment_types": ["regression"],
    "explanation_scopes": ["local_scope"],
    "keywords": ["explains-fairness"],
    "explainer_filter": [{"filter_by": "filter-name", "value": "v"}]
  }
"""
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.prog = "h2o-sonar"

    # positional arguments
    parser.add_argument(
        "action",
        metavar="action",
        type=str,
        help="action to take: 'list', 'run' or 'describe'",
    )
    parser.add_argument(
        "entity",
        metavar="entity",
        type=str,
        help=(
            "entity on which to perform the action: 'interpretation'(s) or "
            "'explainer'(s)"
        ),
    )

    # optional arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="location of the dataset",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="",
        help="target column",
    )
    parser.add_argument(
        "--results-location",
        type=str,
        default="",
        help="location where to store the interpretation results",
    )
    parser.add_argument(
        "--results-formats",
        type=str,
        default="",
        help=(
            "comma separated list of MIME types of the interpretation results to create"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="location of the model",
    )
    parser.add_argument(
        "--validset",
        type=str,
        default="",
        help="location of the validation dataset",
    )
    parser.add_argument(
        "--testset",
        type=str,
        default="",
        help="location of the test dataset",
    )
    parser.add_argument(
        "--use_raw-features",
        type=str,
        default="true",
        help="force the use of transformed features in surrogate models with `false`",
    )
    parser.add_argument(
        "--weight-col",
        type=str,
        default="",
        help="optional dataset column name with examples weights",
    )
    parser.add_argument(
        "--drop-cols",
        type=str,
        default="",
        help="optional list of dataset columns to drop",
    )
    parser.add_argument(
        "--sample-num-rows",
        type=str,
        default="",
        help="optional number of rows to sample from the dataset",
    )
    parser.add_argument(
        "--used-features",
        type=str,
        default="",
        help="optional comma separated list of features used by the model",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["pickle", "mojo"],
        default="",
        help="model type: 'pickle' (.pkl) or 'mojo' (.mojo)",
    )
    parser.add_argument(
        "--explainer",
        type=str,
        default="",
        help="ID of the explainer to describe",
    )
    parser.add_argument(
        "--explainers",
        type=str,
        default="",
        help=(
            "comma separated list of explainer IDs to be run (only the most important "
            "explainers are run by default)"
        ),
    )
    parser.add_argument(
        "--all-explainers",
        action="store_true",
        help=(
            "run all explainers (only the most important explainers are run by default)"
        ),
    )
    parser.add_argument(
        "--explainers-pars",
        type=str,
        default="",
        help=(
            "optional dictionary with explainer parameters - the dictionary key is "
            "explainer ID and value is dictionary with parameters; parameter "
            "dictionary has parameter name as the key and parameter value as the value"
        ),
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="",
        help=(
            "path to JSon or TOML file with H2O Sonar configuration to be used to "
            "override defaults - specify only items you want to change (please refer "
            "to h2o_sonar.config.H2oSonarConfig for more details)"
        ),
    )
    parser.add_argument(
        "--config-type",
        type=str,
        default="",
        help="configuration item type - 'CONNECTION' or 'LICENSE'",
    )
    parser.add_argument(
        "--config-value",
        type=str,
        default="",
        help=(
            "configuration item value represented either as dictionary or as string "
            "with JSon serialization of the configuration item - it is expected that "
            "the config item is NOT encrypted"
        ),
    )
    parser.add_argument(
        "--encryption-key",
        type=str,
        default="",
        help=(
            "encryption key to be used for encrypting/decrypting sensitive data "
            "in the configuration. If not specified, shell environment variable "
            "H2O_SONAR_ENCRYPTION_KEY with the encryption key is used."
        ),
    )
    parser.add_argument(
        "-d",
        "--detailed",
        action="store_true",
        help="show detailed descriptors (only IDs are shown by default)",
    )
    parser.add_argument(
        "--interpretation",
        type=str,
        default="",
        help="path to the interpretation report (PDF or HTML) to be uploaded",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="",
        help=(
            "optional (h2oGPT Enterprise) collection name where to upload the "
            "interpretation report"
        ),
    )
    parser.add_argument(
        "--upload-to",
        type=str,
        default="",
        help=(
            "optional (h2oGPT Enterprise) connection key from the configuration "
            "where the report is to be uploaded in order to talk to it"
        ),
    )
    parser.add_argument(
        "--args-as-json-location",
        type=str,
        default="",
        help=(
            "location of the JSon file with all command arguments (replacing command "
            "line arguments) allowing to load them from the filesystem"
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="warning",
        choices=["error", "warning", "info", "debug"],
        help="log level",
    )
    args = parser.parse_args()

    action = _validate_action(args.action)
    entity = _validate_entity(args.entity)

    if action == ACTION_ADD:
        if entity == ENTITY_CONFIG:
            arg_config_type = _validate_arg_config_type(args.config_type)
            arg_config_value = _validate_arg_config_value(args.config_value)
            arg_config_path = _validate_arg_config_path(args.config_path)
            arg_encryption_key = _validate_arg_encryption_key(args.encryption_key)

            interpret.add_config_item(
                config_type=arg_config_type,
                config_value=arg_config_value,
                h2o_sonar_config_path=arg_config_path,
                encryption_key=arg_encryption_key,
            )

            return 0
        else:
            raise errors.InvalidArgumentError(f"Unknown entity '{entity}' to add.")

    if action == ACTION_SHOW:
        if entity == ENTITY_VERSION:
            print(f"H2O Sonar {version.__version__}")
            return 0
        elif entity == ENTITY_CONFIG:
            if args.config_path:
                arg_config_path = _validate_arg_config_path(args.config_path)
                arg_encryption_key = _validate_arg_encryption_key(args.encryption_key)
            else:
                arg_config_path = arg_encryption_key = ""

            cfg_as_dict = interpret.get_config(
                h2o_sonar_config_path=arg_config_path,
                encryption_key=arg_encryption_key,
            )

            print(json.dumps(cfg_as_dict, indent=2))

            return 0
        else:
            raise errors.InvalidArgumentError(f"Unknown entity '{entity}' to show.")

    if action == ACTION_LIST:
        if entity == ENTITY_EXPLAINERS:
            arg_detailed = _validate_arg_detailed(args.detailed)
            arg_args_as_json_location = _validate_arg_validset(
                args.args_as_json_location
            )

            list_of_explainers = interpret.list_explainers(
                args_as_json_location=arg_args_as_json_location,
            )

            if list_of_explainers:
                if not arg_detailed:
                    p_explainers = [ee.id for ee in list_of_explainers]
                else:
                    p_explainers = [ee.dump() for ee in list_of_explainers]
            else:
                p_explainers = []

            print(json.dumps({"explainers": p_explainers}, indent=2))

            return 0
        elif entity == ENTITY_INTERPRETATIONS:
            arg_results_location = _validate_arg_results_location(
                args.results_location, None
            )
            arg_log_level = _validate_arg_log_level(args.log_level)

            interpretations = interpret.list_interpretations(
                results_location=arg_results_location,
                log_level=arg_log_level,
            )

            print(interpretations)

            return 0
        else:
            raise errors.InvalidArgumentError(
                f"Unsupported command - unknown entity '{entity}' for valid action "
                f"'{action}'"
            )

    if action == ACTION_RUN and entity == ENTITY_INTERPRETATION:
        arg_args_as_json_location = _validate_arg_validset(args.args_as_json_location)
        arg_dataset = _validate_arg_dataset_path(
            args.dataset, arg_args_as_json_location
        )
        arg_model = _validate_arg_model(args.model, arg_args_as_json_location)
        arg_validset = _validate_arg_validset(args.validset)
        arg_testset = _validate_arg_validset(args.testset)
        arg_use_raw_features = _validate_arg_use_raw_features(args.use_raw_features)
        arg_weight_col = _validate_arg_weight_col(args.weight_col)
        arg_drop_cols = _validate_arg_list_of_features(
            args.drop_cols,
            (
                "Unable to parse command line argument with features to drop  - it "
                "must be comma separated list of feature names"
            ),
        )
        arg_sample_num_rows = _validate_sample_num_rows(args.sample_num_rows)
        arg_used_model_features = _validate_arg_list_of_features(
            args.used_features,
            (
                "Unable to parse command line argument with features used by the "
                "model - it must be comma separated list of feature names"
            ),
        )
        # arg_model_type = _validate_arg_model(args.model_type)
        arg_target_col = _validate_arg_target_col(
            args.target_col, arg_args_as_json_location
        )
        arg_all_explainers = _validate_arg_bool(args.all_explainers)
        arg_explainers = _validate_arg_explainers(args.explainers)
        arg_explainers = _validate_arg_explainer_pars(
            args.explainers_pars, arg_explainers
        )
        arg_connection = _validate_arg_connection_key(
            key=args.upload_to,
            args_as_json_location=arg_args_as_json_location,
            is_optional=True,
        )
        arg_results_location = _validate_arg_results_location(
            args.results_location, arg_args_as_json_location
        )
        arg_results_formats = _validate_arg_results_formats(args.results_formats)
        arg_log_level = _validate_arg_log_level(args.log_level)
        arg_encryption_key = _validate_arg_encryption_key(args.encryption_key)
        _resolve_config_path(
            config_path=args.config_path,
            encryption_key=arg_encryption_key or "",
        )

        interpretation = interpret.run_interpretation(
            dataset=arg_dataset,
            model=arg_model,
            validset=arg_validset,
            testset=arg_testset,
            use_raw_features=arg_use_raw_features,
            weight_col=arg_weight_col,
            drop_cols=arg_drop_cols,
            sample_num_rows=arg_sample_num_rows,
            used_features=arg_used_model_features,
            target_col=arg_target_col,
            explainers=arg_explainers,
            explainer_keywords=(
                [commons.KEYWORD_FILTER_ALL] if arg_all_explainers else None
            ),
            upload_to=arg_connection,
            results_location=arg_results_location,
            results_formats=arg_results_formats,
            args_as_json_location=arg_args_as_json_location,
            log_level=arg_log_level,
        )

        interpretation_dict = interpretation.to_dict()
        if interpretation_dict:
            if (
                not interpretation.get_failed_explainer_ids()
                and not interpretation.get_finished_explainer_ids()
            ):
                print(
                    "\nInterpretation FINISHED, but there were NO compatible "
                    "explainers to run."
                )
            else:
                print("\nInterpretation FINISHED.")

            path = _make_url_for_file_path(
                interpretation.result.get_results_dir_location()
            )
            print(f"  Results directory:\n    {path}")

            path = _make_url_for_file_path(
                interpretation.result.get_interpretations_html_index_location()
            )
            print(f"  Interpretations index:\n    {path}")

            path = _make_url_for_file_path(
                interpretation.result.get_html_report_location()
            )
            print(f"  HTML report:\n    {path}")

            if interpretation.result.upload_url:
                print(
                    f"  Interpretation has been uploaded to:"
                    f"\n    {interpretation.result.upload_url}"
                )
        else:
            print("Interpretation finished, but it created no result.")

        return 0
    elif action == ACTION_DESCRIBE and entity == ENTITY_EXPLAINER:
        if not args.explainer:
            raise errors.InvalidArgumentError(
                "Required argument --explainer with the explainer ID is missing"
            )

        description = interpret.describe_explainer(args.explainer)
        print(json.dumps(description, indent=2))

        return 0

    elif action == ACTION_UPLOAD:
        if entity == ENTITY_INTERPRETATION:
            arg_interpretation = _validate_arg_file_path(
                path=args.interpretation, args_as_json_location=None, is_optional=False
            )
            arg_connection = _validate_arg_connection_key(
                key=args.upload_to, args_as_json_location=None, is_optional=False
            )
            arg_encryption_key = _validate_arg_encryption_key(args.encryption_key)
            _resolve_config_path(
                config_path=args.config_path,
                encryption_key=arg_encryption_key or "",
            )

            (_, collection_url) = interpret.upload_interpretation(
                interpretation_result=arg_interpretation,
                connection=arg_connection,
                collection_name=args.collection_name,
            )

            print(f"Interpretation has been uploaded to:\n  {collection_url}")

            return 0

    raise errors.InvalidArgumentValueError(
        f"Unsupported action '{action}' or entity '{entity}' - please refer to help "
        f"for supported actions and entities."
    )


def _validate_action(action) -> str:
    if action:
        if action in ACTIONS:
            return action
        else:
            raise errors.InvalidArgumentValueError(
                f"Unknown action: '{action}' - must be one of {ACTIONS}"
            )

    raise errors.InvalidArgumentError(
        f"Required argument 'action' is missing - must be one of {ACTIONS}"
    )


def _validate_entity(entity) -> str:
    if entity:
        if entity in ENTITIES:
            return entity
        else:
            raise errors.InvalidArgumentError(
                f"Unknown entity: '{entity}' - must be one of {ENTITIES}"
            )

    raise errors.InvalidArgumentError(
        f"Required argument 'entity' is missing - must be one of {ENTITIES}"
    )


def _validate_arg_detailed(detailed) -> bool:
    if detailed:
        if isinstance(detailed, bool):
            return detailed
        elif str(detailed).lower() in ["true", "1"]:
            return True

    return False


def _validate_arg_validset(dataset_path) -> str:
    return dataset_path


def _validate_arg_dataset_path(dataset, args_as_json_location) -> str:
    if dataset or args_as_json_location:
        return dataset

    raise errors.InvalidArgumentError("Missing required option '--dataset'")


def _validate_arg_model(model, args_as_json_location) -> str:
    if model or args_as_json_location:
        return model

    return ""


def _validate_arg_drop_cols(drop_cols) -> list | None:
    if drop_cols:
        try:
            return drop_cols.split(",")
        except Exception as ex:
            raise errors.InvalidArgumentError(
                f"Unable to parse command line argument with columns to drop "
                f"- it must be comma separated list of column names: {ex}"
            )

    return []


def _validate_arg_list_of_features(model_features, err_msg: str) -> list | None:
    if model_features:
        try:
            return model_features.split(",")
        except Exception as ex:
            raise errors.InvalidArgumentError(f"{err_msg}: {ex}")

    return []


def _validate_sample_num_rows(sample_num_rows) -> int | None:
    try:
        return int(sample_num_rows)
    except Exception as ex:
        del ex
        return None


def _validate_arg_weight_col(weight_col) -> str:
    return weight_col


def _validate_arg_target_col(target_col, args_as_json_location) -> str:
    if target_col or args_as_json_location:
        return target_col

    raise errors.InvalidArgumentError("Missing required option '--target-col'")


def _validate_arg_use_raw_features(use_raw_features: str) -> bool:
    if use_raw_features and str(use_raw_features).lower() in ["false", "0"]:
        return False
    return True


def _validate_arg_bool(all_explainers) -> bool:
    if all_explainers:
        if isinstance(all_explainers, bool):
            return all_explainers
        elif str(all_explainers).lower() in ["true", "1"]:
            return True

    return False


def _validate_arg_explainers(explainers) -> list[str]:
    if explainers:
        return explainers.split(",")

    # all explainers will be run by default
    return explainers


def _validate_arg_explainer_pars(explainer_pars, explainers) -> list | None:
    if explainer_pars:
        # parse parameters dictionary
        try:
            parameters_dict = ast.literal_eval(explainer_pars)
        except Exception as ex:
            raise errors.InvalidArgumentError(
                f"Unable to parse command line argument with explainer parameters "
                f"- it must be dictionary of explainer parameter dictionaries: "
                f"{ex}\n\n{explainer_pars}"
            )
        if not isinstance(parameters_dict, dict):
            raise errors.InvalidArgumentError(
                f"Command line argument with explainer parameters has wrong type "
                f"'{type(parameters_dict)}' - it must be dictionary of explainer "
                f"parameter dictionaries"
            )

        if parameters_dict:
            new_explainers = []
            if explainers:
                # process all explainers and inject parameters
                for explainer_id in explainers:
                    new_explainers.append(
                        commons.ExplainerToRun(
                            explainer_id=explainer_id,
                            params=parameters_dict.get(explainer_id, None),
                        )
                    )
            else:
                # if explainers to be run were not specified with IDs, then only
                # explainers with parameters will be run
                for explainer_id in parameters_dict:
                    new_explainers.append(
                        commons.ExplainerToRun(
                            explainer_id=explainer_id,
                            params=parameters_dict.get(explainer_id, None),
                        )
                    )

            return new_explainers if new_explainers else explainers

    return explainers


def _validate_arg_results_location(results_location, args_as_json_location) -> str:
    if results_location or args_as_json_location:
        return results_location

    raise errors.InvalidArgumentError("Missing required option '--results-location'")


def _validate_arg_results_formats(results_formats) -> list[str] | None:
    if results_formats:
        try:
            results_formats = results_formats.split(",")
        except Exception as ex:
            raise ValueError(
                f"Unable to parse result formats from the provided "
                f"arguments (must be comma separated list of MIME types): "
                f"{results_formats}, error: {ex}\n{traceback.format_exc()}"
            )

        return results_formats

    return None


def _resolve_config_path(config_path, encryption_key: str = ""):
    if config_path:
        h2o_sonar_config.config.load_and_override(
            config_path=config_path, encryption_key=encryption_key
        )


def _validate_arg_config_path(config_path) -> str:
    if config_path and os.path.isfile(config_path):
        return config_path

    raise errors.InvalidArgumentError(
        f"Invalid path to H2O Sonar config path: '{config_path}'"
    )


def _validate_arg_config_type(config_type):
    if config_type:
        return config_type

    raise errors.InvalidArgumentError(f"Config type is required: '{config_type}'")


def _validate_arg_config_value(config_value):
    if config_value:
        return config_value

    raise errors.InvalidArgumentError(f"Config value is required: '{config_value}'")


def _validate_arg_encryption_key(encryption_key) -> str:
    if encryption_key:
        return str(encryption_key)
    return encryption_key


def _validate_arg_connection_key(
    key, args_as_json_location, is_optional: bool = True
) -> str:
    if key or args_as_json_location:
        return key

    if is_optional:
        return ""

    raise errors.InvalidArgumentError(f"Missing required connection key: '{key}'")


def _validate_arg_file_path(
    path, args_as_json_location, is_optional: bool = True
) -> str:
    if path or args_as_json_location:
        return path

    if is_optional:
        return ""

    raise errors.InvalidArgumentError(f"Missing required path: '{path}'")


def _validate_arg_log_level(log_level):
    if log_level:
        if str(log_level).lower() in LOG_LEVELS:
            return LOG_LEVELS.get(log_level, logging.WARNING)
        else:
            raise errors.InvalidArgumentValueError(
                f"Unknown logging level: '{log_level}' - must be one of "
                f"{list(LOG_LEVELS.keys())}"
            )

    return None


def _make_url_for_file_path(path: str) -> str:
    if path:
        return f"file://{path}"

    return path


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (errors.InvalidArgumentError, errors.InvalidArgumentValueError) as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logging.exception(e)
        sys.exit(2)
