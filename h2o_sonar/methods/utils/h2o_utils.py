# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
"""H2O-3 utilities with proper cluster lifecycle management.

This module manages H2O-3 cluster lifecycle to prevent memory leaks:

1. Automatic Shutdown: Shuts down old clusters when new ones are created
   to prevent multiple Java VMs from accumulating
2. Comprehensive Cleanup: Removes frames, models, and forces GC to
   immediately reclaim Java heap memory
3. Test Safety: Fixtures properly clean up after each test and shut down
   at session end

Memory Management Best Practices:
- Session-scoped fixtures reuse clusters when possible
- Function-scoped fixtures shut down previous clusters before creating new ones
- clean_up_h2o3() removes frames AND models, then forces GC
- kill_h2o3() shuts down clusters and resets tracking state

"""

import os
import pickle
import socket
import traceback

import datatable
import numpy
import pandas

from h2o_sonar import config as cfg
from h2o_sonar import loggers
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import persistences
from h2o_sonar.methods.core import _data
from h2o_sonar.methods.nlp import _lm_text
from h2o_sonar.methods.nlp import _tokenizers


try:
    import h2o

    HAS_H2O = True
except ImportError:
    HAS_H2O = False


DEFAULT_H2O3_PORT = 54321


def h2o_find_free_port(port: int = DEFAULT_H2O3_PORT, max_attempts: int = 10):
    """Find free port for H2O-3 server.

    Parameters
    ----------
    port : int
        Starting port. If 0, then any/random free port is found.
    max_attempts : int
        Maximum number of attempts.

    Returns
    -------
    int :
        Free port.

    """
    if not port:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", port))  # s.bind(("", 0)) to find any free port
            return s.getsockname()[1]  # get port number if 0 was used ^

    for attempt in range(max_attempts):
        port = port + attempt
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                pass

    raise OSError(
        f"Could not find free port for H2O-3 after {max_attempts} attempts "
        f"starting from port {port}."
    )


def h2o_init(h2o3_config: dict | None = None):
    """Ensure connection to an H2O instance.

    Parameters
    ----------
    h2o3_config : dict
        H2O configuration as dictionary with keys defined in
        ``h2oaxi.config.H2o3Config`` e.g. port or memory.

    """
    if not HAS_H2O:
        commons.raise_opt_import_err("h2o")

    h2o3_config = h2o3_config or {}

    # set port and IP in H2O Sonar config - this config is read by explainers
    cfg.config.h2o_port = h2o3_config.get(cfg.H2o3Config.KEY_PORT, h2o_find_free_port())
    cfg.config.h2o_host = h2o3_config.get(cfg.H2o3Config.KEY_HOST, cfg.config.h2o_host)

    h2o.init(
        ip=cfg.config.h2o_host,
        port=cfg.config.h2o_port,
        as_port=True,  # deprecated
        min_mem_size=h2o3_config.get(cfg.H2o3Config.KEY_MIN_MEM_SIZE, None),
        max_mem_size=h2o3_config.get(cfg.H2o3Config.KEY_MAX_MEM_SIZE, None),
    )
    h2o.connect(
        ip=cfg.config.h2o_host,
        port=cfg.config.h2o_port,
    )


def start_h2o3(h2o3_config_overrides: dict | None = None, logger=None):
    """Start H2O-3 on a local H2O instance."""
    if not HAS_H2O:
        commons.raise_opt_import_err("h2o")

    logger = logger or loggers.SonarPrintLogger()

    h2o3_config = {
        cfg.H2o3Config.KEY_PORT: h2o_find_free_port(),
        cfg.H2o3Config.KEY_MIN_MEM_SIZE: cfg.H2o3Config.DEFAULT_MIN_MEM_SIZE,
        cfg.H2o3Config.KEY_MAX_MEM_SIZE: cfg.H2o3Config.DEFAULT_MAX_MEM_SIZE,
    }

    if h2o3_config_overrides:
        for k in h2o3_config_overrides:
            if k not in cfg.H2o3Config.KEYS:
                logger.warning(
                    f"WARNING when starting H2O-3: invalid H2O-3 configuration "
                    f"key: '{k}'. Valid keys: {cfg.H2o3Config.KEYS}"
                )

            h2o3_config[k] = h2o3_config_overrides[k]

    # shutdown existing cluster if one is running to prevent cluster accumulation
    if is_h2o3_running():
        logger.info(
            f"Shutting down existing H2O-3 cluster before starting new cluster "
            f"with configuration: "
            f"min_mem={h2o3_config[cfg.H2o3Config.KEY_MIN_MEM_SIZE]}, "
            f"max_mem={h2o3_config[cfg.H2o3Config.KEY_MAX_MEM_SIZE]}"
        )
        kill_h2o3()

    h2o_init(h2o3_config)

    # mark ~ auto started ~ that H2O-3 was started by H2O Sonar so cleanup will work
    ensure_h2o3_running.__auto_started__ = True

    return h2o3_config


def is_h2o3_running() -> bool:
    """Determine if H2O-3 instance is running or not."""
    if not HAS_H2O:
        commons.raise_opt_import_err("h2o")

    if h2o.cluster():
        return h2o.cluster().is_running()
    return False


def ensure_h2o3_running(
    auto_start=True, h2o3_config_overrides: dict | None = None, logger=None
):
    """Ensure that H2O-3 server is running - either by starting it or connecting
    to it. H2O-3 server is started even if the ``auto_start`` is not enable in
    H2O Sonar configuration.

    Parameters
    ----------
    auto_start : bool
        If True, the H2O-3 server is started if it is not running.
    h2o3_config_overrides : dict
        H2O-3 configuration overrides.
    logger :
        Logger.

    """
    if not HAS_H2O:
        commons.raise_opt_import_err("h2o")

    from h2o import exceptions

    if not is_h2o3_running():
        if auto_start or cfg.config.h2o_auto_start:
            start_up_method = (
                f"started based on the argument OR configuration "
                f"(arg={auto_start} and config={cfg.config.h2o_auto_start})"
            )
            start_h2o3(h2o3_config_overrides=h2o3_config_overrides, logger=logger)
            ensure_h2o3_running.__auto_started__ = True
        else:
            raise RuntimeError(
                f"Attempt to ensure/start running H2O-3 server failed, "
                f"because auto_start was not set in argument and is disabled in config"
                f"(arg={auto_start} and config={cfg.config.h2o_auto_start})"
            )
    elif not h2o.connection():
        start_up_method = (
            f"connect to the running H2O-3 server "
            f"(arg={auto_start} and config={cfg.config.h2o_auto_start})"
        )
        connect_to_h2o3()
    else:
        start_up_method = (
            f"did NOTHING - H2O-3 should be running and H2O-3 connection established "
            f"(arg={auto_start} and config={cfg.config.h2o_auto_start})"
        )

    if not is_h2o3_running():
        raise exceptions.H2OConnectionError(
            f"No H2O-3 is running, either set auto_start or provide a correct "
            f"connection parameters to create new connection to running H2O-3. "
            f"Ensure H2O-3 method used to start the server: "
            f"{start_up_method}"
        )


ensure_h2o3_running.__auto_started__ = False


def clean_up_h2o3(logger=None):
    """Clean up H2O-3 data to free memory.

    Removes all H2O frames AND models from H2O cluster, then forces Java
    garbage collection to immediately reclaim memory. This is more aggressive
    than relying on automatic GC which may delay memory reclamation.

    If H2O is not installed, this function is a no-op and returns immediately.

    """
    if not HAS_H2O:
        return  # no-op if h2o not installed - nothing to clean up

    logger = logger or loggers.SonarPrintLogger()

    # avoid purge of the cluster if it was not started by the H2O Sonar
    if ensure_h2o3_running.__auto_started__:
        cluster = h2o.cluster()
        if cluster and cluster.is_running():
            # remove all frames
            h2o.remove_all()

            # remove all models explicitly (not covered by remove_all)
            try:
                models_resp = h2o.api("GET /3/Models")
                if models_resp and "models" in models_resp:
                    model_list = models_resp["models"]
                    for model_dict in model_list:
                        model_id = model_dict.get("model_id", {}).get("name")
                        if model_id:
                            try:
                                h2o.api(f"DELETE /3/Models/{model_id}")
                            except Exception as exm:
                                logger.warning(
                                    f"WARNING: failed to delete H2O-3 model "
                                    f"{model_id} on H2O-3 test clean-up: {exm}"
                                )
                                pass  # best-effort cleanup
            except Exception as ex:
                logger.warning(
                    f"WARNING: failed to purge H2O-3 models on test clean-up: {ex}"
                )
                pass  # best-effort cleanup

            # force Java garbage collection
            try:
                h2o.api("POST /3/GarbageCollect")
            except Exception as exgc:
                logger.warning(
                    f"WARNING: failed to run H2O-3's GC on test clean-up: {exgc}"
                )
                pass  # best-effort GC


def kill_h2o3():
    """Shutdown H2O-3 cluster if it was started by H2O Sonar.

    If H2O is not installed, this function is a no-op and returns immediately.

    """
    if not HAS_H2O:
        return  # no-op if h2o not installed - nothing to shut down

    # avoid shutdown of the cluster if it was not started by the H2O Sonar
    if ensure_h2o3_running.__auto_started__:
        cluster = h2o.cluster()
        if cluster and cluster.is_running():
            cluster.shutdown()
            # reset flag after shutdown to ensure clean state
            ensure_h2o3_running.__auto_started__ = False


def connect_to_h2o3():
    """Connect to H2O-3 server."""
    if not HAS_H2O:
        commons.raise_opt_import_err("h2o")

    h2o.connect(ip=cfg.config.h2o_host, port=cfg.config.h2o_port)


def assert_is_type(var, *types, **kwargs):
    """Safe type assert with (cythonized code) bug workaround."""
    if not HAS_H2O:
        commons.raise_opt_import_err("h2o")

    from h2o.utils import typechecks

    try:
        typechecks.assert_is_type(var, *types, **kwargs)
    except IndentationError as ex:
        if "unindent does not match any outer indentation level" in str(ex):
            # This is robust workaround of H2O-3's ``assert_is_type()`` bug.
            #
            # ``assert_is_type()`` does NOT work when it is called from the cythonized
            # code - an inner function in html.utils.typechecks attempts to parse
            # the stacktrace on an invalid type, however, the stacktrace doesn't have
            # format the function expects (and token ``assert_`` in particular),
            # therefore it crashes. In other words, type mismatch is detected, but
            # the method fails when it is gathering detailed information for
            # the ``H2OTypeError`` construction.
            from h2o import exceptions as h2o_exceptions

            raise h2o_exceptions.H2OTypeError(
                message=(
                    f"Attempt to upload non-supported data type ({type(var)} "
                    f"- it must be str, PersistentData or H2OFrame."
                )
            )

        raise ex


def upload_data(data):
    """Uploads the data located at a given path or returns it
    if it's a h2o.H2OFrame, else an exception is raised.

    Parameters
    ----------
    data : str, h2o_sonar.core.data.PersistedData
      Path to the data or h2o.H2OFrame.

    Returns
    -------
    h2o.H2OFrame :
      H2O-3 frame.

    """
    if not HAS_H2O:
        commons.raise_opt_import_err("h2o")

    assert_is_type(
        data,
        str,
        _data.PersistedData,
        h2o.H2OFrame,
        message=f"{type(data)} data cannot be uploaded due to incompatible type",
    )

    if not isinstance(data, h2o.H2OFrame):
        data_location = data if isinstance(data, str) else data.data_location
        upload_config = {} if isinstance(data, str) else data.upload_config
        if "header" not in upload_config:
            upload_config["header"] = 1

        return h2o.import_file(path=data_location, parse=True, **upload_config)

    return data


def _safe_to_h2oframe(data, column_names: list[str]):
    """Ensure that RAW column names (e.g. columns containing ") will not be broken
    on conversion to H2O-3 H2OFrame.

    """
    if not HAS_H2O:
        commons.raise_opt_import_err("h2o")

    frame = h2o.H2OFrame(data)
    frame.names = column_names
    return frame


def to_h2oframe(data, labels=None):
    """Convert H2OFrames.

    Parameters
    ----------
    data: h2o.H2OFrame | list | list of lists | pandas.DataFrame | datatable.Frame
    | str
      Data to be transformed to the frame.
    labels: h2o.H2OFrame | list | pandas.DataFrame | None
      Optional frame labels.

    Returns
    -------
    Tuple[h2o.H2OFrame, bool] :
      Frame created from the data and indicator whether the data were transformed.

    """
    if not HAS_H2O:
        commons.raise_opt_import_err("h2o")

    if isinstance(data, (str, _data.PersistedData)):
        return upload_data(data), True

    if isinstance(data, h2o.H2OFrame) and labels is not None:
        raise ValueError("Can't pass H2OFrame *and* labels at the same time.")

    if not isinstance(data, h2o.H2OFrame):
        empty_labels = (
            labels.empty if isinstance(labels, pandas.DataFrame) else not labels
        )

        same_type = isinstance(data, type(labels))
        if not empty_labels and (not same_type or len(data) != len(labels)):
            raise ValueError("Data and labels need to be of the same type and size.")

        if isinstance(data, datatable.Frame):
            pd_data = data.to_pandas()
            h2o_data = _safe_to_h2oframe(pd_data, list(data.names))
        else:
            if isinstance(data, pandas.DataFrame):
                h2o_data = _safe_to_h2oframe(data, list(data.columns))
            else:
                h2o_data = h2o.H2OFrame(data)
        if not empty_labels:
            # TODO does this need to be removed after cbind?
            h2o_labels = h2o.H2OFrame(labels)
            h2o_data = h2o_data.cbind(h2o_labels)

        return h2o_data, True

    return data, False


def h2o_to_dt(X, col_names=None):
    """Convert H2OFrames to datatables."""

    h2o_to_pd = X.as_data_frame(use_pandas=True, header=False)
    h2o_dt = datatable.Frame(h2o_to_pd)
    fix_names = col_names if col_names else X.names
    if len(fix_names) != len(h2o_dt.names):
        raise ValueError(
            f"The names list has length {len(fix_names)} ({fix_names}), "
            f"while the Frame has {len(h2o_dt.names)} columns ({h2o_dt.names})"
        )
    h2o_dt.names = fix_names

    return h2o_dt


_SURROGATE_TOKEN_PICKLE = "surrogate_tokenizer.pickle"
_SURROGATE_LM_PICKLE = "surrogate_lm.pickle"
_LM_TFIDF = "Linear Model + TF-IDF"


def _load_nlp_method_pickle(mli_path: str, method=_SURROGATE_TOKEN_PICKLE):
    path = persistences.Persistence.make_key(mli_path, "nlp", method)
    with open(path, "rb") as method_pickle:
        return pickle.load(method_pickle)


def _save_nlp_method_pickle(
    persistence: persistences.Persistence,
    path: str,
    method=_SURROGATE_TOKEN_PICKLE,
    method_obj=None,
):
    persistence.make_dir(persistences.Persistence.make_key(path, "nlp"))
    path = os.path.join(path, "nlp", method)
    with open(path, "wb") as method_pickle:
        return pickle.dump(method_obj, method_pickle, protocol=5)


def preprocess_h2o3_data(
    frame_for_h2o3: datatable.Frame,
    contains_text_transformers: bool,
    explainer_work_path,
    config: cfg.H2oSonarConfig,
    sanitization_utils,
    num_labels: int,
    features_metadata: dict,
    meta_keys,
    persistence: persistences.Persistence,
    logger,
    vectorizer_path: str = "",
    lm_path: str = "",
    target_col: str = "",
    dropped_cols: list[str] | None = None,
    remove_preprocessed: bool = True,
):
    """Preprocess data for H2O-3.

    Parameters
    ----------
    frame_for_h2o3 : datatable.Frame
      Frame to be preprocessed.
    target_col : str
      Optional target column name.
    dropped_cols : list[str] | None
      Optional dropped columns list.
    contains_text_transformers : bool
      Indicator of text transformers presence in the model.
    features_metadata : dict
      Model features metadata.
    meta_keys :
      Keys to be used with features metadata dictionary.
    num_labels : int
      Number of target labels (regression vs. binomial vs. multinomial).
    explainer_work_path : str
      Explainer working directory path.
    vectorizer_path : str
      Optional vectorizer path.
    lm_path : str
      Optional linear model path.
    config :
      Global H2O Sonar configuration with config overrides already applied - if
      supported by the container runtime.
    sanitization_utils :
      Feature names sanitization utils.
    remove_preprocessed :
      Control removal of preprocessed columns.
    persistence : persistences.Persistence
      Persistence store.
    logger :
      Logger.

    Returns
    -------
    datatable.Frame :
      Frame for H2O-3.

    """
    # check if model uses text features
    logger.debug("[H2O-3 PREPROCESS] getting features metadata")

    if contains_text_transformers:
        if dropped_cols:
            features_metadata[meta_keys.KEY_TEXT_FEATURES] = [
                i
                for i in features_metadata[meta_keys.KEY_TEXT_FEATURES]
                if i not in dropped_cols
            ]
            if len(features_metadata[meta_keys.KEY_TEXT_FEATURES]) == 0:
                contains_text_transformers = False

    if (
        features_metadata
        and meta_keys.KEY_TEXT_FEATURES in features_metadata
        and contains_text_transformers
        and features_metadata[meta_keys.KEY_TEXT_FEATURES]
    ):
        text_features = features_metadata[meta_keys.KEY_TEXT_FEATURES]
        text_features = sanitization_utils.sanitize_names(text_features)
        text_features = list(set(text_features).intersection(set(frame_for_h2o3.names)))

        if text_features and config.mli_nlp_tokenizer:
            vectorized_text_features = None
            vectorizer = None
            if vectorizer_path:
                try:
                    vectorizer = _load_nlp_method_pickle(vectorizer_path)
                    vectorized_text_features = vectorizer.transform(frame_for_h2o3)
                except (AttributeError, FileNotFoundError) as ex:
                    logger.error(
                        f"Legacy vectorizer no longer supported: {ex}:"
                        f"\n{traceback.format_exc()}",
                    )

            # no path OR obsolete legacy vectorizer (is rewritten as fallback)
            if not vectorized_text_features:
                english_stop_words = config.mli_nlp_append_to_english_stop_words
                vectorizer = _tokenizers.TOKENIZERS[config.mli_nlp_tokenizer](
                    text_features_idx=text_features,
                    max_features=(
                        None
                        if config.mli_nlp_tokenizer == _LM_TFIDF
                        else config.mli_nlp_surrogate_tokens
                    ),
                    min_df=config.mli_nlp_min_df,
                    max_df=config.mli_nlp_max_df,
                    min_ngram=config.mli_nlp_min_ngram,
                    max_ngram=config.mli_nlp_max_ngram,
                    stop_words=config.mli_nlp_stop_words,
                    append_to_english_stop_words=english_stop_words,
                    use_stop_words=config.mli_nlp_use_stop_words,
                )

                vectorizer.fit(frame_for_h2o3)
                _save_nlp_method_pickle(
                    persistence=persistence,
                    path=explainer_work_path,
                    method_obj=vectorizer,
                )
                vectorized_text_features = vectorizer.transform(frame_for_h2o3)

            cols_to_preserve = frame_for_h2o3.names
            if remove_preprocessed:
                cols_to_preserve = [
                    elem for elem in frame_for_h2o3.names if elem not in text_features
                ]

            frame_for_h2o3 = datatable.cbind(
                vectorized_text_features,
                frame_for_h2o3[:, list(cols_to_preserve)],
            )

            if config.mli_nlp_surrogate_tokenizer == _LM_TFIDF:
                if num_labels <= 2:  # VLM does not support multinomial yet
                    linear_model = None
                    if lm_path:
                        try:
                            linear_model = _load_nlp_method_pickle(
                                mli_path=lm_path, method=_SURROGATE_LM_PICKLE
                            )
                        except (AttributeError, FileNotFoundError) as ex:
                            logger.error(
                                f"Legacy linear model + vectorizer no longer "
                                f"supported: {ex}:"
                                f"\n{traceback.format_exc()}",
                            )

                    if not linear_model:
                        vocab = numpy.array([])
                        for txt_feat in text_features:
                            vocab = numpy.append(
                                vocab,
                                numpy.array(
                                    vectorizer.get_transformed_token_names(
                                        vectorizer.per_feature_vectorizers[txt_feat],
                                        txt_feat,
                                    )
                                ),
                            )
                        linear_model = _lm_text.LinearModel(
                            target=target_col,
                            n_classes=num_labels,
                            text_feature=text_features,
                            run_tokenizer=False,
                            vocab=vocab.flatten(),
                            tokenizer=vectorizer,
                        )
                        features = vocab.tolist() + [target_col]
                        linear_model.fit(frame_for_h2o3[:, features])
                        _save_nlp_method_pickle(
                            persistence=persistence,
                            path=explainer_work_path,
                            method=_SURROGATE_LM_PICKLE,
                            method_obj=linear_model,
                        )
                        text_imp_dict = linear_model.get_most_important_words()
                        text_imp_dict = dict(
                            sorted(
                                text_imp_dict.items(),
                                key=lambda x: x[1],
                                reverse=True,
                            )
                        )
                        text_imp_words = list(text_imp_dict)
                        frame_for_h2o3 = frame_for_h2o3[
                            :, text_imp_words + cols_to_preserve
                        ]
                else:
                    logger.info(
                        "VLM does not support multinomial experiments - H2O Sonar "
                        "surrogate models will use text tokens based on TF-IDF.",
                    )

    return frame_for_h2o3
