# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
#
# H2O Sonar public API: from explainers listing, to running interpretations,
# getting their results and purging them when they are no longer needed.
#
import json
import logging
import os.path
import pathlib
from typing import Any

import datatable
import pandas

from h2o_sonar import config
from h2o_sonar import loggers
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import datasets
from h2o_sonar.lib.api import explainers as e8s
from h2o_sonar.lib.api import interpretations
from h2o_sonar.lib.api import models as m4s
from h2o_sonar.lib.api import persistences as persist
from h2o_sonar.lib.container import explainer_container
from h2o_sonar.lib.integrations import genai
from h2o_sonar.utils import crypto
from h2o_sonar.utils import io
from h2o_sonar.utils import progress
from h2o_sonar.utils import sampling


try:
    import gc

    HAS_PKG_GC = True
except ImportError:
    HAS_PKG_GC = False


KEYWORD_FILTER_ALL = commons.KEYWORD_FILTER_ALL
KEYWORD_FILTER_ALL_ASTERISK = commons.KEYWORD_FILTER_ALL_ASTERISK


def list_explainers(
    experiment_types: list[str] | None = None,
    explanation_scopes: list[str] | None = None,
    model_meta: m4s.ExplainableModelMeta | None = None,
    keywords: list[str] | None = None,
    explainer_filter: list[commons.FilterEntry] | None = None,
    container: str | explainer_container.ExplainerContainer | None = None,
    args_as_json_location: pathlib.Path | str | None = None,
    portable: bool = False,
    extra_params: dict | None = None,
) -> list[e8s.ExplainerDescriptor]:
    """List explainers by supported experiment types, scores, keywords and other
    criteria. Parameters are used with ``AND`` logical operator.

    Parameters
    ----------
    experiment_types : list[str] | None
      Filter explainers by supported experiment types - regression, binomial
      or multinomial.
    explanation_scopes : list[str] | None
      Filter explainers by supported explanation scopes - local or global.
    model_meta : models.ExplainableModelMeta | None
      Filter explainers by model metadata.
    keywords : list[str] | None
      Filter explainers by keywords.
    explainer_filter : list[commons.FilterEntry] | None
      Filter explainers by generic filter (forward compatible filtering). See
      ``FilterEntry`` for supported search types.
    container : str | explainer_container.ExplainerContainer | None
      Optional explainer container name or instance to be used to run the
      interpretation.
    args_as_json_location : pathlib.Path | str | None
      Load all positional arguments and keyword arguments from JSon file.
      This is useful when input is generated, persisted, repeated and used from CLI
      (which doesn't support all the options).
      IMPORTANT: if this argument is specified, then all other function parameters
      are ignored.
    portable : bool
      If ``True``, then floats (infinity, NaN) and tuples are converted to be
      portable - from strings to max/min values of respective types.
    extra_params : dict | None
      Extra parameters.

    Returns
    -------
    list[explainers.ExplainerDescriptor] :
      Explainers compliant with provided filters.

    """
    del extra_params

    container = resolve_container(container, logger=loggers.SonarPrintLogger())

    # in order to make the parameter resolution ROBUST, */** is not used > manual
    if args_as_json_location:
        args_dict = io.load_list_explainers_args_json(args_as_json_location)

        experiment_types = args_dict.get(io.KEY_EXPERIMENT_TYPES, experiment_types)
        explanation_scopes = args_dict.get(
            io.KEY_EXPLANATION_SCOPES, explanation_scopes
        )
        model_meta = args_dict.get(io.KEY_MODEL_META, model_meta)
        keywords = args_dict.get(io.KEY_KEYWORDS, keywords)
        explainer_filter = args_dict.get(io.KEY_EXPLAINER_FILTER, explainer_filter)

    return container.list_explainers(
        experiment_types=experiment_types,
        explanation_scopes=explanation_scopes,
        model_meta=model_meta,
        keywords=keywords,
        explainer_filter=explainer_filter,
        portable=portable,
    )


def register_explainer(
    explainer_class: type[e8s.Explainer],
    explainer_id: str = "",
    container: str | explainer_container.ExplainerContainer | None = None,
    extra_params: dict | None = None,
) -> str:
    """Register explainer.

    Parameters
    ----------
    explainer_class : Type[explainers.Explainer]
      Explainer class.
    explainer_id : str
      Optional custom explainer ID to be used for explainer identification.
    container : str | explainer_container.ExplainerContainer | None
      Optional explainer container name or instance to be used to run the
      interpretation.
    extra_params : dict | None
      Extra parameters.

    Returns
    -------
    str :
      Explainer ID.

    """
    container = resolve_container(container)

    return container.register_explainer(
        explainer_class=explainer_class,
        explainer_id=explainer_id,
        extra_params=extra_params,
    )


def unregister_explainer(
    explainer_id: str,
    container: str | explainer_container.ExplainerContainer | None = None,
    extra_params: dict | None = None,
) -> str:
    """Unregister explainer.

    Parameters
    ----------
    explainer_id : str
      Custom explainer to be unregistered.
    container : str | explainer_container.ExplainerContainer | None
      Optional explainer container name or instance to be used to run the
      interpretation.
    extra_params : dict | None
      Extra parameters.

    Returns
    -------
    str :
      ID of the unregistered explainer or ``""`` no explainer was unregistered.

    """
    container = resolve_container(container)

    return container.unregister_explainer(
        explainer_id=explainer_id,
        extra_params=extra_params,
    )


def resolve_container(
    container: str | explainer_container.ExplainerContainer | None = None,
    results_location: str | Any = "",
    persistence_api: persist.PersistenceApi | None = None,
    persistence_type: (
        persist.PersistenceType | None
    ) = persist.PersistenceType.file_system,
    do_setup: bool = True,
    logger: loggers.SonarLogger | None = None,
    log_level: int | None = None,
) -> explainer_container.ExplainerContainer:
    """Get explainer container instance to configure, register explainers and
    tune it.

    Parameters
    ----------
    container : str | explainer_container.ExplainerContainer | None
      Optional explainer container name (str) or container instance to be used
      to run the interpretation.
    results_location : pathlib.Path | str | dict | Any | None
      Where to store interpretation results - filesystem (path as string or ``Path``),
      memory (dictionary) or DB. If ``None``, then results are stored to the current
      directory.
    persistence_api : persist.PersistenceApi | None
      Instance of the persistence API allowing to create various persistence types
      (like file-system or DB)
    persistence_type : persist.PersistenceType | None
      Optional choice of the persistence type: file-system (default), in-memory
      or database. This option does not override persistence type in case that
      container is provided.
    do_setup : bool
      Ensure explainer container set up.
    logger : loggers.SonarLogger | None
      Optional custom logger.
    log_level : int
      Optional container and explainers log level.

    Returns
    -------
    explainer_container.ExplainerContainer
      Explainer container instance.

    """
    if not container or (
        container and not isinstance(container, explainer_container.ExplainerContainer)
    ):
        container = explainer_container.ContainerRegistry.registry().resolve_container(
            container
        )

    if do_setup or results_location or persistence_type:
        container.setup(
            results_location=results_location,
            persistence_api=persistence_api,
            persistence_type=persistence_type,
            logger=logger,
            log_level=log_level,
        )

    return container


def _resolve_run_interpretation_formats(
    results_formats: list[str] | None, upload_to
) -> list[str]:
    results_formats = results_formats or [
        commons.MimeType.MIME_HTML,
        commons.MimeType.MIME_JSON,
    ]

    if upload_to and commons.MimeType.MIME_PDF not in results_formats:
        results_formats.append(commons.MimeType.MIME_PDF)

    return results_formats


def run_interpretation(
    dataset: (
        str
        | pathlib.Path
        | datasets.ExplainableDataset
        | commons.ResourceHandle
        | datatable.Frame
        | pandas.DataFrame
        | datasets.ExplainableDatasetHandle
    ),
    model: (
        str
        | pathlib.Path
        | m4s.ExplainableModel
        | commons.ResourceHandle
        | m4s.ExplainableModelHandle
        | Any
        | None
    ) = None,
    models: (
        list[
            str
            | pathlib.Path
            | m4s.ExplainableModel
            | commons.ResourceHandle
            | m4s.ExplainableModelHandle
            | Any
        ]
        | None
    ) = None,
    target_col: str = "",
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
    container: str | explainer_container.ExplainerContainer | None = None,
    results_location: pathlib.Path | str | dict | Any | None = None,
    results_formats: list[str] | None = None,
    persistence_type: persist.PersistenceType = persist.PersistenceType.file_system,
    run_asynchronously: bool = False,
    run_explainers_in_parallel: bool = False,
    progress_callback: progress.AbstractProgressCallbackContext | None = None,
    logger: loggers.SonarLogger | None = None,
    log_level: int = logging.WARNING,
    args_as_json_location: pathlib.Path | str | None = None,
    upload_to: str | config.ConnectionConfig | None = None,
    key: str = "",
    extra_params: dict | None = None,
) -> interpretations.Interpretation:
    """Run interpretation.

    Parameters
    ----------
    dataset :
      Dataset source: explainable dataset instance, datatable
      frame, string (expect path to CSV, .jay or any other file type supported
      by datatable), dictionary (used to construct frame).
    model :
      Path to model (str, Path), explainable model (``ExplainableModel``) or
      an instance of 3rd party model (like Scikit) to interpret.
    models :
      Paths to models (str, Path), explainable models (``ExplainableModel``) or
      an instances of 3rd party models (like Scikit) to interpret.
    target_col : str
      Target column name - must be valid dataset column name.
    explainers : list[str | commons.ExplainerToRun] | None
      Explainer IDs to run within the interpretation or ``ExplainerToRun`` instances
      with explainer parameters. In case of ``None`` or empty list are run all
      default compatible explainers.
    explainer_keywords: list[str] | None
      Run compatible explainers which have given keyword (AND). This setting is used
      *only* in case that ``explainers`` parameter is empty list (or ``None``).
    validset :
      Optional path to validation dataset (str, Path) or datatable Frame instance.
    testset :
      Optional path to test dataset (str, Path) or datatable Frame instance.
    use_raw_features : bool
      ``True`` to use original features (default), ``False`` to force the use
      of transformed features in surrogate models.
    used_features : list | None
      Optional parameter specifying features (dataset columns) used by the model.
      This parameter is used in case that an instance of the model (not
      ``ExplainableModel``) is provided by the user - therefore ``ExplainableModel``'s
      metadata are not available.
    weight_col : str
      Name of the weight column to be used by explainers.
    prediction_col : str
      Name of the predictions column - in case of 3rd party model (standalone MLI).
    drop_cols : list | None
      List of the columns to drop from the interpretation i.e. columns names which
      should not be explained.
    sample_num_rows : int | None
      If ``None``, then automatically sample based on the dataset and RAM size.
      If > 0, then do sample the ``dataset`` to ``sample_num_rows`` number of rows.
      If == 0, then do NOT sample.
    sampler : DatasetSampler | None
      Sampling method (implementation) to be used - see ``h2o_sonar.utils.sampling``
      module (documentation) for available sampling methods. Use a sampler instance
      to use the specific sampling method.
    container : str | explainer_container.ExplainerContainer | None
      Optional explainer container name (str) or container instance to be used
      to run the interpretation.
    results_location : pathlib.Path | str | dict | Any | None
      Where to store interpretation results - filesystem (path as string or ``Path``),
      memory (dictionary) or DB. If ``None``, then results are stored to the current
      directory.
    results_formats : list[str] | None
      Optional list of the result formats (MIME types) to be generated. If ``None``,
      then report in HTML and JSon is created. Supported formats: ``MIME_PDF``,
      ``MIME_HTML`` and ``MIME_JSON``.
    persistence_type : persist.PersistenceType
      Optional choice of the persistence type: file-system (default), in-memory
      or database. This option does not override persistence type in case that
      container is provided.
    run_asynchronously : bool
      ``True`` to run the interpretation asynchronously - the interpretation is run
      synchronously by default (``False``).
    run_explainers_in_parallel : bool
      ``True`` to run explainers in parallel - explainers are run sequentially by
      default.
    progress_callback : progress.AbstractProgressCallbackContext | None
        Optional progress callback context which is stacked atop default logging
        callback in ``h2o_sonar.lib.api.interpretations::Interpretation`` constructor.
    logger : loggers.SonarLogger | None
        Optional custom logger which implements ``loggers.SonarLogger`` interface.
        If ``logger`` is provided, then ``log_level`` is ignored.
    log_level : int
      Optional container and explainers log level.
      If ``logger`` is provided, then ``log_level`` is ignored.
    args_as_json_location : pathlib.Path | str | None
      Load all positional arguments and keyword arguments from JSon file.
      This is useful when input is generated, persisted, repeated and used from CLI
      (which doesn't support all the options).
      IMPORTANT: if this argument is specified, then all other function parameters
      are ignored.
    upload_to : str | config.ConnectionConfig | None
      Upload the interpretation report to the H2O GPT Enterprise in order to talk to
      the report.
    key : str
      Custom interpretation key which must be valid UUID4 string.
    extra_params : dict | None
      Extra parameters.

    Returns
    -------
    interpretations.Interpretation :
      Interpretations instance with the explainer results (references).

    """
    if isinstance(models, dict):
        raise ValueError("Models must be list of models, not dictionary. ")

    # in order to make the parameter resolution ROBUST, */** is not used > manual
    if args_as_json_location:
        args_dict = io.load_run_interpretation_args_json(args_as_json_location)

        dataset = args_dict.get(io.KEY_DATASET, dataset)
        models = args_dict.get(io.KEY_MODELS, models)
        model = args_dict.get(io.KEY_MODEL, model)
        target_col = args_dict.get(io.KEY_TARGET_COL, target_col)
        explainers = args_dict.get(io.KEY_EXPLAINERS, explainers)
        explainer_keywords = args_dict.get(
            io.KEY_EXPLAINER_KEYWORDS, explainer_keywords
        )
        validset = args_dict.get(io.KEY_VALIDSET, validset)
        testset = args_dict.get(io.KEY_TESTSET, testset)
        use_raw_features = args_dict.get(io.KEY_USE_RAW_FEATURES, use_raw_features)
        used_features = args_dict.get(io.KEY_USED_FEATURES, used_features)
        weight_col = args_dict.get(io.KEY_WEIGHT_COL, weight_col)
        prediction_col = args_dict.get(io.KEY_PREDICTION_COL, prediction_col)
        drop_cols = args_dict.get(io.KEY_DROP_COLS, drop_cols)
        sample_num_rows = args_dict.get(io.KEY_SAMPLE_NUM_ROWS, sample_num_rows)
        # container: is not serializable
        results_location = args_dict.get(io.KEY_RESULTS_LOCATION, results_location)
        results_formats = args_dict.get(io.KEY_RESULTS_FORMATS, results_formats)
        if results_formats and isinstance(results_formats, str):
            try:
                results_formats = [rf.strip() for rf in results_formats.split(",")]
            except Exception as ex:
                raise ValueError(
                    f"Unable to parse results formats: {results_formats} - "
                    f"expected comma separated list of MIME types: {ex}"
                )
        persistence_type = args_dict.get(io.KEY_PERSISTENCE_TYPE, persistence_type)
        run_asynchronously = args_dict.get(
            io.KEY_RUN_ASYNCHRONOUSLY, run_asynchronously
        )
        run_explainers_in_parallel = args_dict.get(
            io.KEY_RUN_E_IN_PARALLEL, run_explainers_in_parallel
        )
        upload_to = args_dict.get(io.KEY_UPLOAD_TO, upload_to)
        key = args_dict.get(io.KEY_KEY, key)
        log_level = args_dict.get(io.KEY_LOG_LEVEL, log_level)
        extra_params = args_dict.get(io.KEY_EXTRA_PARAMS, extra_params)
        # override library config if the config path & encryption key are provided
        if args_dict.get(io.KEY_CONFIG_PATH):
            config.config.load_and_override(
                config_path=args_dict.get(io.KEY_CONFIG_PATH),
                encryption_key=crypto.resolve_encryption_key(
                    args_dict.get(io.KEY_ENCRYPTION_KEY)
                ),
            )

    results_formats = _resolve_run_interpretation_formats(results_formats, upload_to)

    if run_asynchronously:
        if container:
            raise ValueError(
                "Container cannot be specified in case of asynchronous interpretation."
            )

        container = resolve_container(explainer_container.AsyncLocalContainer.TYPE_ID)

    container = resolve_container(
        container=container,
        results_location=results_location,
        persistence_type=persistence_type,
        logger=logger,
        log_level=log_level,
    )

    resolved_results_location = container.persistence.getcwl()

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
        sampler=sampler,
        results_location=resolved_results_location,
        results_formats=results_formats,
        explainer_keywords=explainer_keywords,
        run_asynchronously=run_asynchronously,
        progress_callback=progress_callback,
        run_explainers_in_parallel=run_explainers_in_parallel,
        key=key,
        extra_params=extra_params,
    )

    if upload_to:
        (_, collection_url) = upload_interpretation(
            interpretation_result=interpretation,
            connection=upload_to,
        )
        interpretation.result.upload_url = collection_url

    return interpretation


def __load_interpretation(
    interpretation_key: str,
    results_location: str | pathlib.Path | dict,
    persistence_type: persist.PersistenceType = persist.PersistenceType.file_system,
    log_level: int = logging.WARNING,
    extra_params: dict | None = None,
) -> interpretations.Interpretation:
    """Load persisted interpretation (result).

    Parameters
    ----------
    interpretation_key: str
      Interpretation key.
    results_location : str | pathlib.Path | dict
      From where to load the interpretation - filesystem (path as string
      or ``Path``), memory (dictionary) or DB. If ``None``, then results are
      loaded from the current directory.
    persistence_type : persist.PersistenceType
      Optional choice of the persistence type: file-system (default), in-memory
      or database. This option does not override persistence type in case that
      container is provided.
    log_level : int
      Optional container and explainers log level.
    extra_params : dict | None
      Extra parameters.

    Returns
    -------
    interpretations.Interpretation :
      Interpretation instance.

    """
    raise NotImplementedError("TBD")


def list_interpretations(
    results_location: str | pathlib.Path | dict,
    persistence_type: persist.PersistenceType = persist.PersistenceType.file_system,
    container: str | explainer_container.ExplainerContainer | None = None,
    log_level: int = logging.WARNING,
    extra_params: dict | None = None,
) -> list[str]:
    """List interpretations in given results location.

    Parameters
    ----------
    results_location : str | pathlib.Path | dict
      Location used e.g. by ``run_interpretation()`` to store interpretations
      results - filesystem (path as string or ``Path``), memory (dictionary) or DB.
      If ``None``, then results are loaded from the current directory.
    persistence_type : persist.PersistenceType
      Optional choice of the persistence type: file-system (default), in-memory
      or database. This option does not override persistence type in case that
      container is provided.
    container : str | explainer_container.ExplainerContainer | None
      Optional explainer container name (str) or container instance to be used
      to run the interpretation.
    log_level : int
      Optional container and explainers log level.
    extra_params : dict | None
      Extra parameters.

    Returns
    -------
    list[str] :
      Interpretation keys in given results location.

    """
    container = resolve_container(
        container=container,
        results_location=results_location,
        persistence_type=persistence_type,
        log_level=log_level,
    )

    return container.list_interpretations()


def describe_explainer(
    explainer: type[e8s.Explainer] | str,
    portable: bool = False,
) -> dict:
    """Get explainer description.

    Parameters
    ----------
    explainer : Type[e8s.Explainer] | str
       Explainer to describe.
    portable : bool
       If ``True``, then floats (infinity, NaN) and tuples are converted to be
       portable - from strings to max/min values of respective types.

    Returns
    -------
    dict :
     Dictionary with explainer name and parameters.

    """
    if explainer:
        if isinstance(explainer, str):
            container = resolve_container(
                do_setup=True, logger=loggers.SonarPrintLogger()
            )
            return container.get_explainer(explainer_id=explainer).dump(
                portable=portable
            )
    else:
        raise ValueError("Required explainer ID parameter is missing")

    return explainer().as_descriptor(portable=portable).dump()


def __get_explainer_result(
    results_location: str | pathlib.Path | dict,
    explainer_job_key: str,
    explanation_type: str,
    explanation_format: str,
    page_offset: int,
    page_size: int,
    result_format: str,
    explanation_filter: list[commons.FilterEntry],
):
    """Get (global) explainer result.

    Parameters
    ----------
    results_location : str | pathlib.Path | dict
      From where to load the interpretation - filesystem (path as string
      or ``Path``), memory (dictionary) or DB. If ``None``, then results are
      loaded from the current directory.
    explainer_job_key: str
      (Custom) explainer run job key.
    explanation_type: str | None
      Explanation type.
    explanation_format: str | None
      Explanation format supported by the explainer.
    page_offset: int
      Frame's row to be page start.
    page_size: int
      Page size.
    result_format: MimeType
      Mime type for the result (string) serialization like JSon or CSV.
    explanation_filter: list[FilterEntry]
      Additional filter to ge local explanation like feature or class.

    Returns
    -------
    Any :
      Explanation instance in the requested format specified by MIME.

    """
    raise NotImplementedError("TBD")


def __get_explainer_local_result(
    results_location: str | pathlib.Path | dict,
    explainer_job_key: str,
    explanation_type: str,
    explanation_format: str,
    id_column_name: str | None,
    id_column_value: str,
    page_offset: int,
    page_size: int,
    result_format: str,
    explanation_filter: list[commons.FilterEntry],
) -> str | datatable.Frame:
    """Get (local) explainer result - either by loading pre-calculated persisted
    result or by calculating the local explanation on demand.

    Parameters
    ----------
    results_location : str | pathlib.Path | dict
      From where to load the interpretation - filesystem (path as string
      or ``Path``), memory (dictionary) or DB. If ``None``, then results are
      loaded from the current directory.
    explainer_job_key: str
      (Custom) explainer run job key.
    explanation_type: str | None
      Explanation type.
    explanation_format: str | None
      Explanation format supported by the explainer.
    id_column_name : str | None
      ID column name to be used for row search. This parameter allows to search
      for dataset row using any ID column i.e. column which has all its values
      unique and therefore they identify the row. If this parameter is `None`,
      then `id_column_value` is row number.
    id_column_value: str
      Value to search for in the column specified by ``id_column_name``.
      If ``id_column_name`` is `None`, then `id_column_value` is the row number.
    page_offset: int
      Frame's row to be page start.
    page_size: int
      Page size.
    result_format: MimeType
      Mime type for the result (string) serialization like JSon or CSV.
    explanation_filter: list[FilterEntry]
      Additional filter to ge local explanation like feature or class.

    Returns
    -------
    Any :
      Local explanation instance in the requested format specified by MIME type.

    """
    raise NotImplementedError("TBD")


def upload_interpretation(
    interpretation_result: (
        interpretations.Interpretation
        | interpretations.InterpretationResult
        | str
        | pathlib.Path
    ),
    connection: str | config.ConnectionConfig | None,
    collection_name: str = "",
    extra_params: dict | None = None,
) -> tuple[str, str]:
    """Upload interpretation to an LLM like h2oGPTE in order to talk to the
    interpretation report.

    Returns
    -------

    Tuple[str, str] :
      h2oGPT Enterprise collection ID and URL.

    """
    del extra_params

    logger = loggers.SonarPrintLogger()

    # resolve report path
    if isinstance(interpretation_result, pathlib.Path):
        report_path = interpretation_result
    elif isinstance(interpretation_result, str):
        report_path = pathlib.Path(interpretation_result)
    elif isinstance(interpretation_result, interpretations.Interpretation):
        path = interpretation_result.result.get_pdf_report_location()
        if os.path.exists(path):
            report_path = pathlib.Path(path)
        elif os.path.exists(interpretation_result.result.get_html_report_location()):
            report_path = pathlib.Path(
                interpretation_result.result.get_html_report_location()
            )
        else:
            raise ValueError(
                f"Report cannot be uploaded - no report created by the "
                f"interpretation: {interpretation_result}"
            )
    elif isinstance(interpretation_result, interpretations.InterpretationResult):
        path = interpretation_result.get_pdf_report_location()
        if os.path.exists(path):
            report_path = pathlib.Path(path)
        elif os.path.exists(interpretation_result.get_html_report_location()):
            report_path = pathlib.Path(interpretation_result.get_html_report_location())
        else:
            raise ValueError(
                f"Report cannot be uploaded - no report created by the "
                f"interpretation result: {interpretation_result}"
            )
    else:
        raise ValueError(
            f"Unable to resolve report file to uploaded - unsupported interpretation "
            f"result type: {type(interpretation_result)} ({interpretation_result})"
        )

    # TODO resolve connection
    if isinstance(connection, config.ConnectionConfig):
        resolved_connection = connection
    elif isinstance(connection, str):
        resolved_connection = config.config.get_connection(connection)
        if not resolved_connection:
            c_keys = [cc.key for cc in config.config.connections]
            raise ValueError(
                f"Unable to upload report - connection with the key '{connection}' "
                f"(type: {type(connection)}) not found in the H2O Sonar configuration,"
                f"valid connections: "
                f"{', '.join(c_keys)}"
            )
    else:
        raise ValueError(
            f"Unable to resolve upload connection - unsupported connection type: "
            f"{type(connection)} ({connection})"
        )

    (collection_id, collection_url) = genai.H2oGpteRagClient(
        connection=resolved_connection,
        logger=logger,
    ).create_collection(
        doc_paths=[report_path],
        collection_name=collection_name,
    )

    return collection_id, collection_url


def __delete_interpretation(
    results_location: str | pathlib.Path | dict,
    persistence_type: persist.PersistenceType = persist.PersistenceType.file_system,
    log_level: int = logging.WARNING,
    extra_params: dict | None = None,
):
    """Delete interpretation.

    Parameters
    ----------
    results_location : str | pathlib.Path | dict
      Where to destroy the interpretation - filesystem (path as string
      or ``Path``), memory (dictionary) or DB. If ``None``, then interpretation is
      purged in the current directory.
    persistence_type : persist.PersistenceType
      Optional choice of the persistence type: file-system (default), in-memory
      or database. This option does not override persistence type in case that
      container is provided.
    log_level : int
      Optional container and explainers log level.
    extra_params : dict | None
      Extra parameters.

    """
    raise NotImplementedError("TBD")


def add_config_item(
    config_type: str,  # see config.ConfigItemType,
    config_value: dict | str,
    h2o_sonar_config_path: pathlib.Path | str | None = None,
    encryption_key: str = "",
):
    """Add configuration item.

    Parameters
    ----------
    config_type : str
        Configuration item type - see config.ConfigItemType for options.
    config_value : dict | str
        Configuration item value represented either as dictionary or as string with JSon
        serialization of the configuration item. It is expected that the config item is
        NOT encrypted.
    h2o_sonar_config_path : pathlib.Path | str | None
        Path to the H2O Sonar configuration file. If ``None``, then in-memory (current)
        configuration singleton is modified.
    encryption_key : str
        Encryption key to be used for encrypting sensitive data written in the
        configuration. If ``None``, shell environment variable
        ``H2O_SONAR_ENCRYPTION_KEY`` with the encryption key must be set.

    """
    if h2o_sonar_config_path:
        cfg = config.H2oSonarConfig()
        cfg.load_and_override(h2o_sonar_config_path, encryption_key)
    else:
        cfg = config.config

    if config_type == config.ConfigItemType.CONNECTION.name:
        if isinstance(config_value, str):
            try:
                config_value = json.loads(config_value)
            except json.JSONDecodeError:
                raise ValueError(
                    f"Configuration item value is not a valid JSon string: "
                    f"'{config_value}'"
                )
        cfg.add_connection(
            config.ConnectionConfig.from_dict(
                config_dict=config_value, decrypt=False, encryption_key=encryption_key
            )
        )
    elif config_type == config.ConfigItemType.LICENSE.name:
        if isinstance(config_value, str):
            config_value = json.loads(config_value)
        cfg.add_license(
            config.LicenseConfig.from_dict(
                config_dict=config_value, decrypt=False, encryption_key=encryption_key
            )
        )
    else:
        raise ValueError(
            f"Unknown configuration item type: '{config_type}'. Valid values "
            f"are: {', '.join(config.ConfigItemType.__members__.keys())}"
        )

    if h2o_sonar_config_path:
        cfg.save(h2o_sonar_config_path, encryption_key=encryption_key)


def get_config(
    h2o_sonar_config_path: pathlib.Path | str | None = None,
    encryption_key: str = "",
) -> dict:
    """Get configuration item.

    Parameters
    ----------
    h2o_sonar_config_path : pathlib.Path | str | None
        Path to the H2O Sonar configuration file. If ``None``, then in-memory (
        current) configuration singleton will be returned.
    encryption_key : str | None
        Encryption key to be used for decryption of the sensitive data in the
        configuration. If not set, then the encryption key from shell environment
        variable ``H2O_SONAR_ENCRYPTION_KEY`` will be used. If encryption key is not
        provided, then sensitive data will be returned encrypted.

    """
    if h2o_sonar_config_path:
        cfg = config.H2oSonarConfig()
        cfg.load_and_override(h2o_sonar_config_path, encryption_key)
    else:
        cfg = config.config

    return cfg.to_dict(encrypt=False)


def do_gc():
    """Free system resources:

    - shutdowns process pool(s)
    - runs garbage collector
    - clears temporary files

    """
    # shutdown async container process pool (will be re-initialized on demand)
    container = resolve_container(explainer_container.AsyncLocalContainer.TYPE_ID)
    container.gc()

    # Python GC
    if not HAS_PKG_GC:
        commons.raise_opt_import_err("gc")

    gc.collect()
