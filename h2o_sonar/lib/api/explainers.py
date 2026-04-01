# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import abc
import ast
import enum
import importlib
import importlib.metadata
import json
import os
import pickle
import traceback
import uuid
from abc import abstractmethod

import datatable
import toml

from h2o_sonar import errors
from h2o_sonar import loggers as loggers
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import datasets
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import insights
from h2o_sonar.lib.api import models as m4s
from h2o_sonar.lib.api import persistences
from h2o_sonar.lib.api import problems


class ExplainerDescriptor:
    KEY_ID = "id"
    KEY_NAME = "name"
    KEY_DISPLAY_NAME = "display_name"
    KEY_TAGLINE = "tagline"
    KEY_DESCRIPTION = "description"
    KEY_BRIEF_DESCRIPTION = "brief_description"
    KEY_MODEL_TYPES = "model_types"
    KEY_CAN_EXPLAIN = "can_explain"
    KEY_EXPLANATION_SCOPES = "explanation_scopes"
    KEY_EXPLANATIONS = "explanations"
    KEY_PARAMETERS = "parameters"
    KEY_KEYWORDS = "keywords"
    KEY_METRICS_META = "metrics_meta"

    # loosely coupled constructor w/ defaults (backward compatibility & resilience)
    def __init__(
        self,
        id: str,
        name: str = "",
        display_name: str = "",
        tagline: str = "",
        description: str = "",
        brief_description: str = "",
        model_types: list[str] | None = None,
        can_explain: list[str] | None = None,
        explanation_scopes: list[str] | None = None,
        explanations: list[e10s.ExplanationDescriptor] | None = None,
        parameters: list[commons.ConfigItem] | None = None,
        keywords: list[str] | None = None,
        metrics_meta: commons.MetricsMeta | None = None,
        portable: bool = False,
    ) -> None:
        self.id = id
        self.name = name
        self.display_name = display_name
        self.tagline = tagline
        self.description = description
        self.brief_description = brief_description
        self.model_types = model_types or []
        self.can_explain = can_explain or []
        self.explanation_scopes = explanation_scopes or []
        self.explanations = explanations or []
        self.keywords = keywords or []

        if portable:
            if parameters:
                self.parameters = [p.clone().make_portable() for p in parameters]
            if metrics_meta:
                self.metrics_meta = metrics_meta.clone().make_portable()
        else:
            self.parameters = parameters or []
            self.metrics_meta = metrics_meta

    def dump(self, portable: bool = False) -> dict:
        d = {k: v for k, v in vars(self).items()}
        d[ExplainerDescriptor.KEY_EXPLANATIONS] = [a.dump() for a in self.explanations]
        d[ExplainerDescriptor.KEY_PARAMETERS] = [
            a.dump(portable=portable) for a in self.parameters
        ]
        d[ExplainerDescriptor.KEY_METRICS_META] = (
            self.metrics_meta.dump(portable=portable) if self.metrics_meta else []
        )
        return d

    def __str__(self):
        return str(self.dump())

    def clone(self) -> "ExplainerDescriptor":
        return ExplainerDescriptor(
            id=self.id,
            name=self.name,
            display_name=self.display_name,
            tagline=self.tagline,
            description=self.description,
            brief_description=self.brief_description,
            model_types=self.model_types,
            can_explain=self.can_explain,
            explanation_scopes=self.explanation_scopes,
            explanations=self.explanations,
            parameters=self.parameters,
            keywords=self.keywords,
            metrics_meta=self.metrics_meta,
        )

    @staticmethod
    def load(d: dict) -> "ExplainerDescriptor":
        d[ExplainerDescriptor.KEY_EXPLANATIONS] = [
            e10s.ExplanationDescriptor.load(a)
            for a in d[ExplainerDescriptor.KEY_EXPLANATIONS]
        ]
        d[ExplainerDescriptor.KEY_PARAMETERS] = [
            commons.ConfigItem.load(a) for a in d[ExplainerDescriptor.KEY_PARAMETERS]
        ]
        d[ExplainerDescriptor.KEY_METRICS_META] = commons.MetricsMeta.load(
            d[ExplainerDescriptor.KEY_METRICS_META]
        )

        # loosely coupled dictionary to object (backward compatibility & resilience)
        return ExplainerDescriptor(
            id=d.get(ExplainerDescriptor.KEY_ID, ""),
            name=d.get(ExplainerDescriptor.KEY_NAME, ""),
            display_name=d.get(ExplainerDescriptor.KEY_DISPLAY_NAME, ""),
            tagline=d.get(ExplainerDescriptor.KEY_TAGLINE, ""),
            description=d.get(ExplainerDescriptor.KEY_DESCRIPTION, ""),
            brief_description=d.get(ExplainerDescriptor.KEY_BRIEF_DESCRIPTION, ""),
            model_types=d.get(ExplainerDescriptor.KEY_MODEL_TYPES, []),
            can_explain=d.get(ExplainerDescriptor.KEY_CAN_EXPLAIN, []),
            explanation_scopes=d.get(ExplainerDescriptor.KEY_EXPLANATION_SCOPES, []),
            explanations=d.get(ExplainerDescriptor.KEY_EXPLANATIONS, []),
            parameters=d.get(ExplainerDescriptor.KEY_PARAMETERS, []),
            keywords=d.get(ExplainerDescriptor.KEY_KEYWORDS, []),
            metrics_meta=d.get(
                ExplainerDescriptor.KEY_METRICS_META, commons.MetricsMeta()
            ),
        )


class OnDemandExplainKey:
    """On-demand explainer run parameters keys."""

    METHOD = "method"
    ROW = "row"
    FEATURE = "feature"
    CLASS = "class"

    MLI_KEY = "target_mli_key"
    EXPLAINER_JOB_KEY = "target_explainer_job_key"
    EXPLANATION_TYPE = "target_explanation_type"
    FORMAT = "target_format"

    UPDATE_STRATEGY = "update_strategy"


class OnDemandExplainMethod(enum.Enum):
    explain = enum.auto()
    explain_global = enum.auto()
    explain_local = enum.auto()


class ExplainerParam(commons.Param):
    """Explainer parameter declaration."""

    SRC_ANY = "any"
    SRC_CONFIG_OVERRIDES = "config_overrides"
    SRC_CONFIG_OVERRIDES_ERASE = "config_overrides_erase"
    SRC_EXPLAINER_PARAMS = "explainer_params"
    SRC_EVALUATOR_PARAMS = "evaluator_params"

    TAG_SRC_DATASET_COLUMN_NAMES = "SOURCE_DATASET_COLUMN_NAMES"
    TAG_SRC_DATASET_TEXT_COLUMN_NAMES = "SOURCE_DATASET_TEXT_COLUMN_NAMES"

    def __init__(
        self,
        param_name: str,
        param_type: commons.ExplainerParamType | commons.EvaluatorParamType,
        description: str = "",
        comment: str = "",
        default_value: bool | str | float = "",
        value_min: float = 0.0,
        value_max: float = 0.0,
        predefined: list | None = None,
        tags: list | None = None,
        category: str = "",
        src: str = "",
    ):
        """Constructor.

        Parameters
        ----------
        param_name: str
          Parameter name (valid Python identifier).
        param_type: ExplainerParamType
          Parameter type.
        description: str
          Optional human-readable parameter name.
        comment: str
          Optional parameter description.
        default_value: Any
          Optional default value.
        predefined: Optional[list[Any]]
          Predefined values.
        tags: list[str] | None
          Parameter tags.
        category: str
          Parameter category.
        src:
          Source of the parameter with the highest priority
          (from where should be parameter used if there are multiple sources).

        """
        commons.Param.__init__(
            self,
            param_name=param_name,
            param_type=param_type,
            description=description,
            default_value=default_value,
            value_min=value_min,
            value_max=value_max,
            predefined=predefined,
            tags=tags,
        )

        self.comment = comment
        self.category = category
        self.src = src

    def __str__(self) -> str:
        return str(self.as_descriptor().dump())

    def as_descriptor(self, portable: bool = False) -> commons.ConfigItem:
        """Explainer parameter to descriptor conversion."""
        # parameter descriptor is config based to ensure compatibility and reuse
        return commons.ConfigItem(
            name=self.param_name,  # parameter identifier
            description=(
                self.description  # UI: field description w/ fallback to name
                if self.description
                else self.param_name
            ),
            comment=self.comment,  # UI: tooltip w/ fallback to description
            type=self.param_type.name,
            val=self.default_value,
            predefined=self.predefined,
            tags=self.tags,
            min_=self.value_min,
            max_=self.value_max,
            category=self.category,
        )


class ExplainerArgs:
    """Explainer arguments ~ parameter values."""

    def __init__(self, parameters: list[ExplainerParam] = None):
        """Constructor.

        Parameters
        ----------
        parameters: list[ExplainerParam]
          Explainer's parameters declaration.

        """
        self.parameters: list[ExplainerParam] = parameters if parameters else []
        self.args = {}

    def __str__(self):
        return str(self.as_descriptor())

    def as_descriptor(self) -> list:
        """Save parameters as descriptor: `[{'parameter': {'type': 'str'}}]`"""
        return [str(parameter) for parameter in self.parameters]

    def add_parameter(self, param_type: ExplainerParam):
        if not param_type or not isinstance(param_type, ExplainerParam):
            raise ValueError("Explainer parameter type cannot be undefined")
        self.parameters.append(param_type)

    def get(self, param_name: str, default_value=None):
        return self.args.get(param_name, default_value)

    def resolve_params(
        self,
        explainer_params: dict | None = None,
    ):
        """Resolve explainer's ``self.parameters`` (arguments) as follows to
        ``self.args``.

        Parameters
        ----------
        explainer_params: dict | None
          Explainer parameters as dictionary.

        """
        self.args = {}
        # lowest priority
        for parameter in self.parameters:
            if parameter and parameter.param_name:
                if self.args.get(parameter.param_name, None) is None:
                    self.args[parameter.param_name] = parameter.default_value
        # highest priority
        self.from_dict(explainer_params)

    @staticmethod
    def toml_str_to_dict(toml_str: str, logger=None) -> dict:
        if toml_str:
            try:
                return toml.loads(toml_str)
            except Exception as ex:
                if logger:
                    logger.warning(
                        f"Unable to parse TOML arguments: {ex}\n"
                        f"{traceback.format_exc()}",
                    )
        return {}

    @staticmethod
    def json_str_to_dict(json_str: str, logger=None) -> dict:
        if json_str:
            try:
                return json.loads(json_str)
            except Exception as ex:
                if logger:
                    logger.warning(
                        f"Unable to parse JSon arguments: {ex}\n"
                        f"{traceback.format_exc()}",
                    )
        return {}

    def from_config_overrides(
        self, config_overrides: dict, erase: list[str] | None = None
    ) -> dict:
        """Try to get all arguments which are declared as `parameters` from given
        config overrides and set (or overwrite) in `args`.

        Parameters
        ----------
        config_overrides: dict
          Config overrides as dictionary.
        erase: list[str] | None
          Parameters to erase from config overrides.

        """
        return self.from_dict(args_dict=config_overrides, erase=erase)

    def from_dict(self, args_dict: dict, erase: list[str] | None = None) -> dict:
        """Try to get all arguments which are declared as `parameters` from given
        dictionary and set (or overwrite) in `args`. Erase given parameters -
        arguments dictionary is not cloned, but modified.
        """
        if args_dict and self.parameters:
            erase = erase if erase else []
            for parameter in self.parameters:
                if (
                    parameter
                    and parameter.param_name
                    and parameter.param_name in args_dict
                ):
                    try:
                        value = ExplainerArgs._ensure_param_type(
                            parameter=parameter,
                            value=args_dict[parameter.param_name],
                        )
                        self.args[parameter.param_name] = value
                        if parameter.param_name in erase:
                            del args_dict[parameter.param_name]
                    except (ValueError, TypeError, Exception):
                        pass
        return args_dict

    @staticmethod
    def _ensure_param_type(parameter, value):
        if commons.ExplainerParamType.bool == parameter.param_type:
            return bool(value)
        elif commons.ExplainerParamType.int == parameter.param_type:
            return int(value)
        elif commons.ExplainerParamType.float == parameter.param_type:
            return float(value)
        return value

    #
    # local explanations
    #

    @staticmethod
    def resolve_local_paging_args(args: dict, explainer_name: str = "", logger=None):
        """Resolve local explanation paging arguments."""
        page_offset = args[f5s.ExplanationFormat.KEY_ON_DEMAND_PARAMS].get(
            f5s.ExplanationFormat.KEY_PAGE_OFFSET, None
        )
        if page_offset is None:
            raise ValueError(
                f"On-demand {explainer_name}: parameters do not contain page offset"
            )
        elif not isinstance(page_offset, int):
            page_offset = int(page_offset)

        page_size = args[f5s.ExplanationFormat.KEY_ON_DEMAND_PARAMS].get(
            f5s.ExplanationFormat.KEY_PAGE_SIZE, None
        )
        page_size = page_size if page_size else -1
        page_size = page_size if isinstance(page_size, int) else int(page_size)

        if logger:
            logger.debug(
                f"On-demand {explainer_name}: paging {page_offset}[{page_size}]",
            )

        return page_offset, page_size


class ExplainerResult(abc.ABC):
    def __init__(
        self,
        persistence: persistences.ExplainerPersistence,
        explainer_id: str,
        explanation_format: type[f5s.ExplanationFormat] | None,
        explanation: type[e10s.Explanation] | None,
        h2o_sonar_config,
        logger=None,
    ):
        self.persistence: persistences.ExplainerPersistence = persistence
        self.explainer_id = explainer_id
        self.format = explanation_format
        self.explanation = explanation
        self.config = h2o_sonar_config
        self.logger = logger or loggers.SonarPrintLogger()

    @abstractmethod
    def plot(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def data(self, **kwargs) -> datatable.Frame:
        raise NotImplementedError

    @abstractmethod
    def _raw_data(self, **kwargs) -> list | dict | datatable.Frame:
        raise NotImplementedError

    @staticmethod
    def _create_methods_help() -> dict[str, dict]:
        return {"methods": {}}

    @staticmethod
    def _create_method_help() -> dict[str, str | list]:
        return {
            "parameters": [],
            "doc": "",
        }

    @staticmethod
    def _format_parameter_types(*parm_types: type):
        return " | ".join(parm_type.__name__ for parm_type in parm_types)

    @staticmethod
    def _create_method_parameter_help(
        name: str,
        type_name: str,
        default: str = "",
        required: bool = True,
        doc: str = "",
    ) -> dict[str, str | bool]:
        return {
            "name": name,
            "type": type_name,
            "default": default,
            "required": required,
            "doc": doc,
        }

    @classmethod
    def _clazz_param_doc(cls) -> dict[str, str | bool]:
        return cls._create_method_parameter_help(
            name="clazz",
            type_name=cls._format_parameter_types(str),
            required=False,
            default="Selects the default or first class from the set of available "
            "classes in the multinomial classification model.",
            doc="The name of the class in multinomial classification",
        )

    @classmethod
    def _feature_name_param_doc(cls) -> dict[str, str | bool]:
        return cls._create_method_parameter_help(
            name="feature_name",
            type_name=cls._format_parameter_types(str),
            doc="The name of the feature whose data we want to retrieve.",
        )

    @classmethod
    def help(cls) -> dict[str, dict[str, list[dict[str, str | bool]]]]:
        del cls
        return {}

    def params(self) -> dict:
        try:
            i_json = self.persistence.store.load_json(self.persistence.get_json_path())
            return (
                i_json.get("result", {})
                .get("explainers_parameters", {})
                .get(self.explainer_id, {})
            )
        except errors.MliError as err:
            if self.logger:
                self.logger.debug(
                    f"Cannot load explainer parameters from the filesystem for "
                    f"explainer {self.explainer_id}: {err}"
                )
            return {}

    def zip(self, *, file_path):
        self.persistence.store.make_dir_zip_archive(
            src_key=self.persistence.get_explainer_dir(), zip_key=file_path
        )

    def log(self, *, path):
        self.persistence.store.copy_file(
            from_key=self.persistence.get_explainer_log_path(), to_key=path
        )

    def summary(self) -> dict:
        return self.persistence.load_result_descriptor()


class Explainer:
    """Explainer.

    Explainer instance is NOT meant to be reusable i.e. the instance must be created
    using default constructor, initialized using ``setup()`` method and used at most
    once - ``fit()`` method invocation.

    Explainer lifecycle:

    * ``constructor()``
      Explainer instantiation (for external basic/sanity checks, ...). Note that
      explainer constructor executed by H2O Sonar runtime must not have parameters.

    * ``check_compatibility(params) -> bool``
      Explainer check verifying that explainer will be able to explain given model.
      If compatibility check returns ``False`` or raises error, then it will not be run.
      Compatibility check is optional and does not have to be run by the engine.

    * ``setup(params)``
      Set required and optional parameters, configuration, etc.

    * ``fit(X, y)``
      Optional step to train surrogate model(s) or another explainer means. Method gets
      data needed for training/creation/initialization. This step might be skipped
      in case that explainer doesn't need it.

    * ``explain*(X, y) -> [explainer]``
      Actual computation (persistence and upload) of explainer(s) of given data(set).
      Explanation might be provided by value or reference (in case it would not fit
      in memory).

    * ``get_explanation(type, format)``
      Get (cached/persisted) explanations in desired format.

    Attributes
    ----------
    model: Optional[ExplainerModel]
      Instance of ``ExplainerModel`` class which has predict and fit functions
      of the model to be explained. These methods can be used to create
      predictions using the model/scorer.
    persistence: Optional[ExplainerPersistence] = None
      Instance of ``ExplainerPersistence`` class which provides convenient methods
      to persist explainer data e.g. to its working directory.
    params: Optional[CommonExplainerParameters] = None
      Common explainers parameters specified on explainer run like target column
      or columns to drop.
    explainer_params: str
      This explainer specific parameters specified on explainer run.
    logger:
      Explainer's logger.
    config:
      Driverless AI server configuration copy.

    """

    """Explainer version."""
    _explainer_version: str = "1.0.0"

    """Specifies model types (IID, time-series, ...) explained by the explainer."""
    _iid: bool = False  # evaluates IID ML models
    _time_series: bool = False  # evaluates time-series models
    _image: bool = False  # evaluates image ML models
    _unsupervised: bool = False  # evaluates unsupervised ML experiments
    _llm: bool = False  # evaluates LLMs
    _rag: bool = False  # evaluates RAGs (requires retrieval context)

    """Enable the problem types it can support."""
    # y has shape (N,) and is of numeric type, no missing values
    _regression: bool = False
    # y has shape (N,) and can be numeric or string, cardinality 2, no missing values
    _binary: bool = False
    # y has shape (N,) and can be numeric or string, cardinality 3+, no missing values
    _multiclass: bool = False

    """Specifies supported explainer scopes (declaration)."""
    _global_explanation: bool = False
    _local_explanation: bool = False

    """Specify which explanation types this explainer always creates. Type specifier
    must be class which is child of ``Explanation``. Explainer must create an instance
    of declared explanations.
    """
    # Example: [GlobalFeatImpExplanation, PartialDependenceExplanation]
    _explanation_types: list[type[e10s.Explanation]] = []
    """Specify which explanation types this explainer may create."""
    _optional_explanation_types: list[type[e10s.Explanation]] = []

    """Specify explainer parameters."""
    _parameters: list[ExplainerParam] = []

    """Specify metadata of metrics calculated by this explainer."""
    _metrics_meta = commons.MetricsMeta()

    """Specify whether explainer needs a model or explains dataset, 3rd party model
    (represented as dataset predictions column), etc."""
    _requires_model = True

    """Specify whether explainer explains Driverless AI model (``False``) or
    standalone (3rd party model). Standalone explainer requires dataset column (name)
    3rd party model predictions.
    """
    _requires_predict_method = True

    """Specify which datasets and models representations this explainer supports."""
    _supported_dataset_locators = [commons.ResourceLocatorType.local]
    _supported_model_locators = [commons.ResourceLocatorType.local]

    """Whether this explainer uses internally or builds a surrogate model."""
    _generate_surrogate: bool = False

    """Specify Python package dependencies"""
    # Example ["mypackage==1.3.37"]
    #         (will be installed as follows: pip install mypackage==1.3.37)
    _modules_needed_by_name: list[str] = []

    """Specify other explainers this explainer depends on."""
    _depends_on: list[type["Explainer"]] = []

    """Optional name of this explainer to show in interpretations and results."""
    _display_name: str = "Explainer"
    """Optional tagline used to pitch explainer ~ "Just do it!" or "Think different"."""
    _tagline: str = ""
    """Optional description of this explainer to show in interpretations and results."""
    _description: str = NotImplemented
    """Optional brief description of this explainer."""
    _brief_description: str = ""
    """Optional keywords like ``surrogate``, ``nlp`` or ``ootb``."""
    _keywords: list[str] = []
    """Explainer priority in sequential execution: high priority executed first."""
    _priority: float = 0.0

    """Whether to check for stall, should disable if separate server running task."""
    _check_stall = False

    """When set to true, the MLI engine will prepare a predict method based on
    the pre-loaded fitted pipeline.
    """
    _requires_preloaded_predictor = True

    KEYWORD_H2O_SONAR = "h2o-sonar"
    KEYWORD_H2O_MODEL_VALIDATION = "h2o-model-validation"

    KEYWORD_DEFAULT = "run-by-default"
    KEYWORD_NLP = "nlp"
    KEYWORD_COMPLIANCE_TEST = "compliance-test"
    KEYWORD_PROXY = "proxy-explainer"
    KEYWORD_REQUIRES_H2O3 = "requires-h2o3"
    KEYWORD_REQUIRES_OPENAI_KEY = "requires-openai-api-key"
    KEYWORD_MOCK = "mock"
    KEYWORD_TEMPLATE = "template"
    KEYWORD_UNLISTED = "unlisted"
    KEYWORD_IS_FAST = "is_fast"
    KEYWORD_IS_SLOW = "is_slow"
    KEYWORD_LLM = "llm"  # evaluator which evaluates LLMs and/or RAGs
    KEYWORD_EVALUATES_LLM = "evaluates_llm"
    KEYWORD_EVALUATES_RAG = "evaluates_rag"

    # requirements
    KEYWORD_RQ_J = "requires_llm_judge"
    KEYWORD_RQ_P = "requires_prompts"
    KEYWORD_RQ_EA = "requires_expected_answer"
    KEYWORD_RQ_RC = "requires_retrieved_context"
    KEYWORD_RQ_AA = "requires_actual_answer"
    KEYWORD_RQ_C = "requires_constraints"

    # capabilities
    KEYWORD_PREFIX_CAPABILITY = "capability"

    # explainer type ~ explainability domain
    KEYWORD_PREFIX_EXPLAINS = "explains"
    KEYWORD_EXPLAINS_DATASET = f"{KEYWORD_PREFIX_EXPLAINS}-dataset"
    KEYWORD_EXPLAINS_APPROX_BEHAVIOR = f"{KEYWORD_PREFIX_EXPLAINS}-approximate-behavior"
    KEYWORD_EXPLAINS_O_FEATURE_IMPORTANCE = (
        f"{KEYWORD_PREFIX_EXPLAINS}-original-feature-importance"
    )
    KEYWORD_EXPLAINS_T_FEATURE_IMPORTANCE = (
        f"{KEYWORD_PREFIX_EXPLAINS}-transformed-feature-importance"
    )
    KEYWORD_EXPLAINS_FEATURE_BEHAVIOR = f"{KEYWORD_PREFIX_EXPLAINS}-feature-behavior"
    KEYWORD_EXPLAINS_FAIRNESS = f"{KEYWORD_PREFIX_EXPLAINS}-fairness"
    KEYWORD_EXPLAINS_MODEL_DEBUGGING = f"{KEYWORD_PREFIX_EXPLAINS}-model-debugging"
    KEYWORD_EXPLAINS_UNKNOWN = f"{KEYWORD_PREFIX_EXPLAINS}-model"

    EXPLAINERS_PURPOSES = [
        KEYWORD_EXPLAINS_DATASET,
        KEYWORD_EXPLAINS_APPROX_BEHAVIOR,
        KEYWORD_EXPLAINS_O_FEATURE_IMPORTANCE,
        KEYWORD_EXPLAINS_T_FEATURE_IMPORTANCE,
        KEYWORD_EXPLAINS_FEATURE_BEHAVIOR,
        KEYWORD_EXPLAINS_FAIRNESS,
        KEYWORD_EXPLAINS_MODEL_DEBUGGING,
        KEYWORD_EXPLAINS_UNKNOWN,
    ]

    ARG_EXPLAINER_PARAMS = "explainer_params_as_str"

    @staticmethod
    def is_enabled() -> bool:
        """Return ``True`` in case that explainer is enabled, else `False` which will
        make explainer to be completely ignored (unlisted, not loaded, not executed).

        """
        return True

    @staticmethod
    def _description_builder(
        brief: str,
        metrics_meta: commons.MetricsMeta | None = None,
        keywords: list[str] | None = None,
        leaderboard_type: str | None = None,
        parameters: list[ExplainerParam] | None = None,
        extra_insights: str = "",
        to_rst: bool = False,
    ) -> str:
        """Description is meant to be a detailed explanation of the evaluator
        for the user. It should be written in a way that is easy to understand
        and should not contain any technical jargon. It should contain also
        metadata information declared in the class (above).

        Returns
        -------
        str :
            Description of the evaluator in Markdown.

        """
        keywords = keywords or []

        d = "**Evaluator input requirements**:\n\n"

        # requirement table
        check_str = "✓"
        esc_str = "``" if to_rst else "`"
        flag_q = check_str if Explainer.KEYWORD_RQ_P in keywords else " "
        flag_ea = check_str if Explainer.KEYWORD_RQ_EA in keywords else " "
        flag_rc = check_str if Explainer.KEYWORD_RQ_RC in keywords else " "
        flag_aa = check_str if Explainer.KEYWORD_RQ_AA in keywords else " "
        flag_c = check_str if Explainer.KEYWORD_RQ_C in keywords else " "
        if to_rst:
            # reStructuredText
            d = (
                f"{d}"
                f"+----------+-----------------+-------------------+----------------+"
                f"-------------+\n"
                f"| Question "
                f"| Expected answer "
                f"| Retrieved context "
                f"| Actual answer  "
                f"| Conditions  |\n"
                f"+==========+=================+===================+================+"
                f"=============+\n"
                f"| {flag_q}        "
                f"| {flag_ea}               "
                f"| {flag_rc}                 "
                f"| {flag_aa}              "
                f"| {flag_c}           "
                f"|\n"
                f"+----------+-----------------+-------------------+----------------+"
                f"-------------+\n"
            )
        else:
            # Markdown
            d = (
                f"{d}"
                f"| Question "
                f"| Expected Answer "
                f"| Retrieved Context "
                f"| Actual Answer "
                f"| Conditions  "
                f"|\n"
                f"| --- | --- | --- | --- | --- |\n"
                f"| {flag_q} | {flag_ea} | {flag_rc} | {flag_aa} | {flag_c} |\n"
            )

        # brief description + method + see also
        d = f"{d}\n**Description**:\n"
        d = f"{d}\n{brief}\n\n"
        # metrics
        if metrics_meta:
            d = f"{d}**Metrics** calculated by the evaluator:\n"
            for m in metrics_meta.to_list():
                d = f"{d}\n{m.to_md(to_rst=to_rst)}"

        # problems + insights
        if leaderboard_type == "global-llm-bool-leaderboard":
            d = (
                f"{d}\n\n**Problems** reported by the evaluator:\n"
                f"\n"
                f"- If average score of the metric for an evaluated LLM is below the "
                f"threshold, then the evaluator will report a problem for that LLM.\n"
                f"- If test suite has perturbed test cases, then the evaluator will "
                f"report a problem for each perturbed test case and LLM model "
                f"whose metric flipped (moved above/below threshold) after "
                f"perturbation.\n"
                f"\n"
                f"**Insights** diagnosed by the evaluator:\n"
                f"\n"
                f"- Most accurate, least accurate, fastest, slowest, most expensive "
                f"and cheapest LLM models based on the evaluated primary metric.\n"
                f"- LLM models with best and worst context retrieval performance.\n"
                f"- The most difficult test case for the evaluated LLM models, i.e., "
                f"the prompt, which most of the evaluated LLM models had a problem "
                f"answering correctly.{extra_insights}"
            )
        elif leaderboard_type == "global-llm-heatmap-leaderboard":
            d = (
                f"{d}\n\n**Problems** reported by the evaluator:\n"
                f"\n"
                f"- If average score of the metric for an evaluated LLM is below "
                f"the threshold, then the evaluator will report a problem for that "
                f"LLM.\n"
                f"- If test suite has perturbed test cases, then the evaluator will "
                f"report a problem for each perturbed test case and LLM model whose "
                f"metric flipped (moved above/below threshold) after perturbation.\n"
                f"\n"
                f"**Insights** diagnosed by the evaluator:\n"
                f"\n"
                f"- Best performing LLM model based on the evaluated primary metric.\n"
                f"- The most difficult test case for the evaluated LLM models, i.e., "
                f"the prompt, which most of the evaluated LLM models had a problem "
                f"answering correctly.{extra_insights}"
            )
        elif leaderboard_type == "global-llm-classification-leaderboard":
            d = (
                f"{d}\n\n**Problems** reported by the evaluator:\n"
                f"\n"
                f"- If average score of the metric for an evaluated LLM is below "
                f"the threshold, then the evaluator will report a problem for that "
                f"LLM.\n"
                f"- If test suite has perturbed test cases, then the evaluator will "
                f"report a problem for each perturbed test case and LLM model whose "
                f"metric flipped (moved above/below threshold) after perturbation.\n"
                f"\n"
                f"**Insights** diagnosed by the evaluator:\n"
                f"\n"
                f"- Best performing LLM model based on the evaluated primary metric.\n"
                f"- The most difficult test case for the evaluated LLM models, i.e., "
                f"the prompt, which most of the evaluated LLM models had a problem "
                f"answering correctly.{extra_insights}"
            )

        # parameters
        if parameters:
            d = f"{d}\n\nEvaluator **parameters**:\n\n"
            s_li_pad = "   " if to_rst else "    "
            s_q = "``" if to_rst else "`"
            for p in parameters:
                dv = p.default_value if p.default_value else '""'
                d = (
                    f"{d}"
                    f"- {esc_str}{p.param_name}{esc_str} ({p.param_type.name}):\n"
                    f"{s_li_pad}- {p.description}\n"
                    f"{s_li_pad}- Default value: {s_q}{dv}{s_q}\n"
                )

        return d

    @staticmethod
    def load_descriptor(
        descriptor_path: str,
        persistence: persistences.Persistence | None,
    ) -> ExplainerDescriptor:
        data = persistence.load_json(descriptor_path)

        try:
            # BCKWD COMPATIBILITY: on the fly file descriptor fix (increment. migration)
            if data:
                patched = False
                if "description" not in data:
                    data["description"] = ""
                    patched = True
                if "display_name" not in data:
                    data["display_name"] = data.get("name", "")
                    patched = True
                if patched:
                    descriptor = ExplainerDescriptor.load(data)
                    Explainer.save_descriptor(
                        descriptor_path=descriptor_path,
                        descriptor=descriptor,
                        persistence=persistence,
                    )
                    return descriptor
        except Exception as ex:
            del ex

        return ExplainerDescriptor.load(data)

    @staticmethod
    def save_descriptor(
        descriptor_path: str,
        descriptor: ExplainerDescriptor,
        persistence: persistences.Persistence | None,
    ):
        persistence.save_json(key=descriptor_path, data=descriptor.dump())

    @classmethod
    def explainer_version(cls):
        return cls._explainer_version

    @classmethod
    def is_iid(cls) -> bool:
        return cls._iid

    @classmethod
    def is_time_series(cls) -> bool:
        return cls._time_series

    @classmethod
    def is_image(cls) -> bool:
        return cls._image

    @classmethod
    def is_unsupervised(cls) -> bool:
        return cls._unsupervised

    @classmethod
    def is_rag(cls) -> bool:
        return cls._rag

    @classmethod
    def is_llm(cls) -> bool:
        return cls._llm

    @classmethod
    def requires_model(cls) -> bool:
        return cls._requires_model

    @classmethod
    def requires_predict_method(cls) -> bool:
        return cls._requires_predict_method

    @classmethod
    def requires_preloaded_predictor(cls) -> bool:
        return cls._requires_preloaded_predictor

    @classmethod
    def parameters(cls) -> list[ExplainerParam]:
        return cls._parameters

    @classmethod
    def metrics_meta(cls) -> commons.MetricsMeta:
        return cls._metrics_meta

    @classmethod
    def has_model_type_explanations(cls) -> list[str]:
        model_types: list = list()
        if cls._iid:
            model_types.append(commons.ModelTypeExplanation.IID)
        if cls._time_series:
            model_types.append(commons.ModelTypeExplanation.TIME_SERIES)
        if cls._image:
            model_types.append(commons.ModelTypeExplanation.IMAGE)
        if cls._unsupervised:
            model_types.append(commons.ModelTypeExplanation.UNSUPERVISED)
        if cls._llm:
            model_types.append(commons.ModelTypeExplanation.LLM)
        if cls._rag:
            model_types.append(commons.ModelTypeExplanation.RAG)
        return model_types

    @classmethod
    def has_explanations(cls) -> list[str]:
        """Experiment types this explainer explains."""
        explanation_types: list = list()
        if cls._regression:
            explanation_types.append(commons.ExperimentType.regression.name)
        if cls._binary:
            explanation_types.append(commons.ExperimentType.binomial.name)
        if cls._multiclass:
            explanation_types.append(commons.ExperimentType.multinomial.name)
        return explanation_types

    @classmethod
    def priority(cls) -> float:
        """Priority used to order explainers by sequential execution scheduler. Higher
        number, higher priority.

        """
        return cls._priority

    def explains_regression(self) -> bool:
        return self._regression

    def explains_binary(self) -> bool:
        return self._binary

    def explains_multiclass(self) -> bool:
        return self._multiclass

    @classmethod
    def _can_explain_experiment_type(
        cls, experiment_type: commons.ExperimentType
    ) -> bool:
        if commons.ExperimentType.regression == experiment_type:
            return cls._regression
        elif commons.ExperimentType.binomial == experiment_type:
            return cls._binary
        return cls._multiclass

    @classmethod
    def has_explanation_scopes(cls) -> list[str]:
        scopes: list = list()
        if cls._global_explanation:
            scopes.append(commons.ExplanationScope.global_scope.name)
        if cls._local_explanation:
            scopes.append(commons.ExplanationScope.local_scope.name)
        return scopes

    @classmethod
    def has_explanation_types(cls) -> list[type[e10s.Explanation]]:
        """Explanation types supported by the explainer."""
        return cls._explanation_types

    @classmethod
    def can_explain(
        cls,
        model_meta: m4s.ExplainableModelMeta = None,
        experiment_type: commons.ExperimentType = None,
    ) -> bool:
        """Return `True` if explainer can fit either given Driverless AI model's type or
        Driverless AI experiment type.

        """
        if model_meta and experiment_type:
            raise ValueError(
                "At most one filter argument can be used to get explanations "
                "capabilities specification"
            )

        if model_meta:
            return cls._can_explain_experiment_type(model_meta.get_model_type())
        elif experiment_type is not None:
            return cls._can_explain_experiment_type(experiment_type)
        elif not cls.requires_model():
            return True

        return False

    @classmethod
    def explainer_id(cls) -> str:
        return f"{cls.__module__}.{cls.__name__}"

    @classmethod
    def evaluator_id(cls) -> str:
        return f"{cls.__module__}.{cls.__name__}"

    @classmethod
    def depends_on(cls) -> list:
        return cls._depends_on

    @classmethod
    def supports_dataset_locator(cls, locator: commons.ResourceLocatorType) -> bool:
        return locator in cls._supported_dataset_locators

    @classmethod
    def supports_model_locator(cls, locator: commons.ResourceLocatorType) -> bool:
        return locator in cls._supported_model_locators

    @classmethod
    def class_display_name(cls):
        return cls.__name__

    @property
    def display_name(self):
        cls = self.__class__
        return cls.class_display_name()

    @classmethod
    def class_tagline(cls):
        return cls._tagline or f"{cls.class_display_name()}."

    @property
    def tagline(self):
        return self.__class__.class_tagline()

    @classmethod
    def class_brief_description(cls):
        return cls._brief_description or f"{cls.class_display_name()}."

    @property
    def brief_description(self):
        cls = self.__class__
        return cls.class_brief_description()

    @property
    def description(self):
        cls = self.__class__
        return cls.class_description()

    @description.setter
    def description(self, description: str):
        self._description = description

    @classmethod
    def class_description(cls):
        if hasattr(cls, "_description") and cls._description != NotImplemented:
            return cls._description
        elif cls._description != NotImplemented:
            return cls._description
        else:
            return cls.class_display_name()

    @property
    def class_name(self):
        return self.__name__

    @property
    def keywords(self) -> list[str]:
        return self._keywords

    @keywords.setter
    def keywords(self, keywords):
        self._keywords = keywords

    @property
    def working_dir(self) -> str:
        """Working directory path where explainer can store any data it needs."""
        return (
            self.persistence.get_explainer_working_dir() if self.persistence else None
        )

    @property
    def explanations(self) -> dict | None:
        """Explanations created by this explainer."""
        return self._explanations

    @property
    def dependencies(self) -> list[type["Explainer"]]:
        return self._dependencies

    @dependencies.setter
    def dependencies(self, explainer_dependencies: list):
        self._dependencies = explainer_dependencies

    def __init__(self):
        """Constructor must not take any parameters by convention. ``setup()``
        method to be used for explainer instance initialization prior execution.

        """
        # explanation types created by the explainer
        self._explanation_types = (
            self._explanation_types if self._explanation_types else []
        )
        # explainer job key
        self.key: str | None = None
        # interpretation job key
        self.mli_key: str | None = None
        # explainer dependencies on other explainers
        self.dependencies: list = []
        # interpretation parameters
        self.params: commons.CommonInterpretationParams | None = None
        # explainer specific parameters
        self.explainer_params: dict | None = None
        # explainer specific parameters as string
        self.explainer_params_as_str: str | None = None
        # resolved explainer runtime dependencies: parent explainer ID > parent job key
        self.explainer_deps: dict | None = None
        # dataset metadata
        self.dataset_meta: datasets.ExplainableDatasetMeta | None = None
        # validation set metadata
        self.validset_meta: datasets.ExplainableDatasetMeta | None = None
        # test set medatata
        self.testset_meta: datasets.ExplainableDatasetMeta | None = None
        # model
        self.model: m4s.ExplainableModel | m4s.ExplainableModelHandle | None = None
        # models
        self.models = None
        # model metadata
        self.model_meta: m4s.ExplainableModelMeta | None = None
        # explainer persistence
        self.persistence: persistences.ExplainerPersistence | None = None
        # global explanations created by the explainer
        self.explanations_global = None
        # local explanations created by the explainer
        self.explanations_local = None
        # dataset API to create custom explainable datasets needed by this explainer
        self.dataset_api: datasets.DatasetApi | None = None
        # model API to create custom explainable models needed by this explainer
        self.model_api: m4s.ModelApi | None = None
        # runtime (container host) configuration w/ applied cfg overrides (if supported)
        self.config = None
        # explainer logger
        self.logger: loggers.SonarLogger | None = None
        # progress callback
        self.progress_callback = None

        # actual explanations created by the explainer (must be declared)
        self._explanations: dict | None = None

        # problems identified by the explainer
        self._problems: list[problems.ProblemAndAction] | None = None

        # insights identified by the explainer
        self._insights: list[insights.InsightAndAction] | None = None

        # explainer name for logging
        self.log_name: str = ""

        self.args = None

    def __str__(self):
        str_explainer_params = self.explainer_params if self.explainer_params else ""
        result: str = (
            f"{self.explainer_id()}\n"
            f"  name             : {self.display_name}\n"
            f"  description      : {self.description}\n"
            f"  regression       : {self.explains_regression()}\n"
            f"  binomial         : {self.explains_binary()}\n"
            f"  multinomial      : {self.explains_multiclass()}\n"
            f"  scopes           : {self.has_explanation_scopes()}\n"
            f"  explanation types: {self.has_explanation_types()}\n"
            f"  working dir      : {self.working_dir}\n"
            f"  params           : {self.params.dump() if self.params else ''}\n"
            f"  explainer params : {str_explainer_params}\n"
        )
        return result

    def check_required_modules(self, required_modules: set[str] | None = None):
        """Check whether modules specified in ``self._modules_needed_by_name`` are
        imported.

        Parameters
        ----------
        required_modules : list[str] | None
          If defined, then modules specified in the parameter are checked,
          else ``self._modules_needed_by_name`` is checked.

        Returns
        -------
        bool
          ``True`` if all modules are available, ``False`` otherwise.

        """
        required_raw = required_modules
        if not required_modules:
            if self._modules_needed_by_name:
                required_raw = set(self._modules_needed_by_name)

        if required_raw:
            required_modules = set()
            for r in required_raw:
                if "=" in r:
                    required_modules.add(r[: r.index("=")])
                else:
                    required_modules.add(r)

        if required_modules:
            installed = {
                dist.name.lower().replace("-", "_")
                for dist in importlib.metadata.distributions()
            }
            missing = required_modules - installed
            if missing:
                # check whether the missing modules are on Python path
                for m in missing:
                    is_missing = importlib.util.find_spec(m)
                    if is_missing is None:
                        msg = (
                            f"Explainer {self.explainer_id()} is missing the "
                            f"required Python package(s) which must be installed: "
                            f"'{m}'"
                        )
                    else:
                        msg = (
                            f"Explainer {self.explainer_id()} is missing the "
                            f"required Python package(s) which must be installed: "
                            f"'{is_missing}'"
                        )
                    if self.logger:
                        self.logger.warning(msg)
                    return False

        return True

    def _resolve_explainer_params(self):
        explainer_params = ExplainerArgs.json_str_to_dict(self.explainer_params_as_str)
        self.args = ExplainerArgs(self._parameters)
        self.args.resolve_params(explainer_params=explainer_params)

    def _check_compatibility_pckg_err_msg(self, pckg_names: list[str] | str) -> str:
        if isinstance(pckg_names, list):
            pckg_names_fmt = ", ".join([f"'{p}'" for p in pckg_names])
            return (
                f"{self.log_name}: {', '.join(pckg_names_fmt)} Python packages are "
                f"required, but not installed - NOT COMPATIBLE."
            )
        return (
            f"{self.log_name}: '{pckg_names}' Python package is required, "
            f"but not installed - NOT COMPATIBLE."
        )

    def _raise_pckg_err(self, pckg_names: list[str] | str):
        raise ImportError(self._check_compatibility_pckg_err_msg(pckg_names=pckg_names))

    def check_compatibility(
        self,
        params: commons.CommonInterpretationParams | None = None,
        **explainer_params,
    ) -> bool:
        """Explainer's check (based on parameters) verifying that explainer
        will be able to explain a given model. If this compatibility check returns
        ``False`` or raises error, then it will not be run by the engine. This check
        may, but does not have to be performed by the execution engine.

        """
        self.params = params
        self.explainer_params_as_str = explainer_params.get(
            "explainer_params_as_str", None
        )
        self.model = explainer_params.get("model", None)
        self.models = explainer_params.get("models", None)
        self.model_meta = explainer_params.get("model_meta", None)
        self.mli_key = explainer_params.get("mli_key", None)
        self.dataset_meta = explainer_params.get("dataset_meta", None)
        self.validset_meta = explainer_params.get("validset_meta", None)
        self.testset_meta = explainer_params.get("testset_meta", None)
        self.dataset_api = explainer_params.get("dataset_api", None)
        self.model_api = explainer_params.get("model_api", None)
        self.persistence = explainer_params.get("persistence", None)
        self.logger = explainer_params.get("logger", None)

        # target column required for all explainer types except _llm and _unsupervised
        if not self.params.target_col and (
            self._iid or self._time_series or self._image
        ):
            msg = (
                f"Explainer {self.explainer_id()} requires target column "
                f"specified in the interpretation parameters"
            )
            logger = self.logger or loggers.SonarPrintLogger()
            logger.warning(msg)
            raise errors.ExplainerCompatibilityError(msg)

        return True

    def setup(
        self,
        model: m4s.ExplainableModel | m4s.ExplainableModelHandle | None,
        persistence: persistences.ExplainerPersistence,
        models=None,
        key: str = "",
        params: commons.CommonInterpretationParams | None = None,
        explainer_params_as_str: str | None = "",
        dataset_api: datasets.DatasetApi | None = None,
        model_api: m4s.ModelApi | None = None,
        logger: loggers.SonarLogger | None = None,
        **explainer_params,
    ) -> None:
        """Set all the parameters needed to execute ``fit()`` and ``explain()``.

        Parameters
        ----------
        model :
          Explainable model with (fit and) score methods (or ``None`` if 3rd party).
        models :
          (Explainable) models.
        persistence: ExplainerPersistence
          Persistence API allowing (controlled) saving and loading of explanations.
        key: str
          Optional (given) explainer run key (generated otherwise).
        params: CommonInterpretationParams
          Common explainers parameters specified on explainer run.
        explainer_params_as_str: str | None
          Explainer specific parameters in string representation.
        dataset_api : datasets.DatasetApi | None
          Dataset API to create custom explainable datasets needed by this explainer.
        model_api : Optional[m4s.ModelApi]
          Model API to create custom explainable models needed by this explainer.
        logger : loggers.SonarLogger | None
          Logger.
        explainer_params:
          Other explainers RUNTIME parameters, options, and configuration.

        """
        self.model: m4s.ExplainableModel | None = model
        self.models = models
        self.model_meta = (
            model.meta
            if model and not commons.ResourceHandle.is_handle(model)
            else None
        )
        self.dataset_meta = explainer_params.get("dataset_entity", None)
        self.persistence: persistences.ExplainerPersistence = persistence
        persistence.make_explainer_working_dir()
        if not key:
            self.key: str = str(uuid.uuid4())
        else:
            self.key = key
        if explainer_params and explainer_params.get("mli_key"):
            self.mli_key = explainer_params.get("mli_key")
        self.params = params
        if self.params:
            self.params.drop_cols = self.params.drop_cols or []
        self.explainer_params_as_str = explainer_params_as_str
        self.config = self.config or explainer_params.get("config", None)
        self.logger = logger

        # other RUNTIME explainer parameters:
        self.explainer_params = explainer_params

    def run_fit(self, X, y=None, **kwargs):
        """Build explainer and explainer prerequisites.

        This is a method invoked by explainer execution engine (can add code to
        be executed before/after ``fit()`` overridden by child classes).

        Parameters
        ----------
        X :
          Data frame.
        y :
          Labels.

        """
        self.fit(X, y, **kwargs)
        return self

    def fit(self, X, y=None, **kwargs):
        """Optionally, build/train explainer (model) and explainer prerequisites. This
        method implementation to be overridden by child class (this class
        implementation). It may be empty if explainer doesn't have to be built.

        Parameters
        ----------
        X :
          Data frame.
        y :
          Labels.

        """
        pass

    def run_explain(self, X, y, explanations_types: list = None, **kwargs) -> dict:
        """Execute explainer to calculate (persist and upload) explanations(s) of
        a given model.

        This method invokes explainer implementation of ``explain()`` and then
        performs explanation verifications and eventual later actions. It is
        invoked by explainer execution engine (can add code to be executed
        before/after ``explain()`` overridden by child classes).

        Explanation might be provided by value or reference (in case it would not
        fit in memory).

        Parameters
        ----------
        X :
          Data frame.
        y :
          Labels.
        explanations_types: list[Type[Explanation]]
          Explanation types to build. All will be built if empty list or ``None``
          provided. Get all supported types using ``has_explanation_types()``.

        Returns
        -------
        list[Explanation]:
          Explanations.

        """
        explanations_types = (
            explanations_types if explanations_types else self._explanation_types
        )

        #
        # explain
        #

        explanations = self.explain(
            X=X, y=y, explanations_types=explanations_types, **kwargs
        )

        #
        # validate
        #

        if not explanations:
            raise errors.MliError(
                f"Explain method of {self.explainer_id()} explainer did not build "
                f"any explainer results in interpretation {self.key}",
            )
        if len(explanations_types) > len(explanations):
            raise errors.MliError(
                "Required explanation types and explanations were not built "
                f"(expected {len(explanations_types)}, got {len(explanations)} "
                f"~ {explanations}) created by {self.explainer_id()} in interpretation "
                f"{self.key}",
            )
        # proxy explainers explanations are provided by parent explainers
        explanations = [
            ex
            for ex in explanations
            if ex.explanation_type() != e10s.ProxyExplanation.explanation_type()
        ]
        # IMPROVE perform additional explanations verifications

        #
        # actions
        #

        if self._explanations is None:
            self._explanations = {}
        if explanations:
            for explanation in explanations:
                self._explanations[explanation.explanation_type()] = explanation

        Explainer.save_descriptor(
            descriptor_path=self.persistence.get_result_descriptor_file_path(),
            descriptor=self.as_descriptor(),
            persistence=self.persistence.store,
        )

        model_problems = [p.to_dict() for p in self._problems] if self._problems else []
        self.persistence.save_problems(model_problems)

        model_insights = [p.to_dict() for p in self._insights] if self._insights else []
        self.persistence.save_insights(model_insights)

        # explanations itself are stored by explainers (ref/value) memory
        return self._explanations

    def explain(self, X, y=None, explanations_types: list = None, **kwargs) -> list:
        """Invoke this method to calculate and persist global,
        local or both type of explanation(s) for given data(set). This method
        implementation to be overridden by child class (this class implementation).
        This method is responsible for the calculations, build and persistence of
        explanations.

        X : datatable.Frame
          Dataset frame.
        y :
          Labels.
        explanations_types: list[Type[Explanation]]
          Optional explanations to be built. All will be built if empty list
          or ``None`` provided. Get all supported types using
          ``has_explanation_types()``.

        Returns
        -------
        list[Explanation]:
          Explanations descriptors.

        """
        raise NotImplementedError

    def run_explain_local(self, X, y=None, **kwargs) -> list:
        """Execute explainer to calculate (persist and upload) local explanation(s).

        This method invokes explainer implementation ``explain_local()`` and then
        performs explanations verifications and eventual subsequent actions. It is
        invoked by  explainer execution engine (can add code to be executed
        before/after ``explain_local()`` overridden by child classes).

        Parameters
        ----------
        X :
          Data frame.
        y :
          Labels.

        Returns
        -------
        list[Explanation]:
          Explanations.

        """
        return self.explain_local(X, y, **kwargs)

    def explain_local(self, X, y=None, **kwargs) -> list:
        """Execute explainer to calculate on-demand local explanations. This method is
        expected to be overridden if explainer doesn't pre-compute local
        explanations. Default implementation just returns local instance explanations
        computed by ``explain()`` method.

        X :
          Data frame.
        y :
          Labels.

        Returns
        -------
        list[Explanation]:
          Explanations.

        """
        explanations: list = self.explain(X, y, explanations_types=None, **kwargs)
        local_explanations: list = list()
        for explanation in explanations:
            if not explanation.is_global():
                local_explanations.append(explanation)

        if not local_explanations:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support local explanations"
            )
        return local_explanations

    def run_explain_global(self, X, y=None, **kwargs) -> list:
        """Execute explainer to calculate (persist and upload) global explanation(s).

        This method invokes explainer implementation ``explain_global()`` and
        then performs explanations verifications and eventual subsequent actions. It is
        invoked by explainer execution engine (can add code to be executed
        before/after ``explain_global()`` overridden by child classes).

        Parameters
        ----------
        X :
          Data frame.
        y :
          Labels.

        Returns
        -------
        list[Explanation]:
          Explanations.

        """
        return self.explain_global(X, y, **kwargs)

    def explain_global(self, X, y=None, **kwargs) -> list:
        """Execute explainer to calculate on-demand global explanations. This method is
        expected to be overridden if explainer doesn't pre-compute global
        explanations and/or needs to update global explanation after initial
        computation. Default implementation just returns global instance explanations
        computed by ``explain()`` method.

        X :
          Data frame.
        y :
          Labels.

        Returns
        -------
        list[Explanation]:
          Explanations.

        """
        explanations: list = self.explain(X, y, explanations_types=None, **kwargs)
        global_explanations: list = list()
        for explanation in explanations:
            if explanation.is_global():
                global_explanations.append(explanation)

        if not global_explanations:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support global explanations"
            )
        return global_explanations

    def explainer_params_as_dict(self) -> dict | None:
        if self.explainer_params_as_str:
            try:
                return ast.literal_eval(self.explainer_params_as_str)
            except Exception:
                pass
        return dict()

    def validate_explanations(self) -> bool:
        """Optional method which can be used to verify integrity of explanations.

        Returns
        -------
        bool:
          Returns ``True`` if explanations are valid, ``False`` otherwise.

        """
        # IMPROVE check:
        # - >= 1 explainer
        # - 1>= formats per explainer
        # - explanations declared by explainer fit
        # - formats declared by explanations fit
        return bool(self.explanations)

    def add_problem(self, problem: problems.ProblemAndAction):
        """Add an evaluated/interpreted model(s) problem identified by ``explain()``
        method.

        Parameters
        ----------
        problem : problems.ProblemAndAction
          Model problem to be added.

        """
        if problem:
            self._problems = self._problems or []
            self._problems.append(problem)

    def add_insight(self, insight: insights.InsightAndAction):
        """Add an evaluated/interpreted model(s) insight identified by ``explain()``
        method.

        Parameters
        ----------
        insight : insights.InsightAndAction
          Insight to be added.

        """
        if insight:
            self._insights = self._insights or []
            self._insights.append(insight)

    def explain_problems(self) -> list[problems.ProblemAndAction]:
        """Determine (calculate or get persisted problems identified by ``explain()``
        method) interpreted/evaluated model(s) problems.

        Returns
        -------
        list[ProblemAndAction]:
          Interpreted/evaluated model(s) problems.

        """
        return self._problems or []

    def explain_insights(self) -> list[insights.InsightAndAction]:
        """Determine (calculate or get persisted insights identified by ``explain()``
        method) interpreted/evaluated model(s) problems.

        Returns
        -------
        list[InsightAndAction]:
          Interpreted/evaluated model(s) insights.

        """
        return self._insights or []

    def get_explanations(self, explanation_types: list) -> list:
        """Get instance explanations representations in given format.

        Parameters
        ----------
        explanation_types: list[Type[Explanation]]
          Explanation type to return - must be one of explanations declared
          (supported) by explainer. Returns all supported explanations if ``None``
          or empty.

        Returns
        -------
        list[Explanation]:
          Explanations by value or reference.

        """
        explanation_types = (
            explanation_types if explanation_types else self.has_explanation_types()
        )

        result = list()

        for explanation_type in explanation_types:
            if explanation_type not in self.has_explanation_types():
                raise ValueError(
                    f"Explanation type '{explanation_type}' is not supported"
                    f"by the explainer {self.__class__.__name__}"
                )

            result.append(self._explanations[explanation_type])

        return result

    @staticmethod
    def load(explainer_path: str | None = None):
        """Load pickled explainer snapshot."""
        path = os.path.join(
            explainer_path,
            persistences.ExplainerPersistence.FILE_EXPLAINER_PICKLE,
        )
        with open(path, "rb") as explainer_pickle_path:
            return pickle.load(explainer_pickle_path)

    def save(self, explainer_path: str | None = None):
        """Save explainer snapshot pickle."""
        del explainer_path

        # TODO to be implemented w/o problems w/ pickle failing due to acquired locks
        """
        path = os.path.join(
            explainer_path, ExplainerPersistence.FILE_EXPLAINER_PICKLE
        )
        self.model = None
        self.logger = None
        with open(path, "wb") as explainer_pickle_path:
            pickle.dump(self, explainer_pickle_path)
        """
        pass

    def as_descriptor(
        self, runtime_view: bool = False, portable: bool = False
    ) -> ExplainerDescriptor:
        """Explainer descriptor as PROTO entity.

        Parameters
        ----------
        runtime_view: bool
          Not all descriptor fields (like parameters declaration) are needed in
          runtime (for instance they are needed before running explainer), therefore
          they might be skipped in runtime view.
        portable : bool
          If ``True``, then floats (infinity, NaN) and tuples are converted to be
          portable - from strings to max/min values of respective types.

        Returns
        -------
        ExplainerDescriptor:
          Explainer descriptor.

        """
        # explanations types w/o and w/ formats (computed as explainer goes)
        if self._explanations:
            es: list = [e.as_descriptor() for e in list(self._explanations.values())]
        else:
            es = [e.as_class_descriptor() for e in self._explanation_types]

        parameters = (
            []
            if runtime_view
            else [
                parameter.as_descriptor(portable=portable)
                for parameter in self.parameters()
            ]
        )

        metrics_meta = (
            self.metrics_meta() if not runtime_view else commons.MetricsMeta()
        )

        return ExplainerDescriptor(
            id=self.explainer_id(),
            name=self.display_name,
            display_name=self._display_name,
            tagline=self.tagline,
            description="" if runtime_view else self.description,
            brief_description="" if runtime_view else self.brief_description,
            model_types=self.has_model_type_explanations(),
            can_explain=self.has_explanations(),
            explanation_scopes=self.has_explanation_scopes(),
            explanations=es,
            parameters=parameters,
            keywords=self.keywords,
            metrics_meta=metrics_meta,
            portable=portable,
        )

    def report_progress(self, progress: float, message: str = "", precision: int = 1):
        """Report explainer progress in [0, 1] range and message (`""` removes
        previous message, `None` keeps previous message).

        """
        if self.progress_callback:
            if message:
                message = f"{self._display_name} - {message}"
            else:
                message = (
                    f"{self._display_name} - "
                    f"progress {round(progress * 100.0, precision)}%"
                )

            self.progress_callback.set_progress(progress, message)

    def destroy(self, **destroy_params):
        """Override to release resources created by the explainer (DB entities, files,
        running processes, ...) depending on explainer runtime/container.

        """
        pass

    #
    # helper methods
    #

    def create_explanation_workdir_archive(
        self, display_name: str = "", display_category: str = ""
    ) -> e10s.WorkDirArchiveExplanation:
        """Easily create working directory archive with ZIP of explanations
        representations.

        Parameters
        ----------
        display_name: str
          Display name e.g. to be used for naming tile in UI.
        display_category: str
          Display category e.g. to be used for naming tab in UI.

        """
        work_explanation = e10s.WorkDirArchiveExplanation(
            explainer=self,
            display_name=(
                display_name if display_name else f"{self.display_name} ZIP Archive"
            ),
            display_category=(
                display_category
                if display_category
                else e10s.Explanation.DISPLAY_CAT_CUSTOM
            ),
        )
        work_explanation.add_format(
            f5s.WorkDirArchiveZipFormat(
                explanation=work_explanation,
                persistence=self.persistence.store,
            )
        )

        return work_explanation

    def get_result(self) -> type[ExplainerResult] | None:
        return None


Explainer.expected_custom_class = Explainer


class SurrogateExplainer(Explainer, abc.ABC):
    """Surrogate model explainer."""

    _generate_surrogate = True

    KEYWORD_SURROGATE = "surrogate"

    def __init__(self):
        Explainer.__init__(self)
        self.surrogate_model = None

    def run_predict(self, X, y=None, **kwargs):
        """Surrogate explainer provides predict method allowing to get predictions
        from the surrogate model.

        This is method invoked by explainer execution engine (can add code to
        be executed before/after ``fit()`` overridden by child classes).

        Parameters
        ----------
        X :
          Data frame.
        y :
          Labels.

        """
        self.predict(X, y, **kwargs)
        return self

    @abstractmethod
    def predict(self, X, y=None, **kwargs):
        """
        Surrogate explainer provides predict method allowing to get predictions
        from the surrogate model. This method to be overridden by child classes.

        Parameters
        ----------
        X :
          Data frame.
        y :
          Labels.

        """
        raise NotImplementedError


class ExplainerRegistry:
    """Explainer registry provides list of available OOTB and (registered)
    explainers.

    """

    # SINGLETON container registry
    __registry = None
    # SINGLETON: secret key to prevent instantiation using constructor
    __singleton_secret_key = object()

    @classmethod
    def registry(cls):
        if not cls.__registry:
            cls.__registry = ExplainerRegistry(cls.__singleton_secret_key)
        return cls.__registry

    def __init__(self, singleton_create_key):
        # singleton: constructor instantiation protection
        assert singleton_create_key == ExplainerRegistry.__singleton_secret_key, (
            "Explainer registry must be created using registry() method"
        )

        self._explainer_classes: dict[str, Explainer] = dict()

    def register(self, explainer_class, explainer_id: str = "") -> str:
        if not explainer_class:
            raise ValueError("Unable to register explainer as explainer class is None")
        explainer_id = explainer_id or explainer_class().explainer_id()

        # if explainer ID is used, then it's override with new explainer class
        self._explainer_classes[explainer_id] = explainer_class

        return explainer_id

    def unregister(self, explainer_id: str) -> str:
        return explainer_id if self._explainer_classes.pop(explainer_id, "") else ""

    def get_class(self, explainer_id) -> type[Explainer] | None:
        return self._explainer_classes.get(explainer_id, None)

    def list_explainers(self) -> dict:
        return self._explainer_classes.copy()

    def load(self):
        """Load registry from configuration."""
        raise NotImplementedError

    def save(self):
        raise NotImplementedError
