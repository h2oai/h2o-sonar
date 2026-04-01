# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import copy
import enum
import inspect
import pathlib
import socket
import string
import sys
import traceback
import uuid
from typing import Any

from matplotlib import colors
from matplotlib import pyplot


# defaults
DEFAULT_USER: str = "h2o-sonar"

# misc
ENABLED_STRINGS: list[str] = ["true", "1", "yes", "enabled", "on"]

# explainer keyword-based filtering
KEYWORD_FILTER_ALL: str = "all"  # match all explainers
KEYWORD_FILTER_ALL_ASTERISK: str = "*"  # match all explainers (shortcut)

# error handling
ERROR_LLM_HOST = "INTERNAL ERROR"  # legacy model host internal error prefix
ERROR_MODEL_HOST = f"MODEL HOST {ERROR_LLM_HOST}"  # error on test lab completion
LEGACY_LLM_HOST = f"LLM HOST {ERROR_LLM_HOST}"  # legacy test lab completion error


class Keyword:
    """Keyword."""

    def __init__(
        self,
        key: str,
        name: str,
        description: str,
    ):
        self.key = key
        self.name = name
        self.description = description


class KeywordGroup:
    """Keyword groups."""

    def __init__(
        self,
        prefix: str,
        name: str,
        description: str,
        keywords: list[Keyword] | None = None,
    ):
        self.prefix = prefix
        self.name = name
        self.description = description
        self.keywords = keywords

    def is_member(self, keywords: list[str]) -> bool:
        """Check if the entity (evaluator, explainer, method) with given keywords
        is a member of this keyword group.

        """
        for k in keywords:
            if k.startswith(self.prefix):
                return True

        return False


class KeywordGroups:
    """Keyword groups."""

    def __init__(self, groups: list[KeywordGroup] = None):
        self.groups = groups or []

    def add_group(self, group: KeywordGroup):
        self.groups.append(group)

    def get_group(self, prefix: str) -> KeywordGroup | None:
        for group in self.groups:
            if group.prefix == prefix:
                return group
        return None


class Branding(enum.Enum):
    """Branding."""

    # H2O Sonar: Responsible AI library.
    H2O_SONAR = enum.auto()
    # Eval Studio: modular and extensible studio for LLM evaluation and benchmarks
    EVAL_STUDIO = enum.auto()


class ResourceLocatorType(enum.Enum):
    """Resource locator types."""

    local = enum.auto()
    handle = enum.auto()


class ExperimentType(enum.Enum):
    """Experiment types."""

    regression = enum.auto()
    binomial = enum.auto()
    multinomial = enum.auto()


class ModelTypeExplanation:
    # explainers which support IID models
    IID: str = "iid"
    # explainers which support TS models
    TIME_SERIES: str = "time_series"
    # explainers which support image models
    IMAGE: str = "image"
    # explainers which support unsupervised learning models
    UNSUPERVISED: str = "unsupervised"
    # explainers which support LLM models
    LLM: str = "llm"
    # explainers which support RAG-hosted LLM models
    RAG: str = "rag"


class LlmModelHostType(enum.Enum):
    # LLMs hosted by a service (like OpenAI)
    SERVICE = enum.auto()
    # LLMs hosted as base models by a RAG (like h2oGPTe)
    RAG = enum.auto()


class ExplainerFilter:
    """List explainers filters"""

    # explainers which support IID models
    IID: str = ModelTypeExplanation.IID
    # explainers which support TS models
    TIME_SERIES: str = ModelTypeExplanation.TIME_SERIES
    # explainers which support image models
    IMAGE: str = ModelTypeExplanation.IMAGE
    # explainers which support unsupervised learning models
    UNSUPERVISED: str = ModelTypeExplanation.UNSUPERVISED
    # explainers which require predict method (model)
    REQUIRES_PREDICT: str = "requires_predict_method"
    # explainer ID to get particular explainer descriptor
    EXPLAINER_ID = "explainer_id"
    # explainer's blueprint input name
    BLUEPRINT_INPUT_NAME = "blueprint_input_name"


class ConfigItem:
    KEY_NAME = "name"
    KEY_DESCRIPTION = "description"
    KEY_COMMENT = "comment"
    KEY_TYPE = "type"
    KEY_VAL = "val"
    KEY_PREDEFINED = "predefined"
    KEY_TAGS = "tags"
    KEY_MIN = "min_"
    KEY_MAX = "max_"
    KEY_CATEGORY = "category"

    # loosely coupled constructor w/ defaults (backward compatibility & resilience)
    def __init__(
        self,
        name: str = "",
        description: str = "",
        comment: str = "",
        type: str = "",
        val: Any = "",
        predefined: list | None = None,
        tags: list | None = None,
        min_: float = 0.0,
        max_: float = 0.0,
        category: str = "",
    ) -> None:
        self.name = name
        self.description = description
        self.comment = comment
        self.type = type
        self.val = val
        self.predefined = predefined or []
        self.tags = tags or []
        self.min_ = min_
        self.max_ = max_
        self.category = category

    def make_portable(self) -> "ConfigItem":
        self.min_ = port_float(self.min_, portable=True)
        self.max_ = port_float(self.max_, portable=True)
        return self

    def dump(self, portable: bool = False) -> dict:
        d = {k: port_float(v, portable=portable) for k, v in vars(self).items()}
        return d

    def clone(self) -> "ConfigItem":
        return ConfigItem(
            self.name,
            self.description,
            self.comment,
            self.type,
            self.val,
            self.predefined,
            self.tags,
            self.min_,
            self.max_,
            self.category,
        )

    @staticmethod
    def load(d: dict) -> "ConfigItem":
        # loosely coupled dictionary to object (backward compatibility & resilience)
        return ConfigItem(
            name=d.get(ConfigItem.KEY_NAME, ""),
            description=d.get(ConfigItem.KEY_DESCRIPTION, ""),
            comment=d.get(ConfigItem.KEY_COMMENT, ""),
            type=d.get(ConfigItem.KEY_TYPE, ""),
            val=d.get(ConfigItem.KEY_VAL, ""),
            predefined=d.get(ConfigItem.KEY_PREDEFINED, []),
            tags=d.get(ConfigItem.KEY_TAGS, []),
            min_=d.get(ConfigItem.KEY_MIN, 0.0),
            max_=d.get(ConfigItem.KEY_MAX, 0.0),
            category=d.get(ConfigItem.KEY_CATEGORY, ""),
        )


@enum.unique
class ParamType(enum.Enum):
    bool = enum.auto()  # boolean
    int = enum.auto()  # integer
    float = enum.auto()  # float
    str = enum.auto()  # string
    list = enum.auto()  # selection from predefined list of items
    multilist = enum.auto()  # multi-selection from predefined list of items
    customlist = enum.auto()  # list of strings provided by user, w/o predefined values
    dict = enum.auto()  # dictionary


@enum.unique
class InterpretationParamType(enum.Enum):
    bool = enum.auto()
    int = enum.auto()
    float = enum.auto()
    str = enum.auto()
    list = enum.auto()
    multilist = enum.auto()
    customlist = enum.auto()
    dict = enum.auto()
    any = enum.auto()


@enum.unique
class ExplainerParamType(enum.Enum):
    """Explainer parameters."""

    bool = enum.auto()
    int = enum.auto()
    float = enum.auto()
    str = enum.auto()
    list = enum.auto()
    multilist = enum.auto()
    customlist = enum.auto()
    dict = enum.auto()


@enum.unique
class EvaluatorParamType(enum.Enum):
    """Evaluators parameters."""

    # evaluators to explainers adaptor

    bool = ExplainerParamType.bool
    int = ExplainerParamType.int
    float = ExplainerParamType.float
    str = ExplainerParamType.str
    list = ExplainerParamType.list
    multilist = ExplainerParamType.multilist
    customlist = ExplainerParamType.customlist
    dict = ExplainerParamType.dict

    def describe(self):
        return self.name, self.value


def is_ncname(s: str) -> bool:
    if s:
        ncname_chars = string.ascii_letters + string.digits + "_-"
        for c in s:
            if c not in ncname_chars:
                return False
        return True
    return False


class Param:
    """Generic parameter used as (predecessor) of library, interpretation and
    explainer parameters.

    """

    def __init__(
        self,
        param_name: str,
        param_type: ParamType | InterpretationParamType | ExplainerParamType,
        description: str = "",
        default_value="",
        value_min: float = 0.0,
        value_max: float = 0.0,
        predefined: list | None = None,
        tags: list | None = None,
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
        default_value: Any
          Optional default value.
        predefined: list[Any] | None
          Predefined values.
        tags: list[str] | None
          Parameter tags.

        """
        self.param_name = param_name
        self.param_type = param_type
        self.description = description
        self.default_value = default_value
        self.value_min = value_min
        self.value_max = value_max
        self.predefined = predefined if predefined else []
        self.tags = tags if tags else []

    def __str__(self) -> str:
        return str(self.as_descriptor().dump())

    def as_descriptor(self) -> ConfigItem:
        """Explainer parameter to descriptor conversion."""
        # parameter descriptor is config based to ensure compatibility and reuse
        return ConfigItem(
            name=self.param_name,
            description=self.description if self.description else self.param_name,
            comment=self.description,
            type=self.param_type.name,
            val=self.default_value,
            predefined=self.predefined,
            tags=self.tags,
            min_=self.value_min,
            max_=self.value_max,
            category="",
        )


class ExplainerParamKey:
    # explainer parameters
    KEY_KWARGS = "pk"
    # procedure kwarg keys
    KEY_USER = "user"
    KEY_E_ID = "explainer_id"
    KEY_ON_DEMAND = "on_demand_explanation"
    KEY_ON_DEMAND_PARAMS = "on_demand_params"
    KEY_ON_DEMAND_MLI_KEY = "on_demand_mli_key"
    KEY_RUN_KEY = "run_key"
    KEY_E_JOB_KEY = "explainer_job_key"
    KEY_E_DEPS = "explainer_dependencies"
    KEY_PARAMS = "params"
    KEY_LEGACY_I_PARAMS = "legacy_i_params"
    KEY_E_PARAMS = "explainer_params"
    KEY_MODEL = "model"
    KEY_EXPERIMENT_TYPE = "experiment_type"  # r/b/m
    KEY_MODEL_TYPE = "model_type"  # ftrl/dt/...
    KEY_DATASET = "dataset"
    KEY_VALIDSET = "validset"
    KEY_TESTSET = "testset"
    KEY_I_DATA_PATH = "interpretation_data_path"
    KEY_DESCR_PATH = "result_descriptor_path"
    KEY_FEATURES_META = "features_metadata"
    # params public API keys
    KEY_ALL_EXPLAINERS_PARAMS = "explainers_params"
    KEY_WORKER_NAME = "worker_name"


class ExplanationScope(enum.Enum):
    """Explanation scope."""

    local_scope = enum.auto()
    global_scope = enum.auto()


class MimeType:
    MIME_DATATABLE = "application/vnd.h2oai.datatable.jay"
    MIME_JSON_DATATABLE = "application/vnd.h2oai.json+datatable.jay"
    MIME_JSON_CSV = "application/vnd.h2oai.json+csv"
    MIME_MODEL_PIPELINE = "application/vnd.h2oai.pipeline+zip"
    MIME_CSV = "text/csv"
    MIME_JSON = "application/json"
    MIME_TEXT = "text/plain"
    MIME_HTML = "text/html"
    MIME_PDF = "application/pdf"
    MIME_MARKDOWN = "text/markdown"
    MIME_EVALSTUDIO_MARKDOWN = "application/vnd.h2oai-evalstudio-leaderboard.markdown"
    MIME_DOCX = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    MIME_ZIP = "application/zip"
    MIME_PNG = "image/png"
    MIME_JPG = "image/jpeg"
    MIME_SVG = "image/svg+xml"
    # http://www.w3.org/Protocols/rfc1341/4_Content-Type.html
    MIME_IMAGE = "image/xyz"

    EXT_DATATABLE = "jay"
    EXT_JSON = "json"
    EXT_CSV = "csv"
    EXT_TEXT = "txt"
    EXT_HTML = "html"
    EXT_MARKDOWN = "md"
    EXT_DOCX = "docx"
    EXT_ZIP = "zip"
    EXT_PNG = "png"
    EXT_JPG = "jpg"
    EXT_SVG = "svg"

    @staticmethod
    def ext_for_mime(mime: str):
        # TODO IMPROVE dict rewrite w/ default value > exception
        if mime == MimeType.MIME_DATATABLE:
            return MimeType.EXT_DATATABLE
        if mime == MimeType.MIME_JSON_DATATABLE:
            return MimeType.EXT_JSON
        if mime == MimeType.MIME_JSON_CSV:
            return MimeType.EXT_JSON
        elif mime == MimeType.MIME_MODEL_PIPELINE:
            return MimeType.EXT_ZIP
        elif mime == MimeType.MIME_CSV:
            return MimeType.EXT_CSV
        elif mime == MimeType.MIME_JSON:
            return MimeType.EXT_JSON
        elif mime == MimeType.MIME_TEXT:
            return MimeType.EXT_TEXT
        elif mime == MimeType.MIME_HTML:
            return MimeType.EXT_HTML
        elif mime in [MimeType.MIME_MARKDOWN, MimeType.MIME_EVALSTUDIO_MARKDOWN]:
            return MimeType.EXT_MARKDOWN
        elif mime == MimeType.MIME_DOCX:
            return MimeType.EXT_DOCX
        elif mime == MimeType.MIME_ZIP:
            return MimeType.EXT_ZIP
        elif mime == MimeType.MIME_PNG:
            return MimeType.EXT_PNG
        elif mime == MimeType.MIME_JPG:
            return MimeType.EXT_JPG
        elif mime == MimeType.MIME_SVG:
            return MimeType.EXT_SVG

        raise ValueError(f"Cannot provide extension for unknown MIME '{mime}'")


class FilterEntry:
    KEY_FILTER_BY = "filter_by"
    KEY_VALUE = "value"

    # loosely coupled constructor w/ defaults (backward compatibility & resilience)
    def __init__(self, filter_by: str = "", value=None) -> None:
        self.filter_by = filter_by
        self.value = value

    def dump(self) -> dict:
        d = {k: v for k, v in vars(self).items()}
        return d

    def clone(self) -> "FilterEntry":
        return FilterEntry(self.filter_by, self.value)

    @staticmethod
    def load(d: dict) -> "FilterEntry":
        # loosely coupled dictionary to object (backward compatibility & resilience)
        return FilterEntry(
            filter_by=d.get(FilterEntry.KEY_FILTER_BY, ""),
            value=d.get(FilterEntry.KEY_VALUE, None),
        )


class MetricMeta:
    """Evaluation/explanation metric metadata."""

    KEY_KEY = "key"
    KEY_DISPLAY_NAME = "display_name"
    KEY_DATA_TYPE = "data_type"
    KEY_DISPLAY_FORMAT = "display_value"
    KEY_DESCRIPTION = "description"
    KEY_VALUE_RANGE = "value_range"
    KEY_VALUE_ENUM = "value_enum"
    KEY_HIGHER_IS_BETTER = "higher_is_better"
    KEY_THRESHOLD = "threshold"
    KEY_IS_PRIMARY_METRIC = "is_primary_metric"
    KEY_PARENT_METRIC = "parent_metric"
    KEY_EXCLUDE = "exclude"

    DATA_TYPE_SECONDS = "seconds"

    # avoid INF w/ effective infinity for the metric value range
    EFFECTIVE_INF_FLOAT = 1_234_567.89
    EFFECTIVE_INF_INT = int(EFFECTIVE_INF_FLOAT)
    EFFECTIVE_INF_FLOAT_INT = float(EFFECTIVE_INF_INT)

    @staticmethod
    def clone(metric: "MetricMeta", primary: bool = True) -> "MetricMeta":
        """Clone the metric metadata, optionally changing the primary flag."""
        m = copy.deepcopy(metric)
        m.is_primary_metric = primary
        return m

    # loosely coupled constructor w/ defaults (backward compatibility & resilience)
    def __init__(
        self,
        key: str,
        display_name: str = "",
        data_type: str = "float",
        display_format: str = ".4f",
        description: str = "",
        value_range: tuple[float, float] | None = (0.0, 1.0),
        value_enum: list[str] | None = None,
        higher_is_better: bool = True,
        threshold: float | None = 0.5,
        is_primary_metric: bool = True,
        parent_metric: str = "",
        exclude: bool = False,
    ):
        """Constructor.

        Parameters
        ----------
        key : str
           Metric ID.
        display_name : str
          Human friendly display name of the metric.
        display_format : str
            Display format of the metric described using a limited set of formats
            which use D3 format notation: https://d3js.org/d3-format
            For example:
            ".0%" for ``10%``, ".4f" for ``0.1415``, "$,.2f" for ``$1,234.56``,
            ".2" for ``42``, ...
            To format time, use for instance ".3" and ``seconds`` data type.
        data_type : str
          Data type of the metric (``str(type(value))``) and `seconds`.
        value_range : Tuple[float, float]
            Value range of the metric = ``[min, max]`` in case of the numeric metric.
            In case that the metric is categorical, the valid values are in the
            ``value_enum``.
        value_enum : list[str] | None
            List of possible values of the metric in case of the categorical metric.
            If the metric is numeric, the ``value_enum`` is ``None`` and ``value_range``
            is used.
        higher_is_better : bool
            ``True`` if higher values are better (like answer similarity),
            else ``False`` (like toxicity where lower is better).
        threshold :  float | None
            Threshold value for the metric (where makes sense - consider duration or
            cost).
        is_primary_metric : bool
            ``True`` if the metric is primary, else ``False``. By primary metric is
            meant metric which should be used to sort the leaderboards and/or show it
            in the first leaderboard column.
        parent_metric : str
            Parent metric key if the metric is used to calculate the parent metric.
        exclude : bool
            ``True`` to exclude the metric from the showing it in UI, else ``False``.

        """
        self.key = key
        self.display_name = display_name
        self.data_type = data_type
        self.display_format = display_format
        self.description = description
        self.value_range = value_range
        self.value_enum = value_enum
        self.enum = None
        self.higher_is_better = higher_is_better
        self.threshold = threshold
        self.is_primary_metric = is_primary_metric
        self.parent_metric = parent_metric
        self.exclude = exclude

    @staticmethod
    def is_metric_flip(
        old_value: float, new_value: float, metric_meta: "MetricMeta"
    ) -> bool:
        """Did metric score flip between old and new value?

        Returns
        -------
        bool :
            `True` if metric score flip, `False` otherwise.

        """
        if old_value != new_value:
            if (old_value < metric_meta.threshold < new_value) or (
                new_value < metric_meta.threshold < old_value
            ):
                return True

        return False

    def to_dict(self, threshold: float | None = None, portable: bool = False) -> dict:
        return {
            self.KEY_KEY: self.key,
            self.KEY_DISPLAY_NAME: self.display_name,
            self.KEY_DATA_TYPE: self.data_type,
            self.KEY_DISPLAY_FORMAT: self.display_format,
            self.KEY_DESCRIPTION: self.description,
            self.KEY_VALUE_RANGE: port_float(self.value_range, portable=portable),
            self.KEY_VALUE_ENUM: self.value_enum,
            self.KEY_HIGHER_IS_BETTER: self.higher_is_better,
            self.KEY_THRESHOLD: port_float(
                v=threshold if threshold is not None else self.threshold,
                portable=portable,
            ),
            self.KEY_IS_PRIMARY_METRIC: self.is_primary_metric,
            self.KEY_PARENT_METRIC: self.parent_metric,
            self.KEY_EXCLUDE: self.exclude,
        }

    def to_md(self, to_rst: bool = False) -> str:
        s_li_pad = "   " if to_rst else "    "
        s_primary_metric = (
            f"\n{s_li_pad}- This is **primary** metric."
            if self.is_primary_metric
            else ""
        )
        s_hb = (
            "Higher score is better."
            if self.higher_is_better
            else "Lower score is better."
        )
        s_q = "``" if to_rst else "`"
        return (
            f"- **{self.display_name}** ({self.data_type})\n"
            f"{s_li_pad}- {self.description}\n"
            f"{s_li_pad}- {s_hb}\n"
            f"{s_li_pad}- Range: {s_q}{list(self.value_range)}{s_q}\n"
            f"{s_li_pad}- Default threshold: {s_q}{self.threshold}{s_q}"
            f"{s_primary_metric}"
        )

    def dump(self, portable: bool = False) -> dict:
        return self.to_dict(portable=portable)

    def copy(self) -> "MetricMeta":
        return MetricMeta(
            key=self.key,
            display_name=self.display_name,
            data_type=self.data_type,
            display_format=self.display_format,
            description=self.description,
            value_range=self.value_range,
            value_enum=self.value_enum,
            higher_is_better=self.higher_is_better,
            threshold=self.threshold,
            is_primary_metric=self.is_primary_metric,
            parent_metric=self.parent_metric,
            exclude=self.exclude,
        )

    @staticmethod
    def _decode_range(raw_range: tuple) -> tuple[float, float]:
        if not raw_range:
            return 0.0, 1.0

        if len(raw_range) != 2:
            raise ValueError(
                f"Invalid metric value range '{raw_range}' - "
                "it must be a tuple with two values."
            )

        return (
            SafeJavaScript.decode_to_float(raw_range[0]),
            SafeJavaScript.decode_to_float(raw_range[1]),
        )

    @staticmethod
    def from_dict(data: dict | tuple) -> "MetricMeta":
        data = data[1] if isinstance(data, tuple) else data
        # loosely coupled dictionary to object (backward compatibility & resilience)
        return MetricMeta(
            key=data.get(MetricMeta.KEY_KEY, ""),
            display_name=data.get(MetricMeta.KEY_DISPLAY_NAME, ""),
            data_type=data.get(MetricMeta.KEY_DATA_TYPE, "float"),
            display_format=data.get(MetricMeta.KEY_DISPLAY_FORMAT, "{v:.4f}"),
            description=data.get(MetricMeta.KEY_DESCRIPTION, ""),
            value_range=MetricMeta._decode_range(
                data.get(MetricMeta.KEY_VALUE_RANGE, (0.0, 1.0))
            ),
            value_enum=data.get(MetricMeta.KEY_VALUE_ENUM, None),
            higher_is_better=data.get(MetricMeta.KEY_HIGHER_IS_BETTER, True),
            threshold=data.get(MetricMeta.KEY_THRESHOLD, 0.5),
            is_primary_metric=data.get(MetricMeta.KEY_IS_PRIMARY_METRIC, True),
            parent_metric=data.get(MetricMeta.KEY_PARENT_METRIC, ""),
            exclude=data.get(MetricMeta.KEY_EXCLUDE, False),
        )

    @staticmethod
    def load(data: dict) -> "MetricMeta":
        return MetricMeta.from_dict(data)


class MetricsMeta:
    KEY_META = "metadata"

    def __init__(self, metrics: list[MetricMeta] = None):
        metrics = metrics or []
        self.key_to_metric = {m.key: m for m in metrics}

    def size(self) -> int:
        return len(self.key_to_metric)

    def contains(self, key: str) -> bool:
        return key in self.key_to_metric

    def add_metric(self, metric: MetricMeta):
        self.key_to_metric[metric.key] = metric

    def get_metric_keys(self) -> list[str]:
        return list(self.key_to_metric.keys())

    def get_metric(self, key: str) -> MetricMeta | None:
        return self.key_to_metric.get(key, None)

    def get_metric_best_value(self, key: str) -> float | None:
        """Get the best value for the metric."""
        metric_meta = self.get_metric(key)
        if (
            metric_meta
            and metric_meta.value_range
            and isinstance(metric_meta.value_range, tuple)
        ):
            if metric_meta.higher_is_better:
                return metric_meta.value_range[1]
            else:
                return metric_meta.value_range[0]

        return None

    def get_metric_worst_value(self, key: str) -> float | None:
        """Get the worst value for the metric."""
        metric_meta = self.get_metric(key)
        if (
            metric_meta
            and metric_meta.value_range
            and isinstance(metric_meta.value_range, tuple)
        ):
            if metric_meta.higher_is_better:
                return metric_meta.value_range[0]
            else:
                return metric_meta.value_range[1]

        return None

    def is_metric_passed(self, key: str, value: float) -> bool:
        metric_meta = self.get_metric(key)
        if metric_meta:
            if metric_meta.higher_is_better:
                return bool(value >= metric_meta.threshold)
            else:
                return bool(value <= metric_meta.threshold)

        raise ValueError(
            f"Metric '{key}' not found in the evaluation metrics meta. "
            f"Valid metrics are: {list(self.key_to_metric.keys())}."
        )

    def get_metric_description(self, key: str) -> str:
        m = self.key_to_metric.get(key, None)
        return m.description if m else ""

    def get_primary_metric(self) -> MetricMeta | None:
        """Return the metric which is marked as primary metric."""
        for m in self.key_to_metric.values():
            if m.is_primary_metric:
                return m

        if len(self.key_to_metric) == 1:
            the_only_metric = next(iter(self.key_to_metric.values()))
            the_only_metric.is_primary_metric = True
            return the_only_metric

        raise ValueError("No primary metric found in the evaluation metrics meta.")

    def get_threshold(self, key: str, default_value=None) -> float | None:
        return (
            self.get_metric(key).threshold
            if key in self.key_to_metric
            else default_value
        )

    def is_higher_better(self, key: str) -> bool:
        metric = self.get_metric(key)
        if metric:
            return metric.higher_is_better

        raise ValueError(
            f"Metric '{key}' not found in the evaluation metrics meta. "
            f"Valid metrics are: {list(self.key_to_metric.keys())}."
        )

    def set_threshold(self, threshold: float, key: str = ""):
        if key:
            self.key_to_metric[key].threshold = threshold
        else:
            for m in self.key_to_metric.values():
                m.threshold = threshold

    def make_portable(self) -> "MetricsMeta":
        for m in self.key_to_metric.values():
            m.value_range = port_float(m.value_range, portable=True)
            m.threshold = port_float(m.threshold, portable=True)
        return self

    def to_dict(self, threshold: float | None = None) -> dict:
        metrics_meta = self.key_to_metric.copy()
        for mm in metrics_meta.keys():
            metrics_meta[mm] = self.key_to_metric[mm].to_dict(threshold)
        return metrics_meta

    def dump(self, portable: bool = False) -> list:
        return [m.dump(portable=portable) for m in self.key_to_metric.values()]

    def to_list(self) -> list[MetricMeta]:
        return list(self.key_to_metric.values())

    def copy_with_overrides(self, metric_key_to_overrides: dict) -> "MetricsMeta":
        """Copy metrics meta with updated:

        - display names
        - descriptions
        - exclude flag

        Parameters
        ----------
        metric_key_to_overrides : dict
            Dictionary with metric key to overrides mapping - map:
            ``metric key`` -> ``field key`` -> ``new value``

        Returns
        -------
        MetricsMeta
            Copy of the metrics meta with updated display names and descriptions.

        """

        metrics = []
        for k in self.key_to_metric.keys():
            m = self.key_to_metric[k].copy()
            if k in metric_key_to_overrides:
                if MetricMeta.KEY_DISPLAY_NAME in metric_key_to_overrides[k]:
                    m.display_name = metric_key_to_overrides[k][
                        MetricMeta.KEY_DISPLAY_NAME
                    ]
                if MetricMeta.KEY_DESCRIPTION in metric_key_to_overrides[k]:
                    m.description = metric_key_to_overrides[k][
                        MetricMeta.KEY_DESCRIPTION
                    ]
                if MetricMeta.KEY_EXCLUDE in metric_key_to_overrides[k]:
                    m.exclude = metric_key_to_overrides[k][MetricMeta.KEY_EXCLUDE]
            metrics.append(m)
        return MetricsMeta(metrics=metrics)

    def clone(self) -> "MetricsMeta":
        return self.copy_with_overrides({})

    @staticmethod
    def from_dict(metrics_meta: dict | list) -> "MetricsMeta":
        metrics = []
        metrics_meta_items = (
            metrics_meta.items() if isinstance(metrics_meta, dict) else metrics_meta
        )
        metrics_meta_values = (
            metrics_meta_items.values()
            if isinstance(metrics_meta_items, dict)
            else metrics_meta_items
        )
        for value in metrics_meta_values:
            metrics.append(MetricMeta.from_dict(value))
        return MetricsMeta(metrics=metrics)

    @staticmethod
    def load(metrics_meta: list) -> "MetricsMeta":
        metrics = []
        for m in metrics_meta:
            metrics.append(MetricMeta.load(m))
        return MetricsMeta(metrics=metrics)


FLOAT_INF_PORTABLE = sys.float_info.max
FLOAT_NEG_INF_PORTABLE = -sys.float_info.max


def port_float(v: float | tuple, portable: bool = False) -> float | tuple:
    """Make float portable by replacing inf/-inf with large/small finite values."""
    if not portable:
        return v

    if isinstance(v, float):
        if v == float("inf"):
            return FLOAT_INF_PORTABLE
        if v == float("-inf"):
            return FLOAT_NEG_INF_PORTABLE
        return v
    elif isinstance(v, tuple):
        return tuple(port_float(vi, portable=portable) for vi in v)

    return v


class ResourceHandle:
    H_PREFIX: str = "resource:"
    H_CONNECTION: str = "connection"
    H_KEY: str = "key"
    H_VERSION: str = "version"

    @staticmethod
    def is_handle(handle) -> bool:
        if handle is not None:
            if isinstance(handle, str):
                try:
                    if ResourceHandle.parse_string_handle(handle):
                        return True
                    return False
                except ValueError:
                    return False
            elif isinstance(handle, ResourceHandle):
                return True
            elif isinstance(handle, str):
                return handle.startswith(ResourceHandle.H_PREFIX)

        return False

    @staticmethod
    def parse_string_handle(loc_str: str) -> tuple[str, str, str]:
        """Parse CLI argument into connection, resource key and version."""
        #
        # Handle format:
        #
        #  resource:connection:<connect_id>:key:<resource_key>[:version:<res_version>]
        #
        # Handle example:
        #
        #  resource:connection:mydai:key:12345678-1234-1234-1234-123456789012:version:1
        #  resource:connection:mlops:key:12345678-1234-1234-1234-123456789012
        #
        # Hints:
        #
        # - connect_id is unique identifier of the connection among other connections
        #   (it does not have to be globally unique)
        # - connection_key must be NCName (no spaces, no special characters)
        # - resource_key is unique identifier of the resource in the service/runtime
        #   behind the connection
        #
        # CLI:
        #
        # --model <handle>
        # --dataset <handle>
        # --testset <handle>
        # --validset <handle>
        #

        if not loc_str:
            raise ValueError("Resource locator cannot be empty")
        if not loc_str.startswith(ResourceHandle.H_PREFIX):
            raise ValueError(
                f"Resource locator '{loc_str}' must start with 'resource:'"
            )
        if not loc_str.startswith(
            f"{ResourceHandle.H_PREFIX}{ResourceHandle.H_CONNECTION}"
        ):
            raise ValueError(
                f"Resource locator '{loc_str}' "
                "must start with the resource prefix and connection ID separator "
                "'resource:connection:'"
            )
        if f":{ResourceHandle.H_KEY}:" not in loc_str:
            raise ValueError(
                f"Resource locator '{loc_str}' "
                "must contain resource key separator ':key:'"
            )

        # compact
        loc_wip = loc_str[
            len(ResourceHandle.H_PREFIX) + len(ResourceHandle.H_CONNECTION) + 1 :
        ]
        (connection_id, res_key_and_ver) = loc_wip.split(":key:", 1)
        if f":{ResourceHandle.H_VERSION}:" in res_key_and_ver:
            (resource_key, resource_version) = res_key_and_ver.split(":version:", 1)
        else:
            resource_key = res_key_and_ver
            resource_version = ""

        if not connection_id:
            raise ValueError(
                f"Unable to parse valid connection ID from the resource locator "
                f"'{loc_str}' - it must be a non-empty NCName"
            )
        elif not is_ncname(connection_id):
            raise ValueError(
                f"Connection ID must be a valid NCName (no spaces, no special "
                f"characters) - '{connection_id}'"
            )

        if not resource_key:
            raise ValueError(
                f"Unable to parse valid resource ID from the resource locator "
                f"'{loc_str}' - it must be a non-empty NCName"
            )
        elif not is_ncname(resource_key):
            raise ValueError(
                f"Resource ID must be a valid NCName (no spaces, no special "
                f"characters) - '{resource_key}'"
            )

        return connection_id, resource_key, resource_version

    def __init__(
        self,
        connection_key: str,
        resource_key: str,
        version: str = "",
    ):
        self.connection_key = connection_key
        self.resource_key = resource_key
        self.version = version

    def __str__(self):
        if self.version:
            return (
                f"{ResourceHandle.H_PREFIX}"
                f"{ResourceHandle.H_CONNECTION}:{self.connection_key}"
                f":{ResourceHandle.H_KEY}:{self.resource_key}"
                f":{ResourceHandle.H_VERSION}:{self.version}"
            )
        else:
            return (
                f"{ResourceHandle.H_PREFIX}"
                f"{ResourceHandle.H_CONNECTION}:{self.connection_key}"
                f":{ResourceHandle.H_KEY}:{self.resource_key}"
            )


class UpdateGlobalExplanation:
    """Update mode: ``merge`` (to add new explanations) or ``replace`` (default)."""

    UPDATE_MODE: str = "update_mode"
    OPT_MERGE: str = "merge"
    OPT_REPLACE: str = "replace"

    UPDATE_SCOPE: str = "update_scope"
    OPT_FEATURE: str = "feature"
    OPT_CLASS: str = "class"

    """Driverless AI/common parameters source: ``inherit`` or ``request`` (default)."""
    PARAMS_SOURCE: str = "params_source"
    OPT_INHERIT: str = "inherit"
    OPT_REQUEST: str = "request"


class ExplainerJobStatus(enum.Enum):
    SYNCING = -4
    SCHEDULED = -3
    UNKNOWN = -2
    IN_PROGRESS = -1
    RUNNING = -1
    SUCCESS = 0
    FINISHED = 0
    CANCELLED = 1
    FAILED = 2
    ABORTED_BY_USER = 3
    ABORTED_BY_RESTART = 4
    TIMED_OUT = 5

    @staticmethod
    def from_int(status_code: int) -> "ExplainerJobStatus":
        return ExplainerJobStatus(status_code)

    @staticmethod
    def is_job_failed(status: "ExplainerJobStatus") -> bool:
        return status in [
            ExplainerJobStatus.FAILED,
            ExplainerJobStatus.CANCELLED,
            ExplainerJobStatus.ABORTED_BY_RESTART,
            ExplainerJobStatus.ABORTED_BY_USER,
            ExplainerJobStatus.TIMED_OUT,
        ]

    @staticmethod
    def is_job_running(status: "ExplainerJobStatus") -> bool:
        return status in [
            ExplainerJobStatus.RUNNING,
            ExplainerJobStatus.SYNCING,
            ExplainerJobStatus.SCHEDULED,
            ExplainerJobStatus.UNKNOWN,
        ]

    @staticmethod
    def is_job_finished(status: "ExplainerJobStatus") -> bool:
        return (
            ExplainerJobStatus.is_job_failed(status)
            or status == ExplainerJobStatus.FINISHED
        )

    def to_string(self, status_code: int):
        return


class ExplainerToRun:
    """Parametrized explainer (to run) - ID and explainer parameters
    (dictionary, JSon string or any format explainer is able to process).

    """

    def __init__(
        self,
        explainer_id: str,
        params: str | dict = None,
        extra_params: list | None = None,
    ) -> None:
        self.id = explainer_id
        self.params = params or {}
        self.extra_params = extra_params

    def __str__(self):
        return str(
            {
                "explainer_id": self.id,
                "params": self.params,
                "extra_params": self.extra_params,
            }
        )

    def dump(self) -> dict:
        d = {k: v for k, v in vars(self).items()}
        return d

    def clone(self) -> "ExplainerToRun":
        return ExplainerToRun(
            explainer_id=self.id,
            params=self.params,
            extra_params=self.extra_params,
        )

    @staticmethod
    def load(d: dict) -> "ExplainerToRun":
        return ExplainerToRun(
            explainer_id=d.get("id", ""),
            params=d.get("params", {}),
            extra_params=d.get("extra_params", []),
        )


class EvaluatorToRun(ExplainerToRun):
    def __init__(
        self,
        evaluator_id: str,
        params: str | dict = None,
        extra_params: list | None = None,
    ) -> None:
        ExplainerToRun.__init__(self, evaluator_id, params, extra_params)


class PerturbationIntensity(enum.Enum):
    VERY_LOW = enum.auto()
    LOW = enum.auto()
    MEDIUM = enum.auto()
    HIGH = enum.auto()
    VERY_HIGH = enum.auto()
    EXTREME = enum.auto()


class PerturbatorToRun:
    """Parametrized perturbator (to run)."""

    KEYWORD_PERTURBATOR_ID = "perturbator_id"
    KEYWORD_INTENSITY = "intensity"
    KEYWORD_PARAMS = "params"

    def __init__(
        self,
        perturbator_id: str,
        intensity: str | PerturbationIntensity = PerturbationIntensity.MEDIUM,
        params: str | dict = None,
    ) -> None:
        self.perturbator_id = perturbator_id

        if isinstance(intensity, PerturbationIntensity):
            self.intensity = intensity
        elif isinstance(intensity, str):
            try:
                self.intensity = PerturbationIntensity[intensity.upper()]
            except KeyError:
                raise ValueError(
                    f"Invalid perturbation intensity '{intensity}' - must be "
                    f"one of {PerturbationIntensity.__members__}"
                )
        else:
            raise ValueError(
                f"Invalid perturbation intensity '{intensity}' - must be "
                f"one of {PerturbationIntensity.__members__}"
            )

        self.params = params or {}

    def __str__(self):
        return str(
            {
                PerturbatorToRun.KEYWORD_PERTURBATOR_ID: self.perturbator_id,
                PerturbatorToRun.KEYWORD_INTENSITY: self.intensity.name,
                PerturbatorToRun.KEYWORD_PARAMS: self.params,
            }
        )

    def dump(self) -> dict:
        d = {k: v for k, v in vars(self).items()}
        return d

    def clone(self) -> "PerturbatorToRun":
        return PerturbatorToRun(
            perturbator_id=self.perturbator_id,
            intensity=self.intensity,
            params=self.params,
        )

    @staticmethod
    def load(d: dict) -> "PerturbatorToRun":
        return PerturbatorToRun(
            perturbator_id=d.get(PerturbatorToRun.KEYWORD_PERTURBATOR_ID, ""),
            intensity=d.get(
                PerturbatorToRun.KEYWORD_INTENSITY, PerturbationIntensity.MEDIUM
            ),
            params=d.get(PerturbatorToRun.KEYWORD_PARAMS, {}),
        )


class CommonInterpretationParams:
    # interpretation parameters declaration for documentation, report and introspection
    PARAM_MODEL = Param(
        param_name="model",
        description="Interpreted model.",
        param_type=InterpretationParamType.any,
    )
    PARAM_MODELS = Param(
        param_name="models",
        description="Evaluated LLM/RAG models.",
        param_type=InterpretationParamType.any,
    )
    PARAM_DATASET = Param(
        param_name="dataset",
        description="Dataset used to interpret the model.",
        param_type=InterpretationParamType.any,
    )
    PARAM_TARGET_COL = Param(
        param_name="target_col",
        description="Target column of the interpreted model.",
        param_type=InterpretationParamType.str,
    )
    PARAM_VALIDSET = Param(
        param_name="validset",
        description="Optional validation dataset used to interpret the model.",
        param_type=InterpretationParamType.str,
    )
    PARAM_TESTSET = Param(
        param_name="testset",
        description="Optional test dataset used to interpret the model.",
        param_type=InterpretationParamType.str,
    )
    PARAM_USE_RAW_FEATURES = Param(
        param_name="use_raw_features",
        description=(
            "Whether to use original features for the training of Surrogate model "
            "explainers (True, default) or force transformed features (False)."
        ),
        param_type=InterpretationParamType.bool,
    )
    PARAM_WEIGHT_COL = Param(
        param_name="weight_col",
        description=(
            "Optional name of the dataset column with weights of examples to be used "
            "in the model interpretation."
        ),
        param_type=InterpretationParamType.str,
    )
    PARAM_PREDICTION_COL = Param(
        param_name="prediction_col",
        description=(
            "Optional name of the dataset column with predictions of examples to be "
            "used in the (standalone ~ no model, just predictions) model "
            "interpretation."
        ),
        param_type=InterpretationParamType.str,
    )
    PARAM_DROP_COLS = Param(
        param_name="drop_cols",
        description=(
            "Optional list of the dataset column names to be drop and not used in "
            "the model interpretation."
        ),
        param_type=InterpretationParamType.list,
        default_value=[],
    )
    PARAM_SAMPLE_NUM_ROWS = Param(
        param_name="sample_num_rows",
        description=(
            "The sample size, number of rows, to be used for the surrogate models. "
            "This setting overrides global library sampling configuration."
        ),
        param_type=InterpretationParamType.int,
        default_value=0,
    )
    PARAM_RESULTS_LOCATION = Param(
        param_name="results_location",
        description=(
            "Filesystem path (database connectio or other storage type location) "
            "which specifies where to store interpretation results. Current directory "
            "is used by default"
        ),
        param_type=InterpretationParamType.str,
    )
    PARAM_USED_FEATURES = Param(
        param_name="used_features",
        description=(
            "Optional specification of the features (dataset columns) used by the "
            "interpreted model (in case that model doesn's support used features "
            "introspection)."
        ),
        param_type=InterpretationParamType.list,
        default_value=None,
    )

    _parameters = [
        PARAM_MODEL,
        PARAM_MODELS,
        PARAM_DATASET,
        PARAM_TARGET_COL,
        PARAM_VALIDSET,
        PARAM_TESTSET,
        PARAM_USE_RAW_FEATURES,
        PARAM_WEIGHT_COL,
        PARAM_PREDICTION_COL,
        PARAM_DROP_COLS,
        PARAM_SAMPLE_NUM_ROWS,
        PARAM_RESULTS_LOCATION,
        PARAM_USED_FEATURES,
    ]

    def __init__(
        self,
        model,
        models,
        dataset,
        target_col: str,
        validset=PARAM_VALIDSET.default_value,
        testset=PARAM_TESTSET.default_value,
        use_raw_features: bool = PARAM_USE_RAW_FEATURES.default_value,
        weight_col: str = PARAM_WEIGHT_COL.default_value,
        prediction_col: str = PARAM_PREDICTION_COL.default_value,
        drop_cols: list | None = PARAM_DROP_COLS.default_value,
        sample_num_rows: int | None = PARAM_SAMPLE_NUM_ROWS.default_value,
        results_location: str = PARAM_RESULTS_LOCATION.default_value,
        used_features: list | None = PARAM_USED_FEATURES.default_value,
        extra_params: list | None = None,
    ) -> None:
        """Interpretation parameters.

        Parameters
        ----------
        dataset :
          h2o.H2OFrame, pandas.DataFrame]
          Dataset source: explainable dataset instance, datatable
          frame, H2OFrame, pandas DataFrame, string (expect path to CSV, .jay or any
          other file type supported by datatable), dictionary (used to construct frame).
        model :
          Path to model (str, Path), explainable model (``ExplainableModel``) or
          an instance of 3rd party model (like Scikit) to interpret.
        models :
          Paths to models (str, Path), explainable models (``ExplainableModel``) or
          an instances of 3rd party models (like Scikit) to interpret.
        target_col : str
          Target column name - must be valid dataset column name.
        validset :
          Optional path to validation dataset (str, Path) or datatable Frame instance.
        testset :
          Optional path to test dataset (str, Path) or datatable Frame instance.
        use_raw_features : bool
          ``False`` to use original features for the training of surrogate models,
          ``True`` to use transformed features.
        weight_col : str
          Name of the weight column to be used by explainers.
        prediction_col : str
          DEPRECATED: name of the predictions column - in case of 3rd party model (
          standalone MLI).
        drop_cols : list | None
          List of the columns to drop from the interpretation i.e. columns names which
          should not be explained.
        sample_num_rows : int | None
          Do sample of the ``dataset`` to given number of rows.
        results_location : str | pathlib.Path | dict | None
          Where to store interpretation results - filesystem (path as string or
          ``Path``), memory (dictionary) or DB. If ``None``, then results are stored
          to the current directory.
        extra_params : dict | None
          Extra parameters.

        """
        self.model = model
        self.models = models
        self.dataset = dataset
        self.validset = validset
        self.testset = testset
        self.use_raw_features = use_raw_features
        self.target_col = target_col
        self.weight_col = weight_col
        self.prediction_col = prediction_col
        self.drop_cols = drop_cols
        self.sample_num_rows = sample_num_rows
        self.results_location = results_location
        self.extra_params = extra_params
        self.used_features = used_features

        # introspection: parameters indexation
        self.cfg_items_dict = {}
        for p in CommonInterpretationParams._parameters:
            self.cfg_items_dict[p.param_name] = p

    def describe_config_items(self) -> dict[str, Param]:
        return self.cfg_items_dict

    def describe_config_item(self, config_item_name: str) -> Param | None:
        return self.cfg_items_dict.get(config_item_name, None)

    def to_dict(self) -> dict:
        """Safe string-friendly serialization to dictionary."""

        def _safe_str_field(field):
            return (
                field
                if isinstance(field, str)
                else (
                    str(field)
                    if isinstance(field, pathlib.Path)
                    else (str(type(field)) if field is not None else None)
                )
            )

        return {
            "model": _safe_str_field(self.model),
            "models": _safe_str_field(self.models),
            "dataset": _safe_str_field(self.dataset),
            "validset": _safe_str_field(self.validset),
            "testset": _safe_str_field(self.testset),
            "use_raw_features": self.use_raw_features,
            "target_col": self.target_col,
            "weight_col": self.weight_col,
            "prediction_col": self.prediction_col,
            "drop_cols": self.drop_cols,
            "sample_num_rows": self.sample_num_rows,
            "results_location": self.results_location,
            "used_features": self.used_features,
        }

    def dump(self) -> dict:
        d = {k: v for k, v in vars(self).items()}
        return d

    def clone(self) -> "CommonInterpretationParams":
        return CommonInterpretationParams(
            model=self.model,
            models=self.models,
            dataset=self.dataset,
            target_col=self.target_col,
            validset=self.validset,
            testset=self.testset,
            use_raw_features=self.use_raw_features,
            weight_col=self.weight_col,
            prediction_col=self.prediction_col,
            drop_cols=self.drop_cols,
            sample_num_rows=self.sample_num_rows,
            results_location=self.results_location,
            extra_params=self.extra_params,
            used_features=self.used_features,
        )

    @staticmethod
    def load(d: dict) -> "CommonInterpretationParams":
        return CommonInterpretationParams(**d)


class LookAndFeel:
    KEY_LF = "look_and_feel"

    FORMAT_HEXA = "hexa"

    # yellow to black on white
    H2O_SONAR_THEME = "h2o_sonar"
    # yellow to black on black
    DRIVERLESS_AI_THEME = "driverless_ai"
    # blue to red on white
    BLUE_THEME = "blue"

    # colors
    COLOR_H2OAI_YELLOW = "#fec925"
    COLOR_DAI_GREEN = "#bbc600"
    COLOR_MATPLOTLIB_BLUE = "#3b74b4"
    COLOR_WHITE = "#ffffff"
    COLOR_RED = "#ff0000"
    COLOR_BLACK = "#000000"
    COLOR_HOT_ORANGE = "#fd5800"

    # Driverless AI
    COLORMAP_YELLOW_2_BLACK = [COLOR_H2OAI_YELLOW, COLOR_BLACK]
    # default Shapley plot
    COLORMAP_BLUE_2_RED = ["#00AAEE", "#FF1166"]
    # grayscale
    COLORMAP_WHITE_2_BLACK = [COLOR_WHITE, COLOR_BLACK]

    THEME_2_COLORMAP = {
        H2O_SONAR_THEME: COLORMAP_YELLOW_2_BLACK,
        DRIVERLESS_AI_THEME: COLORMAP_YELLOW_2_BLACK,
        BLUE_THEME: COLORMAP_BLUE_2_RED,
    }
    THEME_2_FG_COLOR = {
        H2O_SONAR_THEME: COLOR_H2OAI_YELLOW,
        DRIVERLESS_AI_THEME: COLOR_H2OAI_YELLOW,
        BLUE_THEME: COLOR_MATPLOTLIB_BLUE,
    }
    THEME_2_BG_COLOR = {
        H2O_SONAR_THEME: COLOR_WHITE,
        DRIVERLESS_AI_THEME: COLOR_BLACK,
        BLUE_THEME: COLOR_WHITE,
    }
    THEME_2_LINE_COLOR = {
        H2O_SONAR_THEME: COLOR_BLACK,
        DRIVERLESS_AI_THEME: COLOR_WHITE,
        BLUE_THEME: COLOR_BLACK,
    }

    @staticmethod
    def get_bg_color(theme: str):
        return LookAndFeel.THEME_2_BG_COLOR.get(theme, LookAndFeel.COLOR_WHITE)

    @staticmethod
    def get_fg_color(theme: str):
        return LookAndFeel.THEME_2_FG_COLOR.get(theme, LookAndFeel.COLOR_H2OAI_YELLOW)

    @staticmethod
    def get_line_color(theme: str):
        return LookAndFeel.THEME_2_LINE_COLOR.get(theme, LookAndFeel.COLOR_BLACK)

    @staticmethod
    def get_colormap(
        colormap_data: list[str] | str = "",
        theme: str = "",
    ):
        """Get Matplotlib colormap.

        Parameters
        ----------
        colormap_data : list[str] | str
          Create color map either from the list of two colors (string hexadecimal color
          specification) or by color map name.
        theme : str
          H2O Sonar theme to create color map based on the theme.

        matplotlib.colors.Colormap :
          Color map.

        """

        if not colormap_data:
            colormap = colors.LinearSegmentedColormap.from_list(
                "default_colormap",
                LookAndFeel.THEME_2_COLORMAP.get(
                    theme, LookAndFeel.COLORMAP_YELLOW_2_BLACK
                ),
            )
        elif isinstance(colormap_data, list):
            colormap = colors.LinearSegmentedColormap.from_list(
                "default_colormap", colormap_data
            )
        else:
            colormap = pyplot.get_cmap(colormap_data)

        return colormap


#
# lang functions
#


def base_pkg(obj):
    """Get base package for given Python object.

    Parameters
    ----------
    obj : Python object

    Returns
    -------
    str :
        Base package of Python object and sub-package, e.g., sklearn or ensemble.

    """
    mod = inspect.getmodule(obj)
    if mod:
        base, _sep, _stem = mod.__name__.partition(".")
        _stem = _stem.partition("._")[0]
        return base, _stem
    return "", ""


def opt_import_err_msg(pckg_names: list[str] | str) -> str:
    """Generate optional package import error message.

    Parameters
    ----------
    pckg_names : list[str] | str
        Name or list of names of the required packages.

    Returns
    -------
    str :
        Generated error message.

    """
    if isinstance(pckg_names, list):
        pckg_names_fmt = ", ".join([f"'{p}'" for p in pckg_names])
        return f"The {pckg_names_fmt} Python packages are required, but not installed"
    return f"The '{pckg_names}' Python package is required, but not installed"


def raise_opt_import_err(pckg_names: list[str] | str) -> None:
    """Raise optional package import error.

    Parameters
    ----------
    pckg_names : list[str] | str
        Name or list of names of the required packages.

    Raises
    ------
    ImportError
        Always.

    """
    raise ImportError(opt_import_err_msg(pckg_names))


def generate_key() -> str:
    return str(uuid.uuid4())


def is_valid_key(key: str) -> bool:
    try:
        uuid.UUID(key, version=4)
        return True
    except ValueError:
        return False


def is_port_used(
    hostname: str = "127.0.0.1",
    port: int = 12345,
    service_name="Driverless AI",
    timeout=15,
    logger=None,
) -> bool:
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)  # avoid hang
        s.connect((hostname, port))
        if logger:
            logger.debug(
                f"A service ('{service_name}') is running on {hostname}:{port} and "
                f"it is accessible"
            )
        return True
    except OSError as e:
        if logger:
            logger.debug(
                f"Unable to determine whether a service (like '{service_name}') uses "
                f"{hostname}:{port} - error: {e}\n{traceback.format_exc()}"
            )
        return False
    finally:
        if s:
            s.close()


def add_string_list(items: list | None, add_items: list | None) -> list:
    """Robust list handling of features to drop, process, use, skip, ..."""
    items = [] if items is None else items.copy()
    add_items = [] if add_items is None else add_items.copy()
    items.extend(add_items)
    items = list(set(items))
    return items


class SafeJavaScript:
    """Safe JavaScript datastructures de/serialization."""

    NAN = "NaN"
    INF = "Infinity"
    NEG_INF = "-Infinity"

    @staticmethod
    def decode_to_float(obj):
        if isinstance(obj, str):
            if obj == SafeJavaScript.NAN:
                return float("nan")
            if obj == SafeJavaScript.INF:
                return float("inf")
            if obj == SafeJavaScript.NEG_INF:
                return float("-inf")

        if isinstance(obj, int) or isinstance(obj, float):
            return float(obj)

        return obj


class SemVer:
    @staticmethod
    def from_str(version: str):
        """Parse a semantic version ``<major>.<minor>.<patch>``.

        Returns
        -------
        SemVer | None :
          Instance of SemVer class if valid version, ``None`` otherwise.

        """
        if version:
            version_split = version.split(".")
            if version_split and len(version_split) == 3:
                try:
                    return SemVer.from_int_list([int(v) for v in version_split])
                except Exception as ex:
                    print(
                        f"Unable to parse semantic version: '{version}': {ex}"
                        f"\n{traceback.format_exc()}"
                    )
                    pass
        return None

    @staticmethod
    def from_int_list(version_list: list[int]):
        """Semantic version from the list of 3 integers.

        Returns
        -------
        SemVer | None :
          Instance of SemVer class if valid version, ``None`` otherwise.

        """
        return SemVer(
            major=version_list[0],
            minor=version_list[1],
            patch=version_list[2],
        )

    def __init__(self, major: int, minor: int, patch: int):
        self.major = major
        self.minor = minor
        self.patch = patch


#
# math
#


def harmonic_mean(xs) -> float:
    if not xs or 0.0 in xs:
        return 0.0

    inv_sum = sum(1.0 / n for n in xs)

    return float(len(xs)) / inv_sum if inv_sum != 0.0 else 0.0
