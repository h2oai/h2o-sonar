# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
import enum
from abc import ABC
from abc import abstractmethod
from typing import Any

from h2o_sonar.lib.api import formats as f5s


class ExplanationDescriptor:
    KEY_EXPLANATION_TYPE = "explanation_type"
    KEY_NAME = "name"
    KEY_DESCRIPTION = "description"
    KEY_CATEGORY = "category"
    KEY_SCOPE = "scope"
    KEY_HAS_LOCAL = "has_local"
    KEY_FORMATS = "formats"

    # loosely coupled constructor w/ defaults (backward compatibility & resilience)
    def __init__(
        self,
        explanation_type: str,
        name: str = "",
        category: str = "",
        scope: str = "",
        has_local: str = "",
        formats: list[str] = None,
    ) -> None:
        self.explanation_type = explanation_type
        self.name = name
        self.category = category
        self.scope = scope
        self.has_local = has_local
        self.formats = formats or []

    def dump(self) -> dict:
        d = {k: v for k, v in vars(self).items()}
        return d

    def clone(self) -> "ExplanationDescriptor":
        return ExplanationDescriptor(
            self.explanation_type,
            self.name,
            self.category,
            self.scope,
            self.has_local,
            self.formats,
        )

    @staticmethod
    def load(d: dict) -> "ExplanationDescriptor":
        # loosely coupled dictionary to object (backward compatibility & resilience)
        return ExplanationDescriptor(
            explanation_type=d.get(ExplanationDescriptor.KEY_EXPLANATION_TYPE, ""),
            name=d.get(ExplanationDescriptor.KEY_NAME, ""),
            category=d.get(ExplanationDescriptor.KEY_CATEGORY, ""),
            scope=d.get(ExplanationDescriptor.KEY_SCOPE, ""),
            has_local=d.get(ExplanationDescriptor.KEY_HAS_LOCAL, ""),
            formats=d.get(ExplanationDescriptor.KEY_FORMATS, []),
        )


class Explanation(ABC):
    """Base class of explainer explanations."""

    """Explanation type."""
    _explanation_type: str = None
    """Explanation scope."""
    _is_global: bool = None

    DISPLAY_CAT_MODEL = "MODEL"
    DISPLAY_CAT_DAI_MODEL = "DAI MODEL"
    DISPLAY_CAT_LLM = "LLM"
    DISPLAY_CAT_DATA = "DATA"
    DISPLAY_CAT_SURROGATES = "SURROGATE MODELS"
    DISPLAY_CAT_SURROGATES_ON_RES = "SURROGATE MODELS ON RESIDUALS"
    DISPLAY_CAT_NLP = "NLP"
    DISPLAY_CAT_COMPLIANCE = "COMPLIANCE TESTS"
    DISPLAY_CAT_AUTOREPORT = "AUTOREPORT"
    DISPLAY_CAT_CUSTOM = "CUSTOM"
    DISPLAY_CAT_MOCK = "MOCK"
    DISPLAY_CAT_TEMPLATE = "TEMPLATE"
    DISPLAY_CAT_EXAMPLE = "EXAMPLE"

    @classmethod
    def explanation_type(cls) -> str:
        """Explanation type may be any string identifier (either defined
        by this class or user ~ extensibility) which is used for validation and
        further processing. It must specify unique explanation name and scope.
        Explanation formats are defined by child classes of this abstract class.

        Format: ``<explanation_scope>-<explanation-type>``

        Example: ``global-feature-importance``

        """
        return f"{cls.explanation_scope()}-{cls._explanation_type}"

    @classmethod
    def explanation_scope(cls) -> str:
        """Explanation scope - either `global` or `local`."""
        return "global" if cls._is_global else "local"

    @classmethod
    def is_global(cls) -> bool:
        """Is the explanation global or local?"""
        return cls._is_global

    @classmethod
    def as_class_descriptor(cls) -> ExplanationDescriptor:
        return ExplanationDescriptor(
            explanation_type=cls.explanation_type(),
            name=cls.__name__,
            category="",
            scope=cls.explanation_scope(),
            has_local="",
            formats=list(),
        )

    def as_descriptor(self) -> ExplanationDescriptor:
        return ExplanationDescriptor(
            explanation_type=self.explanation_type(),
            name=self.display_name,
            category=self.display_category,
            scope=self.explanation_scope(),
            has_local=self.has_local,
            formats=self.format_types,
        )

    @property
    def format_types(self) -> list[str]:
        """Explanation formats provided by the explanation.

        Representations are set by explanations as they are created. This is why
        available format types are initialized as empty instance field, not class one.

        Example:

        .. code-block:: text

            ["application/json", "application/vnd.h2oai.datatable", "application/zip" ]

        Returns
        -------
        list[str]:
          Representations (formats) of this explanation.

        """
        return list(self._formats.keys())

    @property
    def has_local(self) -> str:
        """Does explanation have also related local explanation and which?

        Returns
        -------
        str:
          Local explanation type.

        """
        return self._has_local

    @has_local.setter
    def has_local(self, has_local):
        self._has_local = has_local

    @property
    def explainer(self):
        return self._explainer

    @property
    def display_name(self) -> str:
        return (
            self._display_name if self._display_name else f"{self.__class__.__name__}"
        )

    @display_name.setter
    def display_name(self, display_name: str):
        self._display_name = display_name

    @property
    def display_category(self) -> str:
        return (
            self._display_category
            if self._display_category
            else Explanation.DISPLAY_CAT_CUSTOM
        )

    @display_category.setter
    def display_category(self, display_category: str):
        self._display_category = display_category

    def __init__(
        self,
        explainer,
        display_name: str = "",
        display_category: str = "",
        has_local=None,
    ):
        """New custom explanation.

        Parameters
        ----------
        explainer: Explainer
          Explainer which uses this explanation.
        display_name: str
          Explanation name (used to name tile in the UI).
        display_category: str
          Explanation category (used to choose tab of tiles in the UI).
        has_local: str
          Optional related local explanation type.

        """
        # explanation type check
        if not self.explanation_type:
            raise ValueError("Explanation type name must be non-empty and unique")
        if self.is_global is None:
            raise ValueError("Explanation type must specify global or local scope")

        self._explainer = explainer

        self._display_name = (
            display_name
            if display_name
            else f"{explainer.display_name}: {self.__class__.__name__}"
        )
        self._display_category = (
            display_category if display_category else Explanation.DISPLAY_CAT_CUSTOM
        )

        # optional related local explanation type
        self._has_local: str = has_local

        # formats dictionary: [format type: ? extends CustomExplanation]
        self._formats: dict[str, Any] = dict()

    def __str__(self):
        result = (
            f"{self.__class__.__name__}\n"
            f"  explainer  : {self.explainer.__class__.__name__}\n"
            f"  type       : {self.explanation_type()}\n"
            f"  name       : {self.display_name}\n"
            f"  category   : {self.display_category}\n"
            f"  global     : {self.is_global()}\n"
            f"  scope      : {self.explanation_scope()}\n"
            f"  formats    : {self.format_types}\n"
            f"  description: {self.as_descriptor().dump()}\n"
            f"  has_local  : {self._has_local if self._has_local else ''}\n"
        )
        return result

    @abstractmethod
    def validate(self) -> bool:
        """Method used to validate (perform sanity check) of the canonical result
        (frame) produced by the explainer so that it can be subsequently processed
        without problems, e.g. by grammar of MLI visualization components.
        """
        raise NotImplementedError

    def get_format(self, explanation_format: str) -> f5s.ExplanationFormat:
        """Get explanation in specific representation."""
        if explanation_format in self.format_types:
            return self._formats[explanation_format]
        else:
            raise ValueError(
                f"Explanation representation '{explanation_format}' was not created "
                f"by explainer '{self.explainer.__class__.__name__}'"
                f"in explanation '{self.explanation_type}'"
            )

    @staticmethod
    def _check_args_explanation(explanation_format: f5s.ExplanationFormat):
        """Check explanation representation arguments.

        Parameters
        ----------
        explanation_format: ExplanationFormat
          Explanation representation.

        """
        if not explanation_format:
            raise ValueError(
                "Explanation representation must provide actual explanation"
            )
        if not explanation_format.mime:
            raise ValueError("MIME of explanation representation must provided")

    def add_format(self, explanation_format: f5s.ExplanationFormat) -> None:
        """Add explanation representation in a new format.

        Parameters
        ----------
        explanation_format: ExplanationFormat
          New explanation representation.
        """
        Explanation._check_args_explanation(explanation_format)
        self._formats[explanation_format.mime] = explanation_format


class SentenceComparisonMethod(enum.Enum):
    """Enum for sentence comparison methods.

    Attributes
    ----------
    EXACT_MATCH : str
        Exact string matching - sentences must be identical.
    COSINE_DISTANCE : str
        Cosine distance of sentence embeddings - semantic similarity.
    BERT_SCORE : str
        BERTScore - contextual embeddings similarity using BERT.

    """

    EXACT_MATCH = "exact_match"
    COSINE_DISTANCE = "cosine_distance"
    BERT_SCORE = "bert_score"
