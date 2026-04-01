# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import enum
import json
import pathlib
import uuid
from enum import Enum
from typing import Any

import datatable

from h2o_sonar import loggers
from h2o_sonar.lib.api import commons
from h2o_sonar.utils import perturbations
from h2o_sonar.utils import tokenization


class LlmPromptCategories(enum.Enum):
    unknown = "unknown"
    math = "math"
    writing = "writing"
    planning = "planning"
    knowledge = "knowledge"
    reasoning = "reasoning"
    coding = "coding"
    question_answering = "question_answering"
    harm = "harm"
    troubleshooting = "troubleshooting"
    recommendation = "recommendation"
    facts = "facts"
    summarization = "summarization"
    evaluation = "evaluation"
    classification = "classification"


class LlmInputRelType(enum.Enum):
    """Test case / input relationship types."""

    perturbation_source = enum.auto()


class LlmInputRelTargetType(enum.Enum):
    """Test case / input relationships target types."""

    test_case = enum.auto()


class LlmInputRel:
    """Test case relationship."""

    KEY_REL_TYPE = "type"
    KEY_REL_TARGET = "target"
    KEY_REL_TARGET_TYPE = "target_type"

    def __init__(
        self,
        rel_type: str = LlmInputRelType.perturbation_source.name,
        target: str = "",
        target_type: str = LlmInputRelTargetType.test_case.name,
    ):
        self.rel_type = rel_type
        self.target = target
        self.target_type = target_type

    def to_dict(self):
        return {
            LlmInputRel.KEY_REL_TYPE: self.rel_type,
            LlmInputRel.KEY_REL_TARGET: self.target,
            LlmInputRel.KEY_REL_TARGET_TYPE: self.target_type,
        }

    @staticmethod
    def from_dict(as_dict: dict) -> "LlmInputRel":
        return LlmInputRel(
            rel_type=as_dict.get(LlmInputRel.KEY_REL_TYPE, ""),
            target=as_dict.get(LlmInputRel.KEY_REL_TARGET, ""),
            target_type=as_dict.get(LlmInputRel.KEY_REL_TARGET_TYPE, ""),
        )


class LlmDataset:
    """Dataset used to evaluate LLMs and RAGs."""

    # row key
    KEY_KEY = "key"
    # test suite's/test lab's test case key
    KEY_TC_KEY = "test_case_key"
    # dictionary keys (JSon representation)
    KEY_INPUTS = "inputs"
    # input / prompt / question
    KEY_INPUT = "input"
    # OPTIONAL: URLs/paths to document(s) which were used to fine-tune the RAG for
    KEY_CORPUS = "corpus"
    # OPTIONAL: context (set of document chunks by value i.e. text snippets) returned
    KEY_CONTEXT = "context"
    # OPTIONAL: input categories like: math, knowledge, reasoning, coding or question
    KEY_CATEGORIES = "categories"
    # OPTIONAL: input relationships
    KEY_RELATIONSHIPS = "relationships"
    # OPTIONAL: the key of the H2O Sonar model to be used to get the actual answer
    #           (if none, then all interpretation's models will be used)
    KEY_MODEL_KEY = "model_key"
    # OPTIONAL: the key of the test suite's test where the test case belongs to
    KEY_TEST_KEY = "test_key"
    # OPTIONAL: expected output / answer
    KEY_EXPECTED_OUTPUT = "expected_output"
    # OPTIONAL: output / answer constraint (interpreted by explainer) any data structure
    KEY_OUTPUT_CONSTRAINTS = "output_constraints"
    # OPTIONAL: output / answer string condition (interpreted by explainer)
    KEY_OUTPUT_CONDITION = "output_condition"
    # OPTIONAL: actual answer
    KEY_ACTUAL_OUTPUT = "actual_output"
    # OPTIONAL: how much time it took to get the actual answer
    KEY_ACTUAL_DURATION = "actual_duration"
    # OPTIONAL: answer/inference cost
    KEY_COST = "cost"

    COLUMNS = [
        KEY_KEY,
        KEY_INPUT,
        KEY_CORPUS,
        KEY_CONTEXT,
        KEY_CATEGORIES,
        KEY_RELATIONSHIPS,
        KEY_MODEL_KEY,
        KEY_TEST_KEY,
        KEY_EXPECTED_OUTPUT,
        KEY_OUTPUT_CONSTRAINTS,
        KEY_OUTPUT_CONDITION,
        KEY_ACTUAL_OUTPUT,
        KEY_ACTUAL_DURATION,
        KEY_COST,
    ]

    # tabular dataset column names (datatable representation)
    COL_INPUT = KEY_INPUT
    COL_CORPUS = KEY_CORPUS
    COL_CONTEXT = KEY_CONTEXT
    COL_CATEGORIES = KEY_CATEGORIES
    COL_RELATIONSHIPS = KEY_RELATIONSHIPS
    COL_MODEL_KEY = KEY_MODEL_KEY
    COL_TEST_KEY = KEY_TEST_KEY
    COL_EXPECTED_OUTPUT = KEY_EXPECTED_OUTPUT
    COL_OUTPUT_CONSTRAINTS = KEY_OUTPUT_CONSTRAINTS
    COL_OUTPUT_CONDITION = KEY_OUTPUT_CONDITION
    COL_ACTUAL_OUTPUT = KEY_ACTUAL_OUTPUT
    COL_ACTUAL_DURATION = KEY_ACTUAL_DURATION
    COL_COST = KEY_COST

    class LlmDatasetRow:
        def __init__(
            self,
            i: str,
            context: list[str] | None = None,
            corpus: list[str] | None = None,
            categories: str | list[str] = "",
            relationships: list | None = None,
            expected_output: str = "",
            output_constraints: list[str] | Any | None = None,
            output_condition: str = "",
            actual_output: str = "",
            actual_duration: float = 0.0,
            cost: float = 0.0,
            model_key: str = "",
            test_key: str = "",
            key: str = "",
        ):
            """LLM dataset row - question / prompt / input with related (meta)data.

            Parameters
            ----------
            i : str
                Input / question / prompt.
            corpus : list[str] | None
                URLs/paths to document(s) which were used to fine-tune the RAG for this
                test case.
            context : list[str] | None
                Context (set of document chunks by value i.e. text snippets) returned by
                the vector database for augmentation to LLM.
            categories : str | list[str]
                Categories of the input (question/prompt) like: math, knowledge,
                reasoning,
            relationships : list | None
                Relationships among rows capturing e.g., perturbation source/product.
            expected_output : str
                Expected output / answer.
            output_constraints : list[str] | Any | None
                An optional output / answer constraints that might be any data
                structure that can be serialized to JSON going forward. It is
                interpreted by explainer and used for the validation.
            output_condition : str
                An optional string condition which is interpreted and used by explainer
                 to validate output / answer. ``output_condition`` can use
                ``output_constraints`` or vice versa.
            actual_output : str
                Actual output / answer returned by the LLM / RAG product.
            actual_duration : float
                How much time it took to get the actual answer.
            cost : float
                Answer/inference cost.
            model_key : str
                The key of the H2O Sonar model which was used to get the actual answer.
            test_key : str
                The key of the test where the test case belongs to.
            key : str
                Key of the dataset row.

            """
            self.key = key or str(uuid.uuid4())
            self.i = i  # TODO self.i > self.input (no clash w/ global symbols)
            self.context = context or []
            self.corpus = corpus or []
            self.categories = categories or []
            self.categories = (
                self.categories
                if isinstance(self.categories, list)
                else [str(self.categories)]
            )
            self.relationships = relationships or []
            self.expected_output = expected_output
            self.output_constraints = output_constraints or []
            self.output_condition = output_condition
            self.actual_output = actual_output
            self.actual_duration = actual_duration
            self.cost = cost
            self.model_key = model_key
            self.test_key = test_key

        def add_relationship(
            self, relationship_type: str, target: str, target_type: str
        ):
            self.relationships.append(
                LlmInputRel(
                    rel_type=relationship_type,
                    target=target,
                    target_type=target_type,
                )
            )

        def perturb(
            self,
            perturbators: list[commons.PerturbatorToRun],
            raised_errors: list | None = None,
        ):
            """Perturb the input (prompt) using the specified perturbator.
            The perturbation is always performed in place on the input, which is
            a string.

            Parameters
            ----------
            perturbators : list[commons.PerturbatorToRun]
                List of perturbators to run.
            raised_errors : list | None
                List of raised errors.

            """
            registry = perturbations.PerturbatorRegistry.registry()

            for p2r in perturbators:
                perturbator = registry.get_perturbator(p2r.perturbator_id)
                self.i = perturbator.perturb(
                    text=self.i,
                    intensity=p2r.intensity,
                    raised_errors=raised_errors,
                    **p2r.params,
                )

        def copy(self, update_key: bool = True):
            return LlmDataset.LlmDatasetRow(
                key=str(uuid.uuid4()) if update_key else self.key,
                i=self.i,
                corpus=self.corpus.copy(),
                context=self.context.copy(),
                categories=self.categories.copy(),
                expected_output=self.expected_output,
                output_constraints=self.output_constraints.copy(),
                output_condition=self.output_condition,
                actual_output=self.actual_output,
                actual_duration=self.actual_duration,
                cost=self.cost,
                model_key=self.model_key,
                test_key=self.test_key,
            )

        def to_dict(self) -> dict:
            return {
                LlmDataset.KEY_KEY: self.key or str(uuid.uuid4()),
                LlmDataset.KEY_INPUT: self.i,
                LlmDataset.KEY_CORPUS: self.corpus,
                LlmDataset.KEY_CONTEXT: self.context,
                LlmDataset.KEY_CATEGORIES: self.categories,
                LlmDataset.KEY_RELATIONSHIPS: [r.to_dict() for r in self.relationships],
                LlmDataset.KEY_EXPECTED_OUTPUT: self.expected_output,
                LlmDataset.KEY_OUTPUT_CONSTRAINTS: self.output_constraints,
                LlmDataset.KEY_OUTPUT_CONDITION: self.output_condition,
                LlmDataset.KEY_ACTUAL_OUTPUT: self.actual_output,
                LlmDataset.KEY_ACTUAL_DURATION: self.actual_duration,
                LlmDataset.KEY_COST: self.cost,
                LlmDataset.KEY_MODEL_KEY: self.model_key,
                LlmDataset.KEY_TEST_KEY: self.test_key,
            }

        @staticmethod
        def from_dict(as_dict: dict):
            relationships_dict = as_dict.get(LlmDataset.KEY_RELATIONSHIPS, [])
            # prefer test_case_key over key - key may be compound:
            #   test_case_key_model_key
            # if test_case_key is present, use it as the key otherwise fall back
            # to key field for backward compatibility
            key = as_dict.get(
                LlmDataset.KEY_TC_KEY,
                as_dict.get(LlmDataset.KEY_KEY, str(uuid.uuid4())),
            )
            return LlmDataset.LlmDatasetRow(
                key=key,
                i=as_dict[LlmDataset.KEY_INPUT],
                corpus=as_dict.get(LlmDataset.KEY_CORPUS, []),
                context=as_dict.get(LlmDataset.KEY_CONTEXT, []),
                categories=as_dict.get(LlmDataset.KEY_CATEGORIES, []),
                relationships=[LlmInputRel.from_dict(r) for r in relationships_dict],
                expected_output=as_dict.get(LlmDataset.KEY_EXPECTED_OUTPUT, ""),
                output_constraints=as_dict.get(LlmDataset.KEY_OUTPUT_CONSTRAINTS, None),
                output_condition=as_dict.get(LlmDataset.KEY_OUTPUT_CONDITION, ""),
                actual_output=as_dict.get(LlmDataset.KEY_ACTUAL_OUTPUT, ""),
                actual_duration=as_dict.get(LlmDataset.KEY_ACTUAL_DURATION, 0.0),
                cost=as_dict.get(LlmDataset.KEY_COST, 0.0),
                test_key=as_dict.get(LlmDataset.KEY_TEST_KEY, ""),
                model_key=as_dict.get(LlmDataset.KEY_MODEL_KEY, ""),
            )

    def __init__(self):
        self.inputs: list[LlmDataset.LlmDatasetRow] = []

    def stats(self) -> dict[str, int | dict]:
        # map: model_key -> count
        model_keys = {}
        for i in self.inputs:
            if i.model_key not in model_keys:
                model_keys[i.model_key] = 0
            model_keys[i.model_key] += 1

        return {
            "inputs": len(self.inputs),
            "per_model_inputs": model_keys,
        }

    def __str__(self):
        return f"{len(self.inputs)} inputs"

    def prompts(self) -> list[str]:
        """Return the list of unique prompts."""
        prompts = set()  # prompt might be > 1 @ dataset (multiple models output)
        for p in self.inputs:
            prompts.add(p.i)
        return list(prompts)

    def shape(self) -> list:
        return [len(self.inputs), 9]

    def add_input(
        self,
        i: str,
        corpus: list[str] | None = None,
        context: list[str] | None = None,
        categories: str | list[str] = "",
        relationships: list | None = None,
        expected_output: str = "",
        output_constraints: list[str] | Any | None = None,
        output_condition: str = "",
        actual_output: str = "",
        actual_duration: float = 0.0,
        cost: float = 0.0,
        model_key: str = "",
        test_key: str = "",
        key: str = "",
    ):
        """Add new dataset row - question / prompt / input with related (meta)data.

        Parameters
        ----------
        i : str
            Input / question / prompt.
        corpus : list[str] | None
            URLs/paths to document(s) which were used to fine-tune the RAG for this
            test case.
        context : list[str] | None
            Context (set of document chunks by value i.e. text snippets) returned by
            the vector database for augmentation to LLM.
        categories : str | list[str]
            Categories of the input (question/prompt) like: math, knowledge, reasoning,
        relationships : list | None
            Relationships among rows capturing e.g. perturbation source/product.
        expected_output : str
            Expected output / answer.
        output_constraints : list[str] | Any | None
            An optional output / answer constraints which might be any data
            structure which can be serialized to JSON going forward. It is
            interpreted by explainer and used for the validation.
        output_condition : str
            An optional string condition which is interpreted and used by explainer
            in order to validate output / answer. ``output_condition`` can use
            ``output_constraints`` or vice versa.
        actual_output : str
            Actual output / answer returned by the LLM / RAG product.
        actual_duration : float
            How much time it took to get the actual answer.
        cost : float
            Answer/inference cost.
        model_key : str
            The key of the H2O Sonar model which was used to get the actual answer.
        test_key : str
            The key of the test where the test case belongs to.
        key : str
            Key of the dataset row.

        """

        self.inputs.append(
            LlmDataset.LlmDatasetRow(
                key=key,
                i=i,
                corpus=corpus,
                context=context,
                categories=categories,
                relationships=relationships,
                expected_output=expected_output,
                output_constraints=output_constraints,
                output_condition=output_condition,
                actual_output=actual_output,
                actual_duration=actual_duration,
                cost=cost,
                model_key=model_key,
                test_key=test_key,
            )
        )

    def merge(self, other_llm_dataset: "LlmDataset"):
        """Merge another dataset into this one.

        Parameters
        ----------
        other_llm_dataset : LlmDataset
            LLM dataset to be merged into this one.

        """
        self.inputs.extend(other_llm_dataset.inputs)

    def perturb(
        self,
        perturbators: list[commons.PerturbatorToRun],
        in_place: bool = True,
        raised_errors: list | None = None,
    ) -> "LlmDataset":
        """Perturb the inputs (prompts) using the specified perturbator(s).

        Parameters
        ----------
        perturbators : list[commons.PerturbatorToRun]
            Perturbators to run - includes the perturbator ID, intensity,
            and parameters.
        in_place : bool
            If True, perturb the prompt in place, otherwise create a new perturbed
            rows.
        raised_errors : list | None
            If ``None``, then raise error(s) if the perturbator(s) fail(s),
            otherwise do not raise exceptions and store them in the (empty) list
            provided by the caller.

        """
        if not perturbators:
            raise ValueError("No perturbators provided")

        # cost/performance/resource protection - exclude expensive perturbators
        c_perturbators = perturbations.PerturbatorRegistry.registry().are_compatible(
            perturbators=perturbators, items=len(self.inputs)
        )

        if in_place:
            for p in self.inputs:
                p.perturb(c_perturbators, raised_errors=raised_errors)
        else:
            perturbed_inputs = []
            for p in self.inputs:
                perturbed_p = p.copy()

                # perturbations
                perturbation_errors = []
                perturbed_p.perturb(c_perturbators, raised_errors=perturbation_errors)
                if raised_errors is not None and perturbation_errors:
                    raised_errors.extend(perturbation_errors)
                    continue

                # categories
                cats = [perturbations.CAT_PERTURBED]
                for p2r in c_perturbators:
                    cats.append(
                        f"{perturbations.PREFIX_CAT_PERTURBED_BY}"
                        f"{p2r.perturbator_id}:{p2r.intensity}"
                    )
                    cats.append(
                        f"perturbation_type:{p2r.params.get('perturbation_type', '')}"
                    )
                    encoding_type = p2r.params.get("encoding_type", "")
                    if isinstance(encoding_type, Enum):
                        encoding_type = encoding_type.value
                    cats.append(f"encoding_type:{encoding_type}")
                if not perturbed_p.categories:
                    perturbed_p.categories = cats
                else:
                    for cat in cats:
                        if cat not in perturbed_p.categories:
                            perturbed_p.categories.append(cat)

                # relationships
                perturbed_p.add_relationship(
                    relationship_type=LlmInputRelType.perturbation_source.name,
                    target=p.key,
                    target_type=LlmInputRelTargetType.test_case.name,
                )
                perturbed_inputs.append(perturbed_p)

            if perturbed_inputs:
                self.inputs.extend(perturbed_inputs)

        return self

    def to_dict(self) -> dict:
        return {
            LlmDataset.KEY_INPUTS: [i.to_dict() for i in self.inputs],
        }

    @staticmethod
    def from_dict(as_dict: dict) -> "LlmDataset":
        result = LlmDataset()
        for i in as_dict[LlmDataset.KEY_INPUTS]:
            relationships_dict = i.get(LlmDataset.KEY_RELATIONSHIPS, [])
            result.add_input(
                key=i.get(LlmDataset.KEY_KEY, str(uuid.uuid4())),
                i=i.get(LlmDataset.KEY_INPUT, ""),
                corpus=i.get(LlmDataset.KEY_CORPUS, []),
                context=i.get(LlmDataset.KEY_CONTEXT, []),
                categories=i.get(LlmDataset.KEY_CATEGORIES, []),
                relationships=[LlmInputRel.from_dict(r) for r in relationships_dict],
                expected_output=i.get(LlmDataset.KEY_EXPECTED_OUTPUT, ""),
                output_constraints=i.get(LlmDataset.KEY_OUTPUT_CONSTRAINTS, []),
                output_condition=i.get(LlmDataset.KEY_OUTPUT_CONDITION, ""),
                actual_output=i.get(LlmDataset.KEY_ACTUAL_OUTPUT, ""),
                actual_duration=i.get(LlmDataset.KEY_ACTUAL_DURATION, 0.0),
                cost=i.get(LlmDataset.KEY_COST, 0.0),
                model_key=i.get(LlmDataset.KEY_MODEL_KEY, ""),
                test_key=i.get(LlmDataset.KEY_TEST_KEY, ""),
            )
        return result

    def to_datatable_dict(self) -> dict:
        result = {
            LlmDataset.KEY_KEY: [],
            LlmDataset.KEY_INPUT: [],
            LlmDataset.KEY_CORPUS: [],
            LlmDataset.KEY_CONTEXT: [],
            LlmDataset.KEY_CATEGORIES: [],
            LlmDataset.KEY_RELATIONSHIPS: [],
            LlmDataset.KEY_EXPECTED_OUTPUT: [],
            LlmDataset.KEY_OUTPUT_CONSTRAINTS: [],
            LlmDataset.KEY_OUTPUT_CONDITION: [],
            LlmDataset.KEY_ACTUAL_OUTPUT: [],
            LlmDataset.KEY_ACTUAL_DURATION: [],
            LlmDataset.KEY_COST: [],
            LlmDataset.KEY_MODEL_KEY: [],
            LlmDataset.KEY_TEST_KEY: [],
        }

        for p in self.inputs:
            result[LlmDataset.KEY_KEY].append(p.key)
            result[LlmDataset.KEY_INPUT].append(p.i)
            result[LlmDataset.KEY_CORPUS].append(p.corpus)
            result[LlmDataset.KEY_CONTEXT].append(p.context)
            result[LlmDataset.KEY_CATEGORIES].append(p.categories)
            result[LlmDataset.KEY_RELATIONSHIPS].append(
                [r.to_dict() for r in p.relationships]
            )
            result[LlmDataset.KEY_EXPECTED_OUTPUT].append(p.expected_output)
            result[LlmDataset.KEY_OUTPUT_CONSTRAINTS].append(p.output_constraints)
            result[LlmDataset.KEY_OUTPUT_CONDITION].append(p.output_condition)
            result[LlmDataset.KEY_ACTUAL_OUTPUT].append(p.actual_output)
            result[LlmDataset.KEY_ACTUAL_DURATION].append(p.actual_duration)
            result[LlmDataset.KEY_COST].append(p.cost)
            result[LlmDataset.KEY_MODEL_KEY].append(p.model_key)
            result[LlmDataset.KEY_TEST_KEY].append(p.test_key)

        return result

    @staticmethod
    def from_datatable_dict(as_dict: dict) -> "LlmDataset":
        """Deserialize ``datatable`` dictionary to ``LlmDataset``. Structured
        fields (corpus, categories, and output_constraints) are automatically
        deserialized from JSon string to dictionary if possible.

        Parameters
        ----------
        as_dict : dict
            Dictionary created using datatable.to_dict().

        Returns
        -------
        LlmDataset
            LLM dataset.

        """
        for k in [
            LlmDataset.KEY_KEY,
            LlmDataset.KEY_INPUT,
            LlmDataset.KEY_CORPUS,
            LlmDataset.KEY_CONTEXT,
            LlmDataset.KEY_CATEGORIES,
            LlmDataset.KEY_RELATIONSHIPS,
            LlmDataset.KEY_EXPECTED_OUTPUT,
            LlmDataset.KEY_OUTPUT_CONSTRAINTS,
            LlmDataset.KEY_ACTUAL_OUTPUT,
            LlmDataset.KEY_ACTUAL_DURATION,
            LlmDataset.KEY_COST,
            LlmDataset.KEY_MODEL_KEY,
            LlmDataset.KEY_TEST_KEY,
        ]:
            if k not in as_dict:
                raise ValueError(
                    f"Invalid datatable dictionary for LLM Dataset deserialization "
                    f"- missing '{k}' key"
                )

        result = LlmDataset()
        rows = len(as_dict[LlmDataset.KEY_INPUT])
        for i in range(rows):
            # robust structured columns deserialization
            corpus = LlmDataset.from_datatable_json_enc_col(
                as_dict[LlmDataset.KEY_CORPUS][i]
            )
            context = LlmDataset.from_datatable_json_enc_col(
                as_dict[LlmDataset.KEY_CONTEXT][i]
            )
            categories = LlmDataset.from_datatable_json_enc_col(
                as_dict[LlmDataset.KEY_CATEGORIES][i]
            )
            relationships = LlmDataset.from_datatable_json_enc_col(
                as_dict[LlmDataset.KEY_RELATIONSHIPS][i]
            )
            relationships = relationships or []
            output_constraints = LlmDataset.from_datatable_json_enc_col(
                as_dict[LlmDataset.KEY_OUTPUT_CONSTRAINTS][i]
            )
            result.add_input(
                key=as_dict[LlmDataset.KEY_KEY][i],
                i=as_dict[LlmDataset.KEY_INPUT][i],
                corpus=corpus,
                context=context,
                categories=categories,
                relationships=[
                    LlmInputRel.from_dict(r_dict) for r_dict in relationships
                ],
                expected_output=as_dict[LlmDataset.KEY_EXPECTED_OUTPUT][i],
                output_constraints=output_constraints,
                output_condition=as_dict[LlmDataset.KEY_OUTPUT_CONDITION][i],
                actual_output=as_dict[LlmDataset.KEY_ACTUAL_OUTPUT][i],
                actual_duration=as_dict[LlmDataset.KEY_ACTUAL_DURATION][i],
                cost=as_dict[LlmDataset.KEY_COST][i],
                model_key=as_dict[LlmDataset.KEY_MODEL_KEY][i],
                test_key=as_dict[LlmDataset.KEY_TEST_KEY][i],
            )
        return result

    @staticmethod
    def from_datatable_json_enc_col(enc_json_col: str, logger=None) -> list:
        """Robust deserialization of datatable JSON encoded column w/ a list value."""
        if enc_json_col:
            try:
                return json.loads(enc_json_col)
            except Exception as ex:
                logger = logger or loggers.SonarPrintLogger()
                logger.warning(
                    f"Failed to deserialize datatable JSON encoded column "
                    f"({enc_json_col}): {ex}"
                )
            # FALLBACK: return string as the list item (better than an empty list)
            return [enc_json_col]
        return []

    def to_datatable(self) -> datatable.Frame:
        d = self.to_datatable_dict()
        serializable_dict = {}
        for k in d:
            if k in [
                LlmDataset.COL_CORPUS,
                LlmDataset.COL_CATEGORIES,
                LlmDataset.COL_RELATIONSHIPS,
                LlmDataset.COL_OUTPUT_CONSTRAINTS,
                LlmDataset.COL_CONTEXT,
            ]:
                # d[k] is a list [ ... ] > serialize it to JSon & store as string
                serializable_dict[k] = [json.dumps(v) for v in d[k]]
            else:
                serializable_dict[k] = d[k]

        return datatable.Frame(serializable_dict)

    @staticmethod
    def load_from_json(
        json_file_path: str | pathlib.Path,
        datatable_format: bool = False,
    ):
        with open(json_file_path) as f:
            as_dict = json.load(f)

        return (
            LlmDataset.from_datatable_dict(as_dict)
            if datatable_format
            else LlmDataset.from_dict(as_dict)
        )

    def save_as_json(self, json_path: str | pathlib.Path):
        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

        return json_path


class LlmEvalResults:
    """LLM dataset with metrics values from the evaluation."""

    KEY_RESULTS = "results"

    class LlmEvalResultRow:
        # OPTIONAL: actual answer metadata
        KEY_ACTUAL_OUTPUT_META = "actual_output_meta"
        KEY_METRICS = "metrics"
        KEY_METRIC_KEY = "key"
        KEY_METRIC_VALUE = "value"
        KEY_METRICS_META = "metrics_meta"

        def __init__(
            self,
            dataset_row: LlmDataset.LlmDatasetRow,
            metrics: dict,
            actual_output_meta: list | None = None,
            metrics_meta: dict | None = None,
            use_compound_key: bool = True,
        ):
            """LLM evaluation result row - question / prompt / input with related
            (meta)data and metrics values.

            Parameters
            ----------
            dataset_row : LlmDataset.LlmDatasetRow
                Input / question / prompt.
            metrics : [dict]
                Map of metric IDs to metric values.
            actual_output_meta : list | None
                Optional list of metadata about the actual output like tokenization,
                embeddings, introspection, etc.
            metrics_meta : dict | None
                Optional map of metric IDs to metric metadata like thresholds,
                expected values, etc.
            use_compound_key : bool
                Whether to use compound key (test_case_key_model_key) or simple key
                (test_case_key) in the 'key' field. Default is True (compound).

            """
            # input
            self.dataset_row = dataset_row

            # evaluator's output
            self.metrics = metrics
            self.actual_output_meta = actual_output_meta or []
            self.metrics_meta = metrics_meta or {}
            self._use_compound_key = use_compound_key

        def to_dict(self, type_friendly_metrics: bool = False) -> dict:
            dataset_as_dict = self.dataset_row.to_dict()

            # TODO technical debt: unique key to be generated in test lab for test case
            #   then it should be used as a key for the evaluation results
            #   - test case key + model key = unique key ... for now composite key
            # use compound key if flag is set and model_key exists
            if self._use_compound_key and self.dataset_row.model_key:
                # check if key already ends with model_key to avoid double-appending
                if not self.dataset_row.key.endswith(f"_{self.dataset_row.model_key}"):
                    dataset_as_dict[LlmDataset.KEY_KEY] = (
                        f"{self.dataset_row.key}_{self.dataset_row.model_key}"
                    )
                else:
                    dataset_as_dict[LlmDataset.KEY_KEY] = self.dataset_row.key
            else:
                # use simple key (test_case_key only)
                dataset_as_dict[LlmDataset.KEY_KEY] = self.dataset_row.key
            dataset_as_dict[LlmDataset.KEY_TC_KEY] = self.dataset_row.key

            # LEGACY: metrics are side-by-side to other fields ~ unfortunate design
            for metric_id in self.metrics:
                dataset_as_dict[metric_id] = self.metrics[metric_id]
            if type_friendly_metrics:
                # NEW & PROTO friendly design: metrics are stored in a separate field
                this = LlmEvalResults.LlmEvalResultRow
                dataset_as_dict[this.KEY_METRICS] = [
                    {this.KEY_METRIC_KEY: k, this.KEY_METRIC_VALUE: v}
                    for k, v in self.metrics.items()
                ]

            actual_output_meta_list = []
            if self.actual_output_meta:
                for aom in self.actual_output_meta:
                    if isinstance(aom, tokenization.Tokenization):
                        actual_output_meta_list.append(aom.to_dict())
                    elif isinstance(aom, dict):
                        actual_output_meta_list.append(aom)
                    else:
                        raise ValueError(
                            f"Unsupported actual output metadata type when serializing "
                            f"LLM evaluation results row (must be dictionary): "
                            f"{type(aom)}"
                        )
            dataset_as_dict[LlmEvalResults.LlmEvalResultRow.KEY_ACTUAL_OUTPUT_META] = (
                actual_output_meta_list
            )

            dataset_as_dict[LlmEvalResults.LlmEvalResultRow.KEY_METRICS_META] = (
                self.metrics_meta
            )

            return dataset_as_dict

    COL_ACTUAL_OUTPUT_META = LlmEvalResultRow.KEY_ACTUAL_OUTPUT_META
    COL_METRICS_META = LlmEvalResultRow.KEY_METRICS_META

    def __init__(self):
        self.results: list[LlmEvalResults.LlmEvalResultRow] = []

    def __str__(self):
        return f"{len(self.results)} results"

    def shape(self) -> list:
        return [len(self.results), 9]

    def prompts(self) -> list[str]:
        """Return the list of unique prompts."""
        prompts = set()  # prompt might be > 1 @ dataset (multiple models output)
        for p in self.results:
            prompts.add(p.dataset_row.i)
        return list(prompts)

    def add_result(self, result: LlmEvalResultRow):
        """Add new dataset row - question / prompt / input with related (meta)data.

        Parameters
        ----------
        result : LlmEvalResultRow
            Result row.

        """
        self.results.append(result)

    def to_dict(self) -> dict:
        return {
            LlmEvalResults.KEY_RESULTS: [r.to_dict() for r in self.results],
        }

    @staticmethod
    def from_dict(as_dict: dict) -> "LlmEvalResults":
        llm_results_columns = LlmDataset.COLUMNS + [
            LlmEvalResults.LlmEvalResultRow.KEY_ACTUAL_OUTPUT_META,
            LlmEvalResults.LlmEvalResultRow.KEY_METRICS,  # exclude metrics array
            LlmEvalResults.LlmEvalResultRow.KEY_METRICS_META,  # exclude metrics_meta
            LlmDataset.KEY_TC_KEY,  # exclude test_case_key
        ]

        result = LlmEvalResults()
        for r in as_dict[LlmEvalResults.KEY_RESULTS]:
            actual_output_meta = []
            actual_output_meta_list = r.get(
                LlmEvalResults.LlmEvalResultRow.KEY_ACTUAL_OUTPUT_META, []
            )
            if actual_output_meta_list:
                for m in actual_output_meta_list:
                    if isinstance(m, dict):
                        if "tokenization" in m:
                            actual_output_meta.append(
                                tokenization.Tokenization.from_dict(m)
                            )
                        else:
                            actual_output_meta.append(m)
                    else:
                        raise ValueError(
                            f"Unsupported actual output metadata type when "
                            f"deserializing LLM evaluation results row"
                            f"(must be dictionary): {type(m)}"
                        )

            # extract metrics - support both formats:
            # 1. NEW format: metrics as array [{"key": "rouge_1", "value": 0.5}, ...]
            # 2. LEGACY format: metrics as top-level fields {"rouge_1": 0.5, ...}
            metrics = {}
            metrics_array = r.get(LlmEvalResults.LlmEvalResultRow.KEY_METRICS, None)
            if metrics_array is not None and isinstance(metrics_array, list):
                # NEW format: convert array to dict
                for metric_item in metrics_array:
                    if isinstance(metric_item, dict):
                        metric_key = metric_item.get(
                            LlmEvalResults.LlmEvalResultRow.KEY_METRIC_KEY
                        )
                        metric_value = metric_item.get(
                            LlmEvalResults.LlmEvalResultRow.KEY_METRIC_VALUE
                        )
                        if metric_key:
                            # only add if value exists (some metrics may have no value)
                            if metric_value is not None:
                                metrics[metric_key] = metric_value
            else:
                # LEGACY format: extract metrics from top-level fields
                metrics = {k: r[k] for k in r if k not in llm_results_columns}

            # detect if original JSON used compound key format
            # by checking if key field matches test_case_key_model_key pattern
            original_key = r.get(LlmDataset.KEY_KEY, "")
            test_case_key = r.get(LlmDataset.KEY_TC_KEY, "")
            model_key = r.get(LlmDataset.KEY_MODEL_KEY, "")

            # check if original key was compound (test_case_key_model_key)
            expected_compound = (
                f"{test_case_key}_{model_key}" if test_case_key and model_key else ""
            )
            use_compound_key = original_key == expected_compound

            result.add_result(
                LlmEvalResults.LlmEvalResultRow(
                    dataset_row=LlmDataset.LlmDatasetRow.from_dict(r),
                    metrics=metrics,
                    actual_output_meta=actual_output_meta,
                    use_compound_key=use_compound_key,
                )
            )
        return result

    def to_datatable_dict(self) -> dict:
        result = {
            LlmDataset.KEY_KEY: [],
            LlmDataset.KEY_INPUT: [],
            LlmDataset.KEY_CORPUS: [],
            LlmDataset.KEY_CONTEXT: [],
            LlmDataset.KEY_CATEGORIES: [],
            LlmDataset.KEY_RELATIONSHIPS: [],
            LlmDataset.KEY_EXPECTED_OUTPUT: [],
            LlmDataset.KEY_OUTPUT_CONSTRAINTS: [],
            LlmDataset.KEY_OUTPUT_CONDITION: [],
            LlmDataset.KEY_ACTUAL_OUTPUT: [],
            LlmEvalResults.LlmEvalResultRow.KEY_ACTUAL_OUTPUT_META: [],
            LlmDataset.KEY_ACTUAL_DURATION: [],
            LlmDataset.KEY_COST: [],
            LlmDataset.KEY_MODEL_KEY: [],
        }

        if self.results:
            for metric_id in self.results[0].metrics:
                result[metric_id] = []

            for p in self.results:
                p.dataset_row.relationships = p.dataset_row.relationships or []
                result[LlmDataset.KEY_KEY].append(p.dataset_row.key)
                result[LlmDataset.KEY_INPUT].append(p.dataset_row.i)
                result[LlmDataset.KEY_CORPUS].append(p.dataset_row.corpus)
                result[LlmDataset.KEY_CONTEXT].append(p.dataset_row.context)
                result[LlmDataset.KEY_CATEGORIES].append(p.dataset_row.categories)
                result[LlmDataset.KEY_RELATIONSHIPS].append(
                    [r.to_dict() for r in p.dataset_row.relationships]
                )
                result[LlmDataset.KEY_EXPECTED_OUTPUT].append(
                    p.dataset_row.expected_output
                )
                result[LlmDataset.KEY_OUTPUT_CONSTRAINTS].append(
                    p.dataset_row.output_constraints
                )
                result[LlmDataset.KEY_OUTPUT_CONDITION].append(
                    p.dataset_row.output_condition
                )
                result[LlmDataset.KEY_ACTUAL_OUTPUT].append(p.dataset_row.actual_output)
                actual_output_meta_list = []
                if p.actual_output_meta:
                    for aom in p.actual_output_meta:
                        if isinstance(aom, tokenization.Tokenization):
                            actual_output_meta_list.append(aom.to_dict())
                        elif isinstance(aom, dict):
                            actual_output_meta_list.append(aom)
                        else:
                            raise ValueError(
                                f"Unsupported actual output metadata type when "
                                f"serializing LLM evaluation results row "
                                f"(must be dictionary): {type(aom)}"
                            )
                result[LlmEvalResults.LlmEvalResultRow.KEY_ACTUAL_OUTPUT_META].append(
                    actual_output_meta_list
                )
                result[LlmDataset.KEY_ACTUAL_DURATION].append(
                    p.dataset_row.actual_duration
                )
                result[LlmDataset.KEY_COST].append(p.dataset_row.cost)
                result[LlmDataset.KEY_MODEL_KEY].append(p.dataset_row.model_key)
                for metric_id in p.metrics:
                    result[metric_id].append(p.metrics[metric_id])

        return result

    def to_datatable(self) -> datatable.Frame:
        d = self.to_datatable_dict()
        serializable_dict = {}
        for k in d:
            if k in [
                LlmDataset.COL_CORPUS,
                LlmDataset.COL_OUTPUT_CONSTRAINTS,
                LlmDataset.COL_OUTPUT_CONDITION,
                LlmDataset.COL_CATEGORIES,
                LlmDataset.COL_RELATIONSHIPS,
                LlmDataset.COL_CONTEXT,
                LlmEvalResults.COL_ACTUAL_OUTPUT_META,
                LlmEvalResults.COL_METRICS_META,
            ]:
                # d[k] is a list [ ... ] > serialize it to JSon & store as string
                serializable_dict[k] = [json.dumps(v) for v in d[k]]
            else:
                serializable_dict[k] = d[k]

        return datatable.Frame(serializable_dict)

    def save_as_json(self, json_path: str | pathlib.Path):
        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

        return json_path

    @staticmethod
    def load_from_json(
        json_file_path: str | pathlib.Path,
        datatable_format: bool = False,
    ):
        if datatable_format:
            raise NotImplementedError(
                "Loading LlmEvalResults from datatable format is not supported"
            )

        with open(json_file_path) as f:
            as_dict = json.load(f)

        return LlmEvalResults.from_dict(as_dict)

    def to_llm_dataset(self) -> LlmDataset:
        """Convert evaluation results to the LLM dataset - keep all fields,
        skip metrics.

        """
        result = LlmDataset()
        for r in self.results:
            result.add_input(
                key=r.dataset_row.key,
                i=r.dataset_row.i,
                corpus=r.dataset_row.corpus,
                context=r.dataset_row.context,
                categories=r.dataset_row.categories,
                relationships=r.dataset_row.relationships,
                expected_output=r.dataset_row.expected_output,
                output_constraints=r.dataset_row.output_constraints,
                output_condition=r.dataset_row.output_condition,
                actual_output=r.dataset_row.actual_output,
                actual_duration=r.dataset_row.actual_duration,
                cost=r.dataset_row.cost,
                model_key=r.dataset_row.model_key,
            )
        return result
