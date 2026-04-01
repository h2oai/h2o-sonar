# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import enum
import json
import math
import os
import pathlib
import shutil
import subprocess

import markdown
import pytest

from h2o_sonar import evaluate
from h2o_sonar import interpret
from h2o_sonar import loggers
from h2o_sonar.evaluators import encoding_guardrail_evaluator as e_g_e
from h2o_sonar.lib.api import evaluators as e8s
from h2o_sonar.utils import perturbations
from tests import test_utils
from tests.explainers.doc import example_morris_sa_explainer
from tests.lib.test_cli import given_cli
from tests.lib.test_evaluators_gear import _ALL_EVALUATORS


# constants
_ALL_PERTURBATORS = perturbations.PerturbatorRegistry.registry().list_perturbators()

#
# DOCUMENTATION utility tests: .rst, docstrings (Markdown), ...
#


@pytest.mark.skip(reason="Documentation experiment")
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_markdown_to_html():
    """Test features and requirements of the Markdown to HTML conversion:

    * https://python-markdown.github.io/

    """
    #
    # GIVEN
    #
    md = """# Title
This is a paragraph with **bold** and *italic* text.

This is converted to <code>:
```
print("Hello, World!")
```
This is converted to <pre>:

    print("Hello, World 1!")
    print("Hello, World 2!")
    print("Hello, World 3!")

This is ending paragraph.
"""

    #
    # WHEN
    #
    html = markdown.markdown(
        text=md,
        extensions=["markdown.extensions.tables", "markdown.extensions.fenced_code"],
    )

    #
    # THEN
    #
    print(html)
    assert html


@pytest.mark.parametrize(
    "e",
    _ALL_EVALUATORS,
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_brief_description(e):
    #
    # GIVEN
    #

    print(
        f"Description _: {e._display_name} #########################################"
        f"\n'{e._description}'"
        f"\nDescription (): ########################################################"
        f"\n'{e().description}'"
    )

    #
    # WHEN
    #
    print(
        f"\nBrief description _: ################################################"
        f"\n'{e._brief_description}'"
        f"\nBrief description (): ################################################"
        f"\n  '{e().brief_description}'"
    )

    #
    # THEN
    #
    assert e().brief_description
    assert isinstance(e().brief_description, str)
    assert len(e().brief_description) > len(e().display_name) + 3


@pytest.mark.parametrize(
    "ps",
    [
        _ALL_PERTURBATORS,
    ],
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_perturbator_go_dict(tmp_path, ps):
    """Generate Go dictionary from perturbator description:

    var PerturbatorDescriptors = map[string]WfReportDataPerturbator{
        "antonymPerturbator": WfReportDataPerturbator{
            ID:          "antonymPerturbator",
            Name:        "Antonym perturbator",
            Description: "...",
        },
    }

    """

    go_src = "var perturbatorDescriptors = map[string]wfReportDataPerturbator{"

    for p in ps:
        go_src += f'"{p.display_name}": {{\n'
        go_src += f'    ID:          "{p.perturbator_id()}",\n'
        go_src += f'    Name:        "{p.display_name}",\n'
        go_src += f'    Description: "{p.description}",\n'
        go_src += "},\n"

    go_src += "}"

    # save Go source to file
    go_src_file = tmp_path / "perturbator_descriptors.go"
    with open(go_src_file, "w") as f:
        f.write(go_src)

    #
    # THEN
    #
    print(go_src)
    assert go_src


@pytest.mark.parametrize(
    "es",
    [
        _ALL_EVALUATORS,
    ],
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_evaluator_go_dict(tmp_path, es):
    """Generate Go dictionary from evaluator description:

    var EvaluatorDescriptors = map[string]WfReportDataEvaluator{
        "encodingguardrailEvaluator": WfReportDataEvaluator{
            Requirements: []string{
                "Actual answer",
            },
            Name:        "Encoding guardrail",
            Description: "...",
            Method:      "...",
            Metrics: []WfReportDataMetric{
                {
                    Id:          "...",
                    Name:        "...",
                    Description: "...",
                },
            },
        },
    }

    """

    go_src = "var evaluatorDescriptors = map[string]wfReportDataEvaluator{"

    key_id = "id"
    key_name = "name"
    key_description = "description"
    key_requirements = "requirements"
    key_method = "method"
    key_keywords = "keywords"
    key_metrics = "metrics"

    proto = {}
    for e in es:
        #
        # GIVEN
        #

        # print(
        #     f"\nBEGIN Description {e._display_name} ###########################"
        #     f"\n'{e._description}'"
        #     f"\nEND Description {e._display_name} ###############################"
        # )

        #
        # WHEN
        #

        proto = {
            key_id: e().evaluator_id(),
            key_name: e._display_name,
            key_description: "",
            key_requirements: [],
            key_method: "",
            key_keywords: e._keywords,
            key_metrics: e._metrics_meta.to_dict(),
        }

        # camel_case_name = (
        #     e()
        #     ._display_name.replace(" ", "")
        #     .replace("-", "")
        #     .replace("(", "")
        #     .replace(")", "")
        #     .replace(":", "")
        #     .lower()
        # )
        # print(f"\nCamel case name: {camel_case_name}")

        # requirements
        for k in e._keywords:
            if k.startswith("requires"):
                s = k.replace("requires_", "").replace("_", " ")
                proto[key_requirements].append(s)

        # description
        proto[key_description] = (
            e._description.split("**Description**:")[1].split("**Method**:")[0].strip()
        )
        proto[key_description] = proto[key_description].replace("\n- ", "\n")
        proto[key_description] = proto[key_description].replace("\n", " ")
        proto[key_description] = proto[key_description].replace('"', "'")

        # method
        def process_raw_method(method_str: str) -> list[str]:
            if not method_str:
                return []

            lines = method_str.splitlines(keepends=True)

            result = []
            current_item = []

            for line in lines:
                stripped_line = line.lstrip()
                line = line.replace("\n", " ").replace('"', "'")
                if stripped_line.startswith("- ") or stripped_line.startswith("* "):
                    if current_item:
                        result.append("".join(current_item).strip())
                        current_item = []
                    current_item.append(line[2:])
                else:
                    current_item.append(line)

            if current_item:
                result.append("".join(current_item).strip())

            return result

        if e._description and e._description.find("**Method**:") != -1:
            raw_method_str = (
                e._description.split("**Method**:")[1].split("**Metrics**")[0].strip()
            )
            raw_method = process_raw_method(raw_method_str)
        else:
            raw_method = []

        proto[key_method] = raw_method

        # generate Go source
        go_str_requirements = ""
        for r in proto[key_requirements]:
            go_str_requirements += f'"{r}", '
        go_str_requirements = go_str_requirements[:-2]

        go_src_method = ""
        if proto[key_method]:
            for m in proto[key_method]:
                go_src_method += f'        "{m}",\n '

        go_src_metrics = ""
        for m in proto[key_metrics].values():
            go_src_metric_description = m["description"].replace("'", "''")
            if m["threshold"] in [math.inf, "inf"]:
                m["threshold"] = "math.Inf(1)"
            elif m["threshold"] in [-math.inf, "-inf"]:
                m["threshold"] = "math.Inf(-1)"
            else:
                m["threshold"] = m["threshold"]
            go_src_metrics += f"""
            {{
                ID:          "{m["key"]}",
                Name:        "{m["display_name"]}",
                Description: "{go_src_metric_description}",
                DefaultThreshold:   {m["threshold"]},
                HigherIsBetter: {str(m["higher_is_better"]).lower()},
                IsPrimaryMetric: {str(m["is_primary_metric"]).lower()},
            }},
    """
        go_src += f"""
        "{e().evaluator_id()}": {{
        ID:          "{e().evaluator_id()}",
        Requirements: []string{{
            {go_str_requirements},
        }},
        Name:        "{e._display_name}",
        Description: "{proto[key_description]}",
        Method:      []string{{
    {go_src_method}
        }},
        Metrics: []wfReportDataMetric{{
            {go_src_metrics}
        }},
    }},
    """

    go_src += "}"

    # save Go source to file
    go_src_file = tmp_path / "evaluator_descriptors.go"
    with open(go_src_file, "w") as f:
        f.write(go_src)

    #
    # THEN
    #
    # print(json.dumps(proto, indent=2))
    print("\nBEGIN Go source: ########################################")
    print(go_src)
    print("\nEND Go source: ########################################")
    assert go_src


@pytest.mark.parametrize(
    "e",
    _ALL_EVALUATORS,
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_e_descriptor(e):
    #
    # GIVEN
    #
    print(f"\n\n# Evaluating: {e.evaluator_id()} {30 * '#'}")

    #
    # WHEN
    #

    descriptor = e().as_descriptor()

    #
    # THEN
    #
    print(f"\nDescriptor:\n{json.dumps(e().as_descriptor().dump(), indent=2)}")

    assert e._tagline
    assert descriptor.tagline


@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_e_taglines():
    #
    # GIVEN
    #
    print(f"\n# Taglines {50 * '#'}\n")

    #
    # WHEN
    #
    for e in _ALL_EVALUATORS:
        print(f"{e().display_name}:\n  {e().tagline}")

    #
    # THEN
    #
    print(f"\n{50 * '#'}")


@pytest.mark.parametrize(
    "e",
    _ALL_EVALUATORS,
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_e_description_assembly(e):
    #
    # GIVEN
    #
    print(f"\nDescriptor:\n{json.dumps(e().as_descriptor().dump(), indent=2)}")

    #
    # WHEN
    #
    print(f"\n=== {e._display_name} ===\n{e._description}\n=== {e._display_name} ===\n")

    #
    # THEN
    #
    assert e._description
    assert "Expected Answer" in e._description
    assert "Metric" in e._description
    assert "Problems" in e._description
    assert "Insights" in e._description
    assert "parameters" in e._description

    #
    # WHEN conversion from Markdown to HTML
    #
    md_description = markdown.markdown(
        text=e._description,
        extensions=["markdown.extensions.tables", "markdown.extensions.fenced_code"],
    )
    print(
        f"\n=== HTML {e._display_name} ==="
        f"\n{md_description}"
        f"\n=== HTML {e._display_name} ===\n"
    )


@pytest.mark.skip(reason="Documentation tool")
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_doc_evaluators_rst_overview_requirements():
    #
    # GIVEN
    #
    evaluators = _ALL_EVALUATORS
    # sort evaluators by display name
    # TODO sorted(evaluators, lambda e: e._display_name)

    # max evaluator display name length
    max_e_display_name_lng = 30
    max_dns = max([len(e._display_name) for e in evaluators])
    max_dns = min(max_dns, max_e_display_name_lng)

    yes_str = "✓"

    e_attrs_names = [
        "Evaluator" + (" " * (max_dns - len("Evaluator"))),
        "LLM",
        "RAG",
        "J",  # judge
        "Q",  # question (prompt)
        "EA",  # expected answer (ground truth)
        "RC",  # retrieved context
        "AA",  # actual answer
        "C",  # constraints
    ]

    # separator row
    s_row = ""
    for e in e_attrs_names:
        s_row += f"+{'-' * (len(e) + 2)}"
    s_row += "+"

    # head evaluators attributes row
    h_row = ""
    for e in e_attrs_names:
        h_row += f"| {e} "
    h_row += "|"

    # = row
    q_row = ""
    for e in e_attrs_names:
        q_row += f"+{'=' * (len(e) + 2)}"
    q_row += "+"

    #
    # WHEN
    #
    print("\n")
    print(s_row)
    print(h_row)
    print(q_row)
    for e in evaluators:
        #
        # THEN
        #

        # evaluator name ellipsis
        e_display_name = e._display_name
        if len(e_display_name) > max_e_display_name_lng:
            e_display_name = e_display_name[: max_e_display_name_lng - 3] + "..."
        e_name = e_display_name + (" " * (max_dns - len(e_display_name)))

        def _padded_y(attr_name: str, yes: bool = True) -> str:
            y_or_n = yes_str if yes else ""
            attr_w = len(attr_name)
            return y_or_n + (" " * (attr_w - len(y_or_n)))

        e_keywords = e().keywords

        e_attrs = {
            "Evaluator": e_name,
            "LLM": _padded_y("LLM", e8s.KEYWORD_EVALUATES_LLM in e_keywords),
            "RAG": _padded_y("RAG", e8s.KEYWORD_EVALUATES_RAG in e_keywords),
            "J": _padded_y("J", e8s.KEYWORD_RQ_J in e_keywords),
            "Q": _padded_y("Q", e8s.KEYWORD_RQ_P in e_keywords),
            "EA": _padded_y("EA", e8s.KEYWORD_RQ_EA in e_keywords),
            "RC": _padded_y("RC", e8s.KEYWORD_RQ_RC in e_keywords),
            "AA": _padded_y("AA", e8s.KEYWORD_RQ_AA in e_keywords),
            "C": _padded_y("C", e8s.KEYWORD_RQ_C in e_keywords),
        }
        # values row
        v_row = ""
        for v in e_attrs.values():
            v_row += f"| {v} "
        v_row += "|"
        print(v_row)
        print(s_row)

    # legend
    print(
        """
Legend:

* **LLM** - evaluates Language Model (LLM) models.
* **RAG** - evaluates Retrieval Augmented Generation (RAG) models.
* **J** - evaluator requires an LLM judge.
* **Q** - evaluator requires question (prompt).
* **EA** - evaluator requires expected answer (ground truth).
* **RC** - evaluator requires retrieved context.
* **AA** - evaluator requires actual answer.
* **C** - evaluator requires constraints.
"""
    )


@pytest.mark.skip(reason="Documentation tool")
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_doc_metrics_as_md_or_rst():
    #
    # GIVEN
    #

    #
    # WHEN
    #
    for e in [
        e_g_e.EncodingGuardrailEvaluator,
    ]:
        #
        # THEN
        #
        assert e().get_evaluation_metrics()
        for m in e().get_evaluation_metrics().to_list():
            print(f"\n{m.to_md(to_rst=True)}")
            assert m.to_md()


@pytest.mark.skip(reason="Mock tool")
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_guardrail_cfg_mock(tmp_path):
    #
    # GIVEN
    #
    guardrail_cfg_dict = {"guardrail_config": []}

    #
    # WHEN
    #
    for e in _ALL_EVALUATORS:
        #
        # THEN
        #
        m = e().get_evaluation_metrics().get_primary_metric()
        assert m
        print(f"{e.evaluator_id()}")
        print(f"{m.key}")
        print(f"{m.threshold}")
        guardrail_cfg_dict["guardrail_config"].append(
            {
                "evaluator_id": e.evaluator_id(),
                "metric_key": m.key,
                "calibrated_threshold": m.threshold,
            }
        )

    # save guardrail config to JSon file
    guardrail_cfg_file = tmp_path / "guardrail_config.json"
    with open(guardrail_cfg_file, "w") as f:
        json.dump(guardrail_cfg_dict, f, indent=2)


@pytest.mark.skip(reason="Documentation tool")
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_doc_md_checklist():
    #
    # GIVEN
    #

    #
    # WHEN
    #
    for e in _ALL_EVALUATORS:
        #
        # THEN
        #
        print(f"* [ ] {e._display_name}")


#
# H2O Eval Studio configuration
#


@pytest.mark.documentation
@pytest.mark.h2o_sonar
def test_gen_h2o_es_test_classes_config(tmp_path):
    """Generate H2O Eval Studio test classes configuration file from the evaluators
    metadata:

    The following JSON is a representation of a *single* TestClass:

    ```json
    {
        "id": "nist-ai-rmf-accountable-and-transparent",
        "display_name": "Accountable and Transparent",
        "description": "Trustworthy AI depends upon ... confidence in the AI system.",
        "evaluators": [
        "h2o_sonar.evaluators.rag_answer_relevancy_evaluator.AnswerRelevancyEvaluator",
        "h2o_sonar.evaluators.pii_leakage_evaluator.PiiLeakageEvaluator",
        "h2o_sonar.evaluators.sexism_byop_evaluator.SexismByopEvaluator",
        "h2o_sonar.evaluators.stereotype_byop_evaluator.StereotypeByopEvaluator",
        "h2o_sonar.evaluators.rag_tokens_presence_evaluator.RagStrStrEvaluator",
        "h2o_sonar.evaluators.toxicity_evaluator.ToxicityEvaluator",
        "h2o_sonar.evaluators.fairness_bias_evaluator.FairnessBiasEvaluator",
        ],
        "recommended_tests": null,
        "type": "STANDARD",
        "tags": [
        "NIST"
        ]
    }
    ```

    # Fields description

    | Field | Type | Description |
    | --- | --- | --- |
    | id | `str` |
      The unique identifier of the test case. Must be unique across all the test cases.
      Format as shown above is preferred, UUIDs are an option if needed. |
    | display_name | `str` |
      The name of the test case. This is the name that will be displayed in the UI. |
    | description |`str | None` |
      A description of the test case. This is the description that will be displayed
      in the UI. |
    | evaluators | `list[str]` |
      A list of evaluators that are part of this test case. Each evaluator is
      a fully qualified class name. |
    | recommended_tests | `list[str] | None` |
      Can be omitted for now. In the future, it will be used to recommend `Tests` for
      the `TestClass`. |
    | type | `str` |
      The type of the `TestClass`. `TestClasses` are sorted into "Alza" categories
      based on this. See below for more details. |
    | tags | `list[str] | None` |
      A list of tags that are associated with the TestClass. UI currently uses these
      to filter the TestClasses into the Standard -> NIST and Standard -> SR 11-7.
      But this can be changed. See below for more details.

    # TestClass types

    The valid values for the `type` field are:

    - `STANDARD`:
      - corresponds to the `Suites` category in the "Alza" filter
      - on the [evaluators screen](https://eval-studio.h2o.dev/evaluators)
      UI looks up for the tags `NIST` and `SR 11-7` to filter the `TestClasses` based
      on the selected `Evaluation standard`. `TestClasses` of `STANDARD` type with no
      tags are still displayed under `All`
      - `NIST` and `SR 11-7` badges in the UI are displayed based on the `tags` of
      the `TestClass`
    - `ROLE`:
        - currently unused, the initial idea was that `TestClasses` of this type would
        be used to group `TestClasses` based on the role of the evaluator,
         i.e. Model Validator, Developer, etc.
    - `PROBLEM`:
        - corresponds to the `Problems` category in the "Alza" filter
        - on the [evaluators screen](https://eval-studio.h2o.dev/evaluators)
        `TestClasses` with this type are displayed under the `Problems` tab
        - `PROBLEM` badge in the UI is hardcoded, meaning it is displayed for all
        `TestClasses` of this type
    - `PURPOSE`:
        - corresponds to the `Purpose` category in the "Alza" filter
        - on the [evaluators screen](https://eval-studio.h2o.dev/evaluators)
         `TestClasses` with this type are displayed under the `Purpose` tab
        - `PURPOSE` badge in the UI is hardcoded, meaning it is displayed for all
         `TestClasses` of this type
    - `METHOD`:
        - corresponds to the `Method` category in the "Alza" filter
        - currently not displayed on the
        [evaluators screen](https://eval-studio.h2o.dev/evaluators)
        - in the future, we might add a new tab, with additional filtering, similar
        to how we handle `STANDARD` type today
    - `METHOD_TYPE`:
        - corresponds to the `Method` category in the "Alza" filter
        - currently not displayed on the
        [evaluators screen](https://eval-studio.h2o.dev/evaluators)
        - in the future, we might add a new tab, with additional filtering, similar
        to how we handle `STANDARD` type today

    # TestClasses bootstrapfile

    The bootstrap file for the TestClasses should look like:
    ```json
    "test_classes": [
        {
            // TestClass
        },
        ...
        {
            // TestClass
        }
    ]
    ```

    The file is going to be present in the Eval Studio repo, if it is reasonably small,
    or it is going to be put to AWS S3. The naming might follow the bootstrap file
    naming convention, e.g. `s3://eval-studio-bootstrap/test_classes/2024_10_10.json`.
    The file will be parsed on every start of the Eval Studio server. Users will be
    bootstrapped every time the file changes (this needs to be thought through a bit,
    although there are obvious options, such as persisting the hash of the file in DB
    next to subjects, etc.)."""

    #
    # GIVEN
    #

    # keyword groups for which to generate test classes configuration
    input_keyword_groups = []

    class EvalStudioTestClassType(enum.Enum):
        # NIST AI RMF, SR 11-7, ... and other standards
        STANDARD = "STANDARD"
        # regulator, model validator, developer, ...
        ROLE = "ROLE"
        # problem type
        PROBLEM = "PROBLEM"
        # purpose type
        PURPOSE = "PURPOSE"
        # method type
        METHOD = "METHOD"
        # method type
        METHOD_TYPE = "METHOD_TYPE"

    keyword_group_prefix_2_es_tag = {
        e8s.PREFIX_NIST_AI_RMF: "NIST",
        e8s.PREFIX_SR_11_7: "SR 11-7",
        e8s.PREFIX_PROBLEM_TYPE: "PROBLEM",
        e8s.PREFIX_EVAL_ROLE: "ROLE",
        e8s.PREFIX_ES_PURPOSE: "PURPOSE",
        e8s.PREFIX_EVAL_METHOD: "METHOD",
        e8s.PREFIX_EVAL_METHOD_TYPE: "METHOD_TYPE",
    }

    keyword_group_prefix_2_test_class_type = {
        e8s.PREFIX_PROBLEM_TYPE: EvalStudioTestClassType.PROBLEM,
        e8s.PREFIX_NIST_AI_RMF: EvalStudioTestClassType.STANDARD,
        e8s.PREFIX_SR_11_7: EvalStudioTestClassType.STANDARD,
        e8s.PREFIX_EVAL_ROLE: EvalStudioTestClassType.ROLE,
        e8s.PREFIX_ES_PURPOSE: EvalStudioTestClassType.PURPOSE,
        e8s.PREFIX_EVAL_METHOD: EvalStudioTestClassType.METHOD,
        e8s.PREFIX_EVAL_METHOD_TYPE: EvalStudioTestClassType.METHOD_TYPE,
    }

    #
    # WHEN
    #

    # test classes file to generate
    test_classes_config_file = (
        pathlib.Path(os.getenv("TEST_CLASSES_CONFIG_FILE"))
        if os.getenv("TEST_CLASSES_CONFIG_FILE")
        else tmp_path / "h2o_eval_studio_test_classes.json"
    )

    #
    # index keywords and evaluators
    #
    input_keyword_groups = e8s.KEYWORD_GROUPS.groups

    #
    # generate test classes configuration file
    #

    test_classes_dict = {"test_classes": []}

    for kg in input_keyword_groups:
        for k in kg.keywords:
            test_class_dict = {
                "id": k.key,
                "display_name": k.name,
                "description": k.description,
                "evaluators": [],
                # no longer needed: "recommended_tests": None,
                "type": keyword_group_prefix_2_test_class_type[kg.prefix].value,
                "tags": [keyword_group_prefix_2_es_tag[kg.prefix]],
            }

            # find evaluators for the keyword
            matching_evaluators = evaluate.list_evaluators(keywords=[k.key])
            if matching_evaluators:
                for e in matching_evaluators:
                    if k.key in e.keywords:
                        test_class_dict["evaluators"].append(e.id)
            else:
                print(f"WARNING: No evaluators found for keyword {k.key} ({k.name})")

            # filter out all test classes without evaluators
            if not test_class_dict["evaluators"]:
                continue

            # add new test class
            test_classes_dict["test_classes"].append(test_class_dict)

    with open(test_classes_config_file, "w") as f:
        json.dump(test_classes_dict, f, indent=2)

    #
    # THEN
    #
    print(
        f"Generated test classes configuration file:"
        f"\n  file://{test_classes_config_file}"
    )


@pytest.mark.skip(reason="This test is just documentation example verification")
@pytest.mark.documentation
@pytest.mark.h2o_sonar
@pytest.mark.parametrize(
    "dataset, target_col",
    [
        (
            "creditcard.csv",
            "LIMIT_BAL",
        ),
    ],
)
def test_byoe_getting_started(tmpdir, dataset, target_col):
    # GIVEN
    (dataset_path, explainable_model, target_col) = test_utils.create_sklearn_model(
        dataset_name=dataset, target_col=target_col
    )
    explainer = example_morris_sa_explainer.ExampleMorrisSensitivityAnalysisExplainer

    # WHEN
    explainer_id = interpret.register_explainer(explainer_class=explainer)

    try:
        interpretation = interpret.run_interpretation(
            dataset=dataset_path,
            model=explainable_model,
            target_col=target_col,
            explainers=[explainer_id],
            results_location=tmpdir,
            log_level=loggers.DEBUG,
        )

        result = interpretation.get_explainer_result(explainer_id=explainer_id)

        data = result.data()
        print(data)

        result.plot(file_path=os.path.join(tmpdir, "morris_sa_plot.png"))

        result.zip(file_path=os.path.join(tmpdir, "morris_sa_archive.zip"))

        # THEN
        print(f"Interpretation:\n{interpretation}")
        assert interpretation
        assert interpretation.is_explainer_scheduled()
        assert interpretation.is_explainer_finished()
        assert interpretation.is_explainer_successful()
        assert not interpretation.is_explainer_failed()
        assert interpretation.get_scheduled_explainer_ids()
        assert interpretation.get_finished_explainer_ids()
        assert interpretation.get_successful_explainer_ids()
        assert not interpretation.get_failed_explainer_ids()
        assert interpretation.get_explainer_result(explainer.explainer_id())
    finally:
        interpret.unregister_explainer(explainer_id)


@pytest.mark.skip(reason="This test is just documentation example verification")
@pytest.mark.h2o_sonar
def test_byoe_getting_started_cli(tmpdir):
    # GIVEN
    model_path = "data/predictive/models/creditcard-binomial-sklearn-gbm.pkl"

    src_explainer_path = "tests/explainers/templates/template_featimp_explainer.py"
    test_explainer_module = "byoe_test_explainer"
    test_explainer_file_name = "byoe_test_explainer.py"
    test_explainer_path = os.path.join(tmpdir, test_explainer_file_name)
    test_explainer_class_name = "TemplateFeatureImportanceExplainer"
    test_explainer_descr = f"{test_explainer_module}::{test_explainer_class_name}"
    test_explainer_id = f"{test_explainer_module}.{test_explainer_class_name}"
    # prepare custom BYOE recipe to tmpdir under different name
    if not os.path.isfile(src_explainer_path):
        raise ValueError(f"Invalid explainer file: {src_explainer_path}")
    shutil.copyfile(src=src_explainer_path, dst=test_explainer_path)

    # prepare configuration
    (
        dataset_path,
        target_col,
        mock_model_path,
        config_path,
    ) = given_cli(
        tmpdir=tmpdir,
        h2o_auto_start=False,
        # configure H2O Sonar to register test BYOE
        custom_explainers=[test_explainer_descr],
    )

    cli_cmd = (
        ["h2o-sonar"]
        if os.system("which h2o-sonar") == 0
        else ["python", "h2o_sonar/h2o_sonar_cli.py"]
    )
    child_env = os.environ.copy()
    # add the root of the repo to PYTHONPATH so that the CLI can load model class
    python_path = "."
    # add tmpdir with custom BYOE to PYTHONPATH
    python_path = f"{python_path}:{tmpdir}"

    child_env["PYTHONPATH"] = python_path

    # WHEN run interpretation
    cmd = cli_cmd + [
        "run",
        "interpretation",
        "--dataset",
        dataset_path,
        "--target-col",
        target_col,
        "--model",
        model_path,
        "--results-location",
        tmpdir,
        "--config-path",
        config_path,
        "--explainers",
        test_explainer_id,
        "--log-level",
        "debug",
    ]

    print(f"\nRunning interpretation via CLI:\n{cmd}\n")
    p = subprocess.Popen(cmd, env=child_env)
    p.wait()

    # THEN run interpretation
    p_tree = str(os.popen(f"find {tmpdir}").read())
    print(p_tree)
    assert test_explainer_class_name in p_tree
    assert "featimp_class_A.json" in p_tree


@pytest.mark.skip(reason="This test is just demo verification")
@pytest.mark.documentation
@pytest.mark.h2o_sonar
def test_doc_rst_index(tmpdir):
    dataset_path = test_utils.find_locally("data/predictive/creditcard_12_actuals.csv")

    target_column = "AGE"
    results_path = str(tmpdir)
    archive_path = os.path.join(results_path, "my.zip")

    #
    # BEGIN: DOCUMENTATION
    #

    # dataset

    import pandas

    dataset = pandas.read_csv(dataset_path)
    (X, y) = dataset.drop(target_column, axis=1), dataset[target_column]

    # model

    from sklearn import ensemble

    model = ensemble.GradientBoostingClassifier(learning_rate=0.1)
    model.fit(X, y)

    # interpretation

    from h2o_sonar import interpret

    interpretation = interpret.run_interpretation(
        dataset=dataset_path,
        model=model,
        used_features=list(X.columns),
        target_col=target_column,
        results_location=results_path,
    )

    # result

    print(interpretation)  # or interpretation.to_html()

    # get explanation created by the first explainer of the interpretation
    explanation = interpretation.get_explainer_result(
        interpretation.get_finished_explainer_ids()[0]
    )

    # show explanation summary
    print(explanation.summary())
    # show explanation data
    print(explanation.data(feature_name="EDUCATION", category="disparity"))
    # get explanation plot
    print(explanation.plot(feature_name="EDUCATION"))
    # show explainer log
    print(explanation.log(path=results_path))
    # store all explanation artifacts as ZIP archive
    explanation.zip(file_path=archive_path)

    #
    # END: DOCUMENTATION
    #


@pytest.mark.skip(reason="This a documentation generator")
def test_gen_rst_evaluators():
    """Generation of evaluators documentation - not all evaluators are included
    (templates and disabled to be excluded), this is why Sphinx directives are not used.

    """
    #
    # GIVEN
    #
    modules = [
        "abc_byop_evaluator",
        "agentic_fact_check_evaluator",
        "bleu_evaluator",
        "classification_evaluator",
        "contact_information_byop_evaluator",
        "fairness_bias_evaluator",
        "gptscore_evaluator",
        "gptscore_machine_translation_evaluator",
        "gptscore_question_answering_evaluator",
        "gptscore_summary_without_reference_evaluator",
        "gptscore_summary_with_reference_evaluator",
        "language_mismatch_byop_evaluator",
        "parameterizable_byop_evaluator",
        "perplexity_evaluator",
        "pii_leakage_evaluator",
        "rag_answer_correctness_evaluator",
        "rag_answer_relevancy_evaluator",
        "rag_answer_relevancy_no_judge_evaluator",
        "rag_answer_similarity_evaluator",
        "rag_chunk_relevancy_evaluator",
        "rag_context_precision_evaluator",
        "rag_context_recall_evaluator",
        "rag_context_relevancy_evaluator",
        "rag_faithfulness_evaluator",
        "rag_groundedness_evaluator",
        "rag_hallucination_evaluator",
        "rag_ragas_evaluator",
        "rag_tokens_presence_evaluator",
        "rouge_evaluator",
        "sensitive_data_leakage_evaluator",
        "sexism_byop_evaluator",
        "stereotype_byop_evaluator",
        "summarization_byop_evaluator",
        "summarization_evaluator",
        "toxicity_evaluator",
    ]

    #
    # WHEN
    #
    print()
    for m in modules:
        print()
        s = f"h2o_sonar.evaluators.{m} module"
        print(s)
        print("^" * len(s))
        print()
        print(f".. automodule:: h2o_sonar.evaluators.{m}")
        print("    :members:")
        print("    :undoc-members:")
        print("    :show-inheritance:")


@pytest.mark.skip(reason="This a documentation generator")
@pytest.mark.h2o_sonar
@pytest.mark.documentation
def test_list_evaluators_alpha():
    #
    # GIVEN
    #
    e_lst = evaluate.list_evaluators()
    e_ch = [e.display_name for e in e_lst if e8s.KEYWORD_CAP_AH in e.keywords]
    e_names = [e.display_name for e in e_lst]
    e_names.sort()

    #
    # WHEN
    #
    print("Evaluators:")
    for i, n in enumerate(e_names, 1):
        ch = " *" if n in e_ch else ""
        print(f"{i:>2}. {n}{ch}")


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
