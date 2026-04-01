#!/usr/bin/env python3
# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
"""H2O Sonar evaluation test suite library librarian.

  ./test-suite-librarian.py import dir
    --library-path=../data/generative/evals_library/h2o-eval-studio-suite-library/in-moonshot

  ./test-suite-librarian.py generate markdown
    --library-path=../data/generative/evals_library/h2o-eval-studio-suite-library.json
    --output-path=markdown.md
    --table

"""
import argparse
import re
import subprocess
import json
import pathlib
import sys
from typing import Tuple

ACTION_CONVERT = "convert"
ACTION_GENERATE = "generate"
ACTION_IMPORT = "import"
ACTION_LINT = "lint"
ACTION_LIST = "list"
ACTION_PATCH = "patch"

ENTITY_DIR = "dir"
ENTITY_TS_URL = "test-suite-urls"
ENTITY_INDEX = "index"

FORMAT_HTML = "html"
FORMAT_MARKDOWN = "markdown"

KEY_TEST_SUITES = "test_suites"
KEY_NAME = "name"
KEY_DESCRIPTION = "description"
KEY_TEST_SUITE_URL = "test_suite_url"  # URL of the library test suite
KEY_REFERENCE_URL = "reference_url"  # URL of the reference like paper, blog, etc.
KEY_SOURCE_URL = "source_url"  # URL of the test suite source in a different format
KEY_LICENSE = "license"  # license of the source document
KEY_EVALUATES = "evaluates"
KEY_PURPOSES = "purposes"
KEY_CATS = "categories"
KEY_ORIGIN = "origin"
KEY_TC_COUNT = "test_case_count"
KEY_T_COUNT = "test_count"

KEYWORD_LLM = "LLM"
KEYWORD_RAG = "RAG"
KEYWORD_AGENT = "agent"

PURPOSE_GENERATE = "generation"
PURPOSE_QA = "Q&A"
PURPOSE_PRIVACY = "privacy"
PURPOSE_FAIRNESS = "fairness"
PURPOSE_SECURITY = "security"
PURPOSE_SUMMARIZE = "summarization"
PURPOSE_CLASSIFY = "classification"

# test suite / test case categories
CAT_TROUBLESHOOTING = "troubleshooting"
CAT_MATH = "math"
CAT_WRITING = "writing"
CAT_PLANNING = "planning"
CAT_EVALUATION = "evaluation"
CAT_KNOWLEDGE = "knowledge"
CAT_INFO_RETRIEVE = "information_retrieval"
CAT_CODING = "coding"
CAT_REASONING = "reasoning"
CAT_Q_A = "question_answering"
CAT_SUMMARIZATION = "summarization"
CAT_RECOMMENDATION = "recommendation"
CAT_HARM = "harm"  # How to code bot? Create dangerous device? Do harm?
CAT_PRIVACY = "privacy"
CAT_FAIRNESS = "fairness"

PT_SUMMARIZATION = "summarization"

URL_BASE_S3_LIB = (
    "https://eval-studio-artifacts.s3.us-east-1.amazonaws.com/"
    "h2o-eval-studio-suite-library/"
)


def _check_and_set_paths(
    library_path: str,
    output_path: str,
):
    if not library_path:
        raise ValueError("Library path is not provided.")
    if not output_path:
        raise ValueError("Output path is not provided.")
    library_path = pathlib.Path(library_path)
    output_path = pathlib.Path(output_path)

    return library_path, output_path


def _load_json_lib_index(library_path: pathlib.Path) -> dict:
    with open(library_path, "r") as file:
        return json.load(file)


def convert_gen(h2o_evals_dir: str, output_dir: str):
    """Convert h2oai/h2o-evals repository to test suite JSon files with:

    - name in JSon
    - description (links to references / source) in JSon
    - URL to the test suite in description

    Input structure (``h2o_evals_dir`` to be to h2o-evals dir:

    - h2o-evals/catalog/[dir w/ name]/JSON_files/
       - multi_choice_[name]_output.json
       - question_type_[name]_output.json
       - token_presence_[name]_output.json

    """
    h2o_evals_dir = "/home/user/h/mli/git/h2o-evals"
    h2o_evals_dir = pathlib.Path(h2o_evals_dir)

    output_dir = (
        "/home/user/h/mli/git/h2o-sonar/data/generative/evals_library/"
        "h2o-eval-studio-suite-library/in-gen"
    )
    output_dir = pathlib.Path(output_dir)
    # create output dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # iterate over the catalog
    for catalog_dir in h2o_evals_dir.glob("catalog/*"):
        if not catalog_dir.is_dir():
            continue

        # iterate over the JSON files
        for json_file in catalog_dir.glob("JSON_files/*.json"):
            if not json_file.is_file():
                continue

            # load JSON file
            with open(json_file, "r") as handle:
                ts_dict = json.load(handle)

            raw_name = json_file.stem
            ts_type = ""
            if raw_name.startswith("multi_choice_"):
                ts_type = " (multi choice)"
            elif raw_name.startswith("question_type_"):
                ts_type = " (question type)"
            elif raw_name.startswith("tokens_presence_"):
                ts_type = " (text matching)"

            name = (
                json_file.stem.replace("_output", "")
                .replace("multi_choice_", "")
                .replace("question_type_", "")
                .replace("token_presence_", "")
                .replace("_", " ")
            )

            new_ts_dict = {
                "name": f"{name}{ts_type}",
                "description": (
                    f"Test suite {name}{ts_type} for RAG evaluation."
                    f"\n\nReference: https://github.com/h2oai/h2o-evals"
                ),
                "tests": ts_dict.get("tests", []),
            }

            # save test suite
            ts_file = output_dir / f"{json_file.stem}.json"
            with open(ts_file, "w") as handle:
                json.dump(new_ts_dict, handle, indent=4)

            print(f"Test suite saved: {ts_file}")


def import_dir_print(input_dir: str):
    if not input_dir:
        raise ValueError("Input directory path specification empty")
    input_dir = pathlib.Path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    lib_test_suites = []
    lib_index = {
        KEY_NAME: "H2O Eval Studio Evaluation Test Suite Library",
        KEY_DESCRIPTION: "H2O Eval Studio evaluation test suite library.",
        KEY_TEST_SUITES: lib_test_suites,
    }

    # list *.json files in the directory
    for path in pathlib.Path(input_dir).iterdir():
        if path.suffix == ".json":

            # load test suite & build
            with open(path, "r") as handle:
                ts_dict = json.load(handle)

            # count test cases
            tc_count = 0
            is_rag = False
            for t in ts_dict.get("tests", []):
                tc_count += len(t.get("test_cases", []))

                # detect RAG
                if t.get("documents", None) and not is_rag:
                    is_rag = True

            evaluates = [KEYWORD_LLM] if not is_rag else [KEYWORD_RAG]

            lib_ts_item = {
                KEY_NAME: ts_dict.get("name"),
                KEY_DESCRIPTION: ts_dict.get("description"),
                KEY_TEST_SUITE_URL: f"{URL_BASE_S3_LIB}{path.name}",
                KEY_REFERENCE_URL: "",
                KEY_SOURCE_URL: "",
                KEY_LICENSE: "",
                KEY_EVALUATES: evaluates,
                KEY_PURPOSES: [PURPOSE_QA],
                KEY_T_COUNT: len(ts_dict.get("tests", [])),
                KEY_TC_COUNT: tc_count,
            }

            lib_test_suites.append(lib_ts_item)

    print(json.dumps(lib_index, indent=4))


def _gen_md_header(branding: str) -> str:
    return (
        f"# {branding} Test Suite Library\n"
        f"This is [{branding}](https://h2o.ai/platform/enterprise-h2ogpte/eval-studio/)"
        f" **test suite** library for LLM, RAG and agent evaluation. "
        f"\n\nTest suites can be used for **question answering**, **privacy**, "
        f"**fairness**, **security**, **summarization** and **classification** "
        f"evaluation. In addition to that test suites "
        f"can be **combined**, **sampled**, **perturbed** and **customized** for "
        f"specific evaluation needs. \n\nTest suites are provided normalized in "
        f"{branding} JSON format - see also [details](#test-suites):\n"
        "\n"
    )


def _gen_md_table(
    test_suites: list,
) -> str:
    md = ""
    t_count = 0
    tc_count = 0
    for test_suite in test_suites:
        t_count += test_suite[KEY_T_COUNT]
        tc_count += test_suite[KEY_TC_COUNT]
        md += (
            f" [{test_suite[KEY_NAME]}]({test_suite[KEY_TEST_SUITE_URL]}) "
            f" ({test_suite[KEY_TC_COUNT]})"
            f" | "
            f"{', '.join(test_suite[KEY_EVALUATES])} | "
            f"{', '.join(test_suite[KEY_PURPOSES])} | "
            f"{test_suite[KEY_T_COUNT]} | "
            f"{test_suite[KEY_TC_COUNT]}\n"
        )

    ts_str = f"({len(test_suites)})" if test_suites else ""
    t_str = f"({t_count})" if t_count else ""
    tc_str = f"({tc_count:,})" if tc_count else ""
    md_header = (
        f" Test Suite {ts_str} | Evaluates | Purposes | "
        f"Tests {t_str} | Test Cases {tc_str} \n"
        " --- | --- | --- | --- | ---\n"
    )

    return md_header + md


def _gen_name_to_md_anchor(name):
    cleaned_name = re.sub(r"[^\w\s-]", "", name).strip()
    anchored_name = cleaned_name.replace(" ", "-")
    return anchored_name.lower()


def _gen_md_ul(test_suites: list) -> Tuple[str, str]:
    # generate markdown unordered list
    md = ""
    for test_suite in test_suites:
        md += f"\n- **[{test_suite[KEY_NAME]}]({test_suite[KEY_TEST_SUITE_URL]})**"

        # description to ul
        description = test_suite.get(KEY_DESCRIPTION, "")
        if description:
            md += "\n   - "
            dd = description.split("\n")
            for d in dd:
                if d.startswith("Reference") or d.startswith("Source"):
                    d = d.replace('"', "")
                    md += f"\n      - {d}"
                else:
                    md += d

        md += (
            f"\n   - Evaluates: **{', '.join(test_suite[KEY_EVALUATES])}**"
            f"\n   - Purposes: **{', '.join(test_suite[KEY_PURPOSES])}**"
            f"\n   - Tests: **{test_suite[KEY_T_COUNT]}**"
            f"\n   - Test Cases: **{test_suite[KEY_TC_COUNT]}**"
            "\n"
        )

    # generate table of contents
    toc = ""
    for test_suite in test_suites:
        name = test_suite[KEY_NAME]
        if name:
            toc += f"- [{name}](#{_gen_name_to_md_anchor(name)})\n"
        else:
            toc += f"- [unnamed](#unnamed)\n"

    return md, toc


def generate_html_index(
    markdown_library_path: str,
    output_path: str,
):
    """Generate HTML index."""
    (library_path, output_path) = _check_and_set_paths(
        library_path=markdown_library_path,
        output_path=output_path,
    )

    pandoc_cmd = (
        f"pandoc -s -f markdown -t html5 -o {output_path} {library_path} -c "
        f'style.css --metadata pagetitle="H2O Eval Studio Test Suite Library"'
    )

    # run pandoc
    print(f"Running pandoc command: {pandoc_cmd}")
    subprocess.run(pandoc_cmd, shell=True)
    if not output_path.exists():
        raise FileNotFoundError(f"Output HTML file was not generated: {output_path}")
    print(f"Pandoc created HTML index:\n  file://{output_path.absolute()}")

    # patch HTML
    with open(output_path, "r") as file:
        html = file.read()
    # find <body> and remove all before
    body_start = html.find("<body>")
    if body_start == -1:
        raise ValueError("HTML body start not found.")
    body_start += len("<body>")
    html = html[body_start:]
    # prepend custom header
    header = """<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" lang="" xml:lang="">
<head>
  <meta charset="utf-8" />
  <meta name="generator" content="pandoc" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes" />
  <title>H2O Eval Studio Test Suite Library</title>
  <link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
</head>
<body style="margin: 10%; margin-top: 5%;">"""
    html = header + html
    # replace <table> with style enriched <table>
    html = html.replace("<table>", '<table class="w3-table-all">')

    # save patched HTML
    with open(output_path, "w") as file:
        file.write(html)


def generate_markdown_index(
    library_path: str,
    output_path: str,
    branding: str,
    table: bool = False,
):
    """Generate Markdown index."""
    (library_path, output_path) = _check_and_set_paths(
        library_path=library_path,
        output_path=output_path,
    )
    lib_index = _load_json_lib_index(library_path=library_path)

    # sort test suites by name
    lib_index[KEY_TEST_SUITES] = sorted(
        lib_index[KEY_TEST_SUITES],
        key=lambda x: x[KEY_NAME].lower(),
    )

    # stats
    ts_count = len(lib_index[KEY_TEST_SUITES])

    md = _gen_md_header(branding)
    if table:
        md += _gen_md_table(lib_index[KEY_TEST_SUITES])
    else:
        md += _gen_md_table(lib_index[KEY_TEST_SUITES])
        (ul, _) = _gen_md_ul(lib_index[KEY_TEST_SUITES])
        md += "## Test Suites\n"
        md += ul

    with open(output_path, "w") as file:
        file.write(md)

    print(f"Markdown index generated:\n  file://{output_path.absolute()}")


def list_test_suite_urls(library_path: str):
    if not library_path:
        raise ValueError("Library path is not provided.")
    library_path = pathlib.Path(library_path)

    lib_index = _load_json_lib_index(library_path=library_path)
    for test_suite in lib_index[KEY_TEST_SUITES]:
        url = test_suite[KEY_TEST_SUITE_URL]
        if url:
            print(url)


def patch_index_json_file(library_path: str, patch_type="categories"):
    if not library_path:
        raise ValueError("Library path is not provided.")
    library_path = pathlib.Path(library_path)

    if "categories" == patch_type:
        lib_dict = _load_json_lib_index(library_path=library_path)
        for ts in lib_dict[KEY_TEST_SUITES]:
            d = ts.get("description", "")
            if "evalgpt" in d:
                ts[KEY_CATS] = [
                    CAT_TROUBLESHOOTING,
                    CAT_MATH,
                    CAT_WRITING,
                    CAT_PLANNING,
                    CAT_EVALUATION,
                    CAT_KNOWLEDGE,
                    CAT_CODING,
                    CAT_REASONING,
                    CAT_SUMMARIZATION,
                    CAT_RECOMMENDATION,
                    CAT_HARM,
                ]
            elif "h2o-evals" in d:
                ts[KEY_CATS] = [
                    CAT_INFO_RETRIEVE,
                ]
            else:
                ts[KEY_CATS] = [
                    CAT_Q_A,
                ]
        with open(library_path, "w") as file:
            json.dump(lib_dict, file, indent=4)

    elif "purpose" == patch_type:
        lib_dict = _load_json_lib_index(library_path=library_path)
        for ts in lib_dict[KEY_TEST_SUITES]:
            purpose = ts.get("purpose", "")
            if purpose:
                ts.pop("purpose", None)
                ts[KEY_PURPOSES] = [purpose]
            origin = ts.get(KEY_ORIGIN, "")
            if not origin:
                ts[KEY_ORIGIN] = "3rd-party"  # "h2oai", "generated", "3rd-party"
        with open(library_path, "w") as file:
            json.dump(lib_dict, file, indent=4)

    elif "json-validity" == patch_type:
        with open(library_path, "r") as file:
            for l in file:
                if KEY_TC_COUNT in l:
                    l = l.replace(",", "")
                elif '"description": \'' in l:
                    l = l.replace('"description": \'', '"description": "')
                    l = l.replace("',", '",')
                    l = l.replace('License: "CC-BY-4.0 license"', "License: CC-BY-4.0")
                    l = l.replace('License: "Apache-2.0"', "License: Apache-2.0")
                    l = l.replace('License: "MIT License"', "License: MIT")
                    l = l.replace('License: "MIT license"', "License: MIT")
                    l = l.replace('License: "CC BY-SA 4.0"', "License: CC BY-SA 4.0")
                    l = l.replace(
                        'License: "GNU General Public License v3.0"',
                        "License: GNU General Public License v3.0",
                    )
                    l = l.replace('License: "CC BY-NC-SA"', "License: CC BY-NC-SA")
                    l = l.replace('Reference: "', "Reference: ")
                    l = l.replace('Source: "', "Source: ")
                    l = l.replace('"",', '",')
                    l = l.replace("\\'", "'")
                    l = l.replace('".\\nSource', "\\nSource")
                print(l, end="")


def main() -> int:
    """Main function."""
    parser = argparse.ArgumentParser(
        description=(
            f"""H2O Sonar evaluation test suite library librarian.

optional arguments per action and entity:

  {ACTION_GENERATE} {FORMAT_MARKDOWN}:
    --library_path   path to source library JSon index file
    --output_path    path where to generate Markdown index for the library
    --table          generate table index, default is unordered list

  {ACTION_GENERATE} {FORMAT_HTML}:
    --library_path   path to source library JSon index file
    --output_path    path where to generate HTML index for the library

  {ACTION_LIST} {ENTITY_TS_URL}:
    --library_path   path to source library JSon index file

  {ACTION_IMPORT} {ENTITY_DIR}:
    --dir            path to the dir with test suites for which to print the index

  {ACTION_CONVERT} {ENTITY_DIR}:
    --dir            path to h2oai/h2o-evals GitHub repository dir with test suites
    --output_path    directory path where to save the converted test suites

"""
        ),
        epilog=(
            """Examples:
  test-suite-librarian.py generate html
    --library_path data/h2o-eval-suite-library/index.json
    --output_path data/h2o-eval-suite-library/index.html
  test-suite-librarian.py list test-suite-urls
    --library_path data/h2o-eval-suite-library/index.json
"""
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.prog = "test-suite-librarian.py"

    # positional arguments
    parser.add_argument(
        "action",
        choices=[
            ACTION_GENERATE,
            ACTION_IMPORT,
            ACTION_CONVERT,
            ACTION_LIST,
            ACTION_PATCH,
        ],
        help="Action to perform.",
    )
    parser.add_argument(
        "entity",
        choices=[FORMAT_HTML, FORMAT_MARKDOWN, ENTITY_DIR, ENTITY_TS_URL, ENTITY_INDEX],
        help="Entity to generate or process.",
    )

    # optional arguments
    parser.add_argument(
        "--library-path",
        type=str,
        default="",
        help="Path to source library JSon index file.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="",
        help="Path where to generate index for the library in a format.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="",
        help="Directory with test suites for which to print the index.",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="Whether to generate the table or unordered list.",
    )
    args = parser.parse_args()

    # handling
    action = args.action
    entity = args.entity
    branding = "H2O Eval Studio"

    if action == ACTION_GENERATE:
        if entity == FORMAT_MARKDOWN:
            generate_markdown_index(
                library_path=args.library_path,
                output_path=args.output_path,
                table=args.table,
                branding=branding,
            )
        elif entity == FORMAT_HTML:
            generate_html_index(
                markdown_library_path=args.library_path,
                output_path=args.output_path,
            )
        else:
            raise ValueError(f"Unknown entity: {entity}")
    elif action == ACTION_LIST:
        if entity == ENTITY_TS_URL:
            list_test_suite_urls(
                library_path=args.library_path,
            )
        else:
            raise ValueError(f"Unknown entity: {entity}")
    elif action == ACTION_PATCH:
        if entity == ENTITY_INDEX:
            patch_index_json_file(
                library_path=args.library_path,
            )
        else:
            raise ValueError(f"Unknown entity: {entity}")
    elif action == ACTION_IMPORT:
        if entity == ENTITY_DIR:
            import_dir_print(
                input_dir=args.dir,
            )
    elif action == ACTION_CONVERT:
        if entity == ENTITY_DIR:
            convert_gen(
                h2o_evals_dir=args.dir,
                output_dir=args.output_path,
            )
    else:
        raise RuntimeError(f"Unsupported action: {action}")

    return 0


if __name__ == "__main__":
    """Librarian:

    - by default the location of the library within h2oai/h2o-sonar is:
        data/h2o-eval-suite-library
    - S3:
        s3://eval-studio-artifacts/h2o-eval-studio-suite-library/

    ."""
    sys.exit(main())
