# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
from html import parser as html_parser

import airium
import datatable
import pandas as pd
import pytest

from h2o_sonar.lib.api import insights
from h2o_sonar.lib.api import problems
from h2o_sonar.utils.problem_detection import get_feature_importance_problems


@pytest.mark.parametrize(
    "value,precision,expected",
    [
        # floats
        [1.0, 1, "1.0"],
        [1.0, 4, "1.0"],
        [0.109274, 2, "0.11"],
        [6.708302, 2, "6.71"],
        [0.813976, 2, "0.81"],
        [52.544132, 2, "52.54"],
        [25.768531, 2, "25.77"],
        [-3.213123, 2, "-3.21"],
        [0.109274, 3, "0.109"],
        [6.708302, 3, "6.708"],
        [0.813976, 3, "0.814"],
        [52.544132, 3, "52.544"],
        [25.768531, 3, "25.769"],
        [-3.213123, 3, "-3.213"],
        # ints
        [1, 0, "1.0"],
        [1, 1, "1.0"],
        [2, 2, "2.0"],
        # strings
        ["0.109274", 2, "0.11"],
        ["6.708302", 2, "6.71"],
        ["0.813976", 2, "0.81"],
        ["52.544132", 2, "52.54"],
        ["25.768531", 2, "25.77"],
        ["-3.213123", 2, "-3.21"],
        ["0.109274", 3, "0.109"],
        ["6.708302", 3, "6.708"],
        ["0.813976", 3, "0.814"],
        ["52.544132", 3, "52.544"],
        ["25.768531", 3, "25.769"],
        ["-3.213123", 3, "-3.213"],
        # any
        [None, 2, "None"],
        [True, 2, "True"],
        [[], 2, "[]"],
    ],
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_rounding(value, precision, expected):
    """Test rounding of the float numbers to the 2 decimal places."""

    #
    # GIVEN
    #

    #
    # WHEN
    #
    rounded = insights.r(value=value, precision=precision)

    #
    # THEN
    #
    assert rounded == expected


def _given_html_b_e() -> str:
    html_ast = airium.Airium()
    with html_ast.h1():
        html_ast("BTitleE")
    with html_ast.p():
        html_ast("B.1/3 Paragraph 1 prefix content.E")
        with html_ast.a(href="https://www.h2o.ai"):
            html_ast("BLINK.2/3 Link text ELINK")
        html_ast.br()
        with html_ast.b():
            html_ast("B.3/3 Bold textE")
        html_ast("B.4/4 Paragraph 1 suffix content.E")
    with html_ast.p():
        html_ast("B.1/3 Paragraph 2 prefix content.E")
        with html_ast.a(href="https://www.h2o.ai"):
            html_ast("B.2/3 Link text E")
        html_ast.br()
        with html_ast.b():
            html_ast("B.3/3 Bold textE")
        html_ast("B.4/4 Paragraph 2 suffix content.E")
    html_str = str(html_ast)

    return html_str


@pytest.mark.parametrize("html_str,expected_txt", [(_given_html_b_e(), "")])
@pytest.mark.h2o_sonar
def test_problem_html_to_txt(html_str, expected_txt):
    """Test conversion of problems description in the HTML format to the plain
    text format. In other words, the HTML is not the whole HTML document, but
    just the content of the body tag.

    """

    #
    # GIVEN
    #

    class HTMLFilter(html_parser.HTMLParser):
        text = ""

        def handle_starttag(self, tag, attrs):
            print(f"Start tag: {tag} ({type(tag)})")
            if tag in ["h1", "p"]:
                self.text += "\n"

        def handle_data(self, data):
            data_stripped = data.strip()
            print(f"Data: '{data}' -> '{data_stripped}'")
            self.text += data_stripped

    #
    # WHEN
    #
    f = HTMLFilter()
    f.feed(html_str)
    txt = f.text

    #
    # THEN
    #
    print("#" * 80)
    print(f"HTML content:\n'{html_str}'")
    print("#" * 80)
    print(f"TXT content:\n'{txt}'")
    print("#" * 80)


@pytest.mark.h2o_sonar
@pytest.mark.parametrize(
    "row,should_detect_problems",
    [
        [
            [
                0.109274,
                6.708302,
                0.813976,
                52.544132,
                25.768531,
                -3.213123,
            ],
            True,
        ],
        [
            [
                1.17201791,
                1.275785312,
                1.768504249,
                1.52707261,
                1.047581080,
                -0.12122121,
            ],
            False,
        ],
        [
            [
                29.17201791,
                54.52707261,
                1.275785312,
                3.768504249,
                1.047581080,
                -1.22222222,
            ],
            True,
        ],
    ],
)
def test_problem_detection(row, should_detect_problems):
    # Test for bugfixes #626 and #648

    #
    # GIVEN
    #
    frame = pd.DataFrame([row], columns=["A", "B", "C", "D", "E", "bias"])
    datatable_frame: datatable.Frame = datatable.Frame(frame)

    #
    # WHEN
    #
    problems_list: list[problems.ProblemAndAction] = get_feature_importance_problems(
        {"Label A": datatable_frame}, 0.5, "ExplainerId", "ExplainerDispName"
    )

    #
    # THEN
    #
    if should_detect_problems:
        assert problems_list
        assert len(problems_list) == 1
    else:
        assert len(problems_list) == 0


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
