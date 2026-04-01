# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import pytest

from h2o_sonar import loggers
from h2o_sonar.evaluators import rag_tokens_presence_evaluator as evaluator
from h2o_sonar.utils import parsing


@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_symbol_table():
    """Test symbol table."""

    #
    # GIVEN
    #
    s = parsing.ConditionSymbol.AND
    print(f"name: {s.name}")
    print(f"value: {s.value}")
    print(f"precedence: {s.precedence}")

    #
    # WHEN
    #
    p = s.precedence

    #
    # THEN
    #
    assert 2 == p
    assert "AND" == s.name
    assert "AND" == s.value


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("", False),
        ('"million"', True),
        ('"million\\""', True),
        ('"mill\\"ion"', True),
        ('"million', False),
        ('"million\\"', False),
    ],
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_lexer_is_closed_string(expression, expected):
    assert parsing.ConditionLexer._is_closed_string(expression) == expected


# aliases
OR = parsing.ConditionSymbol.OR.name
AND = parsing.ConditionSymbol.AND.name
NOT = parsing.ConditionSymbol.NOT.name
FN_REGEXP = parsing.ConditionSymbol.FN_REGEXP.name
OPERAND = parsing.ConditionSymbol.OPERAND.name


@pytest.mark.parametrize(
    "expression,expected_lexemes,expected_ast,expected",
    [
        ("", [], [], True),
        ('"million"', ['"million"'], ['"million"', None], True),
        (
            '"phrase with spaces"',
            ['"phrase with spaces"'],
            ['"phrase with spaces"', None],
            False,
        ),
        (
            '"escaped \\" quote"',
            ['"escaped \\" quote"'],
            ['"escaped \\" quote"', None],
            False,
        ),
        # AND
        (
            '"a" AND "a"',
            ['"a"', AND, '"a"'],
            [AND, [['"a"', None], ['"a"', None]]],
            True,
        ),
        (
            '"a" AND "a" AND "a"',
            ['"a"', AND, '"a"', AND, '"a"'],
            ["AND", [['"a"', None], ["AND", [['"a"', None], ['"a"', None]]]]],
            True,
        ),
        (
            'NOT "a" AND NOT "a" AND NOT "a"',
            [NOT, '"a"', AND, NOT, '"a"', AND, NOT, '"a"'],
            [
                "AND",
                [
                    ["NOT", ['"a"', None]],
                    ["AND", [["NOT", ['"a"', None]], ["NOT", ['"a"', None]]]],
                ],
            ],
            False,
        ),
        (
            'NOT "a1" AND NOT "a2" AND NOT "a3" AND NOT "a4"',
            [NOT, '"a1"', AND, NOT, '"a2"', AND, NOT, '"a3"', AND, NOT, '"a4"'],
            [
                "AND",
                [
                    ["NOT", ['"a1"', None]],
                    [
                        "AND",
                        [
                            ["AND", [["NOT", ['"a2"', None]], ["NOT", ['"a3"', None]]]],
                            ["NOT", ['"a4"', None]],
                        ],
                    ],
                ],
            ],
            True,
        ),
        (
            '"15,969" AND "million"',
            ['"15,969"', AND, '"million"'],
            [AND, [['"15,969"', None], ['"million"', None]]],
            True,
        ),
        # OR
        (
            '"a" OR "a"',
            ['"a"', OR, '"a"'],
            [OR, [['"a"', None], ['"a"', None]]],
            True,
        ),
        (
            'NOT "a1" OR NOT "a2" OR NOT "a3" OR NOT "a4"',
            [NOT, '"a1"', OR, NOT, '"a2"', OR, NOT, '"a3"', OR, NOT, '"a4"'],
            [
                "OR",
                [  # BEGIN operands 1
                    [
                        "OR",
                        [  # BEGIN operands 2
                            [
                                "OR",
                                [  # BEGIN operands 3
                                    ["NOT", ['"a1"', None]],
                                    ["NOT", ['"a2"', None]],
                                ],  # END operands 3
                            ],
                            ["NOT", ['"a3"', None]],
                        ],  # END operands 2
                    ],
                    ["NOT", ['"a4"', None]],
                ],  # END operands 1
            ],
            True,
        ),
        # AND + OR
        (
            '"phrase with spaces" OR "escaped \\" quote"',
            ['"phrase with spaces"', OR, '"escaped \\" quote"'],
            [OR, [['"phrase with spaces"', None], ['"escaped \\" quote"', None]]],
            False,
        ),
        (
            '"a" OR "b" AND "c" OR "d"',
            ['"a"', OR, '"b"', AND, '"c"', OR, '"d"'],
            [
                OR,
                [
                    [
                        OR,
                        [['"a"', None], [AND, [['"b"', None], ['"c"', None]]]],
                    ],
                    ['"d"', None],
                ],
            ],
            True,
        ),
        (
            '"a" AND "B" OR "c" AND "d"',
            ['"a"', AND, '"B"', OR, '"c"', AND, '"d"'],
            [
                OR,
                [
                    [AND, [['"a"', None], ['"B"', None]]],
                    [AND, [['"c"', None], ['"d"', None]]],
                ],
            ],
            True,
        ),
        # REGEXP
        (
            'regexp("^15,969 [Mm]illion$") OR regexp("[a b\\\\"]")',
            [
                FN_REGEXP,
                '"^15,969 [Mm]illion$"',
                ")",
                OR,
                FN_REGEXP,
                '"[a b\\\\"]"',
                ")",
            ],
            [
                OR,
                [
                    [
                        FN_REGEXP,
                        ['"^15,969 [Mm]illion$"', None],
                    ],
                    [FN_REGEXP, ['"[a b\\\\"]"', None]],
                ],
            ],
            True,
        ),
        (
            'regexp("^15,969 [Mm]illion$") AND regexp("[a b\\\\"]")',
            [
                FN_REGEXP,
                '"^15,969 [Mm]illion$"',
                ")",
                AND,
                FN_REGEXP,
                '"[a b\\\\"]"',
                ")",
            ],
            [
                AND,
                [
                    [
                        FN_REGEXP,
                        ['"^15,969 [Mm]illion$"', None],
                    ],
                    [FN_REGEXP, ['"[a b\\\\"]"', None]],
                ],
            ],
            False,
        ),
        (
            'regexp("^Brazil revenue was 15,969 [Mm]illion\\.$")',
            [
                FN_REGEXP,
                '"^Brazil revenue was 15,969 [Mm]illion\\.$"',
                ")",
            ],
            [FN_REGEXP, ['"^Brazil revenue was 15,969 [Mm]illion\\.$"', None]],
            True,
        ),
        # NOT
        (
            'NOT "phrase not there"',
            [NOT, '"phrase not there"'],
            [NOT, ['"phrase not there"', None]],
            True,
        ),
        ('NOT "million"', [NOT, '"million"'], ["NOT", ['"million"', None]], False),
        (
            '"15,969" AND NOT "million"',
            ['"15,969"', AND, NOT, '"million"'],
            ["AND", [['"15,969"', None], ["NOT", ['"million"', None]]]],
            False,
        ),
        (
            'NOT "15,969" AND "million"',
            [NOT, '"15,969"', AND, '"million"'],
            ["AND", [["NOT", ['"15,969"', None]], ['"million"', None]]],
            False,
        ),
        (
            'NOT "15,969" OR "million"',
            ["NOT", '"15,969"', "OR", '"million"'],
            ["OR", [["NOT", ['"15,969"', None]], ['"million"', None]]],
            True,
        ),
        (
            'NOT "phrase with spaces" OR NOT "escaped \\" quote"',
            [NOT, '"phrase with spaces"', OR, NOT, '"escaped \\" quote"'],
            [
                OR,
                [
                    [NOT, ['"phrase with spaces"', None]],
                    [NOT, ['"escaped \\" quote"', None]],
                ],
            ],
            True,
        ),
        (
            'NOT "phrase with spaces" AND NOT "escaped \\" quote"',
            [NOT, '"phrase with spaces"', AND, NOT, '"escaped \\" quote"'],
            [
                AND,
                [
                    [NOT, ['"phrase with spaces"', None]],
                    [NOT, ['"escaped \\" quote"', None]],
                ],
            ],
            True,
        ),
        (
            'NOT "N/A" OR NOT regexp("^NA$") AND NOT "N/AA" OR NOT regexp("^NAA$")',
            [
                NOT,
                '"N/A"',
                OR,
                NOT,
                FN_REGEXP,
                '"^NA$"',
                ")",
                AND,
                NOT,
                '"N/AA"',
                OR,
                NOT,
                FN_REGEXP,
                '"^NAA$"',
                ")",
            ],
            [
                "OR",
                [
                    [
                        "OR",
                        [
                            ["NOT", ['"N/A"', None]],
                            [
                                "AND",
                                [
                                    ["NOT", ["FN_REGEXP", ['"^NA$"', None]]],
                                    ["NOT", ['"N/AA"', None]],
                                ],
                            ],
                        ],
                    ],
                    ["NOT", ["FN_REGEXP", ['"^NAA$"', None]]],
                ],
            ],
            True,
        ),
        # PARENTHESIS
        (
            '("B" OR "b") AND "a"',
            ["(", '"B"', "OR", '"b"', ")", "AND", '"a"'],
            [
                "AND",
                [["PARENS", [["OR", [['"B"', None], ['"b"', None]]]]], ['"a"', None]],
            ],
            True,
        ),
        (
            '(regexp("m") OR "m1") AND regexp("m2")',
            [
                "(",
                "FN_REGEXP",
                '"m"',
                ")",
                "OR",
                '"m1"',
                ")",
                "AND",
                "FN_REGEXP",
                '"m2"',
                ")",
            ],
            [
                "AND",
                [
                    [
                        "PARENS",
                        [["OR", [["FN_REGEXP", ['"m"', None]], ['"m1"', None]]]],
                    ],
                    ["FN_REGEXP", ['"m2"', None]],
                ],
            ],
            False,
        ),
        (
            'NOT (regexp("never") OR "never") AND regexp("a")',
            [
                "NOT",
                "(",
                "FN_REGEXP",
                '"never"',
                ")",
                "OR",
                '"never"',
                ")",
                "AND",
                "FN_REGEXP",
                '"a"',
                ")",
            ],
            [
                "AND",
                [
                    [
                        "NOT",
                        [
                            "PARENS",
                            [
                                [
                                    "OR",
                                    [
                                        ["FN_REGEXP", ['"never"', None]],
                                        ['"never"', None],
                                    ],
                                ]
                            ],
                        ],
                    ],
                    ["FN_REGEXP", ['"a"', None]],
                ],
            ],
            True,
        ),
        (
            '("Brazil")',
            ["(", '"Brazil"', ")"],
            ["PARENS", [['"Brazil"', None]]],
            True,
        ),
        (
            '(regexp("[Bb]razil"))',
            ["(", "FN_REGEXP", '"[Bb]razil"', ")", ")"],
            ["PARENS", [["FN_REGEXP", ['"[Bb]razil"', None]]]],
            True,
        ),
        (
            '(("Brazil"))',
            ["(", "(", '"Brazil"', ")", ")"],
            ["PARENS", [["PARENS", [['"Brazil"', None]]]]],
            True,
        ),
        (
            '("never") OR ("Brazil")',
            [
                "(",
                '"never"',
                ")",
                OR,
                "(",
                '"Brazil"',
                ")",
            ],
            [
                OR,
                [
                    ["PARENS", [['"never"', None]]],
                    ["PARENS", [['"Brazil"', None]]],
                ],
            ],
            True,
        ),
        (
            '("never" AND "NEVER") OR (regexp("[Bb]razil"))',
            [
                "(",
                '"never"',
                AND,
                '"NEVER"',
                ")",
                OR,
                "(",
                FN_REGEXP,
                '"[Bb]razil"',
                ")",
                ")",
            ],
            [
                OR,
                [
                    ["PARENS", [[AND, [['"never"', None], ['"NEVER"', None]]]]],
                    ["PARENS", [[FN_REGEXP, ['"[Bb]razil"', None]]]],
                ],
            ],
            True,
        ),
        (
            '("never" OR "NEVER") OR (regexp("[Bb]razil"))',
            [
                "(",
                '"never"',
                OR,
                '"NEVER"',
                ")",
                OR,
                "(",
                FN_REGEXP,
                '"[Bb]razil"',
                ")",
                ")",
            ],
            [
                OR,
                [
                    ["PARENS", [[OR, [['"never"', None], ['"NEVER"', None]]]]],
                    ["PARENS", [[FN_REGEXP, ['"[Bb]razil"', None]]]],
                ],
            ],
            True,
        ),
        (
            '(("never" OR "NEVER") OR (regexp("[Bb]razil")))',
            [
                "(",
                "(",
                '"never"',
                "OR",
                '"NEVER"',
                ")",
                "OR",
                "(",
                "FN_REGEXP",
                '"[Bb]razil"',
                ")",
                ")",
                ")",
            ],
            [
                "PARENS",
                [
                    [
                        "OR",
                        [
                            [
                                "PARENS",
                                [["OR", [['"never"', None], ['"NEVER"', None]]]],
                            ],
                            ["PARENS", [["FN_REGEXP", ['"[Bb]razil"', None]]]],
                        ],
                    ]
                ],
            ],
            True,
        ),
        (
            '"15,969" AND "million" AND (NOT regexp("[a b\\\\"]") OR "1")',
            [
                '"15,969"',
                "AND",
                '"million"',
                "AND",
                "(",
                "NOT",
                "FN_REGEXP",
                '"[a b\\\\"]"',
                ")",
                "OR",
                '"1"',
                ")",
            ],
            [
                "AND",
                [
                    ['"15,969"', None],
                    [
                        "AND",
                        [
                            ['"million"', None],
                            [
                                "PARENS",
                                [
                                    [
                                        "OR",
                                        [
                                            [
                                                "NOT",
                                                ["FN_REGEXP", ['"[a b\\\\"]"', None]],
                                            ],
                                            ['"1"', None],
                                        ],
                                    ]
                                ],
                            ],
                        ],
                    ],
                ],
            ],
            True,
        ),
        # NEGATIVE LEXICAL ERRORS: catch & check the exception message
        ('"m " " AND "million"', "unpaired quotation", None, False),
        ('"m" AND million"', "unknown type of lexeme", None, False),
        ('"million', "unclosed string", None, False),
        ('"m" ANDAND "million"', "unknown type of lexeme", None, False),
        (
            '"15,969" AND "million" AND ( NOT regexp("[a b\\\\"]") OR 1)',
            "unknown type of lexeme",
            None,
            False,
        ),
        # NEGATIVE SYNTAX ERRORS: catch & check the exception message
        ('("15,969"))', ["(", '"15,969"', ")", ")"], "missing left parent", False),
        ('("15,969"', ["(", '"15,969"'], "missing right parent", False),
        ('("15,969"))', ["(", '"15,969"', ")", ")"], "missing left parent", False),
        ("OR", [OR], "misplaced", False),
        ("AND", ["AND"], "misplaced", False),
        ("NOT AND", ["NOT", "AND"], "premature", False),
        ('"a" "b" "c"', ['"a"', '"b"', '"c"'], "consecutive", False),
    ],
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_condition(expression, expected_lexemes, expected_ast, expected):
    #
    # GIVEN
    #
    actual_output = "Brazil revenue was 15,969 million."
    logger = loggers.SonarPrintLogger()
    lexer = parsing.ConditionLexer(logger=logger)
    parser = parsing.ConditionRDParser(logger=logger)

    #
    # WHEN lex
    #
    try:
        lexemes = lexer.tokenize(expression)
    except ValueError as e:
        if isinstance(expected_lexemes, str):
            # OK - if expected lexemes are not LIST, then lexical error is expected
            if expected_lexemes not in str(e):
                print(
                    f"ERROR: exception thrown as expected, BUT with wrong message:"
                    f" {expected_lexemes}"
                )
                raise e
            print(e)
            return
        raise e

    #
    # THEN lex
    #
    print(f"\nInput:\n  >>>{expression}<<<")
    print(f"Tokens:\n  {lexemes}")
    assert expected_lexemes == lexemes

    #
    # WHEN parse
    #
    try:
        ast = parser.parse(lexemes)
    except ValueError as e:
        if isinstance(expected_ast, str):
            # OK - if expected AST is not LIST, then syntax error is expected
            if expected_ast not in str(e):
                print(
                    f"ERROR: exception thrown as expected, BUT with wrong message:"
                    f" {expected_ast}"
                )
                raise e
            print(e)
            return
        raise e

    #
    # THEN parse
    #
    print(f"AST:\n  {ast}")
    assert expected_ast == ast

    #
    # WHEN evaluate
    #

    # lex, parse and evaluate
    (e2e_result, e2e_failed_sub_cs) = evaluator.ConditionEvaluator(
        c=expression, logger=logger
    ).evaluate(s=actual_output)
    # evaluate AST from above
    (ast_result, ast_failed_sub_cs) = evaluator.ConditionEvaluator(
        c="", logger=logger
    ).evaluate(s=actual_output, c_ast=ast)

    #
    # THEN evaluate
    #
    print(f"E2E evaluation result:\n  {e2e_result}\n  {e2e_failed_sub_cs}")
    assert expected == e2e_result
    print(f"AST evaluation result:\n  {ast_result}\n  {ast_failed_sub_cs}")
    assert expected == ast_result

    #
    # THEN AST serialization
    #
    condition_ast = parsing.ConditionAst(lexer=lexer, parser=parser, logger=logger)
    ast_as_string = condition_ast.to_string(ast)
    print(f"Expected AST as   str:\n  {expression}")
    print(f"Serialized AST as str:\n  {ast_as_string}")
    assert expression == ast_as_string


@pytest.mark.parametrize(
    "actual_answer,condition,expected_failed_cs",
    [
        (
            "Brazil revenue was 15,969 million.",
            '"phrase with spaces" OR "escaped \\" quote"',
            '"phrase with spaces" OR "escaped \\" quote"',
        ),
        (
            "Brazil revenue was 15,969 million.",
            '"15,968"',
            '"15,968"',
        ),
        (
            "Brazil revenue was 15,969 million.",
            '"15,968" AND "million"',
            '"15,968"',
        ),
        (
            "Brazil revenue was 15,969 million.",
            '"15,969" AND "MISMATCH"',
            '"MISMATCH"',
        ),
        (
            "Brazil revenue was 15,969 million.",
            '("WRONG" OR "BAD") AND "million"',
            '"WRONG" OR "BAD"',
        ),
        (
            "Brazil revenue was 15,969 million.",
            '"WRONG" OR "BAD" AND "WORSE"',
            '"WRONG" OR "BAD" AND "WORSE"',
        ),
    ],
)
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_evaluation_failure_cond(
    actual_answer: str, condition: str, expected_failed_cs
):
    #
    # GIVEN
    #
    logger = loggers.SonarPrintLogger()
    print(f"\nActual answer:\n  {actual_answer}")
    print(f"Condition:\n  {condition}")

    #
    # WHEN
    #

    (e2e_result, e2e_failed_sub_ast) = evaluator.ConditionEvaluator(
        c=condition, logger=logger
    ).evaluate(s=actual_answer)

    #
    # THEN evaluate
    #
    print(f"E2E evaluation result:\n  {e2e_result}\n  {e2e_failed_sub_ast}")
    assert not e2e_result

    #
    # THEN AST serialization
    #
    condition_ast = parsing.ConditionAst(logger=logger)
    failed_ast_as_string = condition_ast.to_string(e2e_failed_sub_ast)
    print(f"Expected AST as   str:\n  {expected_failed_cs}")
    print(f"Serialized AST as str:\n  {failed_ast_as_string}")
    assert expected_failed_cs == failed_ast_as_string


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
