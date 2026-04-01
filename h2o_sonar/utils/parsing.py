# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
"""This module provides lexer and parser for expressions which use a subset of
the AIP-160 (API Improvement Proposal for filtering) syntax:

- https://google.aip.dev/160

"""

import enum

from h2o_sonar import loggers


class ConditionSymbol(enum.Enum):
    """Enumeration of the AIP-160 subset condition expression symbols."""

    NOT = "NOT", 1  # operator
    AND = "AND", 2  # operator
    OR = "OR", 3  # operator
    OPERAND = "OPERAND", 4  # operand (string)
    FN_REGEXP = "FN_REGEXP", 4  # operand (function)
    LEFT_PAREN = "(", 6  # symbol
    RIGHT_PAREN = ")", 5  # symbol
    PARENS = "PARENS", 6  # () node in AST

    def __new__(cls, name, precedence):
        member = object.__new__(cls)
        member._value_ = name
        member.precedence = precedence
        return member


class ConditionLexer:
    """Lexical analysis of the AIP-160 subset condition expressions:

    - operators: AND, OR, NOT
    - operands: string, regexp (function)
    - symbols: (, )

    """

    # lexer string symbols
    S_AND: str = ConditionSymbol.AND.value
    S_OR: str = ConditionSymbol.OR.value
    S_NOT: str = ConditionSymbol.NOT.value
    S_LEFT_PAREN: str = ConditionSymbol.LEFT_PAREN.value
    S_RIGHT_PAREN: str = ConditionSymbol.RIGHT_PAREN.value
    S_REGEXP_PREFIX: str = f"regexp{S_LEFT_PAREN}"

    RESERVED_SYMBOLS = [
        ConditionSymbol.AND.value,
        ConditionSymbol.OR.value,
        ConditionSymbol.NOT.value,
        ConditionSymbol.LEFT_PAREN.value,
        ConditionSymbol.RIGHT_PAREN.value,
    ]

    @staticmethod
    def _is_closed_string(s: str) -> bool:
        """Check if the string is closed - enclosed in ``"``.

        Parameters
        ----------
        s : str
            The string to check.

        Returns
        -------
        bool :
            True if the string is closed, False otherwise.

        """
        # IMPROVE make it faster
        if s.startswith('"'):
            ss = s.replace('\\"', "")
            return ss.count('"') % 2 == 0
        return False

    def __init__(self, logger):
        """Constructor.

        Parameters
        ----------
        logger : loggers.SonarLogger
            The logger instance.

        """
        self.logger = logger
        self.err_prefix = "Condition lexical analysis error:"

    def tokenize(self, expression: str) -> list[str]:
        """Lexical analysis of the string expression into lexemes.

        Parameters
        ----------
        expression : str
            The expression to analyze.

        Returns
        -------
        list[str] :
            The list of lexemes.

        """
        lexemes: list[str] = []

        if not expression:
            return lexemes

        self.logger.debug("\nLexing:")
        lexemes_stack = expression.split(" ")
        while lexemes_stack:
            lexeme = lexemes_stack.pop(0)

            if lexeme in ConditionLexer.RESERVED_SYMBOLS:
                self.logger.debug(f"  keyword: '{lexeme}'")
                lexemes.append(lexeme)
                continue

            if lexeme.startswith('"'):
                if len(lexeme) > 1:
                    if lexeme.endswith('"') and not lexeme.endswith('\\"'):
                        self.logger.debug(f"  string: '{lexeme}'")
                        lexemes.append(lexeme)
                        continue
                    elif ConditionLexer._is_closed_string(lexeme):
                        if lexeme.endswith(ConditionLexer.S_RIGHT_PAREN):
                            lexemes_stack.insert(0, ConditionLexer.S_RIGHT_PAREN)
                            lexemes_stack.insert(0, lexeme[:-1])
                            continue
                        else:
                            raise ValueError(
                                f"{self.err_prefix} invalid lexeme - unclosed string "
                                f"'{lexeme}' in the expression '{expression}' "
                                f"(analysis failed after: {lexemes})"
                            )
                    else:
                        # join unclosed STRING lexemes which have SPACE in it
                        self.logger.debug(f"  string to join: '{lexeme}'")
                        if lexemes_stack:
                            next_lexeme = lexemes_stack.pop(0)
                            lexemes_stack.insert(0, f"{lexeme} {next_lexeme}")
                            continue
                        else:
                            raise ValueError(
                                f"{self.err_prefix} invalid lexeme - unclosed string "
                                f"'{lexeme}' in the expression '{expression}' "
                                f"(analysis failed after: {lexemes})"
                            )
                else:
                    raise ValueError(
                        f"{self.err_prefix} unexpected lexeme '{lexeme}' in the "
                        f"expression '{expression}' - possibly unpaired quotation "
                        f"marks (analysis failed after: {lexemes})"
                    )

            if lexeme.startswith(ConditionLexer.S_LEFT_PAREN):
                self.logger.debug(f"  left paren: '{lexeme}'")
                lexemes.append(str(ConditionSymbol.LEFT_PAREN.value))
                lexemes_stack.insert(0, lexeme[1:])
                continue

            if lexeme.endswith(ConditionLexer.S_RIGHT_PAREN):
                self.logger.debug(f"  right paren: '{lexeme}'")
                lexemes_stack.insert(0, str(ConditionSymbol.RIGHT_PAREN.value))
                lexemes_stack.insert(0, lexeme[:-1])
                continue

            if lexeme.startswith(ConditionLexer.S_REGEXP_PREFIX):
                self.logger.debug(f"  regexp: '{lexeme}'")
                lexemes.append(str(ConditionSymbol.FN_REGEXP.value))
                lexemes_stack.insert(0, lexeme[len(ConditionLexer.S_REGEXP_PREFIX) :])
                continue

            raise ValueError(
                f"{self.err_prefix} unknown type of lexeme '{lexeme}' in "
                f"the expression: '{expression}' (analysis failed after: {lexemes})"
            )

        return lexemes


# AST shorthands
# - AST root node (index)
ROOT = 0
# - AST node symbol (index)
SYMBOL = 0
# - operands (index)
OPERANDS = 1


class ConditionRDParser:
    """Recursive descent parser for the AIP-160 subset condition expressions:

    - operators: AND, OR, NOT
    - operands: string, regexp (function)
    - symbols: (, )

    Precedence:

    - NOT: 1
    - AND: 2
    - OR: 3
    - OPERAND: 4

    """

    def __init__(self, logger):
        """Constructor.

        Parameters
        ----------
        logger : loggers.SonarLogger
            The logger instance.

        """
        # descent first parser function pointers
        self.dfp = {
            ConditionSymbol.NOT.value: self._parse_not,
            ConditionSymbol.AND.value: self._parse_and,
            ConditionSymbol.OR.value: self._parse_or,
            ConditionSymbol.LEFT_PAREN.value: self._parse_l_paren,
        }

        self.logger = logger
        self.err_prefix = "Condition parsing error:"

    def _parse_descent(self, lexemes: list[str], parens: dict, ast: list):
        """Selects descent methods based on the lexeme in the head of lexemes."""
        self.logger.debug(f"    => {ast}")
        if not lexemes:
            raise ValueError(
                f"{self.err_prefix} premature end of input - remaining lexemes: "
                f"{lexemes}, AST: {ast}"
            )
        return self.dfp.get(lexemes[0], self._parse_op)(lexemes, parens, ast)

    def _parse_paren_check(self, lexemes: list[str]) -> dict:
        """Check if the parenthesis are balanced and create map of matching
        parenthesis pairs with indices counted from the end of the lexemes - all with
        linear time complexity.

        Parameters
        ----------
        lexemes : list[str]
            The list of lexemes to check.

        Returns
        -------
        dict :
            The map of matching parenthesis pairs with indices counted from the end
            of the lexemes.

        """
        # map: opening ( index -> closing ) index
        result = {}
        parent_stack = []
        len_lexemes = len(lexemes) - 1
        is_regexp_open = False
        for e, lexeme in enumerate(lexemes):
            if lexeme == ConditionSymbol.FN_REGEXP.value:
                is_regexp_open = True
                continue
            if lexeme == ConditionSymbol.LEFT_PAREN.value:
                parent_stack.append(len_lexemes - e)
            elif lexeme == ConditionSymbol.RIGHT_PAREN.value:
                if is_regexp_open:
                    is_regexp_open = False
                    continue
                if not parent_stack:
                    raise ValueError(
                        f"{self.err_prefix} unbalanced parenthesis at lexeme offset "
                        f"{e} - missing {'left' if not parent_stack else 'right'} "
                        f"parenthesis - remaining lexemes: '{lexemes}', parenthesis "
                        f"stack: {parent_stack}"
                    )
                result[parent_stack.pop()] = len_lexemes - e

        if parent_stack:
            raise ValueError(
                f"{self.err_prefix} unbalanced parenthesis - missing right parenthesis "
                f"- remaining lexemes: '{lexemes}', parenthesis stack: {parent_stack}"
            )

        self.logger.debug(f"  Parenthesis map:\n    {result}")
        return result

    def _parse_l_paren(self, lexemes: list[str], parens: dict, ast: list):
        """Parse left parenthesis."""

        self.logger.debug(f"  LEFT paren: {lexemes}")
        if not lexemes:
            raise ValueError(
                f"{self.err_prefix} missing operands when parsing (left) parenthesis "
                f"- remaining lexemes: {lexemes}, AST: {ast}"
            )

        lexeme = lexemes.pop(0)
        if not lexeme == ConditionSymbol.LEFT_PAREN.value:
            raise ValueError(
                f"{self.err_prefix} expected '(' but got: '{lexeme}' when parsing "
                f"left parenthesis - remaining lexemes: {lexemes}, AST: {ast}"
            )

        idx = len(lexemes)
        if idx in parens:
            self.logger.debug(f"    Parenthesis: {idx} -> {parens[idx]}")
            # extract segment of lexemes in between the parenthesis
            if parens[idx]:
                segment_lng = idx - parens[idx] - 1
                sub_lexemes = lexemes[0:segment_lng]
                remaining_lexemes = lexemes[segment_lng + 1 :]
            else:
                sub_lexemes = lexemes[0:-1]
                remaining_lexemes = []
            self.logger.debug(f"    Sub-lexemes: {sub_lexemes}")
            # parse sub-lexemes
            self.logger.debug(f"    = BEGIN: sub-AST parsing: {idx} -> {parens[idx]} =")
            sub_ast = self.parse(sub_lexemes)
            sub_ast = [ConditionSymbol.PARENS.value, [sub_ast]]
            self.logger.debug(f"    Parsed sub-AST: {sub_ast}")
            self.logger.debug(f"    Remaining lexemes: {remaining_lexemes}")
            self.logger.debug(f"    = END: sub-AST parsing: {idx} -> {parens[idx]} =")
            # inject sub-AST as normal operand
            return self._parse_op(
                lexemes=remaining_lexemes, parens=parens, ast=ast, sub_ast=sub_ast
            )
        else:
            raise ValueError(
                f"{self.err_prefix} unbalanced parenthesis at lexeme offset {idx} - "
                f"missing right parenthesis - remaining lexemes: {lexemes}, AST: {ast}"
            )

    def _parse_not(self, lexemes: list[str], parens: dict, ast: list) -> list:
        """Parse NOT operator."""
        self.logger.debug(f"  NOT: {lexemes}")
        if not lexemes:
            raise ValueError(
                f"{self.err_prefix} missing operand for operator 'NOT' - remaining "
                f"lexemes: {lexemes}, AST: {ast}"
            )

        lexeme = lexemes.pop(0)
        if not lexeme == ConditionSymbol.NOT.value:
            raise ValueError(
                f"{self.err_prefix} expected 'NOT' lexeme, but got: '{lexeme}' "
                f"- remaining lexemes: {lexemes}, AST: {ast}"
            )

        if ast:
            # DESCENT through OR and AND
            node = ast[SYMBOL]
            tree = ast
            while node in [ConditionSymbol.AND.value, ConditionSymbol.OR.value]:
                if tree[OPERANDS] is None:
                    # INCOMPLETE AND/OR binary operators operands > inject NOT
                    tree.append([lexeme, None])
                    break
                elif len(tree[OPERANDS]) == 1:
                    # INCOMPLETE AND/OR binary operators operands > complete with NOT
                    tree[OPERANDS].append([lexeme, None])
                    break
                elif len(tree[OPERANDS]) == 2:
                    # COMPLETE AND/OR binary operators operands > descent right
                    node = tree[OPERANDS][1]
                    if isinstance(node, list):
                        tree = node
                        node = node[SYMBOL]
                    continue
                else:
                    raise ValueError(
                        f"{self.err_prefix}: unexpected operands in AST root while "
                        f"parsing `NOT` - sub-AST: '{tree}', remaining lexemes:"
                        f" {lexemes}, AST: {ast}"
                    )
        else:
            ast = [lexeme, None]

        return self._parse_descent(lexemes=lexemes, parens=parens, ast=ast)

    def _parse_and(self, lexemes: list[str], parens: dict, ast: list) -> list:
        """Parse AND operator."""
        self.logger.debug(f"  AND:\n    {lexemes}\n    {ast}")
        if not lexemes:
            raise ValueError(
                f"{self.err_prefix} missing operand for operator 'AND' - remaining "
                f"lexemes: {lexemes}, AST: {ast}"
            )

        lexeme = lexemes.pop(0)
        if not lexeme == ConditionSymbol.AND.value:
            raise ValueError(
                f"{self.logger} expected 'AND' lexeme, but got: '{lexeme}' - remaining "
                f"lexemes: {lexemes}, AST: {ast}"
            )

        if ast:
            # DESCENT through OR and AND
            node = ast[SYMBOL]
            tree = ast
            while node in [ConditionSymbol.AND.value, ConditionSymbol.OR.value]:
                if tree[OPERANDS] is None:
                    tree.append([lexeme, None])
                    break
                elif (
                    len(tree[OPERANDS]) == 2
                ):  # binary operators operands > descent right
                    node = tree[OPERANDS][1]
                    continue
                else:
                    raise ValueError(
                        f"{self.err_prefix}: unexpected first operand in the AST root "
                        f"while parsing 'AND' - sub-AST: {tree}, remaining lexemes: "
                        f"{lexemes}, AST: {ast}"
                    )

            # INJECT lexeme
            if tree:
                if isinstance(tree, list):
                    if tree[OPERANDS] is None:
                        operand = tree.pop(0)
                        tree.pop()  # None
                        tree.append(lexeme)
                        tree.append([[operand, None]])
                    elif tree[SYMBOL] in [
                        ConditionSymbol.FN_REGEXP.value,
                        ConditionSymbol.NOT.value,
                        ConditionSymbol.PARENS.value,
                    ]:
                        child = tree.copy()
                        tree.clear()
                        tree.append(lexeme)
                        tree.append([child])
                    else:
                        child = tree[OPERANDS].pop()
                        tree[OPERANDS].append([lexeme, [child]])
            else:
                ast = [lexeme, ast]
        else:
            raise ValueError(
                f"{self.err_prefix} unexpected lexemes - misplaced 'AND' operator: "
                f" {lexemes}, AST: {ast}"
            )

        return self._parse_descent(lexemes=lexemes, parens=parens, ast=ast)

    def _parse_or(self, lexemes: list[str], parens: dict, ast: list) -> list:
        """Parse OR operator."""
        self.logger.debug(f"  OR:\n    {lexemes}\n    {ast}")
        if not lexemes:
            raise ValueError(
                f"{self.err_prefix} missing operand for operator 'OR' - remaining "
                f"lexemes: {lexemes}, AST: {ast}"
            )

        lexeme = lexemes.pop(0)
        if not lexeme == ConditionSymbol.OR.value:
            raise ValueError(
                f"{self.err_prefix} expected 'OR' but got: '{lexeme}' - remaining "
                f"lexemes: {lexemes}, AST: {ast}"
            )

        if ast:
            # DESCENT through OR
            node = ast[SYMBOL]
            tree = ast
            while node in [ConditionSymbol.OR.value]:
                if tree[OPERANDS] is None:
                    tree.append([lexeme, None])
                    break
                elif (
                    len(tree[OPERANDS]) == 2
                ):  # binary operators operands > descent right
                    node = tree[OPERANDS][1]
                    continue
                else:
                    raise ValueError(
                        f"{self.err_prefix}: unexpected exactly one operand (should "
                        f"be 0 or 2) while parsing 'OR' -  sub-AST: {tree}, remaining "
                        f"lexemes: {lexemes}, AST: {ast}"
                    )

            # INJECT lexeme
            if tree:
                if isinstance(tree, list):
                    if tree[OPERANDS] is None:
                        operand = tree.pop(0)
                        tree.pop()  # None
                        tree.append(lexeme)
                        tree.append([[operand, None]])
                    elif (
                        tree[SYMBOL] == ConditionSymbol.FN_REGEXP.value
                        or tree[SYMBOL] == ConditionSymbol.NOT.value
                    ):
                        child = tree.copy()
                        tree.clear()
                        tree.append(lexeme)
                        tree.append([child])
                    else:
                        ast = [lexeme, [tree]]

            else:
                ast = [lexeme, ast]
        else:
            raise ValueError(
                f"{self.err_prefix}: unexpected lexemes - misplaced 'OR' operator:"
                f" {lexemes}, AST: {ast}"
            )

        return self._parse_descent(lexemes=lexemes, parens=parens, ast=ast)

    def _parse_op(
        self,
        lexemes: list[str],
        parens: dict,
        ast: list,
        sub_ast: list | None = None,
    ) -> list:
        """Parse operand."""
        self.logger.debug(f"  operand: {lexemes}\n    {lexemes}\n    {ast}")

        lexeme = None
        is_fn = False
        if not sub_ast:
            if not lexemes:
                raise ValueError(
                    f"{self.err_prefix} missing operand for operator - remaining "
                    f"lexemes: {lexemes}, AST: {ast}"
                )

            lexeme = lexemes.pop(0)
            if lexeme in [
                ConditionSymbol.AND.value,
                ConditionSymbol.OR.value,
                ConditionSymbol.NOT.value,
            ]:
                raise ValueError(
                    f"{self.err_prefix} expected operand, but got: '{lexeme}' - "
                    f"remaining lexemes: {lexemes}, AST: {ast}"
                )

            is_fn = True if lexeme == ConditionSymbol.FN_REGEXP.name else False
            if is_fn:
                if len(lexemes) < 2:
                    raise ValueError(
                        f"{self.err_prefix} invalid function call '{lexeme}' - "
                        f"remaining {lexemes}, AST: {ast}"
                    )
                if not lexemes[0].startswith('"'):
                    raise ValueError(
                        f"{self.err_prefix} invalid function '{lexeme}' operand"
                        f"`{lexemes[0]}` - remaining {lexemes}, AST: {ast}"
                    )
                if lexemes[1] != ConditionSymbol.RIGHT_PAREN.value:
                    raise ValueError(
                        f"{self.err_prefix} expected ')' after function '{lexeme}' - "
                        f"remaining {lexemes}, AST: {ast}"
                    )

        # operator: traverse down to the leaf
        if ast:
            node = ast[0]
            tree = ast
            while node in [
                ConditionSymbol.AND.value,
                ConditionSymbol.OR.value,
                ConditionSymbol.NOT.value,
            ]:
                if tree[OPERANDS] is None:
                    # NOT
                    break
                elif (
                    len(tree[OPERANDS]) == 2
                ):  # binary operators operands > descent right
                    tree = tree[OPERANDS][1]
                    continue

                break

            if tree[OPERANDS] is None:
                if is_fn:
                    tree.pop()
                    fn_operand = lexemes.pop(0)
                    lexemes.pop(0)  # pop ')'
                    tree.append([lexeme, [fn_operand, None]])
                elif tree[SYMBOL] == ConditionSymbol.NOT.value:
                    tree.pop()
                    if sub_ast:
                        tree.append(sub_ast)
                    else:
                        tree.append([lexeme, None])
                else:
                    raise ValueError(
                        f"{self.err_prefix} unexpected sub-AST {tree} with lexeme:"
                        f" {lexeme} - might be consecutive operands without operator - "
                        f"remaining lexemes: {lexemes}, AST: {ast}"
                    )
            elif len(tree[OPERANDS]) == 1:
                if is_fn:
                    fn_operand = lexemes.pop(0)
                    lexemes.pop(0)  # pop ')'
                    tree[OPERANDS].append([lexeme, [fn_operand, None]])
                else:
                    if sub_ast:
                        tree[OPERANDS].append(sub_ast)
                    else:
                        tree[OPERANDS].append([lexeme, None])
        else:
            if is_fn:
                fn_operand = lexemes.pop(0)
                lexemes.pop(0)  # pop ')'
                ast = [lexeme, [fn_operand, None]]
            else:
                if sub_ast:
                    ast = sub_ast
                else:
                    ast = [lexeme, None]

        return (
            self._parse_descent(lexemes=lexemes, parens=parens, ast=ast)
            if lexemes
            else ast
        )

    def parse(self, lexemes: list[str]) -> list:
        """Parse lexemes into an AST.

        Parameters
        ----------
        lexemes : list[str]
            The list of lexemes to parse.

        Returns
        -------
        List :
            The AST of the parsed lexemes. AST is a nested lists (consider clashes
            in the keys in case of the dictionaries) of operators and operands where
            the head of the list is the operator and the tail is/are the operand(s).
            Leaf node is represented as the empty list.

        """
        self.logger.debug("Parsing:")

        ast = []
        parens = self._parse_paren_check(lexemes)
        return (
            self._parse_descent(lexemes=lexemes, parens=parens, ast=ast)
            if lexemes
            else ast
        )


class ConditionAst:
    def __init__(
        self,
        lexer: ConditionLexer | None = None,
        parser: ConditionRDParser | None = None,
        logger=None,
    ):
        self.logger = logger or loggers.SonarPrintLogger()
        self.lexer = lexer or ConditionLexer(logger=logger)
        self.parser = parser or ConditionRDParser(logger=logger)

    def to_string(self, ast: list) -> str:
        """Convert AST to string.

        Parameters
        ----------
        ast : list
            The AST to convert.

        Returns
        -------
        str :
            The string representation of the AST.

        """
        if not ast:
            return ""

        if isinstance(ast, str):
            return ast

        if isinstance(ast, list):
            if ast[0] == ConditionSymbol.PARENS.value:
                return f"({self.to_string(ast[1][0])})"

            if ast[0] == ConditionSymbol.FN_REGEXP.value:
                return f"regexp({ast[1][0]})"

            if ast[0] == ConditionSymbol.NOT.value:
                return f"{ConditionSymbol.NOT.value} {self.to_string(ast[1])}"

            if ast[0] in [ConditionSymbol.AND.value, ConditionSymbol.OR.value]:
                return (
                    f"{self.to_string(ast[1][0])} {ast[0]} {self.to_string(ast[1][1])}"
                )

            return ast[0]

        raise ValueError(f"Invalid AST node: {ast}")
