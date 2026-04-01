# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import enum


class NodeDirection(enum.Enum):
    RIGHT = enum.auto()
    LEFT = enum.auto()


class RuleOperators(enum.Enum):
    EQ = "=="
    GT = ">"
    GE = ">="
    LR = "<"
    LE = "<="
    IS = "IS"
    NONE = ""
    IF = "IF"
    AND = "AND"
    OR = "OR"
    TERMINAL = "THEN AVERAGE VALUE OF TARGET IS"


class CodeStyle(enum.Enum):
    # Python code gen w/ dataset row as dictionary
    DICT_ROW = enum.auto()
    # Python code gen w/ unescaped feature names (invalid for transformed features)
    RAW_VARS = enum.auto()


class Criterion:
    def __init__(self, operator: RuleOperators, value):
        self.operator: RuleOperators = operator
        self.value = value


class RuleCondition:
    def __init__(self, feature_name, criteria: list[Criterion]):
        self.feature_name = feature_name
        self.criteria: list[Criterion] = criteria


class Rule:
    def __init__(
        self,
        rule_value,
        rule_conditions: list[RuleCondition],
        gen_py_code: bool = False,
        code_style: CodeStyle = CodeStyle.DICT_ROW,
    ):
        self.rule_value = rule_value
        self.rule_conditions: list[RuleCondition] = rule_conditions
        self.gen_py_code = gen_py_code
        self.code_style = code_style

    def __str_raw_var(self):
        if not self.gen_py_code:
            rule_str = f"{RuleOperators.IF.value} "
        else:
            rule_str = f"{RuleOperators.IF.value.lower()}"
            rule_str = rule_str + "("

        cond_idx = 0
        for condition in self.rule_conditions:
            rule_str += cond_idx * " "
            rule_str += f"{condition.feature_name} "
            for criteria_idx, criteria in enumerate(condition.criteria):
                if criteria_idx > 0:
                    if not self.gen_py_code:
                        rule_str += (
                            f" {RuleOperators.OR.value} {condition.feature_name} "
                        )
                    else:
                        rule_str += (
                            f" {RuleOperators.OR.value.lower()} "
                            f"{condition.feature_name} "
                        )
                if (
                    self.gen_py_code
                    and criteria.operator.value is RuleOperators.IS.value
                ):
                    criteria_val = (
                        f'"{criteria.value}"'
                        if isinstance(criteria.value, str) and criteria.value != "N/A"
                        else "np.nan"
                        if criteria.value == "N/A"
                        else criteria.value
                    )
                    rule_str += f"{criteria.operator.value.lower()} {criteria_val}"
                else:
                    criteria_val = (
                        f'"{criteria.value}"'
                        if isinstance(criteria.value, str) and criteria.value == ""
                        else criteria.value
                    )
                    rule_str += f"{criteria.operator.value} {criteria_val}"
            if cond_idx < len(self.rule_conditions) - 1:
                if not self.gen_py_code:
                    rule_str += f" {RuleOperators.AND.value} "
                else:
                    rule_str += f" {RuleOperators.AND.value.lower()}"
            if cond_idx >= len(self.rule_conditions) - 1 and self.gen_py_code:
                rule_str += "):"
            rule_str += "\n"
            cond_idx += 1
        rule_str += cond_idx * " "
        if not self.gen_py_code:
            rule_str += f"{RuleOperators.TERMINAL.value} {self.rule_value}\n"
        else:
            rule_str += f"  return {self.rule_value}\n"
        return rule_str + "\n"

    def __str_dict_row(self):
        if not self.gen_py_code:
            rule_str = f"{RuleOperators.IF.value} \n"
        else:
            rule_str = f"{RuleOperators.IF.value.lower()} (\n"

        cond_idx = 0
        for condition in self.rule_conditions:
            esc_feature_name = (
                condition.feature_name.replace('"', '\\"')
                if condition.feature_name
                else condition.feature_name
            )
            # LEGACY INDENT: rule_str += cond_idx * " "
            if not self.gen_py_code:
                rule_str += 5 * " "
            else:
                rule_str += 4 * " "
            feature_str = (
                f'row["{esc_feature_name}"]'
                if self.gen_py_code
                else f"{condition.feature_name}"
            )
            rule_str += f"({feature_str} "
            for criteria_idx, criteria in enumerate(condition.criteria):
                if criteria_idx > 0:
                    if not self.gen_py_code:
                        rule_str += f" {RuleOperators.OR.value} {feature_str} "
                    else:
                        rule_str += f" {RuleOperators.OR.value.lower()} {feature_str} "
                if (
                    self.gen_py_code
                    and criteria.operator.value is RuleOperators.IS.value
                ):
                    criteria_val = (
                        f'"{criteria.value}"'
                        if isinstance(criteria.value, str) and criteria.value != "N/A"
                        else "np.nan"
                        if criteria.value == "N/A"
                        else criteria.value
                    )
                    rule_str += f"{criteria.operator.value.lower()} {criteria_val}"
                else:
                    criteria_val = (
                        f'"{criteria.value}"'
                        if isinstance(criteria.value, str) and criteria.value == ""
                        else criteria.value
                    )
                    rule_str += f"{criteria.operator.value} {criteria_val}"
            if cond_idx < len(self.rule_conditions) - 1:
                if not self.gen_py_code:
                    rule_str += f") {RuleOperators.AND.value} "
                else:
                    rule_str += f") {RuleOperators.AND.value.lower()}"
            if cond_idx >= len(self.rule_conditions) - 1:
                if self.gen_py_code:
                    rule_str += ")\n   ):"
                else:
                    rule_str += ")"
            rule_str += "\n"
            cond_idx += 1
        # LEGACY indent: rule_str += cond_idx * " "
        if not self.gen_py_code:
            rule_str += 3 * " "  # THEN indent
        else:
            rule_str += 2 * " "  # THEN indent
        if not self.gen_py_code:
            rule_str += f"THEN\n     AVERAGE VALUE OF TARGET IS {self.rule_value}\n"
        else:
            rule_str += f"  return {self.rule_value}\n"
        return rule_str + "\n"

    def __str__(self):
        return (
            self.__str_dict_row()
            if self.code_style == CodeStyle.DICT_ROW
            else self.__str_raw_var()
        )


class Rules:
    def __init__(self, rules: list[Rule], gen_py_code: bool = False):
        self.rules = rules
        self.gen_py_code = gen_py_code

    def __str__(self):
        """
        1. IF <FEATURE_0> < 1.5 OR NULL AND
          <FEATURE_2> < 0.5 AND
            <FEATURE_4> > 5533
              THEN AVERAGE VALUE OF TARGET IS <PREDICTION_0>
        2. IF PAY_0 >= 1.5 AND
          PAY_2 >= 0.5 OR NULL AND
            BILL_AMT <= 5533 OR NULL
              THEN AVERAGE VALUE OF DEFAULT PAYMENT NEXT MONTH IS 0.321
        ...
        """
        rule_str = ""
        for idx, rule in enumerate(self.rules):
            rule_str += (
                f"{idx + 1}. " + str(rule) if not self.gen_py_code else str(rule)
            )
        return rule_str.strip()
