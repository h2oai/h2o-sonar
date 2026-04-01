# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.


from h2o_sonar.methods.surrogates.rules.rules import CodeStyle
from h2o_sonar.methods.surrogates.rules.rules import Criterion
from h2o_sonar.methods.surrogates.rules.rules import NodeDirection
from h2o_sonar.methods.surrogates.rules.rules import Rule
from h2o_sonar.methods.surrogates.rules.rules import RuleCondition
from h2o_sonar.methods.surrogates.rules.rules import Rules


class AbstractTreeTraverser:
    def __init__(self, root_node, features):
        self.root_node = root_node
        self.features = features

    def is_leaf_node(self, node):
        raise NotImplementedError

    def left_child(self, node):
        return node.left_child

    def right_child(self, node):
        return node.right_child

    def split_feature(self, node):
        return node.split_feature

    def node_prediction(self, node):
        return node.prediction

    def categorical_conditions(self, node, direction: NodeDirection):
        raise NotImplementedError

    def numerical_conditions(self, node, direction: NodeDirection):
        raise NotImplementedError

    def na_conditions(self, node, direction: NodeDirection):
        raise NotImplementedError

    def extract_rules_from_tree_as_py_code(
        self, code_style: CodeStyle = CodeStyle.DICT_ROW
    ):
        if code_style == CodeStyle.RAW_VARS:
            rules = self.extract_rules_from_tree(
                gen_py_code=True, code_style=code_style
            )
            rules = "\n" + rules.__str__()
            indent = "  "
            rules = indent + rules.replace("\n", "\n" + indent)
            imports = (
                "import numpy as np # For missing value handling (N/A)\nfrom math "
                "import nan # For handling nan thresholds\n\n"
            )
            features = list(filter(None, set(self.features)))
            features.sort()
            fxn_signature = f"def dt_surrogate({', '.join(features)}):"
            return f"{imports}{fxn_signature} \n {rules}"

        # default Python code style: row as dictionary
        rules = self.extract_rules_from_tree(gen_py_code=True, code_style=code_style)
        rules = "\n" + rules.__str__()
        indent = "  "
        rules = indent + rules.replace("\n", "\n" + indent)
        imports = (
            "from typing import Dict\nimport numpy as np\nfrom math import nan\n\n"
        )
        fxn_signature = "def dt_surrogate(row: Dict):"
        return f"{imports}{fxn_signature}\n{rules}"

    def extract_rules_from_tree_as_txt(
        self,
        code_style: CodeStyle = CodeStyle.DICT_ROW,
    ):
        return self.extract_rules_from_tree(gen_py_code=False, code_style=code_style)

    def extract_rules_from_tree(
        self,
        gen_py_code: bool,
        code_style: CodeStyle = CodeStyle.DICT_ROW,
    ) -> Rules:
        def traverse_tree(node, rule_conditions: list[RuleCondition]) -> list[Rule]:
            if self.is_leaf_node(node):
                # Left Node
                return [
                    Rule(
                        rule_value=self.node_prediction(node),
                        rule_conditions=rule_conditions.copy(),
                        gen_py_code=gen_py_code,
                        code_style=code_style,
                    )
                ]

            # Left Node
            rule_condition = RuleCondition(
                feature_name=self.split_feature(node),
                criteria=self.criteria_for_node(node, NodeDirection.LEFT),
            )
            rule_conditions.append(rule_condition)
            rules = traverse_tree(self.left_child(node), rule_conditions)
            rule_conditions.pop()

            # Right Node
            rule_condition = RuleCondition(
                feature_name=self.split_feature(node),
                criteria=self.criteria_for_node(node, NodeDirection.RIGHT),
            )
            rule_conditions.append(rule_condition)
            rules += traverse_tree(self.right_child(node), rule_conditions)
            rule_conditions.pop()

            return rules

        return Rules(traverse_tree(self.root_node, []), gen_py_code)

    def criteria_for_node(self, node, direction: NodeDirection) -> list[Criterion]:
        conditions = []

        conditions += self.categorical_conditions(node, direction)
        conditions += self.numerical_conditions(node, direction)
        conditions += self.na_conditions(node, direction)

        return conditions
