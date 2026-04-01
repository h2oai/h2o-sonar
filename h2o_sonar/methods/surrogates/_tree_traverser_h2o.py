# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
from math import isnan

from h2o_sonar.lib.api import commons
from h2o_sonar.methods.surrogates._abstract_tree_traverser import AbstractTreeTraverser
from h2o_sonar.methods.surrogates.rules.rules import Criterion
from h2o_sonar.methods.surrogates.rules.rules import NodeDirection
from h2o_sonar.methods.surrogates.rules.rules import RuleOperators


try:
    from h2o.tree import H2OLeafNode

    HAS_H2O = True
except ImportError:
    HAS_H2O = False


class H2OTreeTraverser(AbstractTreeTraverser):
    def is_leaf_node(self, node):
        if not HAS_H2O:
            commons.raise_opt_import_err("h2o")
        return isinstance(node, H2OLeafNode)

    def categorical_conditions(self, node, direction: NodeDirection):
        levels = (
            node.left_levels if direction == NodeDirection.LEFT else node.right_levels
        )
        if levels:
            return [Criterion(RuleOperators.IS, level) for level in levels]
        return []

    def numerical_conditions(self, node, direction: NodeDirection):
        if node.threshold:
            threshold_condition_op = (
                RuleOperators.LR
                if direction == NodeDirection.LEFT and not isnan(node.threshold)
                else RuleOperators.GE
                if not isnan(node.threshold)
                else RuleOperators.IS
            )
            return [Criterion(threshold_condition_op, node.threshold)]
        return []

    def na_conditions(self, node, direction: NodeDirection):
        if node.na_direction and node.na_direction.upper() in str(direction):
            return [Criterion(RuleOperators.IS, "N/A")]
        return []
