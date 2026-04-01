# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import pytest

from h2o_sonar.methods.surrogates.rules import rules
from tests.base_h2o_test import BaseH2OTest


class TestRulesSurrogate(BaseH2OTest):
    def test_rules_non_empty(self):
        # GIVEN
        rule_value = "dummy_value"
        criterions = [rules.Criterion(rules.RuleOperators.EQ, "dummy_criteria")]
        rule_conditions = [rules.RuleCondition("dummy_feature", criterions)]
        rule = rules.Rule(
            rule_value, rule_conditions, code_style=rules.CodeStyle.RAW_VARS
        )
        # THEN
        expected_str = (
            f"IF {rule_conditions[0].feature_name} == {criterions[0].value}\n"
            f" THEN AVERAGE VALUE OF TARGET IS {rule_value}\n\n"
        )
        self.assertEqual(str(rule), expected_str)

    def test_rules_empty(self):
        # GIVEN
        rule_value = "dummy_value"
        criterions = [rules.Criterion(rules.RuleOperators.EQ, "")]
        rule_conditions = [rules.RuleCondition("dummy_feature", criterions)]
        rule = rules.Rule(
            rule_value, rule_conditions, code_style=rules.CodeStyle.RAW_VARS
        )
        # THEN
        expected_str = (
            f'IF {rule_conditions[0].feature_name} == ""\n'
            f" THEN AVERAGE VALUE OF TARGET IS {rule_value}\n\n"
        )
        self.assertEqual(str(rule), expected_str)


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
