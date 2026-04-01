# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
from unittest import TestCase

import datatable as dt
from datatable import f
from datatable import max

from h2o_sonar.methods.utils.fairness_utils import get_prroc_dt
from h2o_sonar.methods.utils.fairness_utils import smd_multinomial
from tests.test_utils import find_locally


GROUPS_COL_NAME = "Groups"
GROUP_COUNT_NAME = "N"


class TestUtilsDIA(TestCase):
    def test_basic_binary_dia(self):
        # Import creditcard dataset that contains predictions
        path = find_locally("data/predictive/creditcard_with_preds.csv")
        data = dt.fread(path)

        # Actual and prediction columns for creditcard dataset
        actual_column = "DEFAULT_NEXT_MONTH"
        predict_column = "p_DEFAULT_NEXT_MONTH"

        # Calculate prroc
        prroc_frame = get_prroc_dt(data, actual_column, predict_column)
        print(data.names)
        print(prroc_frame.names)
        best_cut = prroc_frame[f.f1 == max(f.f1), :][:, f.cutoff][0, 0]

        expected_best_cut = 0.21
        assert best_cut == expected_best_cut, (
            f"best_cut should be {expected_best_cut} but got {best_cut}"
        )

    def test_basic_smd_multi(self):
        # Import creditcard dataset that contains multinomial predictions
        path = find_locally("data/predictive/creditcard_multinomial.csv")
        data = dt.fread(path)

        # Actual, group column, and reference level
        y = "EDUCATION"
        group_col = "SEX"
        ref_level = "male"

        # Calculate SMD with multinomial outcome
        smd_mult_frame_gender = smd_multinomial(data, y, group_col, ref_level)
        expected_female_smd_dict = {
            "group": ["female"],
            "graduate school": [0.07589038461446762],
            "high school": [-0.04673143848776817],
            "other": [0.0003939093730878085],
            "university": [-0.029552854597568512],
        }
        calc_female_smd_dict = smd_mult_frame_gender[f.group == "female", :].to_dict()

        assert expected_female_smd_dict == calc_female_smd_dict, (
            "Calculated SMD should equal expected SMD for group, female"
        )

        expected_male_smd_dict = {
            "group": ["male"],
            "graduate school": [0],
            "high school": [0],
            "other": [0],
            "university": [0],
        }
        calc_male_smd_dict = smd_mult_frame_gender[f.group == "male", :].to_dict()

        assert expected_male_smd_dict == calc_male_smd_dict, (
            "Calculated SMD should equal expected SMD for group, male"
        )
