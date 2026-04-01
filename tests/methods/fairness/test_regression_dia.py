# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
from unittest import TestCase

import datatable as dt
from datatable import ltype

from h2o_sonar.methods.fairness._disparate_impact_analysis import (
    RegressionDisparateImpactAnalysis,
)
from tests.test_utils import find_locally


GROUPS_COL_NAME = "Groups"
GROUP_COUNT_NAME = "N"


class TestRegressionDIA(TestCase):
    def test_nan_str_ref_level(self):
        # Import creditcard dataset that contains predictions
        path = find_locally("data/predictive/creditcard_regression_with_preds.csv")
        data = dt.fread(path)

        # Parameters for regression DIA
        actual_column = "LIMIT_BAL"
        predict_column = "p_LIMIT_BAL"
        group_column = "SEX"

        # Run DIA
        dia = RegressionDisparateImpactAnalysis(
            actual_column=actual_column,
            predict_column=predict_column,
            group_column=group_column,
        )

        # Metrics frame
        metrics = dia.get_metrics(data)

        # Used to fail when None was present in a column
        parity = dia.get_disparity(
            metrics,
            ref_level=None,
            fetch_metrics=False,
            pred_group_frame=data[:, [dia.predict_column, dia.group_column]],
        )

        dia.get_parity(parity, ref_level=None, get_disparity=False)

    def test_nan_numeric_ref_level(self):
        # Import creditcard dataset that contains predictions
        path = find_locally("data/predictive/creditcard_regression_with_preds.csv")
        data = dt.fread(path)

        # Parameters for regression DIA
        actual_column = "LIMIT_BAL"
        predict_column = "p_LIMIT_BAL"
        group_column = "AGE"

        # Run DIA
        dia = RegressionDisparateImpactAnalysis(
            actual_column=actual_column,
            predict_column=predict_column,
            group_column=group_column,
        )

        # Metrics frame
        # Used to fail when None was present in a column
        dia.get_metrics(data)

    def test_basic_regression_dia(self):
        # Import creditcard dataset that contains predictions
        path = find_locally("data/predictive/creditcard_regression_with_preds.csv")
        data = dt.fread(path)

        # Parameters for regression DIA
        actual_column = "LIMIT_BAL"
        predict_column = "p_LIMIT_BAL"
        group_column = "EDUCATION"

        # Run DIA
        dia = RegressionDisparateImpactAnalysis(
            actual_column=actual_column,
            predict_column=predict_column,
            group_column=group_column,
        )

        # Get EDUCATION levels for later testing
        ed_levels = dt.unique(data[:, group_column]).to_list()[0]

        # Metrics frame

        # Expectations of metrics frame
        expected_metrics_frame_shape = tuple([len(ed_levels), 8])
        expected_metric_col_names = [GROUPS_COL_NAME] + [GROUP_COUNT_NAME] + dia.metrics
        metrics_frame = dia.get_metrics(data)

        # Check groups are correct
        self.assertEqual(
            metrics_frame[:, GROUPS_COL_NAME].to_list()[0],
            ed_levels,
            msg=f"Group column in metrics frame should have values "
            f"{ed_levels}, but got "
            f"{metrics_frame[:, GROUPS_COL_NAME].to_pandas()}",
        )

        # Check metrics frame shape
        self.assertEqual(
            metrics_frame.shape,
            expected_metrics_frame_shape,
            msg=f"Metrics frame should be of shape {[len(ed_levels), 8]} "
            f"but got {metrics_frame.shape}",
        )

        # Check metrics frame col names
        self.assertEqual(
            list(metrics_frame.names),
            expected_metric_col_names,
            msg=f"Expect column names for metric frame to be "
            f"{expected_metric_col_names} but got {metrics_frame.names}",
        )

        # Disparity frame

        # Expectations of disparity frame
        expected_disp_frame_shape = tuple([len(ed_levels), 9])
        expected_disp_col_names = (
            [GROUPS_COL_NAME]
            + [GROUP_COUNT_NAME]
            + ["Standardized Mean Difference"]
            + [col + " Disparity" for col in dia.metrics]
        )

        # Disparity with original data passed in
        disp_frame = dia.get_disparity(
            frame=data, ref_level="graduate school", fetch_metrics=True
        )

        # Check groups are correct
        self.assertEqual(
            disp_frame[:, GROUPS_COL_NAME].to_list()[0],
            ed_levels,
            msg=f"Group column in disparity frame should have values "
            f"{ed_levels}, but got "
            f"{disp_frame[:, GROUPS_COL_NAME].to_pandas()}",
        )

        # Check disparity frame shape
        self.assertEqual(
            disp_frame.shape,
            expected_disp_frame_shape,
            msg=f"Disparity frame should be of shape {[len(ed_levels), 9]} "
            f"but got {disp_frame.shape}",
        )

        # Check disparity frame col names
        self.assertEqual(
            list(disp_frame.names),
            expected_disp_col_names,
            msg=f"Expect column names for disparity frame to be "
            f"{expected_disp_col_names} but got {disp_frame.names}",
        )

        # Disparity with metrics frame passed in
        disp_with_metrics_frame = dia.get_disparity(
            frame=metrics_frame,
            ref_level="graduate school",
            fetch_metrics=False,
            pred_group_frame=data[:, [dia.predict_column, dia.group_column]],
        )

        # Check groups are correct
        self.assertEqual(
            disp_with_metrics_frame[:, GROUPS_COL_NAME].to_list()[0],
            ed_levels,
            msg=f"Group column in disparity frame should have values "
            f"{ed_levels}, but got "
            f"{disp_with_metrics_frame[:, GROUPS_COL_NAME].to_pandas()}",
        )

        # Check disparity frame shape
        self.assertEqual(
            disp_with_metrics_frame.shape,
            expected_disp_frame_shape,
            msg=f"Disparity frame should be of shape {[len(ed_levels), 9]} "
            f"but got {disp_with_metrics_frame.shape}",
        )

        # Check disparity frame col names
        self.assertEqual(
            list(disp_with_metrics_frame.names),
            expected_disp_col_names,
            msg=f"Expect column names for disparity frame to be "
            f"{expected_disp_col_names} but got {disp_with_metrics_frame.names}",
        )

        # Parity frame

        # Expectations of parity frame
        expected_par_frame_shape = tuple([len(ed_levels) + 1, 9])
        expected_par_col_names = (
            [GROUPS_COL_NAME]
            + [GROUP_COUNT_NAME]
            + [col + " Parity" for col in dia.metrics]
            + [
                "Overall Fairness",
            ]
        )

        # Parity with original dataset passed in
        par_frame = dia.get_parity(
            frame=data, ref_level="graduate school", get_disparity=True
        )

        [
            self.assertTrue(ltype.bool == par_frame[:, name].ltypes[0])
            for name in par_frame.names
            if "Parity" in name
        ]
        # Check groups are correct
        self.assertEqual(
            par_frame[:, GROUPS_COL_NAME].to_list()[0],
            ed_levels + ["all"],
            msg=f"Group column in parity frame should have values "
            f"{ed_levels}, but got "
            f"{par_frame[:, GROUPS_COL_NAME].to_pandas()}",
        )
        # Check parity frame shape
        self.assertEqual(
            par_frame.shape,
            expected_par_frame_shape,
            msg=f"Parity frame should be of shape {[len(ed_levels) + 1, 9]} "
            f"but got {par_frame.shape}",
        )

        # Check parity frame col names
        self.assertEqual(
            list(par_frame.names),
            expected_par_col_names,
            msg=f"Expect column names for parity frame to be "
            f"{expected_par_col_names} but got {par_frame.names}",
        )

        # Ensure all numeric values are either 1 or 0 (boolean)
        self.assertTrue(
            expr=all(
                x[0] == 1 or x[0] == 0
                for x in par_frame[
                    :, expected_par_col_names[2 : len(expected_par_col_names)]
                ]
                .min()
                .to_list()
            ),
            msg=f"All entries in parity frame should be 1 or 0 (boolean) but "
            f"got {par_frame.to_pandas()}",
        )

        # Parity with disparity frame passed in
        par_with_disp_frame = dia.get_parity(
            disp_frame, ref_level="graduate school", get_disparity=False
        )

        [
            self.assertTrue(ltype.bool == par_with_disp_frame[:, name].ltypes[0])
            for name in par_with_disp_frame.names
            if "Parity" in name
        ]
        # Check groups are correct
        self.assertEqual(
            par_with_disp_frame[:, GROUPS_COL_NAME].to_list()[0],
            ed_levels + ["all"],
            msg=f"Group column in parity frame should have values "
            f"{ed_levels}, but got "
            f"{par_with_disp_frame[:, GROUPS_COL_NAME].to_pandas()}",
        )
        # Check parity frame shape
        self.assertEqual(
            par_with_disp_frame.shape,
            expected_par_frame_shape,
            msg=f"Parity frame should be of shape {[len(ed_levels) + 1, 9]} "
            f"but got {par_with_disp_frame.shape}",
        )

        # Check parity frame col names
        self.assertEqual(
            list(par_with_disp_frame.names),
            expected_par_col_names,
            msg=f"Expect column names for parity frame to be "
            f"{expected_par_col_names} but got {par_with_disp_frame.names}",
        )

        # Ensure all numeric values are either 1 or 0 (boolean)
        self.assertTrue(
            expr=all(
                x[0] == 1 or x[0] == 0
                for x in par_with_disp_frame[
                    :, expected_par_col_names[2 : len(expected_par_col_names)]
                ]
                .min()
                .to_list()
            ),
            msg=f"All entries in parity frame should be 1 or 0 (boolean) but "
            f"got {par_with_disp_frame.to_pandas()}",
        )
