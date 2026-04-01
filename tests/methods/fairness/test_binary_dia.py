# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
from unittest import TestCase

import datatable as dt
import numpy as np
import pytest
from datatable import f
from datatable import ltype

from h2o_sonar.methods.fairness._disparate_impact_analysis import (
    BinaryDisparateImpactAnalysis,
)
from tests.test_utils import find_locally


GROUPS_COL_NAME = "Groups"
GROUP_COUNT_NAME = "N"


class TestBinaryDIA(TestCase):
    def test_dia_string_actuals(self):
        self._credit_card_dia_test(True)

    def test_dia_numeric_actuals(self):
        self._credit_card_dia_test()

    def test_dia_numeric_12_actuals(self):
        # Tests for a bug where we the first label is 1
        # and the d_ACTUALS column gets assigned all 1s
        # (which datatable then treats as a bool8 column) and fails when
        # the second label was > 1 (numeric)
        self._credit_card_dia_test(
            data_path="data/predictive/creditcard_12_actuals.csv",
            labels=[1, 2],
            actual_col_names=["actual1", "actual2"],
            ref_level=3,
            metrics_rows=4,
            total_conf=19,
        )

    def _credit_card_dia_test(
        self,
        string_actual=False,
        data_path="data/predictive/creditcard.csv",
        labels=None,
        actual_col_names=None,
        ref_level=0,
        metrics_rows=7,
        total_conf=10000,
    ):
        if actual_col_names is None:
            actual_col_names = ["actualTrue", "actualFalse"]

        # Import creditcard dataset that contains predictions
        path = find_locally(data_path)
        data = dt.fread(path)

        # Parameters for binary DIA
        actual_column = "default payment next month"

        if string_actual:
            data[:, actual_column] = data[:, dt.str32(f[actual_column])]
            labels = ["False", "True"]

        predict_column = "P_DEFAULT_NEXT_MONTH"
        cutoff = 0.21
        group_column = "EDUCATION"

        np.random.seed(1234)
        data[:, predict_column] = dt.Frame(np.random.randint(2, size=data.nrows))

        # Run DIA
        dia = BinaryDisparateImpactAnalysis(
            actual_column=actual_column,
            predict_column=predict_column,
            cutoff=cutoff,
            group_column=group_column,
            labels=labels,
        )

        metrics = dia.get_metrics(data)
        self.assertEqual(metrics.nrows, metrics_rows)

        cm = dia.get_confusion_matrix(data, ref_level)
        total = (
            cm[:, actual_col_names[0]].sum().to_list()[0][0]
            + cm[:, actual_col_names[1]].sum().to_list()[0][0]
        )
        self.assertEqual(total, total_conf)

        # Disparity with original data passed in
        disp = dia.get_disparity(frame=data, ref_level=ref_level, fetch_metrics=True)
        self.assertFalse(all(disp[:, "True Positive Rate Disparity"].to_list()[0]))

        # Disparity with metrics frame passed in
        disp_with_metrics_frame = dia.get_disparity(
            frame=metrics,
            ref_level=ref_level,
            fetch_metrics=False,
            pred_group_frame=data[:, [dia.predict_column, dia.group_column]],
        )
        self.assertFalse(
            all(disp_with_metrics_frame[:, "True Positive Rate Disparity"].to_list()[0])
        )

        # Parity with original dataset passed in
        parity = dia.get_parity(frame=data, ref_level=ref_level)
        [
            self.assertTrue(ltype.bool == parity[:, name].ltypes[0])
            for name in parity.names
            if "Parity" in name
        ]

        self.assertFalse(all(parity[:, "True Positive Rate Parity"].to_list()[0]))

        # Parity with disparity frame passed in
        parity_with_disp_frame = dia.get_parity(
            disp, ref_level=ref_level, get_disparity=False
        )
        [
            self.assertTrue(ltype.bool == parity_with_disp_frame[:, name].ltypes[0])
            for name in parity_with_disp_frame.names
            if "Parity" in name
        ]

        self.assertFalse(
            all(parity_with_disp_frame[:, "True Positive Rate Parity"].to_list()[0])
        )

    def test_basic_binary_dia_sample_weight(self):
        # Import creditcard dataset that contains predictions
        path = find_locally("data/predictive/cc_imbalanced.csv")
        data = dt.fread(path)

        # Parameters for binary DIA
        actual_column = "default.payment.next.month"
        predict_column = "default_payment_next_month.1"
        cutoff = 0.0022
        group_column = "EDUCATION"

        # Run DIA
        dia = BinaryDisparateImpactAnalysis(
            actual_column=actual_column,
            predict_column=predict_column,
            cutoff=cutoff,
            group_column=group_column,
            sample_weight="LIMIT_BAL",
        )

        # Expected shape for binary confusion matrix
        expected_cm_shape = tuple([2, 2])

        # Global confusion matrix
        cm = dia.get_confusion_matrix(frame=data, get_global_cm=True)
        # Ensure shape of confusion matrix is [2,2]
        self.assertEqual(
            cm.shape,
            expected_cm_shape,
            msg=f"Confusion matrix should be of shape [2,2] but got {cm.shape}",
        )

    def test_basic_binary_dia(self):
        # Import creditcard dataset that contains predictions
        path = find_locally("data/predictive/creditcard_with_preds.csv")
        data = dt.fread(path)

        # Parameters for binary DIA
        actual_column = "DEFAULT_NEXT_MONTH"
        predict_column = "p_DEFAULT_NEXT_MONTH"
        cutoff = 0.21
        group_column = "EDUCATION"

        # Run DIA
        dia = BinaryDisparateImpactAnalysis(
            actual_column=actual_column,
            predict_column=predict_column,
            cutoff=cutoff,
            group_column=group_column,
        )

        # Get EDUCATION levels for later testing
        ed_levels = dt.unique(data[:, group_column]).to_list()[0]

        # Confusion matrices globally and per EDUCATION level
        # (['graduate school', 'high school', 'other', 'university'])

        # Expected shape for binary confusion matrix
        expected_cm_shape = tuple([2, 2])

        # Global confusion matrix
        cm = dia.get_confusion_matrix(frame=data, get_global_cm=True)
        # Ensure shape of confusion matrix is [2,2]
        self.assertEqual(
            cm.shape,
            expected_cm_shape,
            msg=f"Confusion matrix should be of shape [2,2] but got {cm.shape}",
        )
        # Golden results for global confusion matrix
        # Based on notebook impl:
        # https://github.com/jphall663/interpretable_machine_learning_with_python/blob/master/dia.ipynb
        self.check_cm(
            cm=cm,
            zero_zero=1253,
            zero_one=1334,
            one_zero=750,
            one_one=5603,
            cm_group="global",
        )

        # `other` confusion matrix
        cm_other = dia.get_confusion_matrix(frame=data, level="other")
        # Ensure shape of confusion matrix is [2,2]
        self.assertEqual(
            cm_other.shape,
            expected_cm_shape,
            msg=f"Confusion matrix should be of shape [2,2] but got {cm_other.shape}",
        )
        # Golden results for `other` confusion matrix
        # Based on notebook impl:
        # https://github.com/jphall663/interpretable_machine_learning_with_python/blob/master/dia.ipynb
        self.check_cm(
            cm=cm_other,
            zero_zero=2,
            zero_one=6,
            one_zero=9,
            one_one=116,
            cm_group="other",
        )

        # `high school` confusion matrix
        cm_hs = dia.get_confusion_matrix(frame=data, level="high school")
        # Ensure shape of confusion matrix is [2,2]
        self.assertEqual(
            cm_hs.shape,
            expected_cm_shape,
            msg=f"Confusion matrix should be of shape [2,2] but got {cm_hs.shape}",
        )
        # Golden results for `high school` confusion matrix
        # Based on notebook impl:
        # https://github.com/jphall663/interpretable_machine_learning_with_python/blob/master/dia.ipynb
        self.check_cm(
            cm=cm_hs,
            zero_zero=253,
            zero_one=280,
            one_zero=102,
            one_one=831,
            cm_group="high school",
        )

        # `university` confusion matrix
        cm_uni = dia.get_confusion_matrix(frame=data, level="university")
        # Ensure shape of confusion matrix is [2,2]
        self.assertEqual(
            cm_uni.shape,
            expected_cm_shape,
            msg=f"Confusion matrix should be of shape [2,2] but got {cm_uni.shape}",
        )
        # Golden results for `university` confusion matrix
        # Based on notebook impl:
        # https://github.com/jphall663/interpretable_machine_learning_with_python/blob/master/dia.ipynb
        self.check_cm(
            cm=cm_uni,
            zero_zero=633,
            zero_one=655,
            one_zero=388,
            one_one=2515,
            cm_group="university",
        )

        # `graduate school` confusion matrix
        cm_gs = dia.get_confusion_matrix(frame=data, level="graduate school")
        # Ensure shape of confusion matrix is [2,2]
        self.assertEqual(
            cm_gs.shape,
            expected_cm_shape,
            msg=f"Confusion matrix should be of shape [2,2] but got {cm_gs.shape}",
        )
        # Golden results for `graduate school` confusion matrix
        # Based on notebook impl:
        # https://github.com/jphall663/interpretable_machine_learning_with_python/blob/master/dia.ipynb
        self.check_cm(
            cm=cm_gs,
            zero_zero=365,
            zero_one=393,
            one_zero=251,
            one_one=2141,
            cm_group="graduate school",
        )

        # Metrics frame

        # Expectations of metrics frame
        expected_metrics_frame_shape = tuple([len(ed_levels), 12])
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
            msg=f"Metrics frame should be of shape "
            f"{list(expected_metrics_frame_shape)} "
            f"but got {metrics_frame.shape}",
        )

        # Check metrics frame col names
        self.assertEqual(
            list(metrics_frame.names),
            expected_metric_col_names,
            msg=f"Expect column names for metric frame to be "
            f"{expected_metric_col_names} but got {metrics_frame.names}",
        )

        # Ensure all numeric values are > 0
        self.assertTrue(
            expr=all(x[0] > 0 for x in metrics_frame[:, dia.metrics].min().to_list()),
            msg=f"All entries in metrics frame should be positive but got "
            f"{metrics_frame}",
        )

        # Disparity frame

        # Expectations of disparity frame
        expected_disp_frame_shape = tuple([len(ed_levels), 14])
        expected_disp_col_names = (
            [GROUPS_COL_NAME]
            + [GROUP_COUNT_NAME]
            + ["Adverse Impact Disparity"]
            + ["Marginal Error", "Standardized Mean Difference"]
            + [x + " Disparity" for x in dia.metrics if x != "Adverse Impact"]
        )

        # Disparity with original dataset passed in
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
            msg=f"Disparity frame should be of shape "
            f"{list(expected_disp_frame_shape)} "
            f"but got {disp_frame.shape}",
        )

        # Check disparity frame col names
        self.assertEqual(
            list(disp_frame.names),
            expected_disp_col_names,
            msg=f"Expect column names for disparity frame to be "
            f"{expected_disp_col_names} but got {disp_frame.names}",
        )

        # Ensure all numeric values are > 0
        disp_frame_pos = disp_frame[
            :,
            [
                x
                for x in disp_frame.names
                if x not in ["Marginal Error", "Standardized Mean Difference"]
            ],
        ]
        expected_disp_pos_col_names = disp_frame_pos.names
        self.assertTrue(
            expr=all(
                x[0] > 0
                for x in disp_frame_pos[
                    :,
                    expected_disp_pos_col_names[1 : len(expected_disp_pos_col_names)],
                ]
                .min()
                .to_list()
            ),
            msg=f"All entries in disparity frame should be positive but got "
            f"{disp_frame}",
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
            msg=f"Disparity frame should be of shape "
            f"{list(expected_disp_frame_shape)} "
            f"but got {disp_with_metrics_frame.shape}",
        )

        # Check disparity frame col names
        self.assertEqual(
            list(disp_with_metrics_frame.names),
            expected_disp_col_names,
            msg=f"Expect column names for disparity frame to be "
            f"{expected_disp_col_names} but got {disp_with_metrics_frame.names}",
        )

        # Ensure all numeric values are > 0
        disp_with_metrics_frame_pos = disp_with_metrics_frame[
            :,
            [
                x
                for x in disp_with_metrics_frame.names
                if x not in ["Marginal Error", "Standardized Mean Difference"]
            ],
        ]
        expected_disp_pos_col_names = disp_with_metrics_frame_pos.names
        self.assertTrue(
            expr=all(
                x[0] > 0
                for x in disp_with_metrics_frame_pos[
                    :,
                    expected_disp_pos_col_names[1 : len(expected_disp_pos_col_names)],
                ]
                .min()
                .to_list()
            ),
            msg=f"All entries in disparity frame should be positive but got "
            f"{disp_with_metrics_frame}",
        )

        # Parity frame

        # Expectations of parity frame
        expected_par_frame_shape = tuple([len(ed_levels) + 1, 17])
        expected_par_col_names = (
            [GROUPS_COL_NAME]
            + ["N"]
            + [col + " Parity" for col in dia.metrics]
            + [
                "Type I Parity",
                "Type II Parity",
                "Equalized Odds",
                "Supervised Fairness",
                "Overall Fairness",
            ]
        )

        # Parity with original dataset passed in
        par_frame = dia.get_parity(
            frame=data, ref_level="graduate school", get_disparity=True
        )

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
            msg=f"Parity frame should be of shape "
            f"{list(expected_par_frame_shape)} "
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
            msg=f"Parity frame should be of shape "
            f"{list(expected_par_frame_shape)} "
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

    def check_cm(
        self,
        cm=None,
        zero_zero=None,
        zero_one=None,
        one_zero=None,
        one_one=None,
        cm_group=None,
    ):
        """

        Parameters
        ----------
        cm:
            Confusion matrix
        zero_zero:
            Expected value for confusion matrix at index [0,0]
        zero_one:
            Expected value for confusion matrix at index [0,1]
        one_zero:
            Expected value for confusion matrix at index [1,0]
        one_one:
            Expected value for confusion matrix at index [1,1]
        cm_group:
            Group of confusion matrix, e.g., `global` means overall confusion
            matrix & `other` would mean confusion matrix for a particular group
            called `other`.

        Returns
        -------

        """
        self.assertEqual(
            cm[0, 0],
            zero_zero,
            msg=f"Value for {cm_group} confusion matrix at index [0,0] should "
            f"be {zero_zero} but got {cm[0, 0]}",
        )
        self.assertEqual(
            cm[0, 1],
            zero_one,
            msg=f"Value for {cm_group} confusion matrix at index [0,1] should "
            f"be {zero_one} but got {cm[0, 1]}",
        )
        self.assertEqual(
            cm[1, 0],
            one_zero,
            msg=f"Value for {cm_group} confusion matrix at index [1,0] should "
            f"be {one_zero} but got {cm[1, 0]}",
        )
        self.assertEqual(
            cm[1, 1],
            one_one,
            msg=f"Value for {cm_group} confusion matrix at index [1,1] should "
            f"be {one_one} but got {cm[1, 1]}",
        )

    def test_cm_creation(self):
        group_column = "group"
        actual_column = "actual"
        predict_column = "preds"
        labels = [False, True]

        dia = BinaryDisparateImpactAnalysis(
            group_column=group_column,
            actual_column=actual_column,
            predict_column=predict_column,
            labels=labels,
            cutoff=0.5,
        )

        frame = dt.Frame(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ],
                [
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                ],  # FP  # TN  # FN  # TP
            ],
            names=list([group_column, actual_column, predict_column]),
        )

        cm = dia.get_confusion_matrix(
            frame=frame,
            level=1,
        )

        """
            tp | fp
            -------
            fn | tn
        """

        assert cm[0, 0] == 1  # tp
        assert cm[0, 1] == 2  # fp
        assert cm[1, 0] == 5  # fn
        assert cm[1, 1] == 4  # tn

    def DISABLED_test_bug_18732_airlines(self):
        #
        # GIVEN
        #
        path = "/tmp/dia-bug-hunt.csv"
        dataset = dt.fread(path)

        # Parameters for binary DIA
        actual_column = "DEP_DEL15"
        predict_column = "C1"
        cutoff = 0.2973864674568176
        group_column = "DAY_OF_MONTH"
        labels = [0, 1]

        # Run DIA
        dia = BinaryDisparateImpactAnalysis(
            actual_column=actual_column,
            predict_column=predict_column,
            cutoff=cutoff,
            group_column=group_column,
            labels=labels,
        )

        #
        # WHEN
        #
        metrics, confusion_matrices = dia.get_metrics(dataset, return_cm=True)

        print(f"DIA: {dia}")
        print(f"Metrics: {metrics}")
        print(f"CMs: {confusion_matrices}")
        self.assertIsNotNone(metrics)
        self.assertIsNotNone(confusion_matrices)

    # https://github.com/h2oai/h2oai/issues/25693
    def DISABLED_test_cm_mismatch_dai_bug_25693(self):
        # GIVEN
        actual_column = "DEFAULT_PAYMENT_NEXT_MONTH"
        predict_column = "predictions"
        group_column = "PAY_5"
        labels = [0, 1]
        cut_off = 0.2898488939
        frame = dt.fread("/tmp/dia-GLOBAL-CM-dataset.csv")

        dia = BinaryDisparateImpactAnalysis(
            actual_column=actual_column,
            predict_column=predict_column,
            group_column=group_column,
            cutoff=cut_off,
            labels=labels,
            sample_weight="",
        )

        # WHEN
        cm = dia.get_confusion_matrix(frame=frame)

        # THEN
        print(f"Global confusion matrix:\n{cm.to_dict()}")
        """
            tp | fp
            -------
            fn | tn
        """
        self.assertIsNotNone(cm)
        assert cm[0, 0] == 2955  # tp
        assert cm[0, 1] == 2526  # fp
        assert cm[1, 0] == 2414  # fn
        assert cm[1, 1] == 16104  # tn

    # https://github.com/h2oai/h2oai/issues/26876
    def test_cm_micro_dataset(self):
        # GIVEN
        frame = dt.Frame(
            {
                "State": [
                    "CO",
                    "CO",
                    "CO",
                    "CO",
                    "CO",
                    "CO",
                    "CO",
                    "CO",
                    "CO",
                    "CO",
                    "CO",
                    "CO",
                    "CO",
                    "CO",
                    "CO",
                    "CO",
                    "CO",
                    "CO",
                    "CO",
                    "CO",
                    "CO",
                ],
                "Account length": [
                    77,
                    77,
                    77,
                    77,
                    77,
                    77,
                    77,
                    77,
                    77,
                    77,
                    77,
                    77,
                    77,
                    77,
                    77,
                    77,
                    77,
                    77,
                    77,
                    77,
                    77,
                ],
                "Area code": [
                    408,
                    408,
                    408,
                    408,
                    408,
                    408,
                    408,
                    408,
                    408,
                    408,
                    408,
                    408,
                    408,
                    408,
                    408,
                    408,
                    408,
                    408,
                    408,
                    408,
                    408,
                ],
                "International plan": [
                    "No",
                    "No",
                    "No",
                    "No",
                    "No",
                    "No",
                    "No",
                    "No",
                    "No",
                    "No",
                    "No",
                    "No",
                    "No",
                    "No",
                    "No",
                    "No",
                    "No",
                    "No",
                    "No",
                    "No",
                    "No",
                ],
                "Voice mail plan": [
                    "No",
                    "No",
                    "No",
                    "No",
                    "No",
                    "No",
                    "No",
                    "No",
                    "No",
                    "No",
                    "No",
                    "No",
                    "No",
                    "No",
                    "No",
                    "No",
                    "No",
                    "No",
                    "No",
                    "No",
                    "No",
                ],
                "Number vmail messages": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                "Total day minutes": [
                    62.4,
                    62.4,
                    62.4,
                    62.4,
                    62.4,
                    62.4,
                    62.4,
                    62.4,
                    62.4,
                    62.4,
                    62.4,
                    62.4,
                    62.4,
                    62.4,
                    62.4,
                    62.4,
                    62.4,
                    62.4,
                    62.4,
                    62.4,
                    62.4,
                ],
                "Total day calls": [
                    89,
                    89,
                    89,
                    89,
                    89,
                    89,
                    89,
                    89,
                    89,
                    89,
                    89,
                    89,
                    89,
                    89,
                    89,
                    89,
                    89,
                    89,
                    89,
                    89,
                    89,
                ],
                "Total day charge": [
                    10.61,
                    0.0,
                    3.14,
                    6.28,
                    9.42,
                    12.56,
                    15.69,
                    18.83,
                    21.97,
                    25.11,
                    28.25,
                    10.61,
                    10.61,
                    10.61,
                    10.61,
                    10.61,
                    10.61,
                    10.61,
                    10.61,
                    10.61,
                    10.61,
                ],
                "Total eve minutes": [
                    169.9,
                    169.9,
                    169.9,
                    169.9,
                    169.9,
                    169.9,
                    169.9,
                    169.9,
                    169.9,
                    169.9,
                    169.9,
                    169.9,
                    169.9,
                    169.9,
                    169.9,
                    169.9,
                    169.9,
                    169.9,
                    169.9,
                    169.9,
                    169.9,
                ],
                "Total eve calls": [
                    121,
                    121,
                    121,
                    121,
                    121,
                    121,
                    121,
                    121,
                    121,
                    121,
                    121,
                    121,
                    121,
                    121,
                    121,
                    121,
                    121,
                    121,
                    121,
                    121,
                    121,
                ],
                "Total eve charge": [
                    14.44,
                    14.44,
                    14.44,
                    14.44,
                    14.44,
                    14.44,
                    14.44,
                    14.44,
                    14.44,
                    14.44,
                    14.44,
                    14.44,
                    14.44,
                    14.44,
                    14.44,
                    14.44,
                    14.44,
                    14.44,
                    14.44,
                    14.44,
                    14.44,
                ],
                "Total night minutes": [
                    209.6,
                    209.6,
                    209.6,
                    209.6,
                    209.6,
                    209.6,
                    209.6,
                    209.6,
                    209.6,
                    209.6,
                    209.6,
                    209.6,
                    209.6,
                    209.6,
                    209.6,
                    209.6,
                    209.6,
                    209.6,
                    209.6,
                    209.6,
                    209.6,
                ],
                "Total night calls": [
                    64,
                    64,
                    64,
                    64,
                    64,
                    64,
                    64,
                    64,
                    64,
                    64,
                    64,
                    64,
                    64,
                    64,
                    64,
                    64,
                    64,
                    64,
                    64,
                    64,
                    64,
                ],
                "Total night charge": [
                    9.43,
                    9.43,
                    9.43,
                    9.43,
                    9.43,
                    9.43,
                    9.43,
                    9.43,
                    9.43,
                    9.43,
                    9.43,
                    9.43,
                    9.43,
                    9.43,
                    9.43,
                    9.43,
                    9.43,
                    9.43,
                    9.43,
                    9.43,
                    9.43,
                ],
                "Total intl minutes": [
                    5.7,
                    5.7,
                    5.7,
                    5.7,
                    5.7,
                    5.7,
                    5.7,
                    5.7,
                    5.7,
                    5.7,
                    5.7,
                    5.7,
                    5.7,
                    5.7,
                    5.7,
                    5.7,
                    5.7,
                    5.7,
                    5.7,
                    5.7,
                    5.7,
                ],
                "Total intl calls": [
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                ],
                "Total intl charge": [
                    1.54,
                    1.54,
                    1.54,
                    1.54,
                    1.54,
                    1.54,
                    1.54,
                    1.54,
                    1.54,
                    1.54,
                    1.54,
                    1.54,
                    1.54,
                    1.54,
                    1.54,
                    1.54,
                    1.54,
                    1.54,
                    1.54,
                    1.54,
                    1.54,
                ],
                "Customer service calls": [
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                ],
                "Churn": [
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                "C1": [
                    0.942444920539856,
                    0.942444920539856,
                    0.942444920539856,
                    0.942444920539856,
                    0.942444920539856,
                    0.942444920539856,
                    0.942444920539856,
                    0.9389113783836365,
                    0.929356038570404,
                    0.8981227874755859,
                    0.8185784816741943,
                    0.942444920539856,
                    0.942444920539856,
                    0.942444920539856,
                    0.942444920539856,
                    0.942444920539856,
                    0.942444920539856,
                    0.942444920539856,
                    0.942444920539856,
                    0.942444920539856,
                    0.942444920539856,
                ],
            }
        )
        dia = BinaryDisparateImpactAnalysis(
            actual_column="Churn",
            predict_column="C1",
            group_column="Account length",
            cutoff=0.3790166974,
            labels=[0, 1],
        )

        # THEN
        with pytest.raises(ValueError):
            # WHEN
            dia.get_confusion_matrix(frame)

    # https://github.com/h2oai/h2oai/issues/27677
    def test_data_precision_issue(self):
        # GIVEN
        actual_column = "DEFAULT_PAYMENT_NEXT_MONTH"
        predict_column = "predictions"
        group_column = "PAY_5"
        labels = [0, 1]
        cut_off = 0.2672095895
        path = find_locally("data/predictive/creditcard_float32_dia.bin")
        test_data = dt.fread(path)

        dia = BinaryDisparateImpactAnalysis(
            actual_column=actual_column,
            predict_column=predict_column,
            group_column=group_column,
            cutoff=cut_off,
            labels=labels,
        )

        # WHEN
        cm = dia.get_confusion_matrix(frame=test_data)

        # THEN
        self.assertIsNotNone(cm)
        print(f"Global confusion matrix:\n{cm.to_dict()}")
        self.check_cm(
            cm=cm,
            zero_zero=2866,
            zero_one=2578,
            one_zero=2503,
            one_one=16052,
        )

    def test_data_sample_weight(self):
        # GIVEN
        actual_column = "default payment next month"
        predict_column = "predictions"
        group_column = "MARRIAGE"
        labels = [0, 1]
        cut_off = 0.4368953705
        weight = "AGE"
        path = find_locally("data/predictive/creditcard100_pred.csv")
        test_data = dt.fread(path)

        dia = BinaryDisparateImpactAnalysis(
            actual_column=actual_column,
            predict_column=predict_column,
            group_column=group_column,
            cutoff=cut_off,
            labels=labels,
            sample_weight=weight,
        )

        # WHEN
        cm = dia.get_confusion_matrix(frame=test_data)

        # THEN
        self.assertIsNotNone(cm)
        print(f"Global confusion matrix:\n{cm.to_dict()}")
        self.check_cm(
            cm=cm,
            zero_zero=2700,
            zero_one=279,
            one_zero=261,
            one_one=399,
        )
