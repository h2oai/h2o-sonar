# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import os
import tempfile
import time
from functools import partial
from unittest import TestCase

import pandas as pd

from h2o_sonar import loggers as logging
from h2o_sonar.methods._ice import ICE
from h2o_sonar.methods._pd import PD
from tests.methods.test_pd_robustness import DATASET_LOAN_10_PATH
from tests.methods.test_pd_robustness import FooScorerRegrSeries
from tests.test_utils import rm_test_dir


class TestPdIceCaching(TestCase):
    """Test PD's caching of ICE."""

    # override
    def setUp(self):
        logging.setLevel(logging.DEBUG)

        self.score_foo = partial(FooScorerRegrSeries().score_batch, fast_approx=True)

    @staticmethod
    def ice_save_and_load_datatable_jay(
        test_case, dataset_path, predict_method, row_index, show, do_then
    ):
        # import must be local to avoid UnitTest to run also test case below
        from tests.methods.test_pd_robustness import TestPdIceRobustness

        # GIVEN
        df = TestPdIceRobustness._given_dataset(dataset_path)
        logging.debug(f"X: {df.shape}")
        target_features = df.columns.values.tolist()
        bins = PD.create_unique_bins(target_features, df)
        ice = ICE("ICE")
        start = time.time()
        ice = ice.explain(target_features, df, predict_method=predict_method, bins=bins)
        logging.debug(f"  ICE computed in {time.time() - start}s")
        explanations = ice.explanations()
        if show:
            logging.debug(f"ICE: {ice}")

        # WHEN
        # save
        tmp_dir = tempfile.mkdtemp(prefix="mli_unit_ice_")
        tmp_file_name = os.path.join(tmp_dir, "unit-ice-save-jay.json")
        logging.debug(f"SAVING datatable to {tmp_file_name}")

        try:
            start = time.time()
            ice.save(tmp_file_name)
            logging.debug(f"  saved in {time.time() - start}s")

            # load
            logging.debug(f"LOADING jay from {tmp_file_name}")
            start = time.time()
            ice.load(tmp_file_name, row_index=row_index)
            logging.debug(f"  loaded in {time.time() - start}s")

            l_explanations = ice.explanations()

            # THEN
            if do_then:
                test_case.assertIsInstance(
                    explanations, dict, "Result to be dictionary"
                )
                if show:
                    for f in l_explanations:
                        logging.debug(
                            f"\nICE explanation for {f} is:\n"
                            f"{str(l_explanations[f][ICE.LABEL_REGRESSION])}"
                        )

                # assert save/load roundtrip
                test_case.assertEqual(
                    len(explanations),
                    len(l_explanations),
                    "Explanations cannot have different size after SAVE/LOAD",
                )
                test_case.assertEqual(
                    explanations.keys(),
                    l_explanations.keys(),
                    "Explanations keys cannot be different after SAVE/LOAD",
                )
                for f in l_explanations:
                    # index is LOST when datatable persistency is used
                    explanations[f][ICE.LABEL_REGRESSION] = explanations[f][
                        ICE.LABEL_REGRESSION
                    ].reset_index(drop=True)

                    if show:
                        logging.debug(
                            f"SRC: {f}\n{str(explanations[f][ICE.LABEL_REGRESSION])}"
                        )
                        logging.debug(
                            f"S/L: {f}\n{str(l_explanations[f][ICE.LABEL_REGRESSION])}"
                        )

                    test_case.assertEqual(
                        explanations[f][ICE.LABEL_REGRESSION].shape,
                        l_explanations[f][ICE.LABEL_REGRESSION].shape,
                    )

                    for c in explanations[f][ICE.LABEL_REGRESSION].columns.values:
                        for r in explanations[f][ICE.LABEL_REGRESSION].index.values:
                            logging.debug(f"Checking: f:{f}, c:{c}, r:{r}")
                            test_case.assertEqual(
                                explanations[f][ICE.LABEL_REGRESSION][c][r],
                                # IMPORTANT .jay screws column types by
                                # converting all of them to string
                                l_explanations[f][ICE.LABEL_REGRESSION][str(c)][r],
                            )
        finally:
            rm_test_dir(tmp_dir)

        return l_explanations

    def test_ice_save_and_load_datatable_jay_sanity(self):
        TestPdIceCaching.ice_save_and_load_datatable_jay(
            self, DATASET_LOAN_10_PATH, self.score_foo, None, True, True
        )

    def test_loads_1_ice_row_per_feature(self):
        exs = TestPdIceCaching.ice_save_and_load_datatable_jay(
            self, DATASET_LOAN_10_PATH, self.score_foo, 1, False, False
        )

        # THEN
        for f in exs:
            logging.debug(f"SRC: {f}\n{str(exs[f][ICE.LABEL_REGRESSION])}")
            logging.debug(f"S/L: {f}\n{str(exs[f][ICE.LABEL_REGRESSION])}")
            self.assertEqual(1, exs[f][ICE.LABEL_REGRESSION].shape[0])

    def test_append_ices(self):
        # import must be local to avoid UnitTest to run also test case below
        from tests.methods.test_pd_robustness import TestPdIceRobustness

        # GIVEN
        data = TestPdIceRobustness._given_dataset(DATASET_LOAN_10_PATH)

        tmp_dir = tempfile.mkdtemp(prefix="mli_unit_ice_")
        tmp_file_name = os.path.join(tmp_dir, "unit-ice-save-jay.json")

        try:
            for i in range(2):
                df = data[[data.columns[i]]]
                logging.debug(f"X{i}: {df.shape}, {str(df)}")

                target_features = df.columns.values.tolist()
                bins = PD.create_unique_bins(target_features, df)
                ice = ICE("ICE")
                ices = ice.explain(
                    target_features,
                    df,
                    predict_method=self.score_foo,
                    bins=bins,
                )

                # WHEN
                logging.debug(f"SAVING jay to {tmp_file_name}")
                ices.save(tmp_file_name, append=True)

            # THEN
            logging.debug(f"LOADING jay from {tmp_file_name}")
            ices.load(tmp_file_name)
            logging.debug(f"ICE: {ices}")

            exs = ices.explanations()
            self.assertEqual(2, len(exs))
            for feature in exs:
                df_ice = exs[feature][PD.LABEL_PREFIX_CLASS + "0"]
                logging.debug(f"{feature}: {df_ice.shape}")
                self.assertIsInstance(df_ice, pd.DataFrame)
                self.assertEqual(9, df_ice.shape[0])
                self.assertGreaterEqual(df_ice.shape[1], 7)

        finally:
            rm_test_dir(tmp_dir)

    def test_negative_ice_load_sampled_instance(self):
        # import must be local to avoid UnitTest to run also test case below
        from tests.methods.test_pd_robustness import TestPdIceRobustness

        # GIVEN
        df = TestPdIceRobustness._given_dataset(DATASET_LOAN_10_PATH)

        tmp_dir = tempfile.mkdtemp(prefix="mli_unit_ice_")
        tmp_file_name = os.path.join(tmp_dir, "unit-ice-save-jay.json")

        try:
            logging.debug(f"X: {df.shape}, {str(df)}")
            target_features = ["loan_amnt"]
            bins = PD.create_unique_bins(target_features, df)
            ice = ICE("ICE")
            ice = ice.explain(
                target_features, df, predict_method=self.score_foo, bins=bins
            )
            logging.debug(f"SAVING jay to {tmp_file_name}")
            ice.save(tmp_file_name, append=True)

            # WHEN
            logging.debug(f"LOADING non-existent row from {tmp_file_name}")
            with self.assertRaises(ValueError):
                # THEN
                ice.load(tmp_file_name, row_index=int(1e5))

        finally:
            rm_test_dir(tmp_dir)
