# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import os
import tempfile
from functools import partial
from unittest import TestCase

import pandas as pd

from h2o_sonar import loggers as logging
from h2o_sonar.errors import MliUnsupportedDataFormatError
from h2o_sonar.methods._pd import PD
from tests.methods.ice_pd_test_commons import FooScorerConstPredictFrame
from tests.methods.ice_pd_test_commons import IceStrategyFactory
from tests.methods.test_pd_regression import FooScorerSumRegrFrame
from tests.test_utils import rm_test_dir


class TestPdResiduals(TestCase):
    """Test Partial Dependency (PD) on residuals."""

    def setUp(self):
        logging.setLevel(logging.DEBUG)

        self.strategy = IceStrategyFactory()

        # data: 3x3 for value checks
        self.x3x3 = pd.DataFrame({"f1": [1, 2, 3], "F": [1, 3, 5], "f2": [0.5, 2, 3]})
        self.y3x1 = pd.DataFrame({"Y": [1, 2, 30]})
        self.y2x1 = pd.DataFrame({"Y": [10, 20]})
        self.y3x3 = pd.DataFrame(
            {"c1": [10, 20, 30], "c2": [10, 30, 50], "c3": [0.5, 0.2, 0.3]}
        )

        # prediction method lambda
        self.score_sum = partial(FooScorerSumRegrFrame().score_batch, fast_approx=True)

        # test case settings
        self.test_current_dir_persistence = False

    def test_negative_init(self):
        with self.assertRaises(ValueError):
            PD("Wrong Y shape").explain(
                ["F"], self.x3x3, Y=self.y2x1, predict_method=self.score_sum
            )
        with self.assertRaises(MliUnsupportedDataFormatError):
            PD("Wrong datatype").explain(
                ["F"], self.x3x3, Y=[[1, 2, 3]], predict_method=self.score_sum
            )

    def test_identity(self):
        logging.debug("# PD: residuals w/ identity transform ###")

        # GIVEN
        logging.debug(f"X:\n{self.x3x3}")
        logging.debug(f"Y:\n{self.y3x1}")
        features = ["F"]
        bins = [[1, 5]]

        # WHEN
        pdp = PD("PD on rs").explain(
            features,
            self.x3x3,
            Y=self.y3x1,
            predict_method=self.score_sum,
            bins=bins,
        )

        # THEN
        self.assertNotEqual(None, pdp.explanations())
        self.assertEqual(
            len(pdp.explanations()),
            len(features),
            "Result and explanations size must fit",
        )

        logging.debug(f"PD w/ residuals PD:\n{pdp}")
        ex = pdp.explanations()[features[0]][PD.LABEL_REGRESSION]

        logging.debug(f"\nX\n{str(ex)}")

        self.assertEqual(
            len(ex.columns), len(bins[0]), "bin and residuals cols must fit"
        )
        self.assertEqual(
            7, len(ex.index), "7 rows expected: 2x(MEAN, SD, SEM) + OOR hint"
        )
        self.assertEqual(6.166666666666667, ex[1][PD.COL_RESIDUAL_MEAN], "Wrong MEAN")
        self.assertEqual(14.597374193098338, ex[1][PD.COL_RESIDUAL_SD], "Wrong SD")
        self.assertEqual(8.427797919847022, ex[1][PD.COL_RESIDUAL_SEM], "Wrong SEM")

    def test_abs(self):
        logging.debug("# PD: residuals w/ abs transform ###")

        # GIVEN
        logging.debug(f"X:\n{self.x3x3}")
        logging.debug(f"Y:\n{self.y3x1}")
        features = ["F"]
        bins = [[1, 5]]

        # WHEN
        pdp = self.strategy.get_pd().explain(
            features,
            self.x3x3,
            Y=self.y3x1,
            predict_method=self.score_sum,
            bins=bins,
            target_transform=abs,
        )

        # THEN
        self.assertNotEqual(None, pdp.explanations())
        self.assertEqual(
            len(pdp.explanations()),
            len(features),
            "Result and explanations size must fit",
        )

        logging.debug(f"PD w/ residuals PD:\n{pdp}")
        ex = pdp.explanations()[features[0]][PD.LABEL_REGRESSION]

        logging.debug(f"\nX\n{str(ex)}")

        self.assertEqual(
            len(ex.columns), len(bins[0]), "bin and residuals cols must fit"
        )
        self.assertEqual(7, len(ex.index), "7 rows expected: 2x(MEAN, SD, SEM) + OOR")
        self.assertEqual(9.166666666666666, ex[1][PD.COL_RESIDUAL_MEAN], "Wrong MEAN")
        self.assertEqual(12.003471720020558, ex[1][PD.COL_RESIDUAL_SD], "Wrong SD")
        self.assertEqual(6.93020762876393, ex[1][PD.COL_RESIDUAL_SEM], "Wrong SEM")

    def test_json(self):
        logging.debug("# PD: residuals ###")

        # GIVEN
        features = ["F"]
        bins = [[1, 5]]

        pdp = self.strategy.get_pd().explain(
            features,
            self.x3x3,
            Y=self.y3x1,
            predict_method=self.score_sum,
            bins=bins,
        )
        explanations = pdp.explanations()

        # save
        logging.debug("SAVING JSon")
        tmp_dir = tempfile.mkdtemp(prefix="mli_unit_pdp_")
        tmp_file_name = os.path.join(tmp_dir, "unit-pdp-save-basic.json")
        try:
            # WHEN
            pdp.save_json(tmp_file_name)

            # NEGATIVE: save check overwrite
            with self.assertRaises(FileExistsError):
                pdp.save_json(tmp_file_name)

            # load
            logging.debug("LOADING JSon...")
            pdp.evict_explanations()
            pdp.load_json(tmp_file_name)
            l_explanations = pdp.explanations()

            # THEN
            self.assertIsInstance(explanations, dict, "Result to be dictionary")
            for f in l_explanations:
                logging.debug(f"PD explanation for {f} is:\n{str(l_explanations[f])}")

            # assert save/load roundtrip
            self.assertEqual(
                len(explanations),
                len(l_explanations),
                "Explanations cannot have different size after SAVE/LOAD",
            )
            self.assertEqual(
                explanations.keys(),
                l_explanations.keys(),
                "Explanations keys cannot be different after SAVE/LOAD",
            )
            for f in l_explanations:
                logging.debug(f"SRC: {f}\n{str(explanations[f])}")
                logging.debug(f"S/L: {f}\n{str(l_explanations[f])}")
                self.assertEqual(
                    explanations[f][PD.LABEL_REGRESSION].shape,
                    l_explanations[f][PD.LABEL_REGRESSION].shape,
                )
                for c in explanations[f][PD.LABEL_REGRESSION].columns.values:
                    for r in explanations[f][PD.LABEL_REGRESSION].index.values:
                        self.assertEqual(
                            explanations[f][PD.LABEL_REGRESSION][c][r],
                            l_explanations[f][PD.LABEL_REGRESSION][c][r],
                        )
        finally:
            rm_test_dir(tmp_dir)
            pass

        # NEGATIVE: load non-existent file
        with self.assertRaises(FileNotFoundError):
            pdp.load_json("/THIS-IS-NON-EXISTENT-FILE")

        # save/load to/from CURRENT dir
        if self.test_current_dir_persistence:
            # save to current dir
            logging.debug("SAVING to current directory: " + os.getcwd())
            pdp.save_json()
            # load from current dir
            logging.debug("LOADING from current directory: " + os.getcwd())
            pdp.load_json()
            self.assertIsInstance(explanations, dict, "Result to be dictionary")
            self.assertEqual(len(explanations), len(pdp.explanations()))

    def test_multinomial(self):
        logging.debug("# PD: residuals for multinomial ###")

        # GIVEN
        logging.debug(f"X:\n{self.x3x3}")
        logging.debug(f"Y:\n{self.y3x3}")
        features = ["F"]
        bins = [[1, 5]]
        scorer = FooScorerConstPredictFrame(self.x3x3.shape[0], 3)

        # WHEN
        pdp = self.strategy.get_pd().explain(
            features,
            self.x3x3,
            Y=self.y3x3,
            predict_method=scorer.score_batch,
            bins=bins,
            target_transform=abs,
        )

        # THEN
        class_1 = PD.LABEL_PREFIX_CLASS + "1"
        self.assertNotEqual(None, pdp.explanations())
        self.assertEqual(
            len(pdp.explanations()),
            len(features),
            "Result and explanations size must fit",
        )

        logging.debug(f"PD w/ residuals PD:\n{pdp}")
        ex = pdp.explanations()[features[0]][class_1]
        logging.debug(f"\nX\n{str(ex)}")

        self.assertEqual(
            len(ex.columns), len(bins[0]), "bin and residuals cols must fit"
        )
        self.assertEqual(7, len(ex.index), "7 rows expected: 2x(MEAN, SD, SEM) + OOR")
        self.assertEqual(24, ex[1][PD.COL_RESIDUAL_MEAN], "Wrong MEAN")
        self.assertEqual(19, ex[1][PD.COL_RESIDUAL_SD], "Wrong SD")
        self.assertEqual(10.96965511460289, ex[1][PD.COL_RESIDUAL_SEM], "Wrong SEM")
