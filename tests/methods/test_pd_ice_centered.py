# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
from unittest import TestCase

import pandas as pd

from h2o_sonar import loggers as logging
from h2o_sonar.methods._pd import ICE
from h2o_sonar.methods._pd import PD
from tests.methods.ice_pd_test_commons import IceStrategyFactory


class TestPdIceCentered(TestCase):
    """Test c-ICE and c-PD."""

    def setUp(self):
        logging.setLevel(logging.DEBUG)

        self.strategy = IceStrategyFactory()

        # data
        self.x6x3 = pd.DataFrame(
            {
                "f1": [1, 2, 3, 4, 5, 6],
                "F": ["cat", "dog", "cat", "sheep", "cat", "dog"],
                "f3": [50, 40, 30, 20, 10, 0],
            }
        )
        self.x18x3 = pd.DataFrame(
            {
                "f1": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 3],
                "F": [
                    "cat",
                    "dog",
                    "cat9",
                    "sheep",
                    "cat5",
                    "dog4",
                    "cat1",
                    "dog",
                    "cat1",
                    "1sheep",
                    "2cat",
                    "1dog",
                    "3cat",
                    "2dog",
                    "cat",
                    "sheep5",
                    "cat",
                    "dog8",
                ],
                "f3": [
                    50,
                    40,
                    30,
                    20,
                    10,
                    0,
                    55,
                    45,
                    35,
                    25,
                    15,
                    3,
                    50,
                    40,
                    30,
                    20,
                    10,
                    0,
                ],
            }
        )

    @staticmethod
    def foo_predict_method(x):
        return (x["f1"] + x["f3"]) / 100

    def test_c_ice(self):
        # GIVEN
        fs = ["f3"]

        # WHEN
        ice = self.strategy.get_ice().explain(
            fs,
            self.x6x3,
            predict_method=TestPdIceCentered.foo_predict_method,
            mins=[0],
            maxs=[55],
            center=False,
        )
        c_ice = self.strategy.get_ice().explain(
            fs,
            self.x6x3,
            predict_method=self.foo_predict_method,
            mins=[0],
            maxs=[55],
            center=True,
        )

        # THEN
        self.assertIsNotNone(ice)
        logging.debug(f"ICEs:\n{ice}")
        self.assertIsNotNone(c_ice)

        logging.debug(f"c-ICEs:\n{c_ice}")
        self.assertIsNotNone(ice.explanations())

        self.assertEqual(
            ice.explanations()[fs[0]][ICE.LABEL_REGRESSION].shape,
            c_ice.explanations()[fs[0]][ICE.LABEL_REGRESSION].shape,
        )
        self.assertEqual(0.01, ice.explanations()[fs[0]][ICE.LABEL_REGRESSION][0][0])
        self.assertEqual(
            -0.29500000000000004,
            c_ice.explanations()[fs[0]][ICE.LABEL_REGRESSION][0][0],
        )

    def test_c_pdp(self):
        # GIVEN
        fs = ["f3"]

        # WHEN
        pdp = self.strategy.get_pd().explain(
            fs,
            self.x6x3,
            predict_method=TestPdIceCentered.foo_predict_method,
            center=False,
        )
        c_pdp = self.strategy.get_pd().explain(
            fs, self.x6x3, predict_method=self.foo_predict_method, center=True
        )

        # THEN
        self.assertIsNotNone(pdp)
        logging.debug(f"PDs:\n{pdp}")
        self.assertIsNotNone(c_pdp)

        logging.debug(f"c-PDs:\n{c_pdp}")
        self.assertIsNotNone(pdp.explanations())

        self.assertEqual(
            pdp.explanations()[fs[0]][PD.LABEL_REGRESSION].shape,
            c_pdp.explanations()[fs[0]][PD.LABEL_REGRESSION].shape,
        )
        self.assertEqual(0.035, pdp.explanations()[fs[0]][PD.LABEL_REGRESSION][0][0])
        self.assertEqual(
            -0.27,
            c_pdp.explanations()[fs[0]][PD.LABEL_REGRESSION][0][0],
        )
