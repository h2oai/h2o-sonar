# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import math
from functools import partial
from unittest import TestCase

import pandas as pd

from h2o_sonar import loggers as logging
from h2o_sonar.methods._ice import ICE
from h2o_sonar.methods._pd import PD
from tests.methods.ice_pd_test_commons import FooScorerRegrSeries


class TestPdCategorical(TestCase):
    """Test PD on binomial and categorical features."""

    def setUp(self):
        logging.setLevel(logging.DEBUG)

        # data
        self.xBool = pd.DataFrame(
            {"f1": [1, 2, 3], "F": [True, False, True], "f2": [1, 2, 3]}
        )
        self.xCat = pd.DataFrame(
            {
                "f1": [1, 2, 3, 4, 5, 6],
                "F": ["cat", "dog", "cat", "sheep", "cat", "dog"],
                "f2": [50, 40, 30, 20, 10, 0],
            }
        )
        self.xCatUnseen = pd.DataFrame(
            {
                "f1": [1, 2, 3, 4, 5, 6],
                "F": ["UNSEEN", "dog", "dog", "sheep", "sheep", "sheep"],
                "f2": [50, 40, 30, 20, 10, 0],
            }
        )
        self.xCatUnseenOrder = pd.DataFrame(
            {
                "f1": [1, 2, 3, 4, 5, 6],
                "F": ["ALBATROS", "DOG", "DOG", "ZEBRA", "ZEBRA", "ZEBRA"],
                "f2": [60, 50, 40, 30, 20, 10],
            }
        )
        self.xManyCats = pd.DataFrame(
            {
                "f1": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 3],
                "F": [
                    "cat",
                    "dog",
                    "dog",
                    "sheep",
                    "sheep",
                    "sheep",
                    "cat1",
                    "cat1",
                    "cat1",
                    "cat1",
                    "1dog",
                    "1dog",
                    "1dog",
                    "1dog",
                    "1dog",
                    "1dog",
                    "1dog",
                    "1dog",
                ],
                "f2": [
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
        self.xCatValFreq = pd.DataFrame(
            {
                # unique
                "f1": [
                    6,
                    1,
                    1,
                    2,
                    2,
                    2,
                    4,
                    4,
                    4,
                    4,
                    5,
                    5,
                    5,
                    5,
                    5,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                ],
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
                    "badger",
                    "badger",
                    "badger",
                ],
                "f2": [
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
                    7,
                    7,
                    7,
                ],
            }
        )

        # predict funtion
        self.score_foo = partial(FooScorerRegrSeries().score_batch, fast_approx=True)

    def test_bool(self):
        """PD on boolean values: if bins are not specified, then it's the only
        situation to be handled: mins/maxs are calculated from boolean values
        (we always get 2 bins).

        """
        # GIVEN
        fs = ["F"]

        # WHEN
        pdp = PD("Bool").explain(fs, self.xBool, predict_method=self.score_foo)

        # THEN
        self.assertIsNotNone(pdp)
        logging.debug(f"Bool PDs:\n{pdp}")
        self.assertIsNotNone(pdp.explanations())
        self.assertListEqual(
            [True, False],
            list(pdp.explanations()[fs[0]][PD.LABEL_REGRESSION].columns.values),
        )

    def test_categorical(self):
        # GIVEN
        fs = ["F"]

        # WHEN
        pdp = PD("Bool").explain(fs, self.xCat, predict_method=self.score_foo)

        # THEN
        self.assertIsNotNone(pdp)
        logging.debug(f"Bool PDs:\n{pdp}")
        self.assertIsNotNone(pdp.explanations())
        # bins to be ordered by categorical values frequency
        self.assertListEqual(
            ["cat", "dog", "sheep"],
            list(pdp.explanations()[fs[0]][PD.LABEL_REGRESSION].columns.values),
        )

    def test_more_cats_than_resolution_categorical(self):
        # GIVEN
        fs = ["F"]

        # WHEN
        pdp = PD("Bool").explain(fs, self.xManyCats, predict_method=self.score_foo)

        # THEN
        self.assertIsNotNone(pdp)
        logging.debug(f"Cat PDs:\n{pdp}")
        self.assertIsNotNone(pdp.explanations())
        # bins to be ordered by categorical values frequency
        logging.debug(
            f"{pdp.explanations()[fs[0]][PD.LABEL_REGRESSION].columns.values}"
        )
        self.assertListEqual(
            ["1dog", "cat1", "sheep", "dog", "cat"],
            list(pdp.explanations()[fs[0]][PD.LABEL_REGRESSION].columns.values),
        )

    def test_boolean_unique_bins_builder(self):
        # WHEN
        bins = PD.create_unique_bins(["F"], self.xBool)

        # THEN
        logging.debug(f"Bool bins: {bins}")
        self.assertListEqual([[True, False]], bins)

    def test_categorical_unique_bins_builder(self):
        # WHEN
        bins = PD.create_unique_bins(["F"], self.xCat)

        # THEN
        logging.debug(f"Bool bins: {bins}")
        self.assertListEqual([["cat", "dog", "sheep"]], bins)

    def test_nums_as_cats_unique_bins_builder(self):
        # WHEN
        bins = PD.create_unique_bins(["f1"], self.xManyCats)

        # THEN
        logging.debug(f"Num 2 cat bins: {bins}")
        self.assertListEqual([[1, 2, 3, 4, 5, 6]], bins)

    def test_quantiles_unique_bins_builder(self):
        # WHEN
        bins = PD.create_unique_bins(
            ["f2"], self.xManyCats, features_meta={PD.KEY_QUANTILE_BINS: ["f2"]}
        )

        # THEN
        logging.debug(f"Quantiles bins: {bins}")
        self.assertListEqual(
            [
                [
                    2.1000000000000005,
                    10.0,
                    15.499999999999998,
                    20.0,
                    27.5,
                    30.999999999999996,
                    39.49999999999999,
                    43.00000000000001,
                    50.0,
                    55.0,
                ]
            ],
            bins,
        )

    def test_quantiles_pdp(self):
        # GIVEN
        fs = ["f2"]

        # WHEN
        pdp = PD("Quantiles").explain(
            fs,
            self.xManyCats,
            predict_method=self.score_foo,
            features_meta={PD.KEY_QUANTILE_BINS: fs},
        )

        # THEN
        self.assertIsNotNone(pdp)
        logging.debug(f"Quntile bin PDs:\n{pdp}")
        self.assertIsNotNone(pdp.explanations())
        self.assertListEqual(
            [
                2.1000000000000005,
                10.0,
                15.499999999999998,
                20.0,
                27.5,
                30.999999999999996,
                39.49999999999999,
                43.00000000000001,
                50.0,
                55.0,
            ],
            list(pdp.explanations()[fs[0]][PD.LABEL_REGRESSION].columns.values),
        )

    def test_quantiles_ice(self):
        # GIVEN
        fs = ["f2"]

        # WHEN
        ice = ICE("Quantiles").explain(
            fs,
            self.xManyCats,
            predict_method=self.score_foo,
            mins=[3],
            maxs=[55],
            features_meta={PD.KEY_QUANTILE_BINS: fs},
        )

        # THEN
        self.assertIsNotNone(ice)
        logging.debug(f"Quantiles bin ICEs:\n{ice}")
        self.assertIsNotNone(ice.explanations())
        self.assertListEqual(
            [
                2.1000000000000005,
                10.0,
                15.499999999999998,
                20.0,
                27.5,
                30.999999999999996,
                39.49999999999999,
                43.00000000000001,
                50.0,
                55.0,
            ],
            list(ice.explanations()[fs[0]][PD.LABEL_REGRESSION].columns.values),
        )

    def test_oor_categorical_unseen_hint(self):
        """Out of range PD cannot be calculated for non-numerical features."""

        fs = ["F"]
        pd_cat = PD("Cats").explain(
            fs,
            self.xCat,
            predict_method=self.score_foo,
            out_of_range_resolution=3,
        )
        print(f"UNSEEN:\n{pd_cat}")
        self.assertIsNotNone(pd_cat)
        self.assertIsNotNone(pd_cat.explanations())
        self.assertListEqual(
            [
                "cat",
                "dog",
                "sheep",
                "UNSEEN",
            ],
            list(pd_cat.explanations()[fs[0]][PD.LABEL_REGRESSION].columns.values),
        )
        self.assertListEqual(
            [
                False,
                False,
                False,
                True,
            ],
            list(
                pd_cat.explanations()[fs[0]][PD.LABEL_REGRESSION]
                .loc[PD.COL_OOR, :]
                .values.tolist()
            ),
        )

        fs = ["F"]
        pd_cat_unseen = PD("xCatUnseen").explain(
            fs,
            self.xCatUnseen,
            predict_method=self.score_foo,
            out_of_range_resolution=3,
        )
        print(f"UNSEEN clash:\n{pd_cat_unseen}")
        self.assertIsNotNone(pd_cat_unseen)
        self.assertIsNotNone(pd_cat_unseen.explanations())
        self.assertListEqual(
            [
                "sheep",
                "dog",
                "UNSEEN",
                "UNSEEN_[1]",
            ],
            list(
                pd_cat_unseen.explanations()[fs[0]][PD.LABEL_REGRESSION].columns.values
            ),
        )
        self.assertListEqual(
            [
                False,
                False,
                False,
                True,
            ],
            list(
                pd_cat_unseen.explanations()[fs[0]][PD.LABEL_REGRESSION]
                .loc[PD.COL_OOR, :]
                .values.tolist()
            ),
        )

        fs = ["F"]
        pd_cat_unseen = PD("xCatUnseenOrder").explain(
            fs,
            self.xCatUnseenOrder,
            predict_method=self.score_foo,
            out_of_range_resolution=3,
        )
        print(f"UNSEEN order:\n{pd_cat_unseen}")
        self.assertIsNotNone(pd_cat_unseen)
        self.assertIsNotNone(pd_cat_unseen.explanations())
        self.assertListEqual(
            [
                # cat value frequency order
                "ZEBRA",
                "DOG",
                "ALBATROS",
                "UNSEEN",
            ],
            list(
                pd_cat_unseen.explanations()[fs[0]][PD.LABEL_REGRESSION].columns.values
            ),
        )
        self.assertListEqual(
            [
                False,
                False,
                False,
                True,
            ],
            list(
                pd_cat_unseen.explanations()[fs[0]][PD.LABEL_REGRESSION]
                .loc[PD.COL_OOR, :]
                .values.tolist()
            ),
        )

        fs = ["F"]
        pd_cat_unseen = PD("xCatUnseenOrder").explain(
            fs,
            self.xCatUnseenOrder,
            predict_method=self.score_foo,
            out_of_range_resolution=3,
            bins_sort=True,
        )
        print(f"UNSEEN order sorted:\n{pd_cat_unseen}")
        self.assertIsNotNone(pd_cat_unseen)
        self.assertIsNotNone(pd_cat_unseen.explanations())
        self.assertListEqual(
            [
                # alphabetical order
                "ALBATROS",
                "DOG",
                "UNSEEN",
                "ZEBRA",
            ],
            list(
                pd_cat_unseen.explanations()[fs[0]][PD.LABEL_REGRESSION].columns.values
            ),
        )
        self.assertListEqual(
            [
                False,
                False,
                True,
                False,
            ],
            list(
                pd_cat_unseen.explanations()[fs[0]][PD.LABEL_REGRESSION]
                .loc[PD.COL_OOR, :]
                .values.tolist()
            ),
        )

        fs = ["f1", "F", "f2"]
        pd_cat_unseen = PD("xCatUnseenOrder").explain(
            fs,
            self.xCatUnseenOrder,
            predict_method=self.score_foo,
            out_of_range_resolution=2,
            bins_sort=True,
        )
        print(f"3x UNSEEN order sorted:\n{pd_cat_unseen}")
        self.assertIsNotNone(pd_cat_unseen)
        self.assertIsNotNone(pd_cat_unseen.explanations())
        self.assertListEqual(
            [
                # alphabetical order
                "ALBATROS",
                "DOG",
                "UNSEEN",
                "ZEBRA",
            ],
            list(
                pd_cat_unseen.explanations()[fs[1]][PD.LABEL_REGRESSION].columns.values
            ),
        )
        self.assertListEqual(
            [True, True, False, False, False, False, False, False, True, True],
            list(
                pd_cat_unseen.explanations()[fs[0]][PD.LABEL_REGRESSION]
                .loc[PD.COL_OOR, :]
                .values.tolist()
            ),
        )
        self.assertListEqual(
            [
                False,
                False,
                True,
                False,
            ],
            list(
                pd_cat_unseen.explanations()[fs[1]][PD.LABEL_REGRESSION]
                .loc[PD.COL_OOR, :]
                .values.tolist()
            ),
        )
        self.assertListEqual(
            [
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
            ],
            list(
                pd_cat_unseen.explanations()[fs[2]][PD.LABEL_REGRESSION]
                .loc[PD.COL_OOR, :]
                .values.tolist()
            ),
        )

        fs = ["F"]
        pd_bool = PD("Bool").explain(
            fs,
            self.xBool,
            predict_method=self.score_foo,
            out_of_range_resolution=3,
        )
        print(f"UNSEEN bool:\n{pd_bool}")
        self.assertIsNotNone(pd_bool)
        self.assertIsNotNone(pd_bool.explanations())
        self.assertListEqual(
            [
                True,
                False,
                "UNSEEN",
            ],
            list(pd_bool.explanations()[fs[0]][PD.LABEL_REGRESSION].columns.values),
        )
        self.assertListEqual(
            [
                False,
                False,
                True,
            ],
            list(
                pd_bool.explanations()[fs[0]][PD.LABEL_REGRESSION]
                .loc[PD.COL_OOR, :]
                .values.tolist()
            ),
        )

    def test_meta_makes_num_features_cats(self):
        """Use features metadata to indicate that numbers are actually categorical
        features.

        """

        # GIVEN
        fs = ["f1"]
        x = self.xCatValFreq

        # WHEN
        pdp = PD("Num 2 cat").explain(
            fs,
            x,
            predict_method=self.score_foo,
            features_meta={PD.KEY_CATEGORICAL_FEATURES: ["f1"]},
        )

        # THEN
        self.assertIsNotNone(pdp)
        logging.debug(f"Cat PDs:\n{pdp}")
        self.assertIsNotNone(pdp.explanations())
        # bins to be ordered by categorical values FREQUENCY
        logging.debug(
            f"{pdp.explanations()[fs[0]][PD.LABEL_REGRESSION].columns.values}"
        )
        self.assertListEqual(
            [3, 5, 4, 2, 1, 6],
            list(pdp.explanations()[fs[0]][PD.LABEL_REGRESSION].columns.values),
        )

    def test_meta_makes_num_feats_cats_with_oor(self):
        """Use features metadata to indicate that numbers are actually categorical
        features and get the right OOR values if they are strings/numbers.

        """

        # GIVEN
        fs = ["f1"]
        oor_resolution = 3

        # WHEN
        pdp = PD("Num 2 cat w/ OOR").explain(
            features=fs,
            X=self.xCatValFreq,
            predict_method=self.score_foo,
            features_meta={PD.KEY_CATEGORICAL_FEATURES: ["f1"]},
            out_of_range_resolution=oor_resolution,
        )
        print(f"UNSEEN num 2 cat:\n{pdp}")

        # THEN
        self.assertIsNotNone(pdp)
        logging.debug(f"Cat PDs:\n{pdp}")
        self.assertIsNotNone(pdp.explanations())
        # bins to be ordered by categorical values frequency
        logging.debug(
            f"{pdp.explanations()[fs[0]][PD.LABEL_REGRESSION].columns.values}"
        )
        print(list(pdp.explanations()[fs[0]][PD.LABEL_REGRESSION].columns.values))
        self.assertListEqual(
            # bins of INT feature are INTs
            [3, 5, 4, 2, 1, 6, 0, -1, -2, 7, 8, 9],
            list(pdp.explanations()[fs[0]][PD.LABEL_REGRESSION].columns.values),
        )
        # assert OOR hints
        oor = pdp.explanations()[fs[0]][PD.LABEL_REGRESSION]
        self.assertEqual(6 + 2 * oor_resolution, len(oor.columns))
        logging.debug(f"OOR hints:\n{oor.loc[PD.COL_OOR]}")
        self.assertEqual(4, oor.shape[0])
        self.assertEqual(False, oor.loc[PD.COL_OOR].iloc[0])
        self.assertEqual(False, oor.loc[PD.COL_OOR].iloc[1])
        self.assertEqual(False, oor.loc[PD.COL_OOR].iloc[5])
        self.assertEqual(True, oor.loc[PD.COL_OOR].iloc[6])
        self.assertEqual(True, oor.loc[PD.COL_OOR].iloc[9])
        self.assertEqual(True, oor.loc[PD.COL_OOR].iloc[11])

    def test_cat_feature_unique_1(self):
        """Test PD/ICE calculation of categorical feature with cardinality 1."""
        # https://github.com/h2oai/h2oai/issues/26168

        # GIVEN
        dataset = pd.DataFrame(
            {
                "f1": [1, 2, 3, 4, 5, 6],
                # unique() == 1 (one value) + missing values
                "F": ["cat", None, "cat", None, "cat", None],
                "f2": [50, 40, 30, 20, 10, 0],
            }
        )
        fs = ["F"]

        # WHEN
        pdp = PD("unique()==1").explain(
            features=fs, X=dataset, predict_method=self.score_foo
        )

        # THEN
        self.assertIsNotNone(pdp)
        logging.debug(f"unique()==1 PDs:\n{pdp}")
        self.assertIsNotNone(pdp.explanations())
        # bins to be ordered by categorical values frequency
        pd_bins = list(pdp.explanations()[fs[0]][PD.LABEL_REGRESSION].columns.values)
        self.assertEqual(pd_bins[0], "cat")
        self.assertTrue(math.isnan(pd_bins[1]))

    def test_pd_merge(self):
        # GIVEN
        dataset = pd.DataFrame(
            {
                "f1": [1, 2, 3, 4, 5, 6],
                "F": ["cat", None, "cat", None, "cat", None],
                "f2": [50, 40, 30, 20, 10, 0],
            }
        )
        fs_f1 = ["f1"]
        pdp_f1 = PD("F1").explain(
            features=fs_f1, X=dataset, predict_method=self.score_foo
        )
        fs_f2 = ["f2"]
        pdp_f2 = PD("F2").explain(
            features=fs_f2, X=dataset, predict_method=self.score_foo
        )

        # WHEN
        pdp_f1.merge(pdp_f2)

        # THEN
        self.assertIsNotNone(pdp_f1)
        logging.debug(f"Merged PDs:\n{pdp_f1}")
        self.assertIsNotNone(pdp_f1.explanations())
        assert pdp_f1.features == ["f1", "f2"]
