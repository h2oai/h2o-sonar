# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import statistics
import unittest
from functools import partial

import pandas

from h2o_sonar import loggers as logging
from h2o_sonar.methods import _ice as e_ice
from tests.methods import ice_pd_test_commons


class TestIntBinning(unittest.TestCase):
    """Test integer features binning for PD, ICE and H-statistic."""

    def setUp(self):
        logging.setLevel(logging.DEBUG)

        # prediction method lambda
        self.score_sum = partial(
            ice_pd_test_commons.FooScorerSumRegrFrame().score_batch,
            fast_approx=True,
        )

    #
    # int bins
    #

    def test_int_bins_builder(self):
        test_data = [
            # feature_values + resolution > expected bins
            ([1], 3, [[1]]),
            ([1, 2], 3, [[1, 2]]),
            ([1, 2, 3], 10, [[1, 2, 3]]),
            ([10, 20, 30], 10, [[10, 14, 16, 18, 20, 22, 24, 26, 28, 30]]),
            ([10, 23], 10, [[10, 12, 13, 15, 16, 17, 19, 20, 21, 23]]),
            (
                [10, 20, 1000],
                10,
                [[10, 208, 307, 406, 505, 604, 703, 802, 901, 1000]],
            ),
        ]

        for td in test_data:
            # GIVEN
            feature_values = td[0]
            min_ = min(feature_values)
            max_ = max(feature_values)
            resolution = td[1]
            print(f"Input {feature_values} -> min/max/std: {min_}/{max_}")
            expected_bins = td[2]
            bins = []

            # WHEN
            e_ice.ICE("bins provider").create_numerical_bins_int(
                grid_resolution=resolution,
                min_=min_,
                max_=max_,
                bins=bins,
            )

            # THEN
            print(f"Bins: {bins}")
            self.assertEqual(expected_bins, bins)

    def test_float_bins_builder(self):
        test_data = [
            # feature_values + resolution > expected bins
            ([1.5], 3, [[1.5]]),
            ([1.5, 2.5], 3, [[1.5, 2.0, 2.5]]),
        ]

        for td in test_data:
            # GIVEN
            feature_values = td[0]
            mins = [min(feature_values)]
            maxs = [max(feature_values)]
            resolution = td[1]
            print(f"Input {feature_values} -> mins/maxs/std: {mins}/{maxs}")
            expected_bins = td[2]
            bins = []

            # WHEN
            ice = e_ice.ICE("bins provider")
            ice._g_resolution = resolution
            ice.create_numerical_bins(
                feature_dtype="f",
                bins=bins,
                idx=0,
                mins=mins,
                maxs=maxs,
            )

            # THEN
            print(f"Bins: {bins}")
            self.assertEqual(expected_bins, bins)

    #
    # int OOR bins
    #

    def test_unsigned_int_oor_bins_builder(self):
        # GIVEN
        feature_values = [10, 20, 30]
        min_ = min(feature_values)
        max_ = max(feature_values)
        std_ = statistics.stdev(feature_values)
        print(f"Input min/max/std: {min_}/{max_}/{std_}")

        # WHEN
        bins = e_ice.ICE.create_oor_bins_int(
            # force unsigned int
            feature_dtype="u",
            min_=10,
            max_=30,
            std_dev=statistics.stdev(feature_values),
            out_of_range_resolution=3,
        )

        # THEN
        print(f"Bins: {bins}")
        self.assertEqual([9, 8, 7, 40, 50, 60], bins)

    def test_int_oor_bins_builder(self):
        # GIVEN
        feature_values = [10, -20, 30]
        min_ = min(feature_values)
        max_ = max(feature_values)
        std_ = statistics.stdev(feature_values)
        print(f"Input min/max/std: {min_}/{max_}/{std_}")

        # WHEN
        bins = e_ice.ICE.create_oor_bins_int(
            feature_dtype="i",
            min_=10,
            max_=30,
            std_dev=statistics.stdev(feature_values),
            out_of_range_resolution=3,
        )

        # THEN
        print(f"Bins: {bins}")
        self.assertEqual([-15, -40, -65, 55, 80, 105], bins)

    #
    # ICE int bins unit
    #

    def test_ice_build_numerical_bins(self):
        test_data = [
            # feature dtype / index / maxs / mins > expected bins
            ("i", 0, [30], [10], [[10, 14, 16, 18, 20, 22, 24, 26, 28, 30]]),
            ("u", 0, [10], [0], [[0, 2, 3, 4, 5, 6, 7, 8, 9, 10]]),
            ("u", 0, [5], [0], [[0, 1, 2, 3, 4, 5]]),
            ("i", 0, [5], [0], [[0, 1, 2, 3, 4, 5]]),
            (
                "u",
                0,
                [110],
                [100],
                [[100, 102, 103, 104, 105, 106, 107, 108, 109, 110]],
            ),
            (
                "f",
                0,
                [10.5],
                [0.5],
                [
                    [
                        0.5,
                        1.6111111111111112,
                        2.7222222222222223,
                        3.8333333333333335,
                        4.944444444444445,
                        6.055555555555555,
                        7.166666666666667,
                        8.277777777777779,
                        9.38888888888889,
                        10.5,
                    ]
                ],
            ),
            ("u", 0, [11], [10], [[10, 11]]),
            (
                "f",
                0,
                [11],
                [10],
                [
                    [
                        10.0,
                        10.11111111111111,
                        10.222222222222221,
                        10.333333333333332,
                        10.444444444444443,
                        10.555555555555554,
                        10.666666666666664,
                        10.777777777777775,
                        10.888888888888886,
                        11.0,
                    ]
                ],
            ),
        ]

        for td in test_data:
            # GIVEN
            ice = e_ice.ICE("bins_provider")
            ice._g_resolution = e_ice.ICE.DEFAULT_GRID_RESOLUTION
            bins_ = []

            # WHEN
            feature_dtype = td[0]
            idx = td[1]
            maxs = td[2]
            mins = td[3]
            ice.create_numerical_bins(
                feature_dtype=feature_dtype,
                bins=bins_,
                idx=idx,
                maxs=maxs,
                mins=mins,
            )

            # THEN
            print(f"Bins for maxs={maxs} and mins={mins}:\n{bins_}")
            self.assertEqual(td[4], bins_)

    #
    # ICE int bins integration
    #

    def test_ice_int_oor(self):
        """Test that integer (not float) OOR bins are created for integer features."""
        # GIVEN
        feature = "feature"
        feature_values = [10, 20, 30]
        x = pandas.DataFrame(
            {feature: feature_values, "f2": [0.5, 2, 4], "F": [1, 3, 5]}
        )

        # WHEN
        ice = e_ice.ICE("high enough int")
        explanation = ice.explain(
            features=[feature],
            X=x,
            predict_method=self.score_sum,
            mins=[10],
            maxs=[30],
            stds=[statistics.stdev(feature_values)],
            out_of_range_resolution=3,
        ).explanations()

        # THEN
        print(explanation)
        oor_bins = list(explanation[feature][e_ice.ICE.LABEL_REGRESSION].columns)
        print(f"OOR bins for feature values {feature_values}:\n{oor_bins}")
        self.assertEqual(
            [10, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0, -10, -20, 40, 50, 60],
            oor_bins,
        )

    def test_ice_int_oor_0(self):
        """Test that integer (not float) OOR bins are created for integer features
        where minimum is too low.

        """
        test_data = [
            # high enough integer
            (
                {"f1": [10, 20, 30], "f2": [0.5, 2, 4], "F": [1, 3, 5]},
                [
                    10,
                    14,
                    16,
                    18,
                    20,
                    22,
                    24,
                    26,
                    28,
                    30,
                    0,
                    -10,
                    -20,
                    40,
                    50,
                    60,
                ],
            ),
            # bins which do NOT fit in positive below
            (
                {"f1": [0, 1, 2], "f2": [0.5, 2, 4], "F": [1, 3, 5]},
                [10, 14, 16, 18, 20, 22, 24, 26, 28, 30, 9, 8, 7, 31, 32, 33],
            ),
        ]

        for td in test_data:
            # GIVEN
            feature = "f1"
            features = [feature]
            feature_values = td[0][feature]
            # datatable is NOT able to convert uint16 / uint32 Pandas frames
            # x = pandas.DataFrame(td[0]).astype(numpy.uint16)
            x = pandas.DataFrame(td[0])

            # WHEN
            ice = e_ice.ICE("0 min int")
            explanation = ice.explain(
                features=features,
                X=x,
                predict_method=self.score_sum,
                mins=[10],
                maxs=[30],
                stds=[statistics.stdev(feature_values)],
                out_of_range_resolution=3,
            ).explanations()

            # THEN
            print(explanation)
            oor_bins = list(explanation[feature][e_ice.ICE.LABEL_REGRESSION].columns)
            print(f"OOR bins for feature values {feature_values}:\n{oor_bins}")
            self.assertEqual(
                td[1],
                oor_bins,
            )

    @unittest.skip(
        "Datatable does NOT support unsigned int (neither 32b nor 64b) therefore"
        "ICE test of unsigned int OOR bins cannot be implemented"
    )
    def SKIP_test_ice_unsigned_int_oor(self):
        """Test that unsigned integer (not float) OOR bins are created for integer
        features.

        """
        # datatable is NOT able to convert uint16 / uint32 Pandas frames
        # x = pandas.DataFrame(td[0]).astype(numpy.uint16)

        raise NotImplementedError
