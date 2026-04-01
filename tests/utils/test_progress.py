# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import functools
from unittest import TestCase

import pandas
import pytest

from h2o_sonar.methods import _h_statistic
from h2o_sonar.methods import _pd
from h2o_sonar.utils import progress as progress_utils
from tests.methods import ice_pd_test_commons


#
# PROGRESS REPORTING: methods and explainers
#


class TestProgress(TestCase):
    def setUp(self):
        # predict function
        self.score_foo = functools.partial(
            ice_pd_test_commons.FooScorerRegrSeries().score_batch,
            fast_approx=True,
        )

    def assertProgressNonDecreasing(self, progress_log: list):
        if progress_log:
            last_p: float = 0.0
            for i, p in enumerate(progress_log):
                self.assertGreaterEqual(
                    p, last_p, f"Progress is decreasing at index {i}"
                )
                last_p = p

    def test_progress_callback_default(self):
        # GIVEN
        c = progress_utils.ProgressCallbackContext()

        # WHEN / THEN
        assert c.progress == 0.0
        assert c.relative_progress == 0.0

        for p in [0.1, 0.5, 1.0]:
            c.set_progress(p)
            self.assertEqual(c.progress, p)

    def test_progress_callback_range(self):
        # GIVEN
        c = progress_utils.ProgressCallbackContext(
            min_progress=0.25,
            max_progress=0.75,
        )

        # WHEN / THEN
        expected_relative_progress = [0.25, 0.5, 0.75]
        for e, p in enumerate([0.0, 0.5, 1.0]):
            c.set_progress(p)
            self.assertEqual(p, c.progress)
            self.assertEqual(expected_relative_progress[e], c.relative_progress)

    def test_progress_callback_steps(self):
        # GIVEN
        c = progress_utils.ProgressCallbackContext(total_steps=100)
        progress_log = []

        # WHEN
        for p in [10, 50, 100]:
            c.set_steps(p)
            self.assertEqual(c.progress, float(p) / 100.0)
            progress_log.append(c.progress)

        # THEN
        self.assertProgressNonDecreasing(progress_log)

    def test_progress_callback_range_steps(self):
        # GIVEN
        c = progress_utils.ProgressCallbackContext(
            min_progress=0.25,
            max_progress=0.75,
            total_steps=100,
        )
        progress_log = []

        # WHEN
        expected_progress = [0.0, 0.5, 1.0]
        expected_relative_progress = [0.25, 0.5, 0.75]
        for e, p in enumerate([0, 50, 100]):
            c.set_steps(p)
            self.assertEqual(expected_progress[e], c.progress)
            self.assertEqual(expected_relative_progress[e], c.relative_progress)
            progress_log.append(c.relative_progress)

        # THEN
        self.assertProgressNonDecreasing(progress_log)

    def test_sub_callback_features_bins_steps(self):
        # GIVEN
        features = 10
        c_level_features = progress_utils.ProgressCallbackContext(total_steps=features)

        # WHEN
        progress_log = []

        # report 3/10 steps done
        c_level_features.set_steps(3)
        progress_log.append(c_level_features.progress)
        # get sub/child callback to report progress within scope of one step
        feature_bins = 20
        c_level_bins = c_level_features.get_sub_callback_for_steps(
            total_steps=feature_bins,
        )
        # report progress between steps 3 and 4
        for b in range(feature_bins):
            c_level_bins.set_steps(b)
            print(
                f"Bin {b} progress:\n"
                f"    {c_level_features.progress} ... all features progress\n"
                f"    {c_level_bins.progress} ... feature progress"
            )
            progress_log.append(c_level_features.progress)

        # report 4/10 steps done
        c_level_features.set_steps(4)
        progress_log.append(c_level_features.progress)
        # get sub/child callback to report progress within scope of one step
        feature_bins = 3
        c_level_bins = c_level_features.get_sub_callback_for_steps(
            total_steps=feature_bins,
        )
        # report progress between steps 3 and 4
        for b in range(feature_bins):
            c_level_bins.set_steps(b)
            print(
                f"Bin {b} progress:\n"
                f"    {c_level_features.progress} ... all features progress\n"
                f"    {c_level_bins.progress} ... feature progress"
            )
            progress_log.append(c_level_features.progress)

        # report 5/10 steps done
        c_level_features.set_steps(5)
        progress_log.append(c_level_features.progress)

        # THEN
        print(f"Progress log:\n{progress_log}")
        self.assertProgressNonDecreasing(progress_log)
        self.assertEqual(
            progress_log,
            [
                0.3,
                0.3,
                0.305,
                0.31,
                0.315,
                0.32,
                0.325,
                0.33,
                0.335,
                0.34,
                0.34500000000000003,
                0.35,
                0.355,
                0.36,
                0.365,
                0.37,
                0.375,
                0.38,
                0.385,
                0.39,
                0.395,
                0.4,
                0.4,
                0.43333333333333335,
                0.4666666666666667,
                0.5,
            ],
        )

    def test_sub_callback_features_bins_progress_pct(self):
        # GIVEN
        features = 10
        features_callback = progress_utils.ProgressCallbackContext(total_steps=features)

        # WHEN
        progress_log = []

        # report 3/10 steps done
        features_callback.set_steps(3)
        progress_log.append(features_callback.progress_percent)
        # get sub/child callback to report progress within scope of one step
        feature_bins = 20
        bins_sub_callback = features_callback.get_sub_callback_for_progress(
            min_progress=features_callback.progress,
            max_progress=features_callback.progress
            + features_callback.get_range_for_step(),
        )
        # report progress between steps 3 and 4
        for b in range(feature_bins):
            bins_sub_callback.set_progress((float(b) + 1.0) / float(feature_bins))
            print(
                f"Bin {b} progress:\n"
                f"    {features_callback.progress_percent}"
                f" ... all features progress\n"
                f"    {bins_sub_callback.progress_percent}"
                f" ... feature progress"
            )
            progress_log.append(features_callback.progress_percent)

        # report 4/10 steps done
        features_callback.set_steps(4)
        progress_log.append(features_callback.progress_percent)
        # get sub/child callback to report progress within scope of one step
        feature_bins = 3
        bins_sub_callback = features_callback.get_sub_callback_for_progress(
            min_progress=features_callback.progress,
            max_progress=features_callback.progress
            + features_callback.get_range_for_step(),
        )
        # report progress between steps 3 and 4
        for b in range(feature_bins):
            bins_sub_callback.set_progress((float(b) + 1.0) / float(feature_bins))
            print(
                f"Bin {b} progress:\n"
                f"    {features_callback.progress_percent}"
                f" ... all features progress\n"
                f"    {bins_sub_callback.progress_percent}"
                f" ... feature progress"
            )
            progress_log.append(features_callback.progress_percent)

        # report 5/10 steps done
        features_callback.set_steps(5)
        progress_log.append(features_callback.progress_percent)

        # THEN
        print(f"Progress log:\n{progress_log}")
        self.assertProgressNonDecreasing(progress_log)
        self.assertEqual(
            progress_log,
            [
                30,
                30,
                31,
                31,
                32,
                32,
                33,
                33,
                34,
                34,
                35,
                35,
                36,
                36,
                37,
                37,
                38,
                38,
                39,
                39,
                40,
                40,
                43,
                46,
                50,
                50,
            ],
        )

    def test_process_callback_bridge_twice(self):
        # GIVEN
        features = ["FEATURE-A", "FEATURE-B", "FEATURE-C"]

        class MyCustomCallback(progress_utils.AbstractProgressCallbackContext):
            def __init__(self):
                self.debug_progress_log: list = []
                progress_utils.AbstractProgressCallbackContext.__init__(self)

            def set_progress(
                self, progress: float, message: str | None = None
            ) -> float:
                print(f"Bridge 2x progress: {progress}")
                self.debug_progress_log.append(progress)
                return progress

        custom_callback = MyCustomCallback()

        splitter_callback = progress_utils.ProgressCallbackContext(
            total_steps=len(features), parent_callback=custom_callback
        )

        # WHEN
        custom_callback.set_progress(progress=0.0, message="Start")
        for _ in features:
            bridge_callback = progress_utils.ProgressCallbackStackingBridge(
                splitter_callback
            )

            # update progress
            bridge_callback.set_progress(0.1)
            bridge_callback.set_progress(0.5)
            bridge_callback.set_progress(1.0)

        # THEN
        print(f"Bridge 2x:\n{custom_callback.debug_progress_log}")
        self.assertEqual(custom_callback.debug_progress_log[0], 0.0)
        self.assertEqual(custom_callback.debug_progress_log[-1], 1.0)
        self.assertProgressNonDecreasing(custom_callback.debug_progress_log)

    #
    # ICE
    #

    def _test_progress_callback_ice(self, opt_1_frame: bool):
        # GIVEN
        dataset = pandas.DataFrame(
            {
                "f1": [1, 2, 3, 4, 5, 6],
                "f2": ["cat", "dog", "cat", "snake", "cat", "dog"],
                "f3": [50, 40, 30, 20, 10, 0],
                "f4": [55, 44, 33, 22, 11, 0],
            }
        )
        fs = ["f1", "f2", "f3", "f4"]

        # WHEN
        class MyCustomCallback(progress_utils.AbstractProgressCallbackContext):
            def __init__(self):
                self.debug_progress_log: list = []

            def set_progress(
                self, progress: float, message: str | None = None
            ) -> float:
                print(f"ICE progress: {progress}")
                self.debug_progress_log.append(progress)
                return progress

        custom_progress_callback = MyCustomCallback()

        ice = _pd.ICE("ICE callback")
        ice.opt_1_prediction = opt_1_frame
        ice.explain(
            features=fs,
            X=dataset,
            mins=[1, "cat", 0, 0],
            maxs=[6, "dog", 50, 55],
            predict_method=self.score_foo,
            progress_callback=custom_progress_callback,
        )

        # THEN
        self.assertIsNotNone(ice)
        print(f"ICEs:\n{ice}")
        self.assertIsNotNone(ice.explanations())
        print(f"ICE progress log:\n{custom_progress_callback.debug_progress_log}")
        if opt_1_frame:
            self.assertProgressNonDecreasing(
                custom_progress_callback.debug_progress_log
            )
            self.assertEqual(
                custom_progress_callback.debug_progress_log,
                [
                    0.1111111111111111,
                    0.2222222222222222,
                    0.33333333333333337,
                    0.5555555555555556,
                    0.6666666666666667,
                    0.7777777777777779,
                    0.8888888888888888,
                    1.0,
                ],
            )
        else:
            self.assertProgressNonDecreasing(
                custom_progress_callback.debug_progress_log
            )
            self.assertEqual(
                [
                    0.04166666666666667,
                    0.08333333333333334,
                    0.125,
                    0.16666666666666669,
                    0.20833333333333337,
                    0.25,
                    0.33333333333333337,
                    0.4166666666666667,
                    0.5,
                    0.525,
                    0.55,
                    0.575,
                    0.6,
                    0.625,
                    0.65,
                    0.675,
                    0.7,
                    0.725,
                    0.75,
                    0.775,
                    0.8,
                    0.825,
                    0.85,
                    0.875,
                    0.9,
                    0.925,
                    0.95,
                    0.975,
                    1.0,
                    1.0,
                ],
                custom_progress_callback.debug_progress_log,
            )

    def test_progress_callback_ice_1_frame(self):
        self._test_progress_callback_ice(True)

    def test_progress_callback_ice(self):
        self._test_progress_callback_ice(False)

    #
    # PD
    #

    def _test_progress_callback_pd(self, opt_1_frame: bool):
        # GIVEN
        dataset = pandas.DataFrame(
            {
                "f1": [1, 2, 3, 4, 5, 6],
                "f2": ["cat", "dog", "cat", "snake", "cat", "dog"],
                "f3": [50, 40, 30, 20, 10, 0],
                "f4": [55, 44, 33, 22, 11, 0],
            }
        )
        fs = ["f1", "f2", "f3", "f4"]

        # WHEN
        class MyCustomCallback(progress_utils.AbstractProgressCallbackContext):
            def __init__(self):
                self.debug_progress_log: list = []
                progress_utils.AbstractProgressCallbackContext.__init__(self)

            def set_progress(
                self, progress: float, message: str | None = None
            ) -> float:
                print(f"PD progress: {progress}")
                self.debug_progress_log.append(progress)
                return progress

        custom_progress_callback = MyCustomCallback()

        pdp = _pd.PD("PD callback")
        pdp.opt_1_prediction = opt_1_frame
        pdp.explain(
            features=fs,
            X=dataset,
            predict_method=self.score_foo,
            progress_callback=custom_progress_callback,
        )

        # THEN
        self.assertIsNotNone(pdp)
        print(f"PDs:\n{pdp}")
        self.assertIsNotNone(pdp.explanations())
        print(f"PD progress log:\n{custom_progress_callback.debug_progress_log}")
        if opt_1_frame:
            self.assertProgressNonDecreasing(
                custom_progress_callback.debug_progress_log
            )
            self.assertEqual(
                custom_progress_callback.debug_progress_log,
                [
                    0.16670000000000001,
                    0.25,
                    0.4167,
                    0.5,
                    0.6667000000000001,
                    0.75,
                    0.9167000000000001,
                    1.0,
                ],
            )
        else:
            self.assertProgressNonDecreasing(
                custom_progress_callback.debug_progress_log
            )
            self.assertEqual(
                [
                    0.04175000000000001,
                    0.08340000000000002,
                    0.12505,
                    0.16670000000000001,
                    0.20835000000000004,
                    0.25,
                    0.33340000000000003,
                    0.4167,
                    0.5,
                    0.52509,
                    0.55008,
                    0.57507,
                    0.60006,
                    0.62505,
                    0.65004,
                    0.67503,
                    0.70002,
                    0.72501,
                    0.75,
                    0.77509,
                    0.80008,
                    0.82507,
                    0.85006,
                    0.87505,
                    0.90004,
                    0.92503,
                    0.95002,
                    0.97501,
                    1.0,
                ],
                custom_progress_callback.debug_progress_log,
            )

    def test_progress_callback_pd_1_frame(self):
        self._test_progress_callback_pd(True)

    def test_progress_callback_pd(self):
        self._test_progress_callback_pd(False)

    #
    # H-statistic
    #

    @staticmethod
    def foo_predict_method_regr(x):
        y_hat = abs(x["a"] + x["F"] + x["G"] * x["H"] - x["b"]) ** (1 / 2)
        # logging.debug("Y_hat:\n{}".format(y_hat))
        return y_hat

    def test_progress_callback_h_statistic(self):
        # GIVEN
        frame = pandas.DataFrame(
            {
                "a": [9.0, 8.0, 7.0, 6.0, 5.0],
                "F": [3.0, 4.0, 5.0, 4.0, 3.0],
                "G": [10.0, 20.0, 30.0, 40.0, 50.0],
                "H": [0.1, 0.2, 0.3, 0.4, 0.5],
                "b": [5.0, 6.0, 7.0, 8.0, 9.0],
            }
        )
        features = list("FGH")
        bins = [[1, 3], [2, 8], [13, 16]]

        # WHEN
        class MyCustomCallback(progress_utils.AbstractProgressCallbackContext):
            def __init__(self):
                self.debug_progress_log: list = []
                progress_utils.AbstractProgressCallbackContext.__init__(self)

            def set_progress(
                self, progress: float, message: str | None = None
            ) -> float:
                print(f"H-statistic progress: {progress}")
                self.debug_progress_log.append(progress)
                return progress

        custom_progress_callback = MyCustomCallback()

        result = _h_statistic.HStatistic("H-statistic progress").explain(
            features,
            frame,
            predict_method=TestProgress.foo_predict_method_regr,
            bins=bins,
            progress_callback=custom_progress_callback,
        )

        # THEN
        self.assertIsNotNone(result)
        print(f"H-statistic:{result}")
        self.assertIsNotNone(result.explanations())
        self.assertProgressNonDecreasing(custom_progress_callback.debug_progress_log)
        print(
            f"H-statistic progress log:\n{custom_progress_callback.debug_progress_log}"
        )
        self.assertEqual(
            custom_progress_callback.debug_progress_log,
            [
                0.041795825833333335,
                0.08344166166666667,
                0.1250874975,
                0.16673333333333332,
                0.20842915916666663,
                0.250074995,
                0.29172083083333333,
                0.33336666666666664,
                0.37506249249999996,
                0.4167083283333333,
                0.4583541641666667,
                0.5,
                0.6112055522222222,
                0.6667333333333333,
                0.7778388855555556,
                0.8333666666666666,
                0.9444722188888889,
                1.0,
            ],
        )


#
# PROGRESS REPORTING: evaluation
#


def test_evaluate_default_progress(custom_progress_callback=None):
    #
    # GIVEN
    #
    evaluators_count = 10

    class InterpretationMock:
        def __init__(self):
            self.progress_callback = None
            self.progress = 0.0

    interpretation_mock = InterpretationMock()

    class JobMock:
        def __init__(self):
            self.progress_callback = None
            self.progress = 0.0

    # container callback (could be connected to a CUSTOM callback)
    container_callback = progress_utils.LoggingProgressCallbackContext(
        parent_callback=custom_progress_callback, do_update=[interpretation_mock]
    )
    # data structures:
    # - interpretation.progress_callback / evaluation.progress_callback
    # - interpretation.progress / evaluation.progress

    print()

    #
    # WHEN
    #

    # container: initialization
    container_callback.set_progress(0.0, "Evaluation started")
    container_callback.set_progress(0.01, "Evaluation preparation")
    # container: explainers compatibility check
    progress_slot_min = container_callback.progress
    progress_slot_size = (0.09 - progress_slot_min) / evaluators_count
    for i in range(evaluators_count):
        e_progress_min = progress_slot_min + progress_slot_size * i
        e_progress_max = e_progress_min + progress_slot_size
        container_callback.set_progress(
            e_progress_min,
            f"Evaluator {i} compatibility check started",
        )
        container_callback.set_progress(
            e_progress_max,
            f"Evaluator {i} compatibility check finished",
        )
    container_callback.set_progress(0.1, "Evaluation prepared")

    # container: 10x evaluators to perform in range [0.1, 0.9]
    progress_slot_min = container_callback.progress
    progress_slot_size = 0.8 / evaluators_count
    for i in range(evaluators_count):
        e_progress_min = progress_slot_min + progress_slot_size * i
        e_progress_max = e_progress_min + progress_slot_size

        # evaluator: setup
        container_callback.set_progress(
            e_progress_min,
            f"Evaluator {i} setup done",
        )

        # evaluator: RUN
        evaluator_job = JobMock()
        evaluator_callback = container_callback.get_sub_callback_for_progress(
            min_progress=e_progress_min,
            max_progress=e_progress_max,
            do_update=[evaluator_job],
        )

        # EVALUATOR: reports progress - evaluator.set_progress(progress, message)
        # data structures:
        # - explainer_job.progress
        # - explainer.progress_callback
        evaluator_callback.set_progress(
            0.0,
            f"@Evaluator {i} start "
            f"[{evaluator_callback._min_progress}, {evaluator_callback._max_progress}]",
        )
        evaluator_callback.set_progress(
            0.1,
            f"@Evaluator {i} initialization "
            f"[{evaluator_callback._min_progress}, {evaluator_callback._max_progress}]",
        )
        assert evaluator_job.progress == 0.1
        evaluator_callback.set_progress(
            0.5,
            f"@Evaluator {i} evaluation in progress "
            f"[{evaluator_callback._min_progress}, {evaluator_callback._max_progress}]",
        )
        assert evaluator_job.progress == 0.5
        evaluator_callback.set_progress(
            1.0,
            f"@Evaluator {i} evaluation DONE "
            f"[{evaluator_callback._min_progress}, {evaluator_callback._max_progress}]",
        )
        assert evaluator_job.progress == 1.0
        assert evaluator_callback.progress == 1.0

        # container: ensures slot progress
        container_callback.set_progress(
            e_progress_max, f"Evaluation progress: evaluator {i} finished"
        )

    # container: post-processing
    container_callback.set_progress(0.8, "Evaluation results processing")
    container_callback.set_progress(0.9, "Evaluation results saved")

    # container: finish / DONE
    container_callback.set_progress(1.0, "Evaluation finished")

    #
    # THEN

    #
    assert container_callback.progress == 1.0
    assert interpretation_mock.progress == 1.0


def test_evaluate_custom_progress(custom_progress_callback=None):
    #
    # GIVEN: custom user-defined progress callback
    #
    class MyCustomListLogCallback(progress_utils.AbstractProgressCallbackContext):
        def __init__(self):
            self.debug_progress_log: list = []
            progress_utils.AbstractProgressCallbackContext.__init__(self)

        def set_progress(self, progress: float, message: str | None = None) -> float:
            print(f"Bridge 2x progress: {progress}")
            self.debug_progress_log.append(f"{progress * 100.0}% {message}")
            return progress

    custom_callback = MyCustomListLogCallback()

    #
    # WHEN
    #
    test_evaluate_default_progress(custom_progress_callback=custom_callback)

    #
    # THEN
    #
    print(f"\n\nCustom progress log:\n{custom_callback.debug_progress_log}")
    for i, p in enumerate(custom_callback.debug_progress_log):
        print(f"Progress {i}: {p}")


def test_progress_bar():
    import time

    print("Test case generation progress:")
    total = 10
    for i in range(1, total + 1):
        time.sleep(1)

        progress = i / total * 100
        msg = f"Test case {i}/{total}"
        print(f"  [{progress:<3}%] {'#' * i:<10} | {msg}", end="\r")


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
