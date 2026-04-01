# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
"""Test H2O-3 memory configuration propagation and cleanup."""

import pytest

from h2o_sonar.methods.utils import h2o_utils
from tests.base_h2o_test import BaseH2OTest


try:
    import h2o

    HAS_H2O = True
except ImportError:
    HAS_H2O = False


@pytest.mark.skipif(not HAS_H2O, reason="H2O-3 not installed")
class TestH2OMemoryConfig(BaseH2OTest):
    """Verify H2O-3 memory configuration and cleanup behavior.

    Note: This test class validates the memory management fixes implemented
    to prevent OOM issues when running multiple surrogate tests with H2O-3.
    """

    def test_cleanup_removes_frames_and_models(self):
        """Verify cleanup removes frames and models to free memory."""
        # GIVEN - create frames and models
        import pandas as pd
        from h2o.estimators import H2ORandomForestEstimator

        df = pd.DataFrame(
            {"x1": [1, 2, 3, 4, 5], "x2": [2, 3, 4, 5, 6], "y": [0, 1, 0, 1, 0]}
        )
        h2o_frame = h2o.H2OFrame(df)

        # train a simple model
        rf = H2ORandomForestEstimator(ntrees=2, max_depth=2, seed=1234)
        rf.train(x=["x1", "x2"], y="y", training_frame=h2o_frame)

        initial_frames = len(h2o.ls())
        initial_models_resp = h2o.api("GET /3/Models")
        initial_models = len(initial_models_resp.get("models", []))

        print(f"Before cleanup: {initial_frames} frames, {initial_models} models")
        assert initial_frames > 0, "Should have frames before cleanup"
        assert initial_models > 0, "Should have models before cleanup"

        # WHEN
        h2o_utils.clean_up_h2o3()

        # THEN
        final_frames = len(h2o.ls())
        final_models_resp = h2o.api("GET /3/Models")
        final_models = len(final_models_resp.get("models", []))

        print(f"After cleanup: {final_frames} frames, {final_models} models")

        assert final_frames == 0, f"Expected 0 frames after cleanup, got {final_frames}"
        assert final_models == 0, f"Expected 0 models after cleanup, got {final_models}"

        print("DONE - Cleanup verification successful")
