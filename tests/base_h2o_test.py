# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
"""Base test class for H2O-3 dependent tests with automatic memory cleanup."""

from unittest import TestCase

from h2o_sonar.methods.utils import h2o_utils


class BaseH2OTest(TestCase):
    """Base class for tests that use H2O-3/HMLI.

    Provides automatic cleanup after each test method and after all tests
    in the class complete to prevent Java heap memory accumulation.

    Usage:
        class TestMyH2OFeature(BaseH2OTest):
            def test_something(self):
                # H2O-3 cleanup happens automatically after this test
                pass
    """

    def tearDown(self):
        """Clean up H2O-3 data after each test method to prevent memory accumulation.

        This is called after EVERY test method in the class, ensuring that
        H2O-3's Java heap is cleared between tests.
        """
        super().tearDown()
        h2o_utils.clean_up_h2o3()

    @classmethod
    def tearDownClass(cls):
        """Final cleanup after all tests in the class complete.

        This ensures any remaining H2O-3 data is cleared after the entire
        test class finishes.
        """
        super().tearDownClass()
        h2o_utils.clean_up_h2o3()
