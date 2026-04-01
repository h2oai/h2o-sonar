# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import pytest

from h2o_sonar.methods.core import _data
from h2o_sonar.methods.utils import h2o_utils
from tests.base_h2o_test import BaseH2OTest
from tests.test_utils import find_locally


try:
    import h2o

    HAS_H2O = True
except ImportError:
    HAS_H2O = False


def assert_bad_upload():
    pass


class TestH2OUtils(BaseH2OTest):
    """Test H2O utilities.

    Inherits from BaseH2OTest which provides automatic H2O-3 cleanup
    after each test method and after the entire test class completes.
    """

    @pytest.mark.skipif(
        not HAS_H2O,
        reason="H2O-3 Python package is not installed",
    )
    def test_fail_on_bad_upload(self):
        """Negative test of bad data upload. Do NOT remove function in ill formed
        data as it's workaround of H2O-3 type check BUG reproducible on the cythonized
        code.

        """
        from h2o.exceptions import H2OTypeError

        with self.assertRaises(H2OTypeError):
            h2o_utils.upload_data([1, 2, 3])

    def test_local_upload(self):
        frame, delete = h2o_utils.to_h2oframe(
            find_locally("data/predictive/test_upload.csv")
        )
        self.assertDictEqual(
            frame.types,
            {
                "sepal_length": "real",
                "sepal_width": "real",
                "petal_length": "real",
                "petal_width": "real",
                "species": "enum",
            },
        )
        self.assertTupleEqual(frame.shape, (3, 5))
        self.assertTrue(delete)

    def test_local_upload_custom_sep(self):
        frame, delete = h2o_utils.to_h2oframe(
            _data.PersistedData(
                find_locally("data/predictive/test_upload_sep.csv"),
                upload_config={"sep": ";"},
            )
        )

        self.assertDictEqual(
            frame.types,
            {
                "sepal_length": "int",
                "sepal_width": "real",
                "petal_length": "int",
                "petal_width": "int",
                "species": "enum",
            },
        )
        self.assertTupleEqual(frame.shape, (3, 5))
        self.assertTrue(delete)

    def test_local_upload_custom_types(self):
        frame, delete = h2o_utils.to_h2oframe(
            _data.PersistedData(
                find_locally("data/predictive/test_upload_sep.csv"),
                upload_config={
                    "sep": ";",
                    "col_types": {"sepal_length": "enum"},
                },
            )
        )
        self.assertDictEqual(
            frame.types,
            {
                "sepal_length": "enum",
                "sepal_width": "real",
                "petal_length": "int",
                "petal_width": "int",
                "species": "enum",
            },
        )
        self.assertTupleEqual(frame.shape, (3, 5))
        self.assertTrue(delete)

    def test_fail_on_h2oframe_labels(self):
        with self.assertRaises(ValueError):
            frame = h2o_utils.to_h2oframe(
                find_locally("data/predictive/test_upload.csv")
            )
            h2o_utils.to_h2oframe(frame, [1, 2, 3])

    @pytest.mark.skipif(
        not HAS_H2O,
        reason="H2O-3 Python package is not installed",
    )
    def test_upload_h2oframe(self):
        original = h2o.import_file(
            path=find_locally("data/predictive/test_upload.csv"), header=1
        )
        uploaded = h2o_utils.upload_data(original)
        self.assertEqual(original, uploaded)

    def test_return_original_h2oframe(self):
        frame, delete = h2o_utils.to_h2oframe(
            find_locally("data/predictive/test_upload.csv")
        )
        frame2, delete2 = h2o_utils.to_h2oframe(frame)
        self.assertEqual(frame, frame2)


@pytest.mark.parametrize(
    "port,expected_port",
    [
        # (44321, 44321),  # this would be flaky test
        (0, 0)
    ],
)
def test_find_free_port(port: int, expected_port: int):
    #
    # WHEN
    #
    port = h2o_utils.h2o_find_free_port(port=port)

    #
    # THEN
    #
    if expected_port == 0:
        assert port != 0
    else:
        assert port == expected_port
