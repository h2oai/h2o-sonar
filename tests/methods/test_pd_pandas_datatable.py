# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
from unittest import TestCase

import numpy as np
from datatable import fread

from h2o_sonar import loggers as logging
from tests.methods.ice_pd_test_commons import DATASET_DATE_FEATURES_PATH
from tests.test_utils import find_locally


class TestPdPandasDatatable(TestCase):
    """Test Pandas vs. datatable discrepancies like data type inferences."""

    # override
    def setUp(self):
        logging.setLevel(logging.DEBUG)

    def test_datatable_to_pandas_data_conversion_ok(self):
        logging.debug("# datatable 2 pandas: date ###")

        # GIVEN
        df = fread(file=find_locally(DATASET_DATE_FEATURES_PATH))
        logging.debug("CSV loaded and parsed to datatable:")
        logging.debug(f"  Shape: {df.shape}")
        logging.debug(f"  Date type: {df.ltypes[0]}")

        # WHEN
        pf = df.to_pandas()
        logging.debug("Frame converted from datatable to Pandas:")
        logging.debug(f"  Shape: {pf.shape}")
        logging.debug(f"  Date type: {pf.dtypes['Date']}")

        pf = df.to_pandas()
        logging.debug("Frame converted from datatable to Pandas (old method):")
        logging.debug(f"  Shape: {pf.shape}")
        logging.debug(f"  Date type: {pf.dtypes['Date']}")

        # THEN
        self.assertEqual(pf.dtypes["Date"], np.int32)

    def test_datatable_to_pandas_data_conversion_nok(self):
        logging.debug("# datatable 2 pandas: date ###")

        # GIVEN
        df = fread(file=find_locally(DATASET_DATE_FEATURES_PATH))
        logging.debug("CSV loaded and parsed to datatable:")
        logging.debug(f"  Shape: {df.shape}")
        logging.debug(f"  Date type: {df.ltypes[0]}")

        # WHEN
        pf = df.to_pandas()
        logging.debug("Frame converted from datatable to Pandas:")
        logging.debug(f"  Shape: {pf.shape}")
        logging.debug(f"  Date type: {pf.dtypes['Date']}")

        pf = df.to_pandas()
        logging.debug("Frame converted from datatable to Pandas (old method):")
        logging.debug(f"  Shape: {pf.shape}")
        logging.debug(f"  Date type: {pf.dtypes['Date']}")

        # THEN
        self.assertEqual(pf.dtypes["Date"], np.int32)
