# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import os
import time
from unittest import TestCase

import datatable
import pandas
import pandas as pd
import pytest

from h2o_sonar import errors
from h2o_sonar import interpret
from h2o_sonar import loggers
from h2o_sonar.utils import sampling
from tests import test_utils


DATASET_CREDITCARD_PATH = "data/predictive/pd_ice_creditcard_train.csv"
DATASET_CREDITCARD_10_ROWS_PATH = "data/predictive/pd_ice_creditcard_10_rows.csv"
# loan_40.csv dataset:
#  - rows :    424 713 579 ~ 424M
#  - bytes: 42 953 397 956 ~  42GB
DATASET_LOAN_40 = "/home/user/h/datasets/load-testing/loan_40.csv"


class TestDatasetSampling(TestCase):
    """Test dataset sampling."""

    def setUp(self):
        loggers.setLevel(loggers.DEBUG)

        # datasets
        self.cats = pd.DataFrame(
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

    @staticmethod
    def _random_sampling(dataset, limit):
        if dataset is not None:
            if isinstance(dataset, pd.DataFrame):
                return dataset.sample(limit)
            elif isinstance(dataset, datatable.Frame):
                # IMPROVE: memory consumption ~ row selection @ datatable
                return dataset.to_pandas().sample(limit)
            else:
                raise ValueError(f"Wrong dataset type {type(dataset)}")
        else:
            raise ValueError("Dataset is empty")

    def test_pandas_random_sampling_cc(self):
        # GIVEN
        df = pd.read_csv(
            test_utils.find_locally(DATASET_CREDITCARD_10_ROWS_PATH), index_col=0
        )
        loggers.debug(f"Dataset ({len(df.shape)}):\n{df.to_string()}")
        limit = 5

        # WHEN
        sampled = TestDatasetSampling._random_sampling(df, limit)

        # THEN
        loggers.debug(f"Sampled dataset ({len(sampled.shape)}):\n{sampled.to_string()}")
        self.assertEqual((limit, 24), sampled.shape)

    def test_pandas_random_sampling_cats(self):
        # GIVEN
        df = self.cats
        loggers.debug(f"Dataset ({len(df.shape)}):\n{df.to_string()}")
        limit = 5

        # WHEN
        sampled = TestDatasetSampling._random_sampling(df, limit)

        # THEN
        loggers.debug(f"Sampled dataset ({len(sampled.shape)}):\n{sampled.to_string()}")
        self.assertEqual((limit, 3), sampled.shape)

    def test_datatable_random_sampling_cc(self):
        # GIVEN
        df = datatable.fread(
            file=test_utils.find_locally(DATASET_CREDITCARD_10_ROWS_PATH)
        )
        loggers.debug(f"Dataset ({len(df.shape)}):\n{str(df)}")
        limit = 5

        # WHEN
        sampled = TestDatasetSampling._random_sampling(df, limit)

        # THEN
        loggers.debug(f"Sampled dataset ({len(sampled.shape)}):\n{sampled.to_string()}")
        self.assertEqual((limit, 25), sampled.shape)


@pytest.mark.skip(reason="This test requires huge dataset")
@pytest.mark.h2o_sonar
def test_oom_protection_checker():
    # GIVEN
    dataset = DATASET_LOAN_40

    # WHEN
    (will_fit, d_size, ram_size) = sampling.DatasetSampler.is_dataset_fit_in_memory(
        dataset
    )

    # THEN
    print(
        f"Dataset will fit: {will_fit} @ system {d_size}/{ram_size} "
        f"({d_size / ram_size} x free RAM)"
    )
    assert not will_fit


@pytest.mark.skip(reason="This test requires huge dataset")
@pytest.mark.skipif(not test_utils.is_mojo_supported(), reason="MOJO is not supported")
@pytest.mark.h2o_sonar
def test_oom_protection(tmp_path):
    import daimojo

    # GIVEN
    dataset = DATASET_LOAN_40

    # WHEN
    try:
        interpret.run_interpretation(
            dataset=dataset,
            model=daimojo.model("data/predictive/models/creditcard-binomial.mojo"),
            target_col="WHATEVER",
            results_location=str(tmp_path),
        )
    except errors.MliError as ex:
        assert "OOM" in str(ex)
        return

    # THEN
    raise AssertionError("Test must fail!")


@pytest.mark.skip(reason="This test requires huge dataset")
@pytest.mark.h2o_sonar
@pytest.mark.parametrize(
    "sampling_limit",
    [
        123,  # 0.005s
        1_000,  # 0.007s
        100_000,  # 0.5s
        1_234_567,  # 8s
        # WHOLE DATASET Pandas chunk READ: 1M x 420 ~= 1h > Dask needed for final filter
    ],
)
def test_head_sampler(tmp_path, sampling_limit):
    # GIVEN
    dataset = DATASET_LOAN_40
    sampled_dataset_path = os.path.join(tmp_path, "pandas_head_sampled_dataset.csv")

    # WHEN
    started = time.time()
    sampler = sampling.HeadOfDatasetSampling(chunk_size=1_000)
    sampler.sample_dataset(
        dataset=dataset,
        sampling_limit=sampling_limit,
        sampled_dataset_path=sampled_dataset_path,
    )
    print(
        f"Dataset {dataset} HEAD sampled to {sampling_limit} rows in "
        f"{time.time() - started}s"
    )

    # THEN
    assert os.path.isfile(sampled_dataset_path)
    assert pandas.read_csv(sampled_dataset_path).shape[0] == sampling_limit


#
# SAMPLING test cases
#
# * additional sampling testing:
#   1. MODIFY interpret.run_interpretation(..., sample_num_rows=5_000, ...)
#   2. make test (all tests) OR pytest -sv tests/lib (all explainers tests)
#   3. ^ will run all the tests and sample all the datasets
#


@pytest.mark.skip(reason="This test is used to prepare test datasets.")
@pytest.mark.h2o_sonar
def test_head_sampling(tmp_path):
    # GIVEN
    # get 10k head rows from the 42GB
    dataset = "/home/user/h/datasets-load-testing/loan_40gb.csv"
    sampling_limit = 10_000
    sampled_dataset_path = os.path.join(tmp_path, "sampled-dataset.csv")

    # WHEN
    started = time.time()
    sampler = sampling.HeadOfDatasetSampling()
    (sampled, sampled_frame, sampled_dataset_path_out) = sampler.sample_dataset(
        dataset=dataset,
        sampling_limit=sampling_limit,
        sampled_dataset_path=sampled_dataset_path,
        # regression: loan_amnt column will be used
    )
    print(
        f"Dataset {dataset} sampled ({sampled}) to {sampled_dataset_path_out} "
        f"given sampling limit {sampling_limit} rows in "
        f"{time.time() - started}s"
    )

    # THEN
    assert os.path.isfile(sampled_dataset_path)
    assert pandas.read_csv(sampled_dataset_path).shape[0] == sampling_limit


@pytest.mark.h2o_sonar
@pytest.mark.parametrize(
    "sampling_limit,dataset,target_col",
    [
        # ### 1MB numeric dataset sanity check
        (5_000, test_utils.find_locally("data/predictive/creditcard.csv"), "AGE"),
        # ### 1MB string dataset LE test: 0.018543720245361328s
        # (5_000, "data/predictive/creditcard_str_10k.csv", "EDUCATION"),
        # ### 1GB dataset: 7.310097932815552s
        # (100_000, "/home/user/h/datasets/loan.csv", "pub_rec"),
        # ### 42GB dataset NUM >>> OOM @ sklearn sampler
        # (
        #    100_000,
        #    "/home/user/h/datasets-load-testing/loan_40gb.csv",
        #    "emp_length",  # classes: [None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # ),
        # ### 42GB dataset CAT >>> OOM @ sklearn sampler
        # (
        #         100_000,
        #         "/home/user/h/datasets-load-testing/loan_40gb.csv",
        #         "home_ownership",  # cls: ['MORTGAGE', 'NONE', 'OTHER', 'OWN', 'RENT']
        # ),
    ],
)
def test_stratified_sampling(tmp_path, sampling_limit, dataset, target_col):
    # GIVEN
    sampled_dataset_path = os.path.join(tmp_path, "stratified_sampling_dataset.csv")

    # WHEN
    started = time.time()
    sampler = sampling.StratifiedDatasetSampling()
    (sampled, sampled_frame, sampled_dataset_path_out) = sampler.sample_dataset(
        dataset=dataset,
        sampling_limit=sampling_limit,
        sampled_dataset_path=sampled_dataset_path,
        target_col=target_col,
        is_classification=True,
    )
    print(
        f"Dataset {dataset} sampled ({sampled}) to {sampled_dataset_path_out} "
        f"given sampling limit {sampling_limit} rows in "
        f"{time.time() - started}s"
    )

    # THEN
    assert os.path.isfile(sampled_dataset_path)
    assert pandas.read_csv(sampled_dataset_path).shape[0] == sampling_limit


@pytest.mark.h2o_sonar
@pytest.mark.parametrize(
    "sampling_limit,dataset",
    [
        # ### 1MB dataset: sanity check
        (5_000, test_utils.find_locally("data/predictive/creditcard.csv")),
        # ### 1GB dataset: 1.060856819152832s
        # (100_000, "/home/user/h/datasets/loan.csv"),
        # ### 42GB dataset: 5'39s ~ 339.07069993019104s
        # (
        #     100_000,
        #     "/home/user/h/datasets-load-testing/loan_40gb.csv",
        # ),
    ],
)
def test_random_sampling(tmp_path, sampling_limit, dataset):
    # GIVEN
    sampled_dataset_path = os.path.join(tmp_path, "stratified_sampling_dataset.csv")

    # WHEN
    started = time.time()
    sampler = sampling.StratifiedDatasetSampling()
    (sampled, sampled_frame, sampled_dataset_path_out) = sampler.sample_dataset(
        dataset=dataset,
        sampling_limit=sampling_limit,
        sampled_dataset_path=sampled_dataset_path,
        is_classification=False,
    )
    print(
        f"Dataset {dataset} sampled ({sampled}) to {sampled_dataset_path_out} "
        f"given sampling limit {sampling_limit} rows in "
        f"{time.time() - started}s"
    )

    # THEN
    assert os.path.isfile(sampled_dataset_path)
    assert pandas.read_csv(sampled_dataset_path).shape[0] == sampling_limit


@pytest.mark.skip(reason="This test requires huge dataset outside H2O Sonar")
@pytest.mark.skipif(not test_utils.is_mojo_supported(), reason="MOJO is not supported")
@pytest.mark.h2o_sonar
@pytest.mark.parametrize(
    "sampling_limit,dataset,target_col,model_path",
    [
        # ### 42GB dataset: regression
        (
            100_000,
            "/home/user/h/datasets-load-testing/loan_40gb.csv",
            "loan_amnt",
            test_utils.find_locally(
                "data/predictive/models/loan-regression-loan_amnt.mojo"
            ),
        ),
    ],
)
def test_interpretation_with_sampling(
    tmp_path, sampling_limit, dataset, target_col, model_path
):
    # GIVEN
    sampling_limit = None  # automatic sampling

    # WHEN
    started = time.time()
    interpretation = interpret.run_interpretation(
        dataset=dataset,
        model=model_path,
        target_col=target_col,
        sample_num_rows=sampling_limit,
        results_location=str(tmp_path),
    )
    print(
        f"Interpretation of {model_path} on {dataset} with sampling limit "
        f"{sampling_limit} finished in {time.time() - started}s"
    )

    # THEN
    assert interpretation
    print(
        f"Sampling limit chosen by H2O Sonar: "
        f"{interpretation.common_params.sample_num_rows}"
    )
    assert interpretation.common_params.sample_num_rows


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
