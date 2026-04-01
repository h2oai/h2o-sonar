# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import logging
import random
from functools import partial

import datatable
import pytest
import pytest_httpserver
from werkzeug import wrappers

from h2o_sonar import interpret
from h2o_sonar.lib.api import interpretations
from h2o_sonar.lib.api import models
from tests import test_utils


i_dict_keys = interpretations.Interpretation


# indicates availability of the REAL REST server endpoint
# DAI_LOCAL_REST_SERVER_URL = f"http://localhost:8080{models.DAI_REST_PATH_MODEL}"
# else VIRTUAL REST sever endpoint
DAI_LOCAL_REST_SERVER_URL = ""


def given_dai_local_rest_server_url(httpserver: pytest_httpserver.HTTPServer):
    return (
        DAI_LOCAL_REST_SERVER_URL
        if DAI_LOCAL_REST_SERVER_URL
        else httpserver.url_for(models.DAI_REST_PATH_MODEL)
    )


def given_virtual_dai_rest_server_handler(request):
    """Construct DYNAMIC REST server response."""

    row_count = len(json.loads(request.get_data().decode("utf-8"))["rows"])
    scores = []
    for _ in range(row_count):
        prediction = random.random()
        scores.append([f"{prediction}", f"{1.0 - prediction}"])

    return wrappers.Response(
        json.dumps(
            {
                "id": "f5b84036-1d65-11ed-a268-a0cec818dc16",
                "fields": [
                    "DEFAULT_PAYMENT_NEXT_MONTH.0",
                    "DEFAULT_PAYMENT_NEXT_MONTH.1",
                ],
                "score": scores,
            }
        )
    )


def given_virtual_dai_rest_server(httpserver: pytest_httpserver.HTTPServer):
    """Set-up virtual DAI REST server for binomial model."""
    if not DAI_LOCAL_REST_SERVER_URL:
        httpserver.expect_request(
            f"{models.DAI_REST_PATH_MODEL}{models.DAI_REST_PATH_SCORE}"
        ).respond_with_handler(given_virtual_dai_rest_server_handler)
        httpserver.expect_request(
            f"{models.DAI_REST_PATH_MODEL}{models.DAI_REST_PATH_SAMPLE}"
        ).respond_with_json(
            # json.dumps(
            {
                "fields": [
                    "LIMIT_BAL",
                    "SEX",
                    "EDUCATION",
                    "MARRIAGE",
                    "AGE",
                    "PAY_1",
                    "PAY_2",
                    "PAY_3",
                    "PAY_4",
                    "PAY_5",
                    "PAY_6",
                    "BILL_AMT1",
                    "BILL_AMT2",
                    "BILL_AMT3",
                    "BILL_AMT4",
                    "BILL_AMT5",
                    "BILL_AMT6",
                    "PAY_AMT1",
                    "PAY_AMT2",
                    "PAY_AMT3",
                    "PAY_AMT4",
                    "PAY_AMT5",
                    "PAY_AMT6",
                ],
                "rows": [
                    [
                        "10000.0",
                        "male",
                        "university",
                        "divorce",
                        "24.0",
                        "-2.0",
                        "1.0",
                        "2.0",
                        "-2.0",
                        "-2.0",
                        "0.0",
                        "-200.0",
                        "-200.0",
                        "-173.0",
                        "-7905.0",
                        "-765.0",
                        "-3272.0",
                        "61.0",
                        "16.0",
                        "10.0",
                        "10.0",
                        "31.0",
                        "3.0",
                    ]
                ],
            }
            # )
        )


@pytest.mark.h2o_sonar
def test_dai_rest_server_predict_function(httpserver: pytest_httpserver.HTTPServer):
    # GIVEN
    dataset_path = test_utils.find_locally(
        "data/predictive/pd_ice_creditcard_10_rows.csv"
    )
    x = datatable.fread(dataset_path)
    del x[:, "DEFAULT_PAYMENT_NEXT_MONTH"]
    del x[:, "ID"]
    x = x[0:1, :]
    given_virtual_dai_rest_server(httpserver)
    predict_function = partial(
        models._dai_rest_server_predict_method,
        given_dai_local_rest_server_url(httpserver),
    )
    print(
        f"Using Driverless AI local REST server: "
        f"{given_dai_local_rest_server_url(httpserver)}"
    )

    # WHEN
    predictions = predict_function(x)

    # THEN
    print(
        f"Driverless AI local REST server predictions ({type(predictions)}):\n"
        f"{predictions}"
    )
    assert isinstance(predictions, datatable.Frame)


@pytest.mark.h2o_sonar
def test_dai_rest_server_model_class(httpserver: pytest_httpserver.HTTPServer):
    # GIVEN
    given_virtual_dai_rest_server(httpserver)

    # WHEN
    model = models.DriverlessAiRestServerModel(
        model_server_url=given_dai_local_rest_server_url(httpserver)
    )

    # THEN
    print(f"Driverless AI REST server model:\n{model}")
    assert model.meta.used_features


@pytest.mark.h2o_sonar
def test_all_explainers(tmpdir, httpserver: pytest_httpserver.HTTPServer):
    # GIVEN
    dataset_path = test_utils.find_locally(
        "data/predictive/pd_ice_creditcard_10_rows.csv"
    )
    target_col = "DEFAULT_PAYMENT_NEXT_MONTH"
    given_virtual_dai_rest_server(httpserver)

    # WHEN
    i = interpret.run_interpretation(
        dataset=dataset_path,
        model=given_dai_local_rest_server_url(httpserver),
        target_col=target_col,
        results_location=tmpdir,
        log_level=logging.DEBUG,
        explainer_keywords=[interpret.KEYWORD_FILTER_ALL],
    )

    # THEN
    print(f"Interpretation: {i}")
    i_dict = i.to_dict()
    assert i
    expected = 6
    assert expected <= len(
        i_dict[i_dict_keys.KEY_RESULT][i_dict_keys.KEY_SCHEDULED_EXPLAINERS]
    )
    assert expected <= len(i.get_scheduled_explainer_ids())
    assert expected <= len(i.get_finished_explainer_ids())
    assert expected <= len(i.get_successful_explainer_ids())
