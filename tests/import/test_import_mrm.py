# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import pathlib
import uuid

import datatable
import pytest


@pytest.mark.skip(reason="One time import of the test data.")
@pytest.mark.generative
def test_import_csv_to_lab(tmp_path):
    #
    # GIVEN
    #
    json_path = pathlib.Path() / "data" / "generative" / "incoming" / "example-y-d.json"
    with open(json_path) as json_file:
        json_data = json.load(json_file)

    #
    # WHEN
    #

    # gather RAGs which were used
    test_lab_model_names = [
        n for n in json_data.keys() if n not in ["question", "answer"]
    ]
    test_lab_model_keys = {n: str(uuid.uuid4()) for n in test_lab_model_names}

    test_lab_models = []
    for m in test_lab_model_names:
        test_lab_models.append(
            {
                "key": test_lab_model_keys[m],
                "connection": "7dd3f684-eca7-478b-b866-7a68572a87ef",  # not used
                "model_type": "openai_rag",  # not used
                "name": f"{m} RAG model - LLM: N/A, corpus: [N/A]",
                "collection_id": "asst_etkHMrK9RhlYDkYLTcyEf5zB",  # not used
                "collection_name": "RAG collection",  # not used
                "llm_model_name": f"{m}",  # unknown - RAG host used as replacement
                "documents": [],  # unused & unknown
            }
        )

    # convert to CSV with questions and answers to the lab format
    test_lab_inputs = []
    for row in range(len(json_data["question"])):
        for model_name in test_lab_model_names:
            test_lab_inputs.append(
                {
                    "key": str(uuid.uuid4()),
                    "input": json_data["question"][row],
                    "corpus": [],
                    "context": [],
                    "categories": [],
                    "relationships": [],
                    "expected_output": json_data["answer"][row],
                    "output_constraints": [],
                    "actual_output": json_data[model_name][row],
                    "actual_duration": 0.0,
                    "cost": 0.0,
                    "model_key": test_lab_model_keys[model_name],
                }
            )

    # assemble test lab
    test_lab_dict = {
        "name": "TestLab",
        "description": "Test lab for RAG / LLM evaluation.",
        "raw_dataset": {
            "inputs": [],
        },
        "dataset": {
            "inputs": test_lab_inputs,
        },
        "models": test_lab_models,
        "llm_model_names": test_lab_model_names,
        "docs_cache": {},  # unused
    }

    #
    # THEN
    #

    # save as JSON
    lab_path = tmp_path / "test-lab-example-y-d.json"
    with open(lab_path, "w") as lab_file:
        json.dump(test_lab_dict, lab_file, indent=4)
    print(f"\nTest lab:\n  file://{lab_path}")


@pytest.mark.skip(reason="One time conversion of the test data.")
@pytest.mark.generative
def test_json_to_csv(tmp_path):
    #
    # GIVEN
    #
    json_path = pathlib.Path() / "data" / "generative" / "incoming" / "example-y-d.json"
    with open(json_path) as json_file:
        json_data = json.load(json_file)

    #
    # WHEN
    #
    csv_path = str(tmp_path / "example-y-d.csv")
    datatable.Frame(json_data).to_csv(str(csv_path))

    #
    # THEN
    #
    print(f"file://{csv_path}")


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
