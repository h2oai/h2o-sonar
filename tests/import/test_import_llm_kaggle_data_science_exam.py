# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import pathlib
import random

import datatable
import pandas
import pytest

from h2o_sonar.lib.api import datasets as d6s
from h2o_sonar.lib.api import models
from h2o_sonar.lib.integrations import genai
from h2o_sonar.utils import testing
from tests import test_utils
from tests.lib import given_generative


@pytest.mark.skip(reason="Kaggle data science exam data normalization")
@pytest.mark.generative
def test_kaggle_data_science_exam_to_classification(tmp_path):
    #
    # GIVEN
    #
    binomial_classification = False

    # alien
    # testset_path = "/mnt/h2oaissd/llm/kaggle-llm-science-exam/train.csv"
    # sonar
    testset_path = "/home/user/Downloads/kaggle-llm-science-exam--train.csv"

    #
    # WHEN
    #
    df = datatable.fread(testset_path)
    print(df.names)
    print(df.shape)

    # create BINOMIAL classification test suite - filter out only rows for 2 classes
    if binomial_classification:
        infix = "bin"
        filtered_df = df[(datatable.f.answer == "A") | (datatable.f.answer == "B"), :]
    else:
        infix = "multi"
        filtered_df = df

    #
    # THEN
    #
    filtered_df_path = (
        tmp_path / "kaggle_llm_science_exam_bin_classification_FILTERED.csv"
    )
    filtered_df.to_csv(str(filtered_df_path))
    print(f"Filtered DF path:\nfile://{filtered_df_path}")

    # LLM dataset: convert filtered frame to fake LLM dataset ######################

    resolved_dataset = d6s.LlmDataset()
    evaluated_models = []

    llm_model_names = [
        genai.OpenAiAssistantsRagClientVersion1.DEFAULT_LLM_MODEL,
        "claude-3-opus-20240229",
        "mistral-large-latest",
    ]

    if binomial_classification:
        answer_options = ["A", "B"]
    else:
        answer_options = ["A", "B", "C", "D", "E"]

    # h2oGPTe (will NOT be used)
    connection = test_utils.health.get_h2ogpte()
    categories = [d6s.LlmPromptCategories.classification.name]
    corpus = None

    for llm_model_name in llm_model_names:
        evaluated_llm_model = models.ExplainableLlmModel(
            connection=connection,
            model_type=models.ExplainableModelType.h2ogpte_llm,
            name=f"LLM: {llm_model_name}",
            llm_model_name=llm_model_name,
        )

        evaluated_models.append(evaluated_llm_model)

        model_key = evaluated_llm_model.key

        for index in range(filtered_df.shape[0]):
            row = filtered_df[index, :]

            resolved_dataset.add_input(
                i=row[0, "prompt"],
                corpus=corpus,
                context=None,
                categories=categories,
                expected_output=row[0, "answer"],
                actual_output=random.choice(answer_options),
                actual_duration=random.uniform(0.0, 1.0),
                cost=random.uniform(0.0, 1.0),
                model_key=model_key,
            )

    resolved_dataset_dict = resolved_dataset.to_dict()

    dataset_path = tmp_path / f"kaggle_llm_science_exam_class_{infix}_dataset.json"
    with open(dataset_path, "w") as f:
        json.dump(resolved_dataset_dict, f, indent=2)
    print(f"LLM dataset:\nfile://{dataset_path}")

    # test LAB: convert filtered frame to fake LLM dataset ######################

    test_lab = testing.RagTestLab(
        llm_host_connection=connection,
        raw_dataset=resolved_dataset,
        evaluated_models=evaluated_models,
        llm_model_names=llm_model_names,
        docs_cache_dir=tmp_path,
    )
    test_lab.dataset = resolved_dataset
    test_lab_path = tmp_path / f"kaggle_llm_science_exam_class_{infix}_test_lab.json"
    test_lab.save_as_json(test_lab_path)

    print(f"Test lab:\nfile://{test_lab_path}")


@pytest.mark.skip(reason="Kaggle data science exam data normalization 2 lab")
@pytest.mark.generative
def test_kaggle_data_science_exam_to_test_lab(tmp_path: pathlib.Path):
    """Import data from:

    https://www.kaggle.com/competitions/kaggle-llm-science-exam/leaderboard

    Parquet file was created using the RAG with the following configuration:

    - e5-large-v2 .... model used for the embeddings and retrieval:
      https://huggingface.co/intfloat/e5-large-v2
    - mistral-7b-openorca ... Mistral model finetuned on OpenOrca dataset is used as
      an LLM: https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca

    """

    #
    # GIVEN
    #

    # v0 non-realistic
    kaggle_model_name = "llm-science-exam/v0/leakage"
    parquet_path = (
        "/home/user/h/mli/eval-studio-gallery/reports/kaggle-llm-science-exam/"
        "raw-test-data/"
        "llm_science_exam_v0_20231124.parquet"
    )

    # v1 @ e5 @ mistral @ orca
    # kaggle_model_name = "team-h2o-llm-studio/e5-large-v2/mistral-7b-openorca"
    # parquet_path = (
    #     "/home/user/h/mli/eval-studio-gallery/reports/kaggle-llm-science-exam/"
    #     "raw-test-data/"
    #     "llm_science_exam_v1_20231127.parquet"
    # )

    df = pandas.read_parquet(parquet_path)
    print(df.columns)

    #
    # WHEN
    #

    # h2oGPTe will NOT be used
    connection = test_utils.health.get_h2ogpte()

    # corpus will not be synchronized, but it is Wikipedia
    corpus = ["https://www.wikipedia.org/"]

    # KGM team winning model
    rag_model = models.ExplainableRagModel(
        connection=connection,
        model_type=models.ExplainableModelType.unknown,
        name=f"{kaggle_model_name}/competition",
        llm_model_name=kaggle_model_name,
        documents=corpus,
    )

    categories = [d6s.LlmPromptCategories.question_answering.name]
    model_key = rag_model.key
    resolved_dataset = d6s.LlmDataset()
    for index, row in df.iterrows():
        expected_output_col = row["answer"]
        expected_output = row[expected_output_col]

        actual_output_col = row["predicted_answer"]
        actual_output = row[actual_output_col]

        chunks_as_list = list(row["context_list"])

        actual_duration = row["inference_time"] if "inference_time" in row else 0.0

        resolved_dataset.add_input(
            i=row["prompt"],
            corpus=corpus,
            context=chunks_as_list,
            categories=categories,
            expected_output=expected_output,
            actual_output=actual_output,
            actual_duration=actual_duration,
            model_key=model_key,
        )

    resolved_dataset.save_as_json(tmp_path / "kaggle_llm_science_exam_dataset.json")

    # test lab
    test_lab = testing.RagTestLab(
        llm_host_connection=connection,
        raw_dataset=resolved_dataset,
        evaluated_models=[rag_model],
        llm_model_names=[rag_model.llm_model_name],
        docs_cache_dir=tmp_path,
    )
    test_lab.dataset = resolved_dataset
    test_lab.save_as_json(tmp_path / "kaggle_llm_science_exam_test_lab.json")

    #
    # THEN
    #
    print(f"Test lab saved to: {tmp_path}")


@pytest.mark.skip(reason="Kaggle data science exam data normalization 2 suite")
@pytest.mark.generative
def test_merge_test_labs(tmp_path):
    #
    # GIVEN
    #

    # h2oGPTe will NOT be used
    connection = test_utils.health.get_h2ogpte()

    test_lab_1 = testing.RagTestLab.load_from_json(
        llm_host_connection=connection,
        # file_path="data/generative/kaggle_llm_science_exam_test_lab_h2o.json",
        file_path="data/generative/kaggle_llm_science_exam_test_lab_2x_small_25.json",
        docs_cache_dir=tmp_path,
    )
    test_lab_2 = testing.RagTestLab.load_from_json(
        llm_host_connection=connection,
        # file_path="data/generative/kaggle_llm_science_exam_test_lab_leak.json",
        file_path="data/generative/kaggle_llm_science_exam_test_lab_cosmos_25x2.json",
        docs_cache_dir=tmp_path,
    )

    entries_to_use = 50

    #
    # WHEN
    #
    test_dataset = d6s.LlmDataset()

    for i in range(entries_to_use):
        print(i)
        test_dataset.add_input(
            i=test_lab_1.dataset.inputs[i].i,
            corpus=test_lab_1.dataset.inputs[i].corpus,
            context=test_lab_1.dataset.inputs[i].context,
            categories=test_lab_1.dataset.inputs[i].categories,
            expected_output=test_lab_1.dataset.inputs[i].expected_output,
            actual_output=test_lab_1.dataset.inputs[i].actual_output,
            actual_duration=test_lab_1.dataset.inputs[i].actual_duration,
            model_key=test_lab_1.dataset.inputs[i].model_key,
        )
        test_dataset.add_input(
            i=test_lab_2.dataset.inputs[i].i,
            corpus=test_lab_2.dataset.inputs[i].corpus,
            context=test_lab_2.dataset.inputs[i].context,
            categories=test_lab_2.dataset.inputs[i].categories,
            expected_output=test_lab_2.dataset.inputs[i].expected_output,
            actual_output=test_lab_2.dataset.inputs[i].actual_output,
            actual_duration=test_lab_2.dataset.inputs[i].actual_duration,
            model_key=test_lab_2.dataset.inputs[i].model_key,
        )

    test_dataset.save_as_json(
        tmp_path / f"kaggle_llm_science_exam_dataset_2x_small_{entries_to_use}.json"
    )

    # test lab
    rag_models_1 = list(test_lab_1.rag_models.values())
    rag_models_2 = list(test_lab_2.rag_models.values())
    rag_models = rag_models_1 + rag_models_2

    test_lab = testing.RagTestLab(
        llm_host_connection=connection,
        raw_dataset=test_dataset,
        evaluated_models=rag_models,
        llm_model_names=[n.name for n in rag_models],
        docs_cache_dir=tmp_path,
    )
    test_lab.dataset = test_dataset
    test_lab.save_as_json(
        tmp_path / f"kaggle_llm_science_exam_test_lab_4x_{entries_to_use}.json"
    )

    #
    # THEN
    #
    print(f"Test lab saved to: {tmp_path}")


@pytest.mark.skip(reason="Kaggle data science exam data normalization 2 suite")
@pytest.mark.generative
def test_kaggle_data_science_exam_to_test_suite(tmp_path: pathlib.Path):
    """Create test suite WITHOUT actual values to be used to get actual data from
    other fine-tuned / non fine-tuned models.

    """
    #
    # GIVEN
    #
    parquet_path = (
        "/home/user/h/mli/eval-studio-gallery/reports/kaggle-llm-science-exam/"
        "raw-test-data/"
        "llm_science_exam_v1_20231127.parquet"
    )
    df = pandas.read_parquet(parquet_path)
    print(df.columns)

    #
    # WHEN
    #

    # corpus will not be synchronized, but it is Wikipedia
    corpus = ["https://www.wikipedia.org/"]

    categories = [d6s.LlmPromptCategories.question_answering.name]
    test_config = testing.RagTestConfig(
        documents=corpus,
    )
    test_suite = testing.RagTestSuiteConfig()
    for index, row in df.iterrows():
        expected_output_col = row["answer"]
        expected_output = row[expected_output_col]

        test_suite.add_test_case(
            testing.RagTestCaseConfig(
                prompt=row["prompt"],
                categories=categories,
                expected_output=expected_output,
                config=test_config,
            )
        )

    test_suite.save_as_json(tmp_path / "kaggle_llm_science_exam_test_suite.json")

    #
    # THEN
    #
    print(f"Test lab saved to: {tmp_path}")


@pytest.mark.skip(reason="Kaggle data science exam data normalization 2 suite")
@pytest.mark.generative
def test_complete_test_suite_bare_llms(tmp_path: pathlib.Path):
    #
    # TODO IMPORTANT: bare LLM cannot be used as there are NO CHUNKS
    #   - therefore therefore there are no chunks to be used for
    #     RETRIEVAL metrics
    #     GENERATION metrics
    #   - dataset for the Kaggle competition is Wikipedia
    #     cannot be fine-tuned in h2oGPTe (too much data)

    #
    # GIVEN
    #
    # h2oGPTe
    connection = test_utils.health.get_h2ogpte()

    llm_model_name = given_generative.s.H2OGPTE_JUDGE_LLM_MODEL_NAME
    # llm_model_name = "h2oai/h2ogpt-4096-llama2-70b-chat-4bit"

    test_suite_path = "data/generative/kaggle_llm_science_exam_test_suite.json"
    test_suite = testing.RagTestSuiteConfig.load_from_json(test_suite_path)

    #
    # WHEN
    #
    test_lab = testing.RagTestLab.from_rag_test_suite(
        rag_connection=connection,
        rag_test_suite=test_suite,
        llm_model_names=[llm_model_name],
        docs_cache_dir=tmp_path,
    )

    # SKIP: test_lab.build() - RAG is not used, LLM instead

    test_lab.complete_dataset()

    #
    # WHEN
    #

    # gather prompts
    prompts = []
    for i, test_case in enumerate(test_suite.test_cases):
        prompts.append(test_case.prompt)

        if i == 3:
            break

    responses = genai.H2oGpteRagClient(connection).ask_model(
        prompts=prompts, llm_model_name=llm_model_name
    )

    for i, r in enumerate(responses):
        test_suite.test_cases[i].actual_output = r[i]
        test_suite.test_cases[i].actual = r[i]

    #
    # THEN
    #
    print("DONE")


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
