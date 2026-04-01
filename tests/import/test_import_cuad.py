# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import pathlib
import shutil
import uuid

import pandas as pd
import pytest

from h2o_sonar.utils import testing
from tests.lib import test_test_suite_lib


"""CUAD: Contract Understanding Atticus Dataset

* corpus 100MB of PDF files
* 510 commercial legal contracts

Resources:

* https://www.atticusprojectai.org/cuad
    - CUAD www
* https://arxiv.org/abs/2103.06268
    - CUAD paper
* https://www.atticusprojectai.org/labeling-handbook
    - CUAD labeling handbook
* https://github.com/TheAtticusProject/cuad
    - GitHub

"""


@pytest.mark.skip(reason="One time import of the test data.")
@pytest.mark.generative
def test_cuad_test_suite(tmp_path: pathlib.Path) -> None:
    #
    # GIVEN
    #
    t_librarian = test_test_suite_lib.PocTestSuiteLibraryLibrarian
    t_item = t_librarian.TestSuiteLibItem

    ds_license_name = "CC BY 4.0"
    ds_reference = "https://www.atticusprojectai.org/cuad"
    ds_source = "https://github.com/TheAtticusProject/cuad"

    # path to extracted CUAD zip file
    cuad_zip_path = pathlib.Path(
        "/home/user/h2o-eval-studio-demo-data/cuad/original/CUAD_v1"
    )
    assert cuad_zip_path.exists(), f"Path {cuad_zip_path} does not exist"
    cuad_flat_pdf_corpus = pathlib.Path(
        "/home/user/h2o-eval-studio-demo-data/cuad/corpus-flat-raw"
    )
    assert cuad_flat_pdf_corpus.exists(), f"Path {cuad_flat_pdf_corpus} does not exist"
    cuad_pdf_corpus = pathlib.Path("/home/user/h2o-eval-studio-demo-data/cuad/corpus")
    assert cuad_pdf_corpus.exists(), f"Path {cuad_pdf_corpus} does not exist"

    # path to category definitions downloaded from:
    # - https://github.com/TheAtticusProject/cuad/blob/main/category_descriptions.csv
    cuad_categories_path = pathlib.Path(
        "data/generative/incoming/cuad/category_descriptions_as_questions.csv"
    )
    assert cuad_categories_path.exists(), f"Path {cuad_categories_path} does not exist"
    print(f"Reading categories from: {cuad_categories_path}")
    cats_df = pd.read_csv(cuad_categories_path, index_col=False)
    # frame to map: category -> question
    cat_question_map = {}
    for r in range(cats_df.shape[0]):
        category = cats_df.iloc[r, 0].replace("Category: ", "")
        question = cats_df.iloc[r, 1].replace("Description: ", "")
        fmt = " (" + cats_df.iloc[r, 2].replace("Format", "format") + ")"
        cat_question_map[category.lower()] = question + fmt
    print(f"Categories to questions:\n{json.dumps(cat_question_map, indent=2)}")

    # load master clauses
    print(f"Reading master clauses from: {cuad_zip_path}")
    master_clauses_df = pd.read_csv(
        cuad_zip_path / "master_clauses.csv",
        index_col=False,
    )
    master_clauses_cols = list(master_clauses_df.columns)
    print(f"Loaded {master_clauses_df.shape} master clauses")

    # S3 documents base path
    s3_base_path = (
        "https://eval-studio-artifacts.s3.us-east-1.amazonaws.com/"
        "h2o-eval-studio-demo-data/cuad/corpus"
    )

    # 1M+ lib index entries
    prompt_lib_entries = []
    prompt_lib_s3_base_path = (
        "https://eval-studio-artifacts.s3.us-east-1.amazonaws.com/"
        "h2o-eval-studio-suite-library"
    )

    #
    # WHEN
    #
    ts_tests = []
    test_cases = []
    for r in range(master_clauses_df.shape[0]):
        # doc names are crazy - rename all corpus documents to ensure S3 compatibility
        raw_doc_name = master_clauses_df.iloc[r, 0]
        print(raw_doc_name)
        raw_doc_path = cuad_flat_pdf_corpus / raw_doc_name
        if not raw_doc_path.exists():
            raise FileNotFoundError(
                f"Document '{raw_doc_name}' not found on corpus path: {raw_doc_path}"
            )
        safe_doc_name = f"cuad_contract_{r + 1:03d}.pdf"
        safe_doc_path = cuad_pdf_corpus / safe_doc_name
        print(f"Document sync '{raw_doc_name}' -> '{safe_doc_name}'")
        if not safe_doc_path.exists():
            print(f"  Copying document w/ safe name: {safe_doc_name}")
            shutil.copy(raw_doc_path, safe_doc_path)
        else:
            print(f"  Document w/ safe name {safe_doc_name} already exists")

        # TEST
        documents = [f"{s3_base_path}/{safe_doc_name}"]
        test_key = str(uuid.uuid4())
        test = testing.RagTestConfig(
            documents=documents,
            key=test_key,
        )
        ts_tests.append(test)
        test_cases_of_1_test = []

        for c in range(0, 41):
            cat_chunk_idx = 1 + c * 2
            cat_a_idx = 1 + 1 + c * 2
            # this is document chunk relevant for the answer
            cat_chunk = master_clauses_df.iloc[r, cat_chunk_idx]
            # question
            cat_q = cat_question_map.get(master_clauses_cols[cat_chunk_idx].lower())
            # this is the answer text
            cat_ea = master_clauses_df.iloc[r, cat_a_idx]
            if pd.isna(cat_ea):
                print(f"  #{c}: {cat_q} -> <no answer> -> SKIP")
                continue
            assert cat_q is not None, (
                f"Category question not found for "
                f"'{master_clauses_cols[cat_chunk_idx]}'"
            )
            print(
                f"  #{c}: {cat_q}\n"
                f"    {master_clauses_cols[cat_chunk_idx]}: {cat_chunk}\n"
                f"    {master_clauses_cols[cat_a_idx]}: {cat_ea}"
            )

            # TEST CASE
            test_case = testing.RagTestCaseConfig(
                prompt=cat_q,
                expected_output=cat_ea,
                constraints=[],
                config=test,
            )
            test_cases.append(test_case)
            test_cases_of_1_test.append(test_case)

        # TEST SUITE for 1 TEST
        ts_file_name = f"cuad_test_suite_d{r + 1}_{len(test_cases_of_1_test)}p.json"
        ts_name = f"CUAD: Contract Understanding Atticus Dataset - Contract Nr. {r + 1}"
        ts_description = (
            "Contract Understanding Atticus Dataset (CUAD) v1 is a corpus of "
            "13,000+ labels in 510 commercial legal contracts that have been "
            "manually labeled under the supervision of experienced lawyers to "
            "identify 41 types of legal clauses that are considered important "
            "in contract review in connection with a corporate transaction, "
            "including mergers and acquisitions. This test suite covers ONLY "
            f"document: {raw_doc_name}."
            '\n\nLicense: "CC BY 4.0"'
            '\n\nReference: "https://www.atticusprojectai.org/cuad"'
            '\n\nSource: "https://github.com/TheAtticusProject/cuad"'
        )
        test_suite_for_1_test = testing.RagTestSuiteConfig(
            test_cases=test_cases_of_1_test,
            name=ts_name,
            description=ts_description,
        )
        # prompt lib
        prompt_lib_entries.append(
            {
                t_item.KEY_NAME: ts_name,
                t_item.KEY_DESCRIPTION: ts_description,
                t_item.KEY_TEST_SUITE_URL: f"{prompt_lib_s3_base_path}/{ts_file_name}",
                t_item.KEY_EVALUATES: ["RAG"],
                t_item.KEY_T_COUNT: 1,
                t_item.KEY_TC_COUNT: len(test_cases_of_1_test),
                t_item.KEY_PURPOSES: ["legal", "retrieval"],
                t_item.KEY_ORIGIN: "3rd-party",
                t_item.KEY_CATS: [
                    "question_answering",
                ],
                t_item.KEY_SOURCE_URL: ds_source,
                t_item.KEY_REFERENCE_URL: ds_reference,
                t_item.KEY_LICENSE: ds_license_name,
            }
        )
        # save test suite for 1 test
        test_suite_path_1_test = pathlib.Path(tmp_path) / ts_file_name
        print(
            f"Saving CUAD test suite for document {raw_doc_name} with "
            f"{len(test_cases_of_1_test)} test cases:"
        )
        test_suite_for_1_test.save_as_json(file_path=test_suite_path_1_test)
        print(
            f"Test suite saved for {raw_doc_name} to: file://{test_suite_path_1_test}"
        )

    # TEST SUITE
    ts_file_name = "cuad_test_suite_510t_19700p.json"
    ts_name = "CUAD: Contract Understanding Atticus Dataset"
    ts_description = (
        f"Contract Understanding Atticus Dataset (CUAD) v1 is a corpus of "
        f"13,000+ labels in 510 commercial legal contracts that have been "
        f"manually labeled under the supervision of experienced lawyers to "
        f"identify 41 types of legal clauses that are considered important "
        f"in contract review in connection with a corporate transaction, "
        f"including mergers and acquisitions. "
        f'\n\nLicense: "{ds_license_name}"'
        f'\n\nReference: "{ds_reference}"'
        f'\n\nSource: "{ds_source}"'
    )
    test_suite = testing.RagTestSuiteConfig(
        test_cases=test_cases,
        name=ts_name,
        description=ts_description,
    )
    prompt_lib_entries.append(
        {
            t_item.KEY_NAME: ts_name,
            t_item.KEY_DESCRIPTION: ts_description,
            t_item.KEY_TEST_SUITE_URL: f"{prompt_lib_s3_base_path}/{ts_file_name}",
            t_item.KEY_EVALUATES: ["RAG"],
            t_item.KEY_T_COUNT: len(ts_tests),
            t_item.KEY_TC_COUNT: len(test_cases),
            t_item.KEY_PURPOSES: ["legal", "retrieval"],
            t_item.KEY_ORIGIN: "3rd-party",
            t_item.KEY_CATS: [
                "question_answering",
            ],
            t_item.KEY_SOURCE_URL: ds_source,
            t_item.KEY_REFERENCE_URL: ds_reference,
            t_item.KEY_LICENSE: ds_license_name,
        }
    )

    #
    # THEN
    #

    test_suite_path = pathlib.Path(tmp_path) / ts_file_name
    print(f"Saving CUAD test suite with {len(test_cases)} test cases ")
    test_suite.save_as_json(file_path=test_suite_path)
    print(f"Test suite saved to: file://{test_suite_path}")

    # save LIB JSON index entries for the test suite
    prompt_lib_path = pathlib.Path(tmp_path) / "h2o_eval_studio_suite_library_cuad.json"
    print(
        f"Saving CUAD test suite library index with {len(prompt_lib_entries)} entries"
    )
    with open(prompt_lib_path, "w", encoding="utf-8") as f:
        json.dump(prompt_lib_entries, f, indent=2)
    print(f"Test suite library index saved to: file://{prompt_lib_path}")


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
