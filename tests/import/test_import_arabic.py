# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import pathlib

import datatable
import pytest

from h2o_sonar.utils import testing


#
# ARC AI2 test suite.
#


@pytest.mark.skip(reason="One time import of the test data.")
@pytest.mark.generative
def test_import_arc_ai2(tmp_path) -> None:
    """ARC (AI2 Reasoning Challenge) in Arabic (machine translation).

    The AI2’s Reasoning Challenge (ARC) dataset is a multiple-choice question-answering
    dataset, containing questions from science exams from grade 3 to grade 9.
    The dataset is split in two partitions: Easy and Challenge, where the latter
    partition contains the more difficult questions that require reasoning. Most of
    the questions have 4 answer choices, with <1% of all the questions having either
    3 or 5 answer choices. ARC includes a supporting KB of 14.3M unstructured text
    passages.

    Example prompt:

    Q: Anna is holding an ice cube. Why does the cube melt in her hand?

    1) Heat moves from her hand to the ice cube.
    2) Cold moves from her hand to the ice cube.
    3) Heat moves from the ice cube to her hand.
    4) Cold moves from the ice cube to her hand.

    A: 1) Heat moves from her hand to the ice cube.

    English version:

    * https://paperswithcode.com/dataset/arc

    Arabic version:

    * https://gitlab.com/tiiuae/alghafa/-/blob/main/arabic-eval/
        arc_challenge_okapi_ar/eval/arc_challenge_ar_okapi_test.csv

    """

    #
    # GIVEN
    #
    raw_dict = datatable.fread(
        pathlib.Path()
        / "data"
        / "generative"
        / "incoming"
        / "llm-benchmarks-arabic"
        / "arc_challenge_ar_okapi_test.csv"
    ).to_dict()

    #
    # WHEN
    #
    test_suite = testing.RagTestSuiteConfig(
        name="Arc AI2 Challenge Okapi Test Suite",
        description=(
            "The AI2’s Reasoning Challenge (ARC) dataset is a multiple-choice "
            "question-answering dataset, containing questions from science exams from "
            "grade 3 to grade 9. The dataset is split in two partitions: Easy and "
            "Challenge, where the latter partition contains the more difficult "
            "questions that require reasoning. Most of the questions have 4 answer "
            "choices, with <1% of all the questions having either 3 or 5 answer "
            "choices. ARC includes a supporting KB of 14.3M unstructured text "
            "passages. See: https://gitlab.com/tiiuae/alghafa/-/blob/main/arabic-eval/ "
            "and https://paperswithcode.com/dataset/arc"
        ),
    )
    test = testing.RagTestConfig(
        documents=[]  # NO documents as it is LLM question-answering test
    )

    rows = len(raw_dict["index"])
    print(f"Building test suite with {rows} prompts...")
    for i in range(rows):
        prompt = raw_dict["instruction"][i]
        expected_answer = raw_dict["response"][i]

        test_suite.add_test_case(
            testing.RagTestCaseConfig(
                prompt=prompt,
                categories=["question-answering"],
                expected_output=expected_answer,
                constraints=[],
                config=test,
            )
        )

    #
    # THEN
    #
    print(json.dumps(test_suite.to_dict(), indent=4))
    test_suite.save_as_json(pathlib.Path(tmp_path) / "test_suite.json")


#
# MMLU test suite.
#

"""

The MMLU (Massive Multitask Language Understanding) benchmark is a comprehensive test
for evaluating the knowledge and reasoning abilities of language models across
diverse subjects. It consists of multiple-choice questions spanning 57 tasks,
from elementary to professional levels, covering various fields like STEM,
humanities, and social sciences. MMLU is designed to assess a model's ability to
generalize knowledge and reason across different domains, making it a challenging and
realistic evaluation.

"""


@pytest.mark.skip(reason="One time import of the test data.")
@pytest.mark.generative
def test_mmlu(tmp_path) -> None:
    """MMLU (Massive Multitask Language Understanding) in Arabic (machine translation).

    The MMLU (Massive Multitask Language Understanding) benchmark is a comprehensive
    test for evaluating the knowledge and reasoning abilities of language models across
    diverse subjects. It consists of multiple-choice questions spanning 57 tasks,
    from elementary to professional levels, covering various fields like STEM,
    humanities, and social sciences. MMLU is designed to assess a model's ability to
    generalize knowledge and reason across different domains, making it a challenging
    and realistic evaluation.

    Example prompt:

    QUESTION: The Nation of Islam appealed to:

    1) Second-generation immigrants born in Britain from Greater India
    2) White Americans wishing to convert to Islam
    3) African Americans who felt excluded from the U.S. "melting pot"
    4) Africans in the Caribbean living in inner cities with a distinctive youth culture

    Answer with the letter of the correct answer.

    ANSWER: 3) African Americans who felt excluded from the U.S. "melting pot"

    English version:

    * https://paperswithcode.com/dataset/arc

    Arabic version:

    * https://gitlab.com/tiiuae/alghafa/-/blob/main/arabic-eval/mmlu_okapi_ar/eval/
        mmlu_ar_okapi_test.csv?ref_type=heads

    """

    #
    # GIVEN
    #
    max_test_cases = 50_000
    raw_dict = datatable.fread(
        pathlib.Path()
        / "data"
        / "generative"
        / "incoming"
        / "llm-benchmarks-arabic"
        / "mmlu_ar_okapi_test.csv"
    ).to_dict()

    #
    # WHEN
    #
    test_suite = testing.RagTestSuiteConfig(
        name="MMLU (Arabic)",
        description=(
            "The MMLU (Massive Multitask Language Understanding) benchmark is "
            "a comprehensive test for evaluating the knowledge and reasoning "
            "abilities of language models across diverse subjects. It consists of "
            "multiple-choice questions spanning 57 tasks, from elementary to "
            "professional levels, covering various fields like STEM, humanities, "
            "and social sciences. MMLU is designed to assess a model's ability to "
            "generalize knowledge and reason across different domains, making it a "
            "challenging and realistic evaluation. This is the Arabic version "
            "of the MMLU benchmark, translated using machine translation. "
            "\n\nSource: https://gitlab.com/tiiuae/alghafa/-/blob/main/arabic-eval"
            "/mmlu_okapi_ar/eval/mmlu_ar_okapi_test.csv"
        ),
    )
    test = testing.RagTestConfig(
        documents=[]  # NO documents as it is LLM question-answering test
    )

    print(f"Keys: {raw_dict.keys()}")
    for i in range(len(raw_dict["question"])):
        if i >= max_test_cases:
            break
        # print("=" * 50)
        # print(f"Question: {raw_dict['question'][i]}")
        # print(f"S1: {raw_dict['sol1'][i]}")
        # print(f"S2: {raw_dict['sol2'][i]}")
        # print(f"S3: {raw_dict['sol3'][i]}")
        # print(f"S4: {raw_dict['sol4'][i]}")
        # print(f"Label: {raw_dict['label'][i]}")

        label_int = int(raw_dict["label"][i])
        options = [
            raw_dict["sol1"][i],
            raw_dict["sol2"][i],
            raw_dict["sol3"][i],
            raw_dict["sol4"][i],
        ]

        tc_prompt = (
            f"{raw_dict['question'][i]} :\n\n"
            f"الإجابة 1: {raw_dict['sol1'][i]}\n\n"
            f"الإجابة 2: {raw_dict['sol2'][i]}\n\n"
            f"الإجابة 3: {raw_dict['sol3'][i]}\n\n"
            f"الإجابة 4: {raw_dict['sol4'][i]}\n\n"
            f"أجب بالرقم المفرد 1. أو 2. أو 3. أو 4."
            f"\n\n"
        )

        tc_expected_answer = f"{label_int + 1}. {options[label_int]}"

        # print("=" * 50)
        # print(f"->>>{tc_prompt}<<<-")
        # print(f"=>>>{tc_expected_answer}<<<=")

        test_suite.add_test_case(
            testing.RagTestCaseConfig(
                prompt=tc_prompt,
                categories=["question-answering"],
                expected_output=tc_expected_answer,
                constraints=[],
                condition=f'"{label_int + 1}"',
                config=test,
            )
        )

    #
    # THEN
    #

    test_suite_path = pathlib.Path(tmp_path) / "arabic_mmlu_test_suite.json"
    with open(test_suite_path, "w", encoding="utf-8") as f:
        json.dump(test_suite.to_dict(), f, ensure_ascii=False, indent=4)


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
