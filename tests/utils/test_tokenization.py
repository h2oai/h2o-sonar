# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json

import pytest

from h2o_sonar.utils import tokenization


@pytest.mark.h2o_sonar
@pytest.mark.generative
@pytest.mark.parametrize(
    "text, fragments, expected",
    [
        # no fragments
        (
            "This is TOXIC text with UGLY words.",
            [],
            {},
        ),
        # no overlapping fragments
        (
            "This is TOXIC text with UGLY words.",
            [(8, 13, {"m": 1.0}), (24, 28, {"m": 2.0})],
            {},
        ),
    ],
)
def test_fragments_to_tokenization(text: str, fragments: list, expected: dict):
    #
    # GIVEN
    #

    #
    # WHEN
    #
    t = tokenization.Tokenization.from_text_fragments(
        tokenization_type=tokenization.TOKENIZATION_TYPE_F,
        text=text,
        fragments=fragments,
    )

    #
    # THEN
    #

    print(f"Text: {text}")
    print(f"Fragments: {fragments}")
    print("Tokenization:")
    print(json.dumps(t.to_dict(), indent=2))


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
