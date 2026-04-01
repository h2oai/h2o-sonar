# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import random

import pytest

from h2o_sonar import config as h2o_sonar_config
from h2o_sonar.lib.api import commons
from h2o_sonar.utils import perturbations
from tests import test_utils


SEED = 0xCAFE  # random.randint(0, sys.maxsize)
print(f"SEED={SEED}")


@pytest.mark.h2o_sonar
@pytest.mark.generative
@pytest.mark.parametrize(
    "before_replacing,after_replacing,y_to_z,intensity,raised_errors",
    [
        (
            (
                "The zebra zigzagged zestfully through the zucchini zoo, "
                "zealously zapping zesty zucchini zappers."
            ),
            (
                "The yebra zigzagged yestfully through the zucchini zoo, "
                "yealously zapping zesty yucchini zappers."
            ),
            False,
            commons.PerturbationIntensity.LOW,
            None,
        ),
        (
            (
                "The zebra zigzagged zestfully through the zucchini zoo, "
                "zealously zapping zesty zucchini zappers."
            ),
            (
                "The zebra zigzagged zestfullz through the zucchini zoo, "
                "zealouslz zapping zestz zucchini zappers."
            ),
            True,
            commons.PerturbationIntensity.HIGH,
            None,
        ),
        # negative test case: no characters can be replaced
        ("Negative.", None, True, commons.PerturbationIntensity.HIGH, []),
    ],
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_qwerty(
    before_replacing: str,
    after_replacing: str,
    y_to_z: bool,
    intensity: commons.PerturbationIntensity,
    raised_errors: list | None,
):
    # GIVEN
    t_qwerty_perturbator = perturbations.QwertyPerturbator()

    # WHEN
    perturbated_text = t_qwerty_perturbator.perturb(
        text=before_replacing,
        intensity=intensity,
        y_to_z=y_to_z,
        raised_errors=raised_errors,
    )

    # THEN
    if raised_errors is None:
        assert perturbated_text == after_replacing
    else:
        print(f"Raised errors: {raised_errors}")
        assert raised_errors


@pytest.mark.h2o_sonar
@pytest.mark.generative
@pytest.mark.parametrize(
    "input_text,intensity",
    [
        (
            "The zebra zigzagged zestfully through the zoo.",
            commons.PerturbationIntensity.LOW,
        ),
        ("Hello, world!", commons.PerturbationIntensity.MEDIUM),
        ("Test 123", commons.PerturbationIntensity.HIGH),
        ("", commons.PerturbationIntensity.LOW),
    ],
)
def test_copy_perturbator(
    input_text: str,
    intensity: commons.PerturbationIntensity,
):
    """Test that CopyPerturbator returns input unchanged."""
    # GIVEN
    copy_perturbator = perturbations.CopyPerturbator()

    # WHEN
    perturbed_text = copy_perturbator.perturb(
        text=input_text,
        intensity=intensity,
    )

    # THEN
    assert perturbed_text == input_text, "CopyPerturbator should return input unchanged"


@pytest.mark.h2o_sonar
@pytest.mark.generative
@pytest.mark.parametrize(
    "before_replacing,after_replacing,intensity,raised_errors",
    [
        (
            (
                "The zebra zigzagged zestfully through the zucchini zoo "
                "zealously zapping zesty zucchini zappers."
            ),
            (
                "The, zebra zigzagged zestfully through the zucchini zoo "
                "zealously zapping zesty, zucchini zappers."
            ),
            commons.PerturbationIntensity.LOW,
            None,
        ),
        (
            (
                "The zebra zigzagged zestfully through the zucchini zoo "
                "zealously zapping zesty zucchini zappers."
            ),
            (
                "The, zebra zigzagged, zestfully through, the zucchini, "
                "zoo zealously, zapping zesty, zucchini zappers."
            ),
            commons.PerturbationIntensity.HIGH,
            None,
        ),
        # negative test case: no characters can be replaced
        ("Negative.", None, commons.PerturbationIntensity.HIGH, []),
    ],
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_comma_perturbator(
    before_replacing: str,
    after_replacing: str,
    intensity: commons.PerturbationIntensity,
    raised_errors: list | None,
):
    # GIVEN
    comma_perturbator = perturbations.CommaPerturbator()
    # WHEN
    perturbed_text = comma_perturbator.perturb(
        text=before_replacing, intensity=intensity, raised_errors=raised_errors
    )
    # THEN
    if raised_errors is None:
        assert len(perturbed_text.split(",")) > len(before_replacing.split(","))
        assert perturbed_text == after_replacing
    else:
        print(f"Raised errors: {raised_errors}")
        assert raised_errors


@pytest.mark.h2o_sonar
@pytest.mark.generative
@pytest.mark.parametrize(
    "before_replacing,after_replacing,intensity,raised_errors",
    [
        (
            (
                "The zebra zigzagged zestfully through the zucchini zoo "
                "zealously zapping zesty zucchini zappers."
            ),
            (
                "zebra The zigzagged zestfully through the zucchini zoo "
                "zealously zapping zucchini zesty zappers."
            ),
            commons.PerturbationIntensity.LOW,
            None,
        ),
        (
            (
                "The zebra zigzagged zestfully through the zucchini zoo "
                "zealously zapping zesty zucchini zappers."
            ),
            (
                "zebra The zestfully zigzagged the through zoo zucchini "
                "zapping zealously zucchini zesty zappers."
            ),
            commons.PerturbationIntensity.HIGH,
            None,
        ),
        # negative test case
        ("Negative.", None, commons.PerturbationIntensity.HIGH, []),
    ],
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_word_perturbator(
    before_replacing: str,
    after_replacing: str,
    intensity: commons.PerturbationIntensity,
    raised_errors: list | None,
):
    # GIVEN
    word_swap_perturbator = perturbations.WordSwapPerturbator()
    # WHEN
    perturbed_text = word_swap_perturbator.perturb(
        text=before_replacing, intensity=intensity, raised_errors=raised_errors
    )
    # THEN
    if raised_errors is None:
        assert perturbed_text == after_replacing
        assert before_replacing != perturbed_text
    else:
        print(f"Raised errors: {raised_errors}")
        assert raised_errors


@pytest.mark.h2o_sonar
@pytest.mark.generative
@pytest.mark.parametrize(
    "before_replacing,after_replacing,intensity,raised_errors",
    [
        (
            (
                "The , zebra, zigzagged, zestfully, through, the, zucchini, zoo, "
                "zealously, ,zapping ,zesty, zucchini ,zappers."
            ),
            (
                "The , zebra, zigzagged, zestily, through, the, zucchini, zoo, "
                "zealously, ,zapping ,barmy, zucchini ,zapper."
            ),
            commons.PerturbationIntensity.LOW,
            None,
        ),
        (
            (
                "Which of the following statements accurately describes the "
                "impact of Modified Newtonian Dynamics (MOND) on the observed "
                '"missing baryonic mass" discrepancy in galaxy clusters?'
            ),
            (
                "Which of the be affirmation accurately describes the impingement "
                'of Modified Newtonian Dynamics (MOND) on the respect "missing '
                'baryonic multitude" disagreement in extragalactic_nebula cluster?'
            ),
            commons.PerturbationIntensity.HIGH,
            None,
        ),
        # negative test case
        ("Negative.", None, commons.PerturbationIntensity.HIGH, []),
    ],
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_synonym_perturbator(
    before_replacing: str,
    after_replacing: str,
    intensity: commons.PerturbationIntensity,
    raised_errors: list | None,
):
    # GIVEN
    random.seed(SEED)
    synonym_perturbator = perturbations.SynonymPerturbator()
    # WHEN
    perturbed_text = synonym_perturbator.perturb(
        text=before_replacing, intensity=intensity, raised_errors=raised_errors
    )
    # THEN
    if raised_errors is None:
        assert before_replacing != perturbed_text
        # do NOT check exact match as the synonyms are randomly selected
        # assert after_replacing == perturbed_text
    else:
        print(f"Raised errors: {raised_errors}")
        assert raised_errors


@pytest.mark.h2o_sonar
@pytest.mark.generative
@pytest.mark.parametrize(
    "before_replacing,intensity,raised_errors",
    [
        (
            (
                " In botany, a tree is a perennial plant with an elongated stem, "
                "or trunk, usually supporting branches and leaves. In some usages, "
                "the definition of a tree may be narrower, including only woody plants "
                "with secondary growth, plants that are usable as lumber or plants "
                "above a specified height. "
            ),
            commons.PerturbationIntensity.LOW,
            None,
        ),
        (
            (
                "Keep your data isolated and secure. We handle provisioning and "
                "deployment built right within H2O GenAI App Store. Choose the most "
                "cost-effective models for your use case, ensure the safety of your "
                "data and create reusable components to scale application development."
            ),
            commons.PerturbationIntensity.HIGH,
            None,
        ),
        # negative test case
        ("Negative.", commons.PerturbationIntensity.HIGH, []),
    ],
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_antonym_perturbator(
    before_replacing: str,
    intensity: commons.PerturbationIntensity,
    raised_errors: list | None,
):
    # GIVEN
    random.seed(SEED)
    antonym_perturbator = perturbations.AntonymPerturbator()
    # WHEN
    perturbed_text = antonym_perturbator.perturb(
        text=before_replacing, intensity=intensity, raised_errors=raised_errors
    )
    # THEN
    if raised_errors is None:
        assert before_replacing != perturbed_text
    else:
        print(f"Raised errors: {raised_errors}")
        assert raised_errors


@pytest.mark.parametrize(
    "character_perturbator,intensity,threshold,input_text,raised_errors",
    [
        (
            perturbations.RandomCharacterInsertPerturbator(),
            commons.PerturbationIntensity.HIGH,
            -1.0,
            "The zebra zigzagged zestfully through the zucchini zoo.",
            None,
        ),
        (
            perturbations.RandomCharacterDeletePerturbator(),
            commons.PerturbationIntensity.VERY_LOW,
            0.06,
            "The zebra zigzagged zestfully through the zucchini zoo.",
            None,
        ),
        (
            perturbations.RandomCharacterReplacementPerturbator(),
            commons.PerturbationIntensity.LOW,
            0.1,
            "The zebra zigzagged zestfully through the zucchini zoo.",
            None,
        ),
        (
            perturbations.KeywordTyposCharacterPerturbator(),
            commons.PerturbationIntensity.MEDIUM,
            0.3,
            "The zebra zigzagged zestfully through the zucchini zoo.",
            None,
        ),
        (
            perturbations.RandomOCRCharacterPerturbator(),
            commons.PerturbationIntensity.LOW,
            0.1,
            "The zebra zigzagged zestfully through the zucchini zoo.",
            None,
        ),
        # negative test case
        (
            perturbations.RandomCharacterInsertPerturbator(),
            commons.PerturbationIntensity.LOW,
            -1.0,
            "",
            [],
        ),
    ],
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_character_perturbator(
    character_perturbator,
    intensity: commons.PerturbationIntensity,
    threshold: float,
    input_text: str,
    raised_errors: list | None,
):
    #
    # GIVEN
    #
    random.seed(SEED)

    def dif(x1, x2):
        return sum(x1 != x2 for x1, x2 in zip(x1, x2, strict=False)) / len(x1)

    #
    # WHEN
    #
    perturbed_text = character_perturbator.perturb(
        text=input_text,
        intensity=intensity,
        raised_errors=raised_errors,
    )

    #
    # THEN
    #
    if raised_errors is None:
        print(f"Random replacement: {perturbed_text}")
        assert perturbed_text != input_text
        if threshold > 0:
            assert dif(input_text, perturbed_text) <= threshold
    else:
        print(f"Raised errors: {raised_errors}")
        assert raised_errors


@pytest.mark.parametrize(
    "encoding_perturbator,input_text,output_text,prompt_type,answer_type,raised_errors",
    [
        (
            perturbations.EncodingPerturbatorBase16(),
            "The zebra zoo.",
            (
                "The following question is encoded in base16_spaces - "
                "answer in the same encoding: 54 68 65 20 7A 65 62 72 61 20 7A 6F 6F 2E"
            ),
            perturbations.EncodingPerturbator.TYPE_PROMPT_ENCODED,
            perturbations.EncodingPerturbator.TYPE_ANSWER_ENCODED,
            None,
        ),
        (
            perturbations.EncodingPerturbatorBase16(),
            "The zebra zoo.",
            (
                "The following question is encoded in base16_spaces - "
                "answer in normal text: 54 68 65 20 7A 65 62 72 61 20 7A 6F 6F 2E"
            ),
            perturbations.EncodingPerturbator.TYPE_PROMPT_ENCODED,
            perturbations.EncodingPerturbator.TYPE_ANSWER_DECODED,
            None,
        ),
        (
            perturbations.EncodingPerturbatorBase16(),
            "The zebra zoo.",
            (
                "The following question is in normal text - "
                "answer in base16_spaces encoding: The zebra zoo."
            ),
            perturbations.EncodingPerturbator.TYPE_PROMPT_DECODED,
            perturbations.EncodingPerturbator.TYPE_ANSWER_ENCODED,
            None,
        ),
        (
            perturbations.EncodingPerturbatorBase16(),
            "The zebra zoo.",
            "The zebra zoo.",
            perturbations.EncodingPerturbator.TYPE_PROMPT_DECODED,
            perturbations.EncodingPerturbator.TYPE_ANSWER_DECODED,
            [],
        ),
    ],
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_encoding_perturbator(
    encoding_perturbator,
    input_text: str,
    output_text: str,
    prompt_type: str,
    answer_type: str,
    raised_errors: list | None,
):
    #
    # GIVEN
    #
    random.seed(SEED)

    #
    # WHEN
    #
    perturbed_text = encoding_perturbator.perturb(
        text=input_text,
        raised_errors=raised_errors,
        prompt_type=prompt_type,
        answer_type=answer_type,
    )

    #
    # THEN
    #
    if raised_errors is None:
        print(f"random_replacement: {perturbed_text}")
        assert perturbed_text != input_text
    else:
        print(f"Raised errors: {raised_errors}")
        assert raised_errors


@pytest.mark.generative
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_diff_perturbed_prompts():
    import difflib

    # GIVEN
    text = (
        "Keep your data isolated and secure."
        "\nWe handle provisioning and deployment built right within H2O App Store."
    )
    perturbed_text = (
        "Keep your own data ioslated and secure."
        "\nWe handle provisooning deployment built right within H2O App Store."
    )

    # WHEN
    differ = difflib.Differ()
    text_diff = differ.compare(text.splitlines(), perturbed_text.splitlines())

    # THEN
    assert text_diff
    print("\nDiff result")
    print("#" * 80)
    for line in text_diff:
        if line.startswith("?"):
            print(line.strip())
        else:
            print(line)
    print("#" * 80)


@pytest.mark.skip(reason="This agent-based perturbator has been disabled")
@pytest.mark.skipif(
    test_utils.GitHubActions.is_in_gha(),
    reason="Skipped on GHA as this test fails due to h2oGPTe client & agent fragility",
)
@pytest.mark.parametrize(
    "configure_agent,text,intensity,raised_errors",
    [
        (
            True,
            "Is Paris the capital of France?",
            commons.PerturbationIntensity.LOW,
            None,
        ),
        # run just 1 test ^ as agents are slow and pricey
        # (
        #     True,
        #     "The capital of France is Paris.",
        #     commons.PerturbationIntensity.LOW,
        #     None,
        # ),
        # (
        #     "Frida Kahlo is painter.",
        #     commons.PerturbationIntensity.HIGH,
        #     [],
        # ),
        # (
        #     "Is Frida Kahlo painter?",
        #     commons.PerturbationIntensity.HIGH,
        #     [],
        # ),
    ],
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
@pytest.mark.generative
def test_agentic_contextual_misinformation(
    configure_agent: bool,
    text: str,
    intensity: commons.PerturbationIntensity,
    raised_errors: list | None,
):
    """Test agentic perturbator for contextual misinformation.

    Parameters
    ----------
    configure_agent : bool
        Whether to configure the agent connection. If ``False``, the test will fail,
        but use ``False`` with caution as the H2O Sonar configuration is global and
        tests may have side effects and impact other tests - especially in CI.
    text : str
        The input text to perturb.
    intensity : commons.PerturbationIntensity
        The perturbation intensity.
    raised_errors : list | None
        Whether to gather the list of raised errors during perturbation.

    """

    # GIVEN
    agentic_perturbator = perturbations.ContextualMisinformationPerturbator()

    if configure_agent:
        # agentic host connection
        agentic_host_connection = test_utils.health.get_h2ogpte()
        print(f"TEST will use agentic host: {agentic_host_connection}")
        agentic_host_connection_json = json.dumps(
            agentic_host_connection.to_dict(encrypt=False), indent=2
        )
        print(f"h2oGPTe agent host connection:\n\n{agentic_host_connection_json}\n")
        h2o_sonar_config.config.add_connection(agentic_host_connection)

    # WHEN
    perturbed_text = agentic_perturbator.perturb(
        text=text, intensity=intensity, raised_errors=raised_errors, retries=1
    )

    # THEN
    print(
        f"PERTURBATION by {agentic_perturbator._display_name}:"
        f"\n  IN : {text}"
        f"\n  OUT: {perturbed_text}"
    )
    if not configure_agent:
        assert raised_errors, (
            "As the agent connection is not configured, the test should fail, but "
            "it did not raised any errors."
        )
        assert perturbed_text is None, (
            "As the agent connection is not configured, the test should fail ~ "
            "perturbed_text should be None, but it is not."
        )
    if raised_errors is None:
        assert text != perturbed_text, (
            "The perturbed text should be different from the input text."
        )
    else:
        print(f"  Raised errors: {raised_errors}")
        assert raised_errors, "The test should have raised errors, but it did not."


@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
