# Copyright 2018-2025 H2O.ai, Inc. All rights reserved.
"""Tests for structured generation using ask_model_structured method."""

import pydantic
import pytest

from h2o_sonar import loggers
from h2o_sonar.lib.integrations import genai
from tests import test_utils
from tests.lib import given_generative


# ==============================================================================
# Pydantic Models for Testing
# ==============================================================================


class Person(pydantic.BaseModel):
    """Simple structured output model for testing."""

    name: str
    age: int
    city: str


class MathAnswer(pydantic.BaseModel):
    """Structured output model for math problems."""

    answer: int
    explanation: str


class Address(pydantic.BaseModel):
    """Nested model for testing complex structures."""

    model_config = pydantic.ConfigDict(populate_by_name=True)

    street: str
    city: str
    country: str
    postal_code: str = pydantic.Field(alias="postalCode")


class PersonWithAddress(pydantic.BaseModel):
    """Complex structured output model with nested objects."""

    name: str
    age: int
    address: Address
    hobbies: list[str]


class Product(pydantic.BaseModel):
    """Model for testing list/array handling."""

    name: str
    price: float
    in_stock: bool | None = None  # optional as some models may omit it
    tags: list[str]


class ProductList(pydantic.BaseModel):
    """Model for testing list of objects."""

    products: list[Product]
    total_count: int


class SentimentAnalysis(pydantic.BaseModel):
    """Model for testing sentiment analysis scenario."""

    sentiment: str  # positive, negative, neutral
    confidence: float  # 0.0 to 1.0
    key_phrases: list[str]


# ==============================================================================
# Test Client Factories
# ==============================================================================


def get_h2ogpte_client():
    """Get h2oGPTe client with appropriate model."""
    client = genai.H2oGpteRagClient(
        connection=test_utils.health.get_h2ogpte(),
        logger=loggers.SonarPrintLogger(),
    )
    llm_model_names = client.list_llm_model_names()
    if given_generative.LLM_CLAUDE_SONNET in llm_model_names:
        llm_model = given_generative.LLM_CLAUDE_SONNET
    else:
        llm_model = llm_model_names[0]
    return client, llm_model


def get_openai_client():
    """Get OpenAI client configured for structured output."""
    return (
        genai.OpenAiLlmClient(
            connection=given_generative.OPENAI_LLM,
            default_llm_model_name="gpt-4o",
            logger=loggers.SonarPrintLogger(),
        ),
        "gpt-4o",
    )


def get_azure_openai_client():
    """Get Azure OpenAI client configured for structured output."""
    return (
        genai.MsAzureOpenAiLlmClient(
            connection=given_generative.AZURE_OPENAI_LLM,
            api_version="2024-08-01-preview",
            logger=loggers.SonarPrintLogger(),
        ),
        None,
    )


def get_anthropic_client():
    """Get Anthropic client."""
    return (
        genai.AnthropicClaudeLlmClient(
            connection=given_generative.ANTHROPIC_LLM,
            default_llm_model_name=genai.AnthropicClaudeLlmClient.DEFAULT_LLM_MODEL,
            logger=loggers.SonarPrintLogger(),
        ),
        genai.AnthropicClaudeLlmClient.DEFAULT_LLM_MODEL,
    )


def get_ollama_client():
    """Get Ollama client with first available model."""
    client = genai.OllamaClient(
        connection=test_utils.health.get_ollama(),
        logger=loggers.SonarPrintLogger(),
    )
    llm_model_names = client.list_llm_model_names()
    if not llm_model_names:
        pytest.skip("No Ollama models available")
    return client, llm_model_names[0]


def get_bedrock_client():
    """Get Amazon Bedrock client with first available model."""
    client = genai.AmazonBedrockRagClient(
        connection=given_generative.AMAZON_BEDROCK,
        logger=loggers.SonarPrintLogger(),
    )
    llm_model_names = client.list_llm_model_names()
    if not llm_model_names:
        pytest.skip("No Amazon Bedrock models available")
    return client, llm_model_names[0]


def get_openai_assistants_client():
    """Get OpenAI Assistants client configured for structured output."""
    return (
        genai.OpenAiAssistantsRagClient(
            connection=given_generative.OPENAI_RAG,
            default_llm_model_name="gpt-4o",
            logger=loggers.SonarPrintLogger(),
        ),
        "gpt-4o",
    )


# ==============================================================================
# Simple Structure Tests - Person Model
# ==============================================================================


@pytest.mark.skipif(
    not test_utils.health.is_h2ogpte(),
    reason="h2oGPTe connection not available",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_h2ogpte_simple_structure():
    """Test h2oGPTe with simple Person structure."""
    # GIVEN
    client, llm_model = get_h2ogpte_client()
    prompts = ["Create a person named Alice who is 30 years old and lives in New York"]

    # WHEN
    results = client.ask_model_structured(
        prompts=prompts,
        output_structure=Person,
        llm_model_name=llm_model,
    )

    # THEN
    assert len(results) == 1
    assert isinstance(results[0], Person)
    assert results[0].name == "Alice"
    assert results[0].age == 30
    assert results[0].city == "New York"
    print(f"Results: {results}")


@pytest.mark.skipif(
    not test_utils.health.is_openai(),
    reason="OpenAI connection not available",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_openai_simple_structure():
    """Test OpenAI with simple Person structure."""
    # GIVEN
    client, _ = get_openai_client()
    prompts = ["Create a person named Bob who is 25 years old and lives in London"]

    # WHEN
    results = client.ask_model_structured(
        prompts=prompts,
        output_structure=Person,
    )

    # THEN
    assert len(results) == 1
    assert isinstance(results[0], Person)
    assert results[0].name == "Bob"
    assert results[0].age == 25
    assert results[0].city == "London"
    print(f"Results: {results}")


@pytest.mark.skipif(
    not test_utils.health.is_azure_openai(),
    reason="Azure OpenAI connection not available",
)
@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_azure_openai_simple_structure():
    """Test Azure OpenAI with simple Person structure."""
    # GIVEN
    client, _ = get_azure_openai_client()
    prompts = ["Create a person named Charlie who is 35 years old and lives in Paris"]

    # WHEN
    results = client.ask_model_structured(
        prompts=prompts,
        output_structure=Person,
    )

    # THEN
    assert len(results) == 1
    assert isinstance(results[0], Person)
    assert results[0].name == "Charlie"
    assert results[0].age == 35
    assert results[0].city == "Paris"
    print(f"Results: {results}")


@pytest.mark.skipif(
    not test_utils.health.is_anthropic(),
    reason="Anthropic connection not available",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_anthropic_simple_structure():
    """Test Anthropic with simple Person structure."""
    # GIVEN
    client, _ = get_anthropic_client()
    prompts = ["Create a person named Diana who is 28 years old and lives in Tokyo"]

    # WHEN
    results = client.ask_model_structured(
        prompts=prompts,
        output_structure=Person,
    )

    # THEN
    assert len(results) == 1
    assert isinstance(results[0], Person)
    assert results[0].name == "Diana"
    assert results[0].age == 28
    assert results[0].city == "Tokyo"
    print(f"Results: {results}")


@pytest.mark.skipif(
    not test_utils.health.is_ollama(),
    reason="Ollama connection not available",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_ollama_simple_structure():
    """Test Ollama with simple Person structure."""
    # GIVEN
    client, llm_model = get_ollama_client()
    prompts = ["Create a person named Eve who is 32 years old and lives in Berlin"]

    # WHEN
    results = client.ask_model_structured(
        prompts=prompts,
        output_structure=Person,
        llm_model_name=llm_model,
    )

    # THEN
    assert len(results) == 1
    assert isinstance(results[0], Person)
    assert results[0].name == "Eve"
    assert results[0].age == 32
    assert results[0].city == "Berlin"
    print(f"Results: {results}")


@pytest.mark.skipif(
    not test_utils.health.is_bedrock(),
    reason="Amazon Bedrock connection not available",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_bedrock_simple_structure():
    """Test Amazon Bedrock with simple Person structure."""
    # GIVEN
    client, llm_model = get_bedrock_client()
    prompts = ["Create a person named Frank who is 40 years old and lives in Sydney"]

    # WHEN
    try:
        results = client.ask_model_structured(
            prompts=prompts,
            output_structure=Person,
            llm_model_name=llm_model,
        )
    except ValueError as e:
        if "guardrail" in str(e).lower():
            pytest.skip(f"Bedrock guardrails blocking request: {e}")
        raise

    # THEN
    assert len(results) == 1
    assert isinstance(results[0], Person)
    assert results[0].name == "Frank"
    assert results[0].age == 40
    assert results[0].city == "Sydney"
    print(f"Results: {results}")


@pytest.mark.skipif(
    not test_utils.health.is_openai(),
    reason="OpenAI connection not available",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_openai_assistants_simple_structure():
    """Test OpenAI Assistants with simple Person structure."""
    # GIVEN
    client, _ = get_openai_assistants_client()
    prompts = ["Create a person named Grace who is 27 years old and lives in Rome"]

    # WHEN
    results = client.ask_model_structured(
        prompts=prompts,
        output_structure=Person,
    )

    # THEN
    assert len(results) == 1
    assert isinstance(results[0], Person)
    assert results[0].name == "Grace"
    assert results[0].age == 27
    assert results[0].city == "Rome"
    print(f"Results: {results}")


# ==============================================================================
# Math/Reasoning Tests - MathAnswer Model
# ==============================================================================


@pytest.mark.skipif(
    not test_utils.health.is_h2ogpte(),
    reason="h2oGPTe connection not available",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_h2ogpte_math_reasoning():
    """Test h2oGPTe with math reasoning."""
    # GIVEN
    client, llm_model = get_h2ogpte_client()
    prompts = ["What is 15 + 27? Provide the answer and a brief explanation."]

    # WHEN
    results = client.ask_model_structured(
        prompts=prompts,
        output_structure=MathAnswer,
        llm_model_name=llm_model,
    )

    # THEN
    assert len(results) == 1
    assert isinstance(results[0], MathAnswer)
    assert results[0].answer == 42
    assert results[0].explanation
    assert len(results[0].explanation) > 0
    print(f"Results: {results}")


@pytest.mark.skipif(
    not test_utils.health.is_openai(),
    reason="OpenAI connection not available",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_openai_math_reasoning():
    """Test OpenAI with math reasoning."""
    # GIVEN
    client, _ = get_openai_client()
    prompts = ["What is 20 + 22? Provide the answer and a brief explanation."]

    # WHEN
    results = client.ask_model_structured(
        prompts=prompts,
        output_structure=MathAnswer,
    )

    # THEN
    assert len(results) == 1
    assert isinstance(results[0], MathAnswer)
    assert results[0].answer == 42
    assert results[0].explanation
    assert len(results[0].explanation) > 0
    print(f"Results: {results}")


@pytest.mark.skipif(
    not test_utils.health.is_azure_openai(),
    reason="Azure OpenAI connection not available",
)
@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_azure_openai_math_reasoning():
    """Test Azure OpenAI with math reasoning."""
    # GIVEN
    client, _ = get_azure_openai_client()
    prompts = ["What is 10 + 32? Provide the answer and a brief explanation."]

    # WHEN
    results = client.ask_model_structured(
        prompts=prompts,
        output_structure=MathAnswer,
    )

    # THEN
    assert len(results) == 1
    assert isinstance(results[0], MathAnswer)
    assert results[0].answer == 42
    assert results[0].explanation
    assert len(results[0].explanation) > 0
    print(f"Results: {results}")


@pytest.mark.skipif(
    not test_utils.health.is_anthropic(),
    reason="Anthropic connection not available",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_anthropic_math_reasoning():
    """Test Anthropic with math reasoning."""
    # GIVEN
    client, _ = get_anthropic_client()
    prompts = ["What is 30 + 12? Provide the answer and a brief explanation."]

    # WHEN
    results = client.ask_model_structured(
        prompts=prompts,
        output_structure=MathAnswer,
    )

    # THEN
    assert len(results) == 1
    assert isinstance(results[0], MathAnswer)
    assert results[0].answer == 42
    assert results[0].explanation
    assert len(results[0].explanation) > 0
    print(f"Results: {results}")


@pytest.mark.skipif(
    not test_utils.health.is_ollama(),
    reason="Ollama connection not available",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_ollama_math_reasoning():
    """Test Ollama with math reasoning."""
    # GIVEN
    client, llm_model = get_ollama_client()
    prompts = ["What is 25 + 17? Provide the answer and a brief explanation."]

    # WHEN
    results = client.ask_model_structured(
        prompts=prompts,
        output_structure=MathAnswer,
        llm_model_name=llm_model,
    )

    # THEN
    assert len(results) == 1
    assert isinstance(results[0], MathAnswer)
    assert results[0].answer == 42
    assert results[0].explanation
    assert len(results[0].explanation) > 0
    print(f"Results: {results}")


@pytest.mark.skipif(
    not test_utils.health.is_bedrock(),
    reason="Amazon Bedrock connection not available",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_bedrock_math_reasoning():
    """Test Amazon Bedrock with math reasoning."""
    # GIVEN
    client, llm_model = get_bedrock_client()
    prompts = ["What is 35 + 7? Provide the answer and a brief explanation."]

    # WHEN
    try:
        results = client.ask_model_structured(
            prompts=prompts,
            output_structure=MathAnswer,
            llm_model_name=llm_model,
        )
    except ValueError as e:
        if "guardrail" in str(e).lower():
            pytest.skip(f"Bedrock guardrails blocking request: {e}")
        raise

    # THEN
    assert len(results) == 1
    assert isinstance(results[0], MathAnswer)
    assert results[0].answer == 42
    assert results[0].explanation
    assert len(results[0].explanation) > 0
    print(f"Results: {results}")


@pytest.mark.skipif(
    not test_utils.health.is_openai(),
    reason="Either OpenAI API key not set or OpenAI Python package is not installed",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_openai_assistants_math_reasoning():
    """Test OpenAI Assistants with math reasoning."""
    # GIVEN
    client, _ = get_openai_assistants_client()
    prompts = ["What is 40 + 2? Provide the answer and a brief explanation."]

    # WHEN
    results = client.ask_model_structured(
        prompts=prompts,
        output_structure=MathAnswer,
    )

    # THEN
    assert len(results) == 1
    assert isinstance(results[0], MathAnswer)
    assert results[0].answer == 42
    assert results[0].explanation
    assert len(results[0].explanation) > 0
    print(f"Results: {results}")


# ==============================================================================
# Complex Nested Structure Tests - PersonWithAddress Model
# ==============================================================================


@pytest.mark.skipif(
    not test_utils.health.is_h2ogpte(),
    reason="h2oGPTe connection not available",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_h2ogpte_complex_nested():
    """Test h2oGPTe with complex nested structure."""
    # GIVEN
    client, llm_model = get_h2ogpte_client()
    prompts = [
        (
            "Create a person named Henry who is 45 years old, "
            "lives at 456 Oak Avenue in Madrid, Spain with "
            "postal code 28001, and has hobbies: painting, cycling, and photography"
        )
    ]

    # WHEN
    results = client.ask_model_structured(
        prompts=prompts,
        output_structure=PersonWithAddress,
        llm_model_name=llm_model,
    )

    # THEN
    assert len(results) == 1
    assert isinstance(results[0], PersonWithAddress)
    assert results[0].name == "Henry"
    assert results[0].age == 45
    assert isinstance(results[0].address, Address)
    # flexible street check - model may omit number or reformat
    assert (
        "Oak Avenue" in results[0].address.street or "Oak" in results[0].address.street
    )
    assert results[0].address.city == "Madrid"
    assert results[0].address.country == "Spain"
    assert results[0].address.postal_code == "28001"
    assert isinstance(results[0].hobbies, list)
    assert len(results[0].hobbies) == 3
    assert "painting" in results[0].hobbies
    assert "cycling" in results[0].hobbies
    assert "photography" in results[0].hobbies
    print(f"Results: {results}")


@pytest.mark.skipif(
    not test_utils.health.is_openai(),
    reason="OpenAI connection not available",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_openai_complex_nested():
    """Test OpenAI with complex nested structure."""
    # GIVEN
    client, _ = get_openai_client()
    prompts = [
        (
            "Create a person named Isabel who is 38 years old, "
            "lives at 789 Pine Street in Toronto, Canada with "
            "postal code M5H 2N2, and has hobbies: tennis, cooking, and gardening"
        )
    ]

    # WHEN
    results = client.ask_model_structured(
        prompts=prompts,
        output_structure=PersonWithAddress,
    )

    # THEN
    assert len(results) == 1
    assert isinstance(results[0], PersonWithAddress)
    assert results[0].name == "Isabel"
    assert results[0].age == 38
    assert isinstance(results[0].address, Address)
    assert results[0].address.street == "789 Pine Street"
    assert results[0].address.city == "Toronto"
    assert results[0].address.country == "Canada"
    assert results[0].address.postal_code == "M5H 2N2"
    assert isinstance(results[0].hobbies, list)
    assert len(results[0].hobbies) == 3
    print(f"Results: {results}")


@pytest.mark.skipif(
    not test_utils.health.is_azure_openai(),
    reason="Azure OpenAI connection not available",
)
@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_azure_openai_complex_nested():
    """Test Azure OpenAI with complex nested structure."""
    # GIVEN
    client, _ = get_azure_openai_client()
    prompts = [
        (
            "Create a person named Jack who is 42 years old, "
            "lives at 321 Elm Road in Amsterdam, Netherlands with "
            "postal code 1012, and has hobbies: sailing, music, and reading"
        )
    ]

    # WHEN
    results = client.ask_model_structured(
        prompts=prompts,
        output_structure=PersonWithAddress,
    )

    # THEN
    assert len(results) == 1
    assert isinstance(results[0], PersonWithAddress)
    assert results[0].name == "Jack"
    assert results[0].age == 42
    assert isinstance(results[0].address, Address)
    assert results[0].address.street == "321 Elm Road"
    assert results[0].address.city == "Amsterdam"
    assert results[0].address.country == "Netherlands"
    assert results[0].address.postal_code == "1012"
    assert isinstance(results[0].hobbies, list)
    assert len(results[0].hobbies) == 3
    print(f"Results: {results}")


@pytest.mark.skipif(
    not test_utils.health.is_anthropic(),
    reason="Anthropic connection not available",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_anthropic_complex_nested():
    """Test Anthropic with complex nested structure."""
    # GIVEN
    client, _ = get_anthropic_client()
    prompts = [
        (
            "Create a person named Kate who is 33 years old, "
            "lives at 654 Maple Drive in Stockholm, Sweden with "
            "postal code 111 22, and has hobbies: skiing, yoga, and writing"
        )
    ]

    # WHEN
    results = client.ask_model_structured(
        prompts=prompts,
        output_structure=PersonWithAddress,
    )

    # THEN
    assert len(results) == 1
    assert isinstance(results[0], PersonWithAddress)
    assert results[0].name == "Kate"
    assert results[0].age == 33
    assert isinstance(results[0].address, Address)
    assert results[0].address.street == "654 Maple Drive"
    assert results[0].address.city == "Stockholm"
    assert results[0].address.country == "Sweden"
    assert results[0].address.postal_code == "111 22"
    assert isinstance(results[0].hobbies, list)
    assert len(results[0].hobbies) == 3
    print(f"Results: {results}")


@pytest.mark.skipif(
    not test_utils.health.is_ollama(),
    reason="Ollama connection not available",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_ollama_complex_nested():
    """Test Ollama with complex nested structure."""
    # GIVEN
    client, llm_model = get_ollama_client()
    prompts = [
        (
            "Create a person named Leo who is 36 years old, "
            "lives at 147 Cedar Lane in Oslo, Norway with "
            "postal code 0150, and has hobbies: hiking, chess, and coding"
        )
    ]

    # WHEN
    results = client.ask_model_structured(
        prompts=prompts,
        output_structure=PersonWithAddress,
        llm_model_name=llm_model,
    )

    # THEN
    assert len(results) == 1
    assert isinstance(results[0], PersonWithAddress)
    assert results[0].name == "Leo"
    assert results[0].age == 36
    assert isinstance(results[0].address, Address)
    # flexible street check - model may omit number or reformat
    assert "Cedar Lane" in results[0].address.street
    assert results[0].address.city == "Oslo"
    assert results[0].address.country == "Norway"
    assert results[0].address.postal_code == "0150"
    assert isinstance(results[0].hobbies, list)
    assert len(results[0].hobbies) == 3
    assert "hiking" in results[0].hobbies
    assert "chess" in results[0].hobbies
    assert "coding" in results[0].hobbies
    print(f"Results: {results}")


@pytest.mark.skipif(
    not test_utils.health.is_bedrock(),
    reason="Amazon Bedrock connection not available",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_bedrock_complex_nested():
    """Test Amazon Bedrock with complex nested structure."""
    # GIVEN
    client, llm_model = get_bedrock_client()
    prompts = [
        (
            "Create a person named Maria who is 39 years old, "
            "lives at 258 Birch Street in Lisbon, Portugal with "
            "postal code 1200, and has hobbies: dancing, travel, and drawing"
        )
    ]

    # WHEN
    try:
        results = client.ask_model_structured(
            prompts=prompts,
            output_structure=PersonWithAddress,
            llm_model_name=llm_model,
        )
    except ValueError as e:
        if "guardrail" in str(e).lower():
            pytest.skip(f"Bedrock guardrails blocking request: {e}")
        raise

    # THEN
    assert len(results) == 1
    assert isinstance(results[0], PersonWithAddress)
    assert results[0].name == "Maria"
    assert results[0].age == 39
    assert isinstance(results[0].address, Address)
    assert results[0].address.street == "258 Birch Street"
    assert results[0].address.city == "Lisbon"
    assert results[0].address.country == "Portugal"
    assert results[0].address.postal_code == "1200"
    assert isinstance(results[0].hobbies, list)
    assert len(results[0].hobbies) == 3
    print(f"Results: {results}")


@pytest.mark.skipif(
    not test_utils.health.is_openai(),
    reason="Either OpenAI API key not set or OpenAI Python package is not installed",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_openai_assistants_complex_nested():
    """Test OpenAI Assistants with complex nested structure."""
    # GIVEN
    client, _ = get_openai_assistants_client()
    prompts = [
        (
            "Create a person named Nina who is 31 years old, "
            "lives at 369 Willow Way in Dublin, Ireland with "
            "postal code D02, and has hobbies: running, baking, and languages"
        )
    ]

    # WHEN
    results = client.ask_model_structured(
        prompts=prompts,
        output_structure=PersonWithAddress,
    )

    # THEN
    assert len(results) == 1
    assert isinstance(results[0], PersonWithAddress)
    assert results[0].name == "Nina"
    assert results[0].age == 31
    assert isinstance(results[0].address, Address)
    assert results[0].address.street == "369 Willow Way"
    assert results[0].address.city == "Dublin"
    assert results[0].address.country == "Ireland"
    assert results[0].address.postal_code == "D02"
    assert isinstance(results[0].hobbies, list)
    assert len(results[0].hobbies) == 3
    print(f"Results: {results}")


# ==============================================================================
# List/Array Handling Tests - Product Model
# ==============================================================================


@pytest.mark.skipif(
    not test_utils.health.is_h2ogpte(),
    reason="h2oGPTe connection not available",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_h2ogpte_list_handling():
    """Test h2oGPTe with list/array handling."""
    # GIVEN
    client, llm_model = get_h2ogpte_client()
    prompts = [
        (
            "Create a product named Laptop with price 999.99, in stock true, "
            "and tags: electronics, computers, portable"
        )
    ]

    # WHEN
    results = client.ask_model_structured(
        prompts=prompts,
        output_structure=Product,
        llm_model_name=llm_model,
    )

    # THEN
    assert len(results) == 1
    assert isinstance(results[0], Product)
    assert results[0].name == "Laptop"
    assert results[0].price == 999.99
    # in_stock is optional - some models may omit it
    if results[0].in_stock is not None:
        assert results[0].in_stock is True
    assert isinstance(results[0].tags, list)
    assert len(results[0].tags) == 3
    assert "electronics" in results[0].tags
    assert "computers" in results[0].tags
    assert "portable" in results[0].tags
    print(f"Results: {results}")


@pytest.mark.skipif(
    not test_utils.health.is_openai(),
    reason="OpenAI connection not available",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_openai_list_handling():
    """Test OpenAI with list/array handling."""
    # GIVEN
    client, _ = get_openai_client()
    prompts = [
        (
            "Create a product named Smartphone with price 699.99, in stock true, "
            "and tags: electronics, mobile, 5G"
        )
    ]

    # WHEN
    results = client.ask_model_structured(
        prompts=prompts,
        output_structure=Product,
    )

    # THEN
    assert len(results) == 1
    assert isinstance(results[0], Product)
    assert results[0].name == "Smartphone"
    assert results[0].price == 699.99
    assert results[0].in_stock is True
    assert isinstance(results[0].tags, list)
    assert len(results[0].tags) == 3
    print(f"Results: {results}")


@pytest.mark.skipif(
    not test_utils.health.is_azure_openai(),
    reason="Azure OpenAI connection not available",
)
@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_azure_openai_list_handling():
    """Test Azure OpenAI with list/array handling."""
    # GIVEN
    client, _ = get_azure_openai_client()
    prompts = [
        (
            "Create a product named Tablet with price 499.99, in stock false, "
            "and tags: electronics, portable, touchscreen"
        )
    ]

    # WHEN
    results = client.ask_model_structured(
        prompts=prompts,
        output_structure=Product,
    )

    # THEN
    assert len(results) == 1
    assert isinstance(results[0], Product)
    assert results[0].name == "Tablet"
    assert results[0].price == 499.99
    assert results[0].in_stock is False
    assert isinstance(results[0].tags, list)
    assert len(results[0].tags) == 3
    print(f"Results: {results}")


@pytest.mark.skipif(
    not test_utils.health.is_anthropic(),
    reason="Anthropic connection not available",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_anthropic_list_handling():
    """Test Anthropic with list/array handling."""
    # GIVEN
    client, _ = get_anthropic_client()
    prompts = [
        (
            "Create a product named Headphones with price 199.99, in stock true, "
            "and tags: audio, wireless, noise-canceling"
        )
    ]

    # WHEN
    results = client.ask_model_structured(
        prompts=prompts,
        output_structure=Product,
    )

    # THEN
    assert len(results) == 1
    assert isinstance(results[0], Product)
    assert results[0].name == "Headphones"
    assert results[0].price == 199.99
    assert results[0].in_stock is True
    assert isinstance(results[0].tags, list)
    assert len(results[0].tags) == 3
    print(f"Results: {results}")


@pytest.mark.skipif(
    not test_utils.health.is_ollama(),
    reason="Ollama connection not available",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_ollama_list_handling():
    """Test Ollama with list/array handling."""
    # GIVEN
    client, llm_model = get_ollama_client()
    prompts = [
        (
            "Create a product named Monitor with price 299.99, in stock true, "
            "and tags: display, 4K, gaming"
        )
    ]

    # WHEN
    results = client.ask_model_structured(
        prompts=prompts,
        output_structure=Product,
        llm_model_name=llm_model,
    )

    # THEN
    assert len(results) == 1
    assert isinstance(results[0], Product)
    assert results[0].name == "Monitor"
    assert results[0].price == 299.99
    assert results[0].in_stock is True
    assert isinstance(results[0].tags, list)
    assert len(results[0].tags) == 3
    print(f"Results: {results}")


@pytest.mark.skipif(
    not test_utils.health.is_bedrock(),
    reason="Amazon Bedrock connection not available",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_bedrock_list_handling():
    """Test Amazon Bedrock with list/array handling."""
    # GIVEN
    client, llm_model = get_bedrock_client()
    prompts = [
        (
            "Create a product named Keyboard with price 149.99, in stock true, "
            "and tags: input, mechanical, RGB"
        )
    ]

    # WHEN
    try:
        results = client.ask_model_structured(
            prompts=prompts,
            output_structure=Product,
            llm_model_name=llm_model,
        )
    except ValueError as e:
        if "guardrail" in str(e).lower():
            pytest.skip(f"Bedrock guardrails blocking request: {e}")
        raise

    # THEN
    assert len(results) == 1
    assert isinstance(results[0], Product)
    assert results[0].name == "Keyboard"
    assert results[0].price == 149.99
    # in_stock is optional - some models may omit it
    if results[0].in_stock is not None:
        assert results[0].in_stock is True
    assert isinstance(results[0].tags, list)
    assert len(results[0].tags) == 3
    print(f"Results: {results}")


@pytest.mark.skipif(
    not test_utils.health.is_openai(),
    reason="Either OpenAI API key not set or OpenAI Python package is not installed",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_openai_assistants_list_handling():
    """Test OpenAI Assistants with list/array handling."""
    # GIVEN
    client, _ = get_openai_assistants_client()
    prompts = [
        (
            "Create a product named Mouse with price 79.99, in stock true, "
            "and tags: input, wireless, ergonomic"
        )
    ]

    # WHEN
    results = client.ask_model_structured(
        prompts=prompts,
        output_structure=Product,
    )

    # THEN
    assert len(results) == 1
    assert isinstance(results[0], Product)
    assert results[0].name == "Mouse"
    assert results[0].price == 79.99
    assert results[0].in_stock is True
    assert isinstance(results[0].tags, list)
    assert len(results[0].tags) == 3
    print(f"Results: {results}")


# ==============================================================================
# Sentiment Analysis Tests
# ==============================================================================


@pytest.mark.skipif(
    not test_utils.health.is_h2ogpte(),
    reason="h2oGPTe connection not available",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_h2ogpte_sentiment_analysis():
    """Test h2oGPTe with sentiment analysis."""
    # GIVEN
    client, llm_model = get_h2ogpte_client()
    prompts = [
        (
            "Analyze the sentiment of this text: 'I absolutely love this product! "
            "It exceeded all my expectations and works perfectly.'"
        )
    ]

    # WHEN
    results = client.ask_model_structured(
        prompts=prompts,
        output_structure=SentimentAnalysis,
        llm_model_name=llm_model,
    )

    # THEN
    assert len(results) == 1
    assert isinstance(results[0], SentimentAnalysis)
    assert results[0].sentiment.lower() == "positive"
    assert 0.0 <= results[0].confidence <= 1.0
    assert isinstance(results[0].key_phrases, list)
    assert len(results[0].key_phrases) > 0
    print(f"Results: {results}")


@pytest.mark.skipif(
    not test_utils.health.is_openai(),
    reason="OpenAI connection not available",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_openai_sentiment_analysis():
    """Test OpenAI with sentiment analysis."""
    # GIVEN
    client, _ = get_openai_client()
    prompts = [
        (
            "Analyze the sentiment of this text: 'This is terrible. "
            "It broke after one day and customer service was unhelpful.'"
        )
    ]

    # WHEN
    results = client.ask_model_structured(
        prompts=prompts,
        output_structure=SentimentAnalysis,
    )

    # THEN
    assert len(results) == 1
    assert isinstance(results[0], SentimentAnalysis)
    assert results[0].sentiment.lower() == "negative"
    assert 0.0 <= results[0].confidence <= 1.0
    assert isinstance(results[0].key_phrases, list)
    assert len(results[0].key_phrases) > 0
    print(f"Results: {results}")


@pytest.mark.skipif(
    not test_utils.health.is_azure_openai(),
    reason="Azure OpenAI connection not available",
)
@pytest.mark.skipif(
    not given_generative.is_config(), reason="Test services config not available"
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_azure_openai_sentiment_analysis():
    """Test Azure OpenAI with sentiment analysis."""
    # GIVEN
    client, _ = get_azure_openai_client()
    prompts = [
        (
            "Analyze the sentiment of this text: "
            "'The product is okay, nothing special but does the job.'"
        )
    ]

    # WHEN
    results = client.ask_model_structured(
        prompts=prompts,
        output_structure=SentimentAnalysis,
    )

    # THEN
    assert len(results) == 1
    assert isinstance(results[0], SentimentAnalysis)
    assert results[0].sentiment.lower() == "neutral"
    assert 0.0 <= results[0].confidence <= 1.0
    assert isinstance(results[0].key_phrases, list)
    assert len(results[0].key_phrases) > 0
    print(f"Results: {results}")


@pytest.mark.skipif(
    not test_utils.health.is_anthropic(),
    reason="Anthropic connection not available",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_anthropic_sentiment_analysis():
    """Test Anthropic with sentiment analysis."""
    # GIVEN
    client, _ = get_anthropic_client()
    prompts = [
        (
            "Analyze the sentiment of this text: "
            "'Great value for money! Highly recommended to everyone.'"
        )
    ]

    # WHEN
    results = client.ask_model_structured(
        prompts=prompts,
        output_structure=SentimentAnalysis,
    )

    # THEN
    assert len(results) == 1
    assert isinstance(results[0], SentimentAnalysis)
    assert results[0].sentiment.lower() == "positive"
    assert 0.0 <= results[0].confidence <= 1.0
    assert isinstance(results[0].key_phrases, list)
    assert len(results[0].key_phrases) > 0
    print(f"Results: {results}")


@pytest.mark.skipif(
    not test_utils.health.is_ollama(),
    reason="Ollama connection not available",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_ollama_sentiment_analysis():
    """Test Ollama with sentiment analysis."""
    # GIVEN
    client, llm_model = get_ollama_client()
    prompts = [
        (
            "Analyze the sentiment of this text: "
            "'Could be better. Some features are missing.'"
        )
    ]

    # WHEN
    results = client.ask_model_structured(
        prompts=prompts,
        output_structure=SentimentAnalysis,
        llm_model_name=llm_model,
    )

    # THEN
    assert len(results) == 1
    assert isinstance(results[0], SentimentAnalysis)
    assert results[0].sentiment.lower() in ["negative", "neutral"]
    assert 0.0 <= results[0].confidence <= 1.0
    assert isinstance(results[0].key_phrases, list)
    assert len(results[0].key_phrases) > 0
    print(f"Results: {results}")


@pytest.mark.skipif(
    not test_utils.health.is_bedrock(),
    reason="Amazon Bedrock connection not available",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_bedrock_sentiment_analysis():
    """Test Amazon Bedrock with sentiment analysis."""
    # GIVEN
    client, llm_model = get_bedrock_client()
    prompts = [
        (
            "Analyze the sentiment of this text: "
            "'Amazing experience! Will definitely buy again.'"
        )
    ]

    # WHEN
    results = client.ask_model_structured(
        prompts=prompts,
        output_structure=SentimentAnalysis,
        llm_model_name=llm_model,
    )

    # THEN
    assert len(results) == 1
    assert isinstance(results[0], SentimentAnalysis)
    assert results[0].sentiment.lower() == "positive"
    assert 0.0 <= results[0].confidence <= 1.0
    assert isinstance(results[0].key_phrases, list)
    assert len(results[0].key_phrases) > 0
    print(f"Results: {results}")


@pytest.mark.skipif(
    not test_utils.health.is_openai(),
    reason="Either OpenAI API key not set or OpenAI Python package is not installed",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_openai_assistants_sentiment_analysis():
    """Test OpenAI Assistants with sentiment analysis."""
    # GIVEN
    client, _ = get_openai_assistants_client()
    prompts = [
        (
            "Analyze the sentiment of this text: "
            "'Fantastic quality and fast shipping. Very satisfied!'"
        )
    ]

    # WHEN
    results = client.ask_model_structured(
        prompts=prompts,
        output_structure=SentimentAnalysis,
    )

    # THEN
    assert len(results) == 1
    assert isinstance(results[0], SentimentAnalysis)
    assert results[0].sentiment.lower() == "positive"
    assert 0.0 <= results[0].confidence <= 1.0
    assert isinstance(results[0].key_phrases, list)
    assert len(results[0].key_phrases) > 0
    print(f"Results: {results}")


# ==============================================================================
# Multi-Prompt Batch Tests
# ==============================================================================


@pytest.mark.skipif(
    not test_utils.health.is_h2ogpte(),
    reason="h2oGPTe connection not available",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_h2ogpte_batch_prompts():
    """Test h2oGPTe with multiple prompts in one call."""
    # GIVEN
    client, llm_model = get_h2ogpte_client()
    prompts = [
        "Create a person named Oliver who is 29 years old and lives in Vienna",
        "Create a person named Paula who is 34 years old and lives in Prague",
        "Create a person named Quinn who is 26 years old and lives in Brussels",
    ]

    # WHEN
    results = client.ask_model_structured(
        prompts=prompts,
        output_structure=Person,
        llm_model_name=llm_model,
    )

    # THEN
    assert len(results) == 3
    assert all(isinstance(r, Person) for r in results)
    assert results[0].name == "Oliver"
    assert results[0].age == 29
    assert results[1].name == "Paula"
    assert results[1].age == 34
    assert results[2].name == "Quinn"
    assert results[2].age == 26
    print(f"Results: {results}")


@pytest.mark.skipif(
    not test_utils.health.is_openai(),
    reason="OpenAI connection not available",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_openai_batch_prompts():
    """Test OpenAI with multiple prompts in one call."""
    # GIVEN
    client, _ = get_openai_client()
    prompts = [
        "What is 5 + 5? Provide the answer and a brief explanation.",
        "What is 7 + 3? Provide the answer and a brief explanation.",
    ]

    # WHEN
    results = client.ask_model_structured(
        prompts=prompts,
        output_structure=MathAnswer,
    )

    # THEN
    assert len(results) == 2
    assert all(isinstance(r, MathAnswer) for r in results)
    assert results[0].answer == 10
    assert results[1].answer == 10
    print(f"Results: {results}")


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
