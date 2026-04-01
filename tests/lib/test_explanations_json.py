# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
"""Tests for JSON diff functionality."""

import pytest

from h2o_sonar.lib.api.explanations import _explanations_diff_json


@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_json_comparator_identical_dicts():
    """Test JSONComparator with identical dictionaries."""
    #
    # GIVEN
    #
    dict1 = {"user": "Alice", "id": 123, "status": "active"}
    dict2 = {"user": "Alice", "id": 123, "status": "active"}
    comparator = _explanations_diff_json.JSONComparator(dict1, dict2)

    #
    # WHEN
    #
    has_diff = comparator.has_differences()
    summary = comparator.get_diff_summary()

    #
    # THEN
    #
    assert not has_diff, "Expected no differences for identical dictionaries"
    assert isinstance(summary, _explanations_diff_json.DiffSummary), (
        "Expected DiffSummary dataclass"
    )
    assert summary.values_changed == 0
    assert summary.dictionary_item_added == 0
    assert summary.dictionary_item_removed == 0
    assert summary.total_changes() == 0


@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_json_comparator_value_changed():
    """Test JSONComparator with changed values."""
    #
    # GIVEN
    #
    dict1 = {"user": "Alice", "id": 123, "status": "active"}
    dict2 = {"user": "Bob", "id": 123, "status": "active"}
    comparator = _explanations_diff_json.JSONComparator(dict1, dict2)

    #
    # WHEN
    #
    has_diff = comparator.has_differences()
    summary = comparator.get_diff_summary()
    changed_paths = comparator.get_changed_paths()

    #
    # THEN
    #
    assert has_diff, "Expected differences when values changed"
    assert summary.values_changed == 1
    assert summary.total_changes() == 1
    assert "root['user']" in changed_paths


@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_json_comparator_item_added():
    """Test JSONComparator with added dictionary items."""
    #
    # GIVEN
    #
    dict1 = {"user": "Alice", "id": 123}
    dict2 = {"user": "Alice", "id": 123, "status": "active"}
    comparator = _explanations_diff_json.JSONComparator(dict1, dict2)

    #
    # WHEN
    #
    has_diff = comparator.has_differences()
    summary = comparator.get_diff_summary()
    changed_paths = comparator.get_changed_paths()

    #
    # THEN
    #
    assert has_diff, "Expected differences when items added"
    assert summary.dictionary_item_added == 1
    assert summary.total_changes() == 1
    assert "root['status']" in changed_paths


@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_json_comparator_item_removed():
    """Test JSONComparator with removed dictionary items."""
    #
    # GIVEN
    #
    dict1 = {"user": "Alice", "id": 123, "status": "active"}
    dict2 = {"user": "Alice", "id": 123}
    comparator = _explanations_diff_json.JSONComparator(dict1, dict2)

    #
    # WHEN
    #
    has_diff = comparator.has_differences()
    summary = comparator.get_diff_summary()
    changed_paths = comparator.get_changed_paths()

    #
    # THEN
    #
    assert has_diff, "Expected differences when items removed"
    assert summary.dictionary_item_removed == 1
    assert summary.total_changes() == 1
    assert "root['status']" in changed_paths


@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_json_comparator_nested_changes():
    """Test JSONComparator with nested dictionary changes."""
    #
    # GIVEN
    #
    dict1 = {
        "user": "Alice",
        "settings": {"theme": "dark", "notifications": True, "language": "en"},
    }
    dict2 = {
        "user": "Alice",
        "settings": {
            "theme": "light",
            "notifications": True,
            "language": "en",
            "timeout": 300,
        },
    }
    comparator = _explanations_diff_json.JSONComparator(dict1, dict2)

    #
    # WHEN
    #
    has_diff = comparator.has_differences()
    summary = comparator.get_diff_summary()
    changed_paths = comparator.get_changed_paths()

    #
    # THEN
    #
    assert has_diff, "Expected differences in nested dictionaries"
    assert summary.values_changed == 1, "Expected theme value changed"
    assert summary.dictionary_item_added == 1, "Expected timeout added"
    assert summary.total_changes() == 2
    assert "root['settings']['theme']" in changed_paths
    assert "root['settings']['timeout']" in changed_paths


@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_json_comparator_list_changes():
    """Test JSONComparator with list changes."""
    #
    # GIVEN
    #
    dict1 = {"roles": ["admin", "editor", "viewer"]}
    dict2 = {"roles": ["editor", "admin", "viewer"]}
    comparator = _explanations_diff_json.JSONComparator(dict1, dict2)

    #
    # WHEN
    #
    has_diff = comparator.has_differences()

    #
    # THEN
    #
    # with ignore_order=False (default), list reordering is detected as difference
    assert has_diff, "Expected differences when list order changed"


@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_json_comparator_list_changes_ignore_order():
    """Test JSONComparator with list changes ignoring order."""
    #
    # GIVEN
    #
    dict1 = {"roles": ["admin", "editor", "viewer"]}
    dict2 = {"roles": ["editor", "admin", "viewer"]}
    comparator = _explanations_diff_json.JSONComparator(dict1, dict2, ignore_order=True)

    #
    # WHEN
    #
    has_diff = comparator.has_differences()

    #
    # THEN
    #
    # with ignore_order=True, list reordering is NOT detected as difference
    assert not has_diff, "Expected no differences when ignoring list order"


@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_json_comparator_complex_diff():
    """Test JSONComparator with complex nested structure changes."""
    #
    # GIVEN
    #
    dict1 = {
        "user": "Alice",
        "id": 123,
        "status": "active",
        "settings": {
            "theme": "dark",
            "notifications": True,
            "language": "en",
        },
        "roles": ["admin", "editor", "viewer"],
        "data_array": [
            {"key": 1, "val": "Data A"},
            {"key": 2, "val": "Data B"},
        ],
        "metadata": None,
    }

    dict2 = {
        "user": "Bob",
        "id": 123,
        "settings": {
            "theme": "light",
            "notifications": True,
            "language": "en",
            "timeout": 300,
        },
        "roles": ["editor", "admin", "viewer"],
        "data_array": [
            {"key": 1, "val": "Data A"},
        ],
        "metadata": None,
    }

    comparator = _explanations_diff_json.JSONComparator(dict1, dict2)

    #
    # WHEN
    #
    has_diff = comparator.has_differences()
    summary = comparator.get_diff_summary()
    diff_dict = comparator.to_dict()
    diff_json = comparator.to_json()

    #
    # THEN
    #
    print(diff_json)
    assert has_diff, "Expected differences in complex structure"
    assert summary.values_changed >= 1, "Expected at least one value changed"
    assert summary.dictionary_item_added >= 1, "Expected at least one item added"
    assert summary.dictionary_item_removed >= 1, "Expected at least one item removed"
    assert summary.total_changes() >= 3
    assert isinstance(diff_dict, dict), "Expected diff_dict to be a dictionary"
    assert isinstance(diff_json, str), "Expected diff_json to be a string"
    assert len(diff_json) > 0, "Expected non-empty JSON string"


@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_json_comparator_to_json():
    """Test JSONComparator JSON serialization."""
    #
    # GIVEN
    #
    dict1 = {"user": "Alice", "id": 123}
    dict2 = {"user": "Bob", "id": 456}
    comparator = _explanations_diff_json.JSONComparator(dict1, dict2)

    #
    # WHEN
    #
    json_output = comparator.to_json(indent=4)

    #
    # THEN
    #
    assert isinstance(json_output, str), "Expected JSON output to be string"
    assert '"values_changed"' in json_output, "Expected values_changed in JSON"
    assert "Alice" in json_output or "Bob" in json_output, "Expected values in JSON"


@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_json_comparator_empty_dicts():
    """Test JSONComparator with empty dictionaries."""
    #
    # GIVEN
    #
    dict1 = {}
    dict2 = {}
    comparator = _explanations_diff_json.JSONComparator(dict1, dict2)

    #
    # WHEN
    #
    has_diff = comparator.has_differences()
    summary = comparator.get_diff_summary()

    #
    # THEN
    #
    assert not has_diff, "Expected no differences for empty dictionaries"
    assert summary.values_changed == 0
    assert summary.dictionary_item_added == 0
    assert summary.total_changes() == 0


@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_json_comparator_model_configs():
    """Test JSONComparator with model configuration-like dictionaries."""
    #
    # GIVEN
    #
    baseline_cfg = {
        "temperature": 0.7,
        "max_tokens": 1000,
        "model_name": "gpt-3.5-turbo",
        "system_prompt": "You are a helpful assistant",
    }
    current_cfg = {
        "temperature": 0.9,
        "max_tokens": 1000,
        "model_name": "gpt-4",
        "system_prompt": "You are a helpful assistant",
        "top_p": 0.95,
    }
    comparator = _explanations_diff_json.JSONComparator(baseline_cfg, current_cfg)

    #
    # WHEN
    #
    has_diff = comparator.has_differences()
    summary = comparator.get_diff_summary()
    changed_paths = comparator.get_changed_paths()

    #
    # THEN
    #
    assert has_diff, "Expected differences in model configurations"
    assert summary.values_changed == 2, "Expected temperature and model_name changed"
    assert summary.dictionary_item_added == 1, "Expected top_p added"
    assert summary.total_changes() == 3
    assert "root['temperature']" in changed_paths
    assert "root['model_name']" in changed_paths
    assert "root['top_p']" in changed_paths


@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_json_cmp_mock():
    #
    # GIVEN
    #
    dict1 = {
        "system_prompt": (
            "You are a helpful AI assistant that provides accurate and informative "
            "responses based on the given context. Please analyze the provided "
            "documents and answer questions clearly and concisely."
        ),
        "llm_args": {
            "temperature": 0.9,
            "top_p": 0.7,
            "top_k": 1.0,
            "seed": 0,
            "use_agent": True,
        },
        "timeout": 600,
    }

    dict2 = {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "system_prompt": (
            "You are a helpful AI assistant that delivers accurate and informative "
            "responses based on the given context. Please analyze the provided "
            "documents and respond to questions clearly and concisely."
        ),
        "llm_args": {
            "temperature": 0.3,
            "top_p": 0.8,
            "top_k": 1.0,
            "repetition_penalty": 1.07,
            "max_new_tokens": 1024,
            "min_max_new_tokens": 512,
            "seed": 0,
        },
        "timeout": 600,
    }

    comparator = _explanations_diff_json.JSONComparator(dict1, dict2)

    #
    # WHEN
    #
    has_diff = comparator.has_differences()
    summary = comparator.get_diff_summary()
    diff_dict = comparator.to_dict()
    diff_json = comparator.to_json()

    #
    # THEN
    #
    print(diff_json)
    assert has_diff, "Expected differences in model configurations"
    assert summary.values_changed == 3, "Expected 3 values changed"
    assert summary.dictionary_item_added == 4, "Expected 4 items added"
    assert summary.dictionary_item_removed == 1, "Expected 1 item removed"
    assert summary.total_changes() == 8
    assert isinstance(diff_dict, dict), "Expected diff_dict to be a dictionary"
    assert isinstance(diff_json, str), "Expected diff_json to be a string"
    assert len(diff_json) > 0, "Expected non-empty JSON string"


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
