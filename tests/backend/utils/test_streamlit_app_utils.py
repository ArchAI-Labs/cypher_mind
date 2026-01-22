import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))

sys.path.insert(0, project_root)

import json
import pytest
from unittest.mock import patch, mock_open

from backend.utils.streamlit_app_utils import (
    format_results_as_table,
    generate_sample_questions,
)


def test_format_results_empty_or_none():
    """Test that empty inputs return an empty list."""
    assert format_results_as_table([]) == []
    assert format_results_as_table(None) == []


def test_format_results_simple_flat():
    """Test standard formatting of keys (title case) for flat dictionaries."""
    data = [
        {"name": "alice", "job_title": "engineer"},
        {"name": "bob", "job_title": "manager"},
    ]

    result = format_results_as_table(data)

    # Keys should be Title Cased
    assert result[0]["Name"] == "alice"
    assert result[0]["Job_Title"] == "engineer"
    assert len(result) == 2


def test_format_results_nested_dictionary():
    """Test that nested dictionaries are flattened with ' - ' separator."""
    data = [{"id": 1, "info": {"city": "rome", "zip_code": 100}}]

    result = format_results_as_table(data)

    # Check flattening format: ParentKey - ChildKey
    assert result[0]["Id"] == 1
    assert result[0]["Info - City"] == "rome"
    assert result[0]["Info - Zip_Code"] == 100
    # ensure original nested dict is gone
    assert "info" not in result[0]


def test_format_results_mixed_keys_and_padding():
    """
    Test that rows with different keys are normalized.
    Missing keys should be filled with empty strings.
    """
    data = [
        {"a": 1, "b": 2},  # Row 1 has A, B
        {"a": 3, "c": 4},  # Row 2 has A, C (missing B)
    ]

    result = format_results_as_table(data)

    # Check Row 1
    assert result[0]["A"] == 1
    assert result[0]["B"] == 2
    assert result[0]["C"] == ""  # Added padding

    # Check Row 2
    assert result[1]["A"] == 3
    assert result[1]["B"] == ""  # Added padding
    assert result[1]["C"] == 4


def test_format_results_sorting():
    """Test that keys in the output dictionaries are sorted alphabetically."""
    data = [{"z": 1, "a": 2, "m": 3}]

    result = format_results_as_table(data)
    keys = list(result[0].keys())

    # The function sorts keys: sorted(list(all_keys))
    assert keys == ["A", "M", "Z"]


# --- Tests for generate_sample_questions ---


def test_generate_sample_questions_success():
    """Test loading JSON from a file path defined in environment variables."""
    mock_env = {"SAMPLE_QUESTIONS": "questions.json"}
    mock_json_content = ["What is X?", "Who is Y?"]

    with patch.dict(os.environ, mock_env):
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(mock_json_content))
        ) as mock_file:
            questions = generate_sample_questions()

            assert questions == mock_json_content
            mock_file.assert_called_once_with("questions.json", "r")


def test_generate_sample_questions_file_not_found():
    """Test behavior when the file does not exist (should raise FileNotFoundError)."""
    mock_env = {"SAMPLE_QUESTIONS": "missing.json"}

    with patch.dict(os.environ, mock_env):
        with patch("builtins.open", side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError):
                generate_sample_questions()


def test_generate_sample_questions_invalid_json():
    """Test behavior when the file contains invalid JSON."""
    mock_env = {"SAMPLE_QUESTIONS": "bad.json"}

    with patch.dict(os.environ, mock_env):
        with patch("builtins.open", mock_open(read_data="INVALID JSON")):
            with pytest.raises(json.JSONDecodeError):
                generate_sample_questions()
