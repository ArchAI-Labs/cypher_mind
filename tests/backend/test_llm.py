import os
import sys
import json
import pytest
import importlib
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

if 'backend.llm' in sys.modules:
    del sys.modules['backend.llm']

import backend.llm as llm

# --- Fixtures ---

@pytest.fixture
def mock_env(monkeypatch):
    """Setta variabili d'ambiente fittizie."""
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "password")
    monkeypatch.setenv("GEMINI_API_KEY", "fake_key")
    monkeypatch.setenv("MODEL", "gemini/gemini-pro")

@pytest.fixture
def mock_completion():
    """Mock per litellm.completion."""
    with patch("backend.llm.completion") as mock:
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "MOCKED CONTENT"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        mock.return_value = mock_response
        yield mock

@pytest.fixture
def mock_neo4j_driver():
    """Mock per il driver Neo4j."""
    with patch("backend.llm.GraphDatabase.driver") as mock_driver:
        driver_instance = MagicMock()
        session_instance = MagicMock()
        result_cursor = MagicMock()
        
        mock_driver.return_value.__enter__.return_value = driver_instance
        driver_instance.session.return_value.__enter__.return_value = session_instance
        session_instance.run.return_value = result_cursor
        
        record = MagicMock()
        record.data.return_value = {"name": "Test Node", "id": 1}
        result_cursor.__iter__.return_value = [record]
        
        yield driver_instance, session_instance

# --- Tests ---

def test_get_schema_success(mock_env, mock_completion):
    mock_completion.return_value.choices[0].message.content = "Schema Description"
    nodes = {"Person": ["name", "age"]}
    rels = {"KNOWS": ["since"]}
    
    result = llm.get_schema(nodes, rels)
    
    assert result == "Schema Description"
    mock_completion.assert_called_once()

def test_get_schema_error(mock_env, mock_completion):
    mock_completion.side_effect = Exception("API Error")
    nodes = {}
    rels = {}
    
    result = llm.get_schema(nodes, rels)
    
    assert "An error occurred" in result
    assert "API Error" in result

def test_generate_cypher_query(mock_env, mock_completion):
    mock_completion.return_value.choices[0].message.content = "MATCH (n) RETURN n"
    schema = "Nodes: Person"
    question = "Find all people"
    
    result = llm.generate_cypher_query(question, schema)
    
    assert result == "MATCH (n) RETURN n"

def test_clean_cypher_query():
    raw_query = "```cypher\nMATCH (n) RETURN n\n```"
    cleaned = llm.clean_cypher_query(raw_query)
    assert cleaned == "MATCH (n) RETURN n"

def test_validate_cypher_syntax():
    assert llm.validate_cypher_syntax("MATCH (n) RETURN n") is True
    assert llm.validate_cypher_syntax("SELECT * FROM table") is False

def test_execute_cypher_query_success(mock_env, mock_neo4j_driver):
    driver, session = mock_neo4j_driver
    query = "MATCH (n) RETURN n"
    results = llm.execute_cypher_query(query)
    
    assert len(results) == 1
    assert results[0] == {"name": "Test Node", "id": 1}

def test_execute_cypher_query_failure(mock_env, mock_neo4j_driver):
    driver, session = mock_neo4j_driver
    session.run.side_effect = Exception("DB Connection Failed")
    
    with pytest.raises(Exception) as excinfo:
        llm.execute_cypher_query("BAD QUERY")
    
    assert "DB Connection Failed" in str(excinfo.value)

def test_ask_neo4j_llm_with_override(mock_env, mock_neo4j_driver):
    # Patchiamo generate_cypher_query all'interno del modulo llm
    with patch("backend.llm.generate_cypher_query") as mock_gen:
        override = "MATCH (o) RETURN o"
        
        result = llm.ask_neo4j_llm("Question", "Schema", override_cypher=override)
        
        mock_gen.assert_not_called()
        assert result["cypher_query"] == override

def test_ask_neo4j_llm_generation(mock_env, mock_neo4j_driver):
    with patch("backend.llm.generate_cypher_query", return_value="MATCH (g) RETURN g") as mock_gen:
        result = llm.ask_neo4j_llm("My Question", "My Schema")
        
        mock_gen.assert_called_once()
        assert result["cypher_query"] == "MATCH (g) RETURN g"

def test_ask_neo4j_llm_error_handling(mock_env):
    with patch("backend.llm.generate_cypher_query", side_effect=Exception("Gen Error")):
        result = llm.ask_neo4j_llm("Q", "S")
        
        assert "error" in result
        assert "Gen Error" in result["error"]

def test_extract_query_intent():
    q1 = "Show top 10 users"
    intent1 = llm.extract_query_intent(q1)
    assert intent1["action"] == "retrieve"
    assert intent1["limit"] == 10

def test_suggest_similar_queries():
    suggestions = llm.suggest_similar_queries("Show users", {})
    assert len(suggestions) > 0

def test_generate_coalesce_expression(mock_env, mock_completion):
    expected_cypher = "coalesce(n.name, 'Unknown')"
    mock_completion.return_value.choices[0].message.content = f"```cypher\n{expected_cypher}\n```"
    schema = {"Person": ["name", "age"]}
    
    result = llm.generate_coalesce_expression(schema)
    
    assert result == expected_cypher