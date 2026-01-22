import os
import sys
import json
import pytest
from unittest.mock import MagicMock, patch

# --- Setup del Path per l'import ---
# Assicurati che questo path punti correttamente alla cartella 'src'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Importiamo il modulo da testare
try:
    from backend.llm import (
        get_schema,
        generate_cypher_query,
        clean_cypher_query,
        validate_cypher_syntax,
        execute_cypher_query,
        ask_neo4j_llm,
        extract_query_intent,
        suggest_similar_queries,
        generate_coalesce_expression
    )
except ImportError:
    # Fallback se necessario, ma preferisci l'import diretto sopra
    import importlib.util
    spec = importlib.util.spec_from_file_location("llm", "./llm.py")
    llm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(llm)
    # Rinomina per compatibilità con il resto del file
    get_schema = llm.get_schema
    generate_cypher_query = llm.generate_cypher_query
    clean_cypher_query = llm.clean_cypher_query
    validate_cypher_syntax = llm.validate_cypher_syntax
    execute_cypher_query = llm.execute_cypher_query
    ask_neo4j_llm = llm.ask_neo4j_llm
    extract_query_intent = llm.extract_query_intent
    suggest_similar_queries = llm.suggest_similar_queries
    generate_coalesce_expression = llm.generate_coalesce_expression

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
    # CORREZIONE QUI: Usa 'backend.llm.completion' invece di 'llm.completion'
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
    # CORREZIONE QUI: Usa 'backend.llm.GraphDatabase.driver'
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
    
    result = get_schema(nodes, rels)
    
    assert result == "Schema Description"
    mock_completion.assert_called_once()
    args, kwargs = mock_completion.call_args
    assert str(nodes) in kwargs['messages'][1]['content']

def test_get_schema_error(mock_env, mock_completion):
    mock_completion.side_effect = Exception("API Error")
    nodes = {}
    rels = {}
    
    result = get_schema(nodes, rels)
    
    assert "An error occurred" in result
    assert "API Error" in result

def test_generate_cypher_query(mock_env, mock_completion):
    mock_completion.return_value.choices[0].message.content = "MATCH (n) RETURN n"
    schema = "Nodes: Person"
    question = "Find all people"
    
    result = generate_cypher_query(question, schema)
    
    assert result == "MATCH (n) RETURN n"
    
    call_args = mock_completion.call_args
    messages = call_args[1]['messages']
    assert schema in messages[0]['content']
    assert question in messages[1]['content']

def test_clean_cypher_query():
    raw_query = "```cypher\nMATCH (n) RETURN n\n```"
    cleaned = clean_cypher_query(raw_query)
    assert cleaned == "MATCH (n) RETURN n"

    raw_query_2 = "MATCH (n) RETURN n"
    cleaned_2 = clean_cypher_query(raw_query_2)
    assert cleaned_2 == "MATCH (n) RETURN n"

def test_validate_cypher_syntax():
    assert validate_cypher_syntax("MATCH (n) RETURN n") is True
    assert validate_cypher_syntax("CREATE (n:Test)") is True
    assert validate_cypher_syntax("SELECT * FROM table") is False
    assert validate_cypher_syntax("Hello world") is False
    assert validate_cypher_syntax("MATCH (n RETURN n") is False

def test_execute_cypher_query_success(mock_env, mock_neo4j_driver):
    driver, session = mock_neo4j_driver
    query = "MATCH (n) RETURN n"
    results = execute_cypher_query(query)
    
    assert len(results) == 1
    assert results[0] == {"name": "Test Node", "id": 1}
    session.run.assert_called_with(query)

def test_execute_cypher_query_failure(mock_env, mock_neo4j_driver):
    driver, session = mock_neo4j_driver
    session.run.side_effect = Exception("DB Connection Failed")
    
    with pytest.raises(Exception) as excinfo:
        execute_cypher_query("BAD QUERY")
    
    assert "DB Connection Failed" in str(excinfo.value)

def test_ask_neo4j_llm_with_override(mock_env, mock_neo4j_driver):
    # CORREZIONE QUI: Usa 'backend.llm.generate_cypher_query'
    with patch("backend.llm.generate_cypher_query") as mock_gen:
        override = "MATCH (o) RETURN o"
        
        result = ask_neo4j_llm("Question", "Schema", override_cypher=override)
        
        mock_gen.assert_not_called()
        assert result["cypher_query"] == override
        assert result["data"][0]["name"] == "Test Node"

def test_ask_neo4j_llm_generation(mock_env, mock_neo4j_driver):
    # CORREZIONE QUI: Usa 'backend.llm.generate_cypher_query'
    with patch("backend.llm.generate_cypher_query", return_value="MATCH (g) RETURN g") as mock_gen:
        result = ask_neo4j_llm("My Question", "My Schema")
        
        mock_gen.assert_called_once()
        assert result["cypher_query"] == "MATCH (g) RETURN g"
        assert result["data"] is not None

def test_ask_neo4j_llm_error_handling(mock_env):
    # CORREZIONE QUI: Usa 'backend.llm.generate_cypher_query'
    with patch("backend.llm.generate_cypher_query", side_effect=Exception("Gen Error")):
        result = ask_neo4j_llm("Q", "S")
        
        assert "error" in result
        assert "Gen Error" in result["error"]
        assert result["data"] == []

def test_extract_query_intent():
    q1 = "Show top 10 users"
    intent1 = extract_query_intent(q1)
    assert intent1["action"] == "retrieve"
    assert intent1["limit"] == 10
    assert "user" in intent1["entities"]

    q2 = "How many projects are there?"
    intent2 = extract_query_intent(q2)
    assert intent2["action"] == "count"
    assert intent2["aggregation"] == "count"
    assert "project" in intent2["entities"]

def test_suggest_similar_queries():
    suggestions = suggest_similar_queries("Show users", {})
    assert len(suggestions) > 0
    assert any("users" in s for s in suggestions)
    suggestions_proj = suggest_similar_queries("List projects", {})
    assert any("projects" in s for s in suggestions_proj)

def test_generate_coalesce_expression(mock_env, mock_completion):
    expected_cypher = "coalesce(n.name, 'Unknown')"
    mock_completion.return_value.choices[0].message.content = f"```cypher\n{expected_cypher}\n```"
    schema = {"Person": ["name", "age"]}
    
    result = generate_coalesce_expression(schema)
    
    assert result == expected_cypher