import os
import sys
import pytest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from backend.import_data import GraphImport

@pytest.fixture
def mock_tx():
    """Mock della transazione Neo4j."""
    tx = MagicMock()
    return tx

@pytest.fixture
def graph_importer(mock_tx):
    """Istanza di GraphImport con driver mockato."""
    with patch("backend.import_data.GraphDatabase.driver") as mock_d:
        driver_instance = MagicMock()
        session_instance = MagicMock()
        
        mock_d.return_value = driver_instance
        driver_instance.session.return_value = session_instance

        def execute_write_side_effect(func, *args, **kwargs):
            return func(mock_tx, *args, **kwargs)
            
        session_instance.execute_write.side_effect = execute_write_side_effect
        
        session_instance.__enter__.return_value = session_instance
        session_instance.__exit__.return_value = None

        importer = GraphImport("bolt://uri", "user", "pass")
        
        importer._mock_session = session_instance 
        yield importer

# --- Tests ---

def test_init(graph_importer):
    assert graph_importer.user == "user"
    assert graph_importer.driver is not None

def test_close(graph_importer):
    graph_importer.close()
    graph_importer.driver.close.assert_called_once()

def test_open(graph_importer):
    session = graph_importer.open()
    graph_importer.driver.session.assert_called()
    assert session is not None

def test_reset_database(graph_importer, mock_tx):
    graph_importer.reset_database()
    
    graph_importer._mock_session.execute_write.assert_called()
    
    mock_tx.run.assert_called_with("MATCH (n) DETACH DELETE n")

def test_import_nodes_no_metadata(graph_importer, mock_tx):
    file_path = "nodes.csv"
    label = "Person"
    
    graph_importer.import_nodes(file_path, label)
    
    args, kwargs = mock_tx.run.call_args
    query = args[0]
    
    assert f"LOAD CSV WITH HEADERS FROM 'file:///{file_path}'" in query
    assert f"CREATE (n:{label} " in query
    assert kwargs['file_path'] == file_path

def test_import_nodes_with_metadata(graph_importer, mock_tx):
    file_path = "data.csv"
    label = "User"
    metadata = ["name", "age"]
    
    graph_importer.import_nodes(file_path, label, node_metadata=metadata)
    
    args, _ = mock_tx.run.call_args
    query = args[0]
    
    assert "name: row['name']" in query
    assert "age: row['age']" in query

def test_import_relationships_simple(graph_importer, mock_tx):
    file_path = "rels.csv"
    rel_type = "FRIEND"
    start_lbl = "Person"
    end_lbl = "Person"
    id_keys = ["id_from", "id_to"]
    
    graph_importer.import_relationships(file_path, rel_type, start_lbl, end_lbl, id_keys)
    
    args, kwargs = mock_tx.run.call_args
    query = args[0]
    
    assert f"MATCH (a:{start_lbl}" in query
    assert f"MATCH (b:{end_lbl}" in query
    assert f"CREATE (a)-[r:{rel_type}]->(b)" in query
    assert kwargs['id_keys'] == id_keys

def test_import_relationships_with_metadata(graph_importer, mock_tx):
    file_path = "rels.csv"
    rel_type = "WORKS_AT"
    start_lbl = "Person"
    end_lbl = "Company"
    id_keys = ["p_id", "c_id"]
    metadata = ["since", "role"]
    
    graph_importer.import_relationships(
        file_path, rel_type, start_lbl, end_lbl, id_keys, relationship_metadata=metadata
    )
    
    args, _ = mock_tx.run.call_args
    query = args[0]
    
    assert "since: row['since']" in query
    assert "role: row['role']" in query
    assert f"CREATE (a)-[r:{rel_type}{{since: row['since'], role: row['role']}}]->(b)" in query.replace('\n', '').replace('  ', ' ')

def test_import_all(graph_importer):
    node_files = {
        "users.csv": ("User", ["name"], "id")
    }
    rel_files = {
        "knows.csv": ("KNOWS", "User", "User", ["id", "friend_id"], ["since"])
    }
    
    with patch.object(graph_importer, 'import_nodes') as mock_nodes, \
         patch.object(graph_importer, 'import_relationships') as mock_rels:
        
        graph_importer.import_all(node_files, rel_files)
        
        mock_nodes.assert_called_once_with("users.csv", "User", ["name"], "id")
        mock_rels.assert_called_once_with("knows.csv", "KNOWS", "User", "User", ["id", "friend_id"], ["since"])