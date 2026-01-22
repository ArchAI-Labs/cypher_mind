import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Configurazione path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

# --- MOCK PREVENTIVO ---
# Mockiamo 'backend.llm' PRIMA di importare 'backend.gds_manager'.
# Questo evita ImportError se llm.py ha dipendenze complesse e ci permette di controllare generate_coalesce_expression.
mock_llm = MagicMock()
sys.modules['backend.llm'] = mock_llm

# Ora possiamo importare la classe in sicurezza
from backend.gds_manager import GDSManager

@pytest.fixture
def mock_driver():
    """
    Mock del metodo neo4j.GraphDatabase.driver.
    Usiamo il path assoluto 'neo4j.GraphDatabase.driver' per garantire che
    venga intercettata qualsiasi chiamata, indipendentemente da come è stata importata.
    """
    with patch("neo4j.GraphDatabase.driver") as mock_d:
        driver_instance = MagicMock()
        session_instance = MagicMock()
        
        # Setup della catena di chiamate: driver -> session -> transazione
        mock_d.return_value = driver_instance
        driver_instance.session.return_value.__enter__.return_value = session_instance
        
        yield driver_instance, session_instance

@pytest.fixture
def gds_manager(mock_driver):
    """Istanza di GDSManager con driver mockato."""
    # Poiché la fixture mock_driver è attiva, GraphDatabase.driver restituirà il mock
    return GDSManager("bolt://localhost", "user", "pass")

# --- Tests ---

def test_init(gds_manager):
    assert gds_manager.coalesce_expr == "gds.util.asNode(nodeId)"
    assert gds_manager.driver is not None

def test_close(gds_manager):
    gds_manager.close()
    gds_manager.driver.close.assert_called_once()

def test_get_schema(gds_manager):
    # Configuriamo il mock del modulo llm che abbiamo iniettato all'inizio
    mock_llm.generate_coalesce_expression.return_value = "coalesce(n.name, 'Unknown')"
    
    gds_manager.get_schema({"Node": ["prop"]})
    
    # Verifichiamo che la funzione importata sia stata chiamata
    mock_llm.generate_coalesce_expression.assert_called_with({"Node": ["prop"]})
    assert gds_manager.coalesce_expr == "coalesce(n.name, 'Unknown')"

def test_create_projection_success(gds_manager, mock_driver):
    _, session = mock_driver
    
    # Mock del risultato della query
    mock_record = {"graphName": "test-graph", "nodeCount": 100, "relationshipCount": 50}
    mock_result = MagicMock()
    mock_result.single.return_value = mock_record
    session.run.return_value = mock_result

    result = gds_manager.create_projection("test-graph", ["Person"], ["KNOWS"])
    
    assert "✅" in result
    assert "100 nodi" in result
    session.run.assert_called_once()

def test_create_projection_error(gds_manager, mock_driver):
    _, session = mock_driver
    # Simuliamo un errore del DB
    session.run.side_effect = Exception("Projection Error")
    
    result = gds_manager.create_projection("test-graph", [], [])
    assert "❌ Errore" in result
    assert "Projection Error" in result

def test_delete_all_projections(gds_manager, mock_driver):
    _, session = mock_driver
    
    # Simuliamo due chiamate a session.run:
    # 1. La prima per ottenere la lista dei grafi
    # 2. Le successive per cancellarli (il valore di ritorno non importa)
    
    graph_list = [{"graphName": "g1"}, {"graphName": "g2"}]
    
    # side_effect accetta una lista di valori di ritorno per chiamate consecutive
    session.run.side_effect = [graph_list, MagicMock(), MagicMock()]
    
    result = gds_manager.delete_all_projections()
    
    assert "✅ 2 proiezioni eliminate" in result
    assert session.run.call_count == 3 # 1 list + 2 drop

def test_list_algorithms(gds_manager, mock_driver):
    _, session = mock_driver
    session.run.side_effect = None # Reset side effect
    
    algo_record = MagicMock()
    algo_record.data.return_value = {"name": "PageRank", "description": "PR algo"}
    session.run.return_value = [algo_record]
    
    algos = gds_manager.list_algorithms()
    
    assert len(algos) == 1
    assert algos[0]["name"] == "PageRank"

def test_run_pagerank(gds_manager, mock_driver):
    _, session = mock_driver
    session.run.side_effect = None
    
    mock_data = MagicMock()
    mock_data.data.return_value = {"label": "Node A", "score": 0.5}
    session.run.return_value = [mock_data]

    results = gds_manager.run_pagerank("my-graph")
    
    assert len(results) == 1
    # Verifica che la query contenga il nome della funzione GDS corretta
    call_args = session.run.call_args
    query_sent = call_args[0][0]
    assert "gds.pageRank.stream" in query_sent

def test_run_algorithms_wrappers(gds_manager, mock_driver):
    """Test generico per assicurarsi che i metodi wrapper chiamino l'algoritmo giusto."""
    _, session = mock_driver
    session.run.side_effect = None
    session.run.return_value = [] 

    # Betweenness
    gds_manager.run_betweenness("g")
    assert "gds.betweenness.stream" in session.run.call_args[0][0]
    
    # Closeness
    gds_manager.run_closeness("g")
    assert "gds.beta.closeness.stream" in session.run.call_args[0][0]

    # Louvain
    gds_manager.run_louvain("g")
    assert "gds.louvain.stream" in session.run.call_args[0][0]

    # Similarity
    gds_manager.run_similarity("g")
    assert "gds.nodeSimilarity.stream" in session.run.call_args[0][0]

def test_run_algorithm_error(gds_manager, mock_driver):
    _, session = mock_driver
    session.run.side_effect = Exception("GDS Error")
    
    result = gds_manager.run_pagerank("fail-graph")
    
    assert "error" in result[0]
    assert "GDS Error" in result[0]["error"]