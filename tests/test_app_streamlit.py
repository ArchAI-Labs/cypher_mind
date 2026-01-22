import os
import sys
import json
import pytest
import asyncio
from unittest.mock import MagicMock, patch, mock_open, ANY  # Added ANY

# --- Setup Path ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# --- Custom Mock Class for Session State ---
class MockSessionState(dict):
    """Simula st.session_state supportando sia accesso a dizionario che attributo."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

# --- Fixtures ---

@pytest.fixture
def mock_env(monkeypatch):
    """Setta variabili d'ambiente necessarie."""
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "password")
    monkeypatch.setenv("NODE_CONTEXT_URL", "nodes.json")
    monkeypatch.setenv("REL_CONTEXT_URL", "rels.json")
    monkeypatch.setenv("QDRANT_COLLECTION", "test_collection")

@pytest.fixture
def mock_cache_hit_class():
    class MockCacheHit:
        def __init__(self, cypher_query, response, confidence, strategy, template_used=None):
            self.cypher_query = cypher_query
            self.response = response
            self.confidence = confidence
            self.strategy = strategy
            self.template_used = template_used
    return MockCacheHit

@pytest.fixture
def mock_cache_instance(mock_cache_hit_class):
    cache = MagicMock()
    cache.smart_search.return_value = mock_cache_hit_class(
        cypher_query="", response=None, confidence=0.0, strategy="no_match"
    )
    # Mocking stats
    cache.get_performance_stats.return_value = {
        "cache_performance": {"hit_rate_percentage": 0, "template_hits": 0, "recent_cache_hits": 0, "embedding_cache_hits": 0, "embedding_cache_misses": 0, "fuzzy_hits": 0, "total_search_hits": 0},
        "storage": {"total_cached_queries": 0, "templates_in_qdrant": 0, "vector_size": 384, "embedding_cache_size": 0, "frequent_queries_cache_size": 0, "recent_results_cache_size": 0},
        "qdrant_metrics": {"indexed_vectors_count": 0, "vectors_count": 0, "segments_count": 0, "status": "ok", "optimizer_status": "ok", "config": {"hnsw_m": 16, "hnsw_ef_construct": 100, "distance": "Cosine", "quantization_enabled": False, "quantization_type": "None", "on_disk": False}},
        "model_info": {"embedder_model": "test", "nlp_enabled": False, "fuzzy_matching_available": False},
        "efficiency": {"templates_loaded": 0, "avg_template_priority": 0, "memory_efficiency": {"max_embedding_cache": 0, "max_frequent_cache": 0, "max_recent_cache": 0}}
    }
    return cache

@pytest.fixture
def mock_gds_instance():
    gds = MagicMock()
    gds.create_projection.return_value = "✅ Projection created"
    gds.delete_all_projections.return_value = "✅ Deleted"
    return gds

# --- Global Mocking & Import ---
@pytest.fixture(autouse=True)
def setup_modules(mock_cache_hit_class):
    
    # Mock dei moduli backend
    mock_llm = MagicMock()
    mock_gds = MagicMock()
    mock_cache = MagicMock()
    mock_utils = MagicMock()
    
    # Mock di Streamlit
    mock_st = MagicMock()
    
    # Session State simulation con classe custom
    session_state = MockSessionState()
    mock_st.session_state = session_state
    mock_st._internal_session_state = session_state 
    
    # Default inputs
    mock_st.selectbox.return_value = "Q1"
    
    # Default button behavior: False by default to prevent triggering all actions
    mock_st.button.return_value = False

    # Gestione columns
    mock_st.columns.side_effect = lambda x: [MagicMock() for _ in range(x)] if isinstance(x, int) else [MagicMock(), MagicMock()]
    
    # Gestione context managers
    mock_st.spinner.return_value.__enter__.return_value = None
    mock_st.sidebar.expander.return_value.__enter__.return_value = None

    # PATCH CRITICA
    with patch.dict(sys.modules, {
        'streamlit': mock_st,
        'backend.llm': mock_llm,
        'backend.gds_manager': mock_gds,
        'backend.semantic_cache': mock_cache,
        'backend.utils.streamlit_app_utils': mock_utils
    }):
        mock_cache.CacheHit = mock_cache_hit_class
        
        # Reload app_streamlit
        if 'app_streamlit' in sys.modules:
            del sys.modules['app_streamlit']
        import app_streamlit
        
        yield {
            'st': mock_st,
            'app': app_streamlit,
            'llm': mock_llm,
            'gds': mock_gds,
            'cache': mock_cache,
            'utils': mock_utils
        }

# --- Tests ---

def test_get_strategy_badge(setup_modules):
    app = setup_modules['app']
    badge = app.get_strategy_badge("template_match")
    assert "template-badge" in badge

def test_display_cache_result_info(setup_modules, mock_cache_hit_class):
    app = setup_modules['app']
    mock_st = setup_modules['st']
    
    hit = mock_cache_hit_class("", None, 0.99, "recent_cache")
    app.display_cache_result_info(hit)
    mock_st.markdown.assert_called()

def test_initialize_cache_optimized_success(setup_modules):
    app = setup_modules['app']
    mock_st = setup_modules['st']
    
    with patch("app_streamlit.create_optimized_cache") as mock_create:
        mock_create.return_value = MagicMock()
        app.initialize_cache()
        mock_create.assert_called_once()
        mock_st.error.assert_not_called()

def test_initialize_cache_fallback(setup_modules):
    app = setup_modules['app']
    mock_st = setup_modules['st']
    
    with patch("app_streamlit.create_optimized_cache", side_effect=Exception("Fail")), \
         patch("app_streamlit.SemanticCache") as mock_direct:
        
        app.initialize_cache()
        mock_st.error.assert_called_once()
        mock_direct.assert_called_once()

def test_main_initial_setup(setup_modules, mock_env, mock_cache_instance, mock_gds_instance):
    app = setup_modules['app']
    mock_st = setup_modules['st']
    
    questions = {"Q1": "Query 1"}
    mock_st.selectbox.return_value = "Q1"

    # Ensure no previous result is set to trigger rendering logic that fails
    if "llm_result" in mock_st.session_state:
        del mock_st.session_state["llm_result"]

    with patch("app_streamlit.initialize_cache", return_value=mock_cache_instance), \
         patch("app_streamlit.GDSManager", return_value=mock_gds_instance), \
         patch("app_streamlit.generate_sample_questions", return_value=questions), \
         patch("builtins.open", mock_open(read_data='{}')):
        
        app.main()
        
        assert mock_st.session_state["qdrant_cache"] == mock_cache_instance
        mock_st.image.assert_called()

def test_run_button_cache_hit_response(setup_modules, mock_env, mock_cache_instance, mock_cache_hit_class):
    app = setup_modules['app']
    mock_st = setup_modules['st']
    
    mock_st.session_state["qdrant_cache"] = mock_cache_instance
    mock_st.session_state["gds"] = MagicMock()
    
    # Specific button behavior: Return True only for "Run", False for others (like "Delete All Projections")
    def button_side_effect(label, **kwargs):
        return label == "Run"
    mock_st.button.side_effect = button_side_effect

    mock_st.selectbox.return_value = "Q1"
    mock_st.text_area.return_value = "Q"
    
    hit = mock_cache_hit_class("MATCH...", [{"n": 1}], 0.9, "exact_match")
    mock_cache_instance.smart_search.return_value = hit
    
    with patch("app_streamlit.generate_sample_questions", return_value={"Q1": "Q"}), \
         patch("builtins.open", mock_open(read_data='{}')), \
         patch("app_streamlit.format_results_as_table", return_value=[{"n": 1}]):
        
        app.main()
        
        assert mock_st.session_state["llm_result"]["cached"] is True
        mock_st.success.assert_called_with("🎉 Response found in cache!")

def test_run_button_cypher_hit_exec_success(setup_modules, mock_env, mock_cache_instance, mock_cache_hit_class):
    app = setup_modules['app']
    mock_st = setup_modules['st']
    
    mock_st.session_state["qdrant_cache"] = mock_cache_instance
    mock_st.session_state["gds"] = MagicMock()
    
    def button_side_effect(label, **kwargs):
        return label == "Run"
    mock_st.button.side_effect = button_side_effect
    
    mock_st.selectbox.return_value = "Q1"
    mock_st.text_area.return_value = "Question"
    
    hit = mock_cache_hit_class("MATCH (n) RETURN n", None, 0.9, "template_match", "tpl_1")
    mock_cache_instance.smart_search.return_value = hit
    
    db_result = {"data": [{"id": 1}], "cypher_query": "MATCH (n) RETURN n"}
    
    with patch("app_streamlit.generate_sample_questions", return_value={"Q1": "V"}), \
         patch("builtins.open", mock_open(read_data='{}')), \
         patch("app_streamlit.ask_neo4j_llm", return_value=db_result) as mock_ask:

        app.main()
        
        mock_ask.assert_called_with(
            question="Question", 
            data_schema=ANY,  # Correct usage of ANY
            override_cypher="MATCH (n) RETURN n"
        )
        mock_cache_instance.store_query_and_response.assert_called()
        mock_st.success.assert_any_call("✅ Query successfully executed from the database!")

def test_run_button_no_match_llm_fallback(setup_modules, mock_env, mock_cache_instance, mock_cache_hit_class):
    app = setup_modules['app']
    mock_st = setup_modules['st']
    
    mock_st.session_state["qdrant_cache"] = mock_cache_instance
    mock_st.session_state["gds"] = MagicMock()
    
    def button_side_effect(label, **kwargs):
        return label == "Run"
    mock_st.button.side_effect = button_side_effect

    mock_st.selectbox.return_value = "Q1"
    mock_st.text_area.return_value = "Q"
    
    hit = mock_cache_hit_class("", None, 0.0, "no_match")
    mock_cache_instance.smart_search.return_value = hit
    
    llm_res = {"data": [{"res": 1}], "cypher_query": "C"}
    
    with patch("app_streamlit.generate_sample_questions", return_value={"Q1": "Q"}), \
         patch("builtins.open", mock_open(read_data='{}')), \
         patch("app_streamlit.ask_neo4j_llm", return_value=llm_res):
        
        app.main()
        
        mock_st.success.assert_any_call("🚀 Query generated and cached!")

def test_sidebar_cache_management(setup_modules, mock_cache_instance):
    app = setup_modules['app']
    mock_st = setup_modules['st']
    
    mock_st.session_state["qdrant_cache"] = mock_cache_instance
    mock_st.session_state["gds"] = MagicMock()
    mock_st.selectbox.return_value = "Q1"
    
    def button_side_effect(label, **kwargs):
        return label == "Clear Memory"
    mock_st.button.side_effect = button_side_effect
    
    with patch("app_streamlit.generate_sample_questions", return_value={"Q1": "Q"}), \
         patch("builtins.open", mock_open(read_data='{}')):
        
        app.main()
        
        mock_cache_instance.clear_caches.assert_called()

def test_sidebar_gds_projection(setup_modules, mock_gds_instance):
    app = setup_modules['app']
    mock_st = setup_modules['st']
    
    mock_st.session_state["qdrant_cache"] = MagicMock()
    mock_st.session_state["gds"] = mock_gds_instance
    mock_st.selectbox.return_value = "Q1"
    
    mock_st.text_input.side_effect = lambda l, **k: "G" if "Projection" in l else ""
    mock_st.text_area.side_effect = lambda l, **k: "L" if "Labels" in l or "Type" in l else ""
    mock_st.button.side_effect = lambda l, **k: l == "Create Projection"
    
    with patch("app_streamlit.generate_sample_questions", return_value={"Q1": "Q"}), \
         patch("builtins.open", mock_open(read_data='{}')):
        
        app.main()
        
        mock_gds_instance.create_projection.assert_called_with("G", ["L"], ["L"])

def test_sidebar_async_batch_search(setup_modules, mock_cache_instance, mock_cache_hit_class):
    app = setup_modules['app']
    mock_st = setup_modules['st']
    
    mock_st.session_state["qdrant_cache"] = mock_cache_instance
    mock_st.session_state["gds"] = MagicMock()
    mock_st.selectbox.return_value = "Q1"
    
    mock_st.text_area.side_effect = lambda l, **k: "Q1" if "Enter questions" in l else ""
    mock_st.button.side_effect = lambda l, **k: l == "Run Async Batch Search"
    
    async_results = [mock_cache_hit_class("C1", None, 1.0, "exact_match")]
    
    async def mock_async_search(questions):
        return async_results
    
    mock_cache_instance.async_batch_search.side_effect = mock_async_search
    
    with patch("app_streamlit.generate_sample_questions", return_value={"Q1": "Q"}), \
         patch("builtins.open", mock_open(read_data='{}')):
        
        app.main()
        
        mock_st.metric.assert_any_call("Confidence", "100.00%")

def test_file_uploader_batch(setup_modules, mock_cache_instance):
    app = setup_modules['app']
    mock_st = setup_modules['st']
    
    mock_st.session_state["qdrant_cache"] = mock_cache_instance
    mock_st.session_state["gds"] = MagicMock()
    mock_st.selectbox.return_value = "Q1"
    
    mock_st.file_uploader.return_value = MagicMock()
    mock_st.button.side_effect = lambda l, **k: l == "Insert Batch"
    mock_cache_instance.store_batch_queries.return_value = True
    
    with patch("json.load", return_value=[{"q":"q"}]), \
         patch("app_streamlit.generate_sample_questions", return_value={"Q1": "Q"}), \
         patch("builtins.open", mock_open(read_data='{}')):
         
         app.main()
         
         mock_st.success.assert_any_call("✅ Successfully inserted 1 queries!")