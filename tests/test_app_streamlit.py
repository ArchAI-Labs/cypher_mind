import os
import sys
import json
import pytest
import asyncio
from unittest.mock import MagicMock, patch, mock_open, ANY
from enum import Enum

# --- Setup Path ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# --- Mock SearchStrategy enum (mirrors medha.SearchStrategy) ---
class MockSearchStrategy(str, Enum):
    L1_CACHE = "l1_cache"
    TEMPLATE_MATCH = "template_match"
    EXACT_MATCH = "exact_match"
    SEMANTIC_MATCH = "semantic_match"
    FUZZY_MATCH = "fuzzy_match"
    NO_MATCH = "no_match"
    ERROR = "error"

# --- Custom Mock Class for Session State ---
class MockSessionState(dict):
    """Simulates st.session_state supporting both dict and attribute access."""
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
    """Set required environment variables."""
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "password")
    monkeypatch.setenv("NODE_CONTEXT_URL", "nodes.json")
    monkeypatch.setenv("REL_CONTEXT_URL", "rels.json")
    monkeypatch.setenv("QDRANT_COLLECTION", "test_collection")

@pytest.fixture
def mock_cache_hit_class():
    class MockCacheHit:
        def __init__(self, generated_query, response_summary, confidence, strategy, template_used=None):
            self.generated_query = generated_query
            self.response_summary = response_summary
            self.confidence = confidence
            self.strategy = strategy
            self.template_used = template_used
    return MockCacheHit

@pytest.fixture
def mock_cache_instance(mock_cache_hit_class):
    cache = MagicMock()
    cache.search_sync.return_value = mock_cache_hit_class(
        generated_query="", response_summary=None, confidence=0.0, strategy=MockSearchStrategy.NO_MATCH
    )
    cache.store_sync.return_value = True
    cache.store_batch = MagicMock()
    cache.clear_caches.return_value = None
    cache.close = MagicMock()
    cache._settings = MagicMock()
    cache.stats = {
        "hit_rate": 0,
        "total_requests": 0,
        "total_cached_queries": 0,
        "templates_count": 0,
        "strategy_hits": {}
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

    # Mock backend modules
    mock_llm = MagicMock()
    mock_gds = MagicMock()
    mock_utils = MagicMock()

    # Mock medha module
    mock_medha = MagicMock()
    mock_medha.CacheHit = mock_cache_hit_class
    mock_medha.SearchStrategy = MockSearchStrategy
    mock_medha.Settings = MagicMock()
    mock_medha.Medha = MagicMock()

    # Mock medha embeddings submodule
    mock_medha_embeddings = MagicMock()
    mock_medha_embeddings_fastembed = MagicMock()

    # Mock Streamlit
    mock_st = MagicMock()

    # Session State simulation
    session_state = MockSessionState()
    mock_st.session_state = session_state
    mock_st._internal_session_state = session_state

    # Default inputs
    mock_st.selectbox.return_value = "Q1"
    mock_st.button.return_value = False
    mock_st.columns.side_effect = lambda x: [MagicMock() for _ in range(x)] if isinstance(x, int) else [MagicMock(), MagicMock()]
    mock_st.spinner.return_value.__enter__.return_value = None
    mock_st.sidebar.expander.return_value.__enter__.return_value = None

    # Mock pandas to prevent numpy re-import issues across test runs
    mock_pd = MagicMock()

    with patch.dict(sys.modules, {
        'streamlit': mock_st,
        'pandas': mock_pd,
        'backend.llm': mock_llm,
        'backend.gds_manager': mock_gds,
        'backend.utils.streamlit_app_utils': mock_utils,
        'medha': mock_medha,
        'medha.embeddings': mock_medha_embeddings,
        'medha.embeddings.fastembed_adapter': mock_medha_embeddings_fastembed,
    }):
        # Reload app_streamlit
        if 'app_streamlit' in sys.modules:
            del sys.modules['app_streamlit']
        import app_streamlit

        yield {
            'st': mock_st,
            'app': app_streamlit,
            'llm': mock_llm,
            'gds': mock_gds,
            'medha': mock_medha,
            'utils': mock_utils
        }

# --- Tests ---

def test_get_strategy_badge(setup_modules):
    app = setup_modules['app']
    badge = app.get_strategy_badge(MockSearchStrategy.TEMPLATE_MATCH)
    assert "template-badge" in badge

def test_get_strategy_badge_llm(setup_modules):
    app = setup_modules['app']
    badge = app.get_strategy_badge("llm")
    assert "llm-badge" in badge

def test_display_cache_result_info(setup_modules, mock_cache_hit_class):
    app = setup_modules['app']
    mock_st = setup_modules['st']

    hit = mock_cache_hit_class("", None, 0.99, MockSearchStrategy.L1_CACHE)
    app.display_cache_result_info(hit)
    mock_st.markdown.assert_called()

def test_initialize_cache_success(setup_modules):
    app = setup_modules['app']
    mock_st = setup_modules['st']

    mock_cache = MagicMock()
    with patch("app_streamlit.FastEmbedAdapter") as mock_embed, \
         patch("app_streamlit.Settings") as mock_settings, \
         patch("app_streamlit.Medha") as mock_medha_cls, \
         patch("app_streamlit.asyncio.run") as mock_run:

        mock_medha_cls.return_value = mock_cache
        result = app.initialize_cache()
        mock_medha_cls.assert_called_once()
        mock_run.assert_called_once()
        assert result == mock_cache

def test_initialize_cache_failure(setup_modules):
    app = setup_modules['app']
    mock_st = setup_modules['st']

    with patch("app_streamlit.FastEmbedAdapter", side_effect=Exception("Fail")):
        result = app.initialize_cache()
        assert result is None
        mock_st.error.assert_called()

def test_main_initial_setup(setup_modules, mock_env, mock_cache_instance, mock_gds_instance):
    app = setup_modules['app']
    mock_st = setup_modules['st']

    questions = {"Q1": "Query 1"}
    mock_st.selectbox.return_value = "Q1"

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

    def button_side_effect(label, **kwargs):
        return label == "Run"
    mock_st.button.side_effect = button_side_effect

    mock_st.selectbox.return_value = "Q1"
    mock_st.text_area.return_value = "Q"

    hit = mock_cache_hit_class("MATCH...", [{"n": 1}], 0.9, MockSearchStrategy.EXACT_MATCH)
    mock_cache_instance.search_sync.return_value = hit

    with patch("app_streamlit.generate_sample_questions", return_value={"Q1": "Q"}), \
         patch("builtins.open", mock_open(read_data='{}')), \
         patch("app_streamlit.format_results_as_table", return_value=[{"n": 1}]):

        app.main()

        assert mock_st.session_state["llm_result"]["cached"] is True
        mock_st.success.assert_any_call("Response found in cache!")

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

    hit = mock_cache_hit_class("MATCH (n) RETURN n", None, 0.9, MockSearchStrategy.TEMPLATE_MATCH, "tpl_1")
    mock_cache_instance.search_sync.return_value = hit

    db_result = {"data": [{"id": 1}], "cypher_query": "MATCH (n) RETURN n"}

    with patch("app_streamlit.generate_sample_questions", return_value={"Q1": "V"}), \
         patch("builtins.open", mock_open(read_data='{}')), \
         patch("app_streamlit.ask_neo4j_llm", return_value=db_result) as mock_ask:

        app.main()

        mock_ask.assert_called_with(
            question="Question",
            data_schema=ANY,
            override_cypher="MATCH (n) RETURN n"
        )
        mock_cache_instance.store_sync.assert_called()
        mock_st.success.assert_any_call("Query successfully executed from the database!")

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

    hit = mock_cache_hit_class("", None, 0.0, MockSearchStrategy.NO_MATCH)
    mock_cache_instance.search_sync.return_value = hit

    llm_res = {"data": [{"res": 1}], "cypher_query": "C"}

    with patch("app_streamlit.generate_sample_questions", return_value={"Q1": "Q"}), \
         patch("builtins.open", mock_open(read_data='{}')), \
         patch("app_streamlit.ask_neo4j_llm", return_value=llm_res):

        app.main()

        mock_st.success.assert_any_call("Query generated and cached!")

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
    mock_st.session_state["qdrant_cache"].stats = {"hit_rate": 0, "total_requests": 0, "total_cached_queries": 0}
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

    async_results = [mock_cache_hit_class("C1", None, 1.0, MockSearchStrategy.EXACT_MATCH)]

    async def mock_search(question):
        return async_results[0]

    mock_cache_instance.search = mock_search

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

    async def mock_store_batch(data):
        return True

    mock_cache_instance.store_batch = mock_store_batch

    with patch("json.load", return_value=[{"q": "q"}]), \
         patch("app_streamlit.generate_sample_questions", return_value={"Q1": "Q"}), \
         patch("builtins.open", mock_open(read_data='{}')):

         app.main()

         mock_st.success.assert_any_call("Successfully inserted 1 queries!")

def test_update_cache_thresholds(setup_modules, mock_cache_instance):
    app = setup_modules['app']
    mock_st = setup_modules['st']

    mock_st.session_state['template_threshold'] = 0.80
    mock_st.session_state['exact_threshold'] = 0.95
    mock_st.session_state['fuzzy_threshold'] = 90.0
    mock_st.session_state['similarity_threshold'] = 0.85

    app._update_cache_thresholds(mock_cache_instance)

    assert mock_cache_instance._settings.score_threshold_template == 0.80
    assert mock_cache_instance._settings.score_threshold_exact == 0.95
    assert mock_cache_instance._settings.score_threshold_fuzzy == 90.0
    assert mock_cache_instance._settings.score_threshold_semantic == 0.85
