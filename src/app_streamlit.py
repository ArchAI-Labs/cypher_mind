import os
import json
import logging
import asyncio

import streamlit as st
import pandas as pd

from backend.llm import (
    ask_neo4j_llm,
    get_schema
)
from backend.gds_manager import GDSManager

from backend.utils.streamlit_app_utils import (
    format_results_as_table,
    generate_sample_questions,
)
from medha import Medha, CacheHit, Settings, SearchStrategy
from medha.embeddings.fastembed_adapter import FastEmbedAdapter


LOGO_PATH = "img/logo_cyphermind.png"

# Customized colors
BACKGROUND_COLOR = "#F8F7F4"
PRIMARY_COLOR = "#00274A"
TEXT_COLOR = PRIMARY_COLOR

# Set app style
st.markdown(
    f"""
    <style>
        body {{
            background-color: {BACKGROUND_COLOR};
            color: {TEXT_COLOR};
        }}
        .stApp {{
            background-color: {BACKGROUND_COLOR};
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: {PRIMARY_COLOR};
        }}
        .stButton > button {{
            color: white;
            background-color: {PRIMARY_COLOR};
            border-color: {PRIMARY_COLOR};
        }}
        .stButton > button:hover {{
            background-color: #585696;
            border-color: #585696;
        }}
        .cache-info {{
            background-color: #e8f4fd;
            padding: 10px;
            border-radius: 5px;
            border-left: 4px solid {PRIMARY_COLOR};
            margin: 10px 0;
        }}
        .strategy-badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
            color: white;
            margin-right: 5px;
        }}
        .template-badge {{ background-color: #28a745; }}
        .exact-badge {{ background-color: #17a2b8; }}
        .similar-badge {{ background-color: #ffc107; color: black; }}
        .fuzzy-badge {{ background-color: #fd7e14; }}
        .l1-badge {{ background-color: #20c997; }}
        .llm-badge {{ background-color: #dc3545; }}
        .performance-metric {{
            background-color: #f8f9fa;
            padding: 8px;
            border-radius: 4px;
            border: 1px solid #dee2e6;
            margin: 4px 0;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

def get_strategy_badge(strategy):
    """Return HTML badge for the strategy used"""
    badges = {
        SearchStrategy.TEMPLATE_MATCH: '<span class="strategy-badge template-badge">TEMPLATE</span>',
        SearchStrategy.EXACT_MATCH: '<span class="strategy-badge exact-badge">EXACT MATCH</span>',
        SearchStrategy.SEMANTIC_MATCH: '<span class="strategy-badge similar-badge">SIMILAR</span>',
        SearchStrategy.FUZZY_MATCH: '<span class="strategy-badge fuzzy-badge">FUZZY MATCH</span>',
        SearchStrategy.L1_CACHE: '<span class="strategy-badge l1-badge">L1 CACHE</span>',
    }
    # Also support string "llm" used for display logic
    if strategy == "llm":
        return '<span class="strategy-badge llm-badge">LLM GENERATED</span>'
    return badges.get(strategy, f'<span class="strategy-badge">{strategy}</span>')

def display_cache_result_info(cache_hit):
    """Display information about cache search results"""
    strategy = cache_hit.strategy

    # Check if this was a manually created CacheHit for an LLM fallback
    if strategy == SearchStrategy.NO_MATCH and cache_hit.generated_query:
        st.markdown(
            f'<div class="cache-info">'
            f'{get_strategy_badge("llm")}'
            f'<strong>New query generated!</strong> No cache match found, used LLM to generate fresh Cypher query.'
            f'<br><small>Confidence: {cache_hit.confidence:.2%}</small>'
            f'</div>',
            unsafe_allow_html=True
        )
    elif strategy == SearchStrategy.L1_CACHE:
        st.markdown(
            f'<div class="cache-info">'
            f'{get_strategy_badge(SearchStrategy.L1_CACHE)}'
            f'<strong>Found in L1 memory cache!</strong> Retrieved from recent queries (in-memory).'
            f'<br><small>Confidence: {cache_hit.confidence:.2%}</small>'
            f'</div>',
            unsafe_allow_html=True
        )
    elif strategy == SearchStrategy.TEMPLATE_MATCH:
        st.markdown(
            f'<div class="cache-info">'
            f'{get_strategy_badge(SearchStrategy.TEMPLATE_MATCH)}'
            f'<strong>Query template matched!</strong> Generated query directly from pattern without LLM call.'
            f'<br><small>Confidence: {cache_hit.confidence:.2%} | Template: {cache_hit.template_used or "Unknown"}</small>'
            f'</div>',
            unsafe_allow_html=True
        )
    elif strategy == SearchStrategy.EXACT_MATCH:
        st.markdown(
            f'<div class="cache-info">'
            f'{get_strategy_badge(SearchStrategy.EXACT_MATCH)}'
            f'<strong>Exact match found in cache!</strong> Retrieved previously processed query.'
            f'<br><small>Confidence: {cache_hit.confidence:.2%}</small>'
            f'</div>',
            unsafe_allow_html=True
        )
    elif strategy == SearchStrategy.SEMANTIC_MATCH:
        st.markdown(
            f'<div class="cache-info">'
            f'{get_strategy_badge(SearchStrategy.SEMANTIC_MATCH)}'
            f'<strong>Similar query found!</strong> Using cached result with semantic similarity.'
            f'<br><small>Confidence: {cache_hit.confidence:.2%}</small>'
            f'</div>',
            unsafe_allow_html=True
        )
    elif strategy == SearchStrategy.FUZZY_MATCH:
        st.markdown(
            f'<div class="cache-info">'
            f'{get_strategy_badge(SearchStrategy.FUZZY_MATCH)}'
            f'<strong>Fuzzy match found!</strong> Using string similarity to find similar query.'
            f'<br><small>Confidence: {cache_hit.confidence:.2%}</small>'
            f'</div>',
            unsafe_allow_html=True
        )
    elif strategy == SearchStrategy.NO_MATCH:
        st.info("No suitable cache match found. Will use LLM to generate new query.")


def initialize_cache():
    """Initialize the Medha semantic cache"""
    try:
        embedder = FastEmbedAdapter(
            model_name=os.environ.get("EMBEDDER", "sentence-transformers/all-MiniLM-L6-v2")
        )
        settings = Settings(
            qdrant_mode=os.environ.get("QDRANT_MODE", "memory"),
            qdrant_host=os.environ.get("QDRANT_HOST", "localhost"),
            qdrant_url=os.environ.get("QDRANT_URL"),
            qdrant_api_key=os.environ.get("QDRANT_API_KEY"),
            template_file=os.environ.get("TEMPLATE_QUERY"),
            score_threshold_exact=float(os.environ.get("EXACT_THRESHOLD", "0.99")),
            score_threshold_semantic=float(os.environ.get("SIMILARITY_THRESHOLD", "0.90")),
            score_threshold_template=float(os.environ.get("TEMPLATE_THRESHOLD", "0.70")),
            score_threshold_fuzzy=float(os.environ.get("FUZZY_THRESHOLD", "85.0")),
        )
        cache = Medha(
            collection_name=os.environ.get("QDRANT_COLLECTION", "semantic_cache"),
            embedder=embedder,
            settings=settings,
        )
        asyncio.run(cache.start())
        return cache
    except Exception as e:
        st.error(f"Error initializing cache: {e}")
        logging.error(f"Cache initialization error: {e}")
        return None


def _update_cache_thresholds(cache):
    """Update cache settings from sidebar slider values"""
    cache._settings.score_threshold_template = st.session_state.get('template_threshold', 0.70)
    cache._settings.score_threshold_exact = st.session_state.get('exact_threshold', 0.99)
    cache._settings.score_threshold_fuzzy = st.session_state.get('fuzzy_threshold', 85.0)
    cache._settings.score_threshold_semantic = st.session_state.get('similarity_threshold', 0.90)


def main():
    st.image(LOGO_PATH, width=200)
    st.title("ArchAI - CypherMind")
    st.subheader("Natural Language to Cypher")

    # Display optimization info banner
    st.info(
        "**Powered by Medha Semantic Cache** | "
        "5-tier waterfall search | "
        "L1 Memory + Templates + Vector + Semantic + Fuzzy"
    )

    # Initialize Semantic Cache
    if "qdrant_cache" not in st.session_state:
        with st.spinner("Initializing semantic cache..."):
            st.session_state.qdrant_cache = initialize_cache()
            if st.session_state.qdrant_cache is None:
                st.error("Failed to initialize semantic cache. Some features may be limited.")
                return

    if "gds" not in st.session_state:
        st.session_state.gds = GDSManager(
            uri=os.environ["NEO4J_URI"],
            user=os.environ["NEO4J_USER"],
            password=os.environ["NEO4J_PASSWORD"]
        )

    # Quick stats dashboard
    if st.session_state.qdrant_cache:
        try:
            cache_stats = st.session_state.qdrant_cache.stats
            col_stat1, col_stat2, col_stat3 = st.columns(3)

            with col_stat1:
                st.metric(
                    "Cache Hit Rate",
                    f"{cache_stats.get('hit_rate', 0):.1f}%",
                    help="Percentage of cache hits"
                )

            with col_stat2:
                st.metric(
                    "Total Requests",
                    cache_stats.get('total_requests', 0),
                    help="Total search requests"
                )

            with col_stat3:
                st.metric(
                    "Cached Queries",
                    cache_stats.get('total_cached_queries', 0),
                    help="Total queries stored in cache"
                )

        except Exception as e:
            logging.debug(f"Could not load quick stats: {e}")

    col1, col2 = st.columns([3, 1])

    with col1:
        sample_questions = generate_sample_questions()

        if "llm_result" not in st.session_state:
            st.session_state.llm_result = None

        if "last_cache_hit" not in st.session_state:
            st.session_state.last_cache_hit = None

        if st.button("Clear Previous Result"):
            st.session_state.llm_result = None
            st.session_state.last_cache_hit = None

        selected_question = st.selectbox(
            "Select an example question:", list(sample_questions.keys())
        )

        question = st.text_area(
            "Enter your question in natural language:",
            value=sample_questions[selected_question],
            height=150,
        )

        # Load schema
        with open(os.environ.get("NODE_CONTEXT_URL"), "r") as file:
            nodes = json.load(file)

        with open(os.environ.get("REL_CONTEXT_URL"), "r") as file:
            relationship = json.load(file)

        st.session_state.gds.get_schema(nodes)
        graph_schema = get_schema(nodes=nodes, relations=relationship)

        if st.button("Run"):
            if question and st.session_state.qdrant_cache:
                with st.spinner("Searching cache and processing query..."):
                    # Update thresholds from sidebar sliders
                    _update_cache_thresholds(st.session_state.qdrant_cache)

                    # Search using medha
                    cache_hit = st.session_state.qdrant_cache.search_sync(question)
                    st.session_state.last_cache_hit = cache_hit
                    generated_query = cache_hit.generated_query
                    result = None

                    # Case 1: Full response found in cache
                    if cache_hit.response_summary:
                        try:
                            # Cache stored as string, decode back to list/dict for display
                            data_decoded = json.loads(cache_hit.response_summary) if isinstance(cache_hit.response_summary, str) else cache_hit.response_summary
                        except:
                            data_decoded = cache_hit.response_summary

                        st.session_state.llm_result = {
                            "data": data_decoded,
                            "cypher_query": generated_query,
                            "cached": True,
                            "strategy": cache_hit.strategy,
                            "confidence": cache_hit.confidence
                        }
                        st.success("Response found in cache!")
                    else:
                        # Case 2: Cypher query found but no response summary
                        if generated_query and cache_hit.confidence > 0.0:
                            st.info(f"Running Cypher queries from cache ({cache_hit.strategy})...")
                            try:
                                current_result = ask_neo4j_llm(
                                    question=question,
                                    data_schema=graph_schema,
                                    override_cypher=generated_query
                                )

                                if current_result and current_result.get("data"):
                                    if isinstance(current_result["data"], list) and not current_result["data"]:
                                        st.warning("Cypher query executed but no results found. Fallback to LLM...")
                                        result = None
                                    else:
                                        # FIX: JSON Stringify the response summary before storing
                                        summary_to_cache = json.dumps(current_result["data"])
                                        
                                        st.session_state.qdrant_cache.store_sync(
                                            question=question,
                                            generated_query=generated_query,
                                            response_summary=summary_to_cache,
                                            template_id=cache_hit.template_used
                                        )
                                        st.session_state.llm_result = {
                                            **current_result,
                                            "cached": True,
                                            "strategy": cache_hit.strategy,
                                            "confidence": cache_hit.confidence
                                        }
                                        st.success("Query successfully executed from the database!")
                                        result = current_result
                                else:
                                    st.warning("No results from the Cypher query. Fallback to LLM...")
                                    result = None
                            except Exception as e:
                                st.error(f"Error executing Cypher query: {e}. Fallback to LLM...")
                                result = None

                        # Case 3: No cache match or fallback, use LLM
                        if not result:
                            st.info("Generating a new query with LLM...")
                            try:
                                result = ask_neo4j_llm(
                                    question=question,
                                    data_schema=graph_schema
                                )
                                if result and result.get("data"):
                                    # FIX: JSON Stringify the response summary before storing
                                    summary_to_cache = json.dumps(result["data"])

                                    st.session_state.qdrant_cache.store_sync(
                                        question=question,
                                        generated_query=result["cypher_query"],
                                        response_summary=summary_to_cache
                                    )
                                    st.session_state.llm_result = {
                                        **result,
                                        "cached": False,
                                        "strategy": "llm",
                                        "confidence": 1.0
                                    }
                                    
                                    # FIX: Use valid Enum (NO_MATCH) and stringified summary for the CacheHit object
                                    st.session_state.last_cache_hit = CacheHit(
                                        generated_query=result["cypher_query"],
                                        response_summary=summary_to_cache,
                                        confidence=1.0,
                                        strategy=SearchStrategy.NO_MATCH,
                                        template_used=None
                                    )
                                    st.success("Query generated and cached!")
                                else:
                                    st.warning("No results from the LLM.")
                                    st.session_state.llm_result = None
                            except Exception as e:
                                st.error(f"Error with the LLM model: {e}")
                                st.session_state.llm_result = None

        # Display cache search result info
        if st.session_state.last_cache_hit:
            display_cache_result_info(st.session_state.last_cache_hit)

        # Display results
        if st.session_state.llm_result:
            result = st.session_state.llm_result

            st.write("**Cypher Query:**")
            st.code(result["cypher_query"], language="cypher")

            st.write("**Results:**")
            table_data = format_results_as_table(result["data"])
            if table_data:
                try:
                    df = pd.DataFrame(table_data)
                    df = df.fillna("")
                    if df.shape[0] > 100:
                        st.dataframe(df.head(100))
                        st.write(f"Showing first 100 of {df.shape[0]} rows.")
                    else:
                        st.dataframe(df)
                except Exception as e:
                    st.write(f"Could not display results as a table: {e}")
                    st.json(result["data"])
            else:
                st.write("No results found.")

    with col2:
        with st.sidebar.expander("Cache Settings", expanded=True):
            st.subheader("Smart Cache Configuration")

            st.session_state.template_threshold = st.slider(
                "Template Match Threshold",
                min_value=0.5,
                max_value=1.0,
                value=st.session_state.get('template_threshold', 0.70),
                step=0.05,
                format="%.2f",
                key='template_threshold_slider',
                help="Minimum confidence for template matching.",
            )

            st.session_state.exact_threshold = st.slider(
                "Exact Match Threshold",
                min_value=0.9,
                max_value=1.0,
                value=st.session_state.get('exact_threshold', 0.99),
                step=0.01,
                format="%.3f",
                key='exact_threshold_slider',
                help="Threshold for considering two queries as exact matches.",
            )

            st.session_state.fuzzy_threshold = st.slider(
                "Fuzzy Match Threshold",
                min_value=70.0,
                max_value=100.0,
                value=st.session_state.get('fuzzy_threshold', 85.0),
                step=1.0,
                format="%.1f",
                key='fuzzy_threshold_slider',
                help="Minimum string similarity (Levenshtein) for fuzzy matching.",
            )

            st.session_state.similarity_threshold = st.slider(
                "Similarity Search Threshold",
                min_value=0.7,
                max_value=0.95,
                value=st.session_state.get('similarity_threshold', 0.90),
                step=0.01,
                format="%.3f",
                key='similarity_threshold_slider',
                help="Minimum similarity for semantic search.",
            )

            st.info(
                "**Cache Strategies (Waterfall):**\n\n"
                "1. **L1 Memory** - Recent queries\n"
                "2. **Template Matching** - Pattern-based generation\n"
                "3. **Exact Match** - High-confidence vector match\n"
                "4. **Semantic Similarity** - Meaning-based matching\n"
                "5. **Fuzzy Matching** - String similarity\n"
                "6. **LLM Fallback** - Generate new query"
            )

        with st.sidebar.expander("Cache Management"):
            col_cache_a, col_cache_b = st.columns(2)

            with col_cache_a:
                if st.button("Reset Cache"):
                    try:
                        if st.session_state.qdrant_cache:
                            asyncio.run(st.session_state.qdrant_cache.close())

                        st.session_state.qdrant_cache = initialize_cache()
                        if st.session_state.qdrant_cache:
                            st.success("Cache reset successfully!")
                    except Exception as e:
                        st.error(f"Error resetting cache: {e}")

            with col_cache_b:
                if st.button("Clear Memory"):
                    if st.session_state.qdrant_cache:
                        st.session_state.qdrant_cache.clear_caches()
                        st.success("Memory caches cleared!")

        with st.sidebar.expander("Performance Analytics"):
            if st.button("View Performance Stats"):
                if st.session_state.qdrant_cache:
                    with st.spinner("Loading stats..."):
                        try:
                            cache_stats = st.session_state.qdrant_cache.stats
                            st.subheader("Cache Performance")
                            col_perf_a, col_perf_b = st.columns(2)
                            with col_perf_a:
                                st.metric("Hit Rate", f"{cache_stats.get('hit_rate', 0):.1f}%")
                                st.metric("Total Requests", cache_stats.get('total_requests', 0))
                            with col_perf_b:
                                st.metric("Cached Queries", cache_stats.get('total_cached_queries', 0))
                                st.metric("Templates", cache_stats.get('templates_count', 0))
                        except Exception as e:
                            st.error(f"Error: {e}")

        with st.sidebar.expander("Batch Operations & Testing"):
            st.subheader("Batch Query Testing")
            batch_file = st.file_uploader("Upload JSON file", type=["json"])

            if batch_file is not None:
                try:
                    batch_data = json.load(batch_file)
                    if st.button("Insert Batch"):
                        with st.spinner(f"Inserting {len(batch_data)} queries..."):
                            # FIX: Ensure response_summary in batch is stringified if provided as object
                            for item in batch_data:
                                if "response_summary" in item and not isinstance(item["response_summary"], str):
                                    item["response_summary"] = json.dumps(item["response_summary"])
                            
                            success = asyncio.run(st.session_state.qdrant_cache.store_batch(batch_data))
                            if success: st.success("Success!")
                except Exception as e:
                    st.error(f"Error: {e}")

        # GDS Functionality
        with st.sidebar.expander("Graph Projections & GDS"):
            graph_name = st.text_input("Projection Name")
            node_labels_input = st.text_area("Node Labels (comma separated)")
            relationship_types_input = st.text_area("Relationship Types (comma separated)")

            if st.button("Create Projection"):
                if graph_name and node_labels_input and relationship_types_input:
                    node_labels_list = [l.strip() for l in node_labels_input.split(",") if l.strip()]
                    relationship_types_list = [r.strip() for r in relationship_types_input.split(",") if r.strip()]
                    with st.spinner("Creating..."):
                        msg = st.session_state.gds.create_projection(graph_name, node_labels_list, relationship_types_list)
                        st.success(msg)

            algo = st.selectbox("GDS Algorithm", ["PageRank", "Betweenness Centrality", "Closeness Centrality", "Louvain", "Node Similarity"])
            if st.button("Run GDS Algorithm"):
                if graph_name:
                    with st.spinner(f"Running {algo}..."):
                        gds = st.session_state.gds
                        if algo == "PageRank": gds_results = gds.run_pagerank(graph_name)
                        elif algo == "Betweenness Centrality": gds_results = gds.run_betweenness(graph_name)
                        elif algo == "Closeness Centrality": gds_results = gds.run_closeness(graph_name)
                        elif algo == "Louvain": gds_results = gds.run_louvain(graph_name)
                        elif algo == "Node Similarity": gds_results = gds.run_similarity(graph_name)
                        
                        st.dataframe(pd.DataFrame(gds_results))

if __name__ == "__main__":
    main()