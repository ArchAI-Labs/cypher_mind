import os
import json
import logging

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
from backend.semantic_cache import SemanticCache, create_optimized_cache


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
        .signature-badge {{ background-color: #6f42c1; }}
        .fuzzy-badge {{ background-color: #fd7e14; }}
        .recent-badge {{ background-color: #20c997; }}
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
        "template_match": '<span class="strategy-badge template-badge">TEMPLATE</span>',
        "exact_match": '<span class="strategy-badge exact-badge">EXACT MATCH</span>',
        "semantic_similarity": '<span class="strategy-badge similar-badge">SIMILAR</span>',
        "semantic_signature": '<span class="strategy-badge signature-badge">SIGNATURE</span>',
        "fuzzy_match": '<span class="strategy-badge fuzzy-badge">FUZZY MATCH</span>',  # NUOVO
        "recent_cache": '<span class="strategy-badge recent-badge">RECENT</span>',  # NUOVO
        "llm": '<span class="strategy-badge llm-badge">LLM GENERATED</span>'
    }
    return badges.get(strategy, f'<span class="strategy-badge">{strategy.upper()}</span>')

def display_cache_result_info(cache_hit):
    """Display information about cache search results using new CacheHit object"""
    if cache_hit.strategy == "recent_cache":
        st.markdown(
            f'<div class="cache-info">'
            f'{get_strategy_badge("recent_cache")}'
            f'<strong>Found in recent cache!</strong> Retrieved from last 3 queries (in-memory).'
            f'<br><small>Confidence: {cache_hit.confidence:.2%}</small>'
            f'</div>',
            unsafe_allow_html=True
        )
    elif cache_hit.strategy == "template_match":
        st.markdown(
            f'<div class="cache-info">'
            f'{get_strategy_badge("template_match")}'
            f'<strong>Query template matched!</strong> Generated Cypher directly from pattern without LLM call.'
            f'<br><small>Confidence: {cache_hit.confidence:.2%} | Template: {cache_hit.template_used or "Unknown"}</small>'
            f'</div>',
            unsafe_allow_html=True
        )
    elif cache_hit.strategy == "exact_match":
        st.markdown(
            f'<div class="cache-info">'
            f'{get_strategy_badge("exact_match")}'
            f'<strong>Exact match found in cache!</strong> Retrieved previously processed query.'
            f'<br><small>Confidence: {cache_hit.confidence:.2%}</small>'
            f'</div>',
            unsafe_allow_html=True
        )
    elif cache_hit.strategy in ["semantic_similarity", "semantic_signature"]:
        st.markdown(
            f'<div class="cache-info">'
            f'{get_strategy_badge(cache_hit.strategy)}'
            f'<strong>Similar query found!</strong> Using cached result with {cache_hit.strategy.replace("_", " ")}.'
            f'<br><small>Confidence: {cache_hit.confidence:.2%}</small>'
            f'</div>',
            unsafe_allow_html=True
        )
    elif cache_hit.strategy == "fuzzy_match":
        st.markdown(
            f'<div class="cache-info">'
            f'{get_strategy_badge("fuzzy_match")}'
            f'<strong>Fuzzy match found!</strong> Using string similarity to find similar query.'
            f'<br><small>Confidence: {cache_hit.confidence:.2%}</small>'
            f'</div>',
            unsafe_allow_html=True
        )
    elif cache_hit.strategy == "no_match":
        st.info("No suitable cache match found. Will use LLM to generate new query.")


def initialize_cache():
    """Initialize the enhanced semantic cache with better error handling"""
    try:
        # Prima prova la funzione factory ottimizzata
        cache = create_optimized_cache(
            collection_name=os.environ.get("QDRANT_COLLECTION", "semantic_cache"),
            mode=os.environ.get("QDRANT_MODE", "memory"),
            embedder=os.environ.get("EMBEDDER", "sentence-transformers/all-MiniLM-L6-v2"),
            vector_size=int(os.environ.get("VECTOR_SIZE", "384"))
        )
        return cache
    except Exception as e:
        st.error(f"Error with optimized cache initialization: {e}")
        logging.error(f"Optimized cache initialization error: {e}")
        
        # Fallback alla inizializzazione diretta
        try:
            cache = SemanticCache(
                collection_name=os.environ.get("QDRANT_COLLECTION", "semantic_cache"),
                mode=os.environ.get("QDRANT_MODE", "memory"),
                embedder=os.environ.get("EMBEDDER", "sentence-transformers/all-MiniLM-L6-v2"),
                vector_size=int(os.environ.get("VECTOR_SIZE", "384"))
            )
            return cache
        except Exception as e2:
            st.error(f"Error with direct cache initialization: {e2}")
            logging.error(f"Direct cache initialization error: {e2}")
            return None

def main():
    st.image(LOGO_PATH, width=200)
    st.title("ArchAI - CypherMind")
    st.subheader("Natural Language to Cypher")

    # Display optimization info banner
    st.info(
        "⚡ **Powered by Optimized Qdrant Vector Search** | "
        "🚀 Quantization enabled (~75% memory reduction) | "
        "📊 HNSW indexing | "
        "⚙️ Batch operations | "
        "🔄 Async concurrent search"
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
            perf_stats = st.session_state.qdrant_cache.get_performance_stats()
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

            with col_stat1:
                st.metric(
                    "Cache Hit Rate",
                    f"{perf_stats['cache_performance']['hit_rate_percentage']:.1f}%",
                    help="Percentage of embedding cache hits"
                )

            with col_stat2:
                st.metric(
                    "Cached Queries",
                    perf_stats['storage']['total_cached_queries'],
                    help="Total queries stored in Qdrant"
                )

            with col_stat3:
                quantization_type = perf_stats['qdrant_metrics']['config'].get('quantization_type', 'None')
                st.metric(
                    "Quantization",
                    quantization_type if quantization_type else "Disabled",
                    help="Vector quantization type for memory optimization"
                )

            with col_stat4:
                st.metric(
                    "Templates",
                    perf_stats['storage']['templates_in_qdrant'],
                    help="Query templates in Qdrant"
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
                    # Recupera i valori dei threshold dalla sessione
                    template_threshold = st.session_state.get('template_threshold', 0.8)
                    exact_threshold = st.session_state.get('exact_threshold', 0.95)
                    similarity_threshold = st.session_state.get('similarity_threshold', 0.8)
                    fuzzy_threshold = st.session_state.get('fuzzy_threshold', 85.0)  # AGGIUNTO

                    # Usa l'enhanced smart search con i threshold dell'interfaccia utente
                    cache_hit = st.session_state.qdrant_cache.smart_search(
                        question,
                        template_threshold=template_threshold,
                        exact_threshold=exact_threshold,
                        similarity_threshold=similarity_threshold,
                        fuzzy_threshold=fuzzy_threshold
                    )
                    st.session_state.last_cache_hit = cache_hit
                    cypher_query = cache_hit.cypher_query
                    result = None # Inizializza result a None

                    # Caso 1: Trovato un risultato completo in cache
                    if cache_hit.response:
                        st.session_state.llm_result = {
                            "data": cache_hit.response,
                            "cypher_query": cypher_query,
                            "cached": True,
                            "strategy": cache_hit.strategy,
                            "confidence": cache_hit.confidence
                        }
                        st.success("🎉 Response found in cache!")
                    else:
                        # Caso 2: Trovata una query Cypher, ma non la response o la response era vuota
                        # Questa parte viene eseguita solo se non c'è una response completa in cache
                        if cypher_query and cache_hit.confidence > 0.0:
                            st.info(f"Running Cypher queries from cache ({cache_hit.strategy})...")
                            try:
                                # Prova ad eseguire la query Cypher
                                current_result = ask_neo4j_llm(
                                    question=question,
                                    data_schema=graph_schema,
                                    override_cypher=cypher_query
                                )

                                if current_result and current_result.get("data"):
                                    # Verifica se i risultati sono vuoti
                                    if isinstance(current_result["data"], list) and not current_result["data"]:
                                        st.warning("Cypher query executed but no results found. Fallback to LLM...")
                                        result = None  # Resetta il risultato per attivare il fallback
                                    else:
                                        # Abbiamo risultati non vuoti, memorizzali in cache e usali
                                        st.session_state.qdrant_cache.store_query_and_response(
                                            question=question,
                                            cypher_query=cypher_query,
                                            response=current_result["data"],
                                            template_used=cache_hit.template_used
                                        )
                                        st.session_state.llm_result = {
                                            **current_result,
                                            "cached": True, # È un hit della cache semantica (query Cypher)
                                            "strategy": cache_hit.strategy,
                                            "confidence": cache_hit.confidence
                                        }
                                        st.success("✅ Query successfully executed from the database!")
                                        result = current_result # Imposta result per evitare il fallback LLM
                                else:
                                    st.warning("No results from the Cypher query. Fallback to LLM...")
                                    result = None # Attiva il fallback LLM
                            except Exception as e:
                                st.error(f"Error executing Cypher query: {e}. Fallback to LLM...")
                                result = None # Attiva il fallback LLM
                        
                        # Caso 3: Nessun match della cache (cypher_query vuota) o fallback da strategia precedente (result è None), usa l'LLM
                        if not result: # Se 'result' è ancora None, significa che dobbiamo usare l'LLM
                            st.info("Generating a new query with LLM...")
                            try:
                                result = ask_neo4j_llm(
                                    question=question,
                                    data_schema=graph_schema
                                )
                                if result and result.get("data"):
                                    # Memorizza la nuova query e risposta in cache
                                    st.session_state.qdrant_cache.store_query_and_response(
                                        question=question,
                                        cypher_query=result["cypher_query"],
                                        response=result["data"]
                                    )
                                    st.session_state.llm_result = {
                                        **result,
                                        "cached": False,
                                        "strategy": "llm",
                                        "confidence": 1.0
                                    }
                                    st.success("🚀 Query generated and cached!")
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
            strategy = result.get("strategy", "unknown")
            confidence = result.get("confidence", 0.0)
            
            st.markdown(
                f'{get_strategy_badge(strategy)} Query processed successfully! '
                f'<small>(Confidence: {confidence:.2%})</small>', 
                unsafe_allow_html=True
            )

            st.write("**Cypher Query:**")
            st.code(result["cypher_query"], language="cypher")

            st.write("**Results:**")
            table_data = format_results_as_table(result["data"])
            if table_data:
                try:
                    df = pd.DataFrame(table_data)
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

            # Inizializza e aggiorna i threshold tramite session_state
            st.session_state.template_threshold = st.slider(
                "Template Match Threshold",
                min_value=0.5,
                max_value=1.0,
                value=st.session_state.get('template_threshold', 0.75),
                step=0.05,
                format="%.2f",
                key='template_threshold_slider',
                help="Minimum confidence for template matching. Lower values allow more flexible matching.",
            )

            st.session_state.exact_threshold = st.slider(
                "Exact Match Threshold",
                min_value=0.9,
                max_value=1.0,
                value=st.session_state.get('exact_threshold', 0.95),
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
                help="Minimum string similarity (Levenshtein) for fuzzy matching (70-100).",
            )

            st.session_state.similarity_threshold = st.slider(
                "Similarity Search Threshold",
                min_value=0.7,
                max_value=0.95,
                value=st.session_state.get('similarity_threshold', 0.8),
                step=0.01,
                format="%.3f",
                key='similarity_threshold_slider',
                help="Minimum similarity for fallback semantic search.",
            )
            
            st.info(
                "🧠 **Cache Strategies:**\n\n"
                "0. **Recent Cache** - Last 3 queries (in-memory)\n"
                "1. **Template Matching** - Pattern-based Cypher generation\n"
                "2. **Exact Match** - Previously cached identical queries\n" 
                "3. **Semantic Similarity** - Semantically related queries\n"
                "4. **Fuzzy Matching** - String similarity (Levenshtein)\n"
                "5. **LLM Fallback** - Generate new query when needed"
            )

        with st.sidebar.expander("Cache Management"):
            col_cache_a, col_cache_b = st.columns(2)
            
            with col_cache_a:
                if st.button("Reset Cache"):
                    try:
                        # Delete collections
                        if st.session_state.qdrant_cache:
                            st.session_state.qdrant_cache.qdrant_client.delete_collection(
                                collection_name=st.session_state.qdrant_cache.collection_name
                            )
                            st.session_state.qdrant_cache.qdrant_client.delete_collection(
                                collection_name=st.session_state.qdrant_cache.template_collection
                            )
                        
                        # Reinitialize
                        st.session_state.qdrant_cache = initialize_cache()
                        if st.session_state.qdrant_cache:
                            st.success("Cache reset successfully!")
                        else:
                            st.error("Error reinitializing cache.")
                    except Exception as e:
                        st.error(f"Error resetting cache: {e}")
            
            with col_cache_b:
                if st.button("Clear Memory"):
                    if st.session_state.qdrant_cache:
                        st.session_state.qdrant_cache.clear_caches()
                        st.success("Memory caches cleared!")
                    else:
                        st.warning("No cache instance available.")

            st.info("**Reset Cache:** Deletes all stored queries and templates\n"
                   "**Clear Memory:** Clears in-memory caches only")

        with st.sidebar.expander("Performance Analytics"):
            if st.button("View Performance Stats"):
                if st.session_state.qdrant_cache:
                    with st.spinner("Loading performance analytics..."):
                        try:
                            perf_stats = st.session_state.qdrant_cache.get_performance_stats()

                            if "error" in perf_stats:
                                st.error(f"Error: {perf_stats['error']}")
                            else:
                                # Cache Performance
                                st.subheader("🎯 Cache Performance")
                                perf = perf_stats["cache_performance"]

                                col_perf_a, col_perf_b = st.columns(2)
                                with col_perf_a:
                                    st.metric("Hit Rate", f"{perf['hit_rate_percentage']:.1f}%")
                                    st.metric("Template Hits", perf["template_hits"])
                                    st.metric("Recent Cache Hits", perf["recent_cache_hits"])

                                with col_perf_b:
                                    st.metric("Embedding Hits", perf["embedding_cache_hits"])
                                    st.metric("Embedding Misses", perf["embedding_cache_misses"])
                                    st.metric("Fuzzy Hits", perf["fuzzy_hits"])

                                st.metric("Total Search Hits", perf["total_search_hits"])

                                # Storage Info
                                st.subheader("💾 Storage")
                                storage = perf_stats["storage"]

                                col_stor_a, col_stor_b = st.columns(2)
                                with col_stor_a:
                                    st.metric("Cached Queries", storage["total_cached_queries"])
                                    st.metric("Templates in Qdrant", storage["templates_in_qdrant"])
                                    st.metric("Vector Size", storage["vector_size"])

                                with col_stor_b:
                                    st.metric("Embedding Cache", storage["embedding_cache_size"])
                                    st.metric("Frequent Cache", storage["frequent_queries_cache_size"])
                                    st.metric("Recent Cache", storage["recent_results_cache_size"])

                                # Qdrant Metrics (NEW!)
                                st.subheader("⚡ Qdrant Optimization Metrics")
                                qdrant = perf_stats["qdrant_metrics"]

                                col_q1, col_q2 = st.columns(2)
                                with col_q1:
                                    st.metric("Indexed Vectors", qdrant["indexed_vectors_count"])
                                    st.metric("Total Vectors", qdrant["vectors_count"])
                                    st.metric("Segments", qdrant["segments_count"])

                                with col_q2:
                                    st.text(f"Status: {qdrant['status']}")
                                    st.text(f"Optimizer: {qdrant['optimizer_status']}")

                                # HNSW Configuration
                                st.markdown("**HNSW Configuration:**")
                                config = qdrant["config"]
                                st.text(f"  • M (edges): {config['hnsw_m']}")
                                st.text(f"  • EF Construct: {config['hnsw_ef_construct']}")
                                st.text(f"  • Distance: {config['distance']}")

                                # Quantization Info
                                st.markdown("**Quantization:**")
                                if config["quantization_enabled"]:
                                    st.success(f"✅ {config['quantization_type']} enabled")
                                    st.info("Memory usage reduced by ~75% with scalar quantization")
                                else:
                                    st.warning("❌ Quantization disabled")

                                st.text(f"On-Disk Storage: {'Yes' if config['on_disk'] else 'No'}")

                                # Model Info
                                st.subheader("🤖 Model Configuration")
                                model_info = perf_stats["model_info"]
                                efficiency = perf_stats["efficiency"]

                                st.text(f"Embedder: {model_info['embedder_model']}")
                                st.text(f"NLP Enabled: {'Yes' if model_info['nlp_enabled'] else 'No'}")
                                st.text(f"Fuzzy Matching: {'Available' if model_info['fuzzy_matching_available'] else 'Not Available'}")
                                st.text(f"Templates Loaded: {efficiency['templates_loaded']}")
                                st.text(f"Avg Template Priority: {efficiency['avg_template_priority']:.1f}")

                                # Memory Efficiency
                                st.markdown("**Memory Efficiency:**")
                                mem_eff = efficiency["memory_efficiency"]
                                st.text(f"  • Max Embedding Cache: {mem_eff['max_embedding_cache']}")
                                st.text(f"  • Max Frequent Cache: {mem_eff['max_frequent_cache']}")
                                st.text(f"  • Max Recent Cache: {mem_eff['max_recent_cache']}")

                        except Exception as e:
                            st.error(f"Error loading performance stats: {e}")
                else:
                    st.warning("No cache instance available.")

            if st.button("Export Analytics"):
                if st.session_state.qdrant_cache:
                    try:
                        perf_stats = st.session_state.qdrant_cache.get_performance_stats()
                        # Create downloadable JSON
                        json_str = json.dumps(perf_stats, indent=2)
                        st.download_button(
                            label="Download Analytics JSON",
                            data=json_str,
                            file_name="cache_analytics.json",
                            mime="application/json"
                        )
                    except Exception as e:
                        st.error(f"Error exporting analytics: {e}")
                else:
                    st.warning("No cache instance available.")

        # NEW: Batch Operations
        with st.sidebar.expander("Batch Operations & Testing"):
            st.subheader("🚀 Batch Query Testing")

            # Batch insert
            st.markdown("**Batch Insert Queries**")
            batch_file = st.file_uploader(
                "Upload JSON file with queries",
                type=["json"],
                help="Upload a JSON file with format: [{\"question\": \"...\", \"cypher_query\": \"...\", \"response\": \"...\"}]"
            )

            if batch_file is not None:
                try:
                    batch_data = json.load(batch_file)
                    st.info(f"Loaded {len(batch_data)} queries from file")

                    if st.button("Insert Batch"):
                        with st.spinner(f"Inserting {len(batch_data)} queries..."):
                            success = st.session_state.qdrant_cache.store_batch_queries(batch_data)
                            if success:
                                st.success(f"✅ Successfully inserted {len(batch_data)} queries!")
                            else:
                                st.error("Failed to insert batch queries")
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON format: {e}")
                except Exception as e:
                    st.error(f"Error loading file: {e}")

            # Sample batch format
            if st.button("Show Example Batch Format"):
                example = [
                    {
                        "question": "Who are the employees?",
                        "cypher_query": "MATCH (e:Employee) RETURN e",
                        "response": "List of employees...",
                        "template_used": "list_employees"
                    },
                    {
                        "question": "Show all projects",
                        "cypher_query": "MATCH (p:Project) RETURN p",
                        "response": "List of projects...",
                        "template_used": None
                    }
                ]
                st.json(example)

            # Async batch search
            st.markdown("**Async Batch Search (Concurrent)**")
            batch_questions = st.text_area(
                "Enter questions (one per line)",
                height=100,
                help="Enter multiple questions to search concurrently using async operations"
            )

            if st.button("Run Async Batch Search"):
                if batch_questions:
                    questions_list = [q.strip() for q in batch_questions.split('\n') if q.strip()]

                    if questions_list:
                        import asyncio

                        with st.spinner(f"Searching {len(questions_list)} queries concurrently..."):
                            try:
                                # Run async batch search
                                results = asyncio.run(
                                    st.session_state.qdrant_cache.async_batch_search(questions_list)
                                )

                                st.success(f"✅ Completed {len(results)} searches!")

                                # Display results
                                for i, (question, result) in enumerate(zip(questions_list, results)):
                                    with st.expander(f"Query {i+1}: {question[:50]}..."):
                                        st.markdown(f"**Strategy:** {get_strategy_badge(result.strategy)}", unsafe_allow_html=True)
                                        st.metric("Confidence", f"{result.confidence:.2%}")
                                        if result.cypher_query:
                                            st.code(result.cypher_query, language="cypher")
                                        else:
                                            st.warning("No match found")
                            except Exception as e:
                                st.error(f"Error in async batch search: {e}")
                    else:
                        st.warning("Please enter at least one question")
                else:
                    st.warning("Please enter questions to search")

        # Keep existing GDS functionality
        with st.sidebar.expander("Create Graph Projection"):
            graph_name = st.text_input("Projection Name")
            node_labels_input = st.text_area("Node Labels (separated by commas)")
            relationship_types_input = st.text_area("Relationship Types (separated by commas)")

            if st.button("Create Projection"):
                gds = st.session_state.gds

                node_labels_list = [label.strip() for label in node_labels_input.split(",") if label.strip()]
                relationship_types_list = [rel.strip() for rel in relationship_types_input.split(",") if rel.strip()]

                if not graph_name:
                    st.warning("Please enter a projection name.")
                elif not node_labels_list:
                    st.warning("Please enter at least one node label.")
                elif not relationship_types_list:
                    st.warning("Please enter at least one relationship type.")
                else:
                    with st.spinner("Creating graph projection..."):
                        msg = gds.create_projection(graph_name, node_labels_list, relationship_types_list)
                        st.success(msg if "✅" in msg else msg)

            if st.button("Delete All Projections"):
                with st.spinner("Deleting all GDS projections..."):
                    msg = st.session_state.gds.delete_all_projections()
                    st.success(msg if "✅" in msg else msg)

        with st.sidebar.expander("Run GDS Algorithm"):
            gds_graph_name = st.text_input("Graph name for GDS algorithm")
            algo = st.selectbox(
                "Choose GDS Algorithm",
                ["PageRank", "Betweenness Centrality", "Closeness Centrality", "Louvain", "Node Similarity"]
            )

            if st.button("Run Algorithm"):
                if not gds_graph_name:
                    st.sidebar.warning("Please provide a graph projection name.")
                else:
                    with st.spinner(f"Running {algo} on {gds_graph_name}..."):
                        gds = st.session_state.gds

                        if algo == "PageRank":
                            gds_results = gds.run_pagerank(gds_graph_name)
                        elif algo == "Betweenness Centrality":
                            gds_results = gds.run_betweenness(gds_graph_name)
                        elif algo == "Closeness Centrality":
                            gds_results = gds.run_closeness(gds_graph_name)
                        elif algo == "Louvain":
                            gds_results = gds.run_louvain(gds_graph_name)
                        elif algo == "Node Similarity":
                            gds_results = gds.run_similarity(gds_graph_name)
                        else:
                            gds_results = [{"error": "Unknown algorithm selected."}]

                        if isinstance(gds_results, list) and "error" in gds_results[0]:
                            st.error(gds_results[0]["error"])
                        else:
                            st.subheader(f"Results of {algo}:")
                            df = pd.DataFrame(gds_results)
                            st.dataframe(df)

if __name__ == "__main__":
    main()