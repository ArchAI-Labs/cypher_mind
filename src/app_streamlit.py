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

# Se hai codice che importa EnhancedSemanticCache, aggiungi questo alias
# EnhancedSemanticCache = SemanticCache

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
        "llm": '<span class="strategy-badge llm-badge">LLM GENERATED</span>'
    }
    return badges.get(strategy, f'<span class="strategy-badge">{strategy.upper()}</span>')

def display_cache_result_info(cache_hit):
    """Display information about cache search results using new CacheHit object"""
    if cache_hit.strategy == "template_match":
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
    st.title("ArchAI - CypherMind Enhanced")
    st.subheader("Natural Language to Cypher with Smart Semantic Caching")

    # Initialize Enhanced Semantic Cache
    if "qdrant_cache" not in st.session_state:
        with st.spinner("Initializing enhanced semantic cache..."):
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

                    # Usa l'enhanced smart search con i threshold dell'interfaccia utente
                    cache_hit = st.session_state.qdrant_cache.smart_search(
                        question,
                        template_threshold=template_threshold,
                        exact_threshold=exact_threshold,
                        similarity_threshold=similarity_threshold
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
                        st.success("🎉 Response trovata in cache!")
                    else:
                        # Caso 2: Trovata una query Cypher, ma non la response o la response era vuota
                        # Questa parte viene eseguita solo se non c'è una response completa in cache
                        if cypher_query and cache_hit.confidence > 0.0:
                            st.info(f"Eseguendo query Cypher da cache ({cache_hit.strategy})...")
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
                                        st.warning("Query Cypher eseguita ma non ha trovato risultati. Fallback all'LLM...")
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
                                        st.success("✅ Query eseguita con successo dal database!")
                                        result = current_result # Imposta result per evitare il fallback LLM
                                else:
                                    st.warning("Nessun risultato dalla query Cypher. Fallback all'LLM...")
                                    result = None # Attiva il fallback LLM
                            except Exception as e:
                                st.error(f"Errore nell'esecuzione della query Cypher: {e}. Fallback all'LLM...")
                                result = None # Attiva il fallback LLM
                        
                        # Caso 3: Nessun match della cache (cypher_query vuota) o fallback da strategia precedente (result è None), usa l'LLM
                        if not result: # Se 'result' è ancora None, significa che dobbiamo usare l'LLM
                            st.info("Generando una nuova query con LLM...")
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
                                    st.success("🚀 Query generata e memorizzata in cache!")
                                else:
                                    st.warning("Nessun risultato dall'LLM.")
                                    st.session_state.llm_result = None
                            except Exception as e:
                                st.error(f"Errore con il modello LLM: {e}")
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
        with st.sidebar.expander("Enhanced Cache Settings", expanded=True):
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
                "🧠 **Enhanced Cache Strategies:**\n\n"
                "1. **Template Matching** - Pattern-based Cypher generation\n"
                "2. **Exact Match** - Previously cached identical queries\n" 
                "3. **Semantic Signature** - Structurally similar queries\n"
                "4. **Similarity Search** - Semantically related queries\n"
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
                                st.subheader("Cache Performance")
                                perf = perf_stats["cache_performance"]
                                
                                col_perf_a, col_perf_b = st.columns(2)
                                with col_perf_a:
                                    st.metric("Hit Rate", f"{perf['hit_rate_percentage']:.1f}%")
                                    st.metric("Template Hits", perf["template_hits"])
                                
                                with col_perf_b:
                                    st.metric("Cache Hits", perf["embedding_cache_hits"])
                                    st.metric("Cache Misses", perf["embedding_cache_misses"])
                                
                                # Storage Info
                                st.subheader("Storage")
                                storage = perf_stats["storage"]
                                
                                col_stor_a, col_stor_b = st.columns(2)
                                with col_stor_a:
                                    st.metric("Cached Queries", storage["total_cached_queries"])
                                    st.metric("Vector Size", storage["vector_size"])
                                
                                with col_stor_b:
                                    st.metric("Embedding Cache", storage["embedding_cache_size"])
                                    st.metric("Frequent Cache", storage["frequent_queries_cache_size"])
                                
                                # Model Info
                                st.subheader("Model Configuration")
                                model_info = perf_stats["model_info"]
                                efficiency = perf_stats["efficiency"]
                                
                                st.text(f"Embedder: {model_info['embedder_model']}")
                                st.text(f"NLP Enabled: {'Yes' if model_info['nlp_enabled'] else 'No'}")
                                st.text(f"Templates Loaded: {efficiency['templates_loaded']}")
                                st.text(f"Avg Template Priority: {efficiency['avg_template_priority']:.1f}")
                        
                        except Exception as e:
                            st.error(f"Error loading performance stats: {e}")
                else:
                    st.warning("No cache instance available.")
            
            if st.button("Export Analytics"):
                st.info("Analytics export feature coming soon!")

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