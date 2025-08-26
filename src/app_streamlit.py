import os
import json

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
from backend.semantic_cache import SemanticCache

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
    </style>
    """,
    unsafe_allow_html=True,
)

def get_strategy_badge(strategy):
    """Return HTML badge for the strategy used"""
    badges = {
        "template": '<span class="strategy-badge template-badge">TEMPLATE</span>',
        "exact_match": '<span class="strategy-badge exact-badge">EXACT MATCH</span>',
        "semantic_similarity": '<span class="strategy-badge similar-badge">SIMILAR</span>',
        "semantic_signature": '<span class="strategy-badge signature-badge">SIGNATURE</span>',
        "llm": '<span class="strategy-badge llm-badge">LLM GENERATED</span>'
    }
    return badges.get(strategy, "")

def display_cache_result_info(search_result):
    """Display information about cache search results"""
    if search_result.get("found_template"):
        st.markdown(
            f'<div class="cache-info">'
            f'{get_strategy_badge("template")}'
            f'<strong>Query template matched!</strong> Generated Cypher directly from pattern without LLM call.'
            f'<br><small>Confidence: {search_result["confidence"]:.2%}</small>'
            f'</div>',
            unsafe_allow_html=True
        )
    elif search_result.get("found_exact"):
        st.markdown(
            f'<div class="cache-info">'
            f'{get_strategy_badge("exact_match")}'
            f'<strong>Exact match found in cache!</strong> Retrieved previously processed query.'
            f'<br><small>Confidence: {search_result["confidence"]:.2%}</small>'
            f'</div>',
            unsafe_allow_html=True
        )
    elif search_result.get("found_similar"):
        strategy = search_result.get("strategy_used", "semantic_similarity")
        st.markdown(
            f'<div class="cache-info">'
            f'{get_strategy_badge(strategy)}'
            f'<strong>Similar query found!</strong> Using cached result with {strategy.replace("_", " ")}.'
            f'<br><small>Confidence: {search_result["confidence"]:.2%}</small>'
            f'</div>',
            unsafe_allow_html=True
        )

def main():
    st.image(LOGO_PATH, width=200)
    st.title("ArchAI - CypherMind")
    st.subheader("Translate Natural Language to Cypher Query")

    # Initialize Semantic Cache
    if "qdrant_cache" not in st.session_state:
        st.session_state.qdrant_cache = SemanticCache(
            collection_name=os.environ.get("QDRANT_COLLECTION"),
            mode=os.environ.get("QDRANT_MODE"),
            embedder=os.environ.get("EMBEDDER"),
            vector_size=int(os.environ.get("VECTOR_SIZE")),
        )

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
        
        if "last_search_result" not in st.session_state:
            st.session_state.last_search_result = None

        if st.button("Clear Previous Result"):
            st.session_state.llm_result = None
            st.session_state.last_search_result = None

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
            if question:
                with st.spinner("Searching cache and processing query..."):
                    # Use smart search
                    search_result = st.session_state.qdrant_cache.smart_search(question)
                    st.session_state.last_search_result = search_result
                    
                    if (search_result["found_template"] or 
                        search_result["found_exact"] or 
                        search_result["found_similar"]):
                        
                        # Use cached/template result
                        cypher_query = search_result["cypher_query"]
                        
                        if search_result.get("response"):
                            # We have a complete cached result
                            st.session_state.llm_result = {
                                "data": search_result["response"],
                                "cypher_query": cypher_query,
                                "cached": True,
                                "strategy": search_result.get("strategy_used", "unknown")
                            }
                        else:
                            # We have Cypher but need to execute it
                            st.info("Executing generated Cypher query...")
                            try:
                                # Execute the Cypher query
                                result = ask_neo4j_llm(
                                    question=question,
                                    data_schema=graph_schema,
                                    override_cypher=cypher_query
                                )
                                
                                if isinstance(result, dict) and result["data"]:
                                    # Store the complete result
                                    template_used = search_result.get("strategy_used") if search_result["found_template"] else None
                                    st.session_state.qdrant_cache.store_query_and_response(
                                        question=question,
                                        cypher_query=cypher_query,
                                        response=result["data"],
                                        template_used=template_used
                                    )
                                    
                                    st.session_state.llm_result = {
                                        **result,
                                        "cached": search_result["found_exact"],
                                        "strategy": search_result.get("strategy_used", "template")
                                    }
                                else:
                                    st.session_state.llm_result = None
                                    st.warning("No results returned from query execution.")
                            except Exception as e:
                                st.session_state.llm_result = None
                                st.error(f"Error executing Cypher query: {e}")
                    else:
                        # No cache hit, use LLM
                        st.info("No suitable cache match found. Generating Cypher query from LLM...")
                        try:
                            result = ask_neo4j_llm(
                                question=question,
                                data_schema=graph_schema
                            )
                            if isinstance(result, dict) and result["data"]:
                                # Store the new result
                                st.session_state.qdrant_cache.store_query_and_response(
                                    question=question,
                                    cypher_query=result["cypher_query"],
                                    response=result["data"],
                                )
                                st.session_state.llm_result = {
                                    **result,
                                    "cached": False,
                                    "strategy": "llm"
                                }
                                st.success("New query generated and stored in semantic cache.")
                            else:
                                st.session_state.llm_result = None
                                st.warning("No results returned from LLM.")
                        except Exception as e:
                            st.session_state.llm_result = None
                            st.error(f"Error: {e}")

        # Display cache search result info
        if st.session_state.last_search_result:
            display_cache_result_info(st.session_state.last_search_result)

        if st.session_state.llm_result:
            result = st.session_state.llm_result
            strategy = result.get("strategy", "unknown")
            
            st.markdown(f'{get_strategy_badge(strategy)} Query processed successfully!', 
                       unsafe_allow_html=True)

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
            else:
                st.write("No results found.")

    with col2:
        with st.sidebar.expander("Advanced Options", expanded=True):
            st.subheader("Cache Settings")
            
            # Template matching threshold
            template_threshold = st.slider(
                "Template Match Threshold",
                min_value=0.5,
                max_value=1.0,
                value=0.75,
                step=0.05,
                format="%.2f",
                help="Threshold for matching query templates. Lower values allow more flexible template matching.",
            )
            
            # Similarity threshold for fallback search
            similarity_threshold = st.slider(
                "Similarity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.98,
                step=0.01,
                format="%.3f",
                help="Threshold for semantic similarity search when no template matches.",
            )
            
            st.info("The cache uses multiple strategies:\n"
                   "1. **Template Matching** - Direct pattern recognition\n"
                   "2. **Exact Match** - Previously seen identical queries\n"
                   "3. **Semantic Signature** - Queries with same structure\n"
                   "4. **Similarity Search** - Semantically similar queries")

        with st.sidebar.expander("Cache Management"):
            if st.button("Reset Semantic Cache"):
                # Reset both main and template collections
                try:
                    st.session_state.qdrant_cache.qdrant_client.delete_collection(
                        collection_name=st.session_state.qdrant_cache.collection_name
                    )
                    st.session_state.qdrant_cache.qdrant_client.delete_collection(
                        collection_name=st.session_state.qdrant_cache.template_collection
                    )
                except:
                    pass  # Collections might not exist
                
                # Reinitialize cache
                st.session_state.qdrant_cache = SemanticCache(
                    collection_name=os.environ.get("QDRANT_COLLECTION"),
                    mode=os.environ.get("QDRANT_MODE"),
                    embedder=os.environ.get("EMBEDDER"),
                    vector_size=int(os.environ.get("VECTOR_SIZE")),
                )
                st.success("Semantic cache successfully reset!")
            
            if st.button("Add Custom Template"):
                st.info("Feature coming soon! Templates can be added programmatically.")

        with st.sidebar.expander("Create Graph Projection"):
            graph_name = st.text_input("Projection Name")
            node_labels_input = st.text_area("Node Labels (separated by commas)")
            relationship_types_input = st.text_area("Relationship Types (separated by commas)")

            if st.button("Create Projection"):
                gds = st.session_state.gds

                node_labels_list = [label.strip() for label in node_labels_input.split(",") if label.strip()]
                relationship_types_list = [rel.strip() for rel in relationship_types_input.split(",") if rel.strip()]

                if not graph_name:
                    st.warning("⚠️ Please enter a projection name.")
                elif not node_labels_list:
                    st.warning("⚠️ Please enter at least one node label.")
                elif not relationship_types_list:
                    st.warning("⚠️ Please enter at least one relationship type.")
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

        with st.sidebar.expander("Cache Analytics"):
            if st.button("View Cache Statistics"):
                with st.spinner("Loading cache statistics..."):
                    cache_stats = st.session_state.qdrant_cache.get_cache_stats()
                    
                    if "error" in cache_stats:
                        st.error(f"Error retrieving cache stats: {cache_stats['error']}")
                    else:
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.metric("Cached Queries", cache_stats.get("total_cached_queries", 0))
                            st.metric("Templates", cache_stats.get("total_templates", 0))
                        
                        with col_b:
                            st.metric("Vector Size", cache_stats.get("vector_size", 0))
                            st.metric("Embedding Cache", cache_stats.get("embedding_cache_size", 0))
                        
                        st.text(f"Model: {cache_stats.get('embedder_model', 'Unknown')}")
                        
                        if cache_stats.get("top_semantic_signatures"):
                            st.subheader("Top Query Patterns")
                            for signature, count in cache_stats["top_semantic_signatures"]:
                                st.text(f"• {signature}: {count} queries")
                        
                        if cache_stats.get("sample_questions"):
                            st.subheader("Recent Cached Questions")
                            for q in cache_stats["sample_questions"]:
                                display_q = f"{q[:60]}..." if len(q) > 60 else q
                                st.text(f"• {display_q}")

            if st.button("Export Cache Data"):
                st.info("Export functionality coming soon!")

if __name__ == "__main__":
    main()