
import os
import json

import streamlit as st
import pandas as pd

from backend.llm import (
    ask_neo4j_gemini,
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
    </style>
    """,
    unsafe_allow_html=True,
)

def main():
    st.image(LOGO_PATH, width=200)
    st.title("ArchAI - CypherMind")
    st.subheader("Translate Natural Language to Cypher Query")

    if "qdrant_cache" not in st.session_state:
        st.session_state.qdrant_cache = SemanticCache(
            collection_name=os.environ.get("QDRANT_COLLECTION"),
            mode=os.environ.get("QDRANT_MODE"),
            embedder=os.environ.get("EMBEDDER"),
            vector_size=os.environ.get("VECTOR_SIZE"),
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

        if st.button("Clear Previous Result"):
            st.session_state.llm_result = None

        selected_question = st.selectbox(
            "Select an example question:", list(sample_questions.keys())
        )

        question = st.text_area(
            "Enter your question in natural language:",
            value=sample_questions[selected_question],
            height=150,
        )

        with open(os.environ.get("NODE_CONTEXT_URL"), "r") as file:
            nodes = json.load(file)

        with open(os.environ.get("REL_CONTEXT_URL"), "r") as file:
            relationship = json.load(file)

        st.session_state.gds.get_schema(nodes)
        graph_schema = get_schema(nodes=nodes, relations=relationship)

        if st.button("Run"):
            if question:
                cached_result = st.session_state.qdrant_cache.search_similar_question(
                    question=question, threshold=0.995
                )

                if cached_result:
                    st.info("Similar query found in cache.")
                    st.session_state.llm_result = {
                        "data": cached_result["response"],
                        "cypher_query": cached_result["cypher_query"],
                        "cached": True
                    }
                else:
                    st.info("Generating Cypher query from LLM...")
                    with st.spinner("Generating query..."):
                        try:
                            result = ask_neo4j_gemini(
                                question=question,
                                data_schema=graph_schema
                            )
                            if isinstance(result, dict) and result["data"]:
                                st.session_state.qdrant_cache.store_query_and_response(
                                    question=question,
                                    cypher_query=result["cypher_query"],
                                    response=result["data"],
                                )
                                st.session_state.llm_result = {
                                    **result,
                                    "cached": False
                                }
                                st.success("Query stored in semantic cache.")
                            else:
                                st.session_state.llm_result = None
                                st.warning("No results returned.")
                        except Exception as e:
                            st.session_state.llm_result = None
                            st.error(f"Error: {e}")

        if st.session_state.llm_result:
            result = st.session_state.llm_result
            label = "cached" if result.get("cached") else "generated"
            st.success(f"Query {label} successfully.")

            st.write("Query Cypher:")
            st.code(result["cypher_query"], language="cypher")

            st.write("Table Results:")
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
        with st.sidebar.expander("Advanced Options"):
            threshold = st.slider(
                "Similarity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.995,
                step=0.001,
                format="%.3f",
                help="Sets the threshold for searching the cache for similar queries. Higher values require higher similarity.",
            )

            if st.button("Reset Semantic Cache"):
                st.session_state.qdrant_cache.qdrant_client.delete_collection(
                    collection_name=st.session_state.qdrant_cache.collection_name
                )
                st.session_state.qdrant_cache = SemanticCache(
                    collection_name=os.environ.get("QDRANT_COLLECTION"),
                    mode=os.environ.get("QDRANT_MODE"),
                    embedder=os.environ.get("EMBEDDER"),
                    vector_size=os.environ.get("VECTOR_SIZE"),
                )
                st.success("Semantic cache successfully reset!")

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

if __name__ == "__main__":
    main()
