import os
import json

import streamlit as st
import pandas as pd

from backend.gemini import ask_neo4j_gemini, get_schema
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
        .st-bb {{ 
            border-color: {PRIMARY_COLOR};
            color: {TEXT_COLOR};
        }}
        .st-c8, .st-c7, .st-b6, .st-b5, .st-b4, .st-b3, .st-b2, .st-b1, .st-b0 {{
            border-color: {PRIMARY_COLOR} !important;
            box-shadow: none !important;
            color: {TEXT_COLOR};
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
        .streamlit-expanderHeader {{
            color: {PRIMARY_COLOR};
        }}
        input {{
            color: {TEXT_COLOR} !important;
        }}
        textarea {{
            color: {TEXT_COLOR} !important;
        }}
        select {{
            color: {TEXT_COLOR} !important;
        }}
        .markdown-box {{
                border: 2px solid {PRIMARY_COLOR};
                padding: 10px;
                border-radius: 5px;
                background-color: white;
                overflow-x: auto;
        }}
        .tree-output {{
            background-color: white;
            border: 1px solid #ddd;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
            font-family: monospace;
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

    col1, col2 = st.columns([3, 1])

    with col1:
        sample_questions = generate_sample_questions()

        selected_question = st.selectbox(
            "Select an example question:", list(sample_questions.keys())
        )

        question = st.text_area(
            "Enter your question in natural language:",
            value=sample_questions[selected_question],
            height=150,
        )

        # Open and read the JSON file
        with open(os.environ.get("NODE_CONTEXT_URL"), "r") as file:
            nodes = json.load(file)

        with open(os.environ.get("REL_CONTEXT_URL"), "r") as file:
            relationship = json.load(file)

        graph_schema = get_schema(nodes=nodes, relations=relationship)

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

        if st.button("Run"):
            if question:
                cached_result = st.session_state.qdrant_cache.search_similar_question(
                    question=question, threshold=threshold
                )
                if cached_result:
                    st.info("Similar query found in cache.")
                    cypher_query = cached_result["cypher_query"]
                    results = {
                        "data": cached_result["response"],
                        "cypher_query": cypher_query,
                    }
                else:
                    st.info("Cypher query generation from LLM.")
                    results = ask_neo4j_gemini(
                        question=question, data_schema=graph_schema
                    )
                    if results["data"]:
                        st.session_state.qdrant_cache.store_query_and_response(
                            question=question,
                            cypher_query=results["cypher_query"],
                            response=results["data"],
                        )
                        st.success("Cached query and answer.")
                    else:
                        st.warning("No data to store.")

                if isinstance(results, str):
                    st.error(results)
                else:
                    st.write("Query Cypher generated:")
                    st.code(results["cypher_query"], language="cypher")

                    st.write("Table Results:")
                    table_data = format_results_as_table(results["data"])
                    if table_data:
                        try:
                            df = pd.DataFrame(table_data)
                            if df.shape[0] > 100:
                                st.dataframe(
                                    df.head(100),
                                )
                                st.write(
                                    f"Show the first 100 lines out of {df.shape[0]}"
                                )
                            else:
                                st.dataframe(df)
                        except Exception as e:
                            st.write(f"The results are not viewable as a table: {e}")
                    else:
                        st.write("No results found.")

    with col2:
        if "stop_processing" not in st.session_state:
            st.session_state.stop_processing = False

        stop_button_pressed = st.button("Stop App", key="stop_button")

        if stop_button_pressed:
            st.session_state.stop_processing = True
            st.warning("User aborted processing.")

        if st.session_state.stop_processing:
            st.info("Processing has been stopped.")


if __name__ == "__main__":
    main()
