import os
from dotenv import load_dotenv
import json

from neo4j import GraphDatabase
from google import genai
from google.genai import types

load_dotenv()

uri = os.environ.get("NEO4J_URI")
user = os.environ.get("NEO4J_USER")
password = os.environ.get("NEO4J_PASSWORD")

gemini_api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_api_key)


def get_schema(nodes, relations):
    try:
        system_prompt = """
            You are a helpful assistant that describes the schema of a Neo4j graph database based on provided JSON structures. 
            You will receive two JSON strings: one describing the node labels and their properties, and the other describing the relationship types. 
            Your goal is to format this information into a human-readable string as specified in the user's example.
            Instructions:
            1. Extract the node labels from the keys of the node properties JSON.
            2. Extract the relationship types from the keys of the relationship properties JSON.
            3. For each node label, list its associated properties.
            4. Format the output string exactly as shown in the user's example, including the "- Nodes:", "- Relationships:", and "- Relevant Properties:" sections with appropriate indentation.
            5. If a node label or relationship type has no properties, indicate this clearly (e.g., "- NodeLabel:") or simply don't list any properties under it.
        """
        prompt = f"""
            Here is the JSON describing the node labels and their properties: {nodes}
            And here is the JSON describing the relationship types: {relations}
            Based on these JSON structures, generate a string describing the Neo4j database schema in the following format:

            ```
            The Neo4j database has the following schema:
                - Nodes: ...
                - Relationships: ...
                - Relevant Properties:
                    - NodeLabel: property1, property2, ...
                    - AnotherNodeLabel: ...
                    - ...
            ```
        """

        response = client.models.generate_content(
            model=os.environ.get("MODEL"),
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                # max_output_tokens=2048,
                top_k=2,
                top_p=0.5,
                temperature=0.2,
                seed=42,
            ),
        )

        return response.text

    except Exception as e:
        return f"An error occurred: {e}"


def get_system_prompt(data_schema: str):
    prefix = """
        You are a highly skilled expert in translating natural language questions into Cypher queries for Neo4j graph databases. 
        Your primary goal is to understand the user's intent and accurately represent it in a functional and efficient Cypher query.

        When presented with a natural language question about a Neo4j database:

        1. **Understand the Intent:** Carefully analyze the question to identify the core information being requested, the entities involved, and the relationships between them. Consider potential synonyms and different ways the user might express the same need.
        2. **Map to Graph Concepts:** Based on your understanding of the underlying graph schema (node labels, relationship types, and properties), determine how the elements of the natural language question correspond to elements within the graph.
        3. **Formulate the Cypher Logic:** Construct the necessary Cypher clauses (MATCH, WHERE, RETURN, etc.) to retrieve the desired information based on the mapping in the previous step.
        4. **Prioritize Accuracy:** Ensure the generated Cypher query will return the correct results that directly answer the user's question, without including irrelevant data or missing key information.
        5. **Aim for Efficiency:** While accuracy is paramount, strive to generate Cypher queries that are reasonably efficient for the expected size and complexity of the graph.

        Your final output should be the well-formed Cypher query that is the most accurate and effective translation of the user's natural language question.
        """

    suffix = """
        Do not add prefixes or suffixes to the Cypher query.
        The Cypher query must be self-contained and executable directly in Neo4j.
        Example of a correct query: "OPTIONAL MATCH (n)-[r]-() RETURN n, r"
        """

    system_prompt = prefix + "\n" + data_schema + "\n" + suffix
    return system_prompt


def ask_neo4j_gemini(question, data_schema):
    """
    Takes a natural language query and translate it into a Cypher query
    """

    try:

        system_prompt = get_system_prompt(data_schema=data_schema)
        prompt = f"""
        Question in natural language: {question}

        Generates a robust Cypher query for the Neo4j database.
        """

        response = client.models.generate_content(
            model=os.environ.get("MODEL"),
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=2048,
                top_k=2,
                top_p=0.5,
                temperature=0.2,
                seed=42,
            ),
        )
        cypher_query = response.text.removeprefix("```cypher").removesuffix("```")

        while "```" in cypher_query:
            cypher_query = cypher_query.replace("```", "")

        print("Query Cypher:", cypher_query)

        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            with driver.session() as session:
                results = session.run(cypher_query)
                formatted_response = []
                for record in results:
                    formatted_response.append(record.data())

                return {"data": formatted_response, "cypher_query": cypher_query}

    except Exception as e:
        return f"An error occurred: {e}"
