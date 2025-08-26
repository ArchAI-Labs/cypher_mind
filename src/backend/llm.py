import os
from dotenv import load_dotenv
import json
import logging
from typing import Optional, Dict, Any, List

from neo4j import GraphDatabase

from litellm import completion

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

uri = os.environ.get("NEO4J_URI")
user = os.environ.get("NEO4J_USER")
password = os.environ.get("NEO4J_PASSWORD")

api_key = os.environ.get("GEMINI_API_KEY")


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
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        response = completion(
            model=os.environ.get("MODEL"),
            messages=messages,
            max_completion_tokens=2048,
            n=2,
            top_p=0.5,
            temperature=0.2,
            logprobs=False,
            timeout=10,
            max_retries=1,
            num_retries=1
            )
        
        
        return response.choices[0].message.content

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


def execute_cypher_query(cypher_query: str) -> List[Dict[str, Any]]:
    """
    Execute a Cypher query against Neo4j database and return results.
    
    Args:
        cypher_query: The Cypher query to execute
        
    Returns:
        List of dictionaries containing query results
        
    Raises:
        Exception: If query execution fails
    """
    try:
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            with driver.session() as session:
                results = session.run(cypher_query)
                formatted_response = []
                for record in results:
                    formatted_response.append(record.data())
                return formatted_response
    except Exception as e:
        logger.error(f"Error executing Cypher query: {e}")
        logger.error(f"Query was: {cypher_query}")
        raise


def ask_neo4j_llm(question: str, data_schema: str, override_cypher: Optional[str] = None) -> Dict[str, Any]:
    """
    Takes a natural language query and translates it into a Cypher query.
    
    Args:
        question: The natural language question
        data_schema: The Neo4j database schema description
        override_cypher: Optional pre-generated Cypher query to execute instead of generating new one
        
    Returns:
        Dictionary containing the query results and the Cypher query used
    """
    try:
        if override_cypher:
            # Use provided Cypher query (from cache/template)
            logger.info(f"Using provided Cypher query: {override_cypher}")
            cypher_query = override_cypher.strip()
        else:
            # Generate new Cypher query using LLM
            logger.info(f"Generating Cypher query for: {question}")
            cypher_query = generate_cypher_query(question, data_schema)

        # Clean the query
        cypher_query = clean_cypher_query(cypher_query)
        logger.info(f"Executing Cypher query: {cypher_query}")

        # Execute the query
        results = execute_cypher_query(cypher_query)
        
        return {
            "data": results, 
            "cypher_query": cypher_query,
            "question": question
        }

    except Exception as e:
        logger.error(f"Error in ask_neo4j_llm: {e}")
        return {
            "error": str(e),
            "data": [],
            "cypher_query": cypher_query if 'cypher_query' in locals() else "",
            "question": question
        }


def generate_cypher_query(question: str, data_schema: str) -> str:
    """
    Generate a Cypher query from natural language using LLM.
    
    Args:
        question: The natural language question
        data_schema: The Neo4j database schema description
        
    Returns:
        Generated Cypher query string
    """
    system_prompt = get_system_prompt(data_schema=data_schema)
    prompt = f"""
    Question in natural language: {question}

    Generate a robust Cypher query for the Neo4j database.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    
    response = completion(
        model=os.environ.get("MODEL"),
        messages=messages,
        max_completion_tokens=2048,
        n=2,
        top_p=0.5,
        temperature=0.2,
        logprobs=False,
        timeout=10,
        max_retries=1,
        num_retries=1
    )
    
    return response.choices[0].message.content


def clean_cypher_query(cypher_query: str) -> str:
    """
    Clean and normalize a Cypher query string.
    
    Args:
        cypher_query: Raw Cypher query string
        
    Returns:
        Cleaned Cypher query string
    """
    # Remove code block markers
    cleaned_query = cypher_query.removeprefix("```cypher").removesuffix("```")
    
    # Remove any remaining markdown markers
    while "```" in cleaned_query:
        cleaned_query = cleaned_query.replace("```", "")
    
    # Remove extra whitespace
    cleaned_query = cleaned_query.strip()
    
    return cleaned_query


def validate_cypher_syntax(cypher_query: str) -> bool:
    """
    Basic validation of Cypher query syntax.
    
    Args:
        cypher_query: The Cypher query to validate
        
    Returns:
        True if query appears to be valid, False otherwise
    """
    # Basic checks for common Cypher keywords
    cypher_keywords = ['MATCH', 'RETURN', 'WHERE', 'WITH', 'CREATE', 'MERGE', 'DELETE', 'SET']
    query_upper = cypher_query.upper()
    
    # Must have at least one Cypher keyword
    if not any(keyword in query_upper for keyword in cypher_keywords):
        return False
    
    # Basic bracket matching
    if query_upper.count('(') != query_upper.count(')'):
        return False
    if query_upper.count('[') != query_upper.count(']'):
        return False
    if query_upper.count('{') != query_upper.count('}'):
        return False
    
    return True


def extract_query_intent(question: str) -> Dict[str, Any]:
    """
    Extract the intent and key parameters from a natural language question.
    This can be used by the semantic cache for better template matching.
    
    Args:
        question: The natural language question
        
    Returns:
        Dictionary with extracted intent information
    """
    question_lower = question.lower().strip()
    
    intent_info = {
        "action": None,
        "entities": [],
        "filters": {},
        "limit": None,
        "aggregation": None
    }
    
    # Extract action intent
    if any(word in question_lower for word in ['get', 'show', 'find', 'list', 'return']):
        intent_info["action"] = "retrieve"
    elif any(word in question_lower for word in ['count', 'how many']):
        intent_info["action"] = "count"
        intent_info["aggregation"] = "count"
    elif any(word in question_lower for word in ['sum', 'total']):
        intent_info["action"] = "aggregate"
        intent_info["aggregation"] = "sum"
    
    # Extract limit/top keywords
    import re
    limit_match = re.search(r'\b(?:top|first|limit)\s+(\d+)', question_lower)
    if limit_match:
        intent_info["limit"] = int(limit_match.group(1))
    
    # Extract common entity types (extend based on your schema)
    common_entities = ['user', 'project', 'task', 'organization', 'team', 'document']
    for entity in common_entities:
        if entity in question_lower:
            intent_info["entities"].append(entity)
    
    return intent_info


def suggest_similar_queries(question: str, schema_info: Dict) -> List[str]:
    """
    Suggest similar queries based on the question and schema.
    This can help users explore the data.
    
    Args:
        question: The original question
        schema_info: Information about the database schema
        
    Returns:
        List of suggested similar questions
    """
    suggestions = []
    intent = extract_query_intent(question)
    
    if intent["action"] == "retrieve" and "user" in intent["entities"]:
        suggestions.extend([
            "Show all users in the system",
            "Get users with their projects",
            "Find active users",
            "List top 10 users by activity"
        ])
    
    if intent["action"] == "retrieve" and "project" in intent["entities"]:
        suggestions.extend([
            "Show all projects",
            "Get projects with team members",
            "Find recent projects",
            "List projects by status"
        ])
    
    # Remove duplicates and limit suggestions
    suggestions = list(set(suggestions))[:5]
    return suggestions


def generate_coalesce_expression(schema: dict) -> str:
    """
    It uses an LLM to generate a Cypher `coalesce(...)` expression based on the schema properties.
    """
    system_prompt = """
    You are an expert in Neo4j and Cypher.
    Your task is to create a `coalesce(...)` function that uses the best available properties 
    to identify nodes in the graph.

    Rules:
    - Give priority to `name`, `title`, `label`, `id_*`.
    - The output must be **only** a Cypher string such as: coalesce(...).
    - Use the format: gds.util.asNode(nodeId).<prop>.
    - If nothing is present, add 'Unknown' as a fallback.
    - Do not add explanation, only code.
    """

    prompt = f"""
    Here is the schema of the graph nodes in JSON format:

    {json.dumps(schema, indent=2)}

    Generate a coalesce function to label nodes in the Cypher:
    """

    response = completion(
        model=os.environ.get("MODEL"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_completion_tokens=150,
        logprobs=False,
        timeout=10,
        max_retries=1,
        num_retries=1
    )

    return response.choices[0].message.content.removeprefix("```cypher").removesuffix("```").strip()


# Backward compatibility - keep the original function name as alias
def ask_neo4j_llm_legacy(question, data_schema):
    """Legacy function for backward compatibility"""
    return ask_neo4j_llm(question, data_schema)