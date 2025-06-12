import os
from neo4j import GraphDatabase
from litellm import completion
import json
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)


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

    Generates a coalesce function to label nodes in the Cypher:
    """

    response = completion(
        model=os.environ.get("MODEL"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_completion_tokens=150
    )

    return response.choices[0].message.content.strip()

class GDSManager:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()
    
    def get_schema(schema: dict):
        coalesce_expr = generate_coalesce_expression(schema)
        return coalesce_expr

    def create_projection(self, graph_name: str, node_labels: list, relationship_types: list) -> str:
        """Create a GDS Projection"""
        try:
            with self.driver.session() as session:
                query = """
                    CALL gds.graph.project(
                        $graph_name,
                        $node_labels,
                        $relationship_types
                    )
                    YIELD graphName, nodeCount, relationshipCount
                """
                result = session.run(query, graph_name=graph_name, node_labels=node_labels, relationship_types=relationship_types)
                record = result.single()
                return f"✅ Proiezione '{record['graphName']}' creata con {record['nodeCount']} nodi e {record['relationshipCount']} relazioni."
        except Exception as e:
            logger.error(f"Errore durante la proiezione: {e}")
            return f"❌ Errore: {e}"

    def delete_all_projections(self) -> str:
        """Delete All GDS Projection."""
        try:
            with self.driver.session() as session:
                query = "CALL gds.graph.list() YIELD graphName"
                result = session.run(query)
                count = 0
                for record in result:
                    session.run("CALL gds.graph.drop($name, false)", name=record["graphName"])
                    count += 1
                return f"✅ {count} proiezioni eliminate."
        except Exception as e:
            logger.error(f"Errore durante eliminazione proiezioni: {e}")
            return f"❌ Errore: {e}"

    def list_algorithms(self) -> list:
        """Restituisce gli algoritmi disponibili nel GDS."""
        try:
            with self.driver.session() as session:
                result = session.run("CALL gds.list() YIELD name, description RETURN name, description LIMIT 100")
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Errore nel recupero degli algoritmi GDS: {e}")
            return [{"error": str(e)}]

    def run_pagerank(self, graph_name: str) -> list:
        """Esegue PageRank sulla proiezione."""
        return self._run_algorithm(
            graph_name,
            """
            CALL gds.pageRank.stream($graph_name)
            YIELD nodeId, score
            RETURN gds.util.asNode(nodeId).name AS name, score
            ORDER BY score DESC
            LIMIT 10
            """,
            "PageRank"
        )

    def run_betweenness(self, graph_name: str) -> list:
        """Esegue Betweenness Centrality."""
        return self._run_algorithm(
            graph_name,
            """
            CALL gds.betweenness.stream($graph_name)
            YIELD nodeId, score
            RETURN gds.util.asNode(nodeId).name AS name, score
            ORDER BY score DESC
            LIMIT 10
            """,
            "Betweenness"
        )

    def run_closeness(self, graph_name: str) -> list:
        """Esegue Closeness Centrality."""
        return self._run_algorithm(
            graph_name,
            """
            CALL gds.beta.closeness.stream($graph_name)
            YIELD nodeId, centrality
            RETURN gds.util.asNode(nodeId).name AS name, centrality
            ORDER BY centrality DESC
            LIMIT 10
            """,
            "Closeness"
        )

    def run_louvain(self, graph_name: str, coalesce_expr:str) -> list:
        """Esegue Louvain Community Detection."""
        query = f"""
            CALL gds.louvain.stream($graph_name)
            YIELD nodeId, communityId
            RETURN {coalesce_expr} AS label, communityId
            ORDER BY communityId
        """
        return self._run_algorithm(
            graph_name,
            query, 
            "Louvain"
        )

    def run_similarity(self, graph_name: str) -> list:
        """Esegue Node Similarity."""
        return self._run_algorithm(
            graph_name,
            """
            CALL gds.nodeSimilarity.stream($graph_name)
            YIELD node1, node2, similarity
            RETURN 
                gds.util.asNode(node1).name AS Node1, 
                gds.util.asNode(node2).name AS Node2, 
                similarity
            ORDER BY similarity DESC
            LIMIT 10
            """,
            "Node Similarity"
        )

    def _run_algorithm(self, graph_name: str, query: str, algo_name: str) -> list:
        """Metodo privato generico per eseguire algoritmi GDS."""
        try:
            with self.driver.session() as session:
                result = session.run(query, graph_name=graph_name)
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Errore in {algo_name}: {e}")
            return [{"error": str(e)}]
