import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
import logging

from .llm import generate_coalesce_expression

load_dotenv()

logger = logging.getLogger(__name__)




class GDSManager:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.coalesce_expr = "gds.util.asNode(nodeId)"

    def close(self):
        self.driver.close()
    
    def get_schema(self, schema: dict):
        self.coalesce_expr = generate_coalesce_expression(schema)

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
        query = f"""
            CALL gds.pageRank.stream($graph_name)
            YIELD nodeId, score
            RETURN {self.coalesce_expr} AS label, score
            ORDER BY score DESC
            LIMIT 10
        """
        return self._run_algorithm(graph_name, query, "PageRank")

    def run_betweenness(self, graph_name: str) -> list:
        query = f"""
            CALL gds.betweenness.stream($graph_name)
            YIELD nodeId, score
            RETURN {self.coalesce_expr} AS label, score
            ORDER BY score DESC
            LIMIT 10
        """
        return self._run_algorithm(graph_name, query, "Betweenness")


    def run_closeness(self, graph_name: str) -> list:
        query = f"""
            CALL gds.beta.closeness.stream($graph_name)
            YIELD nodeId, score
            RETURN {self.coalesce_expr} AS label, score
            ORDER BY score DESC
            LIMIT 10
        """
        return self._run_algorithm(graph_name, query, "Closeness")


    def run_louvain(self, graph_name: str) -> list:
        query = f"""
            CALL gds.louvain.stream($graph_name)
            YIELD nodeId, communityId
            RETURN {self.coalesce_expr} AS label, communityId
            ORDER BY communityId
        """
        return self._run_algorithm(graph_name, query, "Louvain")


    def run_similarity(self, graph_name: str) -> list:
        query = f"""
            CALL gds.nodeSimilarity.stream($graph_name)
            YIELD node1, node2, similarity
            RETURN 
                {self.coalesce_expr.replace('nodeId', 'node1')} AS Node1,
                {self.coalesce_expr.replace('nodeId', 'node2')} AS Node2,
                similarity
            ORDER BY similarity DESC
            LIMIT 10
        """
        return self._run_algorithm(graph_name, query, "Node Similarity")


    def _run_algorithm(self, graph_name: str, query: str, algo_name: str) -> list:
        """Metodo privato generico per eseguire algoritmi GDS."""
        try:
            with self.driver.session() as session:
                result = session.run(query, graph_name=graph_name)
                print(result)
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Errore in {algo_name}: {e}")
            return [{"error": str(e)}]
