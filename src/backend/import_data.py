import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
import json

load_dotenv()


class GraphImport:
    def __init__(self, uri, user, password):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        self.driver.close()

    def open(self):
        """
        Opens and returns a new Neo4j session using the driver.
        """
        return self.driver.session()

    def reset_database(self):
        """
        Clean all nodes and relationships
        """
        with self.driver.session() as session:

            def _reset(tx):
                query = "MATCH (n) DETACH DELETE n"
                tx.run(query)

            session.execute_write(_reset)
            print("Database Cleaned.")

    def import_nodes(self, file_path, node_label, node_metadata=None, id_key=None):
        """
        Imports nodes from csv separated by ;
        """
        with self.driver.session() as session:

            def _import_nodes(tx, file_path, node_label, node_metadata, id_key):
                property_string = "{"
                if node_metadata:
                    for prop in node_metadata:
                        property_string += f"{prop}: row['{prop}'], "
                if property_string:
                    property_string = property_string[:-2] + "}"
                query = f"""
                 LOAD CSV WITH HEADERS FROM 'file:///{file_path}' AS row FIELDTERMINATOR ';'
                 CREATE (n:{node_label} {property_string}
                  )
                 """
                print(f"NODE QUERY: {query}")
                tx.run(
                    query,
                    file_path=file_path,
                    node_label=node_label,
                    node_metadata=node_metadata,
                    id_key=id_key,
                )

            session.execute_write(
                _import_nodes, file_path, node_label, node_metadata, id_key
            )
            print(f"Noded {node_label} imported from {file_path}")

    def import_relationships(
        self,
        file_path,
        relationship_type,
        start_label,
        end_label,
        id_keys,
        relationship_metadata=None,
    ):
        """
        Imports relations from csv separated by ;
        """
        with self.driver.session() as session:

            def _import_relationships(
                tx,
                file_path,
                relationship_type,
                start_label,
                end_label,
                id_keys,
                relationship_metadata,
            ):
                property_string = "{"
                if relationship_metadata:
                    for prop in relationship_metadata:
                        property_string += f"{prop}: row['{prop}'], "
                    if property_string:
                        property_string = property_string[:-2] + "}"
                    query = f"""
                    LOAD CSV WITH HEADERS FROM 'file:///{file_path}' AS row FIELDTERMINATOR ';'
                    MATCH (a:{start_label} {{{id_keys[0]}: row['{id_keys[0]}']}})
                    MATCH (b:{end_label} {{{id_keys[1]}: row['{id_keys[1]}']}})
                    CREATE (a)-[r:{relationship_type}{property_string}]->(b)
                    """
                    print(f"REL QUERY: {query}")
                else:
                    query = f"""
                    LOAD CSV WITH HEADERS FROM 'file:///{file_path}' AS row FIELDTERMINATOR ';'
                    MATCH (a:{start_label} {{{id_keys[0]}: row['{id_keys[0]}']}})
                    MATCH (b:{end_label} {{{id_keys[1]}: row['{id_keys[1]}']}})
                    CREATE (a)-[r:{relationship_type}]->(b)
                    """
                    print(f"REL QUERY: {query}")
                tx.run(
                    query,
                    file_path=file_path,
                    relationship_type=relationship_type,
                    start_label=start_label,
                    end_label=end_label,
                    id_keys=id_keys,
                    relationship_metadata=relationship_metadata,
                )

            session.execute_write(
                _import_relationships,
                file_path,
                relationship_type,
                start_label,
                end_label,
                id_keys,
                relationship_metadata,
            )
            print(f"Relations {relationship_type} imported from {file_path}")

    def import_all(self, node_files, relationship_files):
        for node_file, (node_label, node_metadata, id_key) in node_files.items():
            self.import_nodes(node_file, node_label, node_metadata, id_key)
        for relationship_file, (
            relationship_type,
            start_label,
            end_label,
            id_keys,
            relationship_metadata,
        ) in relationship_files.items():
            self.import_relationships(
                relationship_file,
                relationship_type,
                start_label,
                end_label,
                id_keys,
                relationship_metadata,
            )
        print("Import DONE.")
