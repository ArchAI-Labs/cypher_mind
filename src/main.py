import os
import json
from dotenv import load_dotenv

load_dotenv()

from backend.import_data import GraphImport

if __name__ == "__main__":
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USER")
    password = os.environ.get("NEO4J_PASSWORD")

    importer = GraphImport(uri, user, password)

    node_url = os.environ.get("NODE_URL")
    relationship_url = os.environ.get("RELATIONSHIP_URL")
    reset = os.environ.get("RESET").lower() in ("yes", "true")
    if reset:
        importer.reset_database()

    try:
        if node_url and relationship_url:
            with open(node_url, "r") as f:
                node_files = json.load(f)

            with open(relationship_url, "r") as f:
                relationship_files = json.load(f)
            importer.import_all(node_files, relationship_files)
        else:
            importer.open()
    finally:
        importer.close()
