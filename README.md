# ArchAI - CypherMind

<br>

![logo](https://github.com/ArchAI-Labs/cypher_mind/blob/main/img/logo_cyphermind.png)

<br>

## Overview

ArchAI - CypherMind is a system designed to bridge the gap between natural language and structured graph database queries. It enables users, regardless of their technical expertise, to interact with a Neo4j graph database using intuitive, natural language questions. The system translates these questions into Cypher queries, executes them against the database, and presents the results in a user-friendly format.

The core purpose of this project is to simplify data retrieval from graph databases, making it accessible to a broader audience, including business analysts, domain experts, and non-technical users. By leveraging the power of large language models (LLMs the system automates the complex process of Cypher query construction.

This repository contains the complete implementation of the ArchAI - CypherMind system, including the Streamlit application, Gemini integration, semantic cache, and data import functionalities. Each internal module plays a crucial role in supporting the project's overall goals, from handling user input to optimizing query performance and ensuring data integrity.

## Technology Stack

*   **Language:** Python
*   **Frameworks:** Streamlit
*   **Libraries:**
    *   neo4j-driver: Official Neo4j Python driver.
    *   google-generativeai (genai): Google Gemini API for natural language processing.
    *   Qdrant-client: Client for the Qdrant vector database.
    *   fastembed: For generating text embeddings.
    *   python-dotenv: For loading environment variables.
    *   pandas: For data manipulation and tabular display.
    *   json: For handling JSON data.
*   **Tools:**
    *   Neo4j: Graph database.
    *   Qdrant: Vector database.

## Directory Structure

```
├── src/
│   ├── app_streamlit.py - Main Streamlit application logic
│   ├── main.py - Data import script
│   ├── backend/
│   │   ├── gemini.py - Gemini API integration
│   │   ├── semantic_cache.py - Semantic cache implementation using Qdrant
│   │   ├── import_data.py - Graph data import functionality
│   │   ├── utils/
│   │   │   ├── streamlit_app_utils.py - Utility functions for the Streamlit app
│   ├── img/
│   │   ├── logo_cyphermind.png - Logo image for the application
│   ├── .env - Environment variables
│   ├── README.md - Project documentation
```

## Getting Started

1.  **Prerequisites:**
    *   Python 3.8 or higher
    *   Neo4j instance (local or cloud)
    *   Qdrant instance (local, cloud, or in-memory)
    *   Google Gemini API key
    *   Poetry for dependency management (recommended) or pip

2.  **Installation:**

    Clone the repository:

    ```
    git clone <repository_URL>
    ```

    Navigate to the project directory:

    ```
    cd ArchAI-CypherMind
    ```

    Install dependencies:

    ```
    pip install -r requirements_streamlit.txt
    ```

3.  **Configuration:**

    Create a `.env` file in the project root directory and set the following environment variables:

    ```
    NEO4J_URI=<Neo4j connection URI>
    NEO4J_USER=<Neo4j username>
    NEO4J_PASSWORD=<Neo4j password>
    GEMINI_API_KEY=<Google Gemini API key>
    QDRANT_COLLECTION=<Qdrant collection name>
    QDRANT_MODE=<Qdrant mode: memory, cloud, or docker>
    QDRANT_HOST=<Qdrant host (for cloud mode)>
    QDRANT_API_KEY=<Qdrant API key (for cloud mode)>
    QDRANT_URL=<Qdrant URL (for docker mode)>
    EMBEDDER=<Embedding model name (e.g., "sentence-transformers/all-MiniLM-L6-v2")>
    VECTOR_SIZE=<Embedding vector size (e.g., 384 for all-MiniLM-L6-v2)>
    NODE_URL=<Path to the JSON file containing node definitions>
    REL_URL=<Path to the JSON file containing relationship definitions>
    SAMPLE_QUESTIONS=<Path to the JSON file containing sample questions>
    RESET=<Set to "true" to reset the database on startup>
    NODE_CONTEXT_URL = <Path to the JSON file containing node type and properties used as context>
    REL_CONTEXT_URL = <Path to the JSON file containing relationships type and properties used as context>
    ```

4.  **Running the Application:**

    Build the docker container:

    ```
    docker compose build --no-cache
    ```

    Run containers (and import data if needed):

    ```
    docker compose up -d
    ```

    Run the Streamlit application:

    ```
    streamlit run src/app_streamlit.py
    ```

5.  **Module Usage:**

    *   **`app_streamlit.py` (Streamlit Application):** This is the main entry point for the application. It provides the user interface for querying the Neo4j database using natural language. To use it, simply run the Streamlit command as shown above. The application will connect to the Neo4j and Qdrant instances using the configured environment variables.

        Before to use the streamlit app you need to create the json files for the context:
        ```json
        {
            "Person": [
                "name", 
                "age"
                ]
        }
        ```

        ```json
        {
            "KNOWS": [
                "source_id", 
                "target_id"
                ]
        }
        ```

    *   **`backend/llm.py` (LLM Integration):** This module handles the interaction with the Google Gemini API (**the nly one tested at the moment**) using LiteLLM. It's used internally by `app_streamlit.py` to translate natural language questions into Cypher queries. You typically don't interact with this module directly.  Configuration is handled through the `GEMINI_API_KEY` environment variable.

    *   **`backend/semantic_cache.py` (Semantic Cache):** This module implements the semantic cache using Qdrant. It's used internally by `app_streamlit.py` to store and retrieve previously asked questions and their corresponding Cypher queries. Configuration is managed via environment variables such as `QDRANT_COLLECTION`, `QDRANT_MODE`, `QDRANT_HOST`, `QDRANT_API_KEY`, `QDRANT_URL`, `EMBEDDER`, and `VECTOR_SIZE`.

    *   **`backend/import_data.py` (Graph Data Import):** This module provides functionality to import data into the Neo4j graph database from CSV files. It can be used as a standalone script or integrated into other applications.

        Example usage:

        ```python
        from backend.import_data import GraphImport
        import os

        uri = os.environ.get("NEO4J_URI")
        user = os.environ.get("NEO4J_USER")
        password = os.environ.get("NEO4J_PASSWORD")

        importer = GraphImport(uri, user, password)

        node_files = {
            "path/to/nodes.csv": ("Person", ["name", "age"], "id")
        }
        relationship_files = {
            "path/to/relationships.csv": ("KNOWS", "Person", "Person", ["source_id", "target_id"], None)
        }

        importer.import_all(node_files, relationship_files)
        importer.close()
        ```

    *   **`backend/utils/streamlit_app_utils.py` (Streamlit Utilities):** This module provides utility functions for the Streamlit application, such as formatting results as a table and generating sample questions. It's used internally by `app_streamlit.py` and doesn't require direct interaction.

    *   **`main.py` (Data Import Script):** This script imports data into the Neo4j database. It's typically run as a one-time setup or as part of a data pipeline. Configuration is done through environment variables.

## Functional Analysis

### 1. Main Responsibilities of the System

The primary responsibility of the system is to translate natural language questions into executable Cypher queries for a Neo4j graph database. It manages the entire process, from receiving user input to displaying the query results in a structured format. The system also provides a semantic caching mechanism to optimize query performance and reduce reliance on the LLM for frequently asked questions. Furthermore, it offers data import functionalities to populate the Neo4j database.

### 2. Problems the System Solves

The system addresses the following key problems:

*   **Complexity of Cypher:** Cypher, while powerful, can be challenging for non-technical users to learn and use effectively. This system removes the need for users to write Cypher queries directly.
*   **Data Accessibility:** By providing a natural language interface, the system makes data stored in Neo4j accessible to a wider audience, including business analysts and domain experts.
*   **Query Performance:** The semantic cache improves query performance by storing and retrieving previously asked questions and their corresponding Cypher queries, reducing the need to invoke the LLM for repeated queries.
*   **Data Import:** The system simplifies the process of importing data into Neo4j from CSV files, enabling users to quickly populate the database with relevant information.

### 3. Interaction of Modules and Components

The modules interact as follows:

1.  The `app_streamlit.py` module receives user input through the Streamlit UI.
2.  It first checks the `semantic_cache.py` module for a similar question.
3.  If no similar question is found, it calls the `llm.py` module to generate a Cypher query.
4.  The `llm.py` module interacts with the Google Gemini API to translate the natural language question into a Cypher query. It also retrieves the database schema to provide context to the LLM.
5.  The generated Cypher query is executed against the Neo4j database using the `neo4j-driver`.
6.  The results are formatted by `app_streamlit.py` using utility functions from `streamlit_app_utils.py`.
7.  The question, Cypher query, and results are stored in the `semantic_cache.py` module for future use.
8.  Finally, the formatted results are displayed in the Streamlit UI.

The `main.py` script uses the `import_data.py` module to import data into the Neo4j database.


<br>

![logo](https://github.com/ArchAI-Labs/cypher_mind/blob/main/img/component_diagram.png)

<br>


### 4. User-Facing vs. System-Facing Functionalities

*   **User-Facing:**
    *   Streamlit UI (`app_streamlit.py`): Provides the interface for users to enter natural language questions, view results, and interact with the system.
    *   Sample questions: Predefined questions loaded from a JSON file that users can select.
    *   Similarity threshold: A slider that allows users to adjust the sensitivity of the semantic cache.
    *   Stop button: Allows users to interrupt long-running queries.

*   **System-Facing:**
    *   Gemini API integration (`llm.py`): Handles the communication with the Google Gemini API for query translation.
    *   Semantic cache (`semantic_cache.py`): Stores and retrieves previously asked questions and their corresponding Cypher queries.
    *   Neo4j interaction (`llm.py`): Executes Cypher queries against the Neo4j database.
    *   Data import (`import_data.py`): Imports data into the Neo4j database from CSV files.
    *   Utility functions (`streamlit_app_utils.py`): Provides helper functions for formatting results and generating sample questions.

## Architectural Patterns and Design Principles Applied

*   **Layered Architecture:** The project follows a layered architecture, separating the user interface, business logic, and data access layers.
*   **Model-View-Controller (MVC) -ish:** The Streamlit application acts as a controller and view, while the business logic components serve as the model.
*   **Semantic Caching:** Improves performance and reduces LLM usage by caching frequently asked questions and their corresponding Cypher queries.
*   **Configuration via Environment Variables:** Makes the application more flexible and secure by loading configuration parameters from environment variables.
*   **Separation of Concerns:** Each module has a specific responsibility, promoting modularity and maintainability.
*   **DRY (Don't Repeat Yourself):** Utility functions are placed in a separate module to avoid code duplication.
*   **Abstraction:** The `SemanticCache` and `GraphImport` classes abstract the interaction with external systems (Qdrant and Neo4j), providing a simplified interface for other modules.

## Weaknesses and Areas for Improvement

Based on the project analysis, here are some concrete TODO items for future releases and roadmap planning:

* **Implement comprehensive error handling:** Improve error handling throughout the application, providing more informative error messages to the user.
* **Add unit tests:** Increase test coverage for all modules, especially the `gemini.py` and `semantic_cache.py` modules, to ensure code reliability and prevent regressions.
* **Refactor complex functions:** Identify and refactor complex functions to improve code readability and maintainability.
* **Implement input validation:** Add input validation to the Streamlit application to prevent invalid or malicious input from reaching the backend.
* **Improve documentation:** Add more detailed documentation for all modules and functions, including usage examples and API references.
* **Implement a more robust schema management:** Explore options for automatically extracting and updating the database schema, rather than relying on manual JSON configuration.
* **Add support for different LLMs:** Make the system more flexible by allowing users to choose from different LLMs, not just Google Gemini.
* **Implement user authentication:** Add user authentication to restrict access to the application and protect sensitive data.
* **Improve the UI/UX:** Enhance the user interface and user experience of the Streamlit application based on user feedback.
* **Implement monitoring and logging:** Add monitoring and logging capabilities to track application performance and identify potential issues.
* **Investigate performance bottlenecks:** Identify and address any performance bottlenecks in the system, especially in the `semantic_cache.py` module.
* **Add CI/CD pipeline:** Implement a CI/CD pipeline to automate the build, test, and deployment process.

## Further Areas of Investigation

The following areas warrant further investigation:

*   **Scalability of the semantic cache:** Explore different strategies for scaling the semantic cache to handle a large number of questions and queries.
*   **Integration with other graph databases:** Investigate the possibility of integrating the system with other graph databases besides Neo4j.
*   **Advanced LLM techniques:** Explore the use of more advanced LLM techniques, such as fine-tuning and prompt engineering, to improve the accuracy and efficiency of query translation.
*   **Automated schema discovery:** Research methods for automatically discovering and updating the database schema, reducing the need for manual configuration.
*   **Security vulnerabilities:** Conduct a thorough security audit of the application to identify and address any potential security vulnerabilities.

## Attribution

Generated with the support of [ArchAI](https://github.com/ArchAI-Labs/code_explainer), an automated documentation system.
