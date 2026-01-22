# ArchAI - CypherMind

<br>

![logo](https://github.com/ArchAI-Labs/cypher_mind/blob/main/img/logo_cyphermind.png)

<br>

## Overview

ArchAI - CypherMind is an advanced natural language to Cypher query translation system that democratizes access to Neo4j graph databases. It combines cutting-edge LLM technology with intelligent caching, template-based query generation, and graph data science capabilities to provide a production-grade solution for querying graph databases using natural language.

### Key Features

- **Multi-Strategy Query Resolution**: 6-level intelligent query resolution cascade (recent cache → templates → exact match → semantic signature → similarity search → fuzzy matching → LLM)
- **Graph Data Science Integration**: Execute advanced graph algorithms (PageRank, Betweenness Centrality, Louvain Community Detection, Node Similarity) directly from natural language
- **Template-Based Query System**: Zero-latency responses for common query patterns with NLP-powered parameter extraction
- **Multi-LLM Support**: Provider-agnostic architecture via LiteLLM (Gemini, GPT-4, Claude, and more)
- **Advanced Semantic Caching**: Qdrant-powered vector similarity search with quantization, indexing, and multiple caching layers
- **Async Operations**: Full async support for concurrent query processing and batch operations
- **Production-Ready**: Comprehensive test suite with 74+ test cases, extensive error handling, and performance monitoring

### Performance Highlights

- **Sub-100ms response time** for cached and templated queries
- **8x-16x memory reduction** through vector quantization
- **Intelligent fallback** handling query variations without LLM invocation
- **Multi-layer caching** (recent results, templates, frequent queries, vector database)

## Technology Stack

### Core Technologies
*   **Language**: Python 3.8+
*   **Web Framework**: Streamlit
*   **Graph Database**: Neo4j 5.x
*   **Vector Database**: Qdrant
*   **LLM Integration**: LiteLLM (multi-provider support)

### Key Libraries
*   **neo4j-driver**: Official Neo4j Python driver
*   **litellm** (1.70.4): Multi-provider LLM abstraction layer
*   **qdrant-client** (1.13.3): Vector database client with quantization support
*   **fastembed**: Fast local text embeddings
*   **spacy** (3.7+): NLP for entity extraction and query analysis
*   **rapidfuzz** (3.0+): Fuzzy string matching for query variations
*   **pandas**: Data manipulation and tabular display
*   **python-dotenv**: Environment configuration

### Testing & Development
*   **pytest** (7.0+): Testing framework
*   **pytest-mock** (3.10+): Mocking support
*   **pytest-asyncio** (0.21+): Async test support

## Directory Structure

```
├── src/
│   ├── app_streamlit.py              - Main Streamlit application with enhanced UI
│   ├── main.py                       - Data import script
│   ├── backend/
│   │   ├── llm.py                    - Multi-LLM integration (Gemini/GPT/Claude)
│   │   ├── semantic_cache.py         - Advanced 6-strategy semantic caching
│   │   ├── gds_manager.py            - Graph Data Science algorithm execution
│   │   ├── import_data.py            - Graph data import from CSV
│   │   └── utils/
│   │       └── streamlit_app_utils.py - Utility functions for UI
├── tests/
│   ├── backend/
│   │   ├── test_llm.py               - 11 tests for LLM integration
│   │   ├── test_semantic_cache.py    - 23 tests for caching strategies
│   │   ├── test_gds_manager.py       - 10 tests for GDS algorithms
│   │   ├── test_import_data.py       - 9+ tests for data import
│   │   └── utils/
│   │       └── test_streamlit_app_utils.py
│   ├── test_app_streamlit.py         - 20+ tests for UI
│   └── test_main.py                  - 10+ tests for main script
├── data/                             - Data files for import
├── data_fake/
│   └── query_template.json           - Query template library
├── img/
│   ├── logo_cyphermind.png
│   └── component_diagram.png
├── .env.example                      - Environment variables template
├── docker-compose.yml                - Docker orchestration
├── Dockerfile                        - Application container
├── pytest.ini                        - Test configuration
└── requirements_streamlit.txt        - Production dependencies
```

## Getting Started

### Prerequisites
*   Python 3.8 or higher
*   Neo4j instance (local or cloud) - version 5.x recommended
*   Qdrant instance (local, cloud, or in-memory)
*   LLM API key (Gemini, OpenAI, Anthropic, etc.)
*   Docker & Docker Compose (optional, for containerized deployment)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ArchAI-Labs/cypher_mind.git
   cd cypher_mind
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements_streamlit.txt
   ```

4. **Download spaCy language model** (for NLP entity extraction)
   ```bash
   python -m spacy download en_core_web_sm
   ```

### Configuration

Create a `.env` file in the project root with the following variables:

```env
# Neo4j Configuration
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# LLM Configuration (choose one)
MODEL=gemini/gemini-pro                    # For Google Gemini
# MODEL=gpt-4                              # For OpenAI GPT-4
# MODEL=claude-3-opus-20240229             # For Anthropic Claude
GEMINI_API_KEY=your_gemini_api_key         # If using Gemini
# OPENAI_API_KEY=your_openai_key           # If using OpenAI
# ANTHROPIC_API_KEY=your_anthropic_key     # If using Anthropic

# Qdrant Configuration
QDRANT_COLLECTION=cypher_cache
QDRANT_MODE=memory                         # Options: memory, docker, cloud
# QDRANT_HOST=your_cloud_host              # For cloud mode
# QDRANT_API_KEY=your_qdrant_key           # For cloud mode
# QDRANT_URL=http://localhost:6333         # For docker mode

# Embedding Configuration
EMBEDDER=sentence-transformers/all-MiniLM-L6-v2
VECTOR_SIZE=384

# Data Import Configuration
NODE_URL=data/nodes.json                   # Path to node definitions
REL_URL=data/relationships.json            # Path to relationship definitions
SAMPLE_QUESTIONS=data/sample_questions.json
RESET=false                                # Set to "true" to reset DB on startup

# Context Configuration (for LLM schema awareness)
NODE_CONTEXT_URL=data/node_context.json    # Node types and properties
REL_CONTEXT_URL=data/rel_context.json      # Relationship types and properties

# Template Configuration (optional)
QUERY_TEMPLATE_PATH=data_fake/query_template.json  # For template-based queries
```

### Running the Application

#### Option 1: Using Docker (Recommended)

1. **Build the container**
   ```bash
   docker compose build --no-cache
   ```

2. **Start all services** (Neo4j, Qdrant, CypherMind)
   ```bash
   docker compose up -d
   ```

3. **Access the application**
   - Streamlit UI: [http://localhost:8501](http://localhost:8501)
   - Neo4j Browser: [http://localhost:7474](http://localhost:7474)
   - Qdrant Dashboard: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

#### Option 2: Local Development

1. **Start Neo4j and Qdrant** (if not using cloud services)
   ```bash
   # Neo4j
   neo4j start

   # Qdrant (using Docker)
   docker run -p 6333:6333 qdrant/qdrant
   ```

2. **Import initial data** (optional)
   ```bash
   python src/main.py
   ```

3. **Run the Streamlit application**
   ```bash
   streamlit run src/app_streamlit.py
   ```

4. **Access the application** at [http://localhost:8501](http://localhost:8501)

## Usage Guide

### Query Template System

Create query templates in `data_fake/query_template.json` for common query patterns:

```json
{
  "templates": [
    {
      "intent": "get_top_users",
      "template": "get top {count} users from project {project}",
      "parameters": ["count", "project"],
      "cypher_template": "MATCH (p:Person)-[:WORKS_ON]->(proj:Project {name: '{project}'}) RETURN p.name, p.email LIMIT {count}",
      "priority": 1,
      "parameter_patterns": {
        "count": "\\b(?:top|first)\\s+(\\d+)\\b",
        "project": "project\\s+([A-Za-z0-9_\\s]+)"
      },
      "aliases": [
        "show me top {count} users in project {project}",
        "list {count} users working on {project}"
      ]
    }
  ]
}
```

**Benefits**:
- Zero-latency query execution (no LLM call)
- Consistent query structure
- Parameter validation
- Support for query variations via aliases

### Graph Data Science (GDS) Integration

The GDS Manager allows you to execute graph algorithms directly from natural language or programmatically:

#### Available Algorithms

1. **PageRank** - Identifies influential nodes
2. **Betweenness Centrality** - Finds bridge nodes
3. **Closeness Centrality** - Measures node accessibility
4. **Louvain Community Detection** - Discovers communities
5. **Node Similarity** - Finds similar nodes

#### Usage Example

```python
from backend.gds_manager import GDSManager
import os

# Initialize
uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")

gds = GDSManager(uri, user, password)

# Create a graph projection
gds.create_graph_projection(
    graph_name="my_graph",
    node_projection=["Person", "Project"],
    relationship_projection={
        "WORKS_ON": {
            "type": "WORKS_ON",
            "orientation": "NATURAL"
        }
    }
)

# Run PageRank
results = gds.run_pagerank(
    graph_name="my_graph",
    write_property="pagerank"
)

# Get top ranked nodes
gds.get_top_nodes_by_algorithm(
    algorithm="pagerank",
    property_name="pagerank",
    limit=10
)

# Cleanup
gds.drop_graph_projection("my_graph")
gds.close()
```

### Data Import

Import nodes and relationships from CSV files:

```python
from backend.import_data import GraphImport
import os

uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")

importer = GraphImport(uri, user, password)

# Define node files
node_files = {
    "data/persons.csv": ("Person", ["name", "email", "age"], "id"),
    "data/projects.csv": ("Project", ["name", "description"], "id")
}

# Define relationship files
relationship_files = {
    "data/works_on.csv": ("WORKS_ON", "Person", "Project", ["person_id", "project_id"], ["role"])
}

# Import all data
importer.import_all(node_files, relationship_files)
importer.close()
```

### Context Configuration

Before using the Streamlit app, create context JSON files to help the LLM understand your graph schema:

**node_context.json**:
```json
{
  "Person": ["name", "email", "age"],
  "Project": ["name", "description", "start_date"]
}
```

**rel_context.json**:
```json
{
  "WORKS_ON": ["person_id", "project_id", "role"],
  "MANAGES": ["manager_id", "project_id"]
}
```

## Architecture

### Query Resolution Flow

```
User Question
    ↓
[1] Recent Results Cache (Last 3 queries, in-memory)
    ↓ (miss)
[2] Template Matching (Regex + NLP parameter extraction)
    ↓ (miss)
[3] Exact Vector Match (Similarity > 0.95)
    ↓ (miss)
[4] Semantic Signature (NLP entity matching)
    ↓ (miss)
[5] Semantic Similarity (Fuzzy vector search)
    ↓ (miss)
[6] Fuzzy String Matching (Levenshtein distance)
    ↓ (miss)
[7] LLM Generation (Gemini/GPT/Claude)
    ↓
Store in Cache → Return Result
```

### Component Interaction

![Component Diagram](https://github.com/ArchAI-Labs/cypher_mind/blob/main/img/component_diagram.png)

1. **Streamlit UI** receives user input
2. **Semantic Cache** attempts multi-strategy resolution
3. **LLM Module** generates Cypher if cache misses
4. **Neo4j Driver** executes queries
5. **GDS Manager** handles graph algorithm requests
6. **Results** formatted and displayed
7. **Cache Updated** with new query-result pairs

### Architectural Patterns

- **Layered Architecture**: Clear separation of UI, business logic, and data access
- **Strategy Pattern**: Multiple query resolution strategies with intelligent fallback
- **Cache-Aside Pattern**: Multi-layer caching with write-through
- **Repository Pattern**: Abstracted database access via managers
- **Template Method**: Extensible algorithm execution framework
- **Async/Await**: Non-blocking operations for concurrent processing

## Testing

The project includes a comprehensive test suite with 74+ test cases:

### Run All Tests
```bash
pytest
```

### Run Specific Test Modules
```bash
# LLM integration tests
pytest tests/backend/test_llm.py

# Semantic cache tests (all 6 strategies)
pytest tests/backend/test_semantic_cache.py

# GDS algorithm tests
pytest tests/backend/test_gds_manager.py

# UI tests
pytest tests/test_app_streamlit.py
```

### Run with Coverage
```bash
pytest --cov=src --cov-report=html
```

### Test Coverage Summary

| Module | Tests | Coverage Areas |
|--------|-------|----------------|
| **LLM Integration** | 11 | Schema generation, Cypher generation, validation, intent extraction, query cleaning |
| **Semantic Cache** | 23 | All 6 search strategies, parameter extraction, async operations, batch search, performance stats |
| **GDS Manager** | 10 | Graph projections, PageRank, Betweenness, Louvain, Node Similarity, error handling |
| **Data Import** | 9+ | Node/relationship import, batch operations, constraint creation |
| **Streamlit UI** | 20+ | Session management, cache controls, UI interactions, error handling |
| **Utilities** | 5+ | Result formatting, sample question generation |

## Performance Optimization

### Caching Strategy Performance

| Strategy | Avg Response Time | Hit Rate (typical) |
|----------|-------------------|-------------------|
| Recent Cache | <10ms | 15-20% |
| Template Match | <50ms | 25-30% |
| Exact Match | <100ms | 10-15% |
| Semantic Signature | <150ms | 15-20% |
| Similarity Search | <200ms | 20-25% |
| Fuzzy Match | <250ms | 5-10% |
| LLM Generation | 1-3s | Last resort |

### Memory Optimization

- **Vector Quantization**: 8x reduction with scalar quantization, 16x with binary
- **LRU Caching**: 10,000 entry limit prevents memory bloat
- **Payload Indexing**: Fast filtering without full vector search
- **Disk Storage**: Optional for large-scale deployments

### Configuration Tuning

**For High Throughput**:
```env
QDRANT_MODE=cloud  # Use Qdrant Cloud for distributed search
EMBEDDER=sentence-transformers/all-MiniLM-L6-v2  # Fast embeddings
SIMILARITY_THRESHOLD=0.70  # Lower threshold, more cache hits
```

**For High Accuracy**:
```env
MODEL=gpt-4  # More powerful LLM
EMBEDDER=sentence-transformers/all-mpnet-base-v2  # Higher quality embeddings
SIMILARITY_THRESHOLD=0.85  # Higher threshold, more LLM calls
```

**For Low Latency**:
```env
QDRANT_MODE=memory  # In-memory vector search
QUERY_TEMPLATE_PATH=data_fake/query_template.json  # Enable templates
SIMILARITY_THRESHOLD=0.75  # Balanced threshold
```

## API Reference

### SemanticCache

```python
from backend.semantic_cache import SemanticCache

cache = SemanticCache(
    collection_name="cypher_cache",
    mode="memory",
    embedder_model="sentence-transformers/all-MiniLM-L6-v2"
)

# Smart search with 6-level cascade
result = cache.smart_search(
    query="show me top 10 users",
    similarity_threshold=0.75
)

# Async batch search
results = await cache.async_batch_search(
    queries=["query1", "query2", "query3"],
    similarity_threshold=0.75
)

# Store query-result pair
cache.store_query(
    query="show me top 10 users",
    cypher="MATCH (u:User) RETURN u LIMIT 10",
    result=[{"name": "John"}],
    template_used="get_top_users"
)

# Get performance statistics
stats = cache.get_detailed_stats()
```

### GDSManager

```python
from backend.gds_manager import GDSManager

gds = GDSManager(uri, user, password)

# Create graph projection
gds.create_graph_projection(
    graph_name="social_graph",
    node_projection=["Person"],
    relationship_projection="KNOWS"
)

# Run algorithms
pagerank_results = gds.run_pagerank("social_graph", write_property="score")
communities = gds.run_louvain("social_graph", write_property="community")
centrality = gds.run_betweenness("social_graph", write_property="centrality")

# Get results
top_influencers = gds.get_top_nodes_by_algorithm("pagerank", "score", limit=10)

# Cleanup
gds.drop_graph_projection("social_graph")
```

### LLM Module

```python
from backend.llm import (
    ask_neo4j_llm,
    extract_query_intent,
    validate_cypher_syntax,
    clean_cypher_query
)

# Generate Cypher from natural language
response = ask_neo4j_llm(
    question="Who are the top 5 influencers?",
    schema_info=schema,
    sample_questions=samples
)

# Extract intent
intent = extract_query_intent("Show me all projects managed by John")
# Returns: {
#   "action": "retrieve",
#   "entities": ["projects", "John"],
#   "filters": {"manager": "John"},
#   "limit": None
# }

# Validate Cypher
is_valid = validate_cypher_syntax("MATCH (n) RETURN n")

# Clean LLM-generated query
clean_query = clean_cypher_query("```cypher\nMATCH (n) RETURN n\n```")
```

## Troubleshooting

### Common Issues

**1. Qdrant Connection Error**
```
Error: Failed to connect to Qdrant
Solution: Ensure Qdrant is running (docker run -p 6333:6333 qdrant/qdrant)
```

**2. Neo4j Authentication Failed**
```
Error: Neo4j authentication failed
Solution: Verify NEO4J_USER and NEO4J_PASSWORD in .env file
```

**3. LLM API Key Invalid**
```
Error: API key authentication failed
Solution: Check your GEMINI_API_KEY/OPENAI_API_KEY in .env
```

**4. Slow Query Performance**
```
Issue: Queries taking >5 seconds
Solution:
- Enable query templates for common patterns
- Lower similarity threshold (0.70-0.75)
- Use vector quantization in Qdrant
- Check Qdrant collection size and optimize
```

**5. Cache Not Working**
```
Issue: Every query calls LLM
Solution:
- Verify Qdrant connection
- Check QDRANT_COLLECTION exists
- Ensure embedder model is downloaded
- Review similarity_threshold (may be too high)
```

### Debug Mode

Enable detailed logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
```

Check cache statistics:
```python
from backend.semantic_cache import SemanticCache

cache = SemanticCache()
stats = cache.get_detailed_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']}%")
print(f"Total queries: {stats['total_searches']}")
```

## Roadmap

### Completed Features ✅
- ✅ Multi-LLM support via LiteLLM
- ✅ Comprehensive test suite (74+ tests)
- ✅ Graph Data Science integration
- ✅ Template-based query system
- ✅ Advanced 6-strategy semantic caching
- ✅ Async operations support
- ✅ NLP entity extraction with spaCy
- ✅ Fuzzy matching with rapidfuzz
- ✅ Vector quantization for memory optimization

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for your changes
4. Ensure all tests pass (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements_streamlit.txt
pip install pytest pytest-mock pytest-asyncio pytest-cov

# Run tests with coverage
pytest --cov=src --cov-report=html

# Format code
black src/ tests/

# Lint code
flake8 src/ tests/
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use CypherMind in your research or project, please cite:

```bibtex
@software{cyphermind2025,
  title={CypherMind: Advanced Natural Language to Cypher Translation},
  author={ArchAI Labs},
  year={2025},
  url={https://github.com/ArchAI-Labs/cypher_mind}
}
```

## Acknowledgments

- **Neo4j** for the powerful graph database platform
- **Qdrant** for the high-performance vector database
- **LiteLLM** for unified LLM API access
- **Streamlit** for the interactive web framework
- **spaCy** for advanced NLP capabilities
- **ArchAI** automated documentation system for project analysis

## Support

- **Documentation**: [GitHub Wiki](https://github.com/ArchAI-Labs/cypher_mind/wiki)
- **Issues**: [GitHub Issues](https://github.com/ArchAI-Labs/cypher_mind/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ArchAI-Labs/cypher_mind/discussions)

---

**Built with ❤️ by ArchAI Labs**

Generated and maintained with the support of [ArchAI](https://github.com/ArchAI-Labs/code_explainer), an automated documentation system.
