import os
import sys
import json
import logging
import pathlib
import functools
from contextlib import asynccontextmanager

# Anchor paths to the project root (parent of src/) regardless of CWD
_SRC_DIR = pathlib.Path(__file__).parent.resolve()
_PROJECT_ROOT = _SRC_DIR.parent

# Always execute with project root as CWD so relative env-var paths work
os.chdir(str(_PROJECT_ROOT))

# Ensure src/ is on sys.path so `from backend.* import` resolves
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from dotenv import load_dotenv
load_dotenv(str(_PROJECT_ROOT / ".env"))

from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.concurrency import run_in_threadpool

from backend.llm import ask_neo4j_llm, get_schema
from backend.gds_manager import GDSManager
from backend.utils.streamlit_app_utils import format_results_as_table, generate_sample_questions
from medha import Medha, Settings, SearchStrategy
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

BASE_DIR = pathlib.Path(__file__).parent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app_state: dict = {}

# Exact hex colors mirrored from the Streamlit CSS badges
# (label, badge_bg, badge_text, banner_bg, border_color)
STRATEGY_DISPLAY = {
    # lowercase keys — SearchStrategy.value returns e.g. "exact_match"
    "llm":            ("LLM GENERATED", "#dc3545", "#ffffff", "#fdecea", "#dc3545"),
    "no_match":       ("LLM GENERATED", "#dc3545", "#ffffff", "#fdecea", "#dc3545"),
    "template_match": ("TEMPLATE",      "#28a745", "#ffffff", "#eaf6ec", "#28a745"),
    "exact_match":    ("EXACT MATCH",   "#17a2b8", "#ffffff", "#e3f6f9", "#17a2b8"),
    "semantic_match": ("SIMILAR",       "#ffc107", "#000000", "#fff8e1", "#ffc107"),
    "fuzzy_match":    ("FUZZY MATCH",   "#fd7e14", "#ffffff", "#fff3e0", "#fd7e14"),
    "l1_cache":       ("L1 CACHE",      "#20c997", "#ffffff", "#e0f5f1", "#20c997"),
}


def get_strategy_display(strategy) -> dict:
    raw = strategy.value if hasattr(strategy, "value") else str(strategy)
    key = raw.lower()  # normalise to lowercase regardless of enum definition
    label, badge_bg, badge_text, banner_bg, border = STRATEGY_DISPLAY.get(
        key, (raw, "#6c757d", "#ffffff", "#f8f9fa", "#6c757d")
    )
    return {
        "label": label,
        "badge_style": f"background-color:{badge_bg}; color:{badge_text};",
        "banner_style": f"background-color:{banner_bg}; border-left:4px solid {border};",
    }


def _build_cache_settings() -> tuple[FastEmbedAdapter, Settings]:
    embedder = FastEmbedAdapter(
        model_name=os.environ.get("EMBEDDER", "sentence-transformers/all-MiniLM-L6-v2")
    )
    settings = Settings(
        qdrant_mode=os.environ.get("QDRANT_MODE", "memory"),
        qdrant_host=os.environ.get("QDRANT_HOST", "localhost"),
        qdrant_url=os.environ.get("QDRANT_URL"),
        qdrant_api_key=os.environ.get("QDRANT_API_KEY"),
        template_file=os.environ.get("TEMPLATE_QUERY"),
        score_threshold_exact=float(os.environ.get("EXACT_THRESHOLD", "0.99")),
        score_threshold_semantic=float(os.environ.get("SIMILARITY_THRESHOLD", "0.90")),
        score_threshold_template=float(os.environ.get("TEMPLATE_THRESHOLD", "0.70")),
        score_threshold_fuzzy=float(os.environ.get("FUZZY_THRESHOLD", "85.0")),
    )
    return embedder, settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting CypherMind FastAPI app...")

    # --- Cache ---
    try:
        embedder, settings = _build_cache_settings()
        cache = Medha(
            collection_name=os.environ.get("QDRANT_COLLECTION", "semantic_cache"),
            embedder=embedder,
            settings=settings,
        )
        await cache.start()
        app_state["cache"] = cache
        logger.info("Cache initialized")
    except Exception as e:
        logger.error(f"Cache init failed: {e}")
        app_state["cache"] = None

    # --- GDS ---
    try:
        app_state["gds"] = GDSManager(
            uri=os.environ["NEO4J_URI"],
            user=os.environ["NEO4J_USER"],
            password=os.environ["NEO4J_PASSWORD"],
        )
        logger.info("GDS Manager initialized")
    except Exception as e:
        logger.error(f"GDS init failed: {e}")
        app_state["gds"] = None

    # --- Schema ---
    try:
        with open(os.environ.get("NODE_CONTEXT_URL"), "r") as f:
            nodes = json.load(f)
        with open(os.environ.get("REL_CONTEXT_URL"), "r") as f:
            relationships = json.load(f)
        if app_state.get("gds"):
            app_state["gds"].get_schema(nodes)
        app_state["graph_schema"] = get_schema(nodes=nodes, relations=relationships)
        logger.info("Schema loaded")
    except Exception as e:
        logger.error(f"Schema loading failed: {e}")
        app_state["graph_schema"] = ""

    yield

    if app_state.get("cache"):
        await app_state["cache"].close()
        logger.info("Cache closed")


app = FastAPI(title="CypherMind", lifespan=lifespan)
app.mount("/img", StaticFiles(directory=str(_PROJECT_ROOT / "img")), name="img")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    sample_questions = generate_sample_questions()
    cache = app_state.get("cache")
    stats = cache.stats if cache else {}
    return templates.TemplateResponse("index.html", {
        "request": request,
        "sample_questions": sample_questions,
        "stats": stats,
    })


# ---------------------------------------------------------------------------
# HTMX partials
# ---------------------------------------------------------------------------

@app.get("/stats", response_class=HTMLResponse)
async def get_stats(request: Request):
    cache = app_state.get("cache")
    stats = cache.stats if cache else {}
    return templates.TemplateResponse("partials/stats.html", {
        "request": request,
        "stats": stats,
    })


@app.post("/query", response_class=HTMLResponse)
async def run_query(
    request: Request,
    question: str = Form(...),
    template_threshold: float = Form(0.70),
    exact_threshold: float = Form(0.99),
    fuzzy_threshold: float = Form(85.0),
    similarity_threshold: float = Form(0.90),
):
    cache = app_state.get("cache")
    graph_schema = app_state.get("graph_schema", "")

    def _render(ctx: dict):
        response = templates.TemplateResponse("partials/query_result.html", {"request": request, **ctx})
        # Tell the stats container to refresh immediately after every query
        response.headers["HX-Trigger-After-Settle"] = "refreshStats"
        return response

    if not cache:
        return _render({"error": "Cache not initialized. Check server logs."})

    # Apply sidebar thresholds
    cache._settings.score_threshold_template = template_threshold
    cache._settings.score_threshold_exact = exact_threshold
    cache._settings.score_threshold_fuzzy = fuzzy_threshold
    cache._settings.score_threshold_semantic = similarity_threshold

    try:
        cache_hit = await run_in_threadpool(cache.search_sync, question)
        generated_query = cache_hit.generated_query
        strategy = cache_hit.strategy
        confidence = cache_hit.confidence

        # Case 1: Full response in cache
        if cache_hit.response_summary:
            try:
                data = json.loads(cache_hit.response_summary) if isinstance(cache_hit.response_summary, str) else cache_hit.response_summary
            except Exception:
                data = cache_hit.response_summary
            return _render({
                "cypher_query": generated_query,
                "table_data": format_results_as_table(data),
                "strategy_display": get_strategy_display(strategy),
                "confidence": confidence,
                "message": "Response found in cache!",
            })

        # Case 2: Cypher found — execute against DB
        if generated_query and confidence > 0.0:
            try:
                current_result = await run_in_threadpool(
                    functools.partial(ask_neo4j_llm, question=question, data_schema=graph_schema, override_cypher=generated_query)
                )
                data = (current_result or {}).get("data")
                if data:
                    summary = json.dumps(data)
                    await run_in_threadpool(
                        functools.partial(
                            cache.store_sync,
                            question=question,
                            generated_query=generated_query,
                            response_summary=summary,
                            template_id=cache_hit.template_used,
                        )
                    )
                    return _render({
                        "cypher_query": generated_query,
                        "table_data": format_results_as_table(data),
                        "strategy_display": get_strategy_display(strategy),
                        "confidence": confidence,
                        "message": "Query executed from cache!",
                    })
            except Exception as e:
                logger.error(f"Cache query execution failed: {e}")

        # Case 3: LLM fallback
        result = await run_in_threadpool(
            functools.partial(ask_neo4j_llm, question=question, data_schema=graph_schema)
        )
        data = (result or {}).get("data")
        if data:
            summary = json.dumps(data)
            await run_in_threadpool(
                functools.partial(
                    cache.store_sync,
                    question=question,
                    generated_query=result["cypher_query"],
                    response_summary=summary,
                )
            )
            return _render({
                "cypher_query": result["cypher_query"],
                "table_data": format_results_as_table(data),
                "strategy_display": get_strategy_display("llm"),
                "confidence": 1.0,
                "message": "Query generated and cached!",
            })

        return _render({"error": "No results returned. Check your question or database connection."})

    except Exception as e:
        logger.error(f"Query error: {e}")
        return _render({"error": str(e)})


@app.post("/cache/reset", response_class=HTMLResponse)
async def reset_cache():
    try:
        old_cache = app_state.get("cache")
        if old_cache:
            await old_cache.close()
        embedder, settings = _build_cache_settings()
        new_cache = Medha(
            collection_name=os.environ.get("QDRANT_COLLECTION", "semantic_cache"),
            embedder=embedder,
            settings=settings,
        )
        await new_cache.start()
        app_state["cache"] = new_cache
        return HTMLResponse('<p class="text-green-600 font-medium">Cache reset successfully!</p>')
    except Exception as e:
        return HTMLResponse(f'<p class="text-red-600 font-medium">Error: {e}</p>')


@app.post("/cache/clear-memory", response_class=HTMLResponse)
async def clear_memory():
    cache = app_state.get("cache")
    if cache:
        await run_in_threadpool(cache.clear_caches)
        return HTMLResponse('<p class="text-green-600 font-medium">Memory caches cleared!</p>')
    return HTMLResponse('<p class="text-red-600 font-medium">Cache not available.</p>')


@app.post("/cache/batch", response_class=HTMLResponse)
async def batch_insert(batch_file: UploadFile = File(...)):
    cache = app_state.get("cache")
    if not cache:
        return HTMLResponse('<p class="text-red-600 font-medium">Cache not available.</p>')
    try:
        content = await batch_file.read()
        batch_data = json.loads(content)
        for item in batch_data:
            if "response_summary" in item and not isinstance(item["response_summary"], str):
                item["response_summary"] = json.dumps(item["response_summary"])
        success = await cache.store_batch(batch_data)
        if success:
            return HTMLResponse(f'<p class="text-green-600 font-medium">Inserted {len(batch_data)} queries successfully!</p>')
        return HTMLResponse('<p class="text-red-600 font-medium">Batch insert failed.</p>')
    except Exception as e:
        return HTMLResponse(f'<p class="text-red-600 font-medium">Error: {e}</p>')


@app.post("/gds/projection", response_class=HTMLResponse)
async def create_projection(
    graph_name: str = Form(...),
    node_labels: str = Form(...),
    relationship_types: str = Form(...),
):
    gds = app_state.get("gds")
    if not gds:
        return HTMLResponse('<p class="text-red-600 font-medium">GDS not available.</p>')
    try:
        node_labels_list = [l.strip() for l in node_labels.split(",") if l.strip()]
        rel_types_list = [r.strip() for r in relationship_types.split(",") if r.strip()]
        msg = await run_in_threadpool(functools.partial(gds.create_projection, graph_name, node_labels_list, rel_types_list))
        return HTMLResponse(f'<p class="text-green-600 font-medium">{msg}</p>')
    except Exception as e:
        return HTMLResponse(f'<p class="text-red-600 font-medium">Error: {e}</p>')


@app.post("/gds/algorithm", response_class=HTMLResponse)
async def run_algorithm(
    request: Request,
    graph_name: str = Form(...),
    algorithm: str = Form(...),
):
    gds = app_state.get("gds")
    if not gds:
        return HTMLResponse('<p class="text-red-600 font-medium">GDS not available.</p>')
    try:
        algo_map = {
            "PageRank": gds.run_pagerank,
            "Betweenness Centrality": gds.run_betweenness,
            "Closeness Centrality": gds.run_closeness,
            "Louvain": gds.run_louvain,
            "Node Similarity": gds.run_similarity,
        }
        fn = algo_map.get(algorithm)
        if not fn:
            return HTMLResponse(f'<p class="text-red-600 font-medium">Unknown algorithm: {algorithm}</p>')
        results = await run_in_threadpool(functools.partial(fn, graph_name))
        return templates.TemplateResponse("partials/gds_result.html", {
            "request": request,
            "results": results,
            "algorithm": algorithm,
        })
    except Exception as e:
        return HTMLResponse(f'<p class="text-red-600 font-medium">Error: {e}</p>')


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=8000, reload=True)
