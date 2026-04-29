import sys
import os

# Add parent directory to sys.path so `from backend.xxx import` works
# when running `uvicorn main:app` from the backend/ directory.
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Core (always-required) routers — matches run.py
from backend.routers.graph import router as graph_router
from backend.routers.journal import router as journal_router
from backend.routers.datasets import router as datasets_router
from backend.routers.graphs import router as graphs_router
from backend.routers.analyze import router as analyze_router
from backend.routers.compute import router as compute_router
from backend.routers.stats import router as stats_router
from backend.routers.paper import router as paper_router
from backend.routers.inspector import router as inspector_router
from backend.routers.sync import router as sync_router

app = FastAPI(title="H-Walker Graph API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# graph_router is the only one without an internal "/api" prefix.
app.include_router(graph_router, prefix="/api")
app.include_router(journal_router)
app.include_router(datasets_router)
app.include_router(graphs_router)
app.include_router(analyze_router)
app.include_router(compute_router)
app.include_router(stats_router)
app.include_router(paper_router)
app.include_router(inspector_router)
app.include_router(sync_router)

# Optional routers — same try/except pattern as run.py so missing
# SDKs (ollama, google-api-python-client, anthropic) don't break the
# whole app or test collection.
try:
    from backend.routers.chat import router as chat_router, ws_router as chat_ws_router
    app.include_router(chat_router)
    app.include_router(chat_ws_router)
except ImportError:
    pass

try:
    from backend.routers.drive import router as drive_router
    app.include_router(drive_router)
except ImportError:
    pass

try:
    from backend.routers.feedback import router as feedback_router
    app.include_router(feedback_router)
except ImportError:
    pass

# Claude Haiku integration removed per user directive (Library panel
# replaced the LLM dock). claude.py is kept on disk but only mounted
# when the anthropic SDK is actually installed.
try:
    from backend.routers.claude import router as claude_router
    app.include_router(claude_router)
except ImportError:
    pass


@app.get("/health")
def health():
    return {"status": "ok"}
