#!/usr/bin/env python3
"""
H-Walker Graph App — One-click launcher.

Usage:
    python run.py          # 기본 실행
    python run.py --port 8000  # 포트 지정
"""
import subprocess
import sys
import os
import time
import webbrowser
import threading

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(APP_DIR))) # /Users/chobyeongjun/0xhenry.dev
BACKEND_DIR = os.path.join(APP_DIR, "backend")
FRONTEND_DIST = os.path.join(APP_DIR, "frontend", "dist")


def check_deps():
    """Install missing Python packages silently."""
    required = ["fastapi", "uvicorn", "pydantic", "pandas", "numpy", "matplotlib", "httpx"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"Installing: {', '.join(missing)}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q"] + missing,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def kill_port(port: int):
    """Kills process listening on given port."""
    try:
        print(f"Cleaning up port {port}...")
        if sys.platform == "win32":
            subprocess.run(f"for /f \"tokens=5\" %a in ('netstat -aon ^| findstr :{port}') do taskkill /f /pid %a", shell=True, check=False, capture_output=True)
        else:
            # More robust way to kill process on Mac/Linux
            cmd = f"lsof -ti:{port} | xargs kill -9"
            subprocess.run(cmd, shell=True, check=False, capture_output=True)
        time.sleep(1.5)
    except Exception:
        pass


def open_browser(port: int):
    """Wait for server to start, then open browser."""
    time.sleep(2)
    webbrowser.open(f"http://localhost:{port}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="H-Walker Graph App")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    print("H-Walker Graph App")
    print("=" * 40)

    # 0. Kill existing port
    kill_port(args.port)

    # 1. Check dependencies
    print("Checking dependencies...")
    check_deps()

    # 2. Add paths
    sys.path.insert(0, PROJECT_ROOT)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "tools", "graph_app"))

    # 3. Setup FastAPI with frontend static files
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    from backend.routers.graph import router as graph_router
    from backend.routers.journal import router as journal_router

    app = FastAPI(title="H-Walker Graph App")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routes — match main.py mount structure exactly
    # graph_router has no prefix → needs /api
    app.include_router(graph_router, prefix="/api")
    # journal_router has prefix="/api" → mount without extra prefix
    app.include_router(journal_router)

    try:
        from backend.routers.chat import router as chat_router, ws_router
        # chat_router has prefix="/api/chat" → mount without extra prefix
        app.include_router(chat_router)
        # ws_router has /ws/chat → mount without prefix
        app.include_router(ws_router)
    except ImportError:
        print("  Note: Ollama not installed — chat disabled")

    try:
        from backend.routers.drive import router as drive_router
        # drive_router has prefix="/api/drive" → mount without extra prefix
        app.include_router(drive_router)
    except ImportError:
        print("  Note: Google Drive SDK not installed — drive disabled")

    try:
        from backend.routers.feedback import router as feedback_router
        # feedback_router has prefix="/api/feedback" → mount without extra prefix
        app.include_router(feedback_router)
    except ImportError:
        print("  Note: feedback router unavailable")

    try:
        # Claude Haiku integration removed per user directive — the
        # bookshelf-style Library panel replaced the LLM dock. The
        # backend/routers/claude.py file is kept but not registered
        # so the anthropic SDK import doesn't fail when the package
        # isn't installed.
        from backend.routers.datasets import router as datasets_router
        from backend.routers.graphs import router as graphs_router
        app.include_router(datasets_router)
        app.include_router(graphs_router)
    except ImportError as e:
        print(f"  Note: Phase 2 routers unavailable — {e}")

    try:
        from backend.routers.analyze import router as analyze_router
        from backend.routers.compute import router as compute_router
        from backend.routers.stats import router as stats_router
        from backend.routers.paper import router as paper_router
        from backend.routers.inspector import router as inspector_router
        from backend.routers.sync import router as sync_router
        app.include_router(analyze_router)
        app.include_router(compute_router)
        app.include_router(stats_router)
        app.include_router(paper_router)
        app.include_router(inspector_router)
        app.include_router(sync_router)
    except ImportError as e:
        print(f"  Note: Phase 2A analyze/compute/stats routers unavailable — {e}")

    # Serve public-folder static assets (fonts, svg) separately so /fonts/ works
    if os.path.isdir(os.path.join(FRONTEND_DIST, "fonts")):
        app.mount("/fonts", StaticFiles(directory=os.path.join(FRONTEND_DIST, "fonts")), name="fonts")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    # Serve frontend
    if os.path.isdir(FRONTEND_DIST):
        app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")), name="assets")

        @app.get("/{path:path}")
        async def spa_fallback(path: str):
            # Check if path is a file in dist
            file_path = os.path.join(FRONTEND_DIST, path)
            if os.path.isfile(file_path):
                return FileResponse(file_path)
            # Default to index.html for SPA routing
            return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))
    else:
        print(f"⚠️  Warning: Frontend dist not found at {FRONTEND_DIST}")
        print("   Please run 'npm run build' in the frontend directory first.")

    # 4. Open browser
    if not args.no_browser:
        threading.Thread(target=open_browser, args=(args.port,), daemon=True).start()

    # 5. Run server
    import uvicorn
    print(f"\n  → http://localhost:{args.port}")
    print(f"  → API docs: http://localhost:{args.port}/docs")
    print(f"  Press Ctrl+C to stop\n")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
