#!/usr/bin/env python3
"""
H-Walker CORE — desktop wrapper.

Launches the FastAPI server in a background thread, waits for readiness,
then opens a native pywebview window pointed at http://127.0.0.1:<port>.

Usage (dev):
    python3 desktop.py

Usage (bundled):
    Double-click the generated .app (macOS) / .exe (Windows). The
    entry point is the same `main()` — py2app / pyinstaller configs in
    packaging/ point at this file.

Fallback: if pywebview is not installed (e.g. headless dev server),
`--headless` or the absence of a display triggers the legacy
browser-tab mode (uvicorn in foreground + `webbrowser.open`).
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


APP_DIR = Path(__file__).resolve().parent
FRONTEND_DIST = APP_DIR / "frontend" / "dist"
DESKTOP_ICON = APP_DIR / "desktop" / "icon.png"


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def _pick_port(preferred: int = 8000) -> int:
    if _port_free(preferred):
        return preferred
    # Fall through to ephemeral
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def _build_app():
    """Construct a FastAPI app with every available router mounted.

    Mirrors run.py's progressive-import pattern so missing optional
    deps (ollama, google SDK) don't fatally block the desktop launch.
    """
    # Ensure project root is importable regardless of cwd.
    sys.path.insert(0, str(APP_DIR))

    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    app = FastAPI(title="H-Walker CORE")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1", "http://localhost"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mandatory routers
    from backend.routers.datasets import router as datasets_router
    from backend.routers.graphs import router as graphs_router
    from backend.routers.analyze import router as analyze_router
    from backend.routers.compute import router as compute_router
    from backend.routers.stats import router as stats_router
    from backend.routers.paper import router as paper_router
    from backend.routers.inspector import router as inspector_router
    from backend.routers.sync import router as sync_router
    app.include_router(datasets_router)
    app.include_router(graphs_router)
    app.include_router(analyze_router)
    app.include_router(compute_router)
    app.include_router(stats_router)
    app.include_router(paper_router)
    app.include_router(inspector_router)
    app.include_router(sync_router)

    # Optional: Claude
    try:
        from backend.routers.claude import router as claude_router
        app.include_router(claude_router)
    except Exception as exc:   # anthropic SDK optional at import time
        print(f"[desktop] Claude router disabled: {exc}")

    # Optional: legacy routers (graph/chat/drive/journal/feedback)
    for modname in ("graph", "chat", "drive", "journal", "feedback"):
        try:
            mod = __import__(f"backend.routers.{modname}", fromlist=["router"])
            r = getattr(mod, "router", None)
            if r is not None:
                app.include_router(r, prefix="/api" if modname == "graph" else "")
        except Exception:
            pass

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    # Serve the built SPA
    if FRONTEND_DIST.is_dir():
        assets = FRONTEND_DIST / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")
        fonts = FRONTEND_DIST / "fonts"
        if fonts.is_dir():
            app.mount("/fonts", StaticFiles(directory=str(fonts)), name="fonts")
        icons = FRONTEND_DIST / "icons"
        if icons.is_dir():
            app.mount("/icons", StaticFiles(directory=str(icons)), name="icons")

        @app.get("/{path:path}")
        async def spa(path: str):
            fp = FRONTEND_DIST / path
            if fp.is_file():
                return FileResponse(fp)
            return FileResponse(FRONTEND_DIST / "index.html")
    else:
        print(f"[desktop] WARNING: frontend dist missing at {FRONTEND_DIST}")
        print("           Run `cd frontend && npm install && npm run build` first.")

    return app


def _wait_ready(url: str, timeout: float = 20.0) -> bool:
    """Poll /health until the server responds or timeout expires."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            with urlopen(url, timeout=0.8) as resp:
                if resp.status == 200:
                    return True
        except URLError:
            pass
        except Exception:
            pass
        time.sleep(0.2)
    return False


def _launch_uvicorn(app, port: int) -> None:
    import uvicorn
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning",
                            access_log=False)
    server = uvicorn.Server(config)
    # uvicorn.Server.run() blocks — we're already inside a thread.
    server.run()


def _open_webview(port: int, app_name: str = "H-Walker CORE") -> None:
    import webview  # pywebview
    url = f"http://127.0.0.1:{port}/"
    kwargs: dict = dict(
        title=app_name,
        url=url,
        width=1480,
        height=940,
        min_size=(1100, 720),
        background_color="#0B0E2E",
        confirm_close=False,
    )
    window = webview.create_window(**kwargs)
    # pywebview >=4 uses `icon` kw in start(), not create_window()
    start_kwargs: dict = {}
    if DESKTOP_ICON.exists():
        start_kwargs["icon"] = str(DESKTOP_ICON)
    try:
        webview.start(**start_kwargs)
    except TypeError:
        # Older pywebview — icon param not supported on start()
        webview.start()
    # When the window closes, exit the whole process (uvicorn thread is daemon).
    os._exit(0)
    _ = window  # quiet linters


def main() -> None:
    parser = argparse.ArgumentParser(description="H-Walker CORE desktop")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--headless", action="store_true",
                        help="Force browser-tab mode (skip pywebview).")
    parser.add_argument("--timeout", type=float, default=20.0,
                        help="Seconds to wait for the server before opening the window.")
    args = parser.parse_args()

    port = _pick_port(args.port)

    print(f"[desktop] H-Walker CORE — booting on http://127.0.0.1:{port}/")

    app = _build_app()

    # Start uvicorn in a daemon thread
    t = threading.Thread(target=_launch_uvicorn, args=(app, port), daemon=True)
    t.start()

    ok = _wait_ready(f"http://127.0.0.1:{port}/health", timeout=args.timeout)
    if not ok:
        print(f"[desktop] ERROR: server did not become ready within {args.timeout}s")
        sys.exit(1)

    try:
        import webview  # noqa: F401
        can_webview = not args.headless
    except ImportError:
        can_webview = False
        if not args.headless:
            print("[desktop] pywebview not installed; falling back to browser.")
            print("          → pip install pywebview")

    if can_webview:
        print(f"[desktop] opening pywebview window")
        _open_webview(port)
    else:
        # Legacy browser-tab mode — keep the main thread alive on uvicorn.
        print(f"[desktop] opening default browser → http://127.0.0.1:{port}/")
        webbrowser.open(f"http://127.0.0.1:{port}/")
        try:
            t.join()
        except KeyboardInterrupt:
            print("\n[desktop] stopped.")


if __name__ == "__main__":
    main()
