#!/bin/bash
# ==============================================================
#  H-Walker CORE · double-click launcher (macOS + Linux)
#
#  Double-click this file in Finder. A Terminal opens, the server
#  starts on http://localhost:8000, and your default browser opens
#  the app. The window stays open so you can read errors instead
#  of vanishing in 0.3 s.
#
#  If pywebview is installed, a native desktop window is used
#  (desktop.py). Otherwise a browser tab is opened (run.py).
# ==============================================================

# IMPORTANT: do NOT use `set -e`. The user's .zshrc may source
# missing files (powerlevel10k etc.) that fail; we want the script
# to keep running anyway. Errors from python3 run.py are surfaced
# explicitly via the trap below.

cd "$(dirname "$0")" || {
    echo "[FATAL] could not cd into $(dirname "$0")"
    read -r -p "Press Enter to close…" _
    exit 1
}

# Pause on exit so the user can read whatever happened. macOS
# Terminal closes the window the moment the script finishes when
# launched via the .command-double-click flow, which makes
# diagnosing errors basically impossible.
on_exit () {
    code=$?
    echo
    if [ "$code" -eq 0 ]; then
        echo "[launcher] H-Walker CORE stopped cleanly."
    else
        echo "[launcher] H-Walker CORE exited with code $code."
        echo "           Scroll up to see what went wrong."
    fi
    echo
    read -r -p "Press Enter to close this window…" _
}
trap on_exit EXIT

# Source the user's shell profile so ANTHROPIC_API_KEY (and any
# pyenv / homebrew PATH) is available. Each source is fully
# isolated — a failing zshrc line cannot kill the launcher.
for f in ~/.zprofile ~/.bash_profile ~/.bashrc ~/.zshrc; do
    [ -f "$f" ] && (source "$f") >/dev/null 2>&1 || true
done

# Make sure homebrew python is reachable even if the profile didn't
# export it (common on Apple-Silicon Macs).
for d in /opt/homebrew/bin /usr/local/bin; do
    case ":$PATH:" in *":$d:"*) ;; *) [ -d "$d" ] && PATH="$d:$PATH" ;; esac
done
export PATH

echo "=============================================="
echo " H-Walker CORE — gait analysis engine"
echo " repo : $(pwd)"
echo " py   : $(command -v python3 || echo 'NOT FOUND')"
echo " ver  : $(python3 --version 2>&1 || true)"
echo " key  : ${ANTHROPIC_API_KEY:+(API key present, but Haiku is no longer used — you can ignore)}${ANTHROPIC_API_KEY:-(no API key — fine, the app no longer needs one)}"
echo "=============================================="
echo

if ! command -v python3 >/dev/null 2>&1; then
    echo "[FATAL] python3 was not found in PATH."
    echo "        Install with: brew install python@3.11"
    exit 1
fi

# Defensive: rebuild the frontend dist if it's missing or older
# than the latest source change. After a `git pull` of a UI commit
# this is the most common cause of a black browser screen, or worse —
# the app boots but serves stale labels (the operator sees old card
# titles even though they checked out a newer branch).
if [ -d "frontend" ]; then
    needs_build=0
    if [ ! -f "frontend/dist/index.html" ]; then
        needs_build=1
        echo "[launcher] frontend/dist not found — building once…"
    else
        # Find any source file newer than the existing bundle.
        # `-newer` returns just one match (head -n 1) — that's enough
        # to trigger a rebuild without scanning the whole tree.
        newest_src=$(find frontend/src frontend/index.html frontend/package.json \
                          frontend/vite.config.ts frontend/tsconfig.json \
                          -type f \( -name '*.ts' -o -name '*.tsx' -o -name '*.css' \
                                  -o -name '*.html' -o -name '*.json' \) \
                          -newer frontend/dist/index.html 2>/dev/null | head -n 1)
        if [ -n "$newest_src" ]; then
            needs_build=1
            echo "[launcher] frontend source newer than dist (e.g. $newest_src) — rebuilding…"
        fi
    fi
    if [ "$needs_build" = "1" ]; then
        ( cd frontend && npm install --silent && npm run build ) || {
            echo "[launcher] frontend build failed — fix the error above and re-run."
            exit 1
        }
    fi
fi

# Branch on pywebview availability.
if python3 -c "import webview" 2>/dev/null; then
    echo "[launcher] pywebview detected → native window via desktop.py"
    python3 desktop.py "$@"
else
    echo "[launcher] pywebview not installed → browser mode via run.py"
    echo "           (run 'pip3 install pywebview' for a native window)"
    python3 run.py "$@"
fi
