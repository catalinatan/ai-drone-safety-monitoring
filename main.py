"""
Development entrypoint — launches all services with one command.

Usage:
    python main.py          # backend (8001) + drone API (8000) + React UI (5173)
    python main.py --no-ui  # backend + drone API only

Ctrl+C shuts down all processes cleanly.
"""

import subprocess
import signal
import sys
import os
import time

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.path.join(REPO_ROOT, "src", "ui")

SERVICES = {
    "drone":   {"cmd": [sys.executable, "-m", "src.drone_control.drone"],  "port": 8000},
    "backend": {"cmd": [sys.executable, "-m", "src.backend.server"],       "port": 8001},
}


def main():
    no_ui = "--no-ui" in sys.argv
    procs: dict[str, subprocess.Popen] = {}

    def shutdown(*_):
        print("\n[main] Shutting down all services...")
        for name, p in procs.items():
            if p.poll() is None:
                p.terminate()
        for name, p in procs.items():
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, shutdown)

    print("=" * 60)
    print("  AI SAFETY MONITORING — Development Server")
    print("=" * 60)

    # Launch Python services
    for name, svc in SERVICES.items():
        procs[name] = subprocess.Popen(svc["cmd"], cwd=REPO_ROOT)
        print(f"  [{name:8s}]  http://localhost:{svc['port']}")

    # Launch React UI
    if not no_ui:
        npm = "npm.cmd" if sys.platform == "win32" else "npm"
        procs["ui"] = subprocess.Popen([npm, "run", "dev"], cwd=UI_DIR)
        print(f"  [{'ui':8s}]  http://localhost:5173")

    print("=" * 60)
    print("  Press Ctrl+C to stop all services")
    print("=" * 60)

    # Monitor — if any process exits unexpectedly, shut everything down
    try:
        while True:
            for name, p in list(procs.items()):
                ret = p.poll()
                if ret is not None:
                    print(f"[main] {name} exited with code {ret}")
                    shutdown()
            time.sleep(0.5)
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()
