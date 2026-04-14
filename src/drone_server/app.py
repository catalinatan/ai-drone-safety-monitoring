"""
Drone Control Server — FastAPI application (port 8000).

Usage (production):
    uvicorn src.drone_server.app:app --host 0.0.0.0 --port 8000

Usage (development via main.py):
    Launched automatically by main.py as a subprocess.

Architecture:
    The ``lifespan`` context manager creates an AirSim client, starts the
    control loop as a daemon thread, then tears down cleanly on shutdown.
    All mutable drone state lives in the ``drone_state`` singleton imported
    from ``drone_state.py``.
"""

from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .control_loop import drone_control_loop
from .drone_state import GotoRequest, ModeRequest, MoveRequest, drone_state


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(_CONFIG_PATH, "r") as _f:
    _cfg = yaml.safe_load(_f)


# ---------------------------------------------------------------------------
# MJPEG streaming helpers
# ---------------------------------------------------------------------------


def _generate_frames_forward():
    """Yield MJPEG frames from the forward-looking camera (camera 3)."""
    while True:
        frame = drone_state.get_frame_forward()
        if frame is not None:
            ret, buffer = cv2.imencode(".jpg", frame)
            if ret:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
        time.sleep(0.033)


def _generate_frames_down():
    """Yield MJPEG frames from the downward-looking camera (camera 0)."""
    while True:
        frame = drone_state.get_frame_down()
        if frame is not None:
            ret, buffer = cv2.imencode(".jpg", frame)
            if ret:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
        time.sleep(0.033)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: connect to AirSim and launch the control loop thread.
    Shutdown: signal the loop to stop and wait for the drone to land."""
    client = None
    control_thread = None

    try:
        import airsim  # just verify airsim is installed; client created inside the loop thread

        control_thread = threading.Thread(
            target=drone_control_loop,
            args=(drone_state, None, _cfg),  # None: loop creates its own client
            daemon=True,
            name="drone-control-loop",
        )
        control_thread.start()
        print("[DRONE SERVER] Control loop started")

    except ImportError as e:
        print(f"[DRONE SERVER] AirSim not available — running in limited mode: {e}")
    except Exception as e:
        print(f"[DRONE SERVER] Failed to start control loop: {e}")

    print("[DRONE SERVER] API ready")
    yield

    # Shutdown
    drone_state.request_stop()
    if control_thread is not None and control_thread.is_alive():
        control_thread.join(timeout=10)
    print("[DRONE SERVER] Shutdown complete")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    app = FastAPI(title="Drone Control API", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- Mode --

    @app.post("/mode")
    async def set_mode(request: ModeRequest):
        """Switch between manual and automatic control modes."""
        mode = request.mode.lower()
        if mode not in ("manual", "automatic"):
            raise HTTPException(status_code=400, detail="Mode must be 'manual' or 'automatic'")
        drone_state.set_mode(mode)
        print(f"[API] Mode changed to: {mode}")
        return {"status": "success", "mode": mode}

    # -- Manual velocity --

    @app.post("/move")
    async def move_velocity(request: MoveRequest):
        """Send velocity command to drone (manual mode only)."""
        if drone_state.get_mode() != "manual":
            raise HTTPException(status_code=400, detail="Move commands only work in manual mode")
        drone_state.set_manual_velocity(request.vx, request.vy, request.vz)
        return {"status": "success"}

    # -- Goto --

    @app.post("/goto")
    async def goto_position(request: GotoRequest):
        """Command drone to fly to specified NED coordinates (automatic mode only)."""
        current_mode = drone_state.get_mode()
        if current_mode != "automatic":
            raise HTTPException(
                status_code=400,
                detail=f"Cannot navigate in {current_mode} mode. Switch to automatic first.",
            )
        if drone_state.get_returning_home():
            raise HTTPException(
                status_code=409,
                detail="Drone is returning home. New goto commands are blocked until landing completes.",
            )

        target_ned = (request.x, request.y, request.z)
        drone_state.set_target(target_ned)
        print(f"[API] Goto: NED=({request.x}, {request.y}, {request.z})")

        return {
            "status": "success",
            "message": "Navigation started",
            "target_ned": {"x": target_ned[0], "y": target_ned[1], "z": target_ned[2]},
        }

    # -- Return home --

    @app.post("/return_home")
    async def return_home():
        """Command drone to return to home position.

        Works from any mode — switches to automatic for the return flight.
        On arrival the control loop lands, disarms, and sets mode back to
        automatic so the drone is ready for the next trigger.
        """
        home = drone_state.get_home()
        if home is None:
            raise HTTPException(status_code=400, detail="Home position not set")
        drone_state.set_mode("automatic")
        drone_state.set_returning_home(True)
        drone_state.set_target(home)
        print(f"[API] Return to home: target={home}")
        return {"status": "success", "message": "Returning to home", "home_position": home}

    # -- Status --

    @app.get("/status")
    async def get_status():
        """Get current drone status."""
        target, is_nav, _ = drone_state.get_nav_snapshot()
        pose = drone_state.get_pose()
        return {
            "mode": drone_state.get_mode(),
            "connected": True,
            "is_navigating": is_nav,
            "returning_home": drone_state.get_returning_home(),
            "grounded": drone_state.is_grounded(),
            "target_position": target,
            "pose": {"x": pose[0], "y": pose[1], "z": pose[2]} if pose else None,
        }

    # -- Video feeds --

    @app.get("/video_feed")
    async def video_feed():
        """Stream forward-looking camera (camera 3) — default view."""
        return StreamingResponse(
            _generate_frames_forward(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/video_feed/forward")
    async def video_feed_forward():
        """Stream forward-looking camera (camera 3)."""
        return StreamingResponse(
            _generate_frames_forward(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/video_feed/down")
    async def video_feed_down():
        """Stream downward-looking camera (camera 0)."""
        return StreamingResponse(
            _generate_frames_down(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    return app


# Module-level app instance (for uvicorn src.drone_server.app:app)
app = create_app()
