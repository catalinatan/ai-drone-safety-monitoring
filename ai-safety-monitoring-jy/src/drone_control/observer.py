"""
Observer Camera System — Read-Only CCTV Feed (AirSim Simulation)

Architecture overview
---------------------
This is a lightweight FastAPI server that provides multiple MJPEG video streams
from a fixed observer drone (Drone2) in AirSim. The drone has 4 cameras pointing
in different directions to provide comprehensive coverage.

Purpose:
  - Provides third-person, CCTV-style views of the scene
  - Uses cameras 0-3 on Drone2 (static observer drone)
  - No control inputs — purely observational
  - Runs on port 8001 to avoid conflicts with the drone control system

Thread safety
~~~~~~~~~~~~~
This server captures frames from all 4 cameras simultaneously in a single thread
and stores them in a thread-safe state object.
"""

import airsim
import time
import cv2
import numpy as np
import math
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import threading
import uvicorn
from typing import Optional, Dict

# ============================================================================
# SHARED STATE (Thread-Safe)
# ============================================================================

class ObserverState:
    """Thread-safe storage for the latest camera frame."""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.frame: Optional[np.ndarray] = None
    
    def set_frame(self, frame: np.ndarray):
        with self.lock:
            self.frame = frame.copy() if frame is not None else None
    
    def get_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

# Global state instance
observer_state = ObserverState()

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(title="Observer Camera API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/status")
async def get_status():
    """Get observer camera status."""
    return {
        "camera_name": "Drone2",
        "camera_id": "0",
        "type": "static_observer",
        "connected": True,
    }

def generate_frames():
    """Yields MJPEG frames for the /video_feed endpoint."""
    while True:
        frame = observer_state.get_frame()
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)  # ~30 FPS

@app.get("/video_feed/{camera_id}")
async def video_feed(camera_id: int):
    """Stream video feed from observer camera 0."""
    if camera_id != 0:
        raise HTTPException(status_code=404, detail=f"Only camera 0 is available")
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# ============================================================================
# CAMERA CAPTURE THREAD
# ============================================================================

def camera_capture_loop():
    """Continuously captures frames from camera 0 on Drone2.
    
    This runs in a separate thread to ensure the API remains responsive.
    """
    print("[OBSERVER] Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("[OBSERVER] Connected to AirSim")
    
    # Enable API control and make Drone2 hover at altitude
    print("[OBSERVER] Enabling API control for Drone2...")
    client.enableApiControl(True, vehicle_name="Drone2")
    client.armDisarm(True, vehicle_name="Drone2")
    
    print("[OBSERVER] Taking off to 15m altitude...")
    client.moveToZAsync(-3.3, 5, vehicle_name="Drone2").join()  # Move to 15m height
    client.hoverAsync(vehicle_name="Drone2").join()  # Hold position
    print("[OBSERVER] Drone2 hovering at altitude")
    
    print("[OBSERVER] Camera capture started (Drone2 - camera 0)")
    
    frame_count = 0
    try:
        while True:
            # Request image from camera 0 only
            responses = client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
            ], vehicle_name="Drone2")
            
            if responses and len(responses) > 0:
                response = responses[0]
                if response and len(response.image_data_uint8) > 0:
                    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                    img = img1d.reshape(response.height, response.width, 3)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    
                    # Add overlay text
                    cv2.putText(img, "OBSERVER CAM - LIVE", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    observer_state.set_frame(img)
                    
                    frame_count += 1
                    if frame_count % 100 == 0:
                        print(f"[OBSERVER] Captured {frame_count} frames")
                else:
                    print(f"[WARNING] Empty image data from camera 0")
            else:
                print(f"[WARNING] No response from simGetImages")
            
            time.sleep(0.033)  # ~30 FPS
            
    except KeyboardInterrupt:
        print("[OBSERVER] Shutting down camera capture...")
    except Exception as e:
        print(f"[ERROR] Camera capture error: {e}")

# ============================================================================
# FASTAPI SERVER THREAD
# ============================================================================

def run_api_server():
    """Run FastAPI server in separate thread."""
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("OBSERVER CAMERA SYSTEM - Read-Only CCTV Feed")
    print("="*60)
    print("[OBSERVER] Starting camera capture thread...")
    
    # Thread 1 — Camera capture (daemon: exits when main thread exits)
    capture_thread = threading.Thread(target=camera_capture_loop, daemon=True)
    capture_thread.start()
    
    # Give capture thread time to initialize
    time.sleep(1)
    
    # Thread 2 — REST API (runs on main thread, blocks until Ctrl+C)
    print("[OBSERVER] Starting API server on http://0.0.0.0:8001")
    print("[OBSERVER] Video feed: http://localhost:8001/video_feed/0")
    print("="*60)
    
    try:
        run_api_server()
    except KeyboardInterrupt:
        print("\n[OBSERVER] Shutdown complete")