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
    """Thread-safe storage for the latest camera frames from all 4 cameras."""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.frames: Dict[int, Optional[np.ndarray]] = {0: None, 1: None, 2: None, 3: None}
    
    def set_frame(self, camera_id: int, frame: np.ndarray):
        with self.lock:
            self.frames[camera_id] = frame.copy() if frame is not None else None
    
    def get_frame(self, camera_id: int) -> Optional[np.ndarray]:
        with self.lock:
            frame = self.frames.get(camera_id)
            return frame.copy() if frame is not None else None

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
        "cameras": ["0", "1", "2", "3"],
        "type": "static_observer",
        "connected": True,
    }

def generate_frames(camera_id: int):
    """Yields MJPEG frames for a specific camera."""
    while True:
        frame = observer_state.get_frame(camera_id)
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)  # ~30 FPS

@app.get("/video_feed/{camera_id}")
async def video_feed(camera_id: int):
    """Stream video feed from a specific observer camera (0-3)."""
    if camera_id not in [0, 1, 2, 3]:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    
    return StreamingResponse(
        generate_frames(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# ============================================================================
# CAMERA CAPTURE THREAD
# ============================================================================

def camera_capture_loop():
    """Continuously captures frames from all 4 cameras on Drone2.
    
    This runs in a separate thread to ensure the API remains responsive.
    """
    print("[OBSERVER] Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("[OBSERVER] Connected to AirSim")
    print("[OBSERVER] Camera capture started (Drone2 - 4 cameras)")
    
    camera_labels = {
        0: "CAM SW",
        1: "CAM S", 
        2: "CAM SE",
        3: "CAM WIDE"
    }
    
    try:
        while True:
            # Request images from all 4 cameras simultaneously
            responses = client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                airsim.ImageRequest("1", airsim.ImageType.Scene, False, False),
                airsim.ImageRequest("2", airsim.ImageType.Scene, False, False),
                airsim.ImageRequest("3", airsim.ImageType.Scene, False, False),
            ], vehicle_name="Drone2")
            
            if responses and len(responses) == 4:
                for i, response in enumerate(responses):
                    if response and len(response.image_data_uint8) > 0:
                        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                        img = img1d.reshape(response.height, response.width, 3)
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        
                        # Add overlay text
                        cv2.putText(img, f"{camera_labels[i]} - LIVE", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        observer_state.set_frame(i, img)
            
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
    print("[OBSERVER] Video feeds:")
    print("[OBSERVER]   CAM 0 (SW): http://localhost:8001/video_feed/0")
    print("[OBSERVER]   CAM 1 (S):  http://localhost:8001/video_feed/1")
    print("[OBSERVER]   CAM 2 (SE): http://localhost:8001/video_feed/2")
    print("[OBSERVER]   CAM 3 (W):  http://localhost:8001/video_feed/3")
    print("="*60)
    
    try:
        run_api_server()
    except KeyboardInterrupt:
        print("\n[OBSERVER] Shutdown complete")
