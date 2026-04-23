"""WebSocket endpoint for real-time detection status streaming."""

from __future__ import annotations

import asyncio
from datetime import datetime

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from src.api.dependencies import get_feed_manager
from src.services.feed_manager import FeedManager

router = APIRouter()


@router.websocket("/ws/status")
async def websocket_status(
    websocket: WebSocket,
    fm: FeedManager = Depends(get_feed_manager),
):
    """Stream real-time detection status for all feeds.

    Connects to WebSocket and sends detection state updates at ~1 Hz.
    Sends:
    ```json
    {
        "feeds": [
            {
                "feed_id": "cctv-1",
                "alarm_active": false,
                "caution_active": false,
                "people_count": 0,
                "danger_count": 0,
                "caution_count": 0
            }
        ],
        "timestamp": "2024-01-01T00:00:00"
    }
    ```
    """
    await websocket.accept()

    try:
        while True:
            # Send status every 250ms for responsive UI updates
            await asyncio.sleep(0.25)

            # Collect status for all feeds
            feeds_status = []
            for feed_id in fm.feed_ids():
                state = fm.get_state(feed_id)
                if state:
                    feeds_status.append(
                        {
                            "feed_id": feed_id,
                            "alarm_active": state.alarm_active,
                            "caution_active": state.caution_active,
                            "people_count": state.people_count,
                            "danger_count": state.danger_count,
                            "caution_count": state.caution_count,
                        }
                    )

            # Send as JSON
            message = {
                "feeds": feeds_status,
                "timestamp": datetime.utcnow().isoformat(),
            }
            await websocket.send_json(message)

    except WebSocketDisconnect:
        # Client disconnected — clean exit
        pass
    except Exception as e:
        print(f"[WS/STATUS] Error: {e}")
        try:
            await websocket.close(code=1011, reason="Internal error")
        except Exception:
            pass
