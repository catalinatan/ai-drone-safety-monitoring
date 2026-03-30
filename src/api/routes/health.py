"""GET /health"""

from fastapi import APIRouter, Depends

from src.api.dependencies import get_feed_manager, get_drone_api
from src.services.feed_manager import FeedManager

router = APIRouter()


@router.get("/health")
async def health_check(
    fm: FeedManager = Depends(get_feed_manager),
    drone_api=Depends(get_drone_api),
):
    return {
        "status": "healthy",
        "feeds_count": len(fm.feed_ids()),
        "drone_api_connected": drone_api is not None,
    }
