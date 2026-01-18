from fastapi import APIRouter
from ..schemas.health import HealthOutput

router = APIRouter(prefix="/health", tags=["health"])

@router.get("/")
async def health():
    return HealthOutput()