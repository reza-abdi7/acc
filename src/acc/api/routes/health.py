from fastapi import APIRouter

router = APIRouter()

# health check endpoint
@router.get("/health", status_code=200)
async def health_check() -> dict:
    """
    health check endpoint for the API.

    returns:
        dict: a dictionary with the status of the AP
    """
    return {"status": "healthy"}
