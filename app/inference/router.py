from fastapi import APIRouter, Depends
from app.inference.schemas import RequestData, InferenceResponse

router = APIRouter()


@router.get("/stream", response_model=InferenceResponse)
async def inference(request_data: RequestData = Depends()):
    return {"redis_key": request_data.session_id}
