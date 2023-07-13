from fastapi import APIRouter, Depends
from app.inference.schemas import RequestData, InferenceResponse
from app.models import LoadedLLM


router = APIRouter()


@router.get("/stream", response_model=InferenceResponse)
async def inference(request_data: RequestData = Depends()):
    streamer = LoadedLLM().generate("", request_data.input)

    for stream in streamer:
        print(stream)

    return {"redis_key": request_data.session_id}
