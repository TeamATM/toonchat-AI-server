from fastapi import APIRouter, Depends

from app.inference.schemas import RequestData, RedisKeyResponse, CompletionResponse
from app.models import RequestCounter
from app.llm.models import LoadedLLM
from threading import Thread
from asyncio import AbstractEventLoop, run_coroutine_threadsafe, get_event_loop, sleep
import time

router = APIRouter()
counter = RequestCounter()


def handle_stream(streamer, loop: AbstractEventLoop, queue: list = None):
    for s in streamer:
        print(s, end="")
        if queue is not None:
            queue.append(s)
    print()

    run_coroutine_threadsafe(counter.decrease(), loop)


@router.get("/stream", response_model=RedisKeyResponse)
async def stream_inference(request_data: RequestData = Depends()):
    streamer = LoadedLLM().generate("", request_data.input)

    Thread(target=handle_stream, args=[streamer, get_event_loop()]).start()

    return {"redis_key": request_data.session_id}


@router.get("/completion", response_model=CompletionResponse)
async def completion_inference(request_data: RequestData = Depends()):
    streamer = LoadedLLM().generate("", request_data.input)

    outputs = []

    thread = Thread(target=handle_stream, args=[streamer, get_event_loop(), outputs])
    thread.start()
    start_time = time.time()

    while time.time() - start_time < 30:
        if not thread.is_alive():
            break
        await sleep(0.01)

    return {"result": ("".join(outputs)).strip()}
