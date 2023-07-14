from contextlib import asynccontextmanager
from typing import AsyncGenerator

from os import environ
from os.path import dirname, join


environ["ROOT_DIR"] = dirname(dirname(__file__))

from redis import asyncio as aioredis
from fastapi import FastAPI, Depends

from app.config import app_configs, settings
from app.inference.router import router as inference_router
from app.llm.utils import load_lora
from app.llm.constants import ModelType
from app.llm.models import LLMConfig
from app.models import RequestCounter


loaded_model = load_lora(
    LLMConfig(
        ModelType.LoRA,
        join(environ["ROOT_DIR"], "models", "checkpoint-2200"),
        join(environ["ROOT_DIR"], "models", "beomi", "KoAlpaca-Polyglot-5.8B"),
        stopping_words=["\n\n", "\nHuman:"],
    ),
    tag="Remon-2200",
)


@asynccontextmanager
async def lifespan(_application: FastAPI) -> AsyncGenerator:
    # Start
    pool = aioredis.ConnectionPool.from_url(
        settings.REDIS_URL, max_connections=10, decode_responses=True
    )
    redis_client = aioredis.Redis(connection_pool=pool)

    yield

    # Stop
    await redis_client.close()


app = FastAPI(**app_configs, lifespan=lifespan)
counter = RequestCounter()


# Define API routes and other application logic here
@app.get("/healthcheck", include_in_schema=False)
async def healthcheck():
    return {"status": "ok"}


@app.get("/requestcount", include_in_schema=False)
async def request_count():
    return {"count": counter.count}


app.include_router(inference_router, prefix="/inference", dependencies=[Depends(counter.increase)])

if __name__ == "__main__":
    from pyngrok import ngrok
    import uvicorn
    import nest_asyncio

    ngrok.set_auth_token(input())
    ngrok_tunnel = ngrok.connect(8000)
    print("Public URL:", ngrok_tunnel.public_url)
    nest_asyncio.apply()
    uvicorn.run(app, port=8000)
