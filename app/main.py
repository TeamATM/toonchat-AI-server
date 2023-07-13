from contextlib import asynccontextmanager
from typing import AsyncGenerator

from redis import asyncio as aioredis
from fastapi import FastAPI

from app.config import app_configs, settings
from app.inference.router import router as inference_router


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


# Define API routes and other application logic here
@app.get("/healthcheck", include_in_schema=False)
async def healthcheck():
    return {"status": "ok"}


app.include_router(inference_router, prefix="/inference")
