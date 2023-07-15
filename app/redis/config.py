from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Extra
from redis import asyncio as aioredis


class Redis(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env.redis", env_file_encoding="utf-8", extra=Extra.allow
    )

    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_PASSWORD: str
    REDIS_USER: str

    REQUEST_STREAM: str
    RESPONSE_STREAM: str
    CONSUMER_GROUP: str

    # connection:aioredis.Redis

    def create_connection(self):
        connection_url = (
            f"redis://{self.REDIS_USER}:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}"
        )
        self.connection = aioredis.from_url(connection_url, db=0)

        return self.connection
